#!/usr/bin/env python3
"""Add modes 5 (GEMM) and 6 (GEMM split-K); compile only matmul.cpp."""
import argparse
import difflib
import json
from pathlib import Path
import shutil

from build_mlx_command_timing import PACKAGE, compile_argv, run_command, sha256
from build_mlx_gemv_tuning import replace_once


def build(base, output, rebuild_unrun=False):
    base, output = base.resolve(), output.resolve()
    if base == output or base in output.parents:
        raise ValueError("Output must be a fresh directory outside the existing stage")
    old_report_path = base / "build-provenance.json"
    old_report = json.loads(old_report_path.read_text())
    if old_report.get("status") != "built_not_executed" or old_report.get("maximum_mode") != 4:
        raise ValueError("Expected the successfully built BM1/2 stage")
    source = base / "src/matmul.cpp"
    original = source.read_text()
    modified = replace_once(original, "mode <= 4 ? mode : 0", "mode <= 6 ? mode : 0")
    modified = replace_once(modified, "int anemlx_gdn_gemv_max_mode(void) { return 4; }",
                            "int anemlx_gdn_gemv_max_mode(void) { return 6; }")
    # Restrict the edit to Matmul, leaving AddMM and gathered dispatch intact.
    low = modified.index("void Matmul::eval_gpu(")
    high = modified.index("void AddMM::eval_gpu(", low)
    body = modified[low:high]
    body = replace_once(body, "  // Route to gemv if needed\n  if (std::min(M, N) == 1) {", """  // Experimental modes 5/6: allow these single-row BF16 shapes through
  // to steel_matmul. Its existing runtime/NAX selection remains unchanged.
  const bool anemlx_force_gemm =
      (anemlx_gdn_gemv_mode.load(std::memory_order_relaxed) == 5 ||
       anemlx_gdn_gemv_mode.load(std::memory_order_relaxed) == 6) &&
      M == 1 && !a_transposed && b_transposed &&
      a.dtype() == bfloat16 && b.dtype() == bfloat16 && out.dtype() == bfloat16 &&
      ((K == 2560 && (N == 10240 || N == 6144)) || (K == 6144 && N == 2560));
  // Route to gemv if needed, except for the explicitly selected experiment.
  if (std::min(M, N) == 1 && !anemlx_force_gemm) {""")
    modified = modified[:low] + body + modified[high:]
    low = modified.index("void steel_gemm_splitk_axpby_nax(")
    high = modified.index("void steel_matmul_axpby(", low)
    body = modified[low:high]
    condition = """anemlx_gdn_gemv_mode.load(std::memory_order_relaxed) == 6 &&
      M == 1 && !transpose_a && transpose_b &&
      a.dtype() == bfloat16 && b.dtype() == bfloat16 && out.dtype() == bfloat16 &&
      ((K == 2560 && (N == 10240 || N == 6144)) || (K == 6144 && N == 2560))"""
    body = replace_once(body, "  // Determine how many partitions to split K into\n", """  if (""" + condition + """) {
    bm = bn = 64;
    bk = 256;
    wm = wn = 2;
    split_k_partition_size = 256;
  }

  // Determine how many partitions to split K into
""")
    modified = modified[:low] + body + modified[high:]
    old = """  // Case 2: Large K with sufficient M, N, and NAX is available, use NAX split-K
  if (use_nax && batch_size_out == 1 &&
      (K >= 3 * std::max(M, N) ||
       (std::max(M, N) <= 1024 && K > 2 * std::max(M, N)))) {"""
    new = """  // Case 2: Preserve availability/batch checks; mode 6 opts in target shapes.
  const bool anemlx_force_splitk = """ + condition + """;
  if (use_nax && batch_size_out == 1 &&
      (anemlx_force_splitk || K >= 3 * std::max(M, N) ||
       (std::max(M, N) <= 1024 && K > 2 * std::max(M, N)))) {"""
    modified = replace_once(modified, old, new)
    commands = old_report["commands"]
    compiles = [c for c in commands if "-c" in c["argv"] and str(source) in c["argv"]]
    links = [c for c in commands if "-c" not in c["argv"]
             and str(base / "obj/matmul.cpp.o") in c["argv"]]
    if len(compiles) != 1 or len(links) != 1:
        raise ValueError("Expected exact previous C++ compile and link commands")
    if output.exists() and rebuild_unrun:
        previous = json.loads((output / "build-provenance.json").read_text())
        if previous.get("status") != "built_not_executed" or previous.get("base_stage") != str(base):
            raise ValueError("Only the explicitly authorized unrun stage can be rebuilt")
    output.mkdir(parents=True, exist_ok=rebuild_unrun)
    report = {"schema_version": 1, "status": "building", "base_stage": str(base),
              "base_provenance_sha256": sha256(old_report_path), "maximum_mode": 6,
              "commands": [], "timing_automatically_started": False,
              "mode_5": "gemm: BF16 M1, non-transposed A/transposed B, three target shapes bypass GEMV for existing steel_matmul; includes matching Attention out projections",
              "mode_6": "gemmSplit: same bypass plus NAX split-K on target shapes, BM64 BN64 BK256 WM2 WN2, K partition size 256; NAX availability and batch_size==1 remain required",
              "numerical_validation": "not_run", "performance_validation": "not_run"}
    try:
        for directory in ("src", "obj", "lib", "logs"):
            (output / directory).mkdir(exist_ok=rebuild_unrun)
        copied = output / "src/matmul.cpp"
        copied.write_text(modified)
        (output / "src/matmul.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), modified.splitlines(keepends=True),
            fromfile=str(source), tofile=str(copied))))
        for name in ("libmlxc.dylib", "libjaccl.dylib", "mlx.metallib"):
            shutil.copy2(base / "lib" / name, output / "lib" / name)
        if not (output / "include").exists():
            (output / "include").symlink_to((base / "include").resolve(), target_is_directory=True)
        new_object = output / "obj/matmul.cpp.o"
        compile_command = compile_argv({"arguments": compiles[0]["argv"]}, source,
                                       copied, new_object, output / "lib/mlx.metallib")
        link_command = list(links[0]["argv"])
        if link_command.count("-o") != 1 or link_command.count(str(base / "obj/matmul.cpp.o")) != 1:
            raise ValueError("Unexpected previous link outputs")
        link_command[link_command.index("-o") + 1] = str(output / "lib/libmlx.dylib")
        link_command[link_command.index(str(base / "obj/matmul.cpp.o"))] = str(new_object)
        run_command(compile_command, Path(compiles[0]["cwd"]), output / "logs/compile.log", report["commands"])
        run_command(link_command, Path(links[0]["cwd"]), output / "logs/link.log", report["commands"])
        run_command(["nm", "-gU", str(output / "lib/libmlx.dylib")], output,
                    output / "logs/symbols.log", report["commands"])
        symbols = (output / "logs/symbols.log").read_text()
        for name in ("anemlx_set_gdn_gemv_mode", "anemlx_gdn_gemv_max_mode",
                     "anemlx_timing_start", "anemlx_timing_stop", "anemlx_timing_version"):
            if "_" + name not in symbols:
                raise RuntimeError(f"Missing export: {name}")
        report["status"] = "built_not_executed"
        report["outputs"] = [{"path": str(p), "sha256": sha256(p)}
                             for p in (copied, output / "lib/libmlx.dylib")]
        report["runtime_selection"] = {
            "DYLD_LIBRARY_PATH": str(output / "lib"), "ANERUNNER_MLX_ROOT": str(output),
            "metallib": "Unmodified copy from BM1/2 stage, preferred by colocated lookup",
            "setter": "anemlx_set_gdn_gemv_mode(5 or 6)", "maximum_mode_getter": "anemlx_gdn_gemv_max_mode returns 6"}
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        (output / "build-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return output / "build-provenance.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-stage", type=Path, default=PACKAGE / "results/gdn-gemv-bm12/native")
    parser.add_argument("--output-root", type=Path, default=PACKAGE / "results/gdn-gemv-gemm/native")
    parser.add_argument("--rebuild-unrun-stage", action="store_true", help="Explicitly replace this authorized, not-yet-run experimental stage")
    args = parser.parse_args()
    print(build(args.base_stage, args.output_root, args.rebuild_unrun_stage))


if __name__ == "__main__":
    main()
