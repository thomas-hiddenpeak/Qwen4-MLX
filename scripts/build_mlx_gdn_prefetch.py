#!/usr/bin/env python3
"""Build isolated modes 7/8: BF16 QKV GEMV with four-tile load lookahead.

Requires the existing successful GEMM and BM1/2 experimental stages. Reuses
their compiler recipes and read-only objects; never loads MLX or runs a GPU.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil

from build_mlx_command_timing import PACKAGE, compile_argv, run_command, sha256
from build_mlx_gemv_tuning import replace_once


NATIVE_SOURCE = PACKAGE / "native/gdn_prefetch.metal"


def tune_source(original):
    text = replace_once(original, "mode <= 6 ? mode : 0", "mode <= 8 ? mode : 0")
    text = replace_once(text, "int anemlx_gdn_gemv_max_mode(void) { return 6; }",
                        "int anemlx_gdn_gemv_max_mode(void) { return 8; }")
    text = replace_once(text, "std::atomic<int> anemlx_gdn_gemv_mode{0};",
                        "std::atomic<int> anemlx_gdn_gemv_mode{0};\n"
                        "std::atomic<uint64_t> anemlx_gdn_prefetch_dispatches[2]{};")
    text = replace_once(text, "int anemlx_gdn_gemv_max_mode(void) { return 8; }",
                        """int anemlx_gdn_gemv_max_mode(void) { return 8; }
extern "C" __attribute__((visibility("default")))
uint64_t anemlx_gdn_prefetch_dispatch_count(int mode) {
  return mode == 7 || mode == 8
      ? anemlx_gdn_prefetch_dispatches[mode - 7].load(std::memory_order_relaxed)
      : 0;
}""")
    anchor = '    kname << "gemv_" << type_to_name(out);'
    replacement = '''    const int anemlx_prefetch_mode =
        anemlx_gdn_gemv_mode.load(std::memory_order_relaxed);
    // The fixed kernel has no tail, batching, axpby, or non-contiguous path.
    const bool anemlx_prefetch_qkv =
        (anemlx_prefetch_mode == 7 || anemlx_prefetch_mode == 8) &&
        !CHECK_AB && M == 1 && N == 10240 && K == 2560 &&
        !transpose_a && transpose_b && batch_size_out == 1 &&
        contiguous_kernel && mat_ld == K &&
        mat.dtype() == bfloat16 && vec.dtype() == bfloat16 &&
        out.dtype() == bfloat16 && mat.strides()[mat.ndim() - 2] == 1 &&
        mat.strides().back() == K &&
        mat.offset() % 8 == 0 && vec.offset() % 8 == 0 &&
        vec.flags().row_contiguous && bm == 8 && bn == 1 &&
        sm == 1 && sn == 32 && tm == 4 && tn == 4;
    if (anemlx_prefetch_qkv) {
      anemlx_gdn_prefetch_dispatches[anemlx_prefetch_mode - 7].fetch_add(
          1, std::memory_order_relaxed);
    }
    kname << (anemlx_prefetch_qkv
                  ? (anemlx_prefetch_mode == 7 ? "gdn_prefetch4_"
                                               : "gdn_prefetch4_vector_")
                  : "gemv_")
          << type_to_name(out);'''
    return replace_once(text, anchor, replacement)


def select_command(report, predicate, description):
    matches = [item for item in report["commands"] if predicate(item["argv"])]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {description}")
    return matches[0]


def redirect_output(argv, output):
    result = list(argv)
    if result.count("-o") != 1:
        raise ValueError("Expected exactly one compiler/linker output")
    result[result.index("-o") + 1] = str(output)
    return result


def build(base, output):
    base, output = base.resolve(), output.resolve()
    report_path = base / "build-provenance.json"
    previous = json.loads(report_path.read_text())
    if previous.get("status") != "built_not_executed" or previous.get("maximum_mode") != 6:
        raise ValueError("Expected the successful isolated GEMM tuning stage")
    metal_stage = Path(previous["base_stage"]).resolve()
    metal_report_path = metal_stage / "build-provenance.json"
    metal_report = json.loads(metal_report_path.read_text())
    runtime = Path(metal_report["runtime"]).resolve()
    if metal_report.get("status") != "built_not_executed" or metal_report.get("maximum_mode") != 4:
        raise ValueError("Expected the successful BM1/2 Metal build")
    for protected in (runtime, base, metal_stage):
        if output == protected or protected in output.parents:
            raise ValueError("Output must be fresh and outside all source stages")
    for parent, metadata in ((base, previous), (metal_stage, metal_report)):
        for item in metadata["outputs"]:
            path = Path(item["path"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Recorded stage output changed: {path}")

    source = base / "src/matmul.cpp"
    original = source.read_text()
    modified = tune_source(original)
    cpp_compile = select_command(previous,
        lambda a: "-c" in a and str(source) in a, "C++ compile command")
    cpp_link = select_command(previous,
        lambda a: "-c" not in a and str(base / "obj/matmul.cpp.o") in a,
        "C++ link command")
    metal_source = metal_stage / "src/gemv.metal"
    metal_compile = select_command(metal_report,
        lambda a: "-c" in a and str(metal_source) in a, "Metal compile command")
    metal_link = select_command(metal_report,
        lambda a: "-c" not in a and str(metal_stage / "obj/gemv.air") in a,
        "Metal link command")
    if not any(arg.endswith("/nojit_kernels.cpp.o") for arg in cpp_link["argv"]):
        raise ValueError("Fixed candidate names require the pinned non-JIT Metal dispatcher")
    if "-fno-fast-math" not in metal_compile["argv"]:
        raise ValueError("Expected the pinned no-fast-math Metal compilation")

    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "status": "building", "base_stage": str(base),
        "metal_stage": str(metal_stage), "runtime": str(runtime),
        "developer_dir": os.environ.get("DEVELOPER_DIR"),
        "maximum_mode": 8, "commands": [], "timing_automatically_started": False,
        "modes": {"7": "prefetch4: four K tiles, scalar BF16 loads",
                  "8": "prefetch4Vector: four K tiles, aligned bfloat4 loads"},
        "scope": "Single-vector original BF16 QKV K2560/N10240 only; BM8/BN1/SM1/SN32/TM4/TN4 and scalar accumulation/shuffle order unchanged. Modes 0-6 are preserved.",
        "numerical_validation": "not_run", "performance_validation": "not_run",
    }
    try:
        for name in ("src", "obj", "lib", "logs"):
            (output / name).mkdir()
        copied_cpp = output / "src/matmul.cpp"
        copied_cpp.write_text(modified)
        (output / "src/matmul.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), modified.splitlines(keepends=True),
            fromfile=str(source), tofile=str(copied_cpp))))
        copied_metal = output / "src/gdn_prefetch.metal"
        shutil.copy2(NATIVE_SOURCE, copied_metal)
        for name in ("libmlxc.dylib", "libjaccl.dylib"):
            shutil.copy2(base / "lib" / name, output / "lib" / name)
        (output / "include").symlink_to((base / "include").resolve(), target_is_directory=True)

        new_object = output / "obj/matmul.cpp.o"
        cpp_args = compile_argv({"arguments": cpp_compile["argv"]}, source,
                               copied_cpp, new_object, output / "lib/mlx.metallib")
        cpp_link_args = redirect_output(cpp_link["argv"], output / "lib/libmlx.dylib")
        cpp_link_args[cpp_link_args.index(str(base / "obj/matmul.cpp.o"))] = str(new_object)
        metal_args = redirect_output(metal_compile["argv"], output / "obj/gdn_prefetch.air")
        metal_args[metal_args.index(str(metal_source))] = str(copied_metal)
        metal_link_args = redirect_output(metal_link["argv"], output / "lib/mlx.metallib")
        metal_link_args.append(str(output / "obj/gdn_prefetch.air"))

        # Every pre-existing link input is watched, not copied over or rebuilt.
        watched = {source, metal_source, NATIVE_SOURCE, report_path, metal_report_path,
                   runtime / "lib/mlx-src/mlx/backend/metal/kernels/gemv.h"}
        for command in (cpp_link, metal_link):
            for arg in command["argv"]:
                if arg.endswith((".o", ".air")):
                    path = Path(arg)
                    watched.add(path if path.is_absolute() else Path(command["cwd"]) / path)
        for item in metal_report["original_inputs"]:
            watched.add(Path(item["path"]))
        report["original_inputs"] = [{"path": str(p), "sha256": sha256(p)}
                                     for p in sorted(watched)]
        report["planned_commands"] = [metal_args, metal_link_args, cpp_args, cpp_link_args]
        for argv, cwd, log in (
            (metal_args, output / "obj", "metal-compile.log"),
            (metal_link_args, output / "obj", "metal-link.log"),
            (cpp_args, Path(cpp_compile["cwd"]), "cpp-compile.log"),
            (cpp_link_args, Path(cpp_link["cwd"]), "cpp-link.log"),
        ):
            run_command(argv, cwd, output / "logs" / log, report["commands"])
        run_command(["nm", "-gU", str(output / "lib/libmlx.dylib")], output,
                    output / "logs/symbols.log", report["commands"])
        symbols = (output / "logs/symbols.log").read_text()
        for name in ("anemlx_set_gdn_gemv_mode", "anemlx_gdn_gemv_max_mode",
                     "anemlx_gdn_prefetch_dispatch_count",
                     "anemlx_timing_start", "anemlx_timing_stop", "anemlx_timing_version"):
            if "_" + name not in symbols:
                raise RuntimeError(f"Missing export: {name}")
        if any(sha256(item["path"]) != item["sha256"] for item in report["original_inputs"]):
            raise RuntimeError("An original build input changed")
        report["original_inputs_unchanged"] = True
        report["status"] = "built_not_executed"
        report["outputs"] = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                             for p in (copied_cpp, copied_metal, new_object,
                                       output / "obj/gdn_prefetch.air",
                                       output / "lib/mlx.metallib", output / "lib/libmlx.dylib")]
        report["runtime_selection"] = {
            "DYLD_LIBRARY_PATH": str(output / "lib"), "ANERUNNER_MLX_ROOT": str(output),
            "setter": "anemlx_set_gdn_gemv_mode(7 or 8)",
            "maximum_mode_getter": "anemlx_gdn_gemv_max_mode returns 8",
        }
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        (output / "build-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return output / "build-provenance.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-stage", type=Path, default=PACKAGE / "results/gdn-gemv-gemm/native")
    parser.add_argument("--output-root", type=Path, default=PACKAGE / "results/gdn-prefetch-v1/native")
    parser.add_argument("--developer-dir", type=Path,
                        default=Path(os.environ.get("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")))
    args = parser.parse_args()
    os.environ["DEVELOPER_DIR"] = str(args.developer_dir.resolve())
    try:
        print(build(args.base_stage, args.output_root))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"build failed: {error}\n")


if __name__ == "__main__":
    main()
