#!/usr/bin/env python3
"""Build five isolated sorted-Q4 NAX dispatch configurations, without GPU use.

Reuse the original native objects and all 42 AIR inputs. Compile only a copied
quantized.cpp and three extra template instances of Apple's unchanged shader.
Neither the author runtime nor earlier experiment libraries are modified.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import shlex
import shutil

from build_mlx_command_timing import DEFAULT_RUNTIME, PACKAGE, compile_argv, run_command, sha256
from build_mlx_moe_tiling import clone_link, replace_once, verify_stock_install


DEFAULT_OUTPUT = PACKAGE / "results/moe-prefill-autotune-v1/mlx-tuned"
DEFAULT_DEVELOPER = Path("/Applications/Xcode.app/Contents/Developer")
EXPORTS = ("anemlx_moe_qmm_tune_version", "anemlx_moe_qmm_tune_dispatch_count")
# Tuple order is BM, BN, BK, WM, WN, matching affine_gather_qmm_rhs_nax.
# Config 0 retains the original BM heuristic; the tuple is its short-prefill
# geometry. Config 1 is already present in the original static metallib.
GEOMETRIES = {
    0: (32, 64, 64, 2, 2),
    1: (64, 64, 64, 2, 2),
    2: (32, 128, 64, 2, 2),
    3: (32, 64, 32, 2, 2),
    4: (32, 32, 64, 2, 1),
}


def metal_name(geometry):
    bm, bn, bk, wm, wn = geometry
    return ("affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_64_b_4"
            f"_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}")


def validate_geometry(geometry):
    bm, bn, bk, wm, wn = geometry
    if bm % wm or bn % wn or (bm // wm) % 16 or (bn // wn) % 16:
        raise ValueError(f"Invalid NAX fragment division: {geometry}")
    tm, tn = bm // wm // 16, bn // wn // 16
    if tm < 1 or tn < 1 or not ((tn == 1 and tm % 2 == 0) or tn % 2 == 0):
        raise ValueError(f"No pinned tile_matmad_nax branch: {geometry}")
    if bk > 64 or 64 % bk or bk % 32:
        raise ValueError(f"Unsupported group64 loader / SK32: {geometry}")
    threads, packed_columns = 32 * wm * wn, bk // 2
    reads = max(1, packed_columns * bn // threads)
    if packed_columns * bn % threads or packed_columns % reads:
        raise ValueError(f"Loader threads would cross a quantization row: {geometry}")
    return {"TM": tm, "TN": tn, "threads": threads,
            "loader_packed_bytes_per_thread": reads, "scale_group_steps": 64 // bk}


def tune_source(original):
    text = replace_once(original, '#include "mlx/utils.h"\n', '''#include "mlx/utils.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
''')
    text = replace_once(text, "namespace mlx::core {\n", '''// Experiment counters describe encoded target dispatches, never GPU traffic.
namespace {
std::atomic<uint64_t> anemlx_moe_qmm_tune_counts[5]{};
}
extern "C" __attribute__((visibility("default")))
int anemlx_moe_qmm_tune_version(void) { return 1; }
extern "C" __attribute__((visibility("default")))
uint64_t anemlx_moe_qmm_tune_dispatch_count(int id) {
  return id < 0 || id > 4 ? 0 :
      anemlx_moe_qmm_tune_counts[id].load(std::memory_order_relaxed);
}

namespace mlx::core {
''')
    low = text.index("void gather_qmm_rhs_nax(\n")
    high = text.index("void gather_qmm_rhs(\n", low)
    body = text[low:high]
    dispatch = '''  int wm = 2, wn = 2;

  // Sorted RHS NAX path only; preserve every unrelated operator and dtype.
  const bool anemlx_moe_target =
      mode == "affine" && transpose && bits == 4 && group_size == 64 &&
      E == 512 && w.ndim() == 3 && w.shape(0) == 512 && M > 1 &&
      x.dtype() == bfloat16 && out.dtype() == bfloat16 &&
      w.dtype() == uint32 && scales.dtype() == bfloat16 &&
      biases && biases->dtype() == bfloat16 &&
      ((K == 2560 && N == 640) || (K == 640 && N == 2560));
  int anemlx_moe_config = 0;
  if (anemlx_moe_target) {
    // Read on each dispatch. The caller must drain work before changing this
    // process-wide environment; it is not a concurrent request-local policy.
    const char* requested = std::getenv("ANERUNNER_MOE_QMM_CONFIG");
'''
    for config_id, geometry in GEOMETRIES.items():
        if config_id == 0:
            continue
        bm, bn, bk, wm, wn = geometry
        dispatch += (f'    if (requested && std::strcmp(requested, "{config_id}") == 0) {{\n'
                     f"      anemlx_moe_config = {config_id};\n"
                     f"      bm = {bm}; bn = {bn}; bk = {bk}; wm = {wm}; wn = {wn};\n"
                     "    }\n")
    dispatch += "    // Unset, 0, and unrecognized values preserve the stock heuristic.\n  }\n"
    body = replace_once(body, "  int wm = 2, wn = 2;\n", dispatch)
    body = replace_once(body, "  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);\n", '''  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  if (anemlx_moe_target) {
    anemlx_moe_qmm_tune_counts[anemlx_moe_config].fetch_add(1, std::memory_order_relaxed);
  }
''')
    return text[:low] + body + text[high:]


def static_metal_plan(runtime, old_build, output):
    cache = old_build / "CMakeCache.txt"
    if "MLX_METAL_JIT:BOOL=OFF" not in cache.read_text().splitlines():
        raise ValueError("Expected pinned MLX_METAL_JIT=OFF")
    kernel_dir = old_build / "mlx/backend/metal/kernels"
    makefile = kernel_dir / "CMakeFiles/mlx-metallib.dir/build.make"
    stock_source = runtime / "lib/mlx-src/mlx/backend/metal/kernels/quantized_nax.metal"
    recipes = []
    for line in makefile.read_text().splitlines():
        if not line.startswith("\tcd "):
            continue
        tokens = shlex.split(line)
        if tokens[3:7] != ["xcrun", "-sdk", "macosx", "metal"]:
            continue
        if tokens[2] != "&&" or Path(tokens[1]).resolve() != kernel_dir:
            raise ValueError("Unreviewed Metal working directory")
        recipes.append(tokens[3:])
    compiles = [a for a in recipes if "-c" in a and str(stock_source) in a]
    links = [a for a in recipes if "-c" not in a and "quantized_nax.air" in a]
    if len(compiles) != 1 or len(links) != 1:
        raise ValueError("Expected exact pinned Metal compile and link recipes")
    compile_command, link_command = list(compiles[0]), list(links[0])
    for argv in (compile_command, link_command):
        if argv.count("-o") != 1 or any(a.startswith("@") or a in ("&&", ";", "-MF", "-MJ", "-MD", "-MMD") for a in argv):
            raise ValueError("Unreviewed Metal output/forwarded flags")
    if "-fno-fast-math" not in compile_command or "-ffast-math" in compile_command:
        raise ValueError("Expected pinned precise Metal flags")
    copied, air = output / "src/moe_qmm_autotune.metal", output / "obj/moe_qmm_autotune.air"
    original = stock_source.read_text()
    marker = "#define instantiate_quantized("
    if original.count(marker) != 1:
        raise ValueError("Pinned quantized_nax include prefix missing")
    extra = original[:original.index(marker)]
    extra += "// Isolated BF16/Q4/group64 instances; Apple shader math is unchanged.\n"
    names = []
    for config_id in (2, 3, 4):
        geometry = GEOMETRIES[config_id]
        validate_geometry(geometry)
        name = metal_name(geometry)
        names.append(name)
        args = ", ".join(map(str, geometry))
        extra += f'instantiate_kernel("{name}",\n    affine_gather_qmm_rhs_nax, bfloat16_t, 64, 4, {args}, true);\n'
    copied.write_text(extra)
    compile_command[compile_command.index(str(stock_source))] = str(copied)
    compile_command[compile_command.index("-o") + 1] = str(air)
    original_airs = [kernel_dir / a for a in link_command if a.endswith(".air")]
    if len(original_airs) != 42 or not all(p.is_file() for p in original_airs):
        raise ValueError("The original 42 AIR inputs are required")
    link_command = [str(kernel_dir / a) if a.endswith(".air") else a for a in link_command]
    link_command.insert(link_command.index("-o"), str(air))
    link_command[link_command.index("-o") + 1] = str(output / "lib/mlx.metallib")
    return {"commands": [compile_command, link_command], "inputs": [cache, makefile, stock_source, *original_airs],
            "outputs": [copied, air], "added_functions": names, "original_air_count": 42, "fast_math": False}


def build(runtime, output):
    runtime, output = runtime.resolve(), output.resolve()
    if output == runtime or runtime in output.parents:
        raise ValueError("Output must be fresh and outside the author runtime")
    old_build, stage = runtime / "lib/.mlx-build/mlx", runtime / "lib/mlx"
    source = runtime / "lib/mlx-src/mlx/backend/metal/quantized.cpp"
    database, link_file = old_build / "compile_commands.json", old_build / "CMakeFiles/mlx.dir/link.txt"
    entries = json.loads(database.read_text())
    matches = [e for e in entries if Path(e["file"]).resolve() == source]
    if len(matches) != 1 or Path(matches[0]["directory"]).resolve() != old_build:
        raise ValueError("Exact pinned quantized.cpp compile command is unavailable")
    original, original_link = source.read_text(), link_file.read_text()
    modified = tune_source(original)
    linked_inputs = [old_build / a for a in shlex.split(original_link)
                     if a.endswith((".o", ".a", ".dylib")) and a not in ("@rpath/libmlx.dylib", "libmlx.dylib")]
    if not linked_inputs or not all(p.is_file() for p in linked_inputs):
        raise ValueError("Original native link inputs are incomplete")
    for name in ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib", "mlx.metallib"):
        if not (stage / "lib" / name).is_file():
            raise ValueError(f"Missing staged dependency: {name}")
    if not (stage / "include").is_dir():
        raise ValueError("Original C headers are missing")
    equivalence = verify_stock_install(old_build, stage)
    modes = {str(i): {"geometry_BM_BN_BK_WM_WN": g, **validate_geometry(g)} for i, g in GEOMETRIES.items()}
    modes["0"]["policy"] = "Stock heuristic M/E<64 -> BM32, otherwise BM64; tuple describes BM32 branch"
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "build-provenance.json"
    report = {"schema_version": 1, "status": "preparing", "runtime": str(runtime), "output_root": str(output),
              "developer_dir": os.environ.get("DEVELOPER_DIR"), "author_stage": str(stage),
              "pinned_stamp": (stage / ".version").read_text().strip(), "stock_install_equivalence": equivalence,
              "base": "Original native objects and static AIRs; no prior timing/GDN/tiling overlay", "modes": modes,
              "scope": "Sorted RHS NAX affine Q4/group64, BF16 x/scales/biases/out, U32 rank3 E512 bank, M>1, (K,N)=(2560,640) or (640,2560)",
              "environment": "ANERUNNER_MOE_QMM_CONFIG read on each eligible dispatch; unset/0/unknown use stock and counter0",
              "counter_semantics": "Encoded target QMM dispatches per selected config id, not completion or physical memory; invalid counter id returns 0",
              "gpu_workload_started": False, "library_loaded": False, "numerical_validation": "not_run", "performance_validation": "not_run", "commands": []}
    try:
        for name in ("src", "obj", "lib", "logs"):
            (output / name).mkdir()
        copied, new_object = output / "src/quantized.cpp", output / "obj/quantized.cpp.o"
        copied.write_text(modified)
        (output / "src/quantized.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), modified.splitlines(keepends=True), fromfile=str(source), tofile=str(copied))))
        for name in ("libmlxc.dylib", "libjaccl.dylib"):
            shutil.copy2(stage / "lib" / name, output / "lib" / name)
        shutil.copy2(stage / ".version", output / ".version")
        (output / "include").symlink_to(stage / "include", target_is_directory=True)
        metal_plan = static_metal_plan(runtime, old_build, output)
        report["static_metal_addition"] = {k: v for k, v in metal_plan.items() if k not in ("commands", "inputs", "outputs")}
        metal = runtime / "lib/mlx-src/mlx/backend/metal"
        watched = [Path(__file__), source, database, link_file, stage / ".version", old_build / "libmlx.dylib",
                   old_build / "cmake_install.cmake", metal / "kernels/quantized_nax.h", metal / "kernels/steel/gemm/nax.h",
                   *[stage / "lib" / n for n in ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib", "mlx.metallib")],
                   *linked_inputs, *metal_plan["inputs"]]
        watched = list(dict.fromkeys(p.resolve() for p in watched))
        report["original_inputs"] = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in watched]
        compile_command = compile_argv(matches[0], source, copied, new_object, output / "lib/mlx.metallib")
        link_command = clone_link(original_link, new_object, output / "lib/libmlx.dylib")
        report.update({"status": "building", "original_compile_entry": matches[0], "original_link_command": original_link.strip(),
                       "planned_commands": [*metal_plan["commands"], compile_command, link_command]})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        for i, command in enumerate(metal_plan["commands"]):
            run_command(command, output / "obj", output / f"logs/metal-{i}.log", report["commands"])
        metallib = (output / "lib/mlx.metallib").read_bytes()
        for geometry in GEOMETRIES.values():
            if metal_name(geometry).encode() not in metallib:
                raise RuntimeError(f"Missing static Metal function: {metal_name(geometry)}")
        report["static_metal_addition"]["all_function_names_found_offline"] = True
        run_command(compile_command, old_build, output / "logs/compile.log", report["commands"])
        run_command(link_command, old_build, output / "logs/link.log", report["commands"])
        run_command(["nm", "-gU", str(output / "lib/libmlx.dylib")], output, output / "logs/symbols.log", report["commands"])
        for name in ("libmlx.dylib", "libmlxc.dylib"):
            run_command(["otool", "-L", str(output / "lib" / name)], output, output / f"logs/{name}-linkage.log", report["commands"])
        symbols = (output / "logs/symbols.log").read_text()
        if any("_" + name not in symbols for name in EXPORTS):
            raise RuntimeError("Missing exported tune identity/counter")
        if any(sha256(i["path"]) != i["sha256"] for i in report["original_inputs"]):
            raise RuntimeError("Original build inputs changed during compilation")
        report.update({"status": "built_not_executed", "original_inputs_unchanged": True,
                       "outputs": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                                   for p in (copied, new_object, *metal_plan["outputs"], output / ".version", *sorted((output / "lib").iterdir()))],
                       "runtime_selection": {"DYLD_LIBRARY_PATH": str(output / "lib"), "ANERUNNER_MLX_ROOT": str(output),
                           "identity": "anemlx_moe_qmm_tune_version() returns 1", "counter": "anemlx_moe_qmm_tune_dispatch_count(id), id 0..4",
                           "selection": "Set config before encoding/evaluation; drain before switching. Env changes are not a concurrent request-local API.",
                           "metallib": "Original device object prefers colocated mlx.metallib; new lib includes 42 original AIRs plus 3 new geometries in one AIR",
                           "validation": "Offline symbol/linkage/function-name checks only. No dlopen or GPU execution."}})
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", "--output-root", dest="output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--developer-dir", type=Path, default=Path(os.environ.get("DEVELOPER_DIR", DEFAULT_DEVELOPER)))
    args = parser.parse_args()
    try:
        if not args.developer_dir.is_dir():
            raise ValueError("Xcode Developer directory is unavailable")
        os.environ["DEVELOPER_DIR"] = str(args.developer_dir.resolve())
        print(build(args.runtime_root, args.output))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"build failed: {error}\n")


if __name__ == "__main__":
    main()
