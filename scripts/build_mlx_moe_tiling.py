#!/usr/bin/env python3
"""Build an isolated prefill MoE tile dispatcher from the pinned stock MLX.

Only a copied quantized.cpp and one added BM16 Metal instance are compiled.
Original native objects and Metal AIR inputs are reused read-only; no author
library, shader, install tree, or Swift source is changed.
This helper neither loads the resulting library nor runs a GPU workload.
"""
import argparse
import difflib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from build_mlx_command_timing import DEFAULT_RUNTIME, PACKAGE, compile_argv, run_command, sha256


QUANTIZED_OBJECT = "CMakeFiles/mlx.dir/mlx/backend/metal/quantized.cpp.o"
DEFAULT_OUTPUT = PACKAGE / "results/moe-prefill-tiling-v2/mlx-tuned"
EXPORTS = ("anemlx_moe_qmm_tile_version", "anemlx_moe_qmm_dispatch_count")
METAL_FUNCTION = "affine_gather_qmm_rhs_nax_nt_bfloat16_t_gs_64_b_4_bm_16_bn_64_bk_64_wm_1_wn_2"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned quantized.cpp anchor must occur once: {old!r}")
    return text.replace(old, new, 1)


def tune_source(original):
    text = replace_once(original, '#include "mlx/utils.h"\n', '''#include "mlx/utils.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
''')
    text = replace_once(text, "namespace mlx::core {\n", '''// Isolated experiment: counters describe successfully encoded target-shape
// dispatches, not GPU completion, executed instructions, or memory traffic.
namespace {
std::atomic<uint64_t> anemlx_moe_qmm_dispatches[3]{}; // BM16, BM32, BM64
}
extern "C" __attribute__((visibility("default")))
int anemlx_moe_qmm_tile_version(void) { return 1; }
extern "C" __attribute__((visibility("default")))
uint64_t anemlx_moe_qmm_dispatch_count(int bm) {
  const int slot = bm == 16 ? 0 : bm == 32 ? 1 : bm == 64 ? 2 : -1;
  return slot < 0 ? 0 :
      anemlx_moe_qmm_dispatches[slot].load(std::memory_order_relaxed);
}

namespace mlx::core {
''')
    low = text.index("void gather_qmm_rhs_nax(\n")
    high = text.index("void gather_qmm_rhs(\n", low)
    body = text[low:high]
    body = replace_once(body, "  int wm = 2, wn = 2;\n", '''  int wm = 2, wn = 2;

  // Only the original Qwen3.8 routed affine BF16/Q4/group64 bank shapes.
  // Other types, banks, dimensions, and dispatcher paths remain untouched.
  const bool anemlx_moe_target =
      mode == "affine" && transpose && bits == 4 && group_size == 64 &&
      E == 512 && w.ndim() == 3 && w.shape(0) == 512 && M > 1 &&
      x.dtype() == bfloat16 && out.dtype() == bfloat16 &&
      w.dtype() == uint32 && scales.dtype() == bfloat16 &&
      biases && biases->dtype() == bfloat16 &&
      ((K == 2560 && N == 640) || (K == 640 && N == 2560));
  if (anemlx_moe_target) {
    // Read at dispatch time, not once at initialization. The caller must
    // drain work before changing this process-wide environment variable.
    const char* requested = std::getenv("ANERUNNER_MOE_QMM_BM");
    if (requested && std::strcmp(requested, "16") == 0) {
      bm = 16;
      wm = 1;
      // Keep WN2: TM1/TN2 uses an existing 16x32 NAX MMA shape.
      // WM2 would give TM0; WM1/WN4 gives unsupported TM1/TN1.
    } else if (requested && std::strcmp(requested, "32") == 0) {
      bm = 32; // Explicit control, with the original WM2/WN2 geometry.
    }
    // Unset, "0", and all other strings preserve the stock BM heuristic.
  }
''')
    body = replace_once(body, "  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);\n", '''  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  if (anemlx_moe_target) {
    const int slot = bm == 16 ? 0 : bm == 32 ? 1 : 2;
    anemlx_moe_qmm_dispatches[slot].fetch_add(1, std::memory_order_relaxed);
  }
''')
    return text[:low] + body + text[high:]


def clone_link(original, new_object, new_library):
    argv = shlex.split(original)
    if argv.count(QUANTIZED_OBJECT) != 1 or argv.count("-o") != 1:
        raise ValueError("Expected one stock quantized object and one link output")
    if any(arg.startswith("@") and arg != "@rpath/libmlx.dylib" for arg in argv):
        raise ValueError("Unreviewed linker response file")
    if any(arg in ("-MF", "-MT", "-MQ", "-MJ", "&&", ";") for arg in argv):
        raise ValueError("Unreviewed linker output or shell directive")
    result = list(argv)
    result[result.index("-o") + 1] = str(new_library)
    result[result.index(QUANTIZED_OBJECT)] = str(new_object)
    return ["-Wl,-rpath,@loader_path" if arg.startswith("-Wl,-rpath,") else arg for arg in result]


def verify_stock_install(old_build, stage):
    """Require byte identity, directly or after the exact CMake install edit.

    No Mach-O section or signature differences are ignored: replay the known
    rpath deletion on a temporary copy and compare the complete resulting file.
    """
    original = old_build / "libmlx.dylib"
    installed = stage / "lib/libmlx.dylib"
    report = {"original_sha256": sha256(original), "installed_sha256": sha256(installed)}
    if report["original_sha256"] == report["installed_sha256"]:
        return {**report, "method": "direct_byte_identity", "exact_match": True}
    install_script = old_build / "cmake_install.cmake"
    rpath = str(old_build / "jaccl")
    anchor = ('execute_process(COMMAND /usr/bin/install_name_tool\n'
              '      -delete_rpath "' + rpath + '"\n'
              '      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmlx.dylib")')
    if install_script.read_text().count(anchor) != 1:
        raise ValueError("Stock libraries differ and the exact CMake rpath install edit is unavailable")
    with tempfile.TemporaryDirectory(prefix="anemlx-stock-install-check-") as temporary:
        copied = Path(temporary) / "libmlx.dylib"
        shutil.copy2(original, copied)
        argv = ["/usr/bin/install_name_tool", "-delete_rpath", rpath, str(copied)]
        process = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        report.update({"method": "exact_CMake_rpath_edit_replayed_on_temporary_copy",
                       "install_script": str(install_script), "install_script_sha256": sha256(install_script),
                       "argv": argv, "exit_code": process.returncode, "output": process.stdout,
                       "transformed_sha256": sha256(copied)})
        if process.returncode or report["transformed_sha256"] != report["installed_sha256"]:
            raise ValueError("Stock library still differs after the exact CMake install edit; refusing overlay")
    return {**report, "exact_match": True}


def static_metal_plan(runtime, old_build, output):
    """Add exactly one instance; preserve the original 42 AIR link inputs."""
    cache = old_build / "CMakeCache.txt"
    if "MLX_METAL_JIT:BOOL=OFF" not in cache.read_text().splitlines():
        raise ValueError("This isolated static-library addition requires the pinned MLX_METAL_JIT=OFF build")
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
            raise ValueError("Unreviewed Metal recipe working directory")
        recipes.append(tokens[3:])
    compiles = [argv for argv in recipes if "-c" in argv and str(stock_source) in argv]
    links = [argv for argv in recipes if "-c" not in argv and "quantized_nax.air" in argv]
    if len(compiles) != 1 or len(links) != 1:
        raise ValueError("Expected exact stock quantized_nax compile and metallib link recipes")
    compile_command, link_command = list(compiles[0]), list(links[0])
    for argv in (compile_command, link_command):
        if argv.count("-o") != 1 or any(arg.startswith("@") or arg in ("&&", ";", "-MF", "-MJ", "-MD", "-MMD") for arg in argv):
            raise ValueError("Unreviewed Metal recipe output or forwarded argument")
    if "-fno-fast-math" not in compile_command or "-ffast-math" in compile_command:
        raise ValueError("Expected the pinned precise Metal compiler flags")
    copied = output / "src/moe_qmm_bm16.metal"
    air = output / "obj/moe_qmm_bm16.air"
    original = stock_source.read_text()
    marker = "#define instantiate_quantized("
    if original.count(marker) != 1:
        raise ValueError("Missing pinned quantized_nax include prefix")
    # Retain the stock includes and Apple attribution; do not instantiate its
    # full matrix of types/shapes again, which would duplicate existing names.
    extra = original[:original.index(marker)] + (
        "// Added isolated BF16/Q4/group64 instance; original shader implementation unchanged.\n"
        'instantiate_kernel("' + METAL_FUNCTION + '",\n'
        "    affine_gather_qmm_rhs_nax, bfloat16_t, 64, 4, 16, 64, 64, 1, 2, true);\n")
    copied.write_text(extra)
    compile_command[compile_command.index(str(stock_source))] = str(copied)
    compile_command[compile_command.index("-o") + 1] = str(air)
    original_airs = [kernel_dir / arg for arg in link_command if arg.endswith(".air")]
    if len(original_airs) != 42 or not all(path.is_file() for path in original_airs):
        raise ValueError("Pinned stock 42-file AIR input set is incomplete")
    link_command = [str(kernel_dir / arg) if arg.endswith(".air") else arg for arg in link_command]
    link_command.insert(link_command.index("-o"), str(air))
    link_command[link_command.index("-o") + 1] = str(output / "lib/mlx.metallib")
    return {
        "commands": [compile_command, link_command],
        "inputs": [cache, makefile, stock_source, *original_airs],
        "outputs": [copied, air], "added_function": METAL_FUNCTION,
        "original_air_count": len(original_airs), "fast_math": False,
        "source": str(copied), "stock_source": str(stock_source),
    }


def build(runtime, output):
    runtime, output = runtime.resolve(), output.resolve()
    if output == runtime or runtime in output.parents:
        raise ValueError("Output must be a fresh directory outside the author runtime")
    old_build = runtime / "lib/.mlx-build/mlx"
    stage = runtime / "lib/mlx"
    source = runtime / "lib/mlx-src/mlx/backend/metal/quantized.cpp"
    database = old_build / "compile_commands.json"
    link_file = old_build / "CMakeFiles/mlx.dir/link.txt"
    entries = json.loads(database.read_text())
    matches = [entry for entry in entries if Path(entry["file"]).resolve() == source]
    if len(matches) != 1 or Path(matches[0]["directory"]).resolve() != old_build:
        raise ValueError("Exact pinned quantized.cpp compile command is unavailable")
    original = source.read_text()
    modified = tune_source(original)
    original_link = link_file.read_text()
    link_tokens = shlex.split(original_link)
    linked_inputs = [old_build / arg for arg in link_tokens if arg.endswith((".o", ".a", ".dylib"))
                     and arg != "@rpath/libmlx.dylib" and arg != "libmlx.dylib"]
    if not linked_inputs or not all(path.is_file() for path in linked_inputs):
        raise ValueError("Stock link inputs are incomplete")
    for name in ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib", "mlx.metallib"):
        if not (stage / "lib" / name).is_file():
            raise ValueError(f"Missing original staged library: {name}")
    if not (stage / "include").is_dir():
        raise ValueError("Original C headers are missing")
    stock_install_check = verify_stock_install(old_build, stage)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "build-provenance.json"
    report = {
        "schema_version": 1, "status": "preparing", "runtime": str(runtime),
        "developer_dir": os.environ.get("DEVELOPER_DIR"),
        "output_root": str(output), "author_stage": str(stage),
        "pinned_stamp": (stage / ".version").read_text().strip(),
        "base": "stock original compile_commands.json/link.txt; no GDN or timing overlay",
        "stock_install_equivalence": stock_install_check,
        "compiled_sources": [str(source), str(output / "src/moe_qmm_bm16.metal")], "commands": [],
        "modes": {
            "0": "Stock BM heuristic: M/E<64 -> BM32, otherwise BM64; WM2/WN2",
            "16": "BM16/BN64/BK64/WM1/WN2; TM1/TN2, original shader unchanged",
            "32": "BM32/BN64/BK64/WM2/WN2; explicit stock-geometry control",
        },
        "scope": "gather_qmm_rhs_nax only; affine, transposed RHS, BF16 x/scales/biases/output, uint32 packed weights, Q4/group64, rank3 E512 bank, M>1, (K,N)=(2560,640) or (640,2560)",
        "environment": "ANERUNNER_MOE_QMM_BM is read on each eligible host dispatch; unset/0/invalid preserve stock geometry",
        "counter_semantics": "Cumulative encoded target dispatches by actual BM16/32/64, including stock mode; use per-trial differences after synchronization. Counts are not GPU completion or physical memory counters.",
        "shader_change": "Original shader math unchanged; add one BM16 BF16/Q4/group64 static instance. MLX_METAL_JIT is OFF, so dispatch alone cannot supply it.",
        "shader_constraint": "BM16 requires WM1 so TM>=1; WN2 gives TN2. WM1/WN4 would yield unsupported TM1/TN1 and is intentionally unavailable.",
        "gpu_workload_started": False, "library_loaded": False,
        "numerical_validation": "not_run", "performance_validation": "not_run",
    }
    try:
        for name in ("src", "obj", "lib", "logs"):
            (output / name).mkdir()
        copied = output / "src/quantized.cpp"
        copied.write_text(modified)
        (output / "src/quantized.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), modified.splitlines(keepends=True),
            fromfile=str(source), tofile=str(copied))))
        for name in ("libmlxc.dylib", "libjaccl.dylib"):
            shutil.copy2(stage / "lib" / name, output / "lib" / name)
        shutil.copy2(stage / ".version", output / ".version")
        (output / "include").symlink_to(stage / "include", target_is_directory=True)
        metal_plan = static_metal_plan(runtime, old_build, output)
        report["static_metal_addition"] = {
            key: value for key, value in metal_plan.items() if key not in ("commands", "inputs", "outputs")}
        metal = runtime / "lib/mlx-src/mlx/backend/metal"
        watched = [source, database, link_file, stage / ".version", old_build / "libmlx.dylib",
                   old_build / "cmake_install.cmake",
                   *[stage / "lib" / name for name in ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib", "mlx.metallib")],
                   metal / "jit_kernels.cpp", metal / "kernels/quantized_nax.h",
                   metal / "kernels/steel/gemm/nax.h", *linked_inputs, *metal_plan["inputs"]]
        watched = list(dict.fromkeys(path.resolve() for path in watched))
        report["original_inputs"] = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                                     for path in watched]
        new_object = output / "obj/quantized.cpp.o"
        compile_command = compile_argv(matches[0], source, copied, new_object, output / "lib/mlx.metallib")
        link_command = clone_link(original_link, new_object, output / "lib/libmlx.dylib")
        report["original_compile_entry"] = matches[0]
        report["original_link_command"] = original_link.strip()
        report["planned_commands"] = [*metal_plan["commands"], compile_command, link_command]
        report["status"] = "building"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        for index, command in enumerate(metal_plan["commands"]):
            run_command(command, output / "obj", output / f"logs/metal-{index}.log", report["commands"])
        if METAL_FUNCTION.encode() not in (output / "lib/mlx.metallib").read_bytes():
            raise RuntimeError("Added BM16 function name is absent from the linked static metallib")
        report["static_metal_addition"]["function_name_found_offline"] = True
        run_command(compile_command, old_build, output / "logs/compile.log", report["commands"])
        run_command(link_command, old_build, output / "logs/link.log", report["commands"])
        run_command(["nm", "-gU", str(output / "lib/libmlx.dylib")], output,
                    output / "logs/symbols.log", report["commands"])
        for name in ("libmlx.dylib", "libmlxc.dylib"):
            run_command(["otool", "-L", str(output / "lib" / name)], output,
                        output / f"logs/{name}-linkage.log", report["commands"])
        symbols = (output / "logs/symbols.log").read_text()
        for name in EXPORTS:
            if "_" + name not in symbols:
                raise RuntimeError(f"Missing exported identity/counter: {name}")
        changed = [item["path"] for item in report["original_inputs"] if sha256(item["path"]) != item["sha256"]]
        if changed:
            raise RuntimeError(f"Original build inputs changed: {changed}")
        report["original_inputs_unchanged"] = True
        report["status"] = "built_not_executed"
        report["outputs"] = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                             for path in (copied, new_object, *metal_plan["outputs"], output / ".version", *sorted((output / "lib").iterdir()))]
        report["runtime_selection"] = {
            "DYLD_LIBRARY_PATH": str(output / "lib"), "ANERUNNER_MLX_ROOT": str(output),
            "identity": "extern C int anemlx_moe_qmm_tile_version(void) returns 1",
            "counter": "extern C uint64_t anemlx_moe_qmm_dispatch_count(int bm), bm=16/32/64; invalid bm returns 0",
            "selection": "Set ANERUNNER_MOE_QMM_BM to 0, 16, or 32 before constructing/evaluating each trial; drain before changing it. Environment mutation is not a thread-safe request-local API.",
            "metallib": "Colocated static mlx.metallib contains all original AIR objects plus exactly one added BM16 instance, compiled with original -fno-fast-math flags.",
            "validation": "The helper checked native exports/linkage and the added Metal function name offline, but did not dlopen or execute Metal. The runner must verify function load, identity, counter deltas, output equality, and timing separately.",
        }
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", "--output-root", dest="output", type=Path, default=DEFAULT_OUTPUT,
                        help="Fresh directory outside the author runtime; existing paths are rejected")
    args = parser.parse_args()
    try:
        print(build(args.runtime_root, args.output))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"build failed: {error}\n")


if __name__ == "__main__":
    main()
