#!/usr/bin/env python3
"""Build an isolated GDN GEMV dispatcher; never load the library or run a GPU.

Only a copied matmul.cpp is compiled, plus gemv.metal when --extra-bm is set.
Original MLX link objects and the existing command-timing device.cpp object are
reused read-only. Mode defaults to zero;
the caller explicitly selects a mode with anemlx_set_gdn_gemv_mode(int).
"""
import argparse
import difflib
import json
from pathlib import Path
import shlex
import shutil

from build_mlx_command_timing import (
    DEFAULT_RUNTIME, PACKAGE, compile_argv, link_argv, run_command, sha256,
)


MATMUL_OBJECT = "CMakeFiles/mlx.dir/mlx/backend/metal/matmul.cpp.o"
DEFAULT_TIMING = PACKAGE / "results/gpu-bottleneck-v1/native"
DEFAULT_OUTPUT = PACKAGE / "results/gdn-gemv-quick/native"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned matmul.cpp anchor must occur once: {old!r}")
    return text.replace(old, new, 1)


def tune_source(original, extra_bm=False):
    text = replace_once(original, "#include <algorithm>\n", "#include <algorithm>\n#include <atomic>\n")
    text = replace_once(text, "namespace mlx::core {\n", """// Experimental process-local dispatch switch; no timing is started here.
namespace {
std::atomic<int> anemlx_gdn_gemv_mode{0};
}
extern "C" __attribute__((visibility("default")))
void anemlx_set_gdn_gemv_mode(int mode) {
  anemlx_gdn_gemv_mode.store(mode >= 0 && mode <= 2 ? mode : 0,
                             std::memory_order_relaxed);
}
extern "C" __attribute__((visibility("default")))
int anemlx_gdn_gemv_max_mode(void) { return 2; }

namespace mlx::core {
""")
    anchor = """    n_out_per_tgp = bm * sm * tm;
    kname << "gemv_" << type_to_name(out);
"""
    replacement = """    // Restrict the experiment to the three unfused GDN projection shapes
    // with original BF16 matrix/vector storage, on the non-transposed path.
    const bool anemlx_qkv_or_z =
        K == 2560 && (out_vector_len == 10240 || out_vector_len == 6144);
    const bool anemlx_gdn_out = K == 6144 && out_vector_len == 2560;
    if (out.dtype() == bfloat16 && mat.dtype() == bfloat16 &&
        vec.dtype() == bfloat16 && (anemlx_qkv_or_z || anemlx_gdn_out)) {
      const int anemlx_mode =
          anemlx_gdn_gemv_mode.load(std::memory_order_relaxed);
      if (anemlx_mode == 1 && anemlx_qkv_or_z) {
        bm = 4;
      } else if (anemlx_mode == 2) {
        bm = 4;
        tm = 1;
      }
    }
    n_out_per_tgp = bm * sm * tm;
    kname << "gemv_" << type_to_name(out);
"""
    if extra_bm:
        text = replace_once(text, "mode <= 2 ? mode : 0", "mode <= 4 ? mode : 0")
        text = replace_once(text, "int anemlx_gdn_gemv_max_mode(void) { return 2; }",
                            "int anemlx_gdn_gemv_max_mode(void) { return 4; }")
        replacement = replace_once(replacement, "        tm = 1;\n      }", """        tm = 1;
      } else if (anemlx_mode == 3 || anemlx_mode == 4) {
        bm = anemlx_mode == 3 ? 2 : 1;
        tm = 4;
      }""")
    return replace_once(text, anchor, replacement)


def extra_metal_plan(runtime, old_build, output):
    """Reuse reviewed make recipes as argv, redirecting all output paths."""
    source = runtime / "lib/mlx-src/mlx/backend/metal/kernels/gemv.metal"
    kernel_dir = old_build / "mlx/backend/metal/kernels"
    makefile = kernel_dir / "CMakeFiles/mlx-metallib.dir/build.make"
    original = source.read_text()
    anchor = "instantiate_gemv_blocks(bfloat16, bfloat16_t);\n"
    modified = replace_once(original, anchor, anchor + """
// Additional BF16 GDN-only dispatch variants; original kernels remain intact.
instantiate_gemv(bfloat16, bfloat16_t, 2, 1, 1, 32, 4, 4);
instantiate_gemv(bfloat16, bfloat16_t, 1, 1, 1, 32, 4, 4);
""")
    recipes = []
    for line in makefile.read_text().splitlines():
        if not line.startswith("\tcd "):
            continue
        tokens = shlex.split(line)
        if tokens[3:7] != ["xcrun", "-sdk", "macosx", "metal"]:
            continue
        if len(tokens) < 5 or tokens[2] != "&&" or Path(tokens[1]).resolve() != kernel_dir:
            raise ValueError("Unreviewed Metal recipe working directory")
        recipes.append(tokens[3:])
    compiles = [a for a in recipes if "-c" in a and str(source) in a]
    links = [a for a in recipes if "-c" not in a and "gemv.air" in a]
    if len(compiles) != 1 or len(links) != 1:
        raise ValueError("Expected one pinned gemv compile and one metallib link recipe")
    copied = output / "src/gemv.metal"
    air = output / "obj/gemv.air"
    commands = []
    watched = [source, makefile]
    for args, destination in ((compiles[0], air), (links[0], output / "lib/mlx.metallib")):
        if args[:4] != ["xcrun", "-sdk", "macosx", "metal"] or args.count("-o") != 1:
            raise ValueError("Unexpected Metal compiler/linker recipe")
        if any(a.startswith("@") or a in ("&&", ";", "-MF", "-MJ", "-MD", "-MMD") for a in args):
            raise ValueError("Unreviewed forwarded Metal recipe or dependency output")
        rewritten = list(args)
        rewritten[rewritten.index("-o") + 1] = str(destination)
        for i, arg in enumerate(rewritten):
            if arg == str(source):
                rewritten[i] = str(copied)
            elif arg.endswith(".air") and i != rewritten.index("-o") + 1:
                original_air = kernel_dir / arg
                if not original_air.is_file():
                    raise ValueError(f"Missing original Metal object: {original_air}")
                watched.append(original_air)
                rewritten[i] = str(air if arg == "gemv.air" else original_air)
        commands.append(rewritten)
    copied.write_text(modified)
    (output / "src/gemv.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile=str(source), tofile=str(copied))))
    return commands, watched, [copied, air, output / "lib/mlx.metallib"]


def build(runtime, timing_stage, output, extra_bm=False):
    runtime, timing_stage, output = runtime.resolve(), timing_stage.resolve(), output.resolve()
    if output == runtime or runtime in output.parents or output == timing_stage or timing_stage in output.parents:
        raise ValueError("Output must be a new directory outside both source stages")
    old_build = runtime / "lib/.mlx-build/mlx"
    stage = runtime / "lib/mlx"
    source = runtime / "lib/mlx-src/mlx/backend/metal/matmul.cpp"
    database = old_build / "compile_commands.json"
    link_file = old_build / "CMakeFiles/mlx.dir/link.txt"
    timing_provenance_path = timing_stage / "build-provenance.json"
    timing_provenance = json.loads(timing_provenance_path.read_text())
    timing_object = timing_stage / "obj/device.cpp.o"
    if (timing_provenance.get("status") != "built_not_executed"
            or Path(timing_provenance.get("runtime", "")).resolve() != runtime
            or timing_provenance.get("pinned_stamp") != (stage / ".version").read_text().strip()):
        raise ValueError("Timing object is not from this successful pinned build")
    expected = [x for x in timing_provenance["outputs"] if Path(x["path"]) == timing_object]
    if len(expected) != 1 or sha256(timing_object) != expected[0]["sha256"]:
        raise ValueError("Timing device object checksum mismatch")
    entries = json.loads(database.read_text())
    matches = [e for e in entries if Path(e["file"]).resolve() == source]
    if len(matches) != 1 or Path(matches[0]["directory"]).resolve() != old_build:
        raise ValueError("Exact pinned matmul.cpp compile command is unavailable")
    original = source.read_text()
    modified = tune_source(original, extra_bm)
    original_link = link_file.read_text()
    objects = [old_build / x for x in shlex.split(original_link) if x.endswith(".o")]
    if not objects or not all(x.is_file() for x in objects):
        raise ValueError("Pinned link object set is incomplete")
    # Existing objects may embed their original absolute metallib paths. Both
    # original stages are preserved; this build does not rewrite those objects.
    for root in (stage, timing_stage):
        if not (root / "lib/mlx.metallib").is_file():
            raise ValueError("An existing object's metallib dependency is missing")
    if sha256(stage / "lib/mlx.metallib") != sha256(timing_stage / "lib/mlx.metallib"):
        raise ValueError("Timing and author metallib content differs")
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "build-provenance.json"
    report = {
        "schema_version": 1, "status": "preparing", "runtime": str(runtime),
        "output_root": str(output), "timing_stage": str(timing_stage),
        "commands": [], "pinned_stamp": (stage / ".version").read_text().strip(),
        "modes": {
            "0": "Original dispatch (default; invalid modes also select zero)",
            "1": "QKV/Z only: BM8 -> BM4; TM4 unchanged; GDN out unchanged",
            "2": "QKV/Z/out: BM4 TM1; BN1 SM1 SN32 TN4 unchanged",
        },
        "scope": "Non-transposed BF16 GEMV with (K,N)=(2560,10240),(2560,6144),(6144,2560). This selects shapes, not layer identities. It does not target packed QKV+Z or prefill GEMM.",
        "only_compiled_source": str(source), "timing_automatically_started": False,
        "extra_bm": extra_bm, "maximum_mode": 4 if extra_bm else 2,
        "numerical_validation": "not_run", "performance_validation": "not_run",
    }
    if extra_bm:
        report["modes"].update({"3": "QKV/Z/out: BM2 TM4; BN1 SM1 SN32 TN4 unchanged",
                                "4": "QKV/Z/out: BM1 TM4; BN1 SM1 SN32 TN4 unchanged"})
        report.pop("only_compiled_source")
        report["compiled_sources"] = [str(source), str(runtime / "lib/mlx-src/mlx/backend/metal/kernels/gemv.metal")]
    try:
        for name in ("src", "obj", "lib", "logs"):
            (output / name).mkdir()
        copied = output / "src/matmul.cpp"
        copied.write_text(modified)
        (output / "src/matmul.patch").write_text("".join(difflib.unified_diff(
            original.splitlines(keepends=True), modified.splitlines(keepends=True),
            fromfile=str(source), tofile=str(copied))))
        for name in ("libmlxc.dylib", "libjaccl.dylib", "mlx.metallib"):
            shutil.copy2(stage / "lib" / name, output / "lib" / name)
        (output / "include").symlink_to(stage / "include", target_is_directory=True)
        watched = [source, database, link_file, timing_object, timing_provenance_path,
                   stage / "lib/libmlx.dylib", timing_stage / "lib/libmlx.dylib",
                   stage / "lib/mlx.metallib", timing_stage / "lib/mlx.metallib"] + objects
        metal_commands, metal_outputs = [], []
        if extra_bm:
            metal_commands, metal_inputs, metal_outputs = extra_metal_plan(runtime, old_build, output)
            watched += metal_inputs
        report["original_inputs"] = [{"path": str(p), "sha256": sha256(p)} for p in watched]
        new_object = output / "obj/matmul.cpp.o"
        compile_command = compile_argv(matches[0], source, copied, new_object, output / "lib/mlx.metallib")
        link_command = link_argv(original_link, timing_object, output / "lib/libmlx.dylib")
        if link_command.count(MATMUL_OBJECT) != 1:
            raise ValueError("Exact matmul.cpp link object missing")
        link_command[link_command.index(MATMUL_OBJECT)] = str(new_object)
        report["status"] = "building"
        report["planned_commands"] = [*metal_commands, compile_command, link_command]
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        for index, command in enumerate(metal_commands):
            run_command(command, output / "obj", output / f"logs/metal-{index}.log", report["commands"])
        run_command(compile_command, old_build, output / "logs/compile.log", report["commands"])
        run_command(link_command, old_build, output / "logs/link.log", report["commands"])
        run_command(["nm", "-gU", str(output / "lib/libmlx.dylib")], output,
                    output / "logs/symbols.log", report["commands"])
        run_command(["otool", "-L", str(output / "lib/libmlx.dylib")], output,
                    output / "logs/linkage.log", report["commands"])
        symbols = (output / "logs/symbols.log").read_text()
        for name in ("anemlx_set_gdn_gemv_mode", "anemlx_gdn_gemv_max_mode", "anemlx_timing_start", "anemlx_timing_stop", "anemlx_timing_version"):
            if "_" + name not in symbols:
                raise RuntimeError(f"Required export missing: {name}")
        if any(sha256(x["path"]) != x["sha256"] for x in report["original_inputs"]):
            raise RuntimeError("An original input changed during the build")
        report["original_inputs_unchanged"] = True
        report["status"] = "built_not_executed"
        report["outputs"] = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                             for p in (new_object, copied, output / "lib/libmlx.dylib", *metal_outputs)]
        report["runtime_selection"] = {"DYLD_LIBRARY_PATH": str(output / "lib"),
                                       "ANERUNNER_MLX_ROOT": str(output),
                                       "setter": "extern C void anemlx_set_gdn_gemv_mode(int)",
                                       "maximum_mode_getter": "extern C int anemlx_gdn_gemv_max_mode(void)",
                                       "note": "The colocated lib/mlx.metallib is preferred before an embedded fallback path. Resolve getter/setter in the running process; no library is loaded by this helper."}
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--timing-stage", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extra-bm", action="store_true", help="Compile added BF16 BM2/BM1 TM4 kernels and enable modes 3/4")
    args = parser.parse_args()
    try:
        print(build(args.runtime_root, args.timing_stage, args.output_root, args.extra_bm))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"build failed: {error}\n")


if __name__ == "__main__":
    main()
