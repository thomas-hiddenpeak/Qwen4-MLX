#!/usr/bin/env python3
"""Build an isolated lazy MoE gate/up primitive, without dlopen or GPU work.

Compile one new C++ bridge using the pinned MLX flags and two Metal sources
using the pinned no-fast-math recipe. Link a small plugin against the installed
stock libmlx. No original source, object, AIR, metallib, or library is changed.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil

from build_mlx_command_timing import DEFAULT_RUNTIME, PACKAGE, compile_argv, run_command, sha256
from build_mlx_moe_tiling import verify_stock_install


DEFAULT_OUTPUT = PACKAGE / "results/moe-prefill-expert-v1/native"
DEFAULT_DEVELOPER = Path("/Applications/Xcode.app/Contents/Developer")
EXPORTS = ("anemlx_moe_gateup_version", "anemlx_moe_gateup_last_error",
           "anemlx_moe_gateup_dispatch_count", "anemlx_moe_gateup",
           "anemlx_moe_expert_plan", "anemlx_moe_gateup_planned", "anemlx_moe_grouped_down",
           "anemlx_moe_expert_plan_dispatch_count", "anemlx_moe_grouped_down_dispatch_count")
FUNCTIONS = (
    "anemlx_moe_gateup_fused_bf16_q4_g64_bm32_bn64_bk64_wm2_wn2",
    "anemlx_moe_gateup_fused_bf16_q4_g64_bm32_bn32_bk64_wm2_wn1",
    "anemlx_moe_expert_plan_bm32", "anemlx_moe_expert_plan_bm16",
    "anemlx_moe_gateup_grouped_bf16_q4_g64_bm32_bn32_bk64_wm2_wn1",
    "anemlx_moe_gateup_grouped_bf16_q4_g64_bm16_bn32_bk64_wm1_wn1",
    "anemlx_moe_down_grouped_bf16_q4_g64_bm32_bn32_bk64_wm2_wn1",
    "anemlx_moe_down_grouped_bf16_q4_g64_bm16_bn32_bk64_wm1_wn1",
)


def metal_command(runtime, old_build, source, air):
    kernels = old_build / "mlx/backend/metal/kernels"
    makefile = kernels / "CMakeFiles/mlx-metallib.dir/build.make"
    original = runtime / "lib/mlx-src/mlx/backend/metal/kernels/quantized_nax.metal"
    matches = []
    for line in makefile.read_text().splitlines():
        if not line.startswith("\tcd "):
            continue
        tokens = shlex.split(line)
        if tokens[3:7] != ["xcrun", "-sdk", "macosx", "metal"]:
            continue
        if "-c" not in tokens or str(original) not in tokens:
            continue
        if tokens[2] != "&&" or Path(tokens[1]).resolve() != kernels:
            raise ValueError("Unreviewed Metal recipe working directory")
        matches.append(tokens[3:])
    if len(matches) != 1:
        raise ValueError("Expected one pinned quantized_nax.metal compile recipe")
    command = list(matches[0])
    if "-fno-fast-math" not in command or command.count("-o") != 1:
        raise ValueError("Expected original no-fast-math recipe with one output")
    command[command.index(str(original))] = str(source)
    command[command.index("-o") + 1] = str(air)
    return command, makefile, matches[0]


def build(runtime, output):
    runtime, output = runtime.resolve(), output.resolve()
    if output == runtime or runtime in output.parents:
        raise ValueError("Output must be fresh and outside the author runtime")
    stage, old_build = runtime / "lib/mlx", runtime / "lib/.mlx-build/mlx"
    source_root = runtime / "lib/mlx-src"
    c_source_root = runtime / "lib/mlxc-src"
    stock_library = stage / "lib/libmlx.dylib"
    database = old_build / "compile_commands.json"
    quantized = source_root / "mlx/backend/metal/quantized.cpp"
    entries = [e for e in json.loads(database.read_text())
               if Path(e["file"]).resolve() == quantized]
    if len(entries) != 1 or Path(entries[0]["directory"]).resolve() != old_build:
        raise ValueError("Expected one pinned Metal quantized.cpp compiler command")
    original_files = [PACKAGE / "native" / name for name in
                      ("moe_gateup_bridge.cpp", "moe_gateup_bridge.h", "moe_gateup_fused.metal",
                       "moe_expert_grouped.metal")]
    if not all(path.is_file() for path in original_files):
        raise ValueError("Bridge, header, or frozen shader is missing")
    equivalence = verify_stock_install(old_build, stage)

    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "build-provenance.json"
    report = {
        "schema_version": 1, "status": "preparing", "runtime": str(runtime),
        "output_root": str(output), "developer_dir": os.environ.get("DEVELOPER_DIR"),
        "pinned_stamp": (stage / ".version").read_text().strip(),
        "stock_install_equivalence": equivalence,
        "base": "Small new plugin dynamically linked to installed stock libmlx; no native object or AIR reuse",
        "abi_version": 2, "exports": EXPORTS, "functions": FUNCTIONS,
        "geometries_BM_BN_BK_WM_WN": [[32, 64, 64, 2, 2], [32, 32, 64, 2, 1],
                                      [32, 32, 64, 2, 1], [16, 32, 64, 1, 1]],
        "scope": "Original gateup variants0/1 unchanged; GPU expert plan; variants2/3 planned gateup and grouped down using original affine Q4/group64 E512 banks",
        "input_contract": "All evaluated inputs row-contiguous; indices sorted and in 0..<512; plan must match indices/M/BM; explicit GPU stream for new APIs; no CPU array extraction",
        "plan_layout": "I32[ceil(M/BM)+512,4] including header; row0.x valid descriptor count; rows1+ {expert,start,count,0}; fixed-capacity GPU dispatch and no CPU count readback",
        "ownership": "Official pinned mlx_array_get_/mlx_array_set_ helpers; caller owns result and pins plugin handle",
        "error_contract": "Entry catches C++ exceptions into thread-local last_error and nonzero code; deferred errors propagate through existing MLX evaluation API",
        "counter_semantics": "Successful GPU encoding per variant, not completion, duration, instructions, or memory bytes",
        "metal_fast_math": False, "gpu_workload_started": False, "library_loaded": False,
        "numerical_validation": "not_run", "performance_validation": "not_run", "commands": [],
    }
    try:
        for subdir in ("src", "obj", "lib", "logs"):
            (output / subdir).mkdir()
        for path in original_files:
            shutil.copy2(path, output / "src" / path.name)
        shutil.copy2(source_root / "LICENSE", output / "src/MLX-LICENSE")
        shutil.copy2(c_source_root / "LICENSE", output / "src/MLXC-LICENSE")
        bridge = output / "src/moe_gateup_bridge.cpp"
        shader = output / "src/moe_gateup_fused.metal"
        grouped_shader = output / "src/moe_expert_grouped.metal"
        obj, air = output / "obj/moe_gateup_bridge.o", output / "obj/moe_gateup_fused.air"
        grouped_air = output / "obj/moe_expert_grouped.air"
        metallib = output / "lib/moe_gateup.metallib"
        library = output / "lib/libanemlx_moe_gateup.dylib"
        metal, makefile, original_metal = metal_command(runtime, old_build, shader, air)
        grouped_metal, _, _ = metal_command(runtime, old_build, grouped_shader, grouped_air)
        compile_cpp = compile_argv(entries[0], quantized, bridge, obj, metallib)
        # MLX_EXPORT belongs to the original library build. Imported MLX APIs
        # retain their visibility; only the explicit C symbols are public.
        compile_cpp = [arg for arg in compile_cpp
                       if arg != "-DMLX_EXPORT" and not arg.startswith("-DMETAL_PATH=")]
        library_id = ("anemlx_moe_gateup_v2_" + sha256(shader)[:16] + "_" +
                      sha256(grouped_shader)[:16] + "_" + sha256(bridge)[:16])
        compile_cpp.extend([
            "-I" + str(c_source_root),
            '-DANEMLX_GATEUP_METALLIB="' + str(metallib) + '"',
            '-DANEMLX_GATEUP_LIBRARY_ID="' + library_id + '"',
        ])
        link = [compile_cpp[0], "-dynamiclib", "-arch", "arm64", "-mmacosx-version-min=26.2",
                "-Wl,-undefined,error", "-Wl,-headerpad_max_install_names",
                "-Wl,-install_name,@rpath/libanemlx_moe_gateup.dylib",
                "-Wl,-rpath," + str(stage / "lib"), str(obj), str(stock_library),
                "-framework", "Metal", "-framework", "Foundation", "-o", str(library)]
        metal_link = ["xcrun", "-sdk", "macosx", "metallib", str(air), str(grouped_air), "-o", str(metallib)]
        # Use an absolute dependency in this plugin, while leaving the stock
        # library itself untouched. No DYLD overlay or alternative libmlx copy.
        absolute_dependency = ["/usr/bin/install_name_tool", "-change", "@rpath/libmlx.dylib",
                               str(stock_library), str(library)]
        watched = [Path(__file__), *original_files, database, makefile, stage / ".version",
                   stock_library, source_root / "mlx/array.h", source_root / "mlx/primitives.h",
                   source_root / "mlx/backend/metal/device.h", source_root / "mlx/backend/metal/device.cpp",
                   source_root / "mlx/backend/metal/kernels/quantized_nax.h",
                   source_root / "mlx/backend/metal/kernels/steel/gemm/nax.h",
                   c_source_root / "mlx/c/private/array.h", c_source_root / "mlx/c/array.h",
                   c_source_root / "mlx/c/private/stream.h", c_source_root / "mlx/c/stream.h"]
        report["original_inputs"] = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                                     for p in dict.fromkeys(watched)]
        report.update({"status": "building", "original_compile_entry": entries[0],
                       "original_metal_compile": original_metal, "library_id": library_id,
                       "planned_commands": [compile_cpp, metal, grouped_metal, metal_link, link, absolute_dependency]})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        for index, command in enumerate(report["planned_commands"]):
            run_command(command, output, output / f"logs/step-{index}.log", report["commands"])
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        nm_log, deps_log = output / "logs/nm-exports.log", output / "logs/otool-dependencies.log"
        run_command(["/usr/bin/nm", "-gU", str(library)], output, nm_log, report["commands"])
        for name in EXPORTS:
            if not any(line.endswith(" T _" + name) for line in nm_log.read_text().splitlines()):
                raise ValueError("Missing C export: " + name)
        run_command(["/usr/bin/otool", "-L", str(library)], output, deps_log, report["commands"])
        if str(stock_library) not in deps_log.read_text():
            raise ValueError("Plugin lacks the expected absolute stock libmlx dependency")
        run_command(["/usr/bin/codesign", "--verify", "--verbose=2", str(library)],
                    output, output / "logs/codesign-verify.log", report["commands"])
        changed = [entry["path"] for entry in report["original_inputs"]
                   if sha256(entry["path"]) != entry["sha256"]]
        if changed:
            raise ValueError("Original inputs changed during build: " + ", ".join(changed))
        report.update({"status": "built_not_executed", "original_inputs_unchanged": True,
                       "library_path": str(library), "metallib_path": str(metallib),
                       "artifacts": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                                     for p in (library, metallib, obj, air, grouped_air)]})
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--developer-dir", type=Path, default=DEFAULT_DEVELOPER)
    args = parser.parse_args()
    if not args.developer_dir.is_dir():
        parser.error("A full Xcode Developer directory is required for Metal compilation")
    os.environ["DEVELOPER_DIR"] = str(args.developer_dir.resolve())
    result = build(args.runtime, args.output)
    print(json.dumps({key: result[key] for key in ("status", "library_path", "metallib_path")}, indent=2))


if __name__ == "__main__":
    main()
