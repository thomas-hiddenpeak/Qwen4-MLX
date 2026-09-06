#!/usr/bin/env python3
"""Build the isolated MoE down-pair scheduling probe; never run a GPU workload.

Compile one C++ primitive against pinned headers, then dynamically link the
installed stock MLX. There is no new Metal shader, copied backend, or MLX edit.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from build_mlx_command_timing import DEFAULT_RUNTIME, PACKAGE, compile_argv, run_command, sha256
from build_mlx_moe_tiling import verify_stock_install


EXPORTS = ("anemlx_moe_down_pair_version", "anemlx_moe_down_pair_last_error",
           "anemlx_moe_down_pair_count", "anemlx_moe_down_pair")


def build(runtime, output):
    runtime, output = runtime.resolve(), output.resolve()
    if output == runtime or runtime in output.parents:
        raise ValueError("Output must be fresh and outside the author runtime")
    stage, old_build = runtime / "lib/mlx", runtime / "lib/.mlx-build/mlx"
    source_root, c_root = runtime / "lib/mlx-src", runtime / "lib/mlxc-src"
    stock = stage / "lib/libmlx.dylib"
    database = old_build / "compile_commands.json"
    original_source = source_root / "mlx/backend/metal/quantized.cpp"
    entries = [e for e in json.loads(database.read_text())
               if Path(e["file"]).resolve() == original_source]
    if len(entries) != 1 or Path(entries[0]["directory"]).resolve() != old_build:
        raise ValueError("Expected the pinned quantized.cpp compiler recipe")
    stamp = (stage / ".version").read_text().strip()
    if stamp != "mlx=1f8e74e3f12f mlxc=56b2d39fc831 target=26.2":
        raise ValueError("Down-pair internal API is restricted to the reviewed MLX/MLX C pin")
    originals = [PACKAGE / "native" / name for name in
                 ("moe_down_pair_bridge.cpp", "moe_down_pair_bridge.h")]
    equivalence = verify_stock_install(old_build, stage)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "build-provenance.json"
    report = {
        "schema_version": 1, "status": "preparing", "runtime": str(runtime),
        "output_root": str(output), "pinned_stamp": stamp, "abi_version": 1,
        "developer_dir": os.environ.get("DEVELOPER_DIR"), "exports": EXPORTS,
        "stock_install_equivalence": equivalence,
        "scope": "S1 routed Q4 down-reduce and shared BF16 down only; original child primitive arithmetic",
        "method": "One C++ multi-output primitive, input hazard predeclaration, direct existing Primitive eval_gpu virtual calls",
        "ownership": "Outer inputs/siblings retained by MLX GPU eval; successful dylib handles pinned for process lifetime",
        "error_contract": "Entry exceptions become nonzero plus thread-local error; deferred errors use normal MLX eval",
        "gpu_workload_started": False, "library_loaded": False,
        "numerical_validation": "not_run", "performance_validation": "not_run", "commands": [],
    }
    try:
        for folder in ("src", "obj", "lib", "logs"):
            (output / folder).mkdir()
        for original in originals:
            shutil.copy2(original, output / "src" / original.name)
        shutil.copy2(source_root / "LICENSE", output / "src/MLX-LICENSE")
        shutil.copy2(c_root / "LICENSE", output / "src/MLXC-LICENSE")
        bridge, obj = output / "src/moe_down_pair_bridge.cpp", output / "obj/moe_down_pair_bridge.o"
        library = output / "lib/libanemlx_moe_down_pair.dylib"
        compile_cpp = compile_argv(entries[0], original_source, bridge, obj, output / "unused.metallib")
        compile_cpp = [arg for arg in compile_cpp
                       if arg != "-DMLX_EXPORT" and not arg.startswith("-DMETAL_PATH=")]
        compile_cpp.append("-I" + str(c_root))
        link = [compile_cpp[0], "-dynamiclib", "-arch", "arm64", "-mmacosx-version-min=26.2",
                "-Wl,-undefined,error", "-Wl,-headerpad_max_install_names",
                "-Wl,-install_name,@rpath/libanemlx_moe_down_pair.dylib",
                "-Wl,-rpath," + str(stage / "lib"), str(obj), str(stock),
                "-framework", "Metal", "-framework", "Foundation", "-o", str(library)]
        dependency = ["/usr/bin/install_name_tool", "-change", "@rpath/libmlx.dylib", str(stock), str(library)]
        watched = [Path(__file__), *originals, database, stage / ".version", stock,
                   source_root / "mlx/array.h", source_root / "mlx/primitives.h",
                   source_root / "mlx/backend/metal/device.h", source_root / "mlx/backend/metal/device.cpp",
                   source_root / "mlx/backend/metal/eval.cpp", source_root / "mlx/backend/metal/custom_kernel.cpp",
                   source_root / "mlx/backend/metal/matmul.cpp", c_root / "mlx/c/private/array.h",
                   c_root / "mlx/c/private/vector.h", c_root / "mlx/c/private/stream.h"]
        report["original_inputs"] = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                                     for p in dict.fromkeys(watched)]
        report.update({"status": "building", "original_compile_entry": entries[0],
                       "planned_commands": [compile_cpp, link, dependency]})
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        for index, command in enumerate(report["planned_commands"]):
            run_command(command, output, output / f"logs/step-{index}.log", report["commands"])
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        exports, deps = output / "logs/nm-exports.log", output / "logs/otool-dependencies.log"
        run_command(["/usr/bin/nm", "-gU", str(library)], output, exports, report["commands"])
        for name in EXPORTS:
            if not any(line.endswith(" T _" + name) for line in exports.read_text().splitlines()):
                raise ValueError("Missing C export: " + name)
        run_command(["/usr/bin/otool", "-L", str(library)], output, deps, report["commands"])
        if str(stock) not in deps.read_text():
            raise ValueError("Plugin lacks expected absolute stock MLX dependency")
        run_command(["/usr/bin/codesign", "--verify", "--verbose=2", str(library)],
                    output, output / "logs/codesign-verify.log", report["commands"])
        changed = [entry["path"] for entry in report["original_inputs"]
                   if sha256(entry["path"]) != entry["sha256"]]
        if changed:
            raise ValueError("Original inputs changed during build: " + ", ".join(changed))
        report.update({"status": "built_not_executed", "original_inputs_unchanged": True,
                       "library_path": str(library),
                       "artifacts": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                                     for p in (obj, library)]})
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, default=PACKAGE / "results/moe-down-pair-v1/native")
    parser.add_argument("--developer-dir", type=Path, default=Path("/Applications/Xcode.app/Contents/Developer"))
    args = parser.parse_args()
    if not args.developer_dir.is_dir():
        parser.error("Developer directory does not exist")
    os.environ["DEVELOPER_DIR"] = str(args.developer_dir.resolve())
    result = build(args.runtime, args.output)
    print(json.dumps({key: result[key] for key in ("status", "library_path")}, indent=2))


if __name__ == "__main__":
    main()
