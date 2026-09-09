#!/usr/bin/env python3
"""Build the isolated paged reader or physical KV pool; never execute it.

Reuses the installed MLX compile flags and no-fast-math Metal recipe. Copies
only candidate sources, builds new objects/metallib/executable in a fresh
directory, and dynamically links the existing stock libmlx without changing it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "native/paged-attention"
sys.path.insert(0, str(ROOT / "scripts"))
from build_mlx_command_timing import DEFAULT_RUNTIME, compile_argv, run_command, sha256
from build_mlx_moe_gateup import DEFAULT_DEVELOPER, metal_command
from build_mlx_moe_tiling import verify_stock_install


class BuildFailed(RuntimeError):
    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code if 0 < exit_code < 256 else min(255, 128 + abs(exit_code))


def file_record(path):
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def build(source_dir, output, runtime, bridge_dir=None, pool=False):
    source_dir, output, runtime = (p.resolve() for p in (source_dir, output, runtime))
    bridge_dir = bridge_dir.resolve() if bridge_dir is not None else None
    roots = [runtime, source_dir] + ([bridge_dir] if bridge_dir else [])
    if any(output == root or root in output.parents for root in roots):
        raise ValueError("Output must be fresh and outside the runtime and candidate source trees")
    if output.exists():
        raise ValueError("Output already exists; choose a fresh directory")
    # This is a small standalone candidate, not a general source-tree builder.
    sources = sorted(p for p in source_dir.iterdir() if p.is_file())
    if not sources or len(sources) > 24 or any(p.is_symlink() for p in sources):
        raise ValueError("Expected at most 24 regular candidate files without symlinks")
    cpp_names = {"paged_sdpa_reader.cpp", "paged_sdpa_probe.cpp"}
    shader_names = {"paged_sdpa_vector.metal"}
    if pool:
        cpp_names.update(("immutable_page_pool.cpp", "paged_kv_pool.cpp"))
        shader_names.add("paged_kv_pool.metal")
    cpp = [p for p in sources if p.name in cpp_names]
    shaders = [p for p in sources if p.name in shader_names]
    if {p.name for p in cpp} != cpp_names or {p.name for p in shaders} != shader_names:
        raise ValueError("Missing required reader/pool source files")
    if not 1 <= len(cpp) <= 4 or not 1 <= len(shaders) <= 4:
        raise ValueError("Expected 1..4 standalone C++ and 1..4 Metal source files")
    bridge_inputs = []
    bridge_name = "paged_kv_pool_bridge" if pool else "paged_sdpa_bridge"
    if bridge_dir:
        bridge_inputs = [bridge_dir / (bridge_name + suffix) for suffix in (".cpp", ".h")]
        if any(not p.is_file() or p.is_symlink() for p in bridge_inputs):
            raise ValueError("Bridge directory must contain regular " + bridge_name + ".cpp/.h files")
        if "paged_sdpa_reader.cpp" not in [p.name for p in cpp]:
            raise ValueError("Shared bridge requires the separate paged_sdpa_reader.cpp source")
        if any(p.name == s.name and p != s for p in bridge_inputs for s in sources):
            raise ValueError("Bridge filenames overlap native candidate inputs")
    sources += [p for p in bridge_inputs if p not in sources]
    stage, old_build = runtime / "lib/mlx", runtime / "lib/.mlx-build/mlx"
    mlx_source = runtime / "lib/mlx-src"
    stock = stage / "lib/libmlx.dylib"
    database = old_build / "compile_commands.json"
    original = mlx_source / "mlx/backend/metal/scaled_dot_product_attention.cpp"
    entries = [entry for entry in json.loads(database.read_text())
               if Path(entry["file"]).resolve() == original]
    if len(entries) != 1 or Path(entries[0]["directory"]).resolve() != old_build:
        raise ValueError("Expected one pinned Metal SDPA C++ compiler command")
    equivalence = verify_stock_install(old_build, stage)

    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "build-provenance.json"
    report = {
        "schema": "qwen-paged-reader-native-build-v1", "status": "preparing",
        "source_dir": str(source_dir), "output_root": str(output), "runtime": str(runtime),
        "bridge_dir": str(bridge_dir) if bridge_dir else None, "physical_pool": pool,
        "developer_dir": os.environ.get("DEVELOPER_DIR"),
        "pinned_stamp": (stage / ".version").read_text().strip(),
        "stock_install_equivalence": equivalence,
        "scope": "Standalone reader probe and optional reader/pool shared bridge; no installed library/object/AIR changes or reuse",
        "metal_fast_math": False, "gpu_workload_started": False,
        "probe_executed": False, "numerical_validation": "not_run", "performance_validation": "not_run",
        "commands": [], "original_compile_entry": entries[0],
    }
    try:
        for directory in ("src", "obj", "bin", "lib", "logs"):
            (output / directory).mkdir()
        for path in sources:
            shutil.copy2(path, output / "src" / path.name)
        copied_inputs = [file_record(output / "src" / path.name) for path in sources]
        # copied_inputs is constructed once from exactly sources above; ordinary
        # zip keeps this helper compatible with the host's Python 3.9.
        if len(sources) != len(copied_inputs) or any(
                sha256(path) != copied["sha256"] for path, copied in zip(sources, copied_inputs)):
            raise ValueError("Candidate source changed while copying")
        shutil.copy2(mlx_source / "LICENSE", output / "src/STOCK-MLX-LICENSE")

        metallib = output / "lib/paged_reader.metallib"
        probe = output / "bin/paged-reader-probe"
        shared_library = output / ("lib/paged_kv_pool.dylib" if pool else "lib/paged_reader.dylib") if bridge_dir else None
        library_id = "anemlx_paged_reader_" + hashlib.sha256(
            "".join(item["sha256"] for item in copied_inputs).encode()).hexdigest()[:32]
        cpp_commands, objects, metal_commands, airs = [], [], [], []
        makefile, original_metal = None, None
        for path in cpp:
            obj = output / "obj" / (path.name + ".o")
            command = compile_argv(entries[0], original, output / "src" / path.name, obj, metallib)
            command = [arg for arg in command if arg != "-DMLX_EXPORT" and not arg.startswith("-DMETAL_PATH=")]
            if pool:
                command = ["-std=c++20" if arg.startswith("-std=") else arg for arg in command]
            command.extend(["-I" + str(output / "src"),
                            "-DANEMLX_PAGED_METALLIB=" + json.dumps(str(metallib)),
                            "-DANEMLX_PAGED_LIBRARY_ID=" + json.dumps(library_id)])
            cpp_commands.append(command)
            objects.append(obj)
        bridge_object = None
        if bridge_dir:
            bridge_object = output / ("obj/" + bridge_name + ".cpp.o")
            command = compile_argv(entries[0], original, output / ("src/" + bridge_name + ".cpp"), bridge_object, metallib)
            command = [arg for arg in command if arg != "-DMLX_EXPORT" and not arg.startswith("-DMETAL_PATH=")]
            command.extend(["-I" + str(output / "src"), "-I" + str(runtime / "lib/mlxc-src"),
                            "-I" + str(stage / "include")])
            cpp_commands.append(command)
        for path in shaders:
            air = output / "obj" / (path.name + ".air")
            command, makefile, original_metal = metal_command(runtime, old_build, output / "src" / path.name, air)
            if "-ffast-math" in command or any(arg in command for arg in ("-MD", "-MMD", "-MF", "-MJ")):
                raise ValueError("Unreviewed Metal dependency/fast-math flags")
            metal_commands.append(command)
            airs.append(air)

        link = [cpp_commands[0][0], "-arch", "arm64", "-mmacosx-version-min=26.2",
                "-Wl,-undefined,error", "-Wl,-headerpad_max_install_names",
                "-Wl,-rpath," + str(stage / "lib"), *map(str, objects), str(stock),
                "-framework", "Metal", "-framework", "Foundation", "-o", str(probe)]
        metal_link = ["xcrun", "-sdk", "macosx", "metallib", *map(str, airs), "-o", str(metallib)]
        commands = [*cpp_commands, *metal_commands, metal_link, link]
        linked = [probe]
        if shared_library:
            # Reader/pool core + thin C bridge: never link the probe's main or
            # test fixture objects into the library loaded by Swift.
            commands.append([cpp_commands[0][0], "-dynamiclib", "-arch", "arm64", "-mmacosx-version-min=26.2",
                             "-Wl,-undefined,error", "-Wl,-headerpad_max_install_names",
                             "-Wl,-install_name,@rpath/" + shared_library.name, "-Wl,-rpath," + str(stage / "lib"),
                             *[str(obj) for obj in objects if obj.name != "paged_sdpa_probe.cpp.o"], str(bridge_object), str(stock),
                             "-framework", "Metal", "-framework", "Foundation", "-o", str(shared_library)])
            linked.append(shared_library)
        for binary in linked:
            commands.extend([
                ["/usr/bin/install_name_tool", "-change", "@rpath/libmlx.dylib", str(stock), str(binary)],
                ["/usr/bin/codesign", "--force", "--sign", "-", str(binary)],
                ["/usr/bin/codesign", "--verify", "--verbose=2", str(binary)]])

        watched = [Path(__file__), *sources, database, makefile, stage / ".version", stock,
                   stage / "lib/mlx.metallib", stage / "lib/libjaccl.dylib",
                   mlx_source / "LICENSE", original,
                   mlx_source / "mlx/array.h", mlx_source / "mlx/primitives.h",
                   mlx_source / "mlx/backend/metal/device.h", mlx_source / "mlx/backend/metal/device.cpp",
                   mlx_source / "mlx/backend/metal/kernels/sdpa_vector.h",
                   mlx_source / "mlx/backend/metal/kernels/utils.h",
                   ROOT / "scripts/build_mlx_command_timing.py",
                   ROOT / "scripts/build_mlx_moe_gateup.py",
                   ROOT / "scripts/build_mlx_moe_tiling.py"]
        if bridge_dir:
            c_source = runtime / "lib/mlxc-src"
            watched.extend(c_source / p for p in ("mlx/c/private/array.h", "mlx/c/private/stream.h",
                                                  "mlx/c/array.h", "mlx/c/stream.h", "LICENSE"))
            shutil.copy2(c_source / "LICENSE", output / "src/STOCK-MLXC-LICENSE")
        report.update({"status": "building", "original_metal_compile": original_metal,
                       "library_id": library_id, "copied_candidate_inputs": copied_inputs,
                       "runtime_invocation": [str(probe), "--run"],
                       "runtime_contract": "Explicit --run only; compiled metallib path; default 2pass reduce uses stock MLX; no DYLD overlay",
                       "original_inputs": [file_record(p) for p in dict.fromkeys(watched)],
                       "planned_commands": commands, "probe_path": str(probe), "metallib_path": str(metallib)})
        if shared_library:
            report["library_path"] = str(shared_library)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        for index, command in enumerate(commands):
            run_command(command, output, output / f"logs/step-{index:02}.log", report["commands"])
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        for binary in linked:
            dependencies = output / ("logs/otool-" + binary.name + ".log")
            run_command(["/usr/bin/otool", "-L", str(binary)], output, dependencies, report["commands"])
            if str(stock) not in dependencies.read_text():
                raise ValueError(binary.name + " lacks the expected absolute stock libmlx dependency")
        if shared_library:
            exports = output / "logs/bridge-exports.log"
            run_command(["/usr/bin/nm", "-gU", str(shared_library)], output, exports, report["commands"])
            symbols = exports.read_text().splitlines()
            prefix = "anemlx_paged_kv_pool_" if pool else "anemlx_paged_sdpa_"
            exports = ("version", "last_error", "create", "free", "import", "fork", "append", "state_free", "state_info", "page_ids", "ready", "read", "materialize", "statistics") if pool else ("version", "last_error", "create", "free", "metadata_bytes", "encoded_reads", "read", "dispatch_info")
            for suffix in exports:
                if not any(line.endswith(" T _" + prefix + suffix) for line in symbols):
                    raise ValueError("Missing shared bridge C export: " + suffix)
        changed = [item["path"] for item in report["original_inputs"] if sha256(item["path"]) != item["sha256"]]
        if changed:
            raise ValueError("Original inputs changed during build: " + ", ".join(changed))
        report.update({"status": "built_not_executed", "original_inputs_unchanged": True,
                       "artifacts": [file_record(p) for p in (*linked, metallib, *objects, *airs,
                                                            *([bridge_object] if bridge_object else []))]})
    except Exception as error:
        report.update({"status": "failed", "error": str(error)})
        if report["commands"] and report["commands"][-1]["exit_code"]:
            raise BuildFailed(str(error), report["commands"][-1]["exit_code"]) from error
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=PREP)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--bridge-dir", type=Path, help="Optional C bridge directory; builds paged_reader.dylib or, with --pool, paged_kv_pool.dylib")
    parser.add_argument("--pool", action="store_true", help="Build the physical shared-page pool and its optional bridge")
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--developer-dir", type=Path, default=DEFAULT_DEVELOPER)
    args = parser.parse_args()
    if not args.developer_dir.is_dir():
        parser.error("A full Xcode Developer directory is required for Metal compilation")
    os.environ["DEVELOPER_DIR"] = str(args.developer_dir.resolve())
    try:
        report = build(args.source_dir, args.output, args.runtime, args.bridge_dir, args.pool)
    except BuildFailed as error:
        print(json.dumps({"status": "failed", "error": str(error), "exit_code": error.exit_code}), file=sys.stderr)
        return error.exit_code
    except (ValueError, RuntimeError, OSError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        # run_command retains the exact child code in build-provenance.json.
        return 1
    print(json.dumps({key: report[key] for key in ("status", "probe_path", "metallib_path", "library_path") if key in report}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
