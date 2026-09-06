#!/usr/bin/env python3
"""Build one diagnostic MLX object and relink into a fresh isolated directory.

Never runs CMake, installs over the author stage, builds Swift, or runs a GPU
workload. Reuses the exact existing link objects read-only. Invoke explicitly;
merely importing this module or running its unit tests does not build anything.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


PACKAGE = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = PACKAGE.parent / "qwen38-ssd/runtime/mlx-serve"
NATIVE_HEADER = PACKAGE / "native/mlx-command-timing/command_timing.hpp"
METAL_OBJECT = "CMakeFiles/mlx.dir/mlx/backend/metal/device.cpp.o"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Pinned device.cpp anchor must occur once: {old!r}")
    return text.replace(old, new, 1)


def instrument_source(original):
    text = replace_once(original, '#include "mlx/utils.h"\n',
                        '#include "mlx/utils.h"\n#include "command_timing.hpp"\n')
    text = replace_once(text, 'void CommandEncoder::commit(std::function<void()> completion) {\n',
                        'void CommandEncoder::commit(std::function<void()> completion) {\n'
                        '  const auto anemlx_ticket = anemlx_timing::reserve(buffer_ops_, buffer_sizes_);\n')
    text = replace_once(text, '      [&error_ = error_,\n',
                        '      [anemlx_ticket, &error_ = error_,\n')
    text = replace_once(text, 'completion = std::move(completion)](MTL::CommandBuffer* cbuf) mutable {\n',
                        'completion = std::move(completion)](MTL::CommandBuffer* cbuf) mutable {\n'
                        '        anemlx_timing::completed(anemlx_ticket, cbuf);\n')
    text = replace_once(text, '  buffer_->commit();\n',
                        '  anemlx_timing::before_commit(anemlx_ticket);\n  buffer_->commit();\n')
    return text


def compile_argv(entry, original_source, copied_source, output_object, metallib):
    argv = list(entry["arguments"]) if "arguments" in entry else shlex.split(entry["command"])
    result = []
    source_count = output_count = metal_count = 0
    i = 0
    dependency = output_object.with_suffix(".d")
    # Redirect every dependency/target-output flag even if the pinned command
    # currently has none. Never let a copied compile touch the original .o/.d.
    replacements = {"-o": str(output_object), "-MF": str(dependency),
                    "-MT": str(output_object), "-MQ": str(output_object),
                    "-MJ": str(output_object.with_suffix(".compile.json")),
                    "--serialize-diagnostics": str(output_object.with_suffix(".dia"))}
    while i < len(argv):
        arg = argv[i]
        if arg in replacements:
            if i + 1 == len(argv):
                raise ValueError(f"Missing value after compiler flag {arg}")
            result.extend([arg, replacements[arg]])
            output_count += arg == "-o"
            i += 2
            continue
        joined = next((flag for flag in ("-MF", "-MT", "-MQ", "-MJ") if arg.startswith(flag) and arg != flag), None)
        if joined:
            result.append(joined + replacements[joined])
        elif arg.startswith("-Wp,") or arg == "-Xclang":
            raise ValueError("Unreviewed forwarded compiler flags; refusing possible dependency writes")
        elif arg.startswith("-DMETAL_PATH="):
            # This is an argv value, not a shell string: literal quotes make a
            # C string macro and avoid CMake JSON/shell double-escaping.
            result.append('-DMETAL_PATH="' + str(metallib) + '"')
            metal_count += 1
        elif Path(arg) == original_source:
            result.append(str(copied_source))
            source_count += 1
        else:
            result.append(arg)
        i += 1
    if source_count != 1 or output_count != 1 or metal_count != 1:
        raise ValueError("Compile command must have one exact source, -o and METAL_PATH")
    return result


def link_argv(original, output_object, output_library):
    argv = shlex.split(original)
    if argv.count(METAL_OBJECT) != 1 or argv.count("-o") != 1:
        raise ValueError("Link command must contain one exact device object and one -o")
    result = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-o":
            result.extend([arg, str(output_library)])
            i += 2
            continue
        if arg == METAL_OBJECT:
            result.append(str(output_object))
        elif arg.startswith("-Wl,-rpath,"):
            result.append("-Wl,-rpath,@loader_path")
        elif arg.startswith("@") and arg != "@rpath/libmlx.dylib":
            raise ValueError("Unreviewed linker response file")
        else:
            result.append(arg)
        i += 1
    return result


def run_command(argv, cwd, log, commands):
    started = time.time()
    with log.open("wb") as stream:
        process = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    record = {"argv": argv, "cwd": str(cwd), "log": str(log), "exit_code": process.returncode,
              "elapsed_seconds": time.time() - started}
    commands.append(record)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}")


def build(runtime, output):
    runtime = runtime.resolve()
    output = output.resolve()
    if output == runtime or runtime in output.parents:
        raise ValueError("Diagnostic output must be outside the author runtime tree")
    source = runtime / "lib/mlx-src/mlx/backend/metal/device.cpp"
    old_build = runtime / "lib/.mlx-build/mlx"
    stage = runtime / "lib/mlx"
    database = old_build / "compile_commands.json"
    link_file = old_build / "CMakeFiles/mlx.dir/link.txt"
    entries = json.loads(database.read_text())
    matches = [e for e in entries if Path(e["file"]).resolve() == source]
    if len(matches) != 1 or Path(matches[0]["directory"]).resolve() != old_build:
        raise ValueError("Exact pinned device compile entry is unavailable")
    original_text = source.read_text()
    modified_text = instrument_source(original_text)
    original_link = link_file.read_text()
    objects = [old_build / p for p in shlex.split(original_link) if p.endswith(".o")]
    if not objects or not all(p.is_file() for p in objects):
        raise ValueError("Original link objects are incomplete")
    for name in ("libmlx.dylib", "libmlxc.dylib", "libjaccl.dylib", "mlx.metallib"):
        if not (stage / "lib" / name).is_file():
            raise ValueError(f"Missing original staged file {name}")
    if not (stage / "include").is_dir():
        raise ValueError("Original staged C headers unavailable")
    output.mkdir(parents=True, exist_ok=False)
    provenance_path = output / "build-provenance.json"
    provenance = {"schema_version": 1, "status": "preparing", "runtime": str(runtime), "output_root": str(output),
                  "author_stage": str(stage), "original_build": str(old_build),
                  "pinned_stamp": (stage / ".version").read_text().strip(),
                  "commands": [], "original_files": [], "original_objects": [],
                  "scope": "Only a copied Metal device.cpp is recompiled; original objects are reused read-only. No CMake install, Swift build or GPU workload."}
    try:
        for name in ("src", "obj", "lib", "logs"):
            (output / name).mkdir()
        copied_source = output / "src/device.cpp"
        copied_source.write_text(modified_text)
        shutil.copy2(NATIVE_HEADER, output / "src/command_timing.hpp")
        (output / "src/device.patch").write_text("".join(difflib.unified_diff(
            original_text.splitlines(keepends=True), modified_text.splitlines(keepends=True),
            fromfile=str(source), tofile=str(copied_source))))
        for path in (source, NATIVE_HEADER, database, link_file, stage / "lib/libmlx.dylib",
                     stage / "lib/libmlxc.dylib", stage / "lib/libjaccl.dylib", stage / "lib/mlx.metallib"):
            provenance["original_files"].append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        provenance["original_objects"] = [{"path": str(p), "sha256": sha256(p)} for p in objects]
        for name in ("libmlxc.dylib", "libjaccl.dylib", "mlx.metallib"):
            shutil.copy2(stage / "lib" / name, output / "lib" / name)
        (output / "include").symlink_to(stage / "include", target_is_directory=True)
        object_path = output / "obj/device.cpp.o"
        compile_command = compile_argv(matches[0], source, copied_source, object_path, output / "lib/mlx.metallib")
        link_command = link_argv(original_link, object_path, output / "lib/libmlx.dylib")
        provenance["original_compile_entry"] = matches[0]
        provenance["original_link_command"] = original_link.strip()
        provenance["planned_commands"] = [compile_command, link_command]
        provenance["status"] = "building"
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        run_command(compile_command, old_build, output / "logs/compile.log", provenance["commands"])
        run_command(link_command, old_build, output / "logs/link.log", provenance["commands"])
        run_command(["otool", "-L", str(output / "lib/libmlx.dylib")], output,
                    output / "logs/otool-mlx.log", provenance["commands"])
        run_command(["otool", "-L", str(output / "lib/libmlxc.dylib")], output,
                    output / "logs/otool-mlxc.log", provenance["commands"])
        run_command(["nm", "-gU", str(output / "lib/libmlx.dylib")], output,
                    output / "logs/exported-symbols.log", provenance["commands"])
        symbols = (output / "logs/exported-symbols.log").read_text()
        for name in ("anemlx_timing_start", "anemlx_timing_stop", "anemlx_timing_version"):
            if "_" + name not in symbols:
                raise RuntimeError(f"Diagnostic symbol was not exported: {name}")
        originals = provenance["original_files"] + provenance["original_objects"]
        changed = [item["path"] for item in originals if sha256(item["path"]) != item["sha256"]]
        if changed:
            raise RuntimeError(f"Original inputs changed during build: {changed}")
        provenance["original_inputs_unchanged"] = True
        provenance["outputs"] = [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size}
                                 for p in [object_path, copied_source, output / "src/command_timing.hpp", *sorted((output / "lib").iterdir())]]
        provenance["status"] = "built_not_executed"
        provenance["runtime_selection"] = {"DYLD_LIBRARY_PATH": str(output / "lib"),
                                           "ANERUNNER_MLX_ROOT": str(output),
                                           "verification": "Runner must resolve anemlx_timing_version and record its result; no model or dylib was loaded by this helper."}
    except Exception as exc:
        provenance["status"] = "failed"
        provenance["error"] = str(exc)
        raise
    finally:
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output-root", type=Path, required=True, help="Fresh directory; an existing path is rejected")
    args = parser.parse_args()
    try:
        print(build(args.runtime_root, args.output_root))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
