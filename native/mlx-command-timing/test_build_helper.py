"""CPU-only argument/patch validation. No compiler or GPU process is invoked."""
import importlib.util
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("timing_builder", PACKAGE / "scripts/build_mlx_command_timing.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class BuildHelperTests(unittest.TestCase):
    def test_current_source_has_exactly_the_reviewed_insertions(self):
        source = builder.DEFAULT_RUNTIME / "lib/mlx-src/mlx/backend/metal/device.cpp"
        original = source.read_text()
        modified = builder.instrument_source(original)
        self.assertEqual(modified.count('anemlx_timing::reserve('), 1)
        self.assertEqual(modified.count('anemlx_timing::completed('), 1)
        self.assertEqual(modified.count('anemlx_timing::before_commit('), 1)
        self.assertLess(modified.index('anemlx_timing::before_commit('), modified.index('  buffer_->commit();'))
        self.assertLess(modified.index('anemlx_timing::completed('), modified.index('        if (completion)'))
        with self.assertRaises(ValueError):
            builder.instrument_source(modified)

    def test_dependency_and_object_paths_are_redirected(self):
        source = Path('/original/device.cpp')
        output = Path('/new dir/device.o')
        args = ['c++', '-DMETAL_PATH=old', '-o', '/original/device.o', '-MF', '/original/device.d',
                '-MToriginal-target', '-MQ', 'original-target', '-MJ/original/device.json', '-c', str(source)]
        new = builder.compile_argv({'arguments': args}, source, Path('/new dir/device.cpp'), output, Path('/new dir/mlx.metallib'))
        self.assertEqual(new[new.index('-o') + 1], str(output))
        self.assertEqual(new[new.index('-MF') + 1], '/new dir/device.d')
        self.assertIn('-MT/new dir/device.o', new)
        self.assertEqual(new[new.index('-MQ') + 1], str(output))
        self.assertIn('-MJ/new dir/device.compile.json', new)
        self.assertIn('-DMETAL_PATH="/new dir/mlx.metallib"', new)
        self.assertNotIn(str(source), new)

    def test_forwarded_dependency_flags_are_rejected(self):
        args = ['c++', '-DMETAL_PATH=old', '-Wp,-MD,/original/device.d', '-o', '/original/device.o', '-c', '/original/device.cpp']
        with self.assertRaises(ValueError):
            builder.compile_argv({'arguments': args}, Path('/original/device.cpp'), Path('/new/device.cpp'), Path('/new/device.o'), Path('/new/mlx.metallib'))

    def test_link_uses_new_object_output_and_local_dependency_rpath(self):
        original = 'c++ -dynamiclib -o libmlx.dylib -install_name @rpath/libmlx.dylib ' + builder.METAL_OBJECT + ' other.o -Wl,-rpath,/original/jaccl'
        result = builder.link_argv(original, Path('/new/device.o'), Path('/new/libmlx.dylib'))
        self.assertEqual(result[result.index('-o') + 1], '/new/libmlx.dylib')
        self.assertIn('/new/device.o', result)
        self.assertIn('other.o', result)
        self.assertIn('-Wl,-rpath,@loader_path', result)
        self.assertNotIn(builder.METAL_OBJECT, result)

    def test_pinned_compile_and_link_commands_transform_without_running(self):
        import json
        build = builder.DEFAULT_RUNTIME / 'lib/.mlx-build/mlx'
        source = builder.DEFAULT_RUNTIME / 'lib/mlx-src/mlx/backend/metal/device.cpp'
        entry = next(e for e in json.loads((build / 'compile_commands.json').read_text()) if Path(e['file']) == source)
        compiled = builder.compile_argv(entry, source, Path('/diagnostic/device.cpp'), Path('/diagnostic/device.o'), Path('/diagnostic/mlx.metallib'))
        linked = builder.link_argv((build / 'CMakeFiles/mlx.dir/link.txt').read_text(), Path('/diagnostic/device.o'), Path('/diagnostic/libmlx.dylib'))
        self.assertIn('-mmacosx-version-min=26.2', compiled)
        self.assertIn('-arch', linked)
        self.assertNotIn(str(source), compiled)


if __name__ == '__main__':
    unittest.main()
