#!/usr/bin/env python3
"""Test build.sh with real CMake/Ninja and tiny CPU-only replacement targets."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "build.sh"
TARGETS = ("llama-server", "llama-fit-params", "llama-quantize", "llama-bench")


class BuildTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="airified-build-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "workspace with spaces"
        self.repo = self.root / "original checkout"
        source = self.repo / "vendor/llama.cpp"
        source.mkdir(parents=True)
        (self.repo / "tools").mkdir()
        shutil.copy2(BUILD_SCRIPT, self.repo / "build.sh")
        # Patch application is orthogonal to build-tree relocation. The engine
        # replacement accepts the same target names without invoking CUDA.
        (self.repo / "tools/apply_engine_patch.py").write_text(
            'print("Probe patches checked")\n'
        )
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.24)\n'
            'project(build_driver_probe C)\n'
            'set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n'
            f'foreach(name {" ".join(TARGETS)})\n'
            '  add_executable(${name} main.c)\n'
            'endforeach()\n'
        )
        (source / "main.c").write_text(
            '#include <stdio.h>\nint main(void) { puts("build probe"); return 0; }\n'
        )

    @property
    def default_build(self):
        return self.repo / "build/sm75-q8-persist"

    def run_build(self, *args, build_dir=None, cwd=None, expected=0):
        env = os.environ.copy()
        env.pop("BUILD_DIR", None)
        env.pop("NCCL_ROOT", None)
        env["BUILD_JOBS"] = "2"
        if build_dir is not None:
            env["BUILD_DIR"] = str(build_dir)
        result = subprocess.run(
            ["bash", str(self.repo / "build.sh"), *args],
            cwd=cwd or self.root, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        self.assertEqual(result.returncode, expected, result.stdout)
        return result.stdout

    def assert_cache_paths(self, build):
        cache = (build / "CMakeCache.txt").read_text()
        self.assertIn(
            f'CMAKE_HOME_DIRECTORY:INTERNAL={(self.repo / "vendor/llama.cpp").resolve()}\n',
            cache,
        )
        self.assertIn(f'CMAKE_CACHEFILE_DIR:INTERNAL={build.resolve()}\n', cache)
        for target in TARGETS:
            result = subprocess.check_output([str(build / "bin" / target)], text=True)
            self.assertEqual(result, "build probe\n")

    def add_cache_marker(self, build):
        with (build / "CMakeCache.txt").open("a") as cache:
            cache.write("\nAIRIFIED_TEST_MARKER:STRING=keep-incrementally\n")

    def move_checkout(self):
        moved = self.root / "renamed checkout"
        self.repo.rename(moved)
        self.repo = moved

    def test_unchanged_build_is_incremental(self):
        self.run_build()
        build = self.default_build
        self.add_cache_marker(build)
        binary = build / "bin/llama-server"
        before = binary.stat().st_mtime_ns
        output = self.run_build()
        self.assertNotIn("Stale CMake paths", output)
        self.assertEqual(binary.stat().st_mtime_ns, before)
        self.assertIn("AIRIFIED_TEST_MARKER:STRING=", (build / "CMakeCache.txt").read_text())
        self.assert_cache_paths(build)

    def test_moved_checkout_recovers_automatically(self):
        self.run_build()
        self.move_checkout()
        output = self.run_build()
        self.assertIn("Stale CMake paths", output)
        self.assert_cache_paths(self.default_build)
        before = (self.default_build / "bin/llama-server").stat().st_mtime_ns
        self.run_build()
        self.assertEqual((self.default_build / "bin/llama-server").stat().st_mtime_ns, before)

    def test_moved_source_with_external_build(self):
        build = self.root / "external build"
        self.run_build(build_dir=build)
        self.move_checkout()
        output = self.run_build(build_dir=build)
        self.assertIn("Stale CMake paths", output)
        self.assert_cache_paths(build)

    def test_moved_build_with_unchanged_source(self):
        old = self.root / "external build"
        moved = self.root / "moved build"
        self.run_build(build_dir=old)
        old.rename(moved)
        output = self.run_build(build_dir=moved)
        self.assertIn("Stale CMake paths", output)
        self.assert_cache_paths(moved)

    def test_relative_build_override_remains_incremental(self):
        self.run_build(build_dir="relative build")
        build = self.root / "relative build"
        self.assert_cache_paths(build)
        before = (build / "bin/llama-server").stat().st_mtime_ns
        output = self.run_build(build_dir="relative build/../relative build")
        self.assertNotIn("Stale CMake paths", output)
        self.assertEqual((build / "bin/llama-server").stat().st_mtime_ns, before)

    def test_rebuild_refreshes_cache_and_recompiles(self):
        self.run_build()
        build = self.default_build
        self.add_cache_marker(build)
        binary = build / "bin/llama-server"
        before = binary.stat().st_mtime_ns
        self.run_build("--rebuild")
        self.assertNotEqual(binary.stat().st_mtime_ns, before)
        self.assertNotIn("AIRIFIED_TEST_MARKER", (build / "CMakeCache.txt").read_text())
        self.assert_cache_paths(build)

    def test_cli_does_not_build_for_help_or_invalid_argument(self):
        self.assertIn("Usage:", self.run_build("--help"))
        self.assertIn("Unknown argument:", self.run_build("--typo", expected=2))
        self.assertFalse(self.default_build.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
