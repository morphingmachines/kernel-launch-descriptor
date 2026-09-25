"""Tests for str/Path buffer init resolution and buf_dir pruning in kernel_launch.py.

Run: python3 -m unittest discover -s tests -v   (also registered as a ctest
in examples/CMakeLists.txt).
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DESCRIPTOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESCRIPTOR_ROOT))

from kernel_launch import (  # noqa: E402
    BUF_INIT_DIRNAME, IN, KernelLaunch, SharedBuffer, buffer, save_launches,
)

FILEINIT_DIR = DESCRIPTOR_ROOT / "examples" / "fileinit"


def _exec_launch_file(path: Path):
    """Execute a launch.py the way gen_launches_json.py does."""
    spec = importlib.util.spec_from_file_location("_launch_desc_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TmpDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self._cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def mkdir(self, name: str) -> Path:
        d = self.tmp / name
        d.mkdir(parents=True)
        return d


class FileinitExampleTest(TmpDirCase):
    def test_json_has_absolute_paths_next_to_launch_py(self):
        cwd = self.mkdir("cwd")
        out = self.mkdir("out") / "launches.json"
        subprocess.run(
            [sys.executable, str(DESCRIPTOR_ROOT / "gen_launches_json.py"),
             str(FILEINIT_DIR / "launch.py"), "-o", str(out)],
            cwd=cwd, check=True, capture_output=True, text=True)

        args = json.loads(out.read_text())["kernels"][0]["args"]
        by_name = {a.get("name") or a.get("shared_id"): a for a in args}
        self.assertEqual(by_name["p"]["init"], str(FILEINIT_DIR.resolve() / "p.bin"))
        lut = next(a for a in args if a["kind"] == "shared_buffer")
        self.assertEqual(lut["init"], str(FILEINIT_DIR.resolve() / "lut.bin"))
        # bytes init: absolute, under launch.py's buf_init/ (unchanged behaviour)
        q = Path(by_name["q"]["init"])
        self.assertTrue(q.is_absolute())
        self.assertEqual(q.parent, FILEINIT_DIR.resolve() / BUF_INIT_DIRNAME)
        for a in args:
            if a.get("init"):
                self.assertTrue(Path(a["init"]).is_file(), a["init"])


class ResolutionTest(TmpDirCase):
    def _write_launch(self, d: Path, body: str) -> Path:
        p = d / "launch.py"
        p.write_text("from kernel_launch import *\n" + body)
        return p

    def test_relative_resolves_against_launch_py_not_cwd(self):
        launch_dir = self.mkdir("launch")
        (launch_dir / "x.bin").write_bytes(bytes(8))
        cwd = self.mkdir("cwd")
        (cwd / "x.bin").write_bytes(bytes(4))  # decoy with wrong size
        os.chdir(cwd)
        mod = _exec_launch_file(self._write_launch(
            launch_dir, 'B = buffer(8, IN, init="x.bin")\nS = SharedBuffer(8, init="x.bin")\n'))
        self.assertEqual(mod.B.init, launch_dir / "x.bin")
        self.assertEqual(mod.S.init, launch_dir / "x.bin")

    def test_missing_relative_file_names_resolved_path(self):
        launch_dir = self.mkdir("launch")
        os.chdir(self.mkdir("cwd"))
        with self.assertRaises(ValueError) as cm:
            _exec_launch_file(self._write_launch(launch_dir, 'buffer(8, IN, init="missing.bin")\n'))
        self.assertIn(str(launch_dir / "missing.bin"), str(cm.exception))

    def test_size_mismatch_names_resolved_path(self):
        launch_dir = self.mkdir("launch")
        (launch_dir / "small.bin").write_bytes(bytes(4))
        os.chdir(self.mkdir("cwd"))
        with self.assertRaises(ValueError) as cm:
            _exec_launch_file(self._write_launch(launch_dir, 'SharedBuffer(8, init="small.bin")\n'))
        self.assertIn(str(launch_dir / "small.bin"), str(cm.exception))

    def test_absolute_and_bytes_unchanged(self):
        f = self.tmp / "abs.bin"
        f.write_bytes(bytes(8))
        self.assertEqual(buffer(8, IN, init=str(f)).init, f)
        self.assertEqual(buffer(8, IN, init=f).init, f)
        self.assertEqual(buffer(8, IN, init=b"\x01\x02").init, b"\x01\x02")
        self.assertIsNone(buffer(8, IN).init)


class PruneTest(TmpDirCase):
    def test_user_file_in_buf_dir_survives_and_stale_is_pruned(self):
        buf_dir = self.mkdir("launch/" + BUF_INIT_DIRNAME)
        user = buf_dir / "mine.bin"
        user.write_bytes(bytes(8))
        (buf_dir / "k9_a9.bin").write_bytes(b"stale")
        (buf_dir / "subdir").mkdir()  # must not crash pruning
        launches = [KernelLaunch(elf="x.elf", entry="e", grid=(1, 1, 1), args=[
            buffer(8, IN, init=user), buffer(8, IN, init=bytes(8))])]
        save_launches(launches, self.tmp / "launches.json", buf_dir=buf_dir)
        self.assertTrue(user.is_file())
        self.assertFalse((buf_dir / "k9_a9.bin").exists())
        self.assertTrue((buf_dir / "k0_a1.bin").is_file())
        self.assertTrue((buf_dir / "subdir").is_dir())

    def test_bytes_blob_would_overwrite_user_file(self):
        buf_dir = self.mkdir("launch/" + BUF_INIT_DIRNAME)
        user = buf_dir / "k0_a1.bin"
        user.write_bytes(bytes(8))
        launches = [KernelLaunch(elf="x.elf", entry="e", grid=(1, 1, 1), args=[
            buffer(8, IN, init=user), buffer(8, IN, init=b"\xff" * 8)])]
        with self.assertRaises(ValueError):
            save_launches(launches, self.tmp / "launches.json", buf_dir=buf_dir)
        self.assertEqual(user.read_bytes(), bytes(8))


if __name__ == "__main__":
    unittest.main()
