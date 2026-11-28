"""Durcissement : chemins de manifeste confinés, dossier de sortie, limites d'image, sous-processus, serveur web."""
import http.client
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from plan2cad import pipeline, safety, web


def _alloc_big():
    x = bytearray(2 * 1024 ** 3)           # 2 Go : dépasse la limite d'espace d'adressage du sous-processus
    return len(x)


def _slow():
    time.sleep(30)


def _double(x):
    return x * 2


class Names(unittest.TestCase):
    def test_safe_name_has_no_separator_or_dotdot(self):
        for bad in ("../../etc/passwd", "a/b\\c", "..", "x\x00y", ""):
            n = safety.safe_name(bad)
            self.assertNotIn("/", n); self.assertNotIn("\\", n); self.assertNotIn("..", n); self.assertTrue(n)


class Paths(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name, "proj"); self.root.mkdir()
        self.outside = Path(self.td.name, "secret"); self.outside.mkdir()
        Image.new("L", (10, 10)).save(self.root / "a.png")
        Image.new("L", (10, 10)).save(self.outside / "b.png")

    def tearDown(self):
        self.td.cleanup()

    def test_inside_ok_outside_refused(self):
        safety.check_input_file(self.root / "a.png", [self.root])
        with self.assertRaises(safety.UnsafeInput):
            safety.check_input_file(self.outside / "b.png", [self.root])
        with self.assertRaises(safety.UnsafeInput):
            safety.check_input_file(self.root / ".." / "secret" / "b.png", [self.root])

    def test_symlink_out_refused(self):
        os.symlink(self.outside / "b.png", self.root / "lien.png")
        with self.assertRaises(safety.UnsafeInput):
            safety.check_input_file(self.root / "lien.png", [self.root])

    def test_extension_refused(self):
        (self.root / "x.txt").write_text("hello")
        with self.assertRaises(safety.UnsafeInput):
            safety.check_input_file(self.root / "x.txt", [self.root])

    def test_manifest_with_outside_path_is_refused(self):
        proj = dict(name="p", levels=[dict(name="RDC", sources=[dict(file="../secret/b.png", lot="arc")])])
        with self.assertRaises(safety.UnsafeInput):
            pipeline.normalize_project(proj, self.root, allowed_roots=[self.root])
        ok = dict(name="p", levels=[dict(name="RDC", sources=[dict(file="a.png", lot="arc")])])
        self.assertEqual(len(pipeline.normalize_project(ok, self.root, allowed_roots=[self.root])["levels"][0]["sources"]), 1)

    def test_outdir(self):
        for bad in ("/", "/etc", str(Path.home())):
            with self.assertRaises(safety.UnsafeInput):
                safety.check_outdir(bad)
        safety.check_outdir(Path(self.td.name, "resultats"))


class Limits(unittest.TestCase):
    def test_pixel_bomb_refused_before_decoding(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, "big.png")
            Image.new("L", (100, 100)).save(f)
            old = safety.MAX_PIXELS
            safety.MAX_PIXELS = 1000
            try:
                with self.assertRaises(safety.UnsafeInput):
                    safety.check_image_header(f)
            finally:
                safety.MAX_PIXELS = old

    def test_side_is_clamped(self):
        self.assertEqual(safety.clamp_side(10 ** 9), safety.MAX_SIDE_PX)


class Isolation(unittest.TestCase):
    def test_result_comes_back(self):
        self.assertEqual(safety.run_isolated(_double, 21, mem_mb=None, cpu_s=None), 42)

    @unittest.skipUnless(hasattr(__import__("os"), "fork"), "limites mémoire : Unix")
    def test_memory_bomb_is_contained(self):
        with self.assertRaises(safety.UnsafeInput):
            safety.run_isolated(_alloc_big, mem_mb=800, cpu_s=None, timeout_s=60)

    def test_timeout(self):
        t = time.time()
        with self.assertRaises(safety.UnsafeInput):
            safety.run_isolated(_slow, mem_mb=None, cpu_s=None, timeout_s=2)
        self.assertLess(time.time() - t, 15)


class WebTraversal(unittest.TestCase):
    def test_sibling_directory_with_same_prefix_is_forbidden(self):
        with tempfile.TemporaryDirectory() as d:
            root, sib = Path(d, "out"), Path(d, "out2")
            root.mkdir(); sib.mkdir()
            (sib / "secret.txt").write_text("top secret")
            (root / "ok.txt").write_text("fine")
            srv = web.serve(root, port=0, block=False)
            try:
                port = srv.server_address[1]
                def get(path):
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    c.request("GET", path)
                    r = c.getresponse()
                    return r.status, r.read()
                self.assertEqual(get("/r/_/ok.txt")[0], 200)
                st, body = get("/r/_/../out2/secret.txt")
                self.assertEqual(st, 403)
                self.assertNotIn(b"top secret", body)
            finally:
                srv.shutdown(); srv.server_close()


if __name__ == "__main__":
    unittest.main()
