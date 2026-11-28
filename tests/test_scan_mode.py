"""Mode scan : murs en double trait, portes à battant, échelle d'après les portes, surfaces lues par l'OCR."""
import unittest

import cv2
import numpy as np

from plan2cad import dimensions as dm
from plan2cad import doors, vectorize as vz

MM = 0.1            # mm papier par pixel


def _room(img, x0, y0, x1, y1, gap_px, t=2):
    """Pièce en double trait : deux rectangles concentriques espacés de gap_px."""
    cv2.rectangle(img, (x0, y0), (x1, y1), 255, t)
    cv2.rectangle(img, (x0 + gap_px, y0 + gap_px), (x1 - gap_px, y1 - gap_px), 255, t)


class WallGap(unittest.TestCase):
    def test_double_line_walls_measured(self):
        img = np.zeros((700, 900), np.uint8)
        _room(img, 50, 50, 850, 650, 9)                     # 0,9 mm entre les deux traits
        gaps, diag = vz.detect_wall_gaps(img, MM)
        self.assertTrue(gaps, diag)
        self.assertTrue(any(0.8 <= g <= 1.3 for g in gaps), gaps)

    def test_single_line_plan_is_not_double(self):
        img = np.zeros((700, 900), np.uint8)
        cv2.rectangle(img, (50, 50), (850, 650), 255, 2)
        cv2.line(img, (450, 50), (450, 650), 255, 2)
        gaps, _ = vz.detect_wall_gaps(img, MM)
        self.assertEqual(gaps, [])

    def test_furniture_does_not_hide_the_peak(self):
        img = np.zeros((700, 900), np.uint8)
        _room(img, 50, 50, 850, 650, 9)
        rng = np.random.default_rng(1)
        for _ in range(150):                                # mobilier : petits traits parallèles à 3 px
            x, y = int(rng.integers(100, 780)), int(rng.integers(100, 580))
            cv2.line(img, (x, y), (x + 40, y), 255, 1)
            cv2.line(img, (x, y + 3), (x + 40, y + 3), 255, 1)
        gaps, _ = vz.detect_wall_gaps(img, MM)
        self.assertTrue(any(0.8 <= g <= 1.3 for g in gaps), gaps)


def _door_image(r=40, mm=MM):
    """Mur horizontal en double trait interrompu par une porte : vantail vertical + quart de cercle."""
    img = np.zeros((500, 700), np.uint8)
    y = 200
    for yy in (y, y + 8):
        cv2.line(img, (20, yy), (300, yy), 255, 2)
        cv2.line(img, (300 + r, yy), (680, yy), 255, 2)
    hx = 300                                                  # charnière à gauche, battant ouvert vers le bas
    cv2.line(img, (hx, y + 8), (hx, y + 8 + r), 255, 2)
    cv2.ellipse(img, (hx, y + 8), (r, r), 0, 0, 90, 255, 2)
    return img


class SwingDoors(unittest.TestCase):
    def test_door_found_with_its_radius(self):
        img = _door_image(40)
        found = doors.detect_swing_doors(img, MM, rmin_mm=2.0, rmax_mm=8.0)
        self.assertEqual(len(found), 1, found)
        self.assertAlmostEqual(found[0]["r_mm"], 40 * MM, delta=0.5)

    def test_chair_arc_without_wall_is_ignored(self):
        img = np.zeros((500, 700), np.uint8)
        cv2.line(img, (300, 250), (300, 290), 255, 2)
        cv2.ellipse(img, (300, 250), (40, 40), 0, 0, 90, 255, 2)
        self.assertEqual(doors.detect_swing_doors(img, MM, rmin_mm=2.0, rmax_mm=8.0), [])

    def test_scale_from_single_doors(self):
        ds = [dict(r_mm=3.7 + 0.05 * i, cx=400 * i, cy=0, r_px=40, double=False) for i in range(5)]
        est = doors.scale_from_doors(ds)
        self.assertIsNotNone(est)
        self.assertAlmostEqual(est["scale"], 850 / 3.8, delta=0.1 * 850 / 3.8)
        self.assertTrue(est["approx"])

    def test_no_scale_from_two_doors(self):
        ds = [dict(r_mm=3.7, cx=0, cy=0, r_px=40, double=False), dict(r_mm=3.7, cx=300, cy=0, r_px=40, double=False)]
        self.assertIsNone(doors.scale_from_doors(ds))

    def test_double_doors_do_not_set_the_scale(self):
        ds = [dict(r_mm=2.9, cx=100 * i, cy=0, r_px=30, double=False) for i in (0, 1)]
        ds[0]["cx"], ds[1]["cx"] = 0, 60                      # deux battants face à face (2 r)
        ds += [dict(r_mm=3.9, cx=1000 + 500 * i, cy=0, r_px=45, double=False) for i in range(3)]
        est = doors.scale_from_doors(ds)
        self.assertAlmostEqual(est["largeur_mm_papier"], 3.9, delta=0.05)


class PrintedArea(unittest.TestCase):
    def test_ocr_variants_of_square_metres(self):
        for t in ("13 m²", "13m2", "13m?", "52 m^"):
            self.assertEqual(dm.parse_room_label(t)["kind"], "area", t)


if __name__ == "__main__":
    unittest.main()


class ScaleCrossCheck(unittest.TestCase):
    def _rooms(self, factor):
        return [dict(surface_m2=a / factor ** 2, surface_lue_m2=float(a), portes=0, fenetres=0, ecart_pct=0.0)
                for a in (6, 9, 12, 15, 20, 30)]

    def test_constant_area_ratio_flags_scale(self):
        from plan2cad import metres as mt
        s = mt.summarize(self._rooms(0.93))
        self.assertTrue(s["echelle_douteuse"])
        self.assertAlmostEqual(s["facteur_echelle"], 0.93, places=2)

    def test_correct_scale_not_flagged(self):
        from plan2cad import metres as mt
        self.assertFalse(mt.summarize(self._rooms(1.0))["echelle_douteuse"])


class Recalage(unittest.TestCase):
    def test_rescale_dxf_scales_lengths_not_angles(self):
        import tempfile
        from pathlib import Path
        from plan2cad import recalage as rc
        from plan2cad.dxf import write_dxf, read_dxf_texts
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "a.dxf"), Path(d, "b.dxf")
            write_dxf(str(src), [("polyline", "ARC-MURS-INT", np.array([[0, 0], [1000, 0], [1000, 2000.0]]), False),
                                 ("arc", "ARC-PORTE", (500.0, 500.0), 400.0, 0.0, 90.0)], {})
            rc.rescale_dxf(src, dst, 0.5)
            txt = dst.read_text()
            self.assertIn("500", txt); self.assertIn("200", txt)           # rayon 400 -> 200, sommet 1000 -> 500
            self.assertIn("90", txt)                                       # l'angle ne change pas


class CotesEtCalques(unittest.TestCase):
    def test_height_label_is_not_a_dimension_of_a_long_line(self):
        # « 7,50 » posé près d'un trait de 3,7 m (à l'échelle) : pas une cote de ce trait ; « 1,00 » près d'un trait de 1 m : oui
        mm_px, scale = 0.3528, 229.5
        px = lambda m: m * 1000 / (mm_px * scale)
        dims = [dict(L_px=px(3.74), cands=[7500.0]), dict(L_px=px(1.0), cands=[1000.0]), dict(L_px=px(1.1), cands=[1000.0, 10.0])]
        kept = dm.consistent_dims(dims, mm_px, scale)
        self.assertEqual([d["cands"][0] for d in kept], [1000.0, 1000.0])

    def test_wall_classes_relative_threshold(self):
        from plan2cad import vectorpdf as vp
        # murs 0,57 (0,99) et 0,42 (0,8) ; mobilier 0,72 (0,25) et 0,36 (0,3)
        got = vp.pick_wall_classes({0.36: 0.3, 0.42: 0.8, 0.57: 0.99, 0.72: 0.25})
        self.assertEqual(got, [0.42, 0.57])
        self.assertEqual(vp.pick_wall_classes({0.5: 0.1}), [])
        self.assertEqual(vp.pick_wall_classes({}), [])


class ConcordanceStricte(unittest.TestCase):
    def test_half_agreement_rejected(self):
        from plan2cad import dimensions as dm
        import inspect
        src = inspect.getsource(dm.estimate_scale)
        self.assertIn("-(-6 * len(props) // 10)", src)   # 60 % arrondi au-dessus : 3 sur 6 ne suffit plus
