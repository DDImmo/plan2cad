import unittest
import numpy as np

from plan2cad import fit
from plan2cad.model import Poly


def circle_pts(c, r, a0, a1, step=6.0):
    t = np.radians(np.arange(a0, a1 + 1e-9, step))
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


class FitTests(unittest.TestCase):
    def test_arc_becomes_arc(self):
        pts = circle_pts((100, 50), 400, 20, 110)
        pcs = fit.fit_polyline(pts, False, 0.0)
        self.assertEqual([p[0] for p in pcs], ["arc"])
        self.assertAlmostEqual(pcs[0][3], 400, delta=1.0)

    def test_closed_circle(self):
        pts = circle_pts((0, 0), 250, 0, 360 - 8, 8)
        pcs = fit.fit_polyline(pts, True, 0.0)
        self.assertEqual(pcs[0][0], "circle")

    def test_rectangle_corners_are_not_a_circle(self):
        pts = np.array([[0, 0], [0, 100], [200, 100], [200, 0]], float)     # 4 coins : cocycliques
        self.assertEqual([p[0] for p in fit.fit_polyline(pts, False, 0.0)], ["poly"])

    def test_straight_wall_untouched(self):
        pts = np.c_[np.linspace(0, 700, 120), 0.4 * np.sin(np.linspace(0, 40, 120))]
        self.assertIsNone(fit.refit_chain(pts, False))

    def test_refit_chain_arc_and_circle(self):
        rng = np.random.default_rng(0)
        arc = circle_pts((300, 300), 120, 0, 100, 0.5) + rng.normal(0, 0.25, (201, 2))
        self.assertIsNotNone(fit.refit_chain(arc, False))
        ring = circle_pts((300, 300), 45, 0, 359, 1.0) + rng.normal(0, 0.25, (360, 2))
        self.assertIsNotNone(fit.refit_chain(ring, True))

    def test_arc_plus_leaf_is_split(self):
        arc = circle_pts((0, 0), 130, 0, 90, 5.0)
        leaf = np.array([[130, 0], [130, -1], [0, 0]], float)
        ents, n_arc, _ = fit.to_entities([Poly(np.vstack([leaf[[1, 0]][:1], arc]), "ARC-PORTE")])
        self.assertEqual(n_arc, 1)

    def test_has_arc_guards_regularization(self):
        self.assertTrue(fit.has_arc(circle_pts((0, 0), 250, 0, 352, 8), True))
        self.assertFalse(fit.has_arc(np.array([[0, 0], [0, 100], [200, 100]], float), False))


if __name__ == "__main__":
    unittest.main()
