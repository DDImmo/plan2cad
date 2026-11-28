import unittest
import numpy as np
from plan2cad.model import Poly
from plan2cad import facade


def rect(x0, y0, x1, y1, layer):
    return Poly(pts=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float), layer=layer, closed=True)


class Facade(unittest.TestCase):
    def test_window_vs_furniture(self):
        wall = Poly(pts=np.array([[0, 0], [8000, 0], [8000, 6000], [0, 6000], [0, 0]], float), layer="ARC-MURS-EXT", closed=True)
        win = rect(2000, 20, 3500, 150, "ARC-MOBILIER")       # bande fine collée à la façade
        table = rect(2000, 1500, 3200, 2400, "ARC-MOBILIER")   # meuble au milieu de la pièce
        deep = rect(5000, 20, 5300, 1800, "ARC-MOBILIER")      # perpendiculaire / trop épais
        polys = [wall, win, table, deep]
        n = facade.tag_windows(polys)
        self.assertEqual(n, 1)
        self.assertEqual(win.layer, "ARC-FENETRE")
        self.assertEqual(table.layer, "ARC-MOBILIER")
        self.assertEqual(deep.layer, "ARC-MOBILIER")


if __name__ == "__main__":
    unittest.main()
