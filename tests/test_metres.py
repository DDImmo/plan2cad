import unittest

import numpy as np

from plan2cad import metres as mt


def rect(x0, y0, x1, y1, layer="ARC-MURS-INT"):
    return dict(type="poly", layer=layer, pts=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), closed=True)


def wall_ring(x0, y0, x1, y1, t=200.0):
    """Mur en double trait : face extérieure et face intérieure, épaisseur t."""
    return [rect(x0, y0, x1, y1), rect(x0 + t, y0 + t, x1 - t, y1 - t)]


class RoomTests(unittest.TestCase):
    def test_printed_area(self):
        self.assertEqual(mt.printed_area("5,27 m²"), 5.27)
        self.assertEqual(mt.printed_area("102 m2"), 102.0)
        self.assertIsNone(mt.printed_area("RDC Bureau 4"))

    def test_single_room_net_area(self):
        # pièce 4,0 x 3,0 m nets, murs de 200 mm : la surface doit être celle de la face intérieure
        ents = wall_ring(0, 0, 4400, 3400)
        rooms, info = mt.compute_rooms(ents)
        self.assertEqual(len(rooms), 1)
        self.assertAlmostEqual(rooms[0]["surface_m2"], 12.0, delta=0.15)
        self.assertAlmostEqual(rooms[0]["perimetre_m"], 14.0, delta=0.4)   # angles légèrement arrondis (voir metres.py)

    def test_door_does_not_merge_rooms_but_symbol_closes_gap(self):
        # deux pièces côte à côte, séparées par une cloison avec une porte (ouverture de 900 mm fermée par le segment de porte)
        ents = wall_ring(0, 0, 8400, 3400)
        gap = [dict(type="poly", layer="ARC-MURS-INT", pts=np.array([[4200, 200], [4200, 1000]], float), closed=False),
               dict(type="poly", layer="ARC-MURS-INT", pts=np.array([[4200, 1900], [4200, 3200]], float), closed=False)]
        door = dict(type="poly", layer="ARC-PORTE", pts=np.array([[4200, 1000], [4200, 1900]], float), closed=False)
        rooms, _ = mt.compute_rooms(ents + gap + [door])
        self.assertEqual(len(rooms), 2)                    # la porte ferme le passage : deux pièces, pas une
        self.assertTrue(all(r["portes"] == 1 for r in rooms))
        rooms2, _ = mt.compute_rooms(ents + gap)
        self.assertEqual(len(rooms2), 1)                   # sans symbole de porte, elles communiquent

    def test_exterior_is_not_a_room(self):
        rooms, _ = mt.compute_rooms(wall_ring(0, 0, 3000, 3000))
        self.assertEqual(len(rooms), 1)

    def test_printed_area_check(self):
        ents = wall_ring(0, 0, 4400, 3400)
        texts = [dict(s="Salon", pos=np.array([1500.0, 1500.0]), height=300.0, angle=0.0),
                 dict(s="12,0 m²", pos=np.array([1500.0, 1100.0]), height=300.0, angle=0.0)]
        rooms, _ = mt.compute_rooms(ents, texts)
        self.assertEqual(rooms[0]["surface_lue_m2"], 12.0)
        self.assertLess(abs(rooms[0]["ecart_pct"]), 2.0)
        self.assertEqual(rooms[0]["libelle"], "Salon")


if __name__ == "__main__":
    unittest.main()
