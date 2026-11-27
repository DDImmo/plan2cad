import unittest

from plan2cad.vectorpdf import _group_chars


def ch(text, x, size, matrix, upright, font="ArialMT"):
    return dict(text=text, x0=x, x1=x + 1, top=0, bottom=1, size=size, matrix=matrix, upright=upright,
                fontname=font, adv=1.0, non_stroking_color=(0, 0, 0))


class RotatedTextSize(unittest.TestCase):
    def test_tiny_rotated_label_keeps_its_own_size(self):
        """Même police, même échelle de matrice : nom de pièce horizontal à 5,2 pt et libellé tourné à 1 pt.
        Le libellé tourné ne doit pas hériter du corps du texte horizontal (bug : 5x trop grand, libellés voisins fusionnés)."""
        room = [ch(c, 100 + i * 3, 5.2, (0.03, 0, 0, 0.03, 100 + i * 3, 400), True) for i, c in enumerate("Salle")]
        tiny = [ch(c, 300, 1.0, (0.0, -0.03, 0.03, 0.0, 300, 400 - i * 1.0), False) for i, c in enumerate("8Z")]
        texts = _group_chars(room + tiny, 842.0)
        sizes = {t["s"]: t["size"] for t in texts}
        self.assertAlmostEqual(sizes["Salle"], 5.2, delta=0.3)
        self.assertLess(sizes["8Z"], 1.6)


if __name__ == "__main__":
    unittest.main()
