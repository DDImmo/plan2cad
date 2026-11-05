"""Charte de calques (inspirée ISO 13567) et couleurs CAO."""
from __future__ import annotations

import numpy as np

# nom -> (couleur ACI, description)
CHARTER: dict[str, tuple[int, str]] = {
    "ARC-MURS-EXT":        (8, "Murs extérieurs, façades, murs porteurs"),
    "ARC-MURS-INT":        (8, "Cloisons et murs intérieurs non porteurs"),
    "ARC-MURS-AXE":        (8, "Axe des murs en double trait avec leur épaisseur réelle (déduit des deux faces)"),
    "ARC-PORTE":           (30, "Portes et sens d'ouverture"),
    "ARC-FENETRE":         (4, "Fenêtres, baies vitrées"),
    "ARC-ESCALIER":        (6, "Escaliers, ascenseurs"),
    "ELEC-COURANT-FORT":   (1, "Prises, tableaux, disjoncteurs, alimentations"),
    "ELEC-ECLAIRAGE":      (1, "Luminaires, interrupteurs, circuits lumière"),
    "ELEC-COURANT-FAIBLE": (1, "RJ45, informatique, fibre, téléphonie"),
    "ELEC-SECURITE":       (1, "Alarme incendie, caméras, BAES"),
    "PLUMB-EAU-POTABLE":   (4, "Eau froide et eau chaude sanitaire"),
    "PLUMB-EVAC-EAUX":     (6, "Évacuations EU / EV"),
    "PLUMB-SANITAIRE":     (5, "Appareils sanitaires"),
    "CVC-CHAUFFAGE":       (30, "Radiateurs, chaudières, planchers chauffants"),
    "CVC-VENTIL":          (3, "Gaines de VMC, grilles"),
    "GAZ-RESEAU":          (40, "Réseau gaz"),
    "ARC-DETAIL":          (9, "Détails fins : mobilier, hachures, symboles (traits les plus fins)"),
    "ARC-COULEUR-ROUGE":   (1, "Tracés rouges de l'architecture (couleur d'origine conservée)"),
    "ARC-COULEUR-VERT":    (3, "Tracés verts de l'architecture (couleur d'origine conservée)"),
    "ARC-COULEUR-BLEU":    (5, "Tracés bleus de l'architecture (couleur d'origine conservée)"),
    "ARC-COULEUR-CYAN":    (4, "Tracés cyan de l'architecture (couleur d'origine conservée)"),
    "ARC-COULEUR-MAGENTA": (6, "Tracés magenta de l'architecture (couleur d'origine conservée)"),
    "ARC-COULEUR-ORANGE":  (30, "Tracés orange de l'architecture (couleur d'origine conservée)"),
    "ARC-COULEUR-JAUNE":   (2, "Tracés jaunes de l'architecture (couleur d'origine conservée)"),
    "ARC-COTES":           (7, "Lignes de cote et leurs valeurs"),
    "ARC-TEXTE":           (7, "Noms de pièces, titres, annotations lus sur le plan"),
    "ELEC-TEXTE":          (1, "Textes du plan d'électricité"),
    "PLUMB-TEXTE":         (4, "Textes du plan de plomberie"),
    "CVC-TEXTE":           (30, "Textes du plan CVC"),
    "GAZ-TEXTE":           (40, "Textes du plan gaz"),
    "Z-TEXTE-NON-LU":      (9, "Tracés qui ressemblent à des lettres que l'OCR n'a pas lues (à supprimer ou saisir)"),
    "Z-TEXTE-INCERTAIN":   (9, "Textes dont la lecture est douteuse (à vérifier)"),
    "AUTRE-NON-CLASSE":    (9, "Éléments non classés (à trier)"),
    "Z-FOND-ARCHI":        (9, "Fond d'architecture des plans de lots (optionnel)"),
    "Z-POINTS-ANCRAGE":    (2, "Repères de calage (deux croix)"),
}

ANCHOR_LAYER = "Z-POINTS-ANCRAGE"

# discipline -> métadonnées
DISCIPLINES = {
    "arc":   {"label": "Architecture / Gros œuvre", "default": "ARC-MURS-INT"},
    "elec":  {"label": "Électricité",               "default": "ELEC-COURANT-FORT"},
    "plumb": {"label": "Plomberie / Fluides",       "default": "PLUMB-EAU-POTABLE"},
    "cvc":   {"label": "CVC",                       "default": "CVC-VENTIL"},
    "gaz":   {"label": "Gaz",                       "default": "GAZ-RESEAU"},
    "autre": {"label": "Non classé",                "default": "AUTRE-NON-CLASSE"},
}

# Légende couleur -> calque, modifiable (--legend legend.json). Conventions courantes, MAIS elles varient
# d'un bureau d'études à l'autre : à ajuster à votre dossier.
DEFAULT_LEGEND = {
    "rouge": "ELEC-COURANT-FORT", "orange": "CVC-CHAUFFAGE", "jaune": "GAZ-RESEAU",
    "vert": "CVC-VENTIL", "cyan": "PLUMB-EAU-POTABLE", "bleu": "PLUMB-EAU-POTABLE",
    "magenta": "PLUMB-EVAC-EAUX", "marron": "PLUMB-EVAC-EAUX",
}


def text_layer(disc: str) -> str:
    return {"arc": "ARC-TEXTE", "elec": "ELEC-TEXTE", "plumb": "PLUMB-TEXTE", "cvc": "CVC-TEXTE",
            "gaz": "GAZ-TEXTE"}.get(disc, "ARC-TEXTE")


def disc_of_layer(name: str) -> str:
    p = name.split("-")[0].upper()
    return {"ARC": "arc", "ELEC": "elec", "PLUMB": "plumb", "CVC": "cvc", "GAZ": "gaz"}.get(p, "autre")

# couleur ACI -> hex (lisible sur fond clair ET sombre, pour l'interface web)
ACI_HEX = {
    1: "#e5322d", 2: "#d4a200", 3: "#1f9d55", 4: "#0ea5c6", 5: "#2f5bea",
    6: "#c026d3", 7: "#8a8f98", 8: "#7b8190", 9: "#b4bac4", 30: "#f08a00", 40: "#c98f00",
}


def aci_hex(aci: int) -> str:
    return ACI_HEX.get(aci, "#8a8f98")


def layer_color(name: str, file_colors: dict | None = None) -> int:
    if name in CHARTER:
        return CHARTER[name][0]
    if file_colors and name in file_colors:
        return file_colors[name]
    return 7


def split_walls_by_width(widths_paper_mm: np.ndarray):
    """Sépare traits épais (murs extérieurs) / fins (cloisons) si la distribution est réellement bimodale.
    Renvoie le seuil, ou None si pas de séparation nette (y compris si tous les traits ont la même épaisseur)."""
    w = np.sort(np.asarray(widths_paper_mm, dtype=float))
    w = w[w > 0]
    if len(w) < 8 or w[-1] - w[0] < 0.2 or w[-1] / w[0] < 1.8:
        return None
    # coupe optimale en deux classes (variance inter-classes maximale), sans dépendre d'un histogramme
    csum = np.cumsum(w)
    n = len(w)
    best, t = -1.0, None
    for i in range(1, n):
        if w[i] == w[i - 1]:
            continue
        m1, m2 = csum[i - 1] / i, (csum[-1] - csum[i - 1]) / (n - i)
        score = i * (n - i) * (m2 - m1) ** 2
        if score > best:
            best, t = score, (w[i - 1] + w[i]) / 2
    if t is None:
        return None
    lo, hi = w[w <= t], w[w > t]
    if len(lo) and len(hi) and hi.mean() / max(lo.mean(), 1e-6) >= 1.8 and (hi.mean() - lo.mean()) >= 0.4:
        return float(t)
    return None
