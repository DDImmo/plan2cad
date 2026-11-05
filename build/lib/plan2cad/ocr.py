"""Lecture du texte des plans (titres, noms de pièces, légendes) avec Tesseract, s'il est installé.

Sans Tesseract tout fonctionne comme avant : le texte n'est simplement pas lu (message explicite dans le journal).
"""
from __future__ import annotations

import difflib
import os
import shutil
import subprocess
import tempfile
import unicodedata

import cv2
import numpy as np


def available() -> bool:
    return shutil.which("tesseract") is not None


def _langs() -> str:
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True, timeout=20).stdout.split()
    except Exception:
        return "eng"
    return "fra+eng" if "fra" in out else "eng"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def ocr_words(gray: np.ndarray, min_conf: float = 45.0, psm: int = 11, extra: list | None = None):
    """Renvoie une liste de dicts {text, x, y, w, h, conf, line} en pixels de `gray` (uint8, fond blanc).
    `line` : identifiant de ligne de texte (les mots d'une même ligne sont regroupés par `phrases`)."""
    if not available():
        return []
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "in.png")
        cv2.imwrite(fn, g)
        env = dict(os.environ, OMP_THREAD_LIMIT="2")
        try:
            r = subprocess.run(["tesseract", fn, "stdout", "--psm", str(psm), "-l", _langs()] + list(extra or []) + ["tsv"],
                               capture_output=True, text=True, timeout=300, env=env)
        except Exception:
            return []
    words = []
    for ln in r.stdout.splitlines()[1:]:
        c = ln.split("\t")
        if len(c) < 12:
            continue
        txt = c[11].strip()
        try:
            conf = float(c[10])
        except ValueError:
            continue
        if not txt or conf < min_conf or not any(ch.isalnum() for ch in txt):
            continue
        x, y, w, h = (int(c[i]) for i in (6, 7, 8, 9))
        if w < 3 or h < 5 or h > 0.2 * g.shape[0]:
            continue
        words.append(dict(text=txt, x=x, y=y, w=w, h=h, conf=conf, line=(c[2], c[3], c[4])))
    return words


def phrases(words):
    """Regroupe les mots voisins d'une même ligne : « PLAN NIVEAU 0 »."""
    out = []
    by = {}
    for w in words:
        by.setdefault(w["line"], []).append(w)
    for ws in by.values():
        ws.sort(key=lambda w: w["x"])
        cur = [ws[0]]
        for w in ws[1:]:
            prev = cur[-1]
            if w["x"] - (prev["x"] + prev["w"]) <= 1.6 * max(prev["h"], w["h"]):
                cur.append(w)
            else:
                out.append(cur); cur = [w]
        out.append(cur)
    res = []
    for g in out:
        x0 = min(w["x"] for w in g); y0 = min(w["y"] for w in g)
        x1 = max(w["x"] + w["w"] for w in g); y1 = max(w["y"] + w["h"] for w in g)
        res.append(dict(text=" ".join(w["text"] for w in g), x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                        conf=float(np.mean([w["conf"] for w in g]))))
    return res


# --------------------------------------------------------------- lexique
LOT_WORDS = {
    "elec": ["electricite", "electrique", "elec", "courant fort", "courant faible", "eclairage", "prises",
             "luminaire", "tableau electrique", "tgbt", "interrupteur", "reseau electrique"],
    "plumb": ["plomberie", "sanitaire", "sanitaires", "evacuation", "evacuations", "assainissement",
              "eau froide", "eau chaude", "eaux usees", "eau potable", "alimentation eau"],
    "cvc": ["chauffage", "ventilation", "vmc", "cvc", "climatisation", "radiateur", "chaudiere", "hvac"],
    "gaz": ["gaz", "gaz naturel"],
}
ARCH_WORDS = ["architecte", "architecture", "plan niveau", "niveau", "rdc", "rez de chaussee", "etage",
              "facade", "coupe", "plan masse", "gros oeuvre", "cotes"]
ROOM_WORDS = ["salon", "sejour", "cuisine", "chambre", "bureau", "sdb", "salle de bain", "salle d'eau", "wc",
              "toilettes", "entree", "hall", "couloir", "degagement", "garage", "cellier", "buanderie",
              "dressing", "terrasse", "balcon", "cave", "local", "atelier", "reunion", "accueil", "stockage",
              "sanitaires", "circulation", "palier", "wc", "salle", "bureaux", "archives", "vestiaire"]
STAIR_WORDS = ["escalier", "esc", "ascenseur", "asc", "monte-charge", "monte charge"]


def _match(tokens: str, lex: list[str]) -> bool:
    t = norm(tokens)
    for ch in ".,;:()[]/\\|":
        t = t.replace(ch, " ")
    t = " ".join(t.split())
    for w in lex:
        if " " in w or len(w) <= 3:
            if w in t.split() or w in t:
                if len(w) > 3 or w in t.split():
                    return True
        else:
            for tok in t.replace("'", " ").replace("-", " ").split():
                if tok == w or (len(w) >= 6 and (tok.startswith(w[:-2]) or difflib.SequenceMatcher(None, tok, w).ratio() >= 0.82)):
                    return True
    return False


def lot_votes(phr: list[dict]):
    """Score par lot d'après les textes d'un plan : poids = hauteur du texte (les titres sont plus grands)."""
    if not phr:
        return {}, []
    hs = np.array([p["h"] for p in phr], float)
    ref = float(np.median(hs))
    votes: dict[str, float] = {}
    why: list[str] = []
    for p in phr:
        wgt = float(np.clip(p["h"] / max(ref, 1), 1.0, 3.0))
        for lot, lex in LOT_WORDS.items():
            if _match(p["text"], lex):
                votes[lot] = votes.get(lot, 0.0) + wgt
                why.append(f"« {p['text']} » → {lot}")
        if _match(p["text"], ARCH_WORDS):
            votes["arc"] = votes.get("arc", 0.0) + 0.5 * wgt
            why.append(f"« {p['text']} » → plan d'étage")
    return votes, why


def room_labels(phr: list[dict]):
    return [p for p in phr if _match(p["text"], ROOM_WORDS)]


def stair_labels(phr: list[dict]):
    return [p for p in phr if _match(p["text"], STAIR_WORDS)]


# --------------------------------------------------------------- lecture par blocs (orientations, tailles)
def _text_groups(ink: np.ndarray, mm_px: float, min_h_mm: float, max_h_mm: float):
    """Blocs de texte candidats : composantes de la taille d'une lettre, regroupées en lignes par dilatation,
    par classe de taille (un titre et une petite légende ne se mélangent pas)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return [], lab, st
    w, h = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]
    dim, small = np.maximum(w, h), np.minimum(w, h)
    lo, hi = min_h_mm / mm_px, max_h_mm / mm_px
    tl = (dim >= lo) & (dim <= hi) & (small >= 1) & (dim <= 9 * np.maximum(small, 1))
    tl[0] = False
    groups = []
    ne = int(np.ceil(np.log(hi / lo) / np.log(1.7))) + 1
    edges = [lo * 1.7 ** i for i in range(ne + 1)]
    for orient in ("h", "v"):
        cv_ = h if orient == "h" else w                    # hauteur de lettre : indépendante des lettres collées
        for a, b in zip(edges[:-1], edges[1:]):
            cls = tl & (cv_ >= a / 1.3) & (cv_ < b * 1.3)   # classes qui se recouvrent : pas de lettre coupée en deux
            if cls.sum() < 1:
                continue
            m = float(np.median(cv_[cls]))
            mask = (cls[lab] * 255).astype(np.uint8)
            for fk in (1.1, 2.2):                              # lettres serrées, puis lettres très espacées (enseignes, titres)
                groups += _groups_for(mask, lab, st, cls, m, fk, orient)
    return groups, lab, st


def _groups_for(mask, lab, st, cls, m, fk, orient):
    groups = []
    if True:
        if True:
            k = int(max(3, fk * m))
            ker = (k, 3) if orient == "h" else (3, k)
            dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, ker))
            ng, gl, gs, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
            sel = mask > 0
            cid_px, gid_px = lab[sel], gl[sel]
            uc, first = np.unique(cid_px, return_index=True)       # un pixel de chaque composante -> son groupe
            by = {}
            for cid, g in zip(uc, gid_px[first]):
                if g > 0:
                    by.setdefault(int(g), []).append(int(cid))
            for g, cids in by.items():
                # les lettres d'une ligne sont alignées : on écarte celles qui s'en éloignent (lettre voisine d'un autre texte)
                if len(cids) >= 3:
                    cc = (st[cids, 1] + st[cids, 3] / 2.0) if orient == "h" else (st[cids, 0] + st[cids, 2] / 2.0)
                    keep = np.abs(cc - np.median(cc)) <= 0.75 * m
                    cids = [c for c, kp in zip(cids, keep) if kp]
                    if not cids:
                        continue
                x0 = int(st[cids, 0].min()); y0 = int(st[cids, 1].min())
                x1 = int((st[cids, 0] + st[cids, 2]).max()); y1 = int((st[cids, 1] + st[cids, 3]).max())
                gw, gh = x1 - x0, y1 - y0
                if orient == "v" and (gh < 2.0 * gw or len(cids) < 2):
                    continue
                if orient == "h" and gw < gh * 0.8 and len(cids) < 2:
                    continue
                groups.append(dict(orient=orient, comps=cids, x=x0, y=y0, w=gw, h=gh, n=len(cids)))
    return groups


def _montage(items, target_h=44, gap=36, max_rows=60):
    """Empile des lignes de texte horizontales (déjà mises à l'endroit) en images lisibles d'un coup."""
    pages = []
    for i in range(0, len(items), max_rows):
        chunk = items[i:i + max_rows]
        W = max(c.shape[1] for _, c in chunk) + 40
        rows, y = [], 20
        for k, (_, c) in enumerate(chunk):
            rows.append((k, y, y + c.shape[0]))
            y += c.shape[0] + gap
        img = np.full((y + 20, W), 255, np.uint8)
        for (k, y0, y1), (_, c) in zip(rows, chunk):
            img[y0:y1, 20:20 + c.shape[1]] = c
        pages.append((img, rows, chunk))
    return pages


def _read_hypothesis(gray, groups, rot, pad_f=0.35, target_h=44, extra=None):
    """Lit les groupes en supposant la rotation `rot` (0, 90 : texte lisible de bas en haut, 270 : de haut en bas)."""
    H, W = gray.shape[:2]
    items = []
    for gi, g in enumerate(groups):
        base = g["h"] if rot == 0 else g["w"]
        pad = int(max(3, pad_f * base))
        x0, y0 = max(0, g["x"] - pad), max(0, g["y"] - pad)
        x1, y1 = min(W, g["x"] + g["w"] + pad), min(H, g["y"] + g["h"] + pad)
        c = gray[y0:y1, x0:x1]
        if rot == 90:
            c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 270:
            c = cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE)
        th = max(1, base)
        f = float(np.clip(target_h / th, 0.5, 5.0))
        c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
        if c.shape[1] > 2600:
            c = cv2.resize(c, None, fx=2600 / c.shape[1], fy=2600 / c.shape[1], interpolation=cv2.INTER_AREA)
        items.append((gi, c))
    out = {}
    for img, rows, chunk in _montage(items):
        words = ocr_words(img, 0.0, psm=11, extra=extra)
        for k, y0, y1 in rows:
            ws = [w for w in words if y0 - 10 <= w["y"] + w["h"] / 2 <= y1 + 10]
            if not ws:
                continue
            ws.sort(key=lambda w: w["x"])
            txt = " ".join(w["text"] for w in ws)
            conf = float(np.mean([w["conf"] for w in ws]))
            out[chunk[k][0]] = (txt, conf)
    return out


def _plausible(txt: str) -> bool:
    t = txt.strip()
    al = sum(c.isalnum() for c in t)
    if al < 2 and not t.isdigit():
        return False
    return al >= 0.6 * len(t.replace(" ", ""))


def read_texts(ink: np.ndarray, gray: np.ndarray, mm_px: float, min_h_mm: float = 0.55, max_h_mm: float = 18.0,
               rotations: bool = True):
    """Lit tous les textes d'un plan, horizontaux ou verticaux, petits ou grands.
    Renvoie (phrases, masque_des_lettres). phrase : dict(text, x, y, w, h, th, angle, conf) en pixels ;
    angle = sens de lecture (0, 90 = de bas en haut, 270 = de haut en bas)."""
    if not available():
        return [], None
    groups, lab, st = _text_groups(ink, mm_px, min_h_mm, max_h_mm)
    if not groups:
        return [], None
    groups = sorted(groups, key=lambda g: -g["w"] * g["h"])[:1500]
    # hypothèses : groupes horizontaux -> 0 ; groupes verticaux -> 90 et 270
    hyp = {0: [i for i, g in enumerate(groups) if g["orient"] == "h"]}
    if rotations:
        v = [i for i, g in enumerate(groups) if g["orient"] == "v"]
        hyp[90], hyp[270] = v, v
    res: dict[int, tuple] = {}
    for rot, ids in hyp.items():
        if not ids:
            continue
        got = _read_hypothesis(gray, [groups[i] for i in ids], rot)
        for j, (txt, conf) in got.items():
            gi = ids[j]
            need = 62.0 if rot == 0 else 78.0
            if conf >= need and _plausible(txt) and (len(txt.replace(" ", "")) >= 2 or rot == 0):
                if gi not in res or conf > res[gi][1]:
                    res[gi] = (txt, conf, rot)
    # arbitrage : un groupe vertical lu = ses lettres sont consommées, on ignore les lectures horizontales des mêmes lettres
    used: set[int] = set()
    out = []
    order = sorted(res, key=lambda gi: (-(groups[gi]["orient"] == "v"), -res[gi][1]))
    for gi in order:
        g = groups[gi]
        cids = [c for c in g["comps"] if c not in used]
        if len(cids) < max(1, int(0.6 * len(g["comps"]))):
            continue
        used.update(g["comps"])
        txt, conf, rot = res[gi]
        th = g["h"] if rot == 0 else g["w"]
        out.append(dict(text=txt, x=g["x"], y=g["y"], w=g["w"], h=g["h"], th=th, angle=float(rot), conf=conf))
    kill = np.zeros(lab.shape, bool) if out else None
    if out:
        ids = np.zeros(len(st), bool)
        for gi in order:
            if gi in res:
                for c in groups[gi]["comps"]:
                    ids[c] = c in used
        kill = ids[lab]
    return out, kill
