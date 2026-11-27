"""Lecture DIRECTE d'un PDF vectoriel (export CAO) : tracés et textes exacts, sans passer par un raster ni l'OCR.

Un PDF issu d'un logiciel de CAO contient déjà : des polylignes avec leur épaisseur et leur couleur, et du texte
(chaîne, position, angle, corps). On les lit tels quels :
  - la planche est découpée en plans (amas de tracés) ; chaque plan est titré par le texte voisin
    (« RDC EXISTANT », « R+1 PROJET 3 » ...) ;
  - les calques se déduisent de l'épaisseur du trait, de la couleur, des cotes et des arcs de porte ;
  - les textes (pièces, surfaces, cotes) sont repris sans erreur de lecture.
Dépendance : pdfplumber (pur Python).  pip install pdfplumber
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

PT_MM = 25.4 / 72.0            # 1 point PDF = 0.3528 mm de papier
_CACHE: dict = {}


def available() -> bool:
    try:
        import pdfplumber  # noqa: F401
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ lecture
def _flatten_bezier(p0, p1, p2, p3, n=6):
    t = np.linspace(0, 1, n + 1)[1:, None]
    p0, p1, p2, p3 = (np.asarray(v, float) for v in (p0, p1, p2, p3))
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _color(c):
    if c is None:
        return (0.0, 0.0, 0.0)
    if isinstance(c, (int, float)):
        return (float(c),) * 3
    c = tuple(float(x) for x in c)
    if len(c) == 1:
        return c * 3
    if len(c) == 4:                                    # CMJN
        k = c[3]
        return tuple((1 - min(1.0, v + k)) for v in c[:3])
    if len(c) >= 3:
        return c[:3]
    return (0.0, 0.0, 0.0)


def _path_pts(o):
    """Points (x, y vers le bas) d'un objet pdfplumber ; les courbes de Bézier sont discrétisées.
    Renvoie (pts, closed, n_beziers)."""
    path = o.get("path")
    if not path:
        pts = [tuple(p) for p in (o.get("pts") or [])]
        return (np.array(pts, float) if len(pts) >= 2 else None), False, 0
    out, cur, closed, nb = [], None, False, 0
    for seg in path:
        op = seg[0]
        if op == "m":
            cur = tuple(seg[1])
            out.append(cur)
        elif op == "l" and cur is not None:
            cur = tuple(seg[1])
            out.append(cur)
        elif op == "c" and cur is not None:
            b = _flatten_bezier(cur, seg[1], seg[2], seg[3])
            out.extend(map(tuple, b))
            cur = tuple(seg[3])
            nb += 1
        elif op == "v" and cur is not None:
            b = _flatten_bezier(cur, cur, seg[1], seg[2]); out.extend(map(tuple, b)); cur = tuple(seg[2]); nb += 1
        elif op == "y" and cur is not None:
            b = _flatten_bezier(cur, seg[1], seg[2], seg[2]); out.extend(map(tuple, b)); cur = tuple(seg[2]); nb += 1
        elif op == "h":
            closed = True
        elif op == "re":
            x, y, w, h = seg[1:5] if len(seg) >= 5 else (seg[1][0], seg[1][1], seg[2], seg[3])
            out.extend([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            closed = True
    if len(out) < 2:
        return None, False, 0
    arr = np.array(out, float)
    if len(arr) > 2 and np.allclose(arr[0], arr[-1]):
        closed, arr = True, arr[:-1]
    return arr, closed, nb


def _group_chars(chars, H):
    """Chars -> chaînes. L'ordre du flux du PDF est conservé (une chaîne est écrite d'un seul tenant)."""
    texts, cur = [], None
    # corps de police : pdfplumber donne une taille fausse pour les caractères tournés ; on la déduit de la matrice
    ratios: dict = {}
    for ch in chars:
        m = ch.get("matrix")
        if m and ch.get("upright", True) and m[0] and ch.get("size"):
            ratios.setdefault(ch.get("fontname"), []).append(float(ch["size"]) / math.hypot(m[0], m[1]))
    ratio = {k: float(np.median(v)) for k, v in ratios.items()}
    allr = float(np.median([x for v in ratios.values() for x in v])) if ratios else 1.0

    def flush():
        nonlocal cur
        if cur and cur["s"].strip():
            texts.append(cur)
        cur = None

    for ch in chars:
        m = ch.get("matrix") or (1, 0, 0, 1, ch["x0"], H - ch["bottom"])
        a, b = m[0], m[1]
        ang = math.degrees(math.atan2(b, a)) % 360.0
        ang = round(ang / 90.0) % 4 * 90.0 if min(ang % 90, 90 - ang % 90) < 2.0 else ang
        sc = math.hypot(a, b) or 1.0
        if ratios and ch.get("upright", True):
            size = sc * ratio.get(ch.get("fontname"), allr)
        elif ch.get("size"):
            # caractère tourné : le rapport corps/échelle appris sur le texte horizontal ne vaut PAS pour lui (une même police et
            # une même matrice servent à des corps très différents : AutoCAD, libellés de 1 pt à côté de noms de pièces de 5 pt).
            # La taille mesurée par pdfplumber (largeur du glyphe) est alors la meilleure estimation.
            size = float(ch["size"])
        else:
            size = sc * ratio.get(ch.get("fontname"), allr) if ratios else 8.0
        size = max(size, 0.1)
        ox, oy = float(m[4]), H - float(m[5])                       # origine (ligne de base), y vers le bas
        adv = float(ch["adv"]) * sc if ch.get("adv") else (ch["x1"] - ch["x0"])
        t = ch["text"]
        if cur is not None:
            ca = math.radians(cur["angle"])
            ux, uy = math.cos(ca), -math.sin(ca)                    # direction du texte, y vers le bas
            dx, dy = ox - cur["end"][0], oy - cur["end"][1]
            along, across = dx * ux + dy * uy, -dx * uy + dy * ux
            ok = (abs(ang - cur["angle"]) < 1.0 and abs(size / cur["size"] - 1) < 0.06
                  and abs(across) < 0.35 * size and -0.3 * size < along < 1.4 * size)
            if not ok:
                flush()
        if cur is None:
            cur = dict(s="", angle=ang, size=size, x=ox, y=oy, end=(ox, oy), bb=[1e9, 1e9, -1e9, -1e9],
                       color=_color(ch.get("non_stroking_color")))
        cur["s"] += t
        ca = math.radians(cur["angle"])
        cur["end"] = (ox + adv * math.cos(ca), oy - adv * math.sin(ca))
        cur["bb"][0] = min(cur["bb"][0], ch["x0"]); cur["bb"][1] = min(cur["bb"][1], ch["top"])
        cur["bb"][2] = max(cur["bb"][2], ch["x1"]); cur["bb"][3] = max(cur["bb"][3], ch["bottom"])
    flush()
    for t in texts:
        t["s"] = t["s"].strip()
    return texts


def load(path: str, page: int = 1) -> dict:
    """Page -> dict(w, h, paths=[...], texts=[...]) ; coordonnées en points, y vers le bas."""
    key = (str(path), page)
    if key in _CACHE:
        return _CACHE[key]
    import pdfplumber
    with pdfplumber.open(str(path)) as pdf:
        pg = pdf.pages[page - 1]
        W, H = float(pg.width), float(pg.height)
        paths = []
        for o in list(pg.lines) + list(pg.curves) + list(pg.rects):
            stroked = o.get("stroke", True)
            if not stroked and not o.get("fill"):
                continue
            pts, closed, nb = _path_pts(o)
            if pts is None:
                continue
            col = _color(o.get("stroking_color") if stroked else o.get("non_stroking_color"))
            paths.append(dict(pts=pts, closed=closed, width=float(o.get("linewidth") or 0.0), rgb=col,
                              stroke=bool(stroked), fill=bool(o.get("fill")), nb=nb,
                              bbox=(float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))))
        texts = _group_chars(pg.chars, H)
    res = dict(w=W, h=H, paths=paths, texts=texts)
    _CACHE[key] = res
    return res


def is_vector(path: str, page: int = 1, min_paths: int = 150) -> bool:
    """Vrai si la page est un dessin vectoriel (et non un scan inséré comme image)."""
    if not str(path).lower().endswith(".pdf") or not available():
        return False
    try:
        d = load(path, page)
    except Exception:
        return False
    return len(d["paths"]) >= min_paths


# ------------------------------------------------------------------ planche -> plans
_TITLE = re.compile(r"(rdc|rez|r\s*\+\s*\d|etage|niveau|niv\b|sous.?sol|existant|projet|etat des lieux|batiment|bat\b|bloc)", re.I)


def _strip(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def variant_of(text: str) -> str | None:
    """« EXISTANT », « PROJET 3 » : état / variante du bâtiment (devient le nom du bâtiment)."""
    t = _strip(text).lower()
    if re.search(r"existant|etat des lieux|\bedl\b", t):
        return "Existant"
    m = re.search(r"projet\s*(?:n[°o]\s*)?([a-z0-9]+)?", t)
    if m:
        return "Projet" + (f" {m.group(1).upper()}" if m.group(1) else "")
    return None


def split_plans(d: dict, gap_pt: float = 20.0, min_paths: int = 100, min_size_pt: float = 120.0):
    """Amas de tracés (dilatation de leurs boîtes) = plans. Renvoie [(bbox, [indices de tracés])]."""
    import cv2
    W, H = d["w"], d["h"]
    s = 1.0
    m = np.zeros((int(H * s) + 2, int(W * s) + 2), np.uint8)
    for p in d["paths"]:
        pts = (p["pts"] * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [pts], p["closed"], 255, 1)
    k = int(gap_pt * s) | 1
    dil = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    groups: dict[int, list] = {}
    for i, p in enumerate(d["paths"]):
        cx, cy = int((p["bbox"][0] + p["bbox"][2]) / 2 * s), int((p["bbox"][1] + p["bbox"][3]) / 2 * s)
        cx, cy = min(max(cx, 0), lab.shape[1] - 1), min(max(cy, 0), lab.shape[0] - 1)
        groups.setdefault(int(lab[cy, cx]), []).append(i)
    out = []
    for g, idx in groups.items():
        if g == 0 or len(idx) < min_paths:
            continue
        b = np.array([d["paths"][i]["bbox"] for i in idx])
        bb = (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())
        if min(bb[2] - bb[0], bb[3] - bb[1]) < min_size_pt:
            continue
        out.append((bb, idx))
    out.sort(key=lambda t: (round(t[0][1] / (0.25 * H)), t[0][0]))
    return out


def title_for(d: dict, bb, margin: float = 90.0):
    """Texte-titre voisin (hors du plan) : le plus proche qui ressemble à « RDC EXISTANT »."""
    x0, y0, x1, y1 = bb
    best = None
    for t in d["texts"]:
        b = t["bb"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not _TITLE.search(_strip(t["s"])):
            continue
        if not (x0 - 20 <= cx <= x1 + 20):
            continue
        if y0 <= cy <= y1:
            continue
        dist = (cy - y1) if cy > y1 else (y0 - cy)
        if dist > margin:
            continue
        if best is None or dist < best[0]:
            best = (dist, t["s"])
    return best[1] if best else None


# ------------------------------------------------------------------ classement en calques
_CHROMA = [("rouge", (1, 0, 0)), ("vert", (0, 0.6, 0)), ("bleu", (0, 0, 1)), ("cyan", (0, 1, 1)),
           ("magenta", (1, 0, 1)), ("orange", (1, 0.55, 0)), ("jaune", (1, 0.9, 0))]


def color_name(rgb):
    """Nom de couleur d'un tracé coloré, ou None si noir / gris."""
    r, g, b = rgb
    mx, mn = max(rgb), min(rgb)
    if mx - mn < 0.2:
        return None
    best = min(_CHROMA, key=lambda c: sum((u - v) ** 2 for u, v in zip(rgb, c[1])))
    return best[0]


def circle_fit(pts: np.ndarray):
    """Cercle (cx, cy, r) ajusté par moindres carrés ; résidu relatif et angle balayé (deg)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = float(np.abs(np.hypot(x - cx, y - cy) - r).max() / r)
    ang = np.unwrap(np.arctan2(y - cy, x - cx))
    return cx, cy, r, res, math.degrees(abs(ang[-1] - ang[0]))


def door_scale(paths: list):
    """Échelle d'après les arcs de battant de porte (quart de cercle) : une porte fait environ 0,90 m.
    Renvoie dict(scale, n, of, rayon_median_mm) ou None. Moins fiable qu'une cote : à confirmer."""
    from .dimensions import STD_METRIC
    rs = []
    for p in paths:
        if p["nb"] >= 1 and len(p["pts"]) >= 5 and not p["closed"]:
            f = circle_fit(p["pts"])
            if f and f[3] < 0.03 and 80 <= f[4] <= 100:
                r = f[2] * PT_MM
                if 2.0 <= r <= 8.0:                          # 2-8 mm de papier : exclut sanitaires, façades courbes
                    rs.append(r)
    if len(rs) < 3:
        return None
    med = float(np.median(rs))
    s = min(STD_METRIC, key=lambda x: abs(med * x - 900.0))
    if abs(med * s / 900.0 - 1) > 0.2:
        return None
    ok = [r for r in rs if 650 <= r * s <= 1100]
    if len(ok) < 3 or len(ok) < 0.5 * len(rs):
        return None
    return dict(scale=float(s), n=len(ok), of=len(rs), rayon_median_mm=round(med, 2), source="largeur des portes (≈ 0,90 m)")


# ------------------------------------------------------------------ cotes dessinées en tracés (texte converti en courbes)
_TPL: dict = {}
_TW, _TH = 18, 28


def _norm_glyph(img: np.ndarray):
    """Image binaire (255 = encre) -> vecteur normalisé (boîte englobante ramenée à _TW x _TH, légèrement floutée)."""
    import cv2
    ys, xs = np.nonzero(img)
    if len(xs) == 0:
        return None
    c = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    c = cv2.resize(c, (_TW, _TH), interpolation=cv2.INTER_AREA)
    c = cv2.GaussianBlur(c.astype(np.float32), (0, 0), 1.2)
    c -= c.mean()
    n = float(np.linalg.norm(c))
    return (c / n).ravel() if n > 1e-6 else None


_HOLES = {"0": (1,), "1": (0,), "2": (0,), "3": (0,), "4": (0, 1), "5": (0,), "6": (1,), "7": (0,), "8": (2,), "9": (1,)}


def _holes(img: np.ndarray) -> int:
    """Nombre de trous (boucles fermées) d'un glyphe binaire : sépare 6/5, 9/4, 8/3, 0/..."""
    import cv2
    ys, xs = np.nonzero(img)
    if len(xs) == 0:
        return 0
    c = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    c = cv2.copyMakeBorder(c, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    cnts, hier = cv2.findContours(c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    area_min = 0.004 * c.shape[0] * c.shape[1]
    return sum(1 for i, h in enumerate(hier[0]) if h[3] >= 0 and cv2.contourArea(cnts[i]) > area_min)


def _templates():
    if _TPL:
        return _TPL
    import cv2
    fonts = [(cv2.FONT_HERSHEY_SIMPLEX, 2.2, 3), (cv2.FONT_HERSHEY_PLAIN, 3.2, 3), (cv2.FONT_HERSHEY_DUPLEX, 2.2, 3),
             (cv2.FONT_HERSHEY_COMPLEX_SMALL, 3.0, 3), (cv2.FONT_HERSHEY_SIMPLEX, 2.2, 2)]
    for ch in "0123456789":
        for font, sc_, th_ in fonts:
            im = np.zeros((90, 70), np.uint8)
            cv2.putText(im, ch, (8, 70), font, sc_, 255, th_)
            v = _norm_glyph(im)
            if v is not None:
                _TPL.setdefault(ch, []).append(v)
    return _TPL


def _glyph_vec(paths_: list, x0, y0, x1, y1, S=24.0, want_holes=False):
    import cv2
    pad = 1
    w, h = int((x1 - x0) * S) + 2 * pad + 2, int((y1 - y0) * S) + 2 * pad + 2
    im = np.zeros((max(h, 3), max(w, 3)), np.uint8)
    for p in paths_:
        q = ((p["pts"] - [x0, y0]) * S + pad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(im, [q], p["closed"], 255, 3)
    if want_holes:
        return _norm_glyph(im), _holes(im)
    return _norm_glyph(im)


def read_glyph_word(paths_: list, min_sim: float = 0.45):
    """Texte numérique (« 6,54 ») dessiné en traits -> (chaîne, angle, similarité) ou None.
    On essaie 0°, 90° (lu de bas en haut) et 270° (de haut en bas)."""
    tpl = _templates()
    best = None
    for ang in (0, 90, 270):
        # coordonnées dans le repère du texte (x = sens de lecture, y vers le bas)
        tp = []
        for p in paths_:
            q = p["pts"]
            if ang == 0:
                r = q
            elif ang == 90:
                r = np.stack([-q[:, 1], q[:, 0]], axis=1)
            else:
                r = np.stack([q[:, 1], -q[:, 0]], axis=1)
            tp.append(dict(pts=r, closed=p["closed"]))
        # glyphes = tracés dont les intervalles en x se recouvrent
        iv = sorted(((float(t["pts"][:, 0].min()), float(t["pts"][:, 0].max()), t) for t in tp), key=lambda a: a[0])
        glyphs, cur = [], None
        for a, b, t in iv:
            if cur is not None and a <= cur[1] + 0.05:
                cur[1] = max(cur[1], b)
                cur[2].append(t)
            else:
                if cur is not None:
                    glyphs.append(cur)
                cur = [a, b, [t]]
        if cur is not None:
            glyphs.append(cur)
        allpts = np.vstack([t["pts"] for t in tp])
        ymin, ymax = float(allpts[:, 1].min()), float(allpts[:, 1].max())
        H_ = ymax - ymin
        if H_ <= 0 or not (2 <= len(glyphs) <= 12):
            continue
        s, sims, ok = "", [], True
        for a, b, ts in glyphs:
            ys = np.vstack([t["pts"] for t in ts])
            gh = float(ys[:, 1].max() - ys[:, 1].min())
            gw = b - a
            if gh < 0.3 * H_:                                  # petit signe en bas de ligne : virgule / point
                s += ","
                sims.append(1.0)
                continue
            if gw > 1.6 * gh or gh < 0.7 * H_:
                ok = False
                break
            v, nh = _glyph_vec(ts, a, float(ys[:, 1].min()), b, float(ys[:, 1].max()), want_holes=True)
            if v is None:
                ok = False
                break
            sc = [(float(max(t_ @ v for t_ in tl)), ch) for ch, tl in tpl.items() if nh in _HOLES[ch]]
            if not sc:
                ok = False
                break
            sc.sort(reverse=True)
            if sc[0][0] < min_sim:
                ok = False
                break
            s += sc[0][1]
            sims.append(sc[0][0])
        if not ok or not re.fullmatch(r"\d{1,3}(,\d{1,3})?", s):
            continue
        score = float(np.mean(sims))
        if best is None or score > best[2]:
            best = (s, ang, score)
    return best


def glyph_texts(paths: list, max_h_pt: float = 3.6, gap_pt: float = 2.6):
    """Petits tracés regroupés en mots numériques lisibles. Renvoie [dict(s, angle, x, y, bb, size, idx)]."""
    from scipy.spatial import cKDTree
    small = [i for i, p in enumerate(paths)
             if max(p["bbox"][2] - p["bbox"][0], p["bbox"][3] - p["bbox"][1]) <= max_h_pt]
    if len(small) < 2:
        return []
    c = np.array([[(paths[i]["bbox"][0] + paths[i]["bbox"][2]) / 2, (paths[i]["bbox"][1] + paths[i]["bbox"][3]) / 2] for i in small])
    tr = cKDTree(c)
    par = list(range(len(small)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, j in tr.query_pairs(gap_pt):
        par[f(i)] = f(j)
    groups: dict = {}
    for n in range(len(small)):
        groups.setdefault(f(n), []).append(small[n])
    out = []
    for g in groups.values():
        if len(g) < 2:
            continue
        r = read_glyph_word([paths[i] for i in g])
        if not r:
            continue
        b = np.array([paths[i]["bbox"] for i in g])
        bb = [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]
        s, ang, sc = r
        size = (bb[3] - bb[1]) if ang == 0 else (bb[2] - bb[0])
        # origine (début de la ligne de base) : coin bas-gauche pour 0°, bas-gauche pour 90°, haut-gauche pour 270°
        if ang == 0:
            x, y = bb[0], bb[3]
        elif ang == 90:
            x, y = bb[2], bb[3]
        else:
            x, y = bb[0], bb[1]
        out.append(dict(s=s, angle=float(ang), x=x, y=y, bb=bb, size=size / 0.72, idx=g, sim=sc,
                        color=(0.0, 0.0, 0.0)))
    return out


# ------------------------------------------------------------------ murs, épaisseurs et ouvertures par la géométrie exacte
def wall_network(item_pts: list, k: float, min_extent_m: float = 2.5, S: float = 2.0):
    """Quels tracés appartiennent au réseau des murs ? Les murs forment un réseau étendu ; un meuble (même épaisseur de
    trait) est un petit amas isolé. Renvoie un tableau booléen (même ordre que item_pts). k = mm réels par point."""
    import cv2
    if not item_pts:
        return np.zeros(0, bool)
    allp = np.vstack(item_pts)
    x0, y0 = allp.min(0)
    w, h = int((allp[:, 0].max() - x0) * S) + 4, int((allp[:, 1].max() - y0) * S) + 4
    m = np.zeros((h, w), np.uint8)
    for p in item_pts:
        cv2.polylines(m, [((p - [x0, y0]) * S + 1).astype(np.int32).reshape(-1, 1, 2)], False, 255, 3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ext_m = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) / S * k / 1000.0
    out = np.zeros(len(item_pts), bool)
    for i, p in enumerate(item_pts):
        q = p[len(p) // 2]
        px, py = int((q[0] - x0) * S + 1), int((q[1] - y0) * S + 1)
        l_ = lab[min(max(py, 0), h - 1), min(max(px, 0), w - 1)]
        out[i] = l_ > 0 and ext_m[l_] >= min_extent_m
    return out


def axis_segments(item_pts: list, tol_deg: float = 2.0, min_len: float = 1.0):
    """Segments horizontaux / verticaux des polylignes -> (H, V) = listes de (coord, a, b, idx_tracé)."""
    H, V = [], []
    for i, p in enumerate(item_pts):
        for a, b in zip(p[:-1], p[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            if L < min_len:
                continue
            ang = math.degrees(math.atan2(abs(dy), abs(dx)))
            if ang <= tol_deg:
                H.append((float((a[1] + b[1]) / 2), float(min(a[0], b[0])), float(max(a[0], b[0])), i))
            elif ang >= 90 - tol_deg:
                V.append((float((a[0] + b[0]) / 2), float(min(a[1], b[1])), float(max(a[1], b[1])), i))
    return H, V


def _merge_intervals(segs, eps=0.4):
    segs = sorted(segs, key=lambda s: s[0])
    out = []
    for a, b in segs:
        if out and a <= out[-1][1] + eps:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def wall_pairs(H: list, V: list, k: float, tmin_m: float = 0.06, tmax_m: float = 0.60, min_overlap: float = 0.5):
    """Deux traits parallèles proches = les deux faces d'un mur. Renvoie [(p0, p1, épaisseur_pt, used_idx_set)] en points."""
    out = []
    tmin, tmax = tmin_m * 1000 / k, tmax_m * 1000 / k
    for segs, orient in ((H, "h"), (V, "v")):
        if len(segs) < 2:
            continue
        arr = np.array([(s[0], s[1], s[2]) for s in segs])
        idx = [s[3] for s in segs]
        order = np.argsort(arr[:, 0])
        used = np.zeros(len(arr), bool)
        for ii in range(len(order)):
            i = order[ii]
            if used[i]:
                continue
            best = None
            for jj in range(ii + 1, len(order)):
                j = order[jj]
                dcoord = arr[j, 0] - arr[i, 0]
                if dcoord > tmax:
                    break
                if used[j] or dcoord < tmin:
                    continue
                lo, hi = max(arr[i, 1], arr[j, 1]), min(arr[i, 2], arr[j, 2])
                ov = hi - lo
                short = min(arr[i, 2] - arr[i, 1], arr[j, 2] - arr[j, 1])
                if ov >= min_overlap * short and ov > 0:
                    if best is None or ov > best[0]:
                        best = (ov, j, lo, hi, dcoord)
            if best:
                _, j, lo, hi, dc = best
                used[i] = used[j] = True
                c = (arr[i, 0] + arr[j, 0]) / 2
                p0, p1 = ((lo, c), (hi, c)) if orient == "h" else ((c, lo), (c, hi))
                out.append((np.array([p0, p1], float), float(dc), {idx[i], idx[j]}))
    return out


def find_gaps(H: list, V: list, k: float, wall_bbox, door_boxes: list, min_m: float = 0.55, max_m: float = 3.2):
    """Trous alignés dans les murs (ouvertures). Renvoie [dict(rect=(x0,y0,x1,y1), orient, kind, why, t)] en points.
    kind : porte (un arc de battant à proximité) | fenetre (trou sur la façade) | ouverture."""
    gmin, gmax = min_m * 1000 / k, max_m * 1000 / k
    t_def = 0.2 * 1000 / k
    raw = []                                            # (orient, coord, a, b)
    for segs, orient in ((H, "h"), (V, "v")):
        by: dict = {}
        for c, a, b, _ in segs:
            by.setdefault(round(c / 0.6), []).append((c, a, b))
        for lst in by.values():
            c = float(np.mean([s[0] for s in lst]))
            mi = _merge_intervals([(s[1], s[2]) for s in lst])
            for (a0, b0), (a1, b1) in zip(mi[:-1], mi[1:]):
                g = a1 - b0
                if gmin <= g <= gmax:
                    raw.append((orient, c, b0, a1))
    def crossed(o, c, a, b):
        other = V if o == "h" else H
        for cc, aa, bb, _ in other:
            if a + 0.5 < cc < b - 0.5 and aa - 0.3 <= c <= bb + 0.3:
                return True
        return False
    raw = [r for r in raw if not crossed(*r)]
    used = [False] * len(raw)
    out = []
    X0, Y0, X1, Y1 = wall_bbox
    for i, (o, c, a, b) in enumerate(raw):
        if used[i]:
            continue
        used[i] = True
        c2, t = c, t_def
        for j in range(i + 1, len(raw)):
            o2, cj, aj, bj = raw[j]
            if used[j] or o2 != o or abs(cj - c) > 0.6 * 1000 / k or abs(cj - c) < 0.04 * 1000 / k:
                continue
            ov = min(b, bj) - max(a, aj)
            if ov >= 0.6 * min(b - a, bj - aj):
                used[j] = True
                a, b = min(a, aj), max(b, bj)
                c2, t = (c + cj) / 2, abs(cj - c)
                break
        rect = (a, c2 - t / 2, b, c2 + t / 2) if o == "h" else (c2 - t / 2, a, c2 + t / 2, b)
        kind, why = "ouverture", "trou dans un mur sans symbole"
        m = 0.5 * 1000 / k
        for bx in door_boxes:
            if bx[0] < rect[2] + m and bx[2] > rect[0] - m and bx[1] < rect[3] + m and bx[3] > rect[1] - m:
                kind, why = "porte", "trou dans un mur avec arc de battant"
                break
        else:
            cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
            edge = min(abs(cx - X0), abs(cx - X1), abs(cy - Y0), abs(cy - Y1))
            if edge <= 0.9 * 1000 / k:
                kind, why = "fenetre", "trou dans la façade sans symbole (à vérifier : porte d'entrée possible)"
        out.append(dict(rect=rect, orient=o, kind=kind, why=why, t=t))
    return out
