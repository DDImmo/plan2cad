"""Génère des scans de démonstration réalistes (bruit, ombres, rotation, décalage, échelle)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 2400, 1700
OUTER = (200, 200, 2200, 1500)       # rectangle (axe des murs extérieurs)


def _walls(img, col, thick_out, thick_in):
    x0, y0, x1, y1 = OUTER
    cv2.rectangle(img, (x0, y0), (x1, y1), col, thick_out)
    # refend vertical avec porte
    cv2.line(img, (1000, y0), (1000, 650), col, thick_in)
    cv2.line(img, (1000, 780), (1000, y1), col, thick_in)
    # séparation horizontale avec porte
    cv2.line(img, (1000, 900), (1450, 900), col, thick_in)
    cv2.line(img, (1580, 900), (x1, 900), col, thick_in)
    cv2.line(img, (1600, 900), (1600, y1), col, thick_in)
    # cage d'ascenseur
    cv2.rectangle(img, (280, 280), (500, 500), col, max(thick_in, 30))


def _scan_effects(img, rng, angle, scale, tx, ty):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=235)
    # fond jauni + vignettage + pliure
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    shade = 1.0 - 0.18 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 2
    shade -= 0.10 * np.exp(-((xx - w * 0.62) ** 2) / (2 * 14 ** 2))
    img = img.astype(np.float32) * shade
    img += rng.normal(0, 7, img.shape)
    # poussières
    for _ in range(120):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(1, 4)), float(rng.integers(30, 120)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8), M


def make_demo(outdir: str, seed: int = 7) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # --- architecture : murs pleins (poché), textes
    arc = np.full((H, W), 255, np.uint8)
    _walls(arc, 20, 47, 16)
    for txt, pos in (("SALON", (1250, 450)), ("CUISINE", (1700, 1150)), ("BUREAU", (450, 900)), ("ASC.", (330, 400))):
        cv2.putText(arc, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 40, 3, cv2.LINE_AA)
    arc, M_arc = _scan_effects(arc, rng, 0.0, 1.0, 0, 0)

    # --- électricité : fond mural fin + réseau + symboles
    ele = np.full((H, W), 255, np.uint8)
    _walls(ele, 150, 4, 3)
    for pts in ([(300, 600), (300, 1300), (900, 1300)], [(300, 1300), (300, 1420), (1900, 1420)],
                [(1150, 260), (1150, 800), (2100, 800)], [(1800, 300), (2100, 300), (2100, 800)]):
        cv2.polylines(ele, [np.array(pts, np.int32)], False, 40, 5, cv2.LINE_AA)
    for c in ((300, 600), (900, 1300), (1900, 1420), (1150, 260), (2100, 800), (1800, 300)):
        cv2.circle(ele, c, 22, 30, 4, cv2.LINE_AA)
        cv2.line(ele, (c[0] - 22, c[1]), (c[0] + 22, c[1]), 30, 4)
    cv2.rectangle(ele, (220, 560), (260, 650), 30, 5)       # tableau
    ang_e, sc_e, tx_e, ty_e = 1.2, 1.0, 35, -28
    ele, M_ele = _scan_effects(ele, rng, ang_e, sc_e, tx_e, ty_e)

    # --- plomberie : fond mural fin + réseau eau
    plu = np.full((H, W), 255, np.uint8)
    _walls(plu, 150, 4, 3)
    for pts in ([(1050, 500), (1050, 1350), (1550, 1350)], [(1550, 1350), (1550, 950), (2100, 950)],
                [(1150, 1000), (1550, 1000)]):
        cv2.polylines(plu, [np.array(pts, np.int32)], False, 60, 6, cv2.LINE_AA)
    for c in ((1050, 500), (2100, 950), (1150, 1000)):
        cv2.circle(plu, c, 18, 50, -1)
    ang_p, sc_p, tx_p, ty_p = -0.8, 0.985, -50, 40
    plu, M_plu = _scan_effects(plu, rng, ang_p, sc_p, tx_p, ty_p)

    paths = {}
    for k, im in (("arc", arc), ("elec", ele), ("plumb", plu)):
        p = out / f"demo_{k}.png"
        cv2.imwrite(str(p), im)
        paths[k] = str(p)
    truth = dict(elec=dict(angle=ang_e, scale=sc_e), plumb=dict(angle=ang_p, scale=sc_p))
    (out / "truth.json").write_text(json.dumps(truth))
    return dict(paths=paths, truth=truth, size=(W, H))


def make_double_line_arch(outdir: str, seed: int = 11) -> str:
    """Plan d'architecture dessiné en DOUBLE TRAIT (deux traits fins par mur), avec cotes et textes."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 255, np.uint8)
    x0, y0, x1, y1 = OUTER
    t_out, t_in, lw = 47, 16, 4

    def band_rect(a, b, c, d, th):
        h = th // 2
        cv2.rectangle(img, (a - h, b - h), (c + h, d + h), 20, lw)
        cv2.rectangle(img, (a + h, b + h), (c - h, d - h), 20, lw)

    def band_line(p, q, th):
        h = th // 2
        if p[0] == q[0]:
            cv2.line(img, (p[0] - h, p[1]), (q[0] - h, q[1]), 20, lw); cv2.line(img, (p[0] + h, p[1]), (q[0] + h, q[1]), 20, lw)
        else:
            cv2.line(img, (p[0], p[1] - h), (q[0], q[1] - h), 20, lw); cv2.line(img, (p[0], p[1] + h), (q[0], q[1] + h), 20, lw)

    band_rect(x0, y0, x1, y1, t_out)
    band_line((1000, y0 + 24), (1000, 650), t_in); band_line((1000, 780), (1000, y1 - 24), t_in)
    band_line((1008, 900), (1450, 900), t_in); band_line((1580, 900), (x1 - 24, 900), t_in)
    band_line((1600, 908), (1600, y1 - 24), t_in)
    band_rect(280, 280, 500, 500, 30)
    for txt, pos in (("SALON", (1250, 450)), ("CUISINE", (1700, 1150)), ("BUREAU", (450, 900))):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 40, 3, cv2.LINE_AA)
    cv2.line(img, (200, 1590), (2200, 1590), 30, 3)            # une cote
    img, _ = _scan_effects(img, rng, 0.0, 1.0, 0, 0)
    p = out / "demo_arc_double.png"
    cv2.imwrite(str(p), img)
    return str(p)


def make_color_plan(outdir: str, seed: int = 5, angle=0.9, scale=1.012, tx=-40, ty=25, jpeg=True) -> dict:
    """Plan EN COULEUR qui mêle plusieurs lots : fond mural noir, élec rouge, eau bleue, gaz jaune,
    évacuations magenta. Déformé (rotation/échelle/décalage) par rapport au plan d'architecture."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 255, np.uint8)
    # fond mural fin, noir
    _walls(img, (30, 30, 30), 4, 3)
    RED, BLUE, YEL, MAG = (30, 30, 215), (200, 90, 20), (0, 200, 235), (190, 40, 190)       # BGR
    cv2.polylines(img, [np.array([(300, 600), (300, 1300), (900, 1300)], np.int32)], False, RED, 5, cv2.LINE_AA)
    cv2.polylines(img, [np.array([(1150, 260), (1150, 800), (2100, 800)], np.int32)], False, RED, 5, cv2.LINE_AA)
    cv2.polylines(img, [np.array([(1050, 500), (1050, 1350), (1550, 1350)], np.int32)], False, BLUE, 6, cv2.LINE_AA)
    cv2.polylines(img, [np.array([(1700, 1000), (1700, 1400), (2100, 1400)], np.int32)], False, YEL, 7, cv2.LINE_AA)
    cv2.polylines(img, [np.array([(700, 1450), (1900, 1450)], np.int32)], False, MAG, 6, cv2.LINE_AA)
    for c in ((300, 600), (2100, 800)):
        cv2.circle(img, c, 20, RED, 4, cv2.LINE_AA)
    # même déformation que pour un scan réel
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(238, 238, 238))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    shade = 1.0 - 0.15 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 2
    img = img.astype(np.float32) * shade[..., None] + rng.normal(0, 6, img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (0, 0), 0.8)
    p = out / ("demo_couleur.jpg" if jpeg else "demo_couleur.png")
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 72] if jpeg else [])
    return dict(path=str(p), angle=angle, scale=scale)


# ------------------------------------------------------------------ jeu de test hétérogène (dossier réel simulé)
def _plan_B(col, t_out, t_in, level, lot=None):
    """Bâtiment B : emprise en L ; cloisons différentes au RDC et à l'étage. Canvas 1900x1200 (coordonnées -150)."""
    img = np.full((1200, 1900), 255, np.uint8) if np.isscalar(col) else np.full((1200, 1900, 3), 255, np.uint8)
    off = lambda p: (p[0] - 150, p[1] - 150)
    pts = np.array([off(p) for p in [(200, 200), (1500, 200), (1500, 700), (2000, 700), (2000, 1300), (200, 1300)]], np.int32)
    cv2.polylines(img, [pts], True, col, t_out)
    segs = ([((900, 200), (900, 650)), ((900, 780), (900, 1300)), ((900, 900), (1450, 900)), ((1580, 900), (2000, 900))]
            if level == 0 else
            [((600, 200), (600, 1300)), ((1200, 200), (1200, 560)), ((1200, 680), (1200, 700)), ((600, 800), (1000, 800)), ((1300, 1000), (2000, 1000))])
    for a, b in segs:
        cv2.line(img, off(a), off(b), col, t_in)
    return img


def _plan_C(col, t_out, t_in):
    img = np.full((1100, 1700), 255, np.uint8)
    cv2.rectangle(img, (60, 60), (1560, 960), col, t_out)
    cv2.line(img, (810, 60), (810, 960), col, t_in)
    cv2.line(img, (60, 500), (810, 500), col, t_in)
    return img


def _deform(img, angle, scale, tx, ty, fill=255):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=fill if img.ndim == 2 else (fill,) * 3)


def make_demo_site(outdir: str, seed: int = 21) -> dict:
    """Dossier hétérogène :
      batA/archi_RDC.png       architecture A (murs pleins)         -> nom complet (bâtiment, niveau, lot)
      batA/reseaux_RDC.jpg     plan COULEUR mêlant 4 lots           -> bâtiment/niveau dans le nom, lots par couleur
      batC/plan_inconnu.png    plan C en N&B, aucun indice de lot   -> lot déduit du dessin
      planche_001.png          UNE planche : 2 plans d'architecture B (RDC, étage) + élec + plomberie en N&B,
                               cadre + cartouche, AUCUN indice dans le nom
    """
    out = Path(outdir)
    (out / "batA").mkdir(parents=True, exist_ok=True)
    (out / "batC").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = make_demo(out / "_tmp", seed)
    cv2.imwrite(str(out / "batA" / "archi_RDC.png"), cv2.imread(base["paths"]["arc"], 0))
    col = make_color_plan(out / "_tmp2", seed=seed + 1)
    Image.fromarray(cv2.cvtColor(cv2.imread(col["path"]), cv2.COLOR_BGR2RGB)).save(out / "batA" / "reseaux_RDC.jpg", quality=80)
    import shutil
    shutil.rmtree(out / "_tmp"); shutil.rmtree(out / "_tmp2")

    # plan C : N&B, murs pleins, sans indice
    c, _ = _scan_effects(_plan_C(20, 40, 14), rng, 0.0, 1.0, 0, 0)
    cv2.imwrite(str(out / "batC" / "plan_inconnu.png"), c)

    # planche : 4 plans + cadre + cartouche
    SW, SH = 4600, 3600
    sheet = np.full((SH, SW), 255, np.uint8)
    cv2.rectangle(sheet, (40, 40), (SW - 40, SH - 40), 30, 8)

    def lot_plan(level, network, angle, scale, tx, ty):
        im = _plan_B(150, 4, 3, level)
        for pts, th in network:
            cv2.polylines(im, [np.array([(x - 150, y - 150) for x, y in pts], np.int32)], False, 40, th, cv2.LINE_AA)
        for cx, cy in [n[0][0] for n in network]:
            cv2.circle(im, (cx - 150, cy - 150), 20, 30, 4, cv2.LINE_AA)
        return _deform(im, angle, scale, tx, ty)

    elec = lot_plan(0, [([(300, 400), (300, 1200), (800, 1200)], 5), ([(1000, 300), (1000, 750), (1400, 750)], 5)], 1.0, 1.0, 25, -20)
    plumb = lot_plan(0, [([(1100, 400), (1100, 1200), (1700, 1200)], 6), ([(400, 500), (400, 1100)], 6)], -0.7, 1.01, -30, 20)
    arcs = [_plan_B(20, 47, 16, 0), _plan_B(20, 47, 16, 1)]
    cells = [(200, 150), (2500, 150), (200, 1700), (2500, 1700)]        # 2 archi en haut, élec + plomberie en bas
    for (x, y), im in zip(cells, [arcs[0], arcs[1], elec, plumb]):
        h, w = im.shape
        sheet[y:y + h, x:x + w] = np.minimum(sheet[y:y + h, x:x + w], im)
    for (x, y), t in zip(cells, ["PLAN NIVEAU 0", "PLAN NIVEAU 1", "ELECTRICITE", "PLOMBERIE"]):
        cv2.putText(sheet, t, (x + 600, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 30, 3, cv2.LINE_AA)
    # cartouche (tableau) en bas à droite
    x0, y0, x1, y1 = 2500, 3250, 4540, 3560
    cv2.rectangle(sheet, (x0, y0), (x1, y1), 30, 4)
    for i in range(1, 3):
        cv2.line(sheet, (x0, y0 + i * (y1 - y0) // 3), (x1, y0 + i * (y1 - y0) // 3), 30, 3)
    for j in range(1, 6):
        cv2.line(sheet, (x0 + j * (x1 - x0) // 6, y0), (x0 + j * (x1 - x0) // 6, y1), 30, 3)
    for r in range(3):
        for k in range(6):
            cv2.putText(sheet, f"R{r}C{k}", (x0 + k * (x1 - x0) // 6 + 40, y0 + r * (y1 - y0) // 3 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, 40, 3, cv2.LINE_AA)
    sheet, _ = _scan_effects(sheet, rng, 0.0, 1.0, 0, 0)
    cv2.imwrite(str(out / "planche_001.png"), sheet)
    return dict(root=str(out))


def make_demo_openings(outdir: str, seed: int = 5) -> dict:
    """Plan d'architecture (murs pleins, 1:100, 400 dpi) avec portes (battant + arc), fenêtres, baie, escalier et noms de pièces."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    W, H, t = 2400, 1700, 39
    im = np.full((H, W), 255, np.uint8)
    wall = lambda x0, y0, x1, y1: cv2.rectangle(im, (x0, y0), (x1, y1), 0, -1)
    # murs extérieurs avec ouvertures
    wall(200, 200, 600, 200 + t); wall(790, 200, 2200, 200 + t)                       # haut : fenêtre 600-790
    wall(200, 1500 - t, 1700, 1500); wall(1842, 1500 - t, 2200, 1500)                # bas : porte 1700-1842
    wall(200, 200, 200 + t, 700); wall(200, 842, 200 + t, 1500)                      # gauche : porte 700-842
    wall(2200 - t, 200, 2200, 400); wall(2200 - t, 600, 2200, 1500)                  # droite : fenêtre 400-600
    # cloisons
    wall(1100, 200, 1100 + t, 900); wall(1100, 1042, 1100 + t, 1500)                 # porte 900-1042
    wall(1100 + t, 800, 1500, 800 + t); wall(1700, 800, 2200 - t, 800 + t)           # baie 1500-1700
    g = 142
    # fenêtres : deux traits fins + traits d'extrémité
    for yy in (206, 233):
        cv2.line(im, (600, yy), (790, yy), 0, 3)
    for yy in (406, 433):
        pass
    for xx in (2206 - t, 2200 - t + 30):
        cv2.line(im, (xx, 400), (xx, 600), 0, 3)
    # portes : battant ouvert + arc
    cv2.line(im, (200 + t, 700), (200 + t + g, 700), 0, 4)
    cv2.ellipse(im, (200 + t, 700), (g, g), 0, 0, 90, 0, 3, cv2.LINE_AA)
    cv2.line(im, (1100, 900), (1100 - g, 900), 0, 4)
    cv2.ellipse(im, (1100, 900), (g, g), 0, 90, 180, 0, 3, cv2.LINE_AA)
    cv2.line(im, (1700, 1500 - t), (1700, 1500 - t - g), 0, 4)
    cv2.ellipse(im, (1700, 1500 - t), (g, g), 0, 270, 360, 0, 3, cv2.LINE_AA)
    # escalier : 8 marches de 28 cm, volée 0,95 m
    x0, x1, y0 = 320, 470, 1000
    for i in range(9):
        cv2.line(im, (x0, y0 + i * 44), (x1, y0 + i * 44), 0, 4)
    cv2.line(im, (x0, y0), (x0, y0 + 8 * 44), 0, 4); cv2.line(im, (x1, y0), (x1, y0 + 8 * 44), 0, 4)
    # textes
    for txt, pos in (("SALON", (1300, 450)), ("CUISINE", (1450, 1250)), ("CHAMBRE", (400, 500)),
                     ("PLAN RDC", (900, 130)), ("ESCALIER", (300, 960))):
        cv2.putText(im, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.9, 0, 4, cv2.LINE_AA)
    im, _ = _scan_effects(im, rng, 0.0, 1.0, 0, 0)
    p = out / "plan_ouvertures.png"
    cv2.imwrite(str(p), im)
    return dict(path=str(p))


def make_demo_double(outdir: str, seed: int = 9) -> dict:
    """Plan en DOUBLE TRAIT (1:100, 400 dpi) : porte à arc, porte sans arc (battant seul), fenêtres qui n'interrompent pas le mur
    (traits intérieurs), escalier en éventail, trois cotes (dont une verticale) et noms de pièces."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    W, H, t = 2400, 1700, 39
    mask = np.zeros((H, W), np.uint8)
    wall = lambda x0, y0, x1, y1: cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    wall(200, 200, 2200, 200 + t)                                                     # haut (fenêtre 600-790 : mur continu)
    wall(200, 1500 - t, 1700, 1500); wall(1842, 1500 - t, 2200, 1500)                # bas : porte 1700-1842
    wall(200, 200, 200 + t, 700); wall(200, 842, 200 + t, 1500)                      # gauche : porte 700-842 (battant seul)
    wall(2200 - t, 200, 2200, 1500)                                                  # droite (fenêtre 400-600 : mur continu)
    wall(1100, 200, 1100 + t, 900); wall(1100, 1042, 1100 + t, 1500)                 # cloison : porte 900-1042
    cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    im = np.full((H, W), 255, np.uint8)
    cv2.drawContours(im, cs, -1, 0, 3)
    g = 142
    for yy in (212, 227):                                                            # fenêtre haut : 2 traits dans l'épaisseur
        cv2.line(im, (600, yy), (790, yy), 0, 2)
    for xx in (2171, 2190):                                                          # fenêtre droite
        cv2.line(im, (xx, 400), (xx, 600), 0, 2)
    cv2.line(im, (200 + t, 842), (200 + t + g, 842), 0, 3)                           # porte gauche : battant seul
    cv2.line(im, (1100, 900), (1100 - g, 900), 0, 3)                                 # porte cloison : battant + arc
    cv2.ellipse(im, (1100, 900), (g, g), 0, 90, 180, 0, 2, cv2.LINE_AA)
    cv2.line(im, (1700, 1500 - t), (1700, 1500 - t - g), 0, 3)                       # porte bas : battant + arc
    cv2.ellipse(im, (1700, 1500 - t), (g, g), 0, 270, 360, 0, 2, cv2.LINE_AA)
    # escalier en éventail : 7 marches rayonnantes autour de (450, 1400)
    cx, cy = 450, 1400
    for a in range(270, 361, 15):
        ca, sa = np.cos(np.radians(a)), np.sin(np.radians(a))
        cv2.line(im, (int(cx + 60 * ca), int(cy + 60 * sa)), (int(cx + 170 * ca), int(cy + 170 * sa)), 0, 3)
    cv2.ellipse(im, (cx, cy), (170, 170), 0, 270, 360, 0, 3, cv2.LINE_AA)
    cv2.ellipse(im, (cx, cy), (60, 60), 0, 270, 360, 0, 3, cv2.LINE_AA)
    # cotes
    cv2.line(im, (200, 130), (2200, 130), 0, 2); cv2.line(im, (200, 112), (200, 148), 0, 2); cv2.line(im, (2200, 112), (2200, 148), 0, 2)
    cv2.putText(im, "12.70", (1100, 112), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    cv2.line(im, (120, 200), (120, 1500), 0, 2); cv2.line(im, (102, 200), (138, 200), 0, 2); cv2.line(im, (102, 1500), (138, 1500), 0, 2)
    tt = np.full((70, 260), 255, np.uint8)
    cv2.putText(tt, "8.26", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    tt = cv2.rotate(tt, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im[760:1020, 40:110] = np.minimum(im[760:1020, 40:110], tt)
    cv2.line(im, (200, 1580), (1100, 1580), 0, 2); cv2.line(im, (200, 1562), (200, 1598), 0, 2); cv2.line(im, (1100, 1562), (1100, 1598), 0, 2)
    cv2.putText(im, "5.72", (600, 1566), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    for txt, pos in (("SALON", (1300, 450)), ("CUISINE", (1450, 1250)), ("CHAMBRE", (400, 600)), ("PLAN RDC", (900, 60))):
        cv2.putText(im, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3, cv2.LINE_AA)
    im, _ = _scan_effects(im, rng, 0.0, 1.0, 0, 0)
    p = out / "plan_double.png"
    cv2.imwrite(str(p), im)
    return dict(path=str(p))
