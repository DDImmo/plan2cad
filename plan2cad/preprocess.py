"""Nettoyage automatique du scan : remplace l'étape manuelle GIMP.

chargement (png/tif/jpg/pdf, page, recadrage, couleur) -> aplatissement du fond
-> (redressement) -> binarisation -> suppression des poussières / textes
-> (couleur) classification des teintes pour séparer les lots.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import safety

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(int(safety.MAX_PIXELS)))
import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = int(safety.MAX_PIXELS)        # PIL refuse au-delà de 2× (anti bombe de décompression)

RASTER_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}


# ------------------------------------------------------------------ PDF
def pdf_pages(path: str) -> int:
    n = _pdf_pages(path)
    if n > safety.MAX_PDF_PAGES:
        raise safety.UnsafeInput(f"PDF de {n} pages (> {safety.MAX_PDF_PAGES}) : refusé (PLAN2CAD_MAX_PDF_PAGES)")
    return n


def _pdf_pages(path: str) -> int:
    try:
        return len(_fitz().open(path))
    except ImportError:
        pass
    if shutil.which("pdfinfo"):
        out = subprocess.run(["pdfinfo", path], capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=60).stdout
        m = re.search(r"Pages:\s*(\d+)", out)
        if m:
            return int(m.group(1))
    return 1


def _pdf_page_size(path: str, page: int):
    """(largeur, hauteur) en points."""
    try:
        r = _fitz().open(path)[page - 1].rect
        return r.width, r.height
    except ImportError:
        pass
    if shutil.which("pdfinfo"):
        out = subprocess.run(["pdfinfo", "-f", str(page), "-l", str(page), path],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60).stdout
        m = re.search(r"Page\s+%d size:\s*([\d.]+)\s*x\s*([\d.]+)\s*pts" % page, out) or \
            re.search(r"Page size:\s*([\d.]+)\s*x\s*([\d.]+)\s*pts", out)
        if m:
            return float(m.group(1)), float(m.group(2))
    raise RuntimeError("Lecture PDF : installer poppler-utils (pdftoppm/pdfinfo) ou PyMuPDF (pip install pymupdf).")


def _fitz():
    """PyMuPDF : module `pymupdf` (récent) ou `fitz` (ancien nom)."""
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz
        return fitz


def _render_pdf(path: str, page: int, dpi: float, max_px: int, crop):
    """Rend une page (ou une zone) de PDF en RGB. Renvoie (rgb, dpi_effectif, taille_nominale_px)."""
    w_pt, h_pt = _pdf_page_size(path, page)
    x0f, y0f, x1f, y1f = crop or (0.0, 0.0, 1.0, 1.0)
    cw_in = (x1f - x0f) * w_pt / 72.0
    ch_in = (y1f - y0f) * h_pt / 72.0
    d = min(dpi, max_px / max(cw_in, ch_in, 1e-6))
    nominal = (w_pt / 72.0 * dpi, h_pt / 72.0 * dpi)
    try:
        fitz = _fitz()
        d = max(1, int(round(d)))
        pg = fitz.open(path)[page - 1]
        clip = fitz.Rect(x0f * w_pt, y0f * h_pt, x1f * w_pt, y1f * h_pt)
        pix = pg.get_pixmap(dpi=d, clip=clip, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
        return arr, d, nominal
    except ImportError:
        pass
    if not shutil.which("pdftoppm"):
        raise RuntimeError("Lecture PDF : installer poppler-utils ou PyMuPDF.")
    sx, sy = d / 72.0 * w_pt, d / 72.0 * h_pt          # taille de la page rendue en px
    args = ["pdftoppm", "-r", f"{d:.3f}", "-png", "-f", str(page), "-l", str(page)]
    if crop:
        args += ["-x", str(int(x0f * sx)), "-y", str(int(y0f * sy)),
                 "-W", str(max(1, int((x1f - x0f) * sx))), "-H", str(max(1, int((y1f - y0f) * sy)))]
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(args + [path, os.path.join(td, "p")], check=True, timeout=600)
        files = sorted(glob.glob(os.path.join(td, "p*.png")))
        if not files:
            raise RuntimeError("pdftoppm n'a produit aucune image")
        arr = np.array(Image.open(files[0]).convert("RGB"))
    return arr, d, nominal


# ------------------------------------------------------------------ chargement
def is_colorful(rgb: np.ndarray, chroma_thr: int = 60, min_frac: float = 0.0008, min_px: int = 400) -> bool:
    """Vrai si l'image contient des traits réellement colorés (pas un papier jauni)."""
    h, w = rgb.shape[:2]
    f = min(1.0, 1500 / max(h, w))
    s = rgb if f >= 1 else cv2.resize(rgb, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    mx, mn = s.max(2).astype(np.int16), s.min(2).astype(np.int16)
    n = int(((mx - mn >= chroma_thr) & (mx > 40)).sum())
    return n >= max(min_px * f * f, min_frac * s.shape[0] * s.shape[1])


def load_image(path: str, dpi: float = 400, max_px: int = 7000, page: int = 1, crop=None):
    """Charge une image / une page de PDF / une zone (crop en fractions 0-1 de la page).

    Renvoie (img, info) ; img = (H,W) niveaux de gris ou (H,W,3) RGB si l'image est colorée.
    info : mm_per_px (mm papier par pixel chargé), dpi effectif, scale (px chargés / px nominaux),
           nominal_size, crop_origin (px nominaux), is_color.
    """
    max_px = safety.clamp_side(max_px)
    ext = Path(path).suffix.lower()
    if ext != ".pdf":
        safety.check_image_header(path)                  # refuse une bombe de décompression avant tout décodage
    if ext == ".pdf":
        rgb, eff_dpi, nominal = _render_pdf(path, page, dpi, max_px, crop)
        scale = eff_dpi / dpi
        cx0 = (crop[0] if crop else 0.0) * nominal[0]
        cy0 = (crop[1] if crop else 0.0) * nominal[1]
    else:
        im = Image.open(path)
        if page > 1 and getattr(im, "n_frames", 1) >= page:
            im.seek(page - 1)
        nominal = im.size
        cx0 = cy0 = 0.0
        if crop:
            x0, y0 = int(crop[0] * nominal[0]), int(crop[1] * nominal[1])
            x1, y1 = max(x0 + 1, int(crop[2] * nominal[0])), max(y0 + 1, int(crop[3] * nominal[1]))
            im = im.crop((x0, y0, x1, y1))
            cx0, cy0 = x0, y0
        gray_mode = im.mode in ("1", "L", "LA", "I;16", "I", "F")
        rgb = np.array(im.convert("L" if gray_mode else "RGB"))
        h, w = rgb.shape[:2]
        scale = 1.0
        m = max(h, w)
        if m > max_px:
            f = max_px / m
            k = max(2, int(round(1 / f)))
            # min-pooling avant réduction : conserve les traits fins d'1 pixel
            rgb = cv2.erode(rgb, np.ones((k, k), np.uint8))
            rgb = cv2.resize(rgb, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
            scale = f
        eff_dpi = dpi * scale
    color = rgb.ndim == 3 and is_colorful(rgb)
    if rgb.ndim == 3 and not color:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    info = dict(mm_per_px=25.4 / eff_dpi, dpi=eff_dpi, scale=scale, nominal_size=tuple(nominal),
                crop_origin=(cx0, cy0), is_color=bool(color))
    return rgb, info


# ------------------------------------------------------------------ nettoyage
def _flatten_channel(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    ds = max(1, max(h, w) // 1200)
    small = cv2.resize(gray, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
    k = 2 * max(3, int(60 / ds) // 2) + 1          # plus large que n'importe quel trait
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), max(2.0, k / 4))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    bg = np.maximum(bg, 1)
    return cv2.divide(gray, bg, scale=255)


def flatten(img: np.ndarray) -> np.ndarray:
    """Supprime ombres, pliures, fond jauni : divise par le fond estimé (par canal si couleur)."""
    if img.ndim == 2:
        return _flatten_channel(img)
    return np.dstack([_flatten_channel(np.ascontiguousarray(img[..., c])) for c in range(3)])


def darkness(flat: np.ndarray) -> np.ndarray:
    """Image « encre = sombre » : le minimum des canaux (le jaune, le rouge, le bleu... deviennent sombres)."""
    return flat if flat.ndim == 2 else flat.min(axis=2)


def estimate_skew(norm: np.ndarray, max_deg: float = 3.0) -> float:
    """Angle (degrés) à appliquer pour redresser, par maximisation de la variance des projections."""
    h, w = norm.shape
    f = 1200 / max(h, w)
    small = cv2.resize(norm, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
    ink = (small < 128).astype(np.uint8)
    hh, ww = ink.shape
    c = (ww / 2, hh / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(ink, M, (ww, hh), flags=cv2.INTER_NEAREST)
        return float(r.sum(1, dtype=np.float64).var() + r.sum(0, dtype=np.float64).var())

    best = max(np.arange(-max_deg, max_deg + 1e-9, 0.25), key=score)
    best = max(np.arange(best - 0.25, best + 0.2501, 0.02), key=score)
    return float(best)


def rotate_image(img: np.ndarray, angle: float, fill: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(fill,) * 3 if img.ndim == 3 else fill)


def binarize(norm: np.ndarray, threshold: int | None = None):
    """Noir & blanc pur. Renvoie (ink uint8 0/255 avec encre=255, seuil)."""
    if threshold is None:
        t, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(min(max(t, 90), 215))
    else:
        t = float(threshold)
    ink = (norm < t).astype(np.uint8) * 255
    return ink, t


def clean_ink(ink: np.ndarray, mm_per_px: float, despeckle_mm: float = 0.4,
              remove_text: bool = False, text_mm: float = 4.0, close_px: int = 0):
    """Retire poussières (et optionnellement les textes) ; renvoie (ink, stats)."""
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    maxdim = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]).astype(np.float64)
    thr_dust = despeckle_mm / mm_per_px
    remove = maxdim < thr_dust
    n_dust = int(remove[1:].sum())
    n_text = 0
    if remove_text:
        t = text_mm / mm_per_px
        txt = (maxdim < t) & ~remove
        n_text = int(txt[1:].sum())
        remove |= txt
    remove[0] = False
    if remove.any():
        ink = np.where(remove[labels], 0, ink).astype(np.uint8)
    return ink, dict(components=int(n - 1), dust_removed=n_dust, text_removed=n_text)


# ------------------------------------------------------------------ couleurs
COLOR_NAMES = ["noir", "rouge", "orange", "jaune", "vert", "cyan", "bleu", "magenta", "marron"]
# (nom, teinte min, teinte max) sur l'échelle OpenCV 0-179
_HUES = [("rouge", 0, 8), ("orange", 8, 20), ("jaune", 20, 36), ("vert", 36, 86), ("cyan", 86, 101),
         ("bleu", 101, 131), ("magenta", 131, 170), ("rouge", 170, 181)]


def classify_ink(flat_rgb: np.ndarray, ink: np.ndarray, chroma_thr: int = 60,
                 min_frac: float = 0.004, min_px: int = 250):
    """Répartit les pixels d'encre par classe de couleur.
    Renvoie {nom: masque uint8 0/255}, uniquement les classes non vides. Les classes colorées
    trop petites (artefacts JPEG) sont rattachées au noir."""
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return {"noir": ink}
    px = flat_rgb[ys, xs]
    mx, mn = px.max(1).astype(np.int16), px.min(1).astype(np.int16)
    chroma = mx - mn
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0].astype(np.int16)
    cls = np.zeros(len(ys), np.int8)
    colored = chroma >= chroma_thr
    for name, lo, hi in _HUES:
        sel = colored & (hue >= lo) & (hue < hi)
        cls[sel] = COLOR_NAMES.index(name)
    brown = colored & (mx < 150) & np.isin(cls, [COLOR_NAMES.index(n) for n in ("rouge", "orange", "jaune")])
    cls[brown] = COLOR_NAMES.index("marron")

    lab = np.zeros(ink.shape, np.int8)
    lab[ys, xs] = cls
    keep = []
    for i in range(1, len(COLOR_NAMES)):
        n = int((cls == i).sum())
        if n >= max(min_px, min_frac * len(ys)):
            keep.append(i)
        else:
            lab[lab == i] = 0
    k5 = np.ones((5, 5), np.uint8)
    for i in keep:                       # halo anti-aliasing : le noir contigu à une couleur lui revient
        dil = cv2.dilate((lab == i).astype(np.uint8), k5) > 0
        lab[(lab == 0) & (ink > 0) & dil] = i
    out = {}
    for i in [0] + keep:
        m = ((lab == i) & (ink > 0)).astype(np.uint8) * 255
        if m.any():
            out[COLOR_NAMES[i]] = m
    return out


def class_preview(flat_rgb_shape, masks: dict, palette=None):
    """Aperçu : chaque classe dessinée dans sa teinte (pour l'interface web), BGR."""
    pal = palette or {"noir": (40, 40, 40), "rouge": (50, 50, 230), "orange": (0, 140, 255), "jaune": (0, 200, 220),
                      "vert": (60, 170, 40), "cyan": (210, 190, 0), "bleu": (230, 90, 30),
                      "magenta": (200, 40, 200), "marron": (30, 70, 120)}
    h, w = next(iter(masks.values())).shape
    img = np.full((h, w, 3), 255, np.uint8)
    for n, m in masks.items():
        img[m > 0] = pal.get(n, (0, 0, 0))
    return img
