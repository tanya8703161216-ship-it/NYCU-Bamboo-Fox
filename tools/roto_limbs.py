#!/usr/bin/env python3
"""Mask the fox's paws/feet per frame and composite a recoloured layer over them.

Usage:
  roto_limbs.py <keys.json> <base_png_dir> <layer_png_dir> <out_png_dir> [--debug DIR] [--masks DIR] [--hold-frames DIR]

keys.json maps limb name -> {frame: {"seed": [x, y], "cut": [[x1, y1], [x2, y2]]}}
or {frame: {"seed": [x, y], "poly": [[x, y], ...]}}. Between keyed frames the
points are interpolated linearly (both keys of a span must use the same form and
point count); a limb is only active from its first to its last keyed frame (a
key may be null to end a span early).

For each active limb the mask is the region of the fox's interior (pixels that
are neither outline blue nor transparent) connected to the seed, on the seed's
side of the cut line (or inside the polygon), within MAX_R of the seed. A
polygon is for poses where a straight cut's extension would reach the knee or
body; only its edge across the wrist/ankle needs to be exact. The paw's own outline separates
it from a body or tail it overlaps, and the cut line sets where the brown stops
on the forearm or shin. The mask is then grown by a few pixels so it covers the
paw's outline edge and finger lines (both identical in the two layers), and
feathered across the cut line so the colour change does not alias.

Kling animates on twos: when a frame repeats the previous drawing, the
previous frame's mask is reused unchanged (whatever the keys say) so the brown
edge cannot drift on a held pose. Holds are detected on --hold-frames (the
original green-screen frames as PNGs) when given, else on the base frames. A seed that lands in a tiny closed area (between toe lines, on an
outline) is moved to the nearest reasonably sized component within SNAP px.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("needs opencv-python-headless")

MAX_R = 260  # px, generous: the cut line and outline do the real limiting
GROW = 6  # px of dilation to cover the paw outline/finger lines
FEATHER = 1.5  # px gaussian sigma on the final mask
MIN_AREA = 150  # px: smaller seed components are toe gaps or outline specks
SNAP = 45  # px search radius for a better seed component
HOLD_DIFF = 0.003  # changed-pixel fraction below which a frame is a hold (moving frames: >2.5%)
# the sprint frames, and the blurred mid-turn/back-view frames, have no outline
# around the paws, so their masks get a softer edge that reads as part of the
# blur rather than a visible line
FEATHER_BY_PREFIX = {"run_": 5.0, "soft_": 4.0}


def feather_for(limb):
    for prefix, sigma in FEATHER_BY_PREFIX.items():
        if limb.startswith(prefix):
            return sigma
    return FEATHER


def interp_keys(keys, frame):
    """Return (seed, cut) for frame, or None if the limb is inactive."""
    ks = sorted(int(k) for k in keys)
    if not ks or frame < ks[0] or frame > ks[-1]:
        return None
    lo = max(k for k in ks if k <= frame)
    hi = min(k for k in ks if k >= frame)
    a, b = keys[str(lo)], keys[str(hi)]
    if a is None or b is None:
        return None
    t = 0.0 if hi == lo else (frame - lo) / (hi - lo)
    form = "poly" if "poly" in a else "cut"
    if form not in b or len(a[form]) != len(b[form]):
        sys.exit(f"frame {lo}-{hi}: keys must share form and point count")
    seed = np.array(a["seed"], float) * (1 - t) + np.array(b["seed"], float) * t
    pts = np.array(a[form], float) * (1 - t) + np.array(b[form], float) * t
    return seed, form, pts


def interior_mask(rgba):
    rgb = rgba[..., :3].astype(int)
    alpha = rgba[..., 3]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    outline = (b > r + 40) & (b > 90)  # dark-blue lines
    return (alpha > 128) & ~outline


def limb_mask(interior, seed, form, pts):
    h, w = interior.shape
    yy, xx = np.mgrid[0:h, 0:w]
    if form == "poly":
        clip = np.zeros((h, w), np.uint8)
        cv2.fillPoly(clip, [np.round(pts).astype(np.int32)], 1)
        half = clip > 0
    else:
        (x1, y1), (x2, y2) = pts
        nx, ny = -(y2 - y1), (x2 - x1)  # normal of the cut line
        side = np.sign((seed[0] - x1) * nx + (seed[1] - y1) * ny)
        half = ((xx - x1) * nx + (yy - y1) * ny) * side > 0
    near = (xx - seed[0]) ** 2 + (yy - seed[1]) ** 2 < MAX_R ** 2
    region = (interior & half & near).astype(np.uint8)
    n, lab = cv2.connectedComponents(region, connectivity=4)
    sx, sy = int(round(seed[0])), int(round(seed[1]))
    sx, sy = min(max(sx, 0), w - 1), min(max(sy, 0), h - 1)
    lid = lab[sy, sx]
    if lid == 0 or (lab == lid).sum() < MIN_AREA:
        # seed on a line or in a toe gap: take the nearest real component
        y0, x0 = max(sy - SNAP, 0), max(sx - SNAP, 0)
        win = lab[y0:sy + SNAP + 1, x0:sx + SNAP + 1]
        best, lid = None, 0
        for k in np.unique(win[win > 0]):
            if (lab == k).sum() < MIN_AREA:
                continue
            ys, xs = np.nonzero(win == k)
            d = ((xs + x0 - sx) ** 2 + (ys + y0 - sy) ** 2).min()
            if best is None or d < best:
                best, lid = d, k
        if lid == 0:
            return np.zeros((h, w), bool)
    m = (lab == lid).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROW + 1,) * 2))
    return (m > 0) & half


def main():
    keys_path, base_dir, layer_dir, out_dir = sys.argv[1:5]
    debug = sys.argv[sys.argv.index("--debug") + 1] if "--debug" in sys.argv else None
    masks = sys.argv[sys.argv.index("--masks") + 1] if "--masks" in sys.argv else None
    hold_dir = sys.argv[sys.argv.index("--hold-frames") + 1] if "--hold-frames" in sys.argv else None
    hold_names = sorted(os.listdir(hold_dir)) if hold_dir else None
    if masks:
        os.makedirs(masks, exist_ok=True)
    keys = json.load(open(keys_path))
    os.makedirs(out_dir, exist_ok=True)
    if debug:
        os.makedirs(debug, exist_ok=True)
    names = sorted(os.listdir(base_dir))
    prev_small, prev_mask, prev_cuts = None, None, None
    for i, name in enumerate(names):
        base = np.asarray(Image.open(os.path.join(base_dir, name)).convert("RGBA"))
        layer = np.asarray(Image.open(os.path.join(layer_dir, name)).convert("RGBA"))
        ref = (np.asarray(Image.open(os.path.join(hold_dir, hold_names[i])).convert("L"))
               if hold_dir else cv2.cvtColor(base[..., :3], cv2.COLOR_RGB2GRAY))
        small = cv2.resize(ref, (480, 270), interpolation=cv2.INTER_AREA).astype(np.int16)
        # a held drawing keeps the previous mask even if a limb span starts or
        # ends on it, so the brown never changes on a still pose
        held = (prev_small is not None
                and (np.abs(small - prev_small) > 20).mean() < HOLD_DIFF)
        if held:
            m, cuts = prev_mask, prev_cuts
        else:
            interior = interior_mask(base)
            m = np.zeros(interior.shape, np.float32)
            cuts = []
            for limb, lk in keys.items():
                st = interp_keys(lk, i)
                if st is None:
                    continue
                seed, form, pts = st
                lm = limb_mask(interior, seed, form, pts).astype(np.float32)
                if lm.any():
                    m = np.maximum(m, cv2.GaussianBlur(lm, (0, 0), feather_for(limb)))
                cuts.append((limb, seed, form, pts))
        prev_small, prev_mask, prev_cuts = small, m, cuts
        if masks:
            Image.fromarray(np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8), "L").save(
                os.path.join(masks, name), compress_level=3)
        m = m[..., None]
        out = base.astype(np.float32).copy()
        out[..., :3] = base[..., :3] * (1 - m) + layer[..., :3] * m
        out[..., 3] = base[..., 3]  # transparency stays exactly the v2 alpha
        res = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(res, "RGBA").save(os.path.join(out_dir, name), compress_level=3)
        if debug and cuts:
            bg = Image.new("RGBA", res.shape[1::-1], (0, 255, 0, 255))
            bg.alpha_composite(Image.fromarray(res, "RGBA"))
            d = ImageDraw.Draw(bg)
            for limb, seed, form, pts in cuts:
                line = [tuple(p) for p in pts] + ([tuple(pts[0])] if form == "poly" else [])
                d.line(line, fill=(255, 0, 255, 255), width=3)
                d.ellipse([seed[0] - 6, seed[1] - 6, seed[0] + 6, seed[1] + 6], outline=(255, 255, 0, 255), width=3)
                d.text((seed[0] + 8, seed[1] - 8), limb, fill=(0, 0, 0, 255))
            bg.convert("RGB").save(os.path.join(debug, name.replace(".png", ".jpg")), quality=88)
    print(f"{len(names)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
