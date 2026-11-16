#!/usr/bin/env python3
"""Recolour the orange strip at the tip of each ear to the brown of the keyframes.

Each ear is a white inner-ear shape with a thin orange strip between it and
the dark-blue outline on the side facing the head. The keyframes turn the
upper part of that strip brown. Per frame:

  * the two inner-ear whites are the white blobs reaching highest;
  * each ear gets an axis from the white blob's centroid to its apex (the
    white pixel farthest from a head centre placed below and between them);
  * the strip is the orange fill within STRIP px of that white blob, and a
    pixel turns brown when its position along the axis is beyond the cut
    fraction measured on the reference keyframe for that ear (left/right).

The colour change uses the same orange->brown transform as the paws
(recolor_layer.recolor), so white, outline and green are untouched.
Held drawings (Kling animates on twos) reuse the previous frame's mask.

Usage:
  ear_tips.py measure <reference.jpg>          print the cut for each ear
  ear_tips.py apply <in_dir> <out_dir> <first> <last> <cutL> <cutR> [--debug DIR]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recolor_layer import recolor  # noqa: E402

STRIP = 55  # px from the inner-ear white that the orange strip can reach
FEATHER = 1.2
HOLD_DIFF = 0.003
AREA_BAND = (0.7, 1.3)  # allowed tip area relative to the segment median
MAX_TIP = 2500
EAR_WHITE_MAX = 18000  # px: largest inner-ear white (01's near ear is ~14.7k)  # px: largest plausible brown ear tip at 1080p
BROWN = np.array([153, 59, 21])


def classes(rgb):
    f = rgb.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    dom = g - np.maximum(r, b)
    screen = dom > 40
    blue = (b > r + 40) & (b > 90)
    white = (f.min(-1) > 200) & ~screen
    orange = (r > 170) & (g > 70) & (g < 150) & (b < 90)
    brown = (np.abs(f - BROWN).max(-1) < 40)
    return screen, blue, white, orange, brown


def ears(rgb):
    """Return [(white_mask, centroid, apex)] for the left and right ear."""
    screen, blue, white, orange, brown = classes(rgb)
    n, lab, st, cen = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    # inner-ear whites only: the tail tip and face are far larger
    cands = [i for i in range(1, n) if 1500 < st[i, 4] < EAR_WHITE_MAX]
    if len(cands) < 2:
        return None
    # the ears are the two white blobs reaching highest; the head centre is a
    # point below and between them (face whites and the tail tip vary too
    # much between poses to anchor on)
    rest = sorted(cands, key=lambda i: st[i, 1])[:2]
    c0, c1 = np.array(cen[rest[0]]), np.array(cen[rest[1]])
    spread = np.linalg.norm(c1 - c0)
    fc = (c0 + c1) / 2 + np.array([0.0, spread])
    out = []
    for i in sorted(rest, key=lambda i: cen[i][0]):
        m = lab == i
        ys, xs = np.nonzero(m)
        d = (xs - fc[0]) ** 2 + (ys - fc[1]) ** 2
        k = np.argmax(d)
        out.append((m, np.array(cen[i]), np.array([xs[k], ys[k]], float)))
    return out


def axis_t(shape, c, apex):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    u = apex - c
    L2 = (u ** 2).sum()
    return ((xx - c[0]) * u[0] + (yy - c[1]) * u[1]) / L2


def strip_mask(rgb, ear):
    screen, blue, white, orange, brown = classes(rgb)
    m, c, apex = ear
    dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    return (orange | brown) & (dist < STRIP), axis_t(m.shape, c, apex)


def measure(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    e = ears(rgb)
    _, _, _, _, brown = classes(rgb)
    for side, ear in zip("LR", e):
        strip, t = strip_mask(rgb, ear)
        tb = t[strip & brown]
        to = t[strip & ~brown]
        print(f"{side}: brown t {np.percentile(tb, 5):.3f}..{tb.max():.3f}  "
              f"orange t ..{np.percentile(to, 95):.3f}  px brown {len(tb)}")


def tip_mask(rgb, cuts, targets=None):
    e = ears(rgb)
    mask = np.zeros(rgb.shape[:2], bool)
    if e is None:
        return mask, []
    for k, (cut, ear) in enumerate(zip(cuts, e)):
        strip, t = strip_mask(rgb, ear)
        m = strip & (t >= cut)
        if targets is not None and targets[k] and m.any():
            # head turns change the ear's foreshortening: move the cut so the
            # tip keeps a size close to its usual one in this segment
            lo, hi = AREA_BAND[0] * targets[k], AREA_BAND[1] * targets[k]
            c = cut
            while m.sum() > hi and c < 1.5:
                c += 0.02
                m = strip & (t >= c)
            while m.sum() < lo and c > -1.0:
                c -= 0.02
                m = strip & (t >= c)
        # a strip far larger than an ear tip means the "ear" was a misread
        # (e.g. an ear cut off by the frame edge); leave that ear alone
        if m.sum() <= MAX_TIP:
            mask |= m
    return mask, e


def main():
    if sys.argv[1] == "measure":
        measure(sys.argv[2])
        return
    src, dst, first, last, cl, cr = sys.argv[2:8]
    first, last, cuts = int(first), int(last), (float(cl), float(cr))
    debug = sys.argv[sys.argv.index("--debug") + 1] if "--debug" in sys.argv else None
    os.makedirs(dst, exist_ok=True)
    if debug:
        os.makedirs(debug, exist_ok=True)
    names = sorted(os.listdir(src))
    # pass 1: each ear's usual tip size over the segment
    areas = [[], []]
    for i in range(first, last + 1):
        rgb = np.asarray(Image.open(os.path.join(src, names[i])).convert("RGB"))
        e = ears(rgb)
        if e is None:
            continue
        for k, (cut, ear) in enumerate(zip(cuts, e)):
            strip, t = strip_mask(rgb, ear)
            a = int((strip & (t >= cut)).sum())
            if 0 < a <= MAX_TIP:
                areas[k].append(a)
    targets = [float(np.median(a)) if a else None for a in areas]
    print("median tip area L/R:", targets)
    prev_small, prev_m = None, None
    for i, name in enumerate(names):
        rgb = np.asarray(Image.open(os.path.join(src, name)).convert("RGB"))
        if not first <= i <= last:
            Image.fromarray(rgb).save(os.path.join(dst, name), compress_level=3)
            continue
        small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (480, 270),
                           interpolation=cv2.INTER_AREA).astype(np.int16)
        if prev_small is not None and (np.abs(small - prev_small) > 20).mean() < HOLD_DIFF:
            m = prev_m
        else:
            mask, _ = tip_mask(rgb, cuts, targets)
            m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), FEATHER) if mask.any() \
                else np.zeros(mask.shape, np.float32)
            m[mask] = np.maximum(m[mask], 0.85)
        prev_small, prev_m = small, m
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        br = recolor(rgba)[..., :3].astype(np.float32)
        out = rgb * (1 - m[..., None]) + br * m[..., None]
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(os.path.join(dst, name), compress_level=3)
        if debug:
            dbg = out.copy()
            dbg[(m > 0.5)] = (255, 0, 255)
            Image.fromarray(dbg).save(os.path.join(debug, name))
    print(f"{last - first + 1} frames processed -> {dst}")


if __name__ == "__main__":
    main()
