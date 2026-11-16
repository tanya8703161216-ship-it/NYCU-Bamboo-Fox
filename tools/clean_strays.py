#!/usr/bin/env python3
"""Remove stray strokes and ghosting around the fox on green-screen frames.

Usage: clean_strays.py <in_png_dir> <out_png_dir> <spec> [--debug DIR]
                       [--erase f:x0:y0:x1:y1:ax:ay:bx:by ...]

<spec> is a comma-separated list of frame ranges with a mode, e.g.
"6-21:remove,22-72:gap" (0-based frame indices; frames not listed are copied).

Every connected non-green blob of at least BIG px is the fox (its body, tail
and outline) or the blue corner panel and is never touched. Smaller blobs
detached from them are strays: action lines, speed dashes, smudges.

  remove  every stray is painted back to the frame's green (landing dashes,
          afterimages under the feet), and so are thin dark strokes sticking
          out of the feet or the panel edge in the lowest FOOT_ZONE of the fox
          (dashes that touch an outline). Shapes touching the frame edge, like
          the panel's thin tip, are kept.

--flatten evens out the green screen itself on every frame: pixels that are
clearly screen (green dominance above FLAT_HI) become the frame's median
green, with a soft ramp down to FLAT_LO so outline edges are untouched. This
clears faint smears left where strokes were and helps keying.

--erase handles strokes fused onto the panel edge that no rule can separate:
in frame f, dark pixels inside the box x0..x1, y0..y1 that lie above the line
(ax,ay)-(bx,by) (the panel's top edge) are painted back to green.
  gap     strays are kept but trimmed so no part comes closer than GAP px to
          the fox outline; leftover crumbs and faint, smeared lines are dropped
          so the wave lines that remain are short, solid and clear of the
          hand.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

FLAT_LO, FLAT_HI = 70, 110  # green-dominance ramp for --flatten
GAP = 16  # px of clear green kept between a motion line and the fox
MIN_KEEP = 60  # px: trimmed line pieces smaller than this are dropped
SOLID = -40  # a kept line's median green-dominance must be below this (solid blue)
BIG = 6000  # px: non-screen blobs at least this large are the fox or the panel
SPUR_OPEN = 15  # px: opening size that strips thin strokes off the fox/panel edge
SPUR_MAX = 3000  # px: larger protrusions are real shapes, not strokes
FOOT_ZONE = 0.3  # spurs are only stripped in the lowest part of the fox


def masks(rgb):
    f = rgb.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    screen = (g - np.maximum(r, b)) > 60
    # every non-screen blob: the fox (with the panel it may stand on) is the
    # large ones; detached strokes, dashes and smudges are the small ones
    n, lab, st, _ = cv2.connectedComponentsWithStats((~screen).astype(np.uint8), 8)
    big = np.zeros(n, bool)
    big[1:] = st[1:, 4] >= BIG
    fox = big[lab]
    stray = ~screen & ~fox
    return screen, fox, stray


def spurs(rgb, fox):
    blob = fox.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPUR_OPEN, SPUR_OPEN))
    thin = (blob > 0) & ~(cv2.morphologyEx(blob, cv2.MORPH_OPEN, k) > 0)
    f = rgb.astype(np.int16)
    dark = (f[..., 2] > f[..., 0] + 30) | (f.max(-1) < 110)  # outline blue or a dark smudge
    ys = np.nonzero(fox & ~dark)[0]
    if len(ys) == 0:
        return np.zeros_like(fox)
    y_foot = ys.max() - FOOT_ZONE * (ys.max() - ys.min())
    h, w = fox.shape
    n, lab, st, cen = cv2.connectedComponentsWithStats(thin.astype(np.uint8), 8)
    out = np.zeros_like(fox)
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a > SPUR_MAX or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        if cen[i][1] < y_foot:
            continue
        comp = lab == i
        if dark[comp].mean() > 0.7:
            out |= comp
    return out


def clean(rgb, mode, extra=()):
    screen, fox, stray = masks(rgb)
    if mode == "remove":
        kill = stray | spurs(rgb, fox)
    else:
        dist = cv2.distanceTransform((~fox).astype(np.uint8), cv2.DIST_L2, 5)
        kill = stray & (dist < GAP)
        keep = (stray & ~kill).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(keep, 8)
        f = rgb.astype(np.int16)
        dom = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
        for k in range(1, n):
            comp = lab == k
            if st[k, 4] < MIN_KEEP or np.median(dom[comp]) > SOLID:
                kill |= comp
    for e in extra:
        kill = kill | e
        fox = fox & ~e
    if not kill.any():
        return rgb, kill
    # widen a little to take the anti-aliased rim, but never into the fox
    grow = cv2.dilate(kill.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    grow &= ~cv2.dilate(fox.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    green = np.median(rgb[screen], axis=0) if screen.any() else np.array([8, 237, 7])
    m = cv2.GaussianBlur(grow.astype(np.float32), (0, 0), 1.0)
    m[grow] = 1.0
    out = rgb * (1 - m[..., None]) + green * m[..., None]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), grow


def manual_erase(rgb, box_line):
    x0, y0, x1, y1, ax, ay, bx, by = box_line
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    f = rgb.astype(np.int16)
    dark = (f[..., 2] > f[..., 0] + 30) | (f.max(-1) < 110)
    edge_y = ay + (xx - ax) * (by - ay) / (bx - ax)
    return dark & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1) & (yy < edge_y - 3)


def flatten(rgb):
    f = rgb.astype(np.float32)
    dom = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    screen = dom > FLAT_HI
    if not screen.any():
        return rgb
    green = np.median(rgb[screen], axis=0).astype(np.float32)
    w = np.clip((dom - FLAT_LO) / (FLAT_HI - FLAT_LO), 0, 1)[..., None]
    return np.clip(f * (1 - w) + green * w + 0.5, 0, 255).astype(np.uint8)


def parse(spec):
    plan = {}
    for part in spec.split(","):
        rng, mode = part.split(":")
        a, b = (rng.split("-") + [rng])[:2]
        for i in range(int(a), int(b) + 1):
            plan[i] = mode
    return plan


def main():
    src, dst, spec = sys.argv[1:4]
    debug = sys.argv[sys.argv.index("--debug") + 1] if "--debug" in sys.argv else None
    plan = parse(spec)
    erase = {}
    if "--erase" in sys.argv:
        for item in sys.argv[sys.argv.index("--erase") + 1:]:
            if item.startswith("--"):
                break
            f, *vals = item.split(":")
            erase.setdefault(int(f), []).append([float(v) for v in vals])
    os.makedirs(dst, exist_ok=True)
    if debug:
        os.makedirs(debug, exist_ok=True)
    names = sorted(os.listdir(src))
    changed = 0
    for i, name in enumerate(names):
        rgb = np.asarray(Image.open(os.path.join(src, name)).convert("RGB"))
        if i in plan:
            out, kill = clean(rgb, plan[i], [manual_erase(rgb, e) for e in erase.get(i, [])])
            if kill.any():
                changed += 1
                if debug:
                    dbg = rgb.copy()
                    dbg[kill] = (255, 0, 255)
                    Image.fromarray(dbg).save(os.path.join(debug, name))
        else:
            out = rgb
        if "--flatten" in sys.argv:
            out = flatten(out)
        Image.fromarray(out).save(os.path.join(dst, name), compress_level=3)
    print(f"{len(names)} frames, {changed} cleaned -> {dst}")


if __name__ == "__main__":
    main()
