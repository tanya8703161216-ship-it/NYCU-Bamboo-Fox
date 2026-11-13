#!/usr/bin/env python3
"""Expand sparse roto keys to every frame by optical-flow tracking.

Usage: track_keys.py <sparse_keys.json> <frames_png_dir> <dense_keys.json>

For each limb, every frame between two keyed frames gets its seed and cut
points tracked forward from the earlier key and backward from the later key
(DIS optical flow on the frames), blended linearly by position in the span so
the result lands exactly on both keys. A null key ends a span. The output has
the same format roto_limbs.py reads, with a key on every active frame.
"""
import json
import os
import sys

import cv2
import numpy as np

SCALE = 0.5  # flow is computed at half resolution


def load_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def main():
    sparse_path, frames_dir, out_path = sys.argv[1:4]
    sparse = json.load(open(sparse_path))
    names = sorted(os.listdir(frames_dir))
    grays = [load_gray(os.path.join(frames_dir, n)) for n in names]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fwd, bwd = {}, {}

    def flow(i, j):
        cache = fwd if j == i + 1 else bwd
        if i not in cache:
            cache[i] = dis.calc(grays[i], grays[j], None)
        return cache[i]

    def move(pts, i, j):
        f = flow(i, j)
        h, w = f.shape[:2]
        out = []
        for x, y in pts:
            sx = int(np.clip(round(x * SCALE), 0, w - 1))
            sy = int(np.clip(round(y * SCALE), 0, h - 1))
            # median flow in a small window is steadier on flat cartoon fills
            win = f[max(sy - 3, 0):sy + 4, max(sx - 3, 0):sx + 4].reshape(-1, 2)
            dx, dy = np.median(win, axis=0) / SCALE
            out.append((x + dx, y + dy))
        return out

    def form_of(k):
        return "poly" if "poly" in k else "cut"

    def pts_of(k):
        return [tuple(k["seed"])] + [tuple(p) for p in k[form_of(k)]]

    dense = {}
    for limb, keys in sparse.items():
        ks = sorted(int(k) for k in keys)
        dl = {}
        for a, b in zip(ks, ks[1:] + [None]):
            ka = keys[str(a)]
            if ka is None:
                continue
            dl[str(a)] = ka
            if b is None or keys[str(b)] is None:
                continue
            kb = keys[str(b)]
            form = form_of(ka)
            if form_of(kb) != form or len(ka[form]) != len(kb[form]):
                sys.exit(f"{limb} {a}-{b}: keys must share form and point count")
            f_tr = [pts_of(ka)]
            for i in range(a, b - 1):
                f_tr.append(move(f_tr[-1], i, i + 1))
            b_tr = [pts_of(kb)]
            for i in range(b, a + 1, -1):
                b_tr.append(move(b_tr[-1], i, i - 1))
            b_tr = b_tr[::-1]  # b_tr[k] is frame a+1+k ... b
            for k, fr in enumerate(range(a + 1, b)):
                t = (fr - a) / (b - a)
                p = [((1 - t) * np.array(f_tr[k + 1][q]) + t * np.array(b_tr[k][q])).tolist()
                     for q in range(len(f_tr[0]))]
                dl[str(fr)] = {"seed": p[0], form: p[1:]}
            dl[str(b)] = kb
        # keep explicit span ends
        for k in ks:
            if keys[str(k)] is None and str(k) not in dl:
                dl[str(k)] = None
        dense[limb] = dl
    json.dump(dense, open(out_path, "w"), indent=0)
    print(f"tracked {sum(len(v) for v in dense.values())} limb-frames -> {out_path}")


if __name__ == "__main__":
    main()
