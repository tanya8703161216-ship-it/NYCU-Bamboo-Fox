#!/usr/bin/env python3
"""Splice a re-generated run into a green-screen frame sequence.

Usage: splice_run.py <base_dir> <run_dir> <out_dir> <first> <last> <src0> <step>

Kling drew the new run mostly in place, so the run poses are taken from the
run clip (frame src0, src0+step, ...) and slid across the screen right to left:
output frame <first> shows the fox just entering at the right edge and by
frame <last> the fox and its trailing speed lines have fully left on the
left. Each pose keeps its own pixels (no resampling); only the horizontal
position changes. Frames after <last> and the gap before <first> are plain
green. All frames are written with the base sequence's green so the splice is
seamless for keying.
"""
import os
import sys

import numpy as np
from PIL import Image

ENTER_X = 1800  # left edge of the fox on the first run frame


def screen_green(rgb):
    f = rgb.astype(np.int16)
    dom = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    return np.median(rgb[dom > 110], axis=0).astype(np.uint8)


def recolor_screen(rgb, green):
    f = rgb.astype(np.float32)
    dom = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    w = np.clip((dom - 70) / 40, 0, 1)[..., None]
    return np.clip(f * (1 - w) + green * w + 0.5, 0, 255).astype(np.uint8)


def fox_left(rgb):
    f = rgb.astype(np.int16)
    dom = f[..., 1] - np.maximum(f[..., 0], f[..., 2])
    xs = np.nonzero((dom < 40).any(axis=0))[0]
    return xs.min(), xs.max()


def main():
    base, run, out, first, last, src0, step = sys.argv[1:8]
    first, last, src0, step = int(first), int(last), int(src0), int(step)
    os.makedirs(out, exist_ok=True)
    names = sorted(os.listdir(base))
    green = screen_green(np.asarray(Image.open(os.path.join(base, names[0])).convert("RGB")))
    poses = [np.asarray(Image.open(os.path.join(run, f"{src0 + k * step:03d}.png")).convert("RGB"))
             for k in range(last - first + 1)]
    extents = [fox_left(p) for p in poses]
    # constant speed chosen so the widest pose (incl. speed lines) is gone at <last>
    span = max(x1 - x0 for x0, x1 in extents) + 1
    speed = (ENTER_X + span) / (last - first)
    h, w = poses[0].shape[:2]
    for i, name in enumerate(names):
        if i < first - 3:
            rgb = np.asarray(Image.open(os.path.join(base, name)).convert("RGB"))
            frame = recolor_screen(rgb, green)
        elif first <= i <= last:
            k = i - first
            pose = recolor_screen(poses[k], green)
            dx = int(round(ENTER_X - k * speed - extents[k][0]))
            frame = np.empty_like(pose)
            frame[:] = green
            xs0, xs1 = max(0, -dx), min(w, w - dx)
            if xs1 > xs0:
                frame[:, xs0 + dx:xs1 + dx] = pose[:, xs0:xs1]
        else:
            frame = np.empty((h, w, 3), np.uint8)
            frame[:] = green
        Image.fromarray(frame).save(os.path.join(out, name), compress_level=3)
    print(f"speed {speed:.0f} px/frame, green {green.tolist()} -> {out}")


if __name__ == "__main__":
    main()
