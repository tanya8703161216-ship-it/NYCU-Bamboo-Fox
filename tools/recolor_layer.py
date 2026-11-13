#!/usr/bin/env python3
"""Make a recoloured copy of a straight-alpha RGBA PNG sequence.

Usage: recolor_layer.py <in_png_dir> <out_png_dir>

Every pixel is decomposed (relative to white) into how much of it is the fox's
orange and how much is the dark-blue outline; only the orange share is moved to
the new brown, so white areas, outlines, anti-aliased edges and shading keep
their original look. Alpha is copied unchanged.

The colours come from the keyframes: 01-04 and 07 changed the paws/feet from
orange #EC6D20 (236,109,32) to brown (153,59,21). Applied per channel as a
ratio, which carries Kling's slight shading variations over to the brown.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

ORANGE = np.array([236.0, 109.0, 32.0])
BROWN = np.array([153.0, 59.0, 21.0])
WHITE = np.array([255.0, 255.0, 255.0])
BLUE = np.array([0.0, 51.0, 160.0])
RATIO = BROWN / ORANGE


def recolor(rgba):
    f = rgba[..., :3].astype(np.float32)
    # Solve f - W = a*(O - W) + d*(B - W) per pixel, least squares.
    basis = np.stack([ORANGE - WHITE, BLUE - WHITE], axis=1)  # 3x2
    pinv = np.linalg.pinv(basis)  # 2x3
    coef = (f - WHITE) @ pinv.T
    a = np.clip(coef[..., 0], 0.0, 1.0)[..., None]
    out = f * (1.0 - a + a * RATIO)
    res = rgba.copy()
    res[..., :3] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return res


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.png")))
    for path in files:
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        Image.fromarray(recolor(rgba), "RGBA").save(
            os.path.join(dst, os.path.basename(path)), compress_level=3)
    print(f"{len(files)} frames -> {dst}")


if __name__ == "__main__":
    main()
