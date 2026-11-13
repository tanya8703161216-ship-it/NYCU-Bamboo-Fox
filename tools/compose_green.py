#!/usr/bin/env python3
"""Apply the paw/foot recolour to the original green-screen video.

Usage: compose_green.py <green.mp4> <mask_png_dir> <out_png_dir>

Uses the masks written by roto_limbs.py --masks and the same orange->brown
transform as recolor_layer.py, so the green-screen version matches the
transparent one while keeping Kling's original green, edges and everything
outside the paws/feet untouched.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recolor_layer import recolor  # noqa: E402


def main():
    src, mask_dir, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    masks = sorted(os.listdir(mask_dir))
    w, h = 1920, 1080
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = 0
    while True:
        buf = p.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        m = np.asarray(Image.open(os.path.join(mask_dir, masks[n]))).astype(np.float32)[..., None] / 255
        rgba = np.dstack([rgb, np.full((h, w), 255, np.uint8)])
        brown = recolor(rgba)[..., :3].astype(np.float32)
        out = rgb * (1 - m) + brown * m
        Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB").save(
            os.path.join(out_dir, masks[n]), compress_level=3)
        n += 1
    p.wait()
    print(f"{n} frames -> {out_dir}")


if __name__ == "__main__":
    main()
