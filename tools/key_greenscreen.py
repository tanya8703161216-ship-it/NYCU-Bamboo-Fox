#!/usr/bin/env python3
"""Key a #00FF00 green-screen video into RGBA frames with spill suppression.

Usage: key_greenscreen.py <input.mp4> <out_png_dir>

Alpha is computed per pixel from how much green exceeds the stronger of red/blue
(a "green dominance" key), which suits flat 2D artwork: the fox's orange,
white and dark-blue never have green above red and blue, so they stay opaque.
Spill is suppressed by clamping green to max(red, blue) on semi/opaque pixels.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

# Green dominance (G - max(R, B)) at or below LO is fully opaque, at or above HI fully transparent.
LO, HI = 40.0, 140.0


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", path,
    ]).decode().strip()
    w, h = map(int, out.split(",")[:2])
    return w, h


def key_frame(rgb):
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    dom = g - np.maximum(r, b)
    alpha = 1.0 - np.clip((dom - LO) / (HI - LO), 0.0, 1.0)
    # Despill: green may not exceed the stronger of red/blue.
    g2 = np.minimum(g, np.maximum(r, b))
    out = np.stack([r, g2, b, alpha * 255.0], axis=-1)
    out[alpha == 0] = 0  # clean, fully transparent pixels
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    w, h = probe(src)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    size = w * h * 3
    n = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        n += 1
        Image.fromarray(key_frame(rgb), "RGBA").save(
            os.path.join(out_dir, f"fox_{n:04d}.png"), compress_level=3)
    proc.wait()
    print(f"{n} frames -> {out_dir}")


if __name__ == "__main__":
    main()
