#!/usr/bin/env python3
"""Key a #00FF00 green-screen video into straight-alpha RGBA PNG frames.

Usage: key_greenscreen.py <input.mp4> <out_png_dir> [--blank-last]

Alpha comes from green dominance, G - max(R, B), which suits flat 2D artwork:
the fox's orange, white and dark-blue never have green above red and blue, so
they stay fully opaque, while the green screen (including Kling's compression
drift away from pure #00FF00) goes fully transparent.

Edge pixels are a mix of foreground and screen, C = a*F + (1-a)*G. The screen
colour G is measured per frame, and F is recovered by unmixing so the output is
straight (unmatted) alpha with no green baked into the edges. A final despill
clamps green to max(R, B) to remove any residual spill or fringe.

--blank-last forces the final frame fully transparent (an empty end frame).
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

# Green dominance at or below LO is fully opaque, at or above HI fully transparent.
LO, HI = 40.0, 140.0
FALLBACK_SCREEN = np.array([0.0, 255.0, 0.0], np.float32)


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", path,
    ]).decode().strip()
    w, h = map(int, out.split(",")[:2])
    return w, h


def key_frame(rgb, screen):
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    dom = g - np.maximum(r, b)
    alpha = 1.0 - np.clip((dom - LO) / (HI - LO), 0.0, 1.0)

    # Unmix partially transparent pixels back to their foreground colour.
    partial = (alpha > 0) & (alpha < 1)
    a = alpha[partial][:, None]
    fg = f.copy()
    fg[partial] = np.clip((f[partial] - (1.0 - a) * screen) / a, 0.0, 255.0)

    # Despill: green may not exceed the stronger of red/blue.
    fr, fgr, fb = fg[..., 0], fg[..., 1], fg[..., 2]
    fg[..., 1] = np.minimum(fgr, np.maximum(fr, fb))

    out = np.concatenate([fg, alpha[..., None] * 255.0], axis=-1)
    out[alpha == 0] = 0  # clean, fully transparent pixels
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def screen_colour(rgb):
    f = rgb.astype(np.float32).reshape(-1, 3)
    dom = f[:, 1] - np.maximum(f[:, 0], f[:, 2])
    bg = f[dom >= HI]
    return np.median(bg, axis=0) if len(bg) > 1000 else FALLBACK_SCREEN


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    blank_last = "--blank-last" in sys.argv
    src, out_dir = args
    os.makedirs(out_dir, exist_ok=True)
    w, h = probe(src)
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    size = w * h * 3
    screen = FALLBACK_SCREEN
    frames = []
    n = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        s = screen_colour(rgb)
        if s is not FALLBACK_SCREEN:
            screen = s  # full-frame tail shots reuse the last measured screen
        n += 1
        path = os.path.join(out_dir, f"fox_{n:04d}.png")
        Image.fromarray(key_frame(rgb, screen), "RGBA").save(path, compress_level=3)
        frames.append(path)
    proc.wait()
    if blank_last and frames:
        Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(frames[-1], compress_level=3)
    print(f"{n} frames -> {out_dir}")


if __name__ == "__main__":
    main()
