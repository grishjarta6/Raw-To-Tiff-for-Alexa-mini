#!/usr/bin/env python3
"""
ARRIRAW ALEXA Mini MXF → 4 монохромных TIFF.

Правильная структура essence элемента (11 309 548 байт):
    [  76 байт  ]  frame header
    [ 5 654 736 ]  chunk 1: чётные столбцы (1712 × 2202 пикс, 12-бит)
    [ 5 654 736 ]  chunk 2: нечётные столбцы

Формула 12-бит распаковки (arri_alt):
    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
HDR = 76
HALF_W = W // 2                 # 1712
CHUNK_BYTES = HALF_W * H * 3 // 2  # 5 654 736
FRAME_SIZE = HDR + 2 * CHUNK_BYTES # 11 309 548


# ---------------------------------------------------------------------------
# Формула распаковки 12-бит (найдена эмпирически)
# ---------------------------------------------------------------------------

def unpack_arri_alt(b0, b1, b2):
    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1
    return p1, p2


def unpack_chunk(raw):
    """5654736 байт → 3 769 824 пикселей (uint16)."""
    n_pairs = len(raw) // 3
    d = np.frombuffer(raw[:n_pairs * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1, p2 = unpack_arri_alt(b0, b1, b2)
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


# ---------------------------------------------------------------------------
# Паттерны Bayer
# ---------------------------------------------------------------------------

PATTERNS = {
    # (top-left, top-right, bottom-left, bottom-right)
    "BGGR": ("B",  "G2", "G1", "R"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


def build_bayer(chunk_even, chunk_odd, pattern, swap_rb=False):
    """
    Собирает полный кадр из двух половин.

    chunk_even  →  чётные столбцы  (x = 0, 2, 4, ...)
    chunk_odd   →  нечётные столбцы (x = 1, 3, 5, ...)
    """
    ph_even = chunk_even.reshape(H, HALF_W)
    ph_odd  = chunk_odd.reshape(H, HALF_W)

    bayer = np.empty((H, W), dtype=np.uint16)
    bayer[:, 0::2] = ph_even
    bayer[:, 1::2] = ph_odd

    # Извлекаем 4 канала по паттерну
    tl, tr, bl, br = PATTERNS[pattern]
    chans = {
        tl: bayer[0::2, 0::2],
        tr: bayer[0::2, 1::2],
        bl: bayer[1::2, 0::2],
        br: bayer[1::2, 1::2],
    }

    if swap_rb:
        chans["R"], chans["B"] = chans["B"], chans["R"]

    return chans


# ---------------------------------------------------------------------------
# Обработка
# ---------------------------------------------------------------------------

def process_frame(path, frame_idx, pattern="BGGR", swap_chunks=False,
                  swap_rb=False, out_dir=None):
    """Читает один кадр, сохраняет 4 моно TIFF + RGB-превью."""
    off = mxf_parser.parse(path)["header_offset"]

    with open(path, "rb") as f:
        f.seek(off + frame_idx * FRAME_SIZE)
        raw = f.read(FRAME_SIZE)

    if len(raw) < FRAME_SIZE:
        raise ValueError(f"Кадр {frame_idx} не помещается в файле")

    chunk1 = raw[HDR : HDR + CHUNK_BYTES]
    chunk2 = raw[HDR + CHUNK_BYTES : HDR + 2 * CHUNK_BYTES]

    if swap_chunks:
        chunk1, chunk2 = chunk2, chunk1

    pix1 = unpack_chunk(chunk1)
    pix2 = unpack_chunk(chunk2)

    chans = build_bayer(pix1, pix2, pattern, swap_rb=swap_rb)

    return chans


def save_all(chans, prefix):
    for name, img in chans.items():
        tifffile.imwrite(f"{prefix}_{name}.tiff", img)
        print(f"   💾 {prefix}_{name}.tiff   "
              f"({img.shape[1]}×{img.shape[0]}, "
              f"std={img.std():.0f})")

    # Псевдо-RGB превью (downscale ×4)
    def norm(x):
        p = np.percentile(x.astype(np.float32), 99.0)
        return np.clip(x.astype(np.float32) / max(p, 1), 0, 1)

    R  = norm(chans["R"])
    B  = norm(chans["B"])
    G1 = norm(chans["G1"])
    G2 = norm(chans["G2"])
    G  = 0.5 * (G1 + G2)
    rgb = np.stack([R, G, B], axis=-1)

    h, w = rgb.shape[:2]
    h2, w2 = h // 4, w // 4
    rgb4 = rgb[:h2*4, :w2*4].reshape(h2, 4, w2, 4, 3).mean(axis=(1, 3))
    tifffile.imwrite(f"{prefix}_RGB.tiff", (rgb4 * 255).astype(np.uint8))
    print(f"   💾 {prefix}_RGB.tiff   (превью ×4)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=MXF_FILE)
    ap.add_argument("--frame", type=int, default=0,
                    help="номер кадра (0..16)")
    ap.add_argument("--pattern", default="BGGR", choices=list(PATTERNS))
    ap.add_argument("--swap-chunks", action="store_true",
                    help="поменять местами chunk1 и chunk2")
    ap.add_argument("--swap-rb", action="store_true",
                    help="поменять местами R и B каналы")
    ap.add_argument("--all-patterns", action="store_true",
                    help="сохранить превью для всех 4 паттернов")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        sys.exit(1)

    off = mxf_parser.parse(path)["header_offset"]
    n_frames = (path.stat().st_size - off) // FRAME_SIZE

    print(f"Файл:      {path.name}")
    print(f"Offset:    {off:,}")
    print(f"Кадров:    {n_frames}")
    print(f"Кадр №:    {args.frame}")
    print(f"Pattern:   {args.pattern}")
    print(f"Swap chunks: {args.swap_chunks}")
    print(f"Swap R↔B:    {args.swap_rb}")
    print()

    if args.all_patterns:
        for pat in PATTERNS:
            print(f"=== pattern={pat} ===")
            chans = process_frame(path, args.frame, pattern=pat,
                                  swap_chunks=args.swap_chunks,
                                  swap_rb=args.swap_rb)
            save_all(chans, f"frame{args.frame:03d}_{pat}")
            print()
        return

    chans = process_frame(path, args.frame, pattern=args.pattern,
                          swap_chunks=args.swap_chunks,
                          swap_rb=args.swap_rb)
    save_all(chans, f"frame{args.frame:03d}_{args.pattern}")


if __name__ == "__main__":
    main()