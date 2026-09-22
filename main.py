#!/usr/bin/env python3
"""
ARRIRAW ALEXA Mini MXF → 4 полных монохромных 16-битных TIFF.

Структура одного essence-пакета (11 309 548 байт):
    [  76 байт  ]  frame header
    [ 5 654 736 ]  chunk 1: ВЕРХНЯЯ половина кадра (1101 × 3424)
    [ 5 654 736 ]  chunk 2: НИЖНЯЯ  половина кадра (1101 × 3424)

Между кадрами в MXF лежат служебные KLV-пакеты (index tables),
поэтому читаем каждый кадр по его собственному offset'у из mxf_parser.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

import mxf_parser


# ===========================================================================
# КОНСТАНТЫ
# ===========================================================================

MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
HDR = 76
HALF_H = H // 2
CHUNK_PIX = HALF_H * W
CHUNK_BYTES = CHUNK_PIX * 3 // 2

PATTERNS = {
    # (top-left, top-right, bottom-left, bottom-right)
    "BGGR": ("B",  "G2", "G1", "R"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


# ===========================================================================
# РАСПАКОВКА
# ===========================================================================

def unpack_chunk(raw: bytes) -> np.ndarray:
    """5 654 736 байт → 3 769 824 пикселей. Формула arri_alt."""
    n_pairs = len(raw) // 3
    d = np.frombuffer(raw[:n_pairs * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)

    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1

    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


def assemble_frame(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    top    = c1[:CHUNK_PIX].reshape(HALF_H, W)
    bottom = c2[:CHUNK_PIX].reshape(HALF_H, W)
    frame = np.empty((H, W), dtype=np.uint16)
    frame[:HALF_H, :] = top
    frame[HALF_H:, :] = bottom
    return frame


def split_channels(frame: np.ndarray, pattern: str) -> dict:
    tl, tr, bl, br = PATTERNS[pattern]
    return {
        tl: frame[0::2, 0::2],
        tr: frame[0::2, 1::2],
        bl: frame[1::2, 0::2],
        br: frame[1::2, 1::2],
    }


def build_rgb(chans: dict) -> np.ndarray:
    R  = chans["R"].astype(np.float32)
    B  = chans["B"].astype(np.float32)
    G1 = chans["G1"].astype(np.float32)
    G2 = chans["G2"].astype(np.float32)
    G  = 0.5 * (G1 + G2)

    def norm(x):
        p = np.percentile(x, 99.0)
        return np.clip(x / max(p, 1), 0, 1)

    rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ===========================================================================
# ОБРАБОТКА КАДРА
# ===========================================================================

def process_frame(path: Path, pkt: dict, frame_idx: int,
                  pattern: str, out_dir: Path,
                  save_rgb: bool = True) -> None:
    """
    Читает ОДИН essence-пакет по его offset'у и сохраняет 4 TIFF.
    """
    pos = pkt["value_start"]
    size = pkt["length"]

    with open(path, "rb") as f:
        f.seek(pos)
        raw = f.read(size)

    if len(raw) < HDR + 2 * CHUNK_BYTES:
        raise ValueError(
            f"Кадр {frame_idx}: мало данных {len(raw)} < "
            f"{HDR + 2 * CHUNK_BYTES}"
        )

    c1_raw = raw[HDR : HDR + CHUNK_BYTES]
    c2_raw = raw[HDR + CHUNK_BYTES : HDR + 2 * CHUNK_BYTES]

    c1 = unpack_chunk(c1_raw)
    c2 = unpack_chunk(c2_raw)

    frame = assemble_frame(c1, c2)
    chans = split_channels(frame, pattern)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"frame{frame_idx:04d}_{pattern}"

    stats = []
    for name, img in chans.items():
        tifffile.imwrite(f"{base}_{name}.tiff", img)
        stats.append(f"{name}(std={img.std():.0f},"
                     f"mean={img.mean():.0f},"
                     f"max={img.max()})")
    print(f"   ✅ кадр {frame_idx:02d} → {base.name}_*.tiff")
    print(f"      " + "  ".join(stats))

    if save_rgb:
        rgb = build_rgb(chans)
        tifffile.imwrite(f"{base}_RGB.tiff", rgb)
        print(f"      + _RGB.tiff")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--all-frames", action="store_true")
    ap.add_argument("--pattern", default="GBRG", choices=list(PATTERNS))
    ap.add_argument("--no-rgb", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.file) if args.file else Path(MXF_FILE)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        sys.exit(1)

    print(f"📂 {path.name}")

    # Разбираем MXF — получаем список ВСЕХ essence-пакетов
    result = mxf_parser.parse(path)
    essence = result["essence"]
    print(f"   Essence пакетов: {len(essence)}")
    print(f"   Первый offset:   {essence[0]['value_start']:,}")
    print(f"   Размер пакета:   {essence[0]['length']:,}")

    # Проверяем, что пакеты не подряд — покажем для диагностики
    if len(essence) >= 2:
        gap = essence[1]["value_start"] - essence[0]["value_start"]
        print(f"   Шаг между кадрами: {gap:,} "
              f"(размер пакета: {essence[0]['length']:,}, "
              f"разница: {gap - essence[0]['length']:+d})")
    print()

    out_dir = Path(args.out) if args.out else \
              path.parent / f"{path.stem}_channels"

    if args.all_frames:
        frame_list = list(range(len(essence)))
    else:
        if args.frame >= len(essence):
            print(f"❌ Кадр {args.frame} вне диапазона "
                  f"(0..{len(essence) - 1})")
            sys.exit(1)
        frame_list = [args.frame]

    for idx in frame_list:
        try:
            process_frame(path, essence[idx], idx, args.pattern, out_dir,
                          save_rgb=not args.no_rgb)
        except Exception as e:
            print(f"   ❌ кадр {idx}: {e}")

    print(f"\n✅ Готово: {out_dir}")


if __name__ == "__main__":
    main()