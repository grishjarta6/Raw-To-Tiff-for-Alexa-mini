#!/usr/bin/env python3
"""
ARRIRAW ALEXA Mini MXF → 4 × 16-bit TIFF.

Структура essence element (11 309 548 байт):
    [  76 байт  ]  frame header
    [ 5 654 736 ]  p1-поток: пиксели 0, 2, 4, ... (B, G1 при BGGR)
    [ 5 654 736 ]  p2-поток: пиксели 1, 3, 5, ... (G2, R при BGGR)

Каждый поток: 12-бит, 3 байта → 2 пикселя.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
DEFAULT_HEADER = 76


# ---------------------------------------------------------------------------
# Формулы распаковки 12-бит: 3 байта → 2 пикселя
# ---------------------------------------------------------------------------

def unpack_std(b0, b1, b2):
    """Стандарт SMPTE RDD 30. p0 = b0[7:0] b1[7:4], p1 = b1[3:0] b2[7:0]"""
    return (b0 << 4) | (b1 >> 4), ((b1 & 0x0F) << 8) | b2

def unpack_nibble_swap(b0, b1, b2):
    return (b0 << 4) | (b1 & 0x0F), ((b1 >> 4) << 8) | b2

def unpack_big_endian(b0, b1, b2):
    """p0 = b0[7:0] b2[7:4], p1 = b1[7:0] b2[3:0]"""
    return (b0 << 4) | (b2 >> 4), (b1 << 4) | (b2 & 0x0F)

def unpack_little_endian(b0, b1, b2):
    return ((b1 & 0x0F) << 8) | b0, ((b1 >> 4) << 8) | b2

UNPACK_MODES = {
    "std":           unpack_std,
    "nibble_swap":   unpack_nibble_swap,
    "big_endian":    unpack_big_endian,
    "little_endian": unpack_little_endian,
}


PATTERNS = {
    # (top-left, top-right, bottom-left, bottom-right)
    "BGGR": ("B",  "G2", "G1", "R"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


# ---------------------------------------------------------------------------
# Распаковка потока
# ---------------------------------------------------------------------------

def unpack_stream(raw, mode):
    """12-бит поток (3 байта → 2 пикселя) → flat uint16."""
    n = (len(raw) // 3) * 3
    d = np.frombuffer(raw[:n], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)

    pe, po = UNPACK_MODES[mode](b0, b1, b2)

    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = pe
    out[1::2] = po
    return out


def build_bayer(pix1, pix2, w, h):
    """
    Собирает Bayer из двух потоков:
      pix1 → пиксели с чётными линейными индексами → чётные столбцы
      pix2 → пиксели с нечётными индексами        → нечётные столбцы
    """
    half_w = w // 2
    n = h * half_w
    bayer = np.empty((h, w), dtype=np.uint16)
    bayer[:, 0::2] = pix1[:n].reshape(h, half_w)
    bayer[:, 1::2] = pix2[:n].reshape(h, half_w)
    return bayer


def extract_channels(bayer, pattern):
    tl, tr, bl, br = PATTERNS[pattern]
    return {
        tl: bayer[0::2, 0::2],
        tr: bayer[0::2, 1::2],
        bl: bayer[1::2, 0::2],
        br: bayer[1::2, 1::2],
    }


def stats(ch):
    return {"min": int(ch.min()), "max": int(ch.max()),
            "mean": float(ch.mean()), "std": float(ch.std())}


# ---------------------------------------------------------------------------
# Оценка качества
# ---------------------------------------------------------------------------

def score_bayer(bayer, pattern):
    """
    Возвращает {'noise': ..., 'corr': ...} или None если конфиг плохой.
    Меньше noise = лучше. corr — корреляция G1↔G2.
    """
    B  = bayer[0::2, 0::2].astype(np.int32)
    G2 = bayer[0::2, 1::2].astype(np.int32)
    G1 = bayer[1::2, 0::2].astype(np.int32)
    R  = bayer[1::2, 1::2].astype(np.int32)
    chans = [B, G2, G1, R]

    # 1. Все каналы должны быть "живыми"
    for ch in chans:
        if ch.std() < 200:
            return None

    # 2. Отсечение дубликатов (формулы, дающие один и тот же пиксель)
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.abs(chans[i] - chans[j]).mean()) < 2.0:
                return None

    # 3. Корреляция G1 ↔ G2 (это один и тот же зелёный)
    g1 = G1.flatten().astype(np.float32)
    g2 = G2.flatten().astype(np.float32)
    corr = float(np.corrcoef(g1, g2)[0, 1])
    if corr < 0.3:
        return None

    # 4. Шум: средний градиент / std
    noise = 0.0
    for ch in chans:
        d = float(np.abs(np.diff(ch, axis=1)).mean())
        noise += d / float(ch.std())
    noise /= 4.0

    return {"noise": noise, "corr": corr}


# ---------------------------------------------------------------------------
# Обработка одного файла
# ---------------------------------------------------------------------------

def process(header=76, mode="std", pattern="BGGR", swap=False, out_dir=None):
    path = Path(MXF_FILE)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        sys.exit(1)

    print(f"📂 {path.name}")

    result = mxf_parser.parse(path)
    off = result["header_offset"]
    essence_size = result["frame_size"]

    half_pixels = W * H // 2
    half_bytes = half_pixels * 3 // 2
    expected = header + 2 * half_bytes

    print(f"   Essence offset:      {off:,}")
    print(f"   Essence size:        {essence_size:,}")
    print(f"   Ожидаемый размер:    {expected:,}  "
          f"(diff: {essence_size - expected:+d})")
    print(f"   Разрешение:          {W}×{H}")
    print(f"   Header:              {header}")
    print(f"   Mode:                {mode}")
    print(f"   Pattern:             {pattern}")
    print(f"   Swap streams:        {swap}")

    if expected != essence_size:
        print(f"   ⚠ Размер не совпал — возможно, header не {header}")

    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(essence_size)

    p1_raw = raw[header : header + half_bytes]
    p2_raw = raw[header + half_bytes : header + 2 * half_bytes]

    print(f"   p1: raw[{header} : {header + half_bytes}]")
    print(f"   p2: raw[{header + half_bytes} : {header + 2 * half_bytes}]")

    pix1 = unpack_stream(p1_raw, mode)
    pix2 = unpack_stream(p2_raw, mode)

    if swap:
        pix1, pix2 = pix2, pix1

    bayer = build_bayer(pix1, pix2, W, H)
    chans = extract_channels(bayer, pattern)

    print()
    for name, ch in chans.items():
        s = stats(ch)
        ok = "✅" if s["std"] > 200 else "⚠"
        print(f"   {ok} {name:<3}  min={s['min']:>5}  max={s['max']:>5}  "
              f"mean={s['mean']:>8.1f}  std={s['std']:>7.1f}")

    sc = score_bayer(bayer, pattern)
    if sc:
        print(f"\n   🎯 score={sc['noise']:.4f}   corr(G1,G2)={sc['corr']:+.4f}")
        print(f"      (score < 0.5 — отлично, 0.5..1.0 — средне, >1.0 — плохо)")
    else:
        print(f"\n   ⚠ Конфиг не прошёл фильтр качества")

    if out_dir is None:
        out_dir = path.parent / f"{path.stem}_channels"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, img in chans.items():
        tifffile.imwrite(str(out_dir / f"{path.stem}_{name}.tiff"), img)

    print(f"\n   ✅ Сохранено в {out_dir}")


# ---------------------------------------------------------------------------
# Brute-force
# ---------------------------------------------------------------------------

def brute(hdr_range, sample_rows=300):
    path = Path(MXF_FILE)
    result = mxf_parser.parse(path)
    off = result["header_offset"]

    sample_pixels = W * sample_rows
    sample_half_pixels = sample_pixels // 2
    sample_half_bytes = sample_half_pixels * 3 // 2

    max_hdr = max(hdr_range)
    need = max_hdr + 2 * sample_half_bytes
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(need)

    total = len(hdr_range) * len(UNPACK_MODES) * 2 * len(PATTERNS)
    print(f"Brute-force: {total} комбинаций")
    print(f"   header ∈ [{min(hdr_range)}..{max_hdr}]  ({len(hdr_range)} значений)")
    print(f"   modes    = {list(UNPACK_MODES)}")
    print(f"   patterns = {list(PATTERNS)}")
    print(f"   swap ∈ {{False, True}}")
    print(f"   sample: {sample_rows} строк, "
          f"{sample_half_bytes:,} байт на поток")
    print()

    half_w = W // 2
    results = []
    for hdr in hdr_range:
        p1_raw = raw[hdr : hdr + sample_half_bytes]
        p2_raw = raw[hdr + sample_half_bytes : hdr + 2 * sample_half_bytes]
        if len(p1_raw) < sample_half_bytes or len(p2_raw) < sample_half_bytes:
            continue
        for mode in UNPACK_MODES:
            try:
                p1 = unpack_stream(p1_raw, mode)
                p2 = unpack_stream(p2_raw, mode)
            except Exception:
                continue
            for swap in (False, True):
                p1x, p2x = (p2, p1) if swap else (p1, p2)
                bayer = np.empty((sample_rows, W), dtype=np.uint16)
                bayer[:, 0::2] = p1x[: sample_rows * half_w].reshape(sample_rows, half_w)
                bayer[:, 1::2] = p2x[: sample_rows * half_w].reshape(sample_rows, half_w)
                for pattern in PATTERNS:
                    sc = score_bayer(bayer, pattern)
                    if sc is None:
                        continue
                    results.append({
                        "header": hdr, "mode": mode, "pattern": pattern,
                        "swap": swap, "score": sc["noise"],
                        "corr": sc["corr"],
                    })
        if (hdr - min(hdr_range)) % 20 == 0:
            print(f"   ... header={hdr}, кандидатов {len(results)}")

    if not results:
        print("❌ Ни одна конфигурация не прошла фильтр.")
        print("   Возможные причины:")
        print("   - файл повреждён или обрезан")
        print("   - разрешение не 3424×2202")
        print("   - структура essence не двухпоточная")
        return

    results.sort(key=lambda x: x["score"])

    print(f"\n{'='*72}")
    print(f"ТОП-20 (score = средний noise_ratio, меньше = лучше)")
    print(f"{'='*72}")
    print(f"   #   header   mode              pattern  swap    score    corr")
    print("   " + "-" * 68)
    for i, r in enumerate(results[:20], 1):
        print(f"   {i:>2}   {r['header']:>5}   {r['mode']:<15}   "
              f"{r['pattern']:<6}   {str(r['swap']):<5}   "
              f"{r['score']:.4f}   {r['corr']:+.4f}")

    best = results[0]
    print(f"\n{'='*72}")
    print(f"🎯 РЕКОМЕНДАЦИЯ")
    print(f"{'='*72}")
    print(f"   uv run python main.py --header {best['header']} "
          f"--mode {best['mode']} --pattern {best['pattern']}"
          f"{' --swap' if best['swap'] else ''}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--header", type=int, default=DEFAULT_HEADER)
    ap.add_argument("--mode", default="std", choices=list(UNPACK_MODES))
    ap.add_argument("--pattern", default="BGGR", choices=list(PATTERNS))
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--brute", action="store_true")
    ap.add_argument("--hdr-range", nargs=2, type=int, default=[0, 128])
    ap.add_argument("--sample-rows", type=int, default=300)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.brute:
        brute(range(args.hdr_range[0], args.hdr_range[1]),
              sample_rows=args.sample_rows)
        return

    process(header=args.header, mode=args.mode, pattern=args.pattern,
            swap=args.swap, out_dir=args.out)


if __name__ == "__main__":
    main()