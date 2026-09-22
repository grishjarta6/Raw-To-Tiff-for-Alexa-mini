"""
Проверка гипотезы: b0 = low-нибблы двух пикселей, b1 и b2 = старшие байты.

Перебираем:
  - 2 порядка нибблов в b0 (low4→p1 или low4→p2)
  - 4 паттерна Bayer
  - offset 0..200
Оценка: min std четырёх каналов (BGGR).
"""

import numpy as np
from pathlib import Path
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
ROWS = 500


PATTERNS = {
    "BGGR": ("B",  "G2", "G1", "R"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


def evaluate(bayer, pattern):
    tl, tr, bl, br = PATTERNS[pattern]
    chans = {
        tl: bayer[0::2, 0::2].astype(np.int32),
        tr: bayer[0::2, 1::2].astype(np.int32),
        bl: bayer[1::2, 0::2].astype(np.int32),
        br: bayer[1::2, 1::2].astype(np.int32),
    }
    stds = {k: float(v.std()) for k, v in chans.items()}
    if min(stds.values()) < 100:
        return None
    # Отсекаем дубликаты
    keys = list(chans)
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.abs(chans[keys[i]] - chans[keys[j]]).mean()) < 2.0:
                return None
    return stds


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]

    total_pixels = ROWS * W
    half = total_pixels // 2
    n_bytes = total_pixels * 3 // 2

    print(f"Essence offset: {off:,}")
    print(f"rows: {ROWS}   n_bytes: {n_bytes:,}")
    print()

    max_sh = 200
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(max_sh + n_bytes + 16)

    # Результаты: (min_std, sh, variant, pattern, stds)
    results = []

    for sh in range(max_sh):
        s = raw[sh : sh + n_bytes]
        if len(s) < n_bytes:
            break
        d = np.frombuffer(s, dtype=np.uint8)
        b0 = d[0::3].astype(np.uint16)
        b1 = d[1::3].astype(np.uint16)
        b2 = d[2::3].astype(np.uint16)

        # Вариант A: p1 = b1[7:0] | b0[7:4], p2 = b2[7:0] | b0[3:0]
        p1A = (b1 << 4) | (b0 >> 4)
        p2A = (b2 << 4) | (b0 & 0x0F)

        # Вариант B: p1 = b1[7:0] | b0[3:0], p2 = b2[7:0] | b0[7:4]
        p1B = (b1 << 4) | (b0 & 0x0F)
        p2B = (b2 << 4) | (b0 >> 4)

        variants = {"A": (p1A, p2A), "B": (p1B, p2B)}

        for vname, (p1, p2) in variants.items():
            bayer = np.empty(total_pixels, dtype=np.uint16)
            bayer[0::2] = p1[:half]
            bayer[1::2] = p2[:half]
            bayer = bayer.reshape(ROWS, W)

            for pat in PATTERNS:
                st = evaluate(bayer, pat)
                if st is None:
                    continue
                mn = min(st.values())
                results.append((mn, sh, vname, pat, st))

    if not results:
        print("❌ Ни одна комбинация не прошла фильтр.")
        return

    results.sort(key=lambda r: r[0], reverse=True)

    print(f"{'='*90}")
    print(f"ТОП-20 (min_std — БОЛЬШЕ = ЛУЧШЕ)")
    print(f"{'='*90}")
    print(f"   #  sh   variant  pattern   min_std   "
          f"B_std  G2_std  G1_std  R_std")
    print("   " + "-" * 86)
    for i, (mn, sh, v, pat, st) in enumerate(results[:20], 1):
        tl, tr, bl, br = PATTERNS[pat]
        print(f"   {i:>2} {sh:>4}   {v:<7}  {pat:<6}   {mn:>8.1f}   "
              f"{st[tl]:>6.0f} {st[tr]:>6.0f} {st[bl]:>6.0f} {st[br]:>6.0f}")

    best = results[0]
    mn, sh, v, pat, st = best
    print()
    print(f"🏆 ЛУЧШЕЕ: sh={sh}, variant={v}, pattern={pat}, min_std={mn:.1f}")
    print(f"   stds: B={st['B']:.0f}  G2={st['G2']:.0f}  "
          f"G1={st['G1']:.0f}  R={st['R']:.0f}")

    if mn > 800:
        print(f"\n✅ Все 4 канала живые. Гипотеза верна!")
    else:
        print(f"\n⚠ min_std < 800. Возможно, нужен другой offset или другая модель.")


if __name__ == "__main__":
    main()