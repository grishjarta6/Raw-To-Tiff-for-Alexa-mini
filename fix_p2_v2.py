"""
Поиск правильной формулы p2 при p1=(b0<<4)|(b2>>4), sh=76, pattern=BGGR.

Перебирает:
  - 24 формулы от (b0, b1, b2)
  - 6 перестановок байтов в группе
  - 3 байтовых сдвига (группа начинается с позиции 0, 1 или 2)

Оценка: std каналов G2 и R. Должно быть > 800 для живой картинки.
"""

import argparse
import numpy as np
from pathlib import Path

import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
ROWS = 500


# ---------------------------------------------------------------------------
# Все 24 формулы
# ---------------------------------------------------------------------------

FORMULAS = [
    ("b0[7:0]|b1[7:4]",  lambda b0,b1,b2: (b0<<4)|(b1>>4)),
    ("b0[7:0]|b1[3:0]",  lambda b0,b1,b2: (b0<<4)|(b1&0x0F)),
    ("b0[7:0]|b2[7:4]",  lambda b0,b1,b2: (b0<<4)|(b2>>4)),
    ("b0[7:0]|b2[3:0]",  lambda b0,b1,b2: (b0<<4)|(b2&0x0F)),
    ("b1[7:0]|b0[7:4]",  lambda b0,b1,b2: (b1<<4)|(b0>>4)),
    ("b1[7:0]|b0[3:0]",  lambda b0,b1,b2: (b1<<4)|(b0&0x0F)),
    ("b1[7:0]|b2[7:4]",  lambda b0,b1,b2: (b1<<4)|(b2>>4)),
    ("b1[7:0]|b2[3:0]",  lambda b0,b1,b2: (b1<<4)|(b2&0x0F)),
    ("b2[7:0]|b0[7:4]",  lambda b0,b1,b2: (b2<<4)|(b0>>4)),
    ("b2[7:0]|b0[3:0]",  lambda b0,b1,b2: (b2<<4)|(b0&0x0F)),
    ("b2[7:0]|b1[7:4]",  lambda b0,b1,b2: (b2<<4)|(b1>>4)),
    ("b2[7:0]|b1[3:0]",  lambda b0,b1,b2: (b2<<4)|(b1&0x0F)),
    ("b0[3:0]|b1[7:0]",  lambda b0,b1,b2: ((b0&0x0F)<<8)|b1),
    ("b0[3:0]|b2[7:0]",  lambda b0,b1,b2: ((b0&0x0F)<<8)|b2),
    ("b1[3:0]|b0[7:0]",  lambda b0,b1,b2: ((b1&0x0F)<<8)|b0),
    ("b1[3:0]|b2[7:0]",  lambda b0,b1,b2: ((b1&0x0F)<<8)|b2),
    ("b2[3:0]|b0[7:0]",  lambda b0,b1,b2: ((b2&0x0F)<<8)|b0),
    ("b2[3:0]|b1[7:0]",  lambda b0,b1,b2: ((b2&0x0F)<<8)|b1),
    ("b0[7:4]|b1[7:0]",  lambda b0,b1,b2: ((b0>>4)<<8)|b1),
    ("b0[7:4]|b2[7:0]",  lambda b0,b1,b2: ((b0>>4)<<8)|b2),
    ("b1[7:4]|b0[7:0]",  lambda b0,b1,b2: ((b1>>4)<<8)|b0),
    ("b1[7:4]|b2[7:0]",  lambda b0,b1,b2: ((b1>>4)<<8)|b2),
    ("b2[7:4]|b0[7:0]",  lambda b0,b1,b2: ((b2>>4)<<8)|b0),
    ("b2[7:4]|b1[7:0]",  lambda b0,b1,b2: ((b2>>4)<<8)|b1),
]


# ---------------------------------------------------------------------------
# Перестановки байтов
# ---------------------------------------------------------------------------

PERMUTATIONS = [
    ("b0,b1,b2", (0, 1, 2)),
    ("b0,b2,b1", (0, 2, 1)),
    ("b1,b0,b2", (1, 0, 2)),
    ("b1,b2,b0", (1, 2, 0)),
    ("b2,b0,b1", (2, 0, 1)),
    ("b2,b1,b0", (2, 1, 0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sh", type=int, default=76)
    ap.add_argument("--rows", type=int, default=ROWS)
    args = ap.parse_args()

    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]

    print(f"Essence offset: {off:,}")
    print(f"sh:             {args.sh}")
    print(f"rows:           {args.rows}")

    # Все данные строк
    n_bytes = args.rows * W * 3 // 2
    with open(path, "rb") as f:
        f.seek(off + args.sh)
        raw = f.read(n_bytes + 3)

    total_pixels = args.rows * W

    # p1 фиксирована = big_endian
    d1 = np.frombuffer(raw[:n_bytes], dtype=np.uint8)
    b0_1 = d1[0::3].astype(np.uint16)
    b1_1 = d1[1::3].astype(np.uint16)
    b2_1 = d1[2::3].astype(np.uint16)
    p1 = (b0_1 << 4) | (b2_1 >> 4)

    # Собираем bayer с p1 на чётных позициях
    bayer_base = np.empty(total_pixels, dtype=np.uint16)
    bayer_base[0::2] = p1[:total_pixels // 2]

    print()
    print(f"{'формула':<20} {'perm':<10} {'shift':>5} | "
          f"{'B_std':>7} {'G2_std':>7} {'G1_std':>7} {'R_std':>7} | "
          f"min(G2,R)")
    print("-" * 95)

    results = []
    for shift in range(3):
        d = np.frombuffer(raw[shift:shift + n_bytes], dtype=np.uint8)
        ba = d[0::3].astype(np.uint16)
        bb = d[1::3].astype(np.uint16)
        bc = d[2::3].astype(np.uint16)

        for perm_name, (i, j, k) in PERMUTATIONS:
            b0, b1, b2 = [ba, bb, bc][i], [ba, bb, bc][j], [ba, bb, bc][k]

            for f_name, fn in FORMULAS:
                p2 = fn(b0, b1, b2)

                bayer = bayer_base.copy()
                bayer[1::2] = p2[:total_pixels // 2]
                bayer = bayer.reshape(args.rows, W)

                B  = bayer[0::2, 0::2]
                G2 = bayer[0::2, 1::2]
                G1 = bayer[1::2, 0::2]
                R  = bayer[1::2, 1::2]

                Bs, G2s, G1s, Rs = (float(B.std()), float(G2.std()),
                                    float(G1.std()), float(R.std()))
                mn = min(G2s, Rs)
                results.append({
                    "f_name": f_name, "perm": perm_name, "shift": shift,
                    "Bs": Bs, "G2s": G2s, "G1s": G1s, "Rs": Rs,
                    "min": mn,
                })

    results.sort(key=lambda x: x["min"], reverse=True)

    for r in results[:30]:
        marker = " ★" if r["min"] > 700 else "  "
        print(f"{marker} {r['f_name']:<18} {r['perm']:<10} "
              f"{r['shift']:>5} | "
              f"{r['Bs']:>7.0f} {r['G2s']:>7.0f} "
              f"{r['G1s']:>7.0f} {r['Rs']:>7.0f} | "
              f"{r['min']:>7.0f}")

    best = results[0]
    print()
    print(f"✅ Лучшая p2: {best['f_name']} | perm={best['perm']} "
          f"| shift={best['shift']}")
    print(f"   std: B={best['Bs']:.0f}, G2={best['G2s']:.0f}, "
          f"G1={best['G1s']:.0f}, R={best['Rs']:.0f}")


if __name__ == "__main__":
    main()