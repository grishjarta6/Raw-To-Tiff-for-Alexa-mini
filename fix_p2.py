"""
Поиск правильной формулы p2 при зафиксированных p1=2, sh=0, pattern=BGGR.
"""
import numpy as np
from pathlib import Path
import mxf_parser

MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
SH = 0
ROWS = 500


FORMULAS = [
    ("b0[7:0]|b1[7:4]", lambda b0,b1,b2:(b0<<4)|(b1>>4)),
    ("b0[7:0]|b1[3:0]", lambda b0,b1,b2:(b0<<4)|(b1&0x0F)),
    ("b0[7:0]|b2[7:4]", lambda b0,b1,b2:(b0<<4)|(b2>>4)),
    ("b0[7:0]|b2[3:0]", lambda b0,b1,b2:(b0<<4)|(b2&0x0F)),
    ("b1[7:0]|b0[7:4]", lambda b0,b1,b2:(b1<<4)|(b0>>4)),
    ("b1[7:0]|b0[3:0]", lambda b0,b1,b2:(b1<<4)|(b0&0x0F)),
    ("b1[7:0]|b2[7:4]", lambda b0,b1,b2:(b1<<4)|(b2>>4)),
    ("b1[7:0]|b2[3:0]", lambda b0,b1,b2:(b1<<4)|(b2&0x0F)),
    ("b2[7:0]|b0[7:4]", lambda b0,b1,b2:(b2<<4)|(b0>>4)),
    ("b2[7:0]|b0[3:0]", lambda b0,b1,b2:(b2<<4)|(b0&0x0F)),
    ("b2[7:0]|b1[7:4]", lambda b0,b1,b2:(b2<<4)|(b1>>4)),
    ("b2[7:0]|b1[3:0]", lambda b0,b1,b2:(b2<<4)|(b1&0x0F)),
    ("b0[3:0]|b1[7:0]", lambda b0,b1,b2:((b0&0x0F)<<8)|b1),
    ("b0[3:0]|b2[7:0]", lambda b0,b1,b2:((b0&0x0F)<<8)|b2),
    ("b1[3:0]|b0[7:0]", lambda b0,b1,b2:((b1&0x0F)<<8)|b0),
    ("b1[3:0]|b2[7:0]", lambda b0,b1,b2:((b1&0x0F)<<8)|b2),
    ("b2[3:0]|b0[7:0]", lambda b0,b1,b2:((b2&0x0F)<<8)|b0),
    ("b2[3:0]|b1[7:0]", lambda b0,b1,b2:((b2&0x0F)<<8)|b1),
    ("b0[7:4]|b1[7:0]", lambda b0,b1,b2:((b0>>4)<<8)|b1),
    ("b0[7:4]|b2[7:0]", lambda b0,b1,b2:((b0>>4)<<8)|b2),
    ("b1[7:4]|b0[7:0]", lambda b0,b1,b2:((b1>>4)<<8)|b0),
    ("b1[7:4]|b2[7:0]", lambda b0,b1,b2:((b1>>4)<<8)|b2),
    ("b2[7:4]|b0[7:0]", lambda b0,b1,b2:((b2>>4)<<8)|b0),
    ("b2[7:4]|b1[7:0]", lambda b0,b1,b2:((b2>>4)<<8)|b1),
]


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]
    print(f"Essence offset: {off:,}")

    n_bytes = ROWS * W * 3 // 2
    with open(path, "rb") as f:
        f.seek(off + SH)
        raw = f.read(n_bytes)

    d = np.frombuffer(raw, dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)

    # p1 фиксирована = formula 2
    p1 = FORMULAS[2][1](b0, b1, b2)

    total = ROWS * W
    print(f"\nПроверяю все p2 при p1=2, sh=0, pattern=BGGR")
    print(f"Ожидаем: G2 и R должны дать std ~1000+, B/G1 всегда ~1170\n")
    print(f"  p2  формула               B_std   G2_std   G1_std    R_std")
    print("  " + "-" * 65)

    rows = []
    for i2, (name, fn) in enumerate(FORMULAS):
        p2 = fn(b0, b1, b2)
        bayer = np.empty(total, dtype=np.uint16)
        bayer[0::2] = p1[:total // 2]
        bayer[1::2] = p2[:total // 2]
        bayer = bayer.reshape(ROWS, W)

        # BGGR
        B  = bayer[0::2, 0::2]
        G2 = bayer[0::2, 1::2]
        G1 = bayer[1::2, 0::2]
        R  = bayer[1::2, 1::2]

        Bs, G2s, G1s, Rs = B.std(), G2.std(), G1.std(), R.std()
        # Оценка: min из G2_std и R_std (те, что зависят от p2)
        score = min(G2s, Rs)
        rows.append((score, i2, name, Bs, G2s, G1s, Rs))

    rows.sort(reverse=True)
    for score, i2, name, Bs, G2s, G1s, Rs in rows:
        marker = " ★" if score > 500 else "  "
        print(f"{marker} {i2:>2}  {name:<20}  "
              f"{Bs:>7.0f}  {G2s:>7.0f}  {G1s:>7.0f}  {Rs:>7.0f}")

    best = rows[0]
    print(f"\n✅ Лучшая p2: {best[1]} — {best[2]}  "
          f"(min std G2/R = {best[0]:.0f})")


if __name__ == "__main__":
    main()