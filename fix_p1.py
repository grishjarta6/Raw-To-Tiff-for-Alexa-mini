"""
Дано: sh=76, p2 = b2[7:0]|b1[3:0]  (formula 17) — даёт живые G2 и R.
Ищем: p1, при которой B_std и G1_std > 800 (чтобы B и G1 тоже стали живыми).

Дополнительно перебираем небольшой сдвиг p1 (0..200) — потому что
p1 и p2 могут начинаться с разных позиций в essence.
"""

import numpy as np
from pathlib import Path
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
SH_P2 = 76           # где начинается p2 (уже найдено)
SH_P1_RANGE = range(0, 200)   # где может начинаться p1
ROWS = 500


FORMULAS = [
    ("b0[7:0]|b1[7:4]",    lambda b0,b1,b2: (b0<<4)|(b1>>4)),
    ("b0[7:0]|b1[3:0]",    lambda b0,b1,b2: (b0<<4)|(b1&0x0F)),
    ("b0[7:0]|b2[7:4]",    lambda b0,b1,b2: (b0<<4)|(b2>>4)),
    ("b0[7:0]|b2[3:0]",    lambda b0,b1,b2: (b0<<4)|(b2&0x0F)),
    ("b1[7:0]|b0[7:4]",    lambda b0,b1,b2: (b1<<4)|(b0>>4)),
    ("b1[7:0]|b0[3:0]",    lambda b0,b1,b2: (b1<<4)|(b0&0x0F)),
    ("b1[7:0]|b2[7:4]",    lambda b0,b1,b2: (b1<<4)|(b2>>4)),
    ("b1[7:0]|b2[3:0]",    lambda b0,b1,b2: (b1<<4)|(b2&0x0F)),
    ("b2[7:0]|b0[7:4]",    lambda b0,b1,b2: (b2<<4)|(b0>>4)),
    ("b2[7:0]|b0[3:0]",    lambda b0,b1,b2: (b2<<4)|(b0&0x0F)),
    ("b2[7:0]|b1[7:4]",    lambda b0,b1,b2: (b2<<4)|(b1>>4)),
    ("b2[7:0]|b1[3:0]",    lambda b0,b1,b2: (b2<<4)|(b1&0x0F)),
    ("b0[3:0]|b1[7:0]",    lambda b0,b1,b2: ((b0&0x0F)<<8)|b1),
    ("b0[3:0]|b2[7:0]",    lambda b0,b1,b2: ((b0&0x0F)<<8)|b2),
    ("b1[3:0]|b0[7:0]",    lambda b0,b1,b2: ((b1&0x0F)<<8)|b0),
    ("b1[3:0]|b2[7:0]",    lambda b0,b1,b2: ((b1&0x0F)<<8)|b2),
    ("b2[3:0]|b0[7:0]",    lambda b0,b1,b2: ((b2&0x0F)<<8)|b0),
    ("b2[3:0]|b1[7:0]",    lambda b0,b1,b2: ((b2&0x0F)<<8)|b1),
    ("b0[7:4]|b1[7:0]",    lambda b0,b1,b2: ((b0>>4)<<8)|b1),
    ("b0[7:4]|b2[7:0]",    lambda b0,b1,b2: ((b0>>4)<<8)|b2),
    ("b1[7:4]|b0[7:0]",    lambda b0,b1,b2: ((b1>>4)<<8)|b0),
    ("b1[7:4]|b2[7:0]",    lambda b0,b1,b2: ((b1>>4)<<8)|b2),
    ("b2[7:4]|b0[7:0]",    lambda b0,b1,b2: ((b2>>4)<<8)|b0),
    ("b2[7:4]|b1[7:0]",    lambda b0,b1,b2: ((b2>>4)<<8)|b1),
]


def main():
    path = Path(MXF_FILE)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        return

    off = mxf_parser.parse(path)["header_offset"]

    total_pixels = ROWS * W
    half = total_pixels // 2
    n_bytes = total_pixels * 3 // 2

    print(f"Файл:      {path.name}")
    print(f"Offset:    {off:,}")
    print(f"Фиксируем: sh_p2={SH_P2}, p2=b2[7:0]|b1[3:0] (formula 17)")
    print(f"Ищем:      p1 и sh_p1 из диапазона {min(SH_P1_RANGE)}..{max(SH_P1_RANGE)}")
    print()

    # Читаем сырые данные с запасом
    max_sh = max(SH_P2, max(SH_P1_RANGE))
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(max_sh + n_bytes + 16)

    # --- p2 фиксирована на sh=76
    s2 = raw[SH_P2 : SH_P2 + n_bytes]
    d2 = np.frombuffer(s2, dtype=np.uint8)
    b0_2 = d2[0::3].astype(np.uint16)
    b1_2 = d2[1::3].astype(np.uint16)
    b2_2 = d2[2::3].astype(np.uint16)
    p2 = ((b2_2 & 0x0F) << 8) | b1_2

    print(f"[A] p2 = b2[7:0]|b1[3:0] при sh={SH_P2}:")
    print(f"    len(p2) = {len(p2):,}")

    # Проверим, что p2 действительно даёт живые каналы
    bayer_p2 = np.zeros(total_pixels, dtype=np.uint16)
    bayer_p2[1::2] = p2[:half]
    bayer_p2 = bayer_p2.reshape(ROWS, W)
    G2_std = bayer_p2[0::2, 1::2].std()
    R_std  = bayer_p2[1::2, 1::2].std()
    print(f"    G2_std = {G2_std:.1f}   R_std = {R_std:.1f}")
    if G2_std < 500 or R_std < 500:
        print(f"    ⚠ p2 работает плохо. Проверьте SH_P2.")
        return
    print(f"    ✅ p2 работает — G2 и R живые.\n")

    # --- Перебираем p1 и её сдвиг
    print(f"[B] Перебор p1 × sh_p1 ({len(SH_P1_RANGE)} × 24 = "
          f"{len(SH_P1_RANGE)*24} комбинаций):")
    print(f"    {'sh':>4} {'формула p1':<22} {'B_std':>8} {'G1_std':>8}  "
          f"вердикт")
    print("    " + "-" * 62)

    results = []
    for sh1 in SH_P1_RANGE:
        s1 = raw[sh1 : sh1 + n_bytes]
        if len(s1) < n_bytes:
            break
        d1 = np.frombuffer(s1, dtype=np.uint8)
        a0 = d1[0::3].astype(np.uint16)
        a1 = d1[1::3].astype(np.uint16)
        a2 = d1[2::3].astype(np.uint16)

        for name, fn in FORMULAS:
            p1 = fn(a0, a1, a2)
            bayer = np.zeros(total_pixels, dtype=np.uint16)
            bayer[0::2] = p1[:half]
            bayer[1::2] = p2[:half]
            bayer = bayer.reshape(ROWS, W)

            B_std  = bayer[0::2, 0::2].std()
            G1_std = bayer[1::2, 0::2].std()

            ok = (B_std > 800 and G1_std > 800)
            results.append((ok, sh1, name, B_std, G1_std))

    # Печатаем только топ (или все хорошие + топ плохих)
    winners = [r for r in results if r[0]]
    if winners:
        print(f"\n    ✅ Найдено {len(winners)} рабочих комбинаций:")
        print(f"    {'sh':>4} {'формула p1':<22} {'B_std':>8} {'G1_std':>8}")
        for ok, sh1, name, B, G1 in winners[:20]:
            print(f"    {sh1:>4} {name:<22} {B:>8.1f} {G1:>8.1f}")

        # Финальная рекомендация
        print(f"\n{'='*72}")
        print(f"🏆 ФИНАЛЬНЫЕ ПАРАМЕТРЫ:")
        best = winners[0]
        print(f"   sh_p1 = {best[1]}")
        print(f"   p1    = {best[2]}")
        print(f"   sh_p2 = {SH_P2}")
        print(f"   p2    = b2[7:0]|b1[3:0]")
        print(f"   B_std = {best[3]:.1f}   G1_std = {best[4]:.1f}")
        print(f"   G2_std = {G2_std:.1f}  R_std = {R_std:.1f}")
        print(f"{'='*72}")
    else:
        print(f"\n    ⚠ Рабочих комбинаций не найдено. Топ-10 по B_std:")
        results.sort(key=lambda r: (r[3] + r[4]), reverse=True)
        print(f"    {'sh':>4} {'формула p1':<22} {'B_std':>8} {'G1_std':>8}")
        for ok, sh1, name, B, G1 in results[:10]:
            print(f"    {sh1:>4} {name:<22} {B:>8.1f} {G1:>8.1f}")


if __name__ == "__main__":
    main()