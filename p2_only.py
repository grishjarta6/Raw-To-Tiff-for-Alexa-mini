"""
Поиск правильной p2 при ЖЁСТКО ЗАФИКСИРОВАННОЙ p1 = (b0<<4)|(b2>>4).

Стратегия:
  1. Проверяем, что p1 действительно даёт хорошие B и G1 (B_std, G1_std > 800).
  2. Перебираем все 24 формулы p2 (из той же тройки b0,b1,b2).
  3. Для каждой p2 смотрим std всех 4 каналов BGGR.
  4. Ищем, где G2_std и R_std одновременно > 800,
     и при этом B_std / G1_std ОСТАЛИСЬ высокими.

Ожидаемая модель ARRIRAW:
  биты 0..11 → b0[7:0], b2[7:4]      (это p1, уже найдено)
  биты 12..23 → b2[3:0], b1[7:0]     (это p2, ищем)

  p1 = (b0 << 4) | (b2 >> 4)
  p2 = ((b2 & 0x0F) << 8) | b1       ← кандидат №1, проверим

Но перебираем ВСЕ 24 формулы, чтобы убедиться.
"""

import numpy as np
from pathlib import Path
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
SH = 76
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
    ("b2[3:0]|b1[7:0]",    lambda b0,b1,b2: ((b2&0x0F)<<8)|b1),   # ★ кандидат
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

    total_pixels = ROWS * W          # 500 строк × 3424 = 1 712 000
    half = total_pixels // 2         # 856 000
    n_bytes = total_pixels * 3 // 2  # 2 568 000

    print(f"Файл:     {path.name}")
    print(f"Offset:   {off:,}")
    print(f"sh:       {SH}")
    print(f"rows:     {ROWS}")
    print(f"n_bytes:  {n_bytes:,}")
    print()

    with open(path, "rb") as f:
        f.seek(off + SH)
        raw = f.read(n_bytes + 3)

    d = np.frombuffer(raw[:n_bytes], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)

    # --- Шаг 1. Фиксируем p1 и проверяем её
    p1 = (b0 << 4) | (b2 >> 4)

    test = np.zeros(total_pixels, dtype=np.uint16)
    test[0::2] = p1[:half]
    test = test.reshape(ROWS, W)

    B_check  = test[0::2, 0::2].std()
    G1_check = test[1::2, 0::2].std()

    print(f"[1] Проверка p1 = (b0<<4)|(b2>>4):")
    print(f"    B_std  = {B_check:.1f}")
    print(f"    G1_std = {G1_check:.1f}")
    if B_check < 500 or G1_check < 500:
        print(f"    ❌ p1 не даёт живую картинку на sh={SH}.")
        print(f"       Возможно, sh другой. Пробуйте main.py --sh 0 или --sh 152.")
        return
    print(f"    ✅ p1 работает. B и G1 хорошие.\n")

    # --- Шаг 2. Перебираем p2, сохраняя p1
    print(f"[2] Перебираем p2, p1 не трогаем:")
    print(f"    {'формула p2':<22} {'G2_std':>10} {'R_std':>10} "
          f"{'B_std':>10} {'G1_std':>10}  вердикт")
    print("    " + "-" * 78)

    rows = []
    for name, fn in FORMULAS:
        p2 = fn(b0, b1, b2)

        bayer = np.empty(total_pixels, dtype=np.uint16)
        bayer[0::2] = p1[:half]
        bayer[1::2] = p2[:half]
        bayer = bayer.reshape(ROWS, W)

        Bs  = bayer[0::2, 0::2].std()
        G2s = bayer[0::2, 1::2].std()
        G1s = bayer[1::2, 0::2].std()
        Rs  = bayer[1::2, 1::2].std()

        good = (G2s > 800 and Rs > 800 and Bs > 800 and G1s > 800)
        verdict = "✅ ВСЕ 4" if good else (
            "⚠ только G2/R" if (G2s > 800 and Rs > 800) else ""
        )
        rows.append((name, G2s, Rs, Bs, G1s, good))
        print(f"    {name:<22} {G2s:>10.1f} {Rs:>10.1f} "
              f"{Bs:>10.1f} {G1s:>10.1f}  {verdict}")

    print()
    winners = [r for r in rows if r[5]]
    if winners:
        best = winners[0]
        print(f"🏆 НАЙДЕНО! Формула p2: {best[0]}")
        print(f"    G2_std={best[1]:.1f}  R_std={best[2]:.1f}  "
              f"B_std={best[3]:.1f}  G1_std={best[4]:.1f}")
        print()
        print(f"    Используйте в main.py (с патчем ниже).")
    else:
        print("⚠ Ни одна p2 из 24 не даёт все 4 канала живыми одновременно.")
        best = max(rows, key=lambda r: min(r[1], r[2]))
        print(f"  Лучшая по min(G2,R): {best[0]}")
        print(f"    G2_std={best[1]:.1f}  R_std={best[2]:.1f}  "
              f"B_std={best[3]:.1f}  G1_std={best[4]:.1f}")
        print()
        print("  Это значит, что p2 надо брать НЕ из той же тройки (b0,b1,b2).")
        print("  Скажите — я добавлю поиск p2 со сдвигом по файлу.")


if __name__ == "__main__":
    main()