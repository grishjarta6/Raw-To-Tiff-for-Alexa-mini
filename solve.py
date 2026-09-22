"""
Полный перебор пар формул (f1, f2) для распаковки 12-бит ARRIRAW.

Модель: линейный поток 12-бит, 3 байта → 2 пикселя.
    пиксель 2k   ← f1(b0[k], b1[k], b2[k])
    пиксель 2k+1 ← f2(b0[k], b1[k], b2[k])

Перебираем offset ∈ [0..N], f1 ∈ 24, f2 ∈ 24.
Оцениваем std четырёх каналов Bayer (BGGR) + отсекаем мусор.
"""

import argparse
from pathlib import Path

import numpy as np

import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202


FORMULAS = [
    ("b0[7:0]|b1[7:4]",   lambda b0,b1,b2: (b0<<4)|(b1>>4)),
    ("b0[7:0]|b1[3:0]",   lambda b0,b1,b2: (b0<<4)|(b1&0x0F)),
    ("b0[7:0]|b2[7:4]",   lambda b0,b1,b2: (b0<<4)|(b2>>4)),
    ("b0[7:0]|b2[3:0]",   lambda b0,b1,b2: (b0<<4)|(b2&0x0F)),
    ("b1[7:0]|b0[7:4]",   lambda b0,b1,b2: (b1<<4)|(b0>>4)),
    ("b1[7:0]|b0[3:0]",   lambda b0,b1,b2: (b1<<4)|(b0&0x0F)),
    ("b1[7:0]|b2[7:4]",   lambda b0,b1,b2: (b1<<4)|(b2>>4)),
    ("b1[7:0]|b2[3:0]",   lambda b0,b1,b2: (b1<<4)|(b2&0x0F)),
    ("b2[7:0]|b0[7:4]",   lambda b0,b1,b2: (b2<<4)|(b0>>4)),
    ("b2[7:0]|b0[3:0]",   lambda b0,b1,b2: (b2<<4)|(b0&0x0F)),
    ("b2[7:0]|b1[7:4]",   lambda b0,b1,b2: (b2<<4)|(b1>>4)),
    ("b2[7:0]|b1[3:0]",   lambda b0,b1,b2: (b2<<4)|(b1&0x0F)),
    ("b0[3:0]|b1[7:0]",   lambda b0,b1,b2: ((b0&0x0F)<<8)|b1),
    ("b0[3:0]|b2[7:0]",   lambda b0,b1,b2: ((b0&0x0F)<<8)|b2),
    ("b1[3:0]|b0[7:0]",   lambda b0,b1,b2: ((b1&0x0F)<<8)|b0),
    ("b1[3:0]|b2[7:0]",   lambda b0,b1,b2: ((b1&0x0F)<<8)|b2),
    ("b2[3:0]|b0[7:0]",   lambda b0,b1,b2: ((b2&0x0F)<<8)|b0),
    ("b2[3:0]|b1[7:0]",   lambda b0,b1,b2: ((b2&0x0F)<<8)|b1),
    ("b0[7:4]|b1[7:0]",   lambda b0,b1,b2: ((b0>>4)<<8)|b1),
    ("b0[7:4]|b2[7:0]",   lambda b0,b1,b2: ((b0>>4)<<8)|b2),
    ("b1[7:4]|b0[7:0]",   lambda b0,b1,b2: ((b1>>4)<<8)|b0),
    ("b1[7:4]|b2[7:0]",   lambda b0,b1,b2: ((b1>>4)<<8)|b2),
    ("b2[7:4]|b0[7:0]",   lambda b0,b1,b2: ((b2>>4)<<8)|b0),
    ("b2[7:4]|b1[7:0]",   lambda b0,b1,b2: ((b2>>4)<<8)|b1),
]


def evaluate(bayer):
    """Оценка конфига. Возвращает dict или None если мусор."""
    B  = bayer[0::2, 0::2].astype(np.int32)
    G2 = bayer[0::2, 1::2].astype(np.int32)
    G1 = bayer[1::2, 0::2].astype(np.int32)
    R  = bayer[1::2, 1::2].astype(np.int32)

    stds = {
        "B":  float(B.std()),
        "G2": float(G2.std()),
        "G1": float(G1.std()),
        "R":  float(R.std()),
    }
    if min(stds.values()) < 100:
        return None

    # Отсекаем дубликаты
    if float(np.abs(B - G2).mean()) < 2.0:
        return None
    if float(np.abs(G1 - R).mean()) < 2.0:
        return None

    g1 = G1.flatten().astype(np.float32)
    g2 = G2.flatten().astype(np.float32)
    corr = float(np.corrcoef(g1, g2)[0, 1])

    return {"stds": stds, "corr": corr,
            "min_std": min(stds.values())}


def solve(off_range, rows):
    path = Path(MXF_FILE)
    res = mxf_parser.parse(path)
    off = res["header_offset"]

    print(f"Essence offset: {off:,}")
    print(f"Essence size:   {res['frame_size']:,}")
    print(f"Разрешение:     {W}×{H}")

    total_pixels = rows * W
    n_bytes = total_pixels * 3 // 2        # длина одного 12-бит потока
    half = total_pixels // 2               # сколько пикселей каждого типа

    print(f"Sample:         {rows} строк  ({total_pixels:,} пикселей)")
    print(f"n_bytes:        {n_bytes:,}")
    print(f"half:           {half:,}")

    max_off = max(off_range)
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(max_off + n_bytes + 16)

    print(f"Offset-диапазон: {len(off_range)} значений "
          f"[{min(off_range)}..{max_off}]")
    print(f"Всего комбинаций: {len(off_range)} × 24 × 23 = "
          f"{len(off_range)*24*23}")
    print()

    results = []

    for oi, o in enumerate(off_range):
        slice_ = raw[o : o + n_bytes]
        if len(slice_) < n_bytes:
            break

        d = np.frombuffer(slice_, dtype=np.uint8)
        b0 = d[0::3].astype(np.uint16)
        b1 = d[1::3].astype(np.uint16)
        b2 = d[2::3].astype(np.uint16)

        # 24 предвычисленных массива
        p_all = [fn(b0, b1, b2) for _, fn in FORMULAS]
        # каждый длиной ~half (но может отличаться ±1 из-за floor)

        # Приводим все к одной длине
        L = min(len(p) for p in p_all)
        p_all = [p[:L] for p in p_all]

        for i in range(24):
            pe = p_all[i]
            for j in range(24):
                if i == j:
                    continue
                po = p_all[j]

                flat = np.empty(L * 2, dtype=np.uint16)
                flat[0::2] = pe
                flat[1::2] = po

                # Обрезаем/дополняем до total_pixels
                n_use = min(len(flat), total_pixels)
                flat = flat[:n_use]
                if n_use < total_pixels:
                    continue

                bayer = flat.reshape(rows, W)
                ev = evaluate(bayer)
                if ev is None:
                    continue
                results.append({
                    "off": o, "p1": i, "p2": j,
                    "min_std": ev["min_std"],
                    "corr": ev["corr"],
                    "stds": ev["stds"],
                })

        if (oi + 1) % 20 == 0:
            print(f"   [{oi+1}/{len(off_range)}] offset={o}: "
                  f"кандидатов {len(results)}")

    if not results:
        print("\n❌ Ни одна комбинация не прошла фильтр.")
        return

    results.sort(key=lambda x: x["min_std"], reverse=True)

    print(f"\n{'='*104}")
    print(f"ТОП-25 (min_std — БОЛЬШЕ = ЛУЧШЕ)")
    print(f"{'='*104}")
    print(f"   #   off   p1   p2   min_std   corr    "
          f"B_std  G2_std  G1_std  R_std    f1_formula             f2_formula")
    print("   " + "-" * 100)
    for i, r in enumerate(results[:25], 1):
        s = r["stds"]
        print(f"   {i:>2} {r['off']:>5} {r['p1']:>3} {r['p2']:>3} "
              f"{r['min_std']:>8.1f}  {r['corr']:+.3f}  "
              f"{s['B']:>6.0f} {s['G2']:>6.0f} {s['G1']:>6.0f} "
              f"{s['R']:>6.0f}   "
              f"{FORMULAS[r['p1']][0]:<22} {FORMULAS[r['p2']][0]}")
    print(f"{'='*104}")

    best = results[0]
    print(f"\n🏆 ЛУЧШИЙ РЕЗУЛЬТАТ:")
    print(f"   offset = {best['off']}")
    print(f"   p1 = {best['p1']}   ({FORMULAS[best['p1']][0]})")
    print(f"   p2 = {best['p2']}   ({FORMULAS[best['p2']][0]})")
    print(f"   min_std = {best['min_std']:.1f}")
    print(f"   corr(G1,G2) = {best['corr']:+.3f}")
    print(f"   stds: B={best['stds']['B']:.0f}, "
          f"G2={best['stds']['G2']:.0f}, "
          f"G1={best['stds']['G1']:.0f}, "
          f"R={best['stds']['R']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", nargs=2, type=int, default=[0, 200])
    ap.add_argument("--off-step", type=int, default=1)
    ap.add_argument("--rows", type=int, default=500)
    args = ap.parse_args()

    off_range = list(range(args.off[0], args.off[1], args.off_step))
    solve(off_range, args.rows)


if __name__ == "__main__":
    main()