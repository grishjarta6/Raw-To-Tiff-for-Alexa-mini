"""
Универсальный поиск параметров распаковки ARRIRAW ALEXA Mini.

Ищет по сетке:
  sh ∈ [0..200] × pattern ∈ 4 × p1 ∈ 24 формул × p2 ∈ 24 формул

Метрика: std + noise_ratio + G1↔G2 corr + проверка на дублирование.

В конце печатает топ-30 конфигураций и рекомендует конкретные
команды для запуска main.py.

Использование:
  uv run python search.py                       # дефолтный диапазон
  uv run python search.py --sh 0 200 --rows 30  # явно
  uv run python search.py --sh-near 76          # фокус вокруг 76
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import mxf_parser


# ===========================================================================
# КОНФИГ
# ===========================================================================

MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202


PATTERNS = {
    "GRBG": ("G1", "R",  "B",  "G2"),
    "BGGR": ("B",  "G2", "G1", "R"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


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


# ===========================================================================
# ОЦЕНКА КАЧЕСТВА
# ===========================================================================

def evaluate(bayer, pattern):
    """
    Возвращает dict со score или None, если конфиг плохой.

    score = средний noise_ratio по 4 каналам.
    Меньше = лучше.
    """
    tl, tr, bl, br = PATTERNS[pattern]
    chans = {
        tl: bayer[0::2, 0::2].astype(np.int32),
        tr: bayer[0::2, 1::2].astype(np.int32),
        bl: bayer[1::2, 0::2].astype(np.int32),
        br: bayer[1::2, 1::2].astype(np.int32),
    }

    # 1. std и noise_ratio каждого канала
    stats = {}
    for name, ch in chans.items():
        s = float(ch.std())
        if s < 150:
            return None
        d = float(np.abs(np.diff(ch, axis=1)).mean())
        stats[name] = {"std": s, "noise": d / s}

    # 2. Дублирование: B == G2 или G1 == R
    #    В этих парах каналы должны отличаться (разные цвета).
    if float(np.abs(chans[tl] - chans[tr]).mean()) < 2.0:
        return None
    if float(np.abs(chans[bl] - chans[br]).mean()) < 2.0:
        return None
    if float(np.abs(chans[tl] - chans[bl]).mean()) < 2.0:
        return None
    if float(np.abs(chans[tr] - chans[br]).mean()) < 2.0:
        return None

    # 3. Корреляция G1 ↔ G2 (обе зелёные)
    g1 = chans["G1"].flatten().astype(np.float32)
    g2 = chans["G2"].flatten().astype(np.float32)
    if g1.std() < 150 or g2.std() < 150:
        return None
    corr = float(np.corrcoef(g1, g2)[0, 1])

    score = sum(s["noise"] for s in stats.values()) / 4
    return {
        "score": score,
        "corr": corr,
        "stats": stats,
        "chans": chans,
    }


# ===========================================================================
# ПОИСК
# ===========================================================================

def search_one_sh(b0, b1, b2, total, rows, w):
    """Возвращает список кандидатов для одного sh."""
    p1_all = [fn(b0, b1, b2) for _, fn in FORMULAS]
    p2_all = [fn(b0, b1, b2) for _, fn in FORMULAS]

    out = []
    for i1 in range(24):
        p1 = p1_all[i1]
        for i2 in range(24):
            if i1 == i2:
                continue
            p2 = p2_all[i2]

            flat = np.empty(total, dtype=np.uint16)
            flat[0::2] = p1[:total // 2]
            flat[1::2] = p2[:total // 2]
            bayer = flat.reshape(rows, w)

            for pattern in PATTERNS:
                ev = evaluate(bayer, pattern)
                if ev is None:
                    continue
                out.append({
                    "p1": i1, "p2": i2, "pattern": pattern,
                    "score": ev["score"], "corr": ev["corr"],
                    "stats": ev["stats"],
                })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sh", nargs=2, type=int, default=[0, 200],
                    help="диапазон sh (start end)")
    ap.add_argument("--sh-step", type=int, default=1)
    ap.add_argument("--sh-near", type=int, default=None,
                    help="искать только вокруг этого sh ±10")
    ap.add_argument("--rows", type=int, default=20)
    args = ap.parse_args()

    path = Path(MXF_FILE)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        sys.exit(1)

    result = mxf_parser.parse(path)
    off = result["header_offset"]

    print(f"Essence offset:   {off:,}")
    print(f"Кадров:           {result.get('frame_count')}")
    print(f"Размер кадра:     {result.get('frame_size'):,} байт")
    print(f"Разрешение:       {W}×{H}")
    print()

    rows = args.rows
    total = rows * W
    n_bytes = total * 3 // 2

    if args.sh_near is not None:
        sh_list = list(range(max(0, args.sh_near - 10),
                             args.sh_near + 11, args.sh_step))
    else:
        sh_list = list(range(args.sh[0], args.sh[1] + 1, args.sh_step))

    print(f"Сканирую {len(sh_list)} значений sh × 4 patterns × 24×23 формул")
    print(f"Rows per sample: {rows}, объём буфера: {n_bytes:,} байт")
    print()

    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(max(sh_list) + n_bytes + 100)

    all_results = []
    for idx, sh in enumerate(sh_list, 1):
        chunk = raw[sh:sh + n_bytes]
        if len(chunk) < n_bytes:
            continue

        d = np.frombuffer(chunk, dtype=np.uint8)
        b0 = d[0::3].astype(np.uint16)
        b1 = d[1::3].astype(np.uint16)
        b2 = d[2::3].astype(np.uint16)

        cands = search_one_sh(b0, b1, b2, total, rows, W)
        for c in cands:
            c["sh"] = sh
        all_results.extend(cands)

        if idx % 20 == 0 or idx == len(sh_list):
            print(f"  [{idx}/{len(sh_list)}] sh={sh}: "
                  f"найдено {len(cands)} кандидатов")

    if not all_results:
        print("\n❌ Ни одна конфигурация не прошла фильтр.")
        sys.exit(1)

    all_results.sort(key=lambda x: x["score"])

    print(f"\n{'='*100}")
    print(f"ТОП-30 конфигураций (score = средний noise_ratio, меньше = лучше)")
    print(f"{'='*100}")
    print(f"  #  sh   pattern  p1  p2  corr    score  "
          f"B_std  G2_std  G1_std  R_std   p1_formula  p2_formula")
    print("  " + "-" * 98)
    for i, r in enumerate(all_results[:30], 1):
        s = r["stats"]
        print(f"  {i:>2} {r['sh']:>4} {r['pattern']:<6} {r['p1']:>2} "
              f"{r['p2']:>2} {r['corr']:.3f}  {r['score']:.3f}  "
              f"{s['B']['std']:>6.0f} {s['G2']['std']:>6.0f} "
              f"{s['G1']['std']:>6.0f} {s['R']['std']:>6.0f}   "
              f"{FORMULAS[r['p1']][0]:<16}  {FORMULAS[r['p2']][0]}")

    print(f"\n{'='*100}")
    print(f"ТОП-5 РЕКОМЕНДАЦИЙ (запускайте по одной для проверки визуально)")
    print(f"{'='*100}")
    for i, r in enumerate(all_results[:5], 1):
        s = r["stats"]
        print(f"\n[{i}] corr={r['corr']:.3f}  score={r['score']:.3f}")
        print(f"    uv run python main.py --bayer={r['pattern']} "
              f"--p1={r['p1']} --p2={r['p2']} --sh={r['sh']}")
        print(f"    p1 = {FORMULAS[r['p1']][1].__name__ if hasattr(FORMULAS[r['p1']][1], '__name__') else FORMULAS[r['p1']][0]}")
        print(f"    p1 формула: {FORMULAS[r['p1']][0]}")
        print(f"    p2 формула: {FORMULAS[r['p2']][0]}")
        print(f"    std по каналам: B={s['B']['std']:.0f}, "
              f"G2={s['G2']['std']:.0f}, G1={s['G1']['std']:.0f}, "
              f"R={s['R']['std']:.0f}")


if __name__ == "__main__":
    main()