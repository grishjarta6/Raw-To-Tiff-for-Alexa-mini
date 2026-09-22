"""
Авто-определение параметров распаковки ARRIRAW ALEXA Mini.

Метрика: корреляция G1↔G2 + штрафы за дубликаты и плоские каналы.

При правильной распаковке G1 и G2 — это один и тот же зелёный канал,
снятый в соседних пикселях. Их корреляция должна быть высокой (>0.7).
При неправильной — близка к нулю.
"""

import sys
from pathlib import Path

import numpy as np

import mxf_parser


# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202

# Сколько строк анализировать (больше — точнее, медленнее)
TARGET_ROWS = 300

# Диапазон sub-header для перебора
SH_RANGE = range(0, 100)

# Минимальный std, ниже которого канал считается пустым
MIN_STD = 100.0


# ---------------------------------------------------------------------------
# Паттерны Байера (top-left, top-right, bottom-left, bottom-right)
# ---------------------------------------------------------------------------

BAYER_PATTERNS = {
    "GRBG": ("G1", "R",  "B",  "G2"),
    "BGGR": ("B",  "G2", "G1", "R"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}

# Позиции G1 и G2 в каждом паттерне (row_parity, col_parity)
G_POS = {
    "GRBG": ((0, 0), (1, 1)),
    "BGGR": ((1, 0), (0, 1)),
    "RGGB": ((0, 1), (1, 0)),
    "GBRG": ((0, 0), (1, 1)),
}


# ---------------------------------------------------------------------------
# 24 формулы распаковки 12-бит
# ---------------------------------------------------------------------------

FORMULAS = [
    lambda b0, b1, b2: (b0 << 4) | (b1 >> 4),
    lambda b0, b1, b2: (b0 << 4) | (b1 & 0x0F),
    lambda b0, b1, b2: (b0 << 4) | (b2 >> 4),
    lambda b0, b1, b2: (b0 << 4) | (b2 & 0x0F),
    lambda b0, b1, b2: (b1 << 4) | (b0 >> 4),
    lambda b0, b1, b2: (b1 << 4) | (b0 & 0x0F),
    lambda b0, b1, b2: (b1 << 4) | (b2 >> 4),
    lambda b0, b1, b2: (b1 << 4) | (b2 & 0x0F),
    lambda b0, b1, b2: (b2 << 4) | (b0 >> 4),
    lambda b0, b1, b2: (b2 << 4) | (b0 & 0x0F),
    lambda b0, b1, b2: (b2 << 4) | (b1 >> 4),
    lambda b0, b1, b2: (b2 << 4) | (b1 & 0x0F),
    lambda b0, b1, b2: ((b0 & 0x0F) << 8) | b1,
    lambda b0, b1, b2: ((b0 & 0x0F) << 8) | b2,
    lambda b0, b1, b2: ((b1 & 0x0F) << 8) | b0,
    lambda b0, b1, b2: ((b1 & 0x0F) << 8) | b2,
    lambda b0, b1, b2: ((b2 & 0x0F) << 8) | b0,
    lambda b0, b1, b2: ((b2 & 0x0F) << 8) | b1,
    lambda b0, b1, b2: ((b0 >> 4) << 8) | b1,
    lambda b0, b1, b2: ((b0 >> 4) << 8) | b2,
    lambda b0, b1, b2: ((b1 >> 4) << 8) | b0,
    lambda b0, b1, b2: ((b1 >> 4) << 8) | b2,
    lambda b0, b1, b2: ((b2 >> 4) << 8) | b0,
    lambda b0, b1, b2: ((b2 >> 4) << 8) | b1,
]


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------

def load_essence(path, off, n_bytes):
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(n_bytes)


def analyze():
    path = Path(MXF_FILE)
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        sys.exit(1)

    result = mxf_parser.parse(path)
    off = result["header_offset"]

    print(f"Essence offset:        {off:,}")
    print(f"Кадров в MXF:          {result.get('frame_count')}")
    print(f"Размер одного кадра:   {result.get('frame_size'):,} байт")
    print(f"Разрешение:            {W}×{H}")

    # Сколько байт нужно
    row_bytes = W * 3 // 2              # 5136 байт на строку пикселей
    n_bytes = TARGET_ROWS * row_bytes   # сколько нужно
    max_sh = max(SH_RANGE)
    n_read = n_bytes + max_sh + 100

    print(f"Читаю {n_read:,} байт "
          f"({TARGET_ROWS} строк + {max_sh} байт на sub-header)")
    raw = load_essence(path, off, n_read)
    if len(raw) < n_bytes:
        print(f"❌ Недостаточно данных: {len(raw)} < {n_bytes}")
        sys.exit(1)

    total_pixels = TARGET_ROWS * W

    results = []

    for sh in SH_RANGE:
        chunk = raw[sh:sh + n_bytes]
        if len(chunk) < n_bytes:
            break
        d = np.frombuffer(chunk, dtype=np.uint8)
        b0 = d[0::3].astype(np.uint16)
        b1 = d[1::3].astype(np.uint16)
        b2 = d[2::3].astype(np.uint16)

        # Предвычисляем 24 значения для p1 и p2 один раз
        p_arr = [f(b0, b1, b2) for f in FORMULAS]

        for pattern in BAYER_PATTERNS:
            (g1_r, g1_c), (g2_r, g2_c) = G_POS[pattern]

            for i1 in range(24):
                p1 = p_arr[i1]
                for i2 in range(24):
                    # Отсекаем одинаковые формулы — они дают дубликаты
                    if i1 == i2:
                        continue

                    p2 = p_arr[i2]

                    pixels = np.empty(total_pixels, dtype=np.uint16)
                    pixels[0::2] = p1[:total_pixels // 2]
                    pixels[1::2] = p2[:total_pixels // 2]
                    bayer = pixels.reshape(TARGET_ROWS, W)

                    g1 = bayer[g1_r::2, g1_c::2].flatten().astype(np.float32)
                    g2 = bayer[g2_r::2, g2_c::2].flatten().astype(np.float32)

                    if g1.std() < MIN_STD or g2.std() < MIN_STD:
                        continue

                    try:
                        corr = float(np.corrcoef(g1, g2)[0, 1])
                    except Exception:
                        continue

                    results.append({
                        "sh": sh, "pattern": pattern,
                        "p1": i1, "p2": i2,
                        "corr": corr,
                        "g1_std": float(g1.std()),
                        "g2_std": float(g2.std()),
                    })

    if not results:
        print("❌ Ни одна комбинация не прошла фильтр. "
              "Проверьте разрешение или essence offset.")
        sys.exit(1)

    results.sort(key=lambda x: x["corr"], reverse=True)

    print(f"\n📊 Проверено комбинаций: {len(results)}")
    print(f"\nТоп-20 по корреляции G1↔G2 (больше = лучше):")
    print(f"   #  sh  pattern   p1  p2      corr    G1_std  G2_std")
    print("  " + "-" * 60)
    for i, r in enumerate(results[:20], 1):
        print(f"  {i:>2}  {r['sh']:>2}  {r['pattern']:<6}  "
              f"{r['p1']:>2}  {r['p2']:>2}   {r['corr']:+.4f}  "
              f"{r['g1_std']:>7.1f}  {r['g2_std']:>7.1f}")

    best = results[0]
    print(f"\n✅ Лучшее: sh={best['sh']} pattern={best['pattern']} "
          f"p1={best['p1']} p2={best['p2']}  (corr={best['corr']:.4f})")

    if best["corr"] < 0.5:
        print("\n⚠ Корреляция < 0.5 даже у лучшего кандидата. "
              "Возможные причины:")
        print("   - файл повреждён или обрезан")
        print("   - разрешение не 3424×2202")
        print("   - формат упаковки не 12-бит (может быть 10-бит или 14-бит)")
    else:
        print(f"\n🎯 Используйте:")
        print(f"   --bayer={best['pattern']} --p1={best['p1']} "
              f"--p2={best['p2']} --sh={best['sh']}")


if __name__ == "__main__":
    analyze()