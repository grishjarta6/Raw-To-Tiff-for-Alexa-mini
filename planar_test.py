"""
Проверка гипотезы 4-planar by color.

essence_size = 76 + 4 × 2 827 368
              ↑      ↑
              hdr    = 1 884 912 пикселей × 1.5 байта
                    = W/2 × H/2 × 1.5
"""

import numpy as np
from pathlib import Path
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202


def unpack12(raw, half_w, half_h):
    """12-бит unpack: 3 байта → 2 пикселя (стандарт SMPTE RDD 30)."""
    n = half_w * half_h
    need = n * 3 // 2
    raw = raw[:need]
    d = np.frombuffer(raw, dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p0 = (b0 << 4) | (b1 >> 4)
    p1 = ((b1 & 0x0F) << 8) | b2
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p0
    out[1::2] = p1
    return out[:n].reshape(half_h, half_w)


def assemble_bayer(ch_B, ch_G2, ch_G1, ch_R, pattern):
    """Собирает полный кадр Bayer из 4 полукадров."""
    hh, hw = ch_B.shape
    bayer = np.zeros((H, W), dtype=np.uint16)

    # Позиции для BGGR:
    #   B  at (2i, 2j)   → bayer[0::2, 0::2]
    #   G2 at (2i, 2j+1) → bayer[0::2, 1::2]
    #   G1 at (2i+1, 2j) → bayer[1::2, 0::2]
    #   R  at (2i+1, 2j+1) → bayer[1::2, 1::2]
    if pattern == "BGGR":
        bayer[0::2, 0::2] = ch_B
        bayer[0::2, 1::2] = ch_G2
        bayer[1::2, 0::2] = ch_G1
        bayer[1::2, 1::2] = ch_R
    elif pattern == "GRBG":
        bayer[0::2, 0::2] = ch_G1
        bayer[0::2, 1::2] = ch_R
        bayer[1::2, 0::2] = ch_B
        bayer[1::2, 1::2] = ch_G2
    elif pattern == "RGGB":
        bayer[0::2, 0::2] = ch_R
        bayer[0::2, 1::2] = ch_G1
        bayer[1::2, 0::2] = ch_G2
        bayer[1::2, 1::2] = ch_B
    elif pattern == "GBRG":
        bayer[0::2, 0::2] = ch_G1
        bayer[0::2, 1::2] = ch_B
        bayer[1::2, 0::2] = ch_R
        bayer[1::2, 1::2] = ch_G2
    return bayer


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]
    ess = path.stat().st_size - off

    print(f"Essence offset: {off:,}")
    print(f"Essence size:   {ess:,}")

    hw, hh = W // 2, H // 2
    chunk_pixels = hw * hh
    chunk_bytes = chunk_pixels * 3 // 2     # 2 827 368

    print(f"Полукадр:       {hw}×{hh} = {chunk_pixels:,} пикселей")
    print(f"Размер чанка:   {chunk_bytes:,} байт")
    print(f"4 чанка + 76:   {76 + 4 * chunk_bytes:,} байт")
    print(f"Совпадает:      {76 + 4 * chunk_bytes == ess}")
    print()

    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(ess)

    hdr = 76

    # Читаем 4 чанка
    c_B  = unpack12(raw[hdr + 0 * chunk_bytes : hdr + 1 * chunk_bytes], hw, hh)
    c_G2 = unpack12(raw[hdr + 1 * chunk_bytes : hdr + 2 * chunk_bytes], hw, hh)
    c_G1 = unpack12(raw[hdr + 2 * chunk_bytes : hdr + 3 * chunk_bytes], hw, hh)
    c_R  = unpack12(raw[hdr + 3 * chunk_bytes : hdr + 4 * chunk_bytes], hw, hh)

    print("Статистика чанков (по файлу):")
    for name, c in [("c0", c_B), ("c1", c_G2), ("c2", c_G1), ("c3", c_R)]:
        print(f"  {name}: min={c.min():>5} max={c.max():>5} "
              f"mean={c.mean():>8.1f} std={c.std():>8.1f}")

    # Пробуем все 24 перестановки 4 каналов → 4 набора (B, G2, G1, R)
    # Но проще: назначаем c0..c3 в позиции B, G2, G1, R последовательно
    # и проверяем общий скор.
    # Перестановок 24, но по факту интересует, какой cX идёт в какую позицию.
    from itertools import permutations

    print()
    print("Перебираю 24 назначения чанков в позиции (B, G2, G1, R) "
          "× 4 паттерна:")
    print(f"{'assign':<20} {'pattern':<8} {'G1↔G2':>8} "
          f"{'score':>8}")
    print("-" * 60)

    chunks = [c_B, c_G2, c_G1, c_R]
    names = ["c0", "c1", "c2", "c3"]

    results = []
    for perm in permutations(range(4)):
        B_ch, G2_ch, G1_ch, R_ch = (chunks[perm[0]], chunks[perm[1]],
                                     chunks[perm[2]], chunks[perm[3]])
        assign = f"({names[perm[0]]},{names[perm[1]]}," \
                 f"{names[perm[2]]},{names[perm[3]]})"

        for pattern in ["BGGR", "GRBG", "RGGB", "GBRG"]:
            bayer = assemble_bayer(B_ch, G2_ch, G1_ch, R_ch, pattern)

            # Проверяем по 4 каналам в позициях BGGR
            B_  = bayer[0::2, 0::2].astype(np.float32).flatten()
            G2_ = bayer[0::2, 1::2].astype(np.float32).flatten()
            G1_ = bayer[1::2, 0::2].astype(np.float32).flatten()
            R_  = bayer[1::2, 1::2].astype(np.float32).flatten()

            stds = [B_.std(), G2_.std(), G1_.std(), R_.std()]
            if min(stds) < 100:
                continue

            corr_g1g2 = float(np.corrcoef(G1_, G2_)[0, 1])
            score = corr_g1g2
            results.append((score, assign, pattern, corr_g1g2, stds))

    if not results:
        print("❌ Ни одна комбинация не прошла фильтр std>100.")
        return

    results.sort(key=lambda r: r[0], reverse=True)

    for score, assign, pattern, corr, stds in results[:15]:
        marker = " ★" if corr > 0.5 else "  "
        print(f"{marker} {assign:<20} {pattern:<8} {corr:>+8.3f} "
              f"{score:>+8.3f}   "
              f"[{stds[0]:>5.0f} {stds[1]:>5.0f} "
              f"{stds[2]:>5.0f} {stds[3]:>5.0f}]")

    best = results[0]
    print()
    print(f"🏆 Лучшее назначение: {best[1]}  pattern={best[2]}  "
          f"corr(G1,G2)={best[3]:+.3f}")
    print(f"   stds: B={best[4][0]:.0f}  G2={best[4][1]:.0f}  "
          f"G1={best[4][2]:.0f}  R={best[4][3]:.0f}")


if __name__ == "__main__":
    main()