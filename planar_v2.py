"""
Уточнённый тест planar-структуры.

Признак правильной сборки:
  - внутриканальная пространственная корреляция между соседними
    пикселями высокая (>0.9)
  - G1↔G2 корреляция высокая
  - B↔R корреляция низкая
"""
import numpy as np
import tifffile
from pathlib import Path
from itertools import permutations
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
HDR = 76


def unpack12(raw, hw, hh):
    n = hw * hh
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
    return out[:n].reshape(hh, hw)


def spatial_corr(img):
    """Средняя корреляция горизонтальных и вертикальных соседей."""
    x = img.astype(np.float32)
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    hx = np.corrcoef(x[:, :-1].flatten(), x[:, 1:].flatten())[0, 1]
    hy = np.corrcoef(x[:-1, :].flatten(), x[1:, :].flatten())[0, 1]
    return float((hx + hy) / 2)


def assemble(c_B, c_G2, c_G1, c_R, pattern):
    bayer = np.zeros((H, W), dtype=np.uint16)
    if pattern == "BGGR":
        bayer[0::2, 0::2] = c_B
        bayer[0::2, 1::2] = c_G2
        bayer[1::2, 0::2] = c_G1
        bayer[1::2, 1::2] = c_R
    elif pattern == "GRBG":
        bayer[0::2, 0::2] = c_G1
        bayer[0::2, 1::2] = c_R
        bayer[1::2, 0::2] = c_B
        bayer[1::2, 1::2] = c_G2
    elif pattern == "RGGB":
        bayer[0::2, 0::2] = c_R
        bayer[0::2, 1::2] = c_G1
        bayer[1::2, 0::2] = c_G2
        bayer[1::2, 1::2] = c_B
    elif pattern == "GBRG":
        bayer[0::2, 0::2] = c_G1
        bayer[0::2, 1::2] = c_B
        bayer[1::2, 0::2] = c_R
        bayer[1::2, 1::2] = c_G2
    return bayer


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]

    hw, hh = W // 2, H // 2
    chunk_bytes = hw * hh * 3 // 2

    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(HDR + 4 * chunk_bytes)

    c = [
        unpack12(raw[HDR + i * chunk_bytes : HDR + (i + 1) * chunk_bytes], hw, hh)
        for i in range(4)
    ]

    print("Пространственная корреляция чанков:")
    for i, ch in enumerate(c):
        sc = spatial_corr(ch)
        print(f"  c{i}: std={ch.std():>7.1f}  spatial={sc:>+.4f}")
    print()

    results = []
    for perm in permutations(range(4)):
        B, G2, G1, R = c[perm[0]], c[perm[1]], c[perm[2]], c[perm[3]]
        for pattern in ["BGGR", "GRBG", "RGGB", "GBRG"]:
            bayer = assemble(B, G2, G1, R, pattern)
            B_  = bayer[0::2, 0::2].astype(np.float32).flatten()
            G2_ = bayer[0::2, 1::2].astype(np.float32).flatten()
            G1_ = bayer[1::2, 0::2].astype(np.float32).flatten()
            R_  = bayer[1::2, 1::2].astype(np.float32).flatten()

            if min(B_.std(), G2_.std(), G1_.std(), R_.std()) < 100:
                continue

            corr_g = float(np.corrcoef(G1_, G2_)[0, 1])
            corr_br = float(np.corrcoef(B_, R_)[0, 1])

            # Пространственная корреляция по исходным чанкам
            sp = (spatial_corr(B) + spatial_corr(G2)
                  + spatial_corr(G1) + spatial_corr(R)) / 4

            # Итоговый скор: хотим высокую corr_g, низкую corr_br,
            # высокую пространственную
            score = corr_g + sp - corr_br

            results.append((score, perm, pattern, corr_g, corr_br, sp))

    results.sort(reverse=True)

    print(f"{'assign':<18} {'pattern':<8} "
          f"{'G1-G2':>8} {'B-R':>8} {'spatial':>8} {'score':>8}")
    print("-" * 70)
    for score, perm, pattern, cg, cbr, sp in results[:15]:
        assign = f"({perm[0]},{perm[1]},{perm[2]},{perm[3]})"
        marker = " ★" if score > 1.5 else "  "
        print(f"{marker} {assign:<18} {pattern:<8} "
              f"{cg:>+8.3f} {cbr:>+8.3f} {sp:>+8.3f} {score:>+8.3f}")

    # Сохраняем 4 монохромных TIFF для топ-3 конфигураций
    print()
    for rank, (score, perm, pattern, cg, cbr, sp) in enumerate(results[:3], 1):
        B, G2, G1, R = c[perm[0]], c[perm[1]], c[perm[2]], c[perm[3]]
        for name, img in [("B", B), ("G2", G2), ("G1", G1), ("R", R)]:
            tifffile.imwrite(f"planar_top{rank}_{pattern}_{name}.tiff", img)
        # Псевдо-RGB
        def norm(x):
            p = np.percentile(x, 99.0)
            return np.clip(x.astype(np.float32) / max(p, 1), 0, 1)
        rgb = np.stack([norm(R), norm((G1.astype(np.float32) + G2) / 2),
                        norm(B)], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        h2, w2 = h // 4, w // 4
        rgb = rgb[:h2*4, :w2*4].reshape(h2, 4, w2, 4, 3).mean(axis=(1, 3))
        tifffile.imwrite(f"planar_top{rank}_{pattern}_rgb.tiff",
                         rgb.astype(np.uint8))

    best = results[0]
    print(f"🏆 Лучшее: assign={best[1]}  pattern={best[2]}")
    print(f"   G1↔G2={best[3]:+.3f}  B↔R={best[4]:+.3f}  "
          f"spatial={best[5]:+.3f}")
    print()
    print("Сохранены TIFF:")
    print(f"   planar_top1_*_*.tiff  → 4 канала + RGB превью")
    print(f"   planar_top2_*_*.tiff  → 4 канала + RGB превью")
    print(f"   planar_top3_*_*.tiff  → 4 канала + RGB превью")
    print()
    print("Откройте planar_top*_*_rgb.tiff и посмотрите — "
          "где сцена читается лучше.")


if __name__ == "__main__":
    main()