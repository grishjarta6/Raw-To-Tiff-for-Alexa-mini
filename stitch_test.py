"""
Перебирает 4 варианта сборки полного кадра из двух чанков.
Правильный — тот, где нет видимых швов/креста.
"""
import numpy as np
import tifffile
from pathlib import Path
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
HDR = 76
HALF_W = W // 2                # 1712
HALF_H = H // 2                # 1101
CHUNK_BYTES = (W * H // 2) * 3 // 2   # 5 654 736
FRAME_SIZE = HDR + 2 * CHUNK_BYTES


def unpack12(b0, b1, b2):
    """arri_alt — найденная правильная формула."""
    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1
    return p1, p2


def unpack_chunk(raw):
    n = len(raw) // 3
    d = np.frombuffer(raw[:n * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1, p2 = unpack12(b0, b1, b2)
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


def assemble_even_odd_columns(c1, c2):
    """Вариант A: чётные/нечётные СТОЛБЦЫ (то, что было)."""
    a = c1.reshape(H, HALF_W)
    b = c2.reshape(H, HALF_W)
    bayer = np.empty((H, W), dtype=np.uint16)
    bayer[:, 0::2] = a
    bayer[:, 1::2] = b
    return bayer


def assemble_even_odd_rows(c1, c2):
    """Вариант B: чётные/нечётные СТРОКИ."""
    a = c1.reshape(HALF_H, W)
    b = c2.reshape(HALF_H, W)
    bayer = np.empty((H, W), dtype=np.uint16)
    bayer[0::2, :] = a
    bayer[1::2, :] = b
    return bayer


def assemble_left_right(c1, c2):
    """Вариант C: левая и правая половины кадра."""
    a = c1.reshape(H, HALF_W)
    b = c2.reshape(H, HALF_W)
    bayer = np.empty((H, W), dtype=np.uint16)
    bayer[:, :HALF_W] = a
    bayer[:, HALF_W:] = b
    return bayer


def assemble_top_bottom(c1, c2):
    """Вариант D: верхняя и нижняя половины кадра."""
    a = c1.reshape(HALF_H, W)
    b = c2.reshape(HALF_H, W)
    bayer = np.empty((H, W), dtype=np.uint16)
    bayer[:HALF_H, :] = a
    bayer[HALF_H:, :] = b
    return bayer


STITCHERS = {
    "A_even_odd_columns": assemble_even_odd_columns,
    "B_even_odd_rows":    assemble_even_odd_rows,
    "C_left_right":       assemble_left_right,
    "D_top_bottom":       assemble_top_bottom,
}


def rgb_preview(bayer, pattern="BGGR", downscale=4):
    tl, tr, bl, br = {
        "BGGR": ("B", "G2", "G1", "R"),
    }[pattern]
    B  = bayer[0::2, 0::2].astype(np.float32)
    G2 = bayer[0::2, 1::2].astype(np.float32)
    G1 = bayer[1::2, 0::2].astype(np.float32)
    R  = bayer[1::2, 1::2].astype(np.float32)

    def norm(x):
        p = np.percentile(x, 99.0)
        return np.clip(x / max(p, 1), 0, 1)

    G = 0.5 * (G1 + G2)
    rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    h2, w2 = h // downscale, w // downscale
    rgb = rgb[:h2*downscale, :w2*downscale].reshape(
        h2, downscale, w2, downscale, 3).mean(axis=(1, 3))
    return rgb.astype(np.uint8)


def spatial_corr(img):
    """Горизонтальная + вертикальная корреляция соседей."""
    x = img.astype(np.float32)
    hx = np.corrcoef(x[:, :-1].flatten(), x[:, 1:].flatten())[0, 1]
    hy = np.corrcoef(x[:-1, :].flatten(), x[1:, :].flatten())[0, 1]
    return float(hx), float(hy)


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]
    print(f"Essence offset: {off:,}")
    print(f"Frame size:     {FRAME_SIZE:,}")
    print()

    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(FRAME_SIZE)

    c1_raw = raw[HDR : HDR + CHUNK_BYTES]
    c2_raw = raw[HDR + CHUNK_BYTES : HDR + 2 * CHUNK_BYTES]

    c1 = unpack_chunk(c1_raw)
    c2 = unpack_chunk(c2_raw)
    print(f"chunk1: {len(c1):,} пикс.  std={c1.std():.0f}")
    print(f"chunk2: {len(c2):,} пикс.  std={c2.std():.0f}")
    print()

    for name, fn in STITCHERS.items():
        try:
            bayer = fn(c1, c2)
        except Exception as e:
            print(f"❌ {name}: {e}")
            continue

        # Метрики
        B  = bayer[0::2, 0::2].astype(np.float32)
        G2 = bayer[0::2, 1::2].astype(np.float32)
        G1 = bayer[1::2, 0::2].astype(np.float32)
        R  = bayer[1::2, 1::2].astype(np.float32)

        sp_B  = spatial_corr(B)[0]
        sp_G2 = spatial_corr(G2)[0]
        sp_G1 = spatial_corr(G1)[0]
        sp_R  = spatial_corr(R)[0]
        avg_sp = (sp_B + sp_G2 + sp_G1 + sp_R) / 4

        # Корреляция G1↔G2
        cg = float(np.corrcoef(G1.flatten(), G2.flatten())[0, 1])

        print(f"{name}:")
        print(f"  sp_h B={sp_B:+.2f}  G2={sp_G2:+.2f}  "
              f"G1={sp_G1:+.2f}  R={sp_R:+.2f}  "
              f"avg={avg_sp:+.3f}  G1↔G2={cg:+.3f}")

        # Сохраняем RGB-превью
        rgb = rgb_preview(bayer)
        out = f"stitch_{name}.tiff"
        tifffile.imwrite(out, rgb)
        print(f"  💾 {out}")
        print()

    print("=== Откройте stitch_*.tiff и сравните ===")
    print("Правильный вариант — тот, где НЕТ видимого креста и швов")
    print("(гладкая непрерывная сцена по всему кадру).")


if __name__ == "__main__":
    main()