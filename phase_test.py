"""
Гипотеза: 4-way column interleave.
Каждый чанк = одна колоночная фаза 856 × 2202.
"""
import numpy as np
import tifffile
from pathlib import Path
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
HDR = 76
PHASE_W = W // 4        # 856
PHASE_H = H             # 2202
PHASE_PIX = PHASE_W * PHASE_H


def unpack12(raw, n_pix):
    need = n_pix * 3 // 2
    d = np.frombuffer(raw[:need], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p0 = (b0 << 4) | (b1 >> 4)
    p1 = ((b1 & 0x0F) << 8) | b2
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p0
    out[1::2] = p1
    return out[:n_pix]


def spatial_hv(img):
    x = img.astype(np.float32)
    hx = float(np.corrcoef(x[:, :-1].flatten(), x[:, 1:].flatten())[0, 1])
    hy = float(np.corrcoef(x[:-1, :].flatten(), x[1:, :].flatten())[0, 1])
    return hx, hy


PATTERNS = {
    "BGGR": ("B",  "G2", "G1", "R"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


def get_channel(frame, name):
    for pat, (tl, tr, bl, br) in PATTERNS.items():
        # Найти позицию name в BGGR-координатах:
        # row_parity, col_parity
        pass
    # Проще — обходим все 4 паттерна явно:
    return None


def channel_by_pos(frame, row_p, col_p):
    return frame[row_p::2, col_p::2]


def find_color_positions(pattern):
    """
    Возвращает {color: (row_parity, col_parity)} для данного паттерна.
    """
    tl, tr, bl, br = PATTERNS[pattern]
    return {
        tl: (0, 0),
        tr: (0, 1),
        bl: (1, 0),
        br: (1, 1),
    }


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]

    chunk_bytes = PHASE_PIX * 3 // 2
    print(f"Essence offset: {off:,}")
    print(f"Размер чанка:   {chunk_bytes:,}")
    print(f"Всего:          76 + 4 × {chunk_bytes:,} = "
          f"{76 + 4 * chunk_bytes:,}")
    print()

    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(HDR + 4 * chunk_bytes)

    phases = []
    for i in range(4):
        chunk = raw[HDR + i * chunk_bytes : HDR + (i + 1) * chunk_bytes]
        p = unpack12(chunk, PHASE_PIX).reshape(PHASE_H, PHASE_W)
        phases.append(p)
        hx, hy = spatial_hv(p)
        print(f"phase {i}: std={p.std():>7.1f}  "
              f"sp_h={hx:>+.4f}  sp_v={hy:>+.4f}")

    # Собираем полный кадр с 4-фазным чередованием (x mod 4)
    frame = np.empty((H, W), dtype=np.uint16)
    for k in range(4):
        frame[:, k::4] = phases[k]

    print()
    print("=== Анализ паттернов Bayer ===")
    print(f"{'pattern':<8} {'G1↔G2':>8} {'B↔R':>8}  "
          f"{'sp_B':>7} {'sp_G2':>7} {'sp_G1':>7} {'sp_R':>7}  "
          f"{'avg_sp':>7}")
    print("-" * 78)

    results = []
    for pattern in PATTERNS:
        pos = find_color_positions(pattern)
        chans = {c: channel_by_pos(frame, rp, cp).astype(np.float32)
                 for c, (rp, cp) in pos.items()}

        G1 = chans["G1"].flatten()
        G2 = chans["G2"].flatten()
        B  = chans["B"].flatten()
        R  = chans["R"].flatten()

        cg = float(np.corrcoef(G1, G2)[0, 1])
        cbr = float(np.corrcoef(B, R)[0, 1])

        def sp(img):
            hx, hy = spatial_hv(img)
            return (hx + hy) / 2

        sB = sp(chans["B"]); sG1 = sp(chans["G1"])
        sG2 = sp(chans["G2"]); sR = sp(chans["R"])
        savg = (sB + sG1 + sG2 + sR) / 4

        results.append((savg, cg, pattern, chans, (sB, sG1, sG2, sR)))

        print(f"{pattern:<8} {cg:>+8.3f} {cbr:>+8.3f}  "
              f"{sB:>+7.3f} {sG2:>+7.3f} {sG1:>+7.3f} {sR:>+7.3f}  "
              f"{savg:>+7.3f}")

    # Превью для каждого паттерна
    print()
    for savg, cg, pattern, chans, _ in results:
        R_c = chans["R"]; B_c = chans["B"]
        G1_c = chans["G1"]; G2_c = chans["G2"]

        def norm(x):
            p = np.percentile(x, 99.0)
            return np.clip(x / max(p, 1), 0, 1)

        G = 0.5 * (G1_c + G2_c)
        rgb = np.stack([norm(R_c), norm(G), norm(B_c)], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        h, w = rgb.shape[:2]
        h2, w2 = h // 4, w // 4
        rgb = rgb[:h2*4, :w2*4].reshape(h2, 4, w2, 4, 3).mean(axis=(1, 3))
        tifffile.imwrite(f"phase_{pattern}.tiff", rgb.astype(np.uint8))
        print(f"  phase_{pattern}.tiff  (sp_avg={savg:+.3f}, "
              f"G1↔G2={cg:+.3f})")

    results.sort(reverse=True)
    best = results[0]
    print()
    print(f"🏆 Лучший по spatial: pattern={best[2]}  "
          f"sp_avg={best[0]:+.3f}  G1↔G2={best[1]:+.3f}")


if __name__ == "__main__":
    main()