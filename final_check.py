"""
Финальная проверка: sh=1, variant=B, все 4 паттерна.
Правильный паттерн определяется по корреляции G1↔G2 (должна быть высокой).

Также сохраняет цветное превью для визуальной оценки.
"""
import numpy as np
import tifffile
from pathlib import Path
from PIL import Image
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
SH = 1
ROWS = 1500   # больше строк — стабильнее статистика

PATTERNS = {
    "BGGR": ("B",  "G2", "G1", "R"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}

VARIANTS = {
    "A": lambda b0, b1, b2: ((b1 << 4) | (b0 >> 4),
                             (b2 << 4) | (b0 & 0x0F)),
    "B": lambda b0, b1, b2: ((b1 << 4) | (b0 & 0x0F),
                             (b2 << 4) | (b0 >> 4)),
}


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]

    total_pixels = ROWS * W
    half = total_pixels // 2
    n_bytes = total_pixels * 3 // 2

    print(f"Essence offset: {off:,}")
    print(f"sh: {SH}   rows: {ROWS}")
    print()

    with open(path, "rb") as f:
        f.seek(off + SH)
        raw = f.read(n_bytes)

    d = np.frombuffer(raw, dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)

    for vname, fn in VARIANTS.items():
        p1, p2 = fn(b0, b1, b2)

        bayer = np.empty(total_pixels, dtype=np.uint16)
        bayer[0::2] = p1[:half]
        bayer[1::2] = p2[:half]
        bayer = bayer.reshape(ROWS, W)

        print(f"=== variant {vname} ===")
        print(f"{'pattern':<8} "
              f"{'G1↔G2':>8} {'G1↔B':>8} {'G1↔R':>8} "
              f"{'G2↔B':>8} {'G2↔R':>8} {'B↔R':>8}  "
              f"{'score':>8}")

        results = []
        for pat, (tl, tr, bl, br) in PATTERNS.items():
            chans = {
                tl: bayer[0::2, 0::2].astype(np.float32),
                tr: bayer[0::2, 1::2].astype(np.float32),
                bl: bayer[1::2, 0::2].astype(np.float32),
                br: bayer[1::2, 1::2].astype(np.float32),
            }
            g1 = chans["G1"].flatten()
            g2 = chans["G2"].flatten()
            bb = chans["B"].flatten()
            rr = chans["R"].flatten()

            def cc(a, b):
                if a.std() < 1 or b.std() < 1:
                    return 0.0
                return float(np.corrcoef(a, b)[0, 1])

            c_g1g2 = cc(g1, g2)
            c_g1b  = cc(g1, bb)
            c_g1r  = cc(g1, rr)
            c_g2b  = cc(g2, bb)
            c_g2r  = cc(g2, rr)
            c_br   = cc(bb, rr)

            # Правильный паттерн: G1↔G2 высоко, остальные с G — низко
            score = c_g1g2 - 0.25 * (c_g1b + c_g1r + c_g2b + c_g2r)
            results.append((score, pat, c_g1g2, c_g1b, c_g1r,
                            c_g2b, c_g2r, c_br, chans))

            print(f"{pat:<8} "
                  f"{c_g1g2:>+8.3f} {c_g1b:>+8.3f} {c_g1r:>+8.3f} "
                  f"{c_g2b:>+8.3f} {c_g2r:>+8.3f} {c_br:>+8.3f}  "
                  f"{score:>+8.3f}")

        results.sort(reverse=True)
        best = results[0]
        print(f"  🏆 Лучший pattern: {best[1]}  (score={best[0]:+.3f})\n")

        if vname == "B":
            # Сохраним превью для лучшего паттерна обоих вариантов
            for i, (sc, pat, *_rest) in enumerate(results[:2]):
                chans = _rest[-1]
                img = build_rgb(chans)
                out = f"preview_{vname}_{pat}.png"
                Image.fromarray(img).save(out)
                print(f"  💾 {out}")

    print("Откройте preview_*.png и сравните — где структура сцены видна лучше.")


def build_rgb(chans):
    """Сборка псевдо-RGB из 4 каналов Bayer."""
    R = chans["R"].astype(np.float32)
    B = chans["B"].astype(np.float32)
    G1 = chans["G1"].astype(np.float32)
    G2 = chans["G2"].astype(np.float32)
    G = 0.5 * (G1 + G2)

    def norm(x):
        p = np.percentile(x, 99.0)
        return np.clip(x / max(p, 1), 0, 1)

    rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    # Downscale ×4 для превью
    h, w = rgb.shape[:2]
    h2, w2 = h // 4, w // 4
    rgb = rgb[:h2*4, :w2*4].reshape(h2, 4, w2, 4, 3).mean(axis=(1, 3))
    return rgb.astype(np.uint8)


if __name__ == "__main__":
    main()