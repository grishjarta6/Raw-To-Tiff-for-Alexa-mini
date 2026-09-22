"""
Гипотеза: кадр = [76 байт] + [chunk1: чётные столбцы] + [chunk2: нечётные столбцы].

Чётные и нечётные столбцы хранятся отдельными 12-бит потоками.
"""
import numpy as np
import tifffile
from pathlib import Path
import mxf_parser


MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
HDR = 76
HALF_W = W // 2       # 1712
HALF_PIX = HALF_W * H # 3 769 824
CHUNK_BYTES = HALF_PIX * 3 // 2  # 5 654 736


UNPACK_FORMULAS = {
    "rdd30":      lambda b0, b1, b2: ((b0 << 4) | (b1 >> 4),
                                      ((b1 & 0x0F) << 8) | b2),
    "arri_be":    lambda b0, b1, b2: ((b0 << 4) | (b2 >> 4),
                                      (b1 << 4) | (b2 & 0x0F)),
    "arri_alt":   lambda b0, b1, b2: ((b0 << 4) | (b2 >> 4),
                                      ((b2 & 0x0F) << 8) | b1),
    "arri_be_rev":lambda b0, b1, b2: ((b1 << 4) | (b2 & 0x0F),
                                      (b0 << 4) | (b2 >> 4)),
}


def unpack_chunk(raw, formula):
    """12-бит unpack: 3 байта → 2 пикселя."""
    n_pairs = len(raw) // 3
    d = np.frombuffer(raw[:n_pairs * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1, p2 = formula(b0, b1, b2)
    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


def assemble_bayer(ch1, ch2, pattern):
    """Чётные столбцы из ch1, нечётные из ch2."""
    hw = W // 2
    bayer = np.empty((H, W), dtype=np.uint16)

    ch1 = ch1[:HALF_PIX].reshape(H, hw)
    ch2 = ch2[:HALF_PIX].reshape(H, hw)

    if pattern == "BGGR":
        # B   (even rows, even cols) ← ch1 even rows
        # G2  (even rows, odd  cols) ← ch2 even rows
        # G1  (odd  rows, even cols) ← ch1 odd rows
        # R   (odd  rows, odd  cols) ← ch2 odd rows
        bayer[0::2, 0::2] = ch1[0::2, :]
        bayer[0::2, 1::2] = ch2[0::2, :]
        bayer[1::2, 0::2] = ch1[1::2, :]
        bayer[1::2, 1::2] = ch2[1::2, :]
    elif pattern == "GRBG":
        bayer[0::2, 0::2] = ch1[0::2, :]
        bayer[0::2, 1::2] = ch2[0::2, :]
        bayer[1::2, 0::2] = ch1[1::2, :]
        bayer[1::2, 1::2] = ch2[1::2, :]
    else:
        # RGGB, GBRG — те же позиции каналов, но роли цветов другие
        bayer[0::2, 0::2] = ch1[0::2, :]
        bayer[0::2, 1::2] = ch2[0::2, :]
        bayer[1::2, 0::2] = ch1[1::2, :]
        bayer[1::2, 1::2] = ch2[1::2, :]
    return bayer, ch1, ch2


def stats(img):
    return img.min(), img.max(), img.mean(), img.std()


def main():
    path = Path(MXF_FILE)
    off = mxf_parser.parse(path)["header_offset"]

    print(f"Essence offset: {off:,}")
    print(f"HDR:            {HDR}")
    print(f"Chunk bytes:    {CHUNK_BYTES:,} (×2 = {2*CHUNK_BYTES:,})")
    print(f"Total:          76 + 2×{CHUNK_BYTES:,} = "
          f"{76 + 2*CHUNK_BYTES:,}")

    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(HDR + 2 * CHUNK_BYTES)

    ch1 = raw[HDR : HDR + CHUNK_BYTES]
    ch2 = raw[HDR + CHUNK_BYTES : HDR + 2 * CHUNK_BYTES]

    print()
    for mode, fn in UNPACK_FORMULAS.items():
        print(f"=== {mode} ===")
        p1 = unpack_chunk(ch1, fn)
        p2 = unpack_chunk(ch2, fn)

        print(f"  chunk1: len={len(p1):,}  "
              f"min={p1.min():>5} max={p1.max():>5} std={p1.std():>7.1f}")
        print(f"  chunk2: len={len(p2):,}  "
              f"min={p2.min():>5} max={p2.max():>5} std={p2.std():>7.1f}")

        for pattern in ["BGGR", "GRBG", "RGGB", "GBRG"]:
            bayer, c1, c2 = assemble_bayer(p1, p2, pattern)
            tl, tr, bl, br = {
                "BGGR": ("B",  "G2", "G1", "R"),
                "GRBG": ("G1", "R",  "B",  "G2"),
                "RGGB": ("R",  "G1", "G2", "B"),
                "GBRG": ("G1", "B",  "R",  "G2"),
            }[pattern]

            def ch(name):
                rp, cp = {"tl": (0, 0), "tr": (0, 1),
                          "bl": (1, 0), "br": (1, 1)}[{
                    tl: "tl", tr: "tr", bl: "bl", br: "br"}[name]]
                return bayer[rp::2, cp::2].astype(np.float32)

            B  = ch("B");  G2 = ch("G2")
            G1 = ch("G1"); R  = ch("R")

            # Spatial: горизонтальная корреляция внутри канала
            def sp(x):
                hx = np.corrcoef(x[:, :-1].flatten(),
                                 x[:, 1:].flatten())[0, 1]
                return float(hx)

            sB  = sp(B);  sG2 = sp(G2)
            sG1 = sp(G1); sR  = sp(R)
            avg = (sB + sG2 + sG1 + sR) / 4

            marker = " ★" if avg > 0.3 else ""
            print(f"  {pattern}: sp=({sB:+.2f},{sG2:+.2f},"
                  f"{sG1:+.2f},{sR:+.2f}) avg={avg:+.3f}{marker}")

            # Сохраняем превью для BGGR
            if pattern == "BGGR":
                tifffile.imwrite(f"chunked_{mode}_B.tiff", B.astype(np.uint16))
                tifffile.imwrite(f"chunked_{mode}_G1.tiff", G1.astype(np.uint16))
                tifffile.imwrite(f"chunked_{mode}_G2.tiff", G2.astype(np.uint16))
                tifffile.imwrite(f"chunked_{mode}_R.tiff", R.astype(np.uint16))

                def norm(x):
                    p = np.percentile(x, 99.0)
                    return np.clip(x / max(p, 1), 0, 1)
                G = 0.5 * (G1 + G2)
                rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
                rgb = (rgb * 255).astype(np.uint8)
                h, w = rgb.shape[:2]
                h2, w2 = h // 4, w // 4
                rgb = rgb[:h2*4, :w2*4].reshape(
                    h2, 4, w2, 4, 3).mean(axis=(1, 3))
                tifffile.imwrite(f"chunked_{mode}.tiff",
                                 rgb.astype(np.uint8))
        print()


if __name__ == "__main__":
    main()