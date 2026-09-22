"""Проверка: насколько G1 и G2 отличаются численно."""
import numpy as np
from pathlib import Path
import mxf_parser

MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
HDR = 76
HALF_H = H // 2
CHUNK_PIX = HALF_H * W
CHUNK_BYTES = CHUNK_PIX * 3 // 2

def unpack(raw):
    n = len(raw) // 3
    d = np.frombuffer(raw[:n*3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)
    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1
    out = np.empty(len(b0)*2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out

path = Path(MXF_FILE)
pkt = mxf_parser.parse(path)["essence"][0]
with open(path, "rb") as f:
    f.seek(pkt["value_start"])
    raw = f.read(pkt["length"])

c1 = unpack(raw[HDR:HDR+CHUNK_BYTES])
c2 = unpack(raw[HDR+CHUNK_BYTES:HDR+2*CHUNK_BYTES])
frame = np.vstack([c1[:CHUNK_PIX].reshape(HALF_H, W),
                   c2[:CHUNK_PIX].reshape(HALF_H, W)])

# GBRG:
#   G1 (tl)  = frame[0::2, 0::2]
#   B  (tr)  = frame[0::2, 1::2]
#   R  (bl)  = frame[1::2, 0::2]
#   G2 (br)  = frame[1::2, 1::2]

G1 = frame[0::2, 0::2]
G2 = frame[1::2, 1::2]
B  = frame[0::2, 1::2]
R  = frame[1::2, 0::2]

print("=== Статистика ===")
for name, ch in [("G1", G1), ("G2", G2), ("R", R), ("B", B)]:
    print(f"  {name}: mean={ch.mean():.1f} std={ch.std():.1f} "
          f"min={ch.min()} max={ch.max()}")

print()
print("=== Попарная разница (средний |A-B|) ===")
def diff(a, b):
    return float(np.abs(a.astype(np.int32) - b.astype(np.int32)).mean())

print(f"  |G1 - G2| = {diff(G1, G2):.2f}")
print(f"  |G1 - R|  = {diff(G1, R):.2f}")
print(f"  |G1 - B|  = {diff(G1, B):.2f}")
print(f"  |G2 - R|  = {diff(G2, R):.2f}")
print(f"  |G2 - B|  = {diff(G2, B):.2f}")
print(f"  |R  - B|  = {diff(R, B):.2f}")

print()
print("=== Корреляции ===")
def corr(a, b):
    return float(np.corrcoef(a.flatten().astype(np.float32),
                             b.flatten().astype(np.float32))[0, 1])

print(f"  corr(G1, G2) = {corr(G1, G2):+.4f}")
print(f"  corr(G1, R)  = {corr(G1, R):+.4f}")
print(f"  corr(G1, B)  = {corr(G1, B):+.4f}")
print(f"  corr(R,  B)  = {corr(R, B):+.4f}")

print()
print("=== Проверка: одинаковые ли данные? ===")
print(f"  G1 идентично G2? {np.array_equal(G1, G2)}")
print(f"  G1 идентично G2 бит-в-бит? "
      f"{np.array_equal(G1.view(np.uint8), G2.view(np.uint8))}")
print(f"  Кол-во отличающихся пикселей G1 vs G2: "
      f"{int((G1 != G2).sum()):,} из {G1.size:,} "
      f"({100 * (G1 != G2).mean():.2f}%)")