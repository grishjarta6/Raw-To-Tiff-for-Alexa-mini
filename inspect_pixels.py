"""Печатает значения пикселей в маленьком регионе, чтобы понять Bayer."""
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

top = c1[:CHUNK_PIX].reshape(HALF_H, W)
bot = c2[:CHUNK_PIX].reshape(HALF_H, W)
frame = np.vstack([top, bot])

print("Значения пикселей 8×8 в точке (y=500, x=1000):")
y0, x0 = 500, 1000
sub = frame[y0:y0+8, x0:x0+8]
print("     " + " ".join(f"x{x0+j:<5}" for j in range(8)))
for i in range(8):
    print(f"y{y0+i:<4} " +
          " ".join(f"{sub[i,j]:<6}" for j in range(8)))

print()
print("Значения пикселей 8×8 в точке (y=1500, x=1000) — НИЖНЯЯ половина:")
y0 = 1500
sub = frame[y0:y0+8, x0:x0+8]
for i in range(8):
    print(f"y{y0+i:<4} " +
          " ".join(f"{sub[i,j]:<6}" for j in range(8)))

print()
print("Средние по 4 позициям Bayer (чётные/нечётные строки и столбцы):")
print(f"  (even,even): {frame[0::2, 0::2].mean():.1f}")
print(f"  (even,odd):  {frame[0::2, 1::2].mean():.1f}")
print(f"  (odd, even): {frame[1::2, 0::2].mean():.1f}")
print(f"  (odd, odd):  {frame[1::2, 1::2].mean():.1f}")
print()
print("Отдельно для верхней половины (y<1101):")
print(f"  (even,even): {frame[:1101, 0::2][0::2, :].mean():.1f}")
print(f"  (even,odd):  {frame[:1101, 1::2][0::2, :].mean():.1f}")
print()
print("Отдельно для нижней половины (y>=1101):")
print(f"  (odd, even): {frame[1101:, 0::2][0::2, :].mean():.1f}")
print(f"  (odd, odd):  {frame[1101:, 1::2][0::2, :].mean():.1f}")