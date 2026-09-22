"""Гипотеза: 16-бит моно, 3 байта = 1.5 пикселя → отбрасываем каждый 3-й байт."""
import numpy as np
from pathlib import Path
import tifffile
import mxf_parser

MXF_FILE = "A003C007_191110_R3MG.mxf"
W, H = 3424, 2202
SH = 76

off = mxf_parser.parse(MXF_FILE)["header_offset"]
total = W * H

# Читаем с запасом
n_bytes = total * 3 // 2
with open(MXF_FILE, "rb") as f:
    f.seek(off + SH)
    raw = f.read(n_bytes)

d = np.frombuffer(raw, dtype=np.uint8)
b0 = d[0::3].astype(np.uint16)
b1 = d[1::3].astype(np.uint16)
b2 = d[2::3].astype(np.uint16)

print("Статистика байтов:")
print(f"  b0:  min={b0.min()} max={b0.max()} std={b0.std():.1f}")
print(f"  b1:  min={b1.min()} max={b1.max()} std={b1.std():.1f}")
print(f"  b2:  min={b2.min()} max={b2.max()} std={b2.std():.1f}")
print(f"  b2 & 0x0F: уникальные значения = {np.unique(b2 & 0x0F)[:20]}")
print(f"  b2 >> 4:   уникальные значения = {np.unique(b2 >> 4)[:20]}")
print(f"  b2 == 1:   доля = {(b2 == 1).mean()*100:.2f}%")