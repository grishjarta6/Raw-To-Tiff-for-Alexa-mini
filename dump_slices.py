"""
Печатает hex 48 байт с нескольких смещений внутри essence.
"""
from pathlib import Path
import mxf_parser

MXF_FILE = "A003C007_191110_R3MG.mxf"
OFFSETS = [0, 100, 1000, 10000, 100_000, 1_000_000, 5_000_000, 10_000_000]

path = Path(MXF_FILE)
base = mxf_parser.parse(path)["header_offset"]
size = path.stat().st_size

print(f"Файл:         {path.name}  ({size:,} байт)")
print(f"Essence base: {base:,}")
print()

with open(path, "rb") as f:
    for o in OFFSETS:
        pos = base + o
        if pos + 48 > size:
            print(f"--- base + {o:,} — за пределами файла ---")
            continue
        f.seek(pos)
        d = f.read(48)
        hex_str = " ".join(f"{b:02x}" for b in d)
        # Группируем по 3 байта для наглядности
        groups = "  ".join(
            " ".join(f"{d[i+j]:02x}" for j in range(3))
            for i in range(0, 48, 3)
        )
        print(f"--- base + {o:>12,} ---")
        print(f"  raw: {hex_str}")
        print(f"  tri: {groups}")
        print()
        