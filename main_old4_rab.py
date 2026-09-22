#!/usr/bin/env python3
"""
ARRIRAW ALEXA Mini MXF → 4 полных монохромных 16-битных TIFF.

Структура essence элемента (11 309 548 байт = один кадр):
    [  76 байт  ]  frame header
    [ 5 654 736 ]  chunk 1: ВЕРХНЯЯ половина кадра  (1101 × 3424 пикс)
    [ 5 654 736 ]  chunk 2: НИЖНЯЯ  половина кадра  (1101 × 3424 пикс)

Формула распаковки 12-бит (arri_alt):
    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

import mxf_parser


# ===========================================================================
# КОНСТАНТЫ
# ===========================================================================

MXF_FILE  = "A003C007_191110_R3MG.mxf"
W, H      = 3424, 2202
HDR       = 76                       # заголовок кадра
HALF_H    = H // 2                   # 1101
CHUNK_PIX = HALF_H * W               # 3 769 824 пикселей
CHUNK_BYTES = CHUNK_PIX * 3 // 2     # 5 654 736 байт
FRAME_SIZE = HDR + 2 * CHUNK_BYTES   # 11 309 548


PATTERNS = {
    # (top-left, top-right, bottom-left, bottom-right)
    "BGGR": ("B",  "G2", "G1", "R"),
    "GRBG": ("G1", "R",  "B",  "G2"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


# ===========================================================================
# РАСПАКОВКА
# ===========================================================================

def unpack_chunk(raw: bytes) -> np.ndarray:
    """
    5 654 736 байт → 3 769 824 пикселей (uint16).
    Формула arri_alt.
    """
    n_pairs = len(raw) // 3
    d = np.frombuffer(raw[:n_pairs * 3], dtype=np.uint8)
    b0 = d[0::3].astype(np.uint16)
    b1 = d[1::3].astype(np.uint16)
    b2 = d[2::3].astype(np.uint16)

    p1 = (b0 << 4) | (b2 >> 4)
    p2 = ((b2 & 0x0F) << 8) | b1

    out = np.empty(len(b0) * 2, dtype=np.uint16)
    out[0::2] = p1
    out[1::2] = p2
    return out


def assemble_frame(chunk_top: np.ndarray, chunk_bottom: np.ndarray) -> np.ndarray:
    """Верхняя половина из chunk1, нижняя из chunk2."""
    top    = chunk_top[:CHUNK_PIX].reshape(HALF_H, W)
    bottom = chunk_bottom[:CHUNK_PIX].reshape(HALF_H, W)
    frame  = np.empty((H, W), dtype=np.uint16)
    frame[:HALF_H, :] = top
    frame[HALF_H:, :] = bottom
    return frame


def split_channels(frame: np.ndarray, pattern: str) -> dict:
    tl, tr, bl, br = PATTERNS[pattern]
    return {
        tl: frame[0::2, 0::2],
        tr: frame[0::2, 1::2],
        bl: frame[1::2, 0::2],
        br: frame[1::2, 1::2],
    }


def build_rgb(chans: dict) -> np.ndarray:
    """Псевдо-RGB из 4 каналов (полный размер)."""
    R  = chans["R"].astype(np.float32)
    B  = chans["B"].astype(np.float32)
    G1 = chans["G1"].astype(np.float32)
    G2 = chans["G2"].astype(np.float32)
    G  = 0.5 * (G1 + G2)

    def norm(x):
        p = np.percentile(x, 99.0)
        return np.clip(x / max(p, 1), 0, 1)

    rgb = np.stack([norm(R), norm(G), norm(B)], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ===========================================================================
# ОБРАБОТКА ОДНОГО КАДРА
# ===========================================================================

def process_frame(path: Path, frame_idx: int, pattern: str,
                  out_dir: Path, save_rgb: bool = True) -> None:
    off = mxf_parser.parse(path)["header_offset"]
    pos = off + frame_idx * FRAME_SIZE

    with open(path, "rb") as f:
        f.seek(pos)
        raw = f.read(FRAME_SIZE)

    if len(raw) < FRAME_SIZE:
        raise ValueError(
            f"Кадр {frame_idx} не помещается в файле: "
            f"прочитано {len(raw)} из {FRAME_SIZE}"
        )

    c1_raw = raw[HDR : HDR + CHUNK_BYTES]
    c2_raw = raw[HDR + CHUNK_BYTES : HDR + 2 * CHUNK_BYTES]

    c1 = unpack_chunk(c1_raw)
    c2 = unpack_chunk(c2_raw)

    frame = assemble_frame(c1, c2)
    chans = split_channels(frame, pattern)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"frame{frame_idx:04d}_{pattern}"

    for name, img in chans.items():
        tifffile.imwrite(f"{base}_{name}.tiff", img)
        print(f"   💾 {base.name}_{name}.tiff   "
              f"({img.shape[1]}×{img.shape[0]}, "
              f"std={img.std():.0f}, "
              f"min={img.min()}, max={img.max()})")

    if save_rgb:
        rgb = build_rgb(chans)
        tifffile.imwrite(f"{base}_RGB.tiff", rgb)
        print(f"   💾 {base.name}_RGB.tiff   "
              f"({rgb.shape[1]}×{rgb.shape[0]}, псевдо-RGB)")


# ===========================================================================
# ПОИСК ФАЙЛОВ И МЕНЮ
# ===========================================================================

def find_raw_files(directory: Path) -> list[Path]:
    found = set()
    for ext in ("*.mxf", "*.MXF", "*.ari", "*.ARI"):
        found.update(directory.glob(ext))
    return sorted(found)


def interactive_menu(files: list[Path]) -> list[Path]:
    print("\n📁 Найдены файлы:")
    for i, f in enumerate(files, 1):
        mb = f.stat().st_size / (1024 * 1024)
        print(f"   [{i:>2}] {f.name}  ({mb:.1f} МБ)")
    print("   [a]  Обработать все")
    print("   [q]  Выход")
    choice = input("\nВыбор: ").strip().lower()
    if choice == "q":
        return []
    if choice == "a":
        return files
    try:
        return [files[int(choice) - 1]]
    except (ValueError, IndexError):
        print("❌ Неверный выбор")
        return []


# ===========================================================================
# CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="ARRIRAW ALEXA Mini MXF → 4 полных TIFF."
    )
    ap.add_argument("--file", default=None,
                    help="MXF файл (по умолчанию — из папки запуска)")
    ap.add_argument("--frame", type=int, default=0,
                    help="номер кадра (0..N-1)")
    ap.add_argument("--all-frames", action="store_true",
                    help="экспортировать все кадры")
    ap.add_argument("--pattern", default="BGGR",
                    choices=list(PATTERNS))
    ap.add_argument("--no-rgb", action="store_true",
                    help="не сохранять псевдо-RGB превью")
    ap.add_argument("--out", default=None,
                    help="папка вывода (по умолчанию <file>_channels)")
    args = ap.parse_args()

    # Определяем файл(ы)
    if args.file:
        target = Path(args.file)
        if not target.exists():
            print(f"❌ Файл не найден: {target}")
            sys.exit(1)
        files = [target]
    else:
        cwd = Path.cwd()
        files = find_raw_files(cwd)
        if not files:
            print(f"В папке {cwd} не найдено .mxf/.ari")
            sys.exit(1)
        if len(files) == 1:
            files = files
            print(f"📂 Использую единственный файл: {files[0].name}")
        else:
            files = interactive_menu(files)
            if not files:
                return

    for filepath in files:
        print(f"\n📂 {filepath.name}")
        off = mxf_parser.parse(filepath)["header_offset"]
        total_frames = (filepath.stat().st_size - off) // FRAME_SIZE
        print(f"   Essence offset: {off:,}")
        print(f"   Всего кадров:   {total_frames}")
        print()

        out_dir = Path(args.out) if args.out else \
                  filepath.parent / f"{filepath.stem}_channels"

        if args.all_frames:
            frame_list = list(range(total_frames))
        else:
            if args.frame >= total_frames:
                print(f"❌ Кадр {args.frame} вне диапазона "
                      f"(0..{total_frames - 1})")
                continue
            frame_list = [args.frame]

        for idx in frame_list:
            print(f"   --- кадр {idx} ---")
            try:
                process_frame(filepath, idx, args.pattern, out_dir,
                              save_rgb=not args.no_rgb)
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")

        print(f"\n   ✅ Готово: {out_dir}")


if __name__ == "__main__":
    main()