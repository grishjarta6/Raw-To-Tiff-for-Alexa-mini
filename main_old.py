#!/usr/bin/env python3
"""
Разделение ARRIRAW (.ari / .mxf) на 4 монохромных 16-битных TIFF.

Режимы:
  --list-modes   список именованных режимов и формул
  --bruteforce   перебор всех (pattern × p1 × p2) с оценкой std
  --p1=N --p2=M  явное указание формул (см. --list-modes)
  --sh=K         смещение данных в essence element
  --diagnose     таблица std по sh=0..11
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile

import mxf_parser


# ===========================================================================
# КОНФИГУРАЦИЯ
# ===========================================================================

MANUAL_W = 3424
MANUAL_H = 2202

ESSENCE_SUBHEADER = 0
DEFAULT_BAYER     = "BGGR"
DEFAULT_MODE      = "interleave_msb"

MIN_STD = 50.0


BAYER_PATTERNS = {
    "GRBG": ("G1", "R",  "B",  "G2"),
    "BGGR": ("B",  "G2", "G1", "R"),
    "RGGB": ("R",  "G1", "G2", "B"),
    "GBRG": ("G1", "B",  "R",  "G2"),
}


# ===========================================================================
# 24 ФОРМУЛЫ РАСПАКОВКИ 12-БИТ
# ===========================================================================
# Каждая формула: f(b0, b1, b2) -> массив 12-битных значений (uint16)
# b0, b1, b2 — numpy-векторы uint16 (по 8 бит).
# ---------------------------------------------------------------------------

FORMULAS = [
    #  0: b0 — high8, b1[7:4] — low4
    lambda b0, b1, b2: (b0 << 4) | (b1 >> 4),
    #  1: b0 — high8, b1[3:0] — low4
    lambda b0, b1, b2: (b0 << 4) | (b1 & 0x0F),
    #  2: b0 — high8, b2[7:4] — low4
    lambda b0, b1, b2: (b0 << 4) | (b2 >> 4),
    #  3: b0 — high8, b2[3:0] — low4
    lambda b0, b1, b2: (b0 << 4) | (b2 & 0x0F),
    #  4: b1 — high8, b0[7:4] — low4
    lambda b0, b1, b2: (b1 << 4) | (b0 >> 4),
    #  5: b1 — high8, b0[3:0] — low4
    lambda b0, b1, b2: (b1 << 4) | (b0 & 0x0F),
    #  6: b1 — high8, b2[7:4] — low4
    lambda b0, b1, b2: (b1 << 4) | (b2 >> 4),
    #  7: b1 — high8, b2[3:0] — low4
    lambda b0, b1, b2: (b1 << 4) | (b2 & 0x0F),
    #  8: b2 — high8, b0[7:4] — low4
    lambda b0, b1, b2: (b2 << 4) | (b0 >> 4),
    #  9: b2 — high8, b0[3:0] — low4
    lambda b0, b1, b2: (b2 << 4) | (b0 & 0x0F),
    # 10: b2 — high8, b1[7:4] — low4
    lambda b0, b1, b2: (b2 << 4) | (b1 >> 4),
    # 11: b2 — high8, b1[3:0] — low4
    lambda b0, b1, b2: (b2 << 4) | (b1 & 0x0F),
    # 12: b0[3:0] — high4, b1 — low8
    lambda b0, b1, b2: ((b0 & 0x0F) << 8) | b1,
    # 13: b0[3:0] — high4, b2 — low8
    lambda b0, b1, b2: ((b0 & 0x0F) << 8) | b2,
    # 14: b1[3:0] — high4, b0 — low8
    lambda b0, b1, b2: ((b1 & 0x0F) << 8) | b0,
    # 15: b1[3:0] — high4, b2 — low8
    lambda b0, b1, b2: ((b1 & 0x0F) << 8) | b2,
    # 16: b2[3:0] — high4, b0 — low8
    lambda b0, b1, b2: ((b2 & 0x0F) << 8) | b0,
    # 17: b2[3:0] — high4, b1 — low8
    lambda b0, b1, b2: ((b2 & 0x0F) << 8) | b1,
    # 18: b0[7:4] — high4, b1 — low8
    lambda b0, b1, b2: ((b0 >> 4) << 8) | b1,
    # 19: b0[7:4] — high4, b2 — low8
    lambda b0, b1, b2: ((b0 >> 4) << 8) | b2,
    # 20: b1[7:4] — high4, b0 — low8
    lambda b0, b1, b2: ((b1 >> 4) << 8) | b0,
    # 21: b1[7:4] — high4, b2 — low8
    lambda b0, b1, b2: ((b1 >> 4) << 8) | b2,
    # 22: b2[7:4] — high4, b0 — low8
    lambda b0, b1, b2: ((b2 >> 4) << 8) | b0,
    # 23: b2[7:4] — high4, b1 — low8
    lambda b0, b1, b2: ((b2 >> 4) << 8) | b1,
]


FORMULA_DESCRIPTIONS = [
    "b0[7:0] | b1[7:4]",
    "b0[7:0] | b1[3:0]",
    "b0[7:0] | b2[7:4]",
    "b0[7:0] | b2[3:0]",
    "b1[7:0] | b0[7:4]",
    "b1[7:0] | b0[3:0]",
    "b1[7:0] | b2[7:4]",
    "b1[7:0] | b2[3:0]",
    "b2[7:0] | b0[7:4]",
    "b2[7:0] | b0[3:0]",
    "b2[7:0] | b1[7:4]",
    "b2[7:0] | b1[3:0]",
    "b0[3:0] | b1[7:0]",
    "b0[3:0] | b2[7:0]",
    "b1[3:0] | b0[7:0]",
    "b1[3:0] | b2[7:0]",
    "b2[3:0] | b0[7:0]",
    "b2[3:0] | b1[7:0]",
    "b0[7:4] | b1[7:0]",
    "b0[7:4] | b2[7:0]",
    "b1[7:4] | b0[7:0]",
    "b1[7:4] | b2[7:0]",
    "b2[7:4] | b0[7:0]",
    "b2[7:4] | b1[7:0]",
]


# ---------------------------------------------------------------------------
# Именованные режимы (совместимость)
# ---------------------------------------------------------------------------
def _make_mode(f1_idx, f2_idx):
    f1 = FORMULAS[f1_idx]
    f2 = FORMULAS[f2_idx]
    def _fn(b0, b1, b2):
        return f1(b0, b1, b2), f2(b0, b1, b2)
    return _fn


UNPACK_MODES = {
    "std":                 _make_mode(0, 15),
    "nibble_swap":         _make_mode(1, 21),
    "interleave_msb":      _make_mode(2, 7),
    "interleave_msb_swap": _make_mode(6, 3),
    "alt_pair":            _make_mode(2, 17),
    "alt_pair_2":          _make_mode(6, 16),
    "le_full":             _make_mode(18, 21),
    "le_full_swap":        _make_mode(20, 19),
    "le_nibble":           _make_mode(14, 10),
    "le_nibble_swap":      _make_mode(15, 8),
    "std_rev":             _make_mode(15, 0),
    "interleave_rev":      _make_mode(7, 2),
}


# ===========================================================================
# 1. ЛОКАЛЬНЫЙ FFMPEG
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent
FFPROBE_PATH = None

ffprobe_candidates = list(BASE_DIR.glob("ffmpeg-*/bin/ffprobe.exe"))
if ffprobe_candidates:
    FFPROBE_PATH = str(ffprobe_candidates[0])
    os.environ["PATH"] = (str(ffprobe_candidates[0].parent)
                          + os.pathsep + os.environ["PATH"])
    print(f"✅ Найден локальный ffprobe: {FFPROBE_PATH}")
elif shutil.which("ffprobe"):
    FFPROBE_PATH = "ffprobe"
    print("✅ Используется системный ffprobe")
else:
    raise RuntimeError("❌ ffprobe не найден!")


# ===========================================================================
# 2. FFPROBE
# ===========================================================================

def _run_ffprobe(args):
    cmd = [FFPROBE_PATH, "-v", "error"] + args + ["-of", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe error:\n{res.stderr.strip()}")
    return json.loads(res.stdout)


def probe_stream(filepath):
    data = _run_ffprobe(["-show_streams", "-select_streams", "v:0",
                         str(filepath)])
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"Видеопоток не найден в {filepath.name}")
    return streams[0]


def probe_first_packet(filepath):
    try:
        data = _run_ffprobe([
            "-select_streams", "v:0",
            "-show_packets",
            "-read_intervals", "%+#1",
            "-show_entries", "packet=pos,size",
            str(filepath),
        ])
    except RuntimeError:
        return None, 0
    packets = data.get("packets", [])
    if not packets:
        return None, 0
    pkt = packets[0]
    size = int(pkt.get("size", 0))
    return (int(pkt["pos"]) if "pos" in pkt else None), size


# ===========================================================================
# 3. АНАЛИЗ ФАЙЛА
# ===========================================================================

def analyse(filepath, verbose=False, manual_w=None, manual_h=None):
    info = {
        "path": filepath, "width": None, "height": None,
        "header_offset": None, "codec": "unknown", "pix_fmt": "unknown",
        "packet_size": 0, "offset_method": "unknown",
        "frame_count": None, "frame_size": None,
    }

    try:
        stream = probe_stream(filepath)
        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))
        if w > 0 and h > 0:
            info["width"]   = w
            info["height"]  = h
            info["codec"]   = stream.get("codec_name", "unknown")
            info["pix_fmt"] = stream.get("pix_fmt", "unknown")
    except Exception as e:
        if verbose:
            print(f"   ⚠ ffprobe (streams): {e}")

    if info["width"] is None:
        if manual_w and manual_h:
            info["width"], info["height"] = manual_w, manual_h
            print(f"   ℹ Ручное разрешение: {manual_w}×{manual_h}")
        else:
            raise RuntimeError("Задайте MANUAL_W / MANUAL_H в main.py.")

    if filepath.suffix.lower() == ".ari":
        info["header_offset"] = 4096
        info["offset_method"] = "fixed (ari)"
        return info

    pos, size = probe_first_packet(filepath)
    info["packet_size"] = size

    if pos is not None and pos > 0:
        info["header_offset"] = pos
        info["offset_method"] = "ffprobe"
        return info

    print("   ⚠ ffprobe не вернул pos, использую mxf_parser...")
    result = mxf_parser.parse(filepath, verbose=verbose)
    info["header_offset"] = result["header_offset"]
    info["offset_method"] = "mxf_parser"
    info["essence_key"]   = result["essence_key"]
    info["klv_packets"]   = result["packet_count"]
    info["frame_count"]   = result.get("frame_count")
    info["frame_size"]    = result.get("frame_size")
    return info


# ===========================================================================
# 4. РАСПАКОВКА
# ===========================================================================

def unpack_with(raw, width, height, f1_idx, f2_idx):
    """Распаковка с явными индексами формул."""
    total_pixels = width * height
    bytes_needed = total_pixels * 3 // 2
    if len(raw) < bytes_needed:
        raise ValueError(f"Мало данных: {len(raw)} < {bytes_needed}")

    data = np.frombuffer(raw[:bytes_needed], dtype=np.uint8)
    b0 = data[0::3].astype(np.uint16)
    b1 = data[1::3].astype(np.uint16)
    b2 = data[2::3].astype(np.uint16)

    p1 = FORMULAS[f1_idx](b0, b1, b2)
    p2 = FORMULAS[f2_idx](b0, b1, b2)

    bayer = np.empty(total_pixels, dtype=np.uint16)
    bayer[0::2] = p1
    bayer[1::2] = p2
    return bayer.reshape((height, width))


def unpack_frame(raw, width, height, mode):
    """Именованный режим."""
    total_pixels = width * height
    bytes_needed = total_pixels * 3 // 2
    if len(raw) < bytes_needed:
        raise ValueError(f"Мало данных: {len(raw)} < {bytes_needed}")

    data = np.frombuffer(raw[:bytes_needed], dtype=np.uint8)
    b0 = data[0::3].astype(np.uint16)
    b1 = data[1::3].astype(np.uint16)
    b2 = data[2::3].astype(np.uint16)

    p1, p2 = UNPACK_MODES[mode](b0, b1, b2)
    bayer = np.empty(total_pixels, dtype=np.uint16)
    bayer[0::2] = p1
    bayer[1::2] = p2
    return bayer.reshape((height, width))


def split_into_channels(bayer, pattern):
    tl, tr, bl, br = BAYER_PATTERNS[pattern]
    return {
        tl: bayer[0::2, 0::2],
        tr: bayer[0::2, 1::2],
        bl: bayer[1::2, 0::2],
        br: bayer[1::2, 1::2],
    }


def _stats(img):
    return {"min": int(img.min()), "max": int(img.max()),
            "mean": float(img.mean()), "std": float(img.std())}


# ===========================================================================
# 5. BRUTE-FORCE
# ===========================================================================

def bruteforce(filepath, off, w, h, sample_rows=20, sh=0):
    """
    Перебирает pattern × p1 ∈ [0..24) × p2 ∈ [0..24) = 4*24*24 = 2304 комбинации.
    Оценка: min(std всех 4 каналов). Больше = лучше.
    """
    total_pixels = w * sample_rows
    bytes_needed = total_pixels * 3 // 2
    with open(filepath, "rb") as f:
        f.seek(off + sh)
        raw = f.read(bytes_needed)
    if len(raw) < bytes_needed:
        raise RuntimeError("Мало данных для brute-force")

    data = np.frombuffer(raw[:bytes_needed], dtype=np.uint8)
    b0 = data[0::3].astype(np.uint16)
    b1 = data[1::3].astype(np.uint16)
    b2 = data[2::3].astype(np.uint16)

    # Предвычисляем все 24 формулы для p1 и p2
    p1_values = [f(b0, b1, b2) for f in FORMULAS]
    p2_values = [f(b0, b1, b2) for f in FORMULAS]

    results = []
    for pattern in BAYER_PATTERNS:
        tl, tr, bl, br = BAYER_PATTERNS[pattern]

        for i1 in range(24):
            p1 = p1_values[i1]
            for i2 in range(24):
                p2 = p2_values[i2]

                bayer = np.empty(total_pixels, dtype=np.uint16)
                bayer[0::2] = p1
                bayer[1::2] = p2
                bayer = bayer.reshape((sample_rows, w))

                chans = {
                    tl: bayer[0::2, 0::2],
                    tr: bayer[0::2, 1::2],
                    bl: bayer[1::2, 0::2],
                    br: bayer[1::2, 1::2],
                }

                stds = {k: float(v.std()) for k, v in chans.items()}
                min_std = min(stds.values())
                mean_std = sum(stds.values()) / 4

                results.append({
                    "pattern": pattern, "p1": i1, "p2": i2,
                    "min_std": min_std, "mean_std": mean_std,
                    "stds": stds,
                })

    results.sort(key=lambda x: x["min_std"], reverse=True)
    return results


# ===========================================================================
# 6. ДИАГНОСТИКА
# ===========================================================================

def diagnose(filepath, off, w, h, mode, pattern, sh_range=range(0, 12)):
    expected = w * h * 3 // 2
    max_sh = max(sh_range)
    with open(filepath, "rb") as f:
        f.seek(off)
        raw = f.read(max_sh + expected)

    ch_names = list(BAYER_PATTERNS[pattern])
    print(f"\n   🔬 Диагностика: pattern={pattern}, mode={mode}")
    print()
    print("      {:>3} | ".format("sh") +
          " | ".join(f"{c:<28}" for c in ch_names))
    print("      " + "-" * (5 + 31 * len(ch_names)))

    for sh in sh_range:
        chunk = raw[sh:sh + expected]
        if len(chunk) < expected:
            break
        try:
            bayer = unpack_frame(chunk, w, h, mode)
        except Exception:
            continue
        chans = split_into_channels(bayer, pattern)
        cells = [f"mn{_stats(chans[c])['min']:>4} "
                 f"mx{_stats(chans[c])['max']:>4} "
                 f"μ{_stats(chans[c])['mean']:>6.0f} "
                 f"σ{_stats(chans[c])['std']:>6.1f}" for c in ch_names]
        print(f"      {sh:>3} | " + " | ".join(cells))


# ===========================================================================
# 7. ОБРАБОТКА ФАЙЛА
# ===========================================================================

def process_file(filepath, output_dir=None, verbose=False,
                 manual_w=None, manual_h=None,
                 bayer=None, mode=None,
                 p1=None, p2=None, sh=None,
                 do_diagnose=False, do_bruteforce=False):

    print(f"\n📂 {filepath.name}")

    info = analyse(filepath, verbose=verbose,
                   manual_w=manual_w, manual_h=manual_h)
    w, h, off = info["width"], info["height"], info["header_offset"]

    print(f"   Кодек:            {info['codec']} ({info['pix_fmt']})")
    print(f"   Разрешение:       {w}×{h}")
    print(f"   Смещение данных:  {off:,} байт "
          f"({off / 1024:.1f} КБ, метод: {info['offset_method']})")

    expected_packed = w * h * 3 // 2

    if info.get("frame_size"):
        print(f"   Кадров:           {info['frame_count']}")
        print(f"   Размер кадра:     {info['frame_size']:,} байт "
              f"(ожидаемый 12-бит: {expected_packed:,})")
        delta = info["frame_size"] - expected_packed
        if delta != 0:
            print(f"   ℹ Разница:        {delta:+d} байт")

    bayer = bayer or DEFAULT_BAYER
    mode  = mode  or DEFAULT_MODE

    # --- Brute-force
    if do_bruteforce:
        sh_use = sh if sh is not None else 0
        print(f"\n   🔬 BRUTE-FORCE: перебор 4 patterns × 24 p1 × 24 p2 "
              f"= 2304 комбинаций (sh={sh_use})...")

        results = bruteforce(filepath, off, w, h, sample_rows=20, sh=sh_use)

        print(f"\n      Топ-20 по min(std по 4 каналам):")
        print(f"      {'#':>3}  {'pattern':<6} {'p1':>3} {'p2':>3}  "
              f"{'min_std':>8}  {'mean_std':>9}  детали")
        for i, r in enumerate(results[:20], 1):
            detail = "  ".join(f"{k}={v:>6.0f}" for k, v in r["stds"].items())
            print(f"      {i:>3}  {r['pattern']:<6} {r['p1']:>3} {r['p2']:>3}  "
                  f"{r['min_std']:>8.1f}  {r['mean_std']:>9.1f}  {detail}")

        best = results[0]
        print(f"\n      ✅ Лучшее: pattern={best['pattern']} "
              f"p1={best['p1']} p2={best['p2']}  "
              f"(min_std={best['min_std']:.1f})")
        print(f"         p1: {FORMULA_DESCRIPTIONS[best['p1']]}")
        print(f"         p2: {FORMULA_DESCRIPTIONS[best['p2']]}")
        return

    # --- Обычная диагностика
    if do_diagnose:
        diagnose(filepath, off, w, h, mode, bayer, range(0, 12))
        return

    # --- Обработка
    if sh is None:
        sh = ESSENCE_SUBHEADER

    if p1 is not None and p2 is not None:
        print(f"   pattern={bayer}, sh={sh}, "
              f"p1={p1} ({FORMULA_DESCRIPTIONS[p1]}), "
              f"p2={p2} ({FORMULA_DESCRIPTIONS[p2]})")
        unpack_fn = lambda raw: unpack_with(raw, w, h, p1, p2)
    else:
        print(f"   pattern={bayer}, mode={mode}, sh={sh}")
        unpack_fn = lambda raw: unpack_frame(raw, w, h, mode)

    with open(filepath, "rb") as f:
        f.seek(off + sh)
        raw = f.read(expected_packed)
    if len(raw) < expected_packed:
        raise ValueError(f"Мало данных: {len(raw)} < {expected_packed}")

    bayer_img = unpack_fn(raw)
    channels = split_into_channels(bayer_img, bayer)

    print(f"\n   📊 Статистика по каналам:")
    for ch, img in channels.items():
        s = _stats(img)
        marker = "✅" if s["std"] > 200 else "⚠"
        print(f"      {marker} {ch:<3}: "
              f"min={s['min']:>5}, max={s['max']:>5}, "
              f"mean={s['mean']:>8.1f}, std={s['std']:>7.1f}")

    if output_dir is None:
        output_dir = filepath.parent / f"{filepath.stem}_channels"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = output_dir / filepath.stem
    for ch, img in channels.items():
        tifffile.imwrite(f"{base}_{ch}.tiff", img)

    print(f"\n   ✅ Сохранено: {output_dir}")


# ===========================================================================
# 8. ФАЙЛЫ И МЕНЮ
# ===========================================================================

def find_raw_files(directory):
    exts = ("*.mxf", "*.MXF", "*.ari", "*.ARI")
    found = set()
    for ext in exts:
        found.update(directory.glob(ext))
    return sorted(found)


def interactive_menu(files):
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
# 9. ТОЧКА ВХОДА
# ===========================================================================

def _get_opt(flags, name):
    prefix = name + "="
    for f in flags:
        if f.startswith(prefix):
            return f[len(prefix):]
    return None


def main():
    args  = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]

    verbose       = "--verbose" in flags or "-v" in flags
    do_diagnose   = "--diagnose" in flags
    do_bruteforce = "--bruteforce" in flags
    show_modes    = "--list-modes" in flags

    if show_modes:
        print("Именованные режимы:")
        for m in UNPACK_MODES:
            print(f"   {m}")
        print("\nФормулы (для --p1=N / --p2=M):")
        for i, desc in enumerate(FORMULA_DESCRIPTIONS):
            print(f"   {i:>2}: {desc}")
        return

    bayer = _get_opt(flags, "--bayer")
    mode  = _get_opt(flags, "--mode")
    sh    = _get_opt(flags, "--sh")
    p1    = _get_opt(flags, "--p1")
    p2    = _get_opt(flags, "--p2")

    if bayer and bayer not in BAYER_PATTERNS:
        print(f"❌ Неизвестный паттерн: {bayer}")
        sys.exit(1)
    if mode and mode not in UNPACK_MODES:
        print(f"❌ Неизвестный режим: {mode}")
        sys.exit(1)

    sh_val = int(sh) if sh is not None else None
    p1_val = int(p1) if p1 is not None else None
    p2_val = int(p2) if p2 is not None else None

    if p1_val is not None and not (0 <= p1_val < 24):
        print(f"❌ p1 вне диапазона 0..23")
        sys.exit(1)
    if p2_val is not None and not (0 <= p2_val < 24):
        print(f"❌ p2 вне диапазона 0..23")
        sys.exit(1)

    target = Path(args[0]) if args else Path.cwd()

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = find_raw_files(target)
        if not files:
            print(f"В директории {target} не найдено .mxf/.ari файлов")
            sys.exit(1)
        files = interactive_menu(files)
        if not files:
            return
    else:
        print(f"❌ Путь не найден: {target}")
        sys.exit(1)

    for f in files:
        try:
            process_file(f,
                         verbose=verbose,
                         manual_w=MANUAL_W,
                         manual_h=MANUAL_H,
                         bayer=bayer, mode=mode,
                         p1=p1_val, p2=p2_val, sh=sh_val,
                         do_diagnose=do_diagnose,
                         do_bruteforce=do_bruteforce)
        except Exception as e:
            print(f"❌ Ошибка при обработке {f.name}: {e}")


if __name__ == "__main__":
    main()