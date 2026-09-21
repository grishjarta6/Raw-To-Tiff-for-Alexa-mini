#!/usr/bin/env python3
"""
Разделение ARRIRAW (.ari / .mxf) на 4 монохромных 16-битных TIFF.

Определение смещения данных сенсора:
  1. ffprobe (packet.pos первого пакета) — работает для большинства MXF.
  2. mxf_parser (Python-сканер KLV + ARRIRAW-специфичный UL) — fallback.

Режим --auto-sh подбирает ESSENCE_SUBHEADER ОТДЕЛЬНО для каждого канала
(R, G1, B, G2), потому что в некоторых MXF-контейнерах чётные и нечётные
пиксельные позиции имеют разные смещения.

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

# Для ALEXA Mini 3.4K Open Gate: 3424×2202
# Для ALEXA Mini 2.8K 16:9:     2880×1620
# Для ALEXA Mini 4:3 2.8K:      2880×2160
MANUAL_W = 3424
MANUAL_H = 2202

# Смещение пиксельных данных внутри essence element (для всех каналов).
# Если --auto-sh не задан, используется это значение.
ESSENCE_SUBHEADER = 0

# Диапазон и шаг авто-поиска subheader
AUTO_SH_MAX  = 200
AUTO_SH_STEP = 1
AUTO_SH_SAMPLE_ROWS = 200


# ===========================================================================
# 1. ПОИСК ЛОКАЛЬНОГО FFMPEG
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parent

FFPROBE_PATH = None

ffprobe_candidates = list(BASE_DIR.glob("ffmpeg-*/bin/ffprobe.exe"))
if ffprobe_candidates:
    FFPROBE_PATH = str(ffprobe_candidates[0])
    os.environ["PATH"] = str(ffprobe_candidates[0].parent) + os.pathsep + os.environ["PATH"]
    print(f"✅ Найден локальный ffprobe: {FFPROBE_PATH}")
elif shutil.which("ffprobe"):
    FFPROBE_PATH = "ffprobe"
    print("✅ Используется системный ffprobe")
else:
    raise RuntimeError("❌ ffprobe не найден!")


# ===========================================================================
# 2. ОБЁРТКИ НАД FFPROBE
# ===========================================================================

def _run_ffprobe(args: list[str]) -> dict:
    cmd = [FFPROBE_PATH, "-v", "error"] + args + ["-of", "json"]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe error:\n{res.stderr.strip()}")
    return json.loads(res.stdout)


def probe_stream(filepath: Path) -> dict:
    data = _run_ffprobe([
        "-show_streams", "-select_streams", "v:0", str(filepath),
    ])
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"Видеопоток не найден в {filepath.name}")
    return streams[0]


def probe_first_packet(filepath: Path) -> tuple[int | None, int]:
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
    if "pos" not in pkt:
        return None, size
    return int(pkt["pos"]), size


# ===========================================================================
# 3. АНАЛИЗ ФАЙЛА
# ===========================================================================

def analyse(filepath: Path,
            verbose: bool = False,
            manual_w: int | None = None,
            manual_h: int | None = None) -> dict:
    info = {
        "path":          filepath,
        "width":         None,
        "height":        None,
        "header_offset": None,
        "codec":         "unknown",
        "pix_fmt":       "unknown",
        "packet_size":   0,
        "offset_method": "unknown",
        "frame_count":   None,
        "frame_size":    None,
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
            print(f"   ℹ ffprobe не определил размеры, "
                  f"использую ручные: {manual_w}×{manual_h}")
        else:
            raise RuntimeError(
                "Не удалось определить разрешение. FFmpeg не поддерживает этот файл. "
                "Задайте MANUAL_W и MANUAL_H в начале main.py."
            )

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
# 4. РАСПАКОВКА 12-БИТ ARRIRAW
# ===========================================================================

def unpack_arriraw(raw: bytes, width: int, height: int) -> np.ndarray:
    """
    Распаковывает 12-битный поток ARRIRAW (2 пикселя = 3 байта)
    в 16-битную матрицу Байера.
    """
    total_pixels = width * height
    bytes_needed = total_pixels * 3 // 2
    if len(raw) < bytes_needed:
        raise ValueError(
            f"Недостаточно данных: {len(raw)} < {bytes_needed} "
            f"для {width}×{height}"
        )

    data = np.frombuffer(raw[:bytes_needed], dtype=np.uint8)
    b0 = data[0::3].astype(np.uint16)
    b1 = data[1::3].astype(np.uint16)
    b2 = data[2::3].astype(np.uint16)

    pix1 = (b0 << 4) | (b1 >> 4)
    pix2 = ((b1 & 0x0F) << 8) | b2

    bayer = np.empty(total_pixels, dtype=np.uint16)
    bayer[0::2] = pix1
    bayer[1::2] = pix2
    return bayer.reshape((height, width))


def extract_channel(bayer: np.ndarray, channel: str) -> np.ndarray:
    """Извлекает один канал из матрицы Байера GRBG."""
    if channel == "R":
        return bayer[0::2, 1::2]
    if channel == "G1":
        return bayer[0::2, 0::2]
    if channel == "B":
        return bayer[1::2, 0::2]
    if channel == "G2":
        return bayer[1::2, 1::2]
    raise ValueError(f"Неизвестный канал: {channel}")


# ===========================================================================
# 5. АВТО-ПОДБОР SUBHEADER (per-channel)
# ===========================================================================

def _roughness(img: np.ndarray) -> float:
    """Средний модуль горизонтального градиента. Меньше = глаже."""
    if img.size < 8:
        return float("inf")
    d = np.abs(np.diff(img.astype(np.int32), axis=1))
    return float(d.mean())


def find_best_subheaders_per_channel(
    filepath: Path,
    header_offset: int,
    width: int,
    height: int,
    max_subheader: int = AUTO_SH_MAX,
    sample_rows: int = AUTO_SH_SAMPLE_ROWS,
    step: int = AUTO_SH_STEP,
) -> dict[str, int]:
    """
    Подбирает ESSENCE_SUBHEADER отдельно для каждого канала.

    Возвращает: {"R": sh, "G1": sh, "B": sh, "G2": sh}
    """
    sample_h = min(sample_rows, height)
    sample_pixels = width * sample_h
    sample_bytes = sample_pixels * 3 // 2
    buf_size = max_subheader + sample_bytes

    with open(filepath, "rb") as f:
        f.seek(header_offset)
        raw = f.read(buf_size)

    if len(raw) < sample_bytes:
        raise RuntimeError("Недостаточно данных для анализа subheader")

    # Собираем статистику
    results: list[dict] = []
    for sh in range(0, max_subheader + 1, step):
        chunk = raw[sh:sh + sample_bytes]
        if len(chunk) < sample_bytes:
            break
        try:
            bayer = unpack_arriraw(chunk, width, sample_h)
        except Exception:
            continue

        results.append({
            "sh": sh,
            "R":  _roughness(extract_channel(bayer, "R")),
            "G1": _roughness(extract_channel(bayer, "G1")),
            "B":  _roughness(extract_channel(bayer, "B")),
            "G2": _roughness(extract_channel(bayer, "G2")),
        })

    if not results:
        raise RuntimeError("Не удалось оценить ни одного subheader")

    best = {}
    for ch in ("R", "G1", "B", "G2"):
        best[ch] = min(results, key=lambda x: x[ch])["sh"]

    # --- Диагностика
    print("\n   🔍 Авто-подбор ESSENCE_SUBHEADER (по каналам)")
    print(f"      Проверено {len(results)} значений "
          f"(0..{max_subheader}, шаг {step})")

    top = sorted(results, key=lambda x: x["R"] + x["G1"] + x["B"] + x["G2"])[:5]
    print(f"\n      Топ-5 по сумме (меньше = чище):")
    print(f"      {'sh':>4}  {'R':>9}  {'G1':>9}  {'B':>9}  {'G2':>9}")
    for r in top:
        print(f"      {r['sh']:>4}  {r['R']:>9.2f}  {r['G1']:>9.2f}  "
              f"{r['B']:>9.2f}  {r['G2']:>9.2f}")

    print(f"\n      ✅ Оптимум по каналам:")
    print(f"         R  → sh = {best['R']}")
    print(f"         G1 → sh = {best['G1']}")
    print(f"         B  → sh = {best['B']}")
    print(f"         G2 → sh = {best['G2']}")

    # Проверка: если R и G2 имеют одинаковый sh, а G1 и B — другой,
    # это подтверждает гипотезу о «двух потоках» с разными смещениями.
    if best["R"] == best["G2"] and best["G1"] == best["B"]:
        print(f"\n      ℹ Подтверждено: чётный поток sh={best['G1']}, "
              f"нечётный поток sh={best['R']}")
    else:
        print(f"\n      ⚠ Каналы требуют разные смещения — "
              f"возможно, у файла более сложная структура")

    return best


# ===========================================================================
# 6. ОБРАБОТКА ОДНОГО ФАЙЛА
# ===========================================================================

def process_file(filepath: Path,
                 output_dir: Path | None = None,
                 verbose: bool = False,
                 manual_w: int | None = None,
                 manual_h: int | None = None,
                 auto_sh: bool = False,
                 subheader_override: dict[str, int] | None = None) -> None:
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

    # --- Определяем subheader для каждого канала
    if subheader_override:
        subheaders = subheader_override
        print(f"   Subheader (вручную): R={subheaders['R']}, "
              f"G1={subheaders['G1']}, B={subheaders['B']}, "
              f"G2={subheaders['G2']}")
    elif auto_sh:
        subheaders = find_best_subheaders_per_channel(filepath, off, w, h)
    else:
        sh = ESSENCE_SUBHEADER
        subheaders = {"R": sh, "G1": sh, "B": sh, "G2": sh}
        print(f"   ESSENCE_SUBHEADER = {sh} (для всех каналов)")

    # --- Читаем максимально необходимое количество байт
    max_sh = max(subheaders.values())
    read_size = max_sh + expected_packed

    with open(filepath, "rb") as f:
        f.seek(off)
        raw_big = f.read(read_size)

    if len(raw_big) < expected_packed:
        raise ValueError(
            f"Недостаточно данных: {len(raw_big)} < {expected_packed}"
        )

    print(f"   Читаю с offset:   {off:,}, "
          f"{read_size:,} байт (с запасом на subheader)")

    # --- Извлекаем каждый канал с собственным subheader
    channels: dict[str, np.ndarray] = {}
    for ch in ("R", "G1", "B", "G2"):
        sh = subheaders[ch]
        chunk = raw_big[sh:sh + expected_packed]
        if len(chunk) < expected_packed:
            raise ValueError(f"Недостаточно данных для канала {ch}")
        bayer = unpack_arriraw(chunk, w, h)
        channels[ch] = extract_channel(bayer, ch)

    # --- Сохранение
    if output_dir is None:
        output_dir = filepath.parent / f"{filepath.stem}_channels"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = output_dir / filepath.stem
    tifffile.imwrite(f"{base}_R.tiff",  channels["R"])
    tifffile.imwrite(f"{base}_B.tiff",  channels["B"])
    tifffile.imwrite(f"{base}_G1.tiff", channels["G1"])
    tifffile.imwrite(f"{base}_G2.tiff", channels["G2"])

    print(f"   ✅ Сохранено: {output_dir}")
    print(f"      Размер канала: {channels['R'].shape[1]}×"
          f"{channels['R'].shape[0]}")


# ===========================================================================
# 7. ПОИСК ФАЙЛОВ И МЕНЮ
# ===========================================================================

def find_raw_files(directory: Path) -> list[Path]:
    exts = ("*.mxf", "*.MXF", "*.ari", "*.ARI")
    found: set[Path] = set()
    for ext in exts:
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
# 8. ТОЧКА ВХОДА
# ===========================================================================

def main() -> None:
    args  = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    verbose = "--verbose" in flags or "-v" in flags
    auto_sh = "--auto-sh" in flags

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
                         auto_sh=auto_sh)
        except Exception as e:
            print(f"❌ Ошибка при обработке {f.name}: {e}")


if __name__ == "__main__":
    main()