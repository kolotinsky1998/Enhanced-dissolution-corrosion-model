"""
Поиск положений максимумов локального оттока Fe3O4 по сохраненному .npz-файлу
после расчета диффузионной задачи.

Скрипт использует файл результатов, создаваемый fipy_fe_pb_oxidation.py, и
определяет расстояния от границы стали z = 0 до локальных максимумов профиля
потока Fe3O4(z).

Примеры запуска:
    python find_fe3o4_flux_peaks.py
    python find_fe3o4_flux_peaks.py --npz data/fipy_fe_pb_oxidation_results.npz
    python find_fe3o4_flux_peaks.py --save-csv fe3o4_peak_distances.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_NPZ = "fipy_fe_pb_oxidation_results.npz"


@dataclass
class PeakInfo:
    index: int
    z_m: float
    flux_kg_m2_s: float
    is_global: bool


def find_local_maxima(
    z: np.ndarray,
    flux: np.ndarray,
    min_rel_height: float = 1.0e-6,
    include_edges: bool = True,
) -> list[PeakInfo]:
    """
    Находит локальные максимумы профиля потока.

    Параметр min_rel_height задает относительный порог по сравнению с глобальным
    максимумом профиля. Это позволяет не считать численный шум отдельными пиками.
    """
    z = np.asarray(z, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if z.ndim != 1 or flux.ndim != 1:
        raise ValueError("z и flux должны быть одномерными массивами")
    if z.size != flux.size:
        raise ValueError("z и flux должны иметь одинаковую длину")
    if z.size < 2:
        return []

    flux_pos = np.maximum(flux, 0.0)
    global_idx = int(np.argmax(flux_pos))
    global_max = float(flux_pos[global_idx])

    if global_max <= 0.0:
      return []

    threshold = min_rel_height * global_max
    peak_indices: list[int] = []

    if include_edges:
        if flux_pos[0] >= flux_pos[1] and flux_pos[0] >= threshold:
            peak_indices.append(0)

    for i in range(1, flux_pos.size - 1):
        if (
            flux_pos[i] >= flux_pos[i - 1]
            and flux_pos[i] >= flux_pos[i + 1]
            and flux_pos[i] >= threshold
            and (flux_pos[i] > flux_pos[i - 1] or flux_pos[i] > flux_pos[i + 1])
        ):
            peak_indices.append(i)

    if include_edges:
        last = flux_pos.size - 1
        if flux_pos[last] >= flux_pos[last - 1] and flux_pos[last] >= threshold:
            peak_indices.append(last)

    peak_indices = sorted(set(peak_indices))

    peaks = [
        PeakInfo(
            index=i,
            z_m=float(z[i]),
            flux_kg_m2_s=float(flux_pos[i]),
            is_global=(i == global_idx),
        )
        for i in peak_indices
    ]
    peaks.sort(key=lambda item: item.z_m)
    return peaks


def analyze_flux_profiles(
    npz_path: str | Path = DEFAULT_NPZ,
    min_rel_height: float = 1.0e-6,
) -> dict[str, object]:
    """
    Анализирует все сохраненные профили локального потока Fe3O4.

    Возвращает словарь с двумя разделами:
    - transient: список профилей в сохраненные моменты времени
    - steady: пики стационарного профиля
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    z = np.asarray(data["z"], dtype=float)
    time_profiles = np.asarray(data["time_profiles"], dtype=float)
    flux_profiles = np.asarray(data["Fe3O4_flux_profiles"], dtype=float)
    steady_flux = np.asarray(data["steady_Fe3O4_flux"], dtype=float)

    transient = []
    for time_s, flux in zip(time_profiles, flux_profiles):
        peaks = find_local_maxima(z, flux, min_rel_height=min_rel_height)
        transient.append(
            {
                "time_s": float(time_s),
                "peaks": peaks,
            }
        )

    steady = {
        "time_s": float(np.asarray(data["steady_time"], dtype=float).ravel()[0]),
        "peaks": find_local_maxima(z, steady_flux, min_rel_height=min_rel_height),
    }

    return {
        "npz_path": str(npz_path),
        "transient": transient,
        "steady": steady,
        "T_C": float(np.asarray(data["T_C"], dtype=float).ravel()[0]) if "T_C" in data else None,
        "L": float(np.asarray(data["L"], dtype=float).ravel()[0]) if "L" in data else None,
        "C_O_bulk": float(np.asarray(data["C_O_bulk"], dtype=float).ravel()[0]) if "C_O_bulk" in data else None,
    }


def print_report(report: dict[str, object]) -> None:
    print("Положения максимумов локального потока Fe3O4")
    print("=" * 72)
    print(f"Файл результатов: {report['npz_path']}")
    if report["T_C"] is not None:
        print(f"T = {report['T_C']:.2f} °C")
    if report["L"] is not None:
        print(f"l = {report['L']:.6e} m")
    if report["C_O_bulk"] is not None:
        print(f"C_O^bulk = {report['C_O_bulk']:.6e} mol/m^3")
    print("=" * 72)

    print("\nСохраненные нестационарные профили:")
    for item in report["transient"]:
        time_s = item["time_s"]
        peaks: list[PeakInfo] = item["peaks"]
        if not peaks:
            print(f"t = {time_s:.6e} s: максимумов нет")
            continue
        print(f"t = {time_s:.6e} s:")
        for peak in peaks:
            flag = " [global]" if peak.is_global else ""
            print(
                f"  z = {peak.z_m:.6e} m, "
                f"flux = {peak.flux_kg_m2_s:.6e} kg/(m^2 s){flag}"
            )

    print("\nСтационарный профиль:")
    steady = report["steady"]
    steady_peaks: list[PeakInfo] = steady["peaks"]
    if not steady_peaks:
        print(f"t = {steady['time_s']:.6e} s: максимумов нет")
    else:
        print(f"t = {steady['time_s']:.6e} s:")
        for peak in steady_peaks:
            flag = " [global]" if peak.is_global else ""
            print(
                f"  z = {peak.z_m:.6e} m, "
                f"flux = {peak.flux_kg_m2_s:.6e} kg/(m^2 s){flag}"
            )


def save_csv(report: dict[str, object], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "profile_type",
                "time_s",
                "peak_index",
                "z_m",
                "distance_from_steel_m",
                "flux_kg_m2_s",
                "is_global",
            ]
        )

        for item in report["transient"]:
            time_s = item["time_s"]
            for peak in item["peaks"]:
                writer.writerow(
                    [
                        "transient",
                        f"{time_s:.16e}",
                        peak.index,
                        f"{peak.z_m:.16e}",
                        f"{peak.z_m:.16e}",
                        f"{peak.flux_kg_m2_s:.16e}",
                        int(peak.is_global),
                    ]
                )

        steady = report["steady"]
        for peak in steady["peaks"]:
            writer.writerow(
                [
                    "steady",
                    f"{steady['time_s']:.16e}",
                    peak.index,
                    f"{peak.z_m:.16e}",
                    f"{peak.z_m:.16e}",
                    f"{peak.flux_kg_m2_s:.16e}",
                    int(peak.is_global),
                ]
            )

    print(f"\nCSV с положениями максимумов сохранен в {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Поиск расстояний от границы стали до максимумов локального потока Fe3O4."
    )
    parser.add_argument(
        "--npz",
        default=DEFAULT_NPZ,
        help=f"Путь к .npz-файлу результатов. По умолчанию: {DEFAULT_NPZ}",
    )
    parser.add_argument(
        "--min-rel-height",
        type=float,
        default=1.0e-6,
        help=(
            "Относительный порог для отсечения малых максимумов. "
            "Задается как доля от глобального максимума профиля."
        ),
    )
    parser.add_argument(
        "--save-csv",
        default="",
        help="Если задано, результаты дополнительно сохраняются в CSV-файл.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args, _unknown = parser.parse_known_args()

    report = analyze_flux_profiles(
        npz_path=args.npz,
        min_rel_height=args.min_rel_height,
    )
    print_report(report)

    if args.save_csv:
        save_csv(report, args.save_csv)


if __name__ == "__main__":
    main()
