"""
Постобработка результатов расчета из файла fipy_fe_pb_oxidation_results.npz.

Запускать в отдельной ячейке Colab после выполнения расчетного скрипта.
Сохраняет красивые графики в PNG.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


RESULTS_FILE = "fipy_fe_pb_oxidation_results.npz"
FIG_DPI = 220
SAVE_PREFIX = ""


def setup_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8.6, 5.4),
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "lines.linewidth": 2.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 10,
        }
    )


def load_results(filename: str = RESULTS_FILE) -> dict[str, np.ndarray]:
    raw = np.load(filename)
    return {key: raw[key] for key in raw.files}


def savefig(fig: plt.Figure, name: str) -> None:
    filename = f"{SAVE_PREFIX}{name}.png"
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", facecolor="white")
    print(f"saved: {filename}")


def add_metadata_box(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    rho_pb = 11441.0 - 1.2795 * float(data["T_K"][0])
    c_o_bulk_masspct = float(data["C_O_bulk"][0] * 15.999e-3 / rho_pb * 100.0)
    text = (
        f"T = {data['T_C'][0]:.0f} °C\n"
        f"L = {data['L'][0] * 100:.2f} cm\n"
        f"Nx = {int(data['NX'][0])}\n"
        f"dt = {data['DT'][0]:.3g} s\n"
        f"C_O^bulk = {c_o_bulk_masspct:.3e} mass %"
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="0.75"),
    )


def plot_stationary_concentrations(data: dict[str, np.ndarray]) -> None:
    z_cm = data["z"] * 100.0
    fig, ax = plt.subplots()
    ax.plot(z_cm, data["steady_C_Fe_masspct"], color="#b33c2e", label="Fe")
    ax.plot(z_cm, data["steady_C_O_masspct"], color="#1f77b4", label="O")
    ax.set_xlabel("Координата z, см")
    ax.set_ylabel("Концентрация, масс. %")
    ax.set_title("Стационарные профили концентраций")
    ax.legend(frameon=True)
    add_metadata_box(ax, data)
    savefig(fig, "stationary_concentrations")


def plot_stationary_fe3o4_flux(data: dict[str, np.ndarray]) -> None:
    z_cm = data["z"] * 100.0
    fig, ax = plt.subplots()
    ax.plot(z_cm, data["steady_Fe3O4_flux"], color="black")
    ax.fill_between(z_cm, data["steady_Fe3O4_flux"], color="#a6a6a6", alpha=0.25)
    ax.set_xlabel("Координата z, см")
    ax.set_ylabel(r"Поток $\mathrm{Fe_3O_4}$, кг/(м$^2$ c)")
    ax.set_title("Стационарный локальный отток Fe$_3$O$_4$")
    add_metadata_box(ax, data)
    savefig(fig, "stationary_fe3o4_flux")


def plot_unsteady_profiles(data: dict[str, np.ndarray]) -> None:
    z_cm = data["z"] * 100.0
    times = data["time_profiles"]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.95, len(times)))

    fig, ax = plt.subplots()
    for color, t, profile in zip(colors, times, data["C_Fe_profiles_masspct"]):
        ax.plot(z_cm, profile, color=color, label=f"{t:.2e} s")
    ax.set_xlabel("Координата z, см")
    ax.set_ylabel("Концентрация Fe, масс. %")
    ax.set_title("Эволюция профиля железа")
    ax.legend(ncol=2, frameon=True)
    savefig(fig, "unsteady_fe_profiles")

    fig, ax = plt.subplots()
    for color, t, profile in zip(colors, times, data["C_O_profiles_masspct"]):
        ax.plot(z_cm, profile, color=color, label=f"{t:.2e} s")
    ax.set_xlabel("Координата z, см")
    ax.set_ylabel("Концентрация O, масс. %")
    ax.set_title("Эволюция профиля кислорода")
    ax.legend(ncol=2, frameon=True)
    savefig(fig, "unsteady_o_profiles")

    fig, ax = plt.subplots()
    for color, t, profile in zip(colors, times, data["Fe3O4_flux_profiles"]):
        ax.plot(z_cm, profile, color=color, label=f"{t:.2e} s")
    ax.set_xlabel("Координата z, см")
    ax.set_ylabel(r"Поток $\mathrm{Fe_3O_4}$, кг/(м$^2$ c)")
    ax.set_title("Эволюция локального оттока Fe$_3$O$_4$")
    ax.legend(ncol=2, frameon=True)
    savefig(fig, "unsteady_fe3o4_flux_profiles")


def plot_boundary_histories(data: dict[str, np.ndarray]) -> None:
    time = data["time"]
    c_fe_wall = data["left_boundary_Fe_masspct"]
    c_o_wall = data["left_boundary_O_masspct"]
    c_fe_sat = float(data["C_Fe_s"][0] * 55.845e-3 / (11441.0 - 1.2795 * data["T_K"][0]) * 100.0)
    c_o_sat = float(data["C_O_s"][0] * 15.999e-3 / (11441.0 - 1.2795 * data["T_K"][0]) * 100.0)

    # Линия равновесного по магнетиту кислорода при текущей концентрации Fe на стенке:
    # C_O,eq = (K_Fe3O4 / C_Fe^3)^(1/4)
    c_fe_wall_mol = np.maximum(c_fe_wall / 100.0 * (11441.0 - 1.2795 * data["T_K"][0]) / 55.845e-3, 1.0e-30)
    c_o_eq_mol = (float(data["K_Fe3O4"][0]) / c_fe_wall_mol**3) ** 0.25
    c_o_eq_masspct = c_o_eq_mol * 15.999e-3 / (11441.0 - 1.2795 * data["T_K"][0]) * 100.0

    fig, ax = plt.subplots()
    ax.plot(time, c_fe_wall, color="#b33c2e", label=r"$C_{Fe}(0,t)$")
    ax.plot(time, c_o_wall, color="#1f77b4", label=r"$C_{O}(0,t)$")
    ax.axhline(c_fe_sat, color="#b33c2e", linestyle="--", alpha=0.85, label=r"$C_{Fe}^{s}$")
    ax.axhline(c_o_sat, color="#1f77b4", linestyle="--", alpha=0.85, label=r"$C_{O}^{s}$")
    ax.plot(time, c_o_eq_masspct, color="#2f6b2f", linestyle="-.", label=r"$C_{O,\mathrm{eq}}^{Fe_3O_4}(t)$")
    ax.set_xlabel("Время, c")
    ax.set_ylabel("Концентрация, масс. %")
    ax.set_title("Граничные концентрации на поверхности стали")
    ax.set_yscale("log")
    ax.legend(frameon=True, ncol=2)
    savefig(fig, "boundary_concentrations_vs_time")

    fig, ax = plt.subplots()
    ax.plot(time, data["integral_Fe3O4_flux"], color="black")
    ax.fill_between(time, data["integral_Fe3O4_flux"], color="#8c8c8c", alpha=0.22)
    ax.set_xlabel("Время, c")
    ax.set_ylabel(r"Интегральный отток $\mathrm{Fe_3O_4}$, кг/(м$^2$ c)")
    ax.set_title("Интегральный отток Fe$_3$O$_4$ из системы")
    savefig(fig, "integral_fe3o4_flux_vs_time")


def main() -> None:
    setup_matplotlib()
    data = load_results()
    plot_stationary_concentrations(data)
    plot_stationary_fe3o4_flux(data)
    plot_unsteady_profiles(data)
    plot_boundary_histories(data)
    plt.show()


if __name__ == "__main__":
    main()
