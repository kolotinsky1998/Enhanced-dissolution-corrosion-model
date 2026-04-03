"""
Построение зависимости критической толщины диффузионного слоя

    l_diff^* = 3 D_O C_O^bulk / (4 k_Fe C_Fe^s)

от концентрации кислорода в объеме жидкого свинца при T = 420 °C.

По оси абсцисс используется C_O_bulk в mass % с логарифмической шкалой.
На график также наносится вертикальная пунктирная линия для C_O^s.
График сохраняется в PNG и отображается на экране.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


T_C = 420.0
T_K = T_C + 273.15

OUTPUT_PNG = "ldiff_critical_vs_co_bulk_420C.png"
FIG_DPI = 220


R = 8.31446261815324
M_O = 15.999e-3
M_FE = 55.845e-3


def rho_pb(T: float) -> float:
    return 11441.0 - 1.2795 * T


def c_fe_s_mol_m3(T: float) -> float:
    return 10.0 ** (1.824 - 4860.0 / T) / 100.0 / M_FE * rho_pb(T)


def c_o_s_mol_m3(T: float) -> float:
    return 10.0 ** (3.21 - 5100.0 / T) / 100.0 / M_O * rho_pb(T)


def d_o_pb(T: float) -> float:
    return 1.0e-4 * 2.79e-3 * math.exp(-45587.0 / (R * T))


def k_fe(T: float) -> float:
    return 1.22e-4 * math.exp(-9418.0 / (R * T)) / c_fe_s_mol_m3(T)


def mol_to_mass_percent(concentration_mol_m3: np.ndarray | float, molar_mass: float, rho_lead: float) -> np.ndarray | float:
    return concentration_mol_m3 * molar_mass / rho_lead * 100.0


def ldiff_critical(c_o_bulk_mol_m3: np.ndarray, T: float) -> np.ndarray:
    return 3.0 * d_o_pb(T) * c_o_bulk_mol_m3 / (4.0 * k_fe(T) * c_fe_s_mol_m3(T))


def make_plot() -> Path:
    rho_lead = rho_pb(T_K)
    c_o_s_masspct = float(mol_to_mass_percent(c_o_s_mol_m3(T_K), M_O, rho_lead))

    c_o_bulk_mol = np.logspace(-6, 1.5, 600)
    c_o_bulk_masspct = mol_to_mass_percent(c_o_bulk_mol, M_O, rho_lead)
    ldiff_star = ldiff_critical(c_o_bulk_mol, T_K)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 16,
            "axes.labelsize": 19,
            "axes.titlesize": 21,
            "legend.fontsize": 14,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        c_o_bulk_masspct,
        ldiff_star,
        color="#1f4e5f",
        linewidth=3.0,
        label=r"$l_{diff}^{*}=\dfrac{3D_O C_O^{bulk}}{4K_{Fe}C_{Fe}^{s}}$",
        zorder=2,
    )

    ax.axvline(
        c_o_s_masspct,
        color="#8f2738",
        linestyle="--",
        linewidth=2.0,
        alpha=0.9,
        label=r"$C_O^{s}$",
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$C_O^{bulk}$, mass %")
    ax.set_ylabel(r"$l_{diff}^{*}$, m")
    ax.set_title(r"Критическая толщина $l_{diff}^{*}$ как функция $C_O^{bulk}$ при $420^\circ$C")

    for spine in ax.spines.values():
        spine.set_color("#262626")
        spine.set_linewidth(1.4)

    ax.tick_params(axis="both", which="major", width=1.3, length=7, colors="#1f1f1f")
    ax.tick_params(axis="both", which="minor", width=1.0, length=4, colors="#303030")

    ax.grid(which="major", color="#cfcfcf", linewidth=0.8, alpha=0.55)
    ax.grid(which="minor", color="#e6e6e6", linewidth=0.55, alpha=0.45)

    info_text = "\n".join(
        [
            rf"$T = {T_C:.0f}\,^{{\circ}}\mathrm{{C}}$",
            rf"$D_O = {d_o_pb(T_K):.3e}\,\mathrm{{m^2/s}}$",
            rf"$K_{{Fe}} = {k_fe(T_K):.3e}\,\mathrm{{m/s}}$",
            rf"$C_O^s = {c_o_s_masspct:.3e}\,\mathrm{{mass\ \%}}$",
            rf"$C_{{Fe}}^s = {mol_to_mass_percent(c_fe_s_mol_m3(T_K), M_FE, rho_lead):.3e}\,\mathrm{{mass\ \%}}$",
        ]
    )
    ax.text(
        0.03,
        0.05,
        info_text,
        transform=ax.transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#c9c9c9", alpha=0.98),
    )

    leg = ax.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.95)
    leg.get_frame().set_edgecolor("#bfbfbf")
    leg.get_frame().set_facecolor("white")

    output_path = Path(OUTPUT_PNG)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"График сохранен в {output_path.resolve()}")
    plt.show()
    return output_path


if __name__ == "__main__":
    make_plot()
