"""
Построение зависимости положения главного пика локального потока Fe3O4
от концентрации кислорода в объеме жидкого свинца для случая l = 1e-6 м.

На график наносятся:
1. численные точки по результатам расчетов;
2. аналитическая кривая

    z_max = l / (1 + (4/3) * (C_O_bulk / C_Fe_s) * (D_O / D_Fe)).

По оси абсцисс используется C_O_bulk в mass % и логарифмическая шкала.
График сохраняется в PNG и отображается на экране.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


T_C = 620.0
T_K = T_C + 273.15
L = 1.0e-6

OUTPUT_PNG = "main_peak_vs_co_bulk_l1e6.png"
FIG_DPI = 220


# Численные точки из логов расчета при l = 1e-6 м
C_O_BULK_MOL_M3 = np.array(
    [
        1.017448e-02,
        2.034895e-02,
        4.069790e-02,
        1.017448e-01,
        2.034895e-01,
    ],
    dtype=float,
)

MAIN_PEAK_Z_M = np.array(
    [
        8.362500e-07,
        8.037500e-07,
        5.512500e-07,
        1.250000e-09,
        1.250000e-09,
    ],
    dtype=float,
)


R = 8.31446261815324
M_O = 15.999e-3
M_FE = 55.845e-3


def rho_pb(T: float) -> float:
    return 11441.0 - 1.2795 * T


def c_fe_s_mol_m3(T: float) -> float:
    return 10.0 ** (1.824 - 4860.0 / T) / 100.0 / M_FE * rho_pb(T)


def d_o_pb(T: float) -> float:
    return 1.0e-4 * 2.79e-3 * math.exp(-45587.0 / (R * T))


def d_fe_pb(T: float) -> float:
    return 4.898e-7 * math.exp(-43934.77 / 8.31 / T)


def k_fe(T: float) -> float:
    return 1.22e-4 * math.exp(-9418.0 / (R * T)) / c_fe_s_mol_m3(T)


def mol_to_mass_percent(concentration_mol_m3: np.ndarray | float, molar_mass: float, rho_lead: float) -> np.ndarray | float:
    return concentration_mol_m3 * molar_mass / rho_lead * 100.0


def analytical_peak_position(c_o_bulk_mol_m3: np.ndarray, T: float, l: float) -> np.ndarray:
    c_fe_sat = c_fe_s_mol_m3(T)
    d_o = d_o_pb(T)
    d_fe = d_fe_pb(T)
    k = k_fe(T)

    numerator = 4.0 * d_fe * k * c_fe_sat * l - 3.0 * d_o * c_o_bulk_mol_m3 * d_fe
    denominator = 4.0 * d_fe * k * c_fe_sat + 3.0 * d_o * c_o_bulk_mol_m3 * k
    z_peak = numerator / denominator
    return np.clip(z_peak, 0.0, l)


def make_plot() -> Path:
    rho_lead = rho_pb(T_K)
    c_o_bulk_masspct = mol_to_mass_percent(C_O_BULK_MOL_M3, M_O, rho_lead)

    x_curve_mol = np.logspace(
        np.log10(C_O_BULK_MOL_M3.min() * 0.7),
        np.log10(C_O_BULK_MOL_M3.max() * 1.4),
        500,
    )
    x_curve_masspct = mol_to_mass_percent(x_curve_mol, M_O, rho_lead)
    y_curve = analytical_peak_position(x_curve_mol, T_K, L)

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
        x_curve_masspct,
        y_curve,
        color="#1f4e5f",
        linewidth=3.0,
        label=r"Аналитическая оценка с учетом конечного $k_{Fe}$",
        zorder=2,
    )

    ax.scatter(
        c_o_bulk_masspct,
        MAIN_PEAK_Z_M,
        s=90,
        color="#b23a48",
        edgecolor="white",
        linewidth=1.5,
        label="Численные результаты",
        zorder=3,
    )

    ax.plot(
        c_o_bulk_masspct,
        MAIN_PEAK_Z_M,
        color="#8f2738",
        linewidth=1.8,
        alpha=0.65,
        zorder=2,
    )

    for x_val, y_val in zip(c_o_bulk_masspct, MAIN_PEAK_Z_M):
        ax.annotate(
            f"{y_val:.2e}",
            xy=(x_val, y_val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color="#5a2430",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$C_O^{bulk}$, mass %")
    ax.set_ylabel(r"$z_{\max}$, m")
    ax.set_title("Положение главного пика потока магнетита при $l=10^{-6}$ м")

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
            rf"$l = {L:.1e}\,\mathrm{{m}}$",
            rf"$D_O = {d_o_pb(T_K):.3e}\,\mathrm{{m^2/s}}$",
            rf"$D_{{Fe}} = {d_fe_pb(T_K):.3e}\,\mathrm{{m^2/s}}$",
            rf"$K_{{Fe}} = {k_fe(T_K):.3e}\,\mathrm{{m/s}}$",
            rf"$C_{{Fe}}^s = {mol_to_mass_percent(c_fe_s_mol_m3(T_K), M_FE, rho_lead):.3e}\,\mathrm{{mass\ \%}}$",
        ]
    )
    ax.text(
        0.03,
        0.04,
        info_text,
        transform=ax.transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#c9c9c9", alpha=0.98),
    )

    leg = ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.95)
    leg.get_frame().set_edgecolor("#bfbfbf")
    leg.get_frame().set_facecolor("white")

    output_path = Path(OUTPUT_PNG)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"График сохранен в {output_path.resolve()}")
    plt.show()
    return output_path


if __name__ == "__main__":
    make_plot()
