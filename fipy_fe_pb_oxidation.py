"""
Численное решение одномерной модели начального окисления железа
в жидком свинце с использованием FiPy.

Перед запуском в Colab установите зависимости:
    !pip install fipy

Скрипт строит:
1. стационарные профили концентраций O и Fe;
2. локальный профиль потока удаляемого Fe3O4;
3. нестационарные профили концентраций и потока Fe3O4;
4. эволюцию граничных концентраций во времени;
5. интегральный отток Fe3O4 из всей системы во времени.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from fipy import CellVariable, DiffusionTerm, Grid1D, TransientTerm

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# ============================================================
# Параметры расчета
# ============================================================

T_C = 620.0
T_K = T_C + 273.15

L = 1e-4  # толщина диффузионного слоя, м
NX = 400  # число ячеек по координате z

DT = 0.1  # шаг по времени, с
MAX_STEPS = 500
STEADY_TOL = 1.0e-14
MIN_STEPS_BEFORE_STEADY = 200

SAVE_PROFILE_TIMES = np.array([1.0, 10.0, 100.0, 1.0e3, 1.0e4, 5.0e4])

FIG_DPI = 180
PLOT_DIR = "."
C_O_BULK_FACTOR = 1e-7  # C_O^bulk = C_O_BULK_FACTOR * C_O^s(T)
INITIAL_C_O_FACTOR = 0.0  # C_O(z,0) = INITIAL_C_O_FACTOR * C_O^s(T)
INITIAL_C_FE_FACTOR = 1.0  # C_Fe(z,0) = INITIAL_C_FE_FACTOR * C_Fe^bulk
RESULTS_FILE = "fipy_fe_pb_oxidation_results.npz"


# ============================================================
# Физические константы и термодинамика
# ============================================================

R = 8.31446261815324
m_O = 15.999e-3
m_Fe = 55.845e-3
M_Fe3O4 = 3.0 * m_Fe + 4.0 * m_O
MACHINE_EPS = np.finfo(float).eps


def rho_pb(T: float) -> float:
    return 11441.0 - 1.2795 * T


def C_O_s(T: float) -> float:
    return 10.0 ** (3.21 - 5100.0 / T) / 100.0 / m_O * rho_pb(T)


def C_Fe_s(T: float) -> float:
    return 10.0 ** (1.824 - 4860.0 / T) / 100.0 / m_Fe * rho_pb(T)


def D_O_Pb(T: float) -> float:
    return 1.0e-4 * 2.79e-3 * math.exp(-45587.0 / (R * T))


def D_Fe_Pb(T: float) -> float:
    return 4.898e-7 * math.exp(-43934.77 / 8.31 / T)


class Fe3O4Solid:
    def __init__(self) -> None:
        self.A = 104.2096
        self.B = 178.5108
        self.C = 10.61510
        self.D = 1.132534
        self.E = -0.994202
        self.F = -1163.336
        self.G = 212.0585
        self.H = -1120.894

    def H_f(self) -> float:
        return -1120.89 * 1000.0

    def H_m(self, T: float) -> float:
        t = T / 1000.0
        return (
            1000.0
            * (
                self.A * t
                + self.B * t**2 / 2.0
                + self.C * t**3 / 3.0
                + self.D * t**4 / 4.0
                - self.E / t
                + self.F
                - self.H
            )
            + self.H_f()
        )

    def S_m(self, T: float) -> float:
        t = T / 1000.0
        return (
            self.A * math.log(t)
            + self.B * t
            + self.C * t**2 / 2.0
            + self.D * t**3 / 3.0
            - self.E / (2.0 * t**2)
            + self.G
        )

    def G_m(self, T: float) -> float:
        return self.H_m(T) - self.S_m(T) * T


class PbLiquid:
    def __init__(self) -> None:
        self.A = 38.00449
        self.B = -14.62249
        self.C = 7.255475
        self.D = -1.033370
        self.E = -0.330775
        self.F = -7.944328
        self.G = 118.7992
        self.H = 4.282993

    def H_f(self) -> float:
        return 4.28 * 1000.0

    def H_m(self, T: float) -> float:
        t = T / 1000.0
        return (
            1000.0
            * (
                self.A * t
                + self.B * t**2 / 2.0
                + self.C * t**3 / 3.0
                + self.D * t**4 / 4.0
                - self.E / t
                + self.F
                - self.H
            )
            + self.H_f()
        )

    def S_m(self, T: float) -> float:
        t = T / 1000.0
        return (
            self.A * math.log(t)
            + self.B * t
            + self.C * t**2 / 2.0
            + self.D * t**3 / 3.0
            - self.E / (2.0 * t**2)
            + self.G
        )

    def G_m(self, T: float) -> float:
        return self.H_m(T) - self.S_m(T) * T


class FeSolid:
    def __init__(self) -> None:
        self.T_I_II = 700.0
        self.A_I = 18.42868
        self.B_I = 24.64301
        self.C_I = -8.913720
        self.D_I = 9.664706
        self.E_I = -0.012643
        self.F_I = -6.573022
        self.G_I = 42.51488
        self.H_I = 0.0
        self.A_II = -57767.65
        self.B_II = 137919.7
        self.C_II = -122773.2
        self.D_II = 38682.42
        self.E_II = 3993.080
        self.F_II = 24078.67
        self.G_II = -87364.01
        self.H_II = 0.0

    def H_m(self, T: float) -> float:
        t = T / 1000.0
        if T < self.T_I_II:
            return 1000.0 * (
                self.A_I * t
                + self.B_I * t**2 / 2.0
                + self.C_I * t**3 / 3.0
                + self.D_I * t**4 / 4.0
                - self.E_I / t
                + self.F_I
                - self.H_I
            )
        return 1000.0 * (
            self.A_II * t
            + self.B_II * t**2 / 2.0
            + self.C_II * t**3 / 3.0
            + self.D_II * t**4 / 4.0
            - self.E_II / t
            + self.F_II
            - self.H_II
        )

    def S_m(self, T: float) -> float:
        t = T / 1000.0
        if T < self.T_I_II:
            return (
                self.A_I * math.log(t)
                + self.B_I * t
                + self.C_I * t**2 / 2.0
                + self.D_I * t**3 / 3.0
                - self.E_I / (2.0 * t**2)
                + self.G_I
            )
        return (
            self.A_II * math.log(t)
            + self.B_II * t
            + self.C_II * t**2 / 2.0
            + self.D_II * t**3 / 3.0
            - self.E_II / (2.0 * t**2)
            + self.G_II
        )

    def G_m(self, T: float) -> float:
        return self.H_m(T) - self.S_m(T) * T


class PbOYellowSolid:
    def __init__(self) -> None:
        self.T_I_II = 762.0
        self.A_I = 7.465570
        self.B_I = 179.5860
        self.C_I = -233.5490
        self.D_I = 109.2070
        self.E_I = 0.233832
        self.F_I = -226.9830
        self.G_I = 32.54460
        self.H_I = -219.4090
        self.A_II = 47.86340
        self.B_II = 12.55480
        self.C_II = -0.001810
        self.D_II = 0.000416
        self.E_II = 0.000200
        self.F_II = -234.8160
        self.G_II = 118.9100
        self.H_II = -219.4090

    def H_f(self) -> float:
        return -219.41 * 1000.0

    def H_m(self, T: float) -> float:
        t = T / 1000.0
        if T < self.T_I_II:
            return (
                1000.0
                * (
                    self.A_I * t
                    + self.B_I * t**2 / 2.0
                    + self.C_I * t**3 / 3.0
                    + self.D_I * t**4 / 4.0
                    - self.E_I / t
                    + self.F_I
                    - self.H_I
                )
                + self.H_f()
            )
        return (
            1000.0
            * (
                self.A_II * t
                + self.B_II * t**2 / 2.0
                + self.C_II * t**3 / 3.0
                + self.D_II * t**4 / 4.0
                - self.E_II / t
                + self.F_II
                - self.H_II
            )
            + self.H_f()
        )

    def S_m(self, T: float) -> float:
        t = T / 1000.0
        if T < self.T_I_II:
            return (
                self.A_I * math.log(t)
                + self.B_I * t
                + self.C_I * t**2 / 2.0
                + self.D_I * t**3 / 3.0
                - self.E_I / (2.0 * t**2)
                + self.G_I
            )
        return (
            self.A_II * math.log(t)
            + self.B_II * t
            + self.C_II * t**2 / 2.0
            + self.D_II * t**3 / 3.0
            - self.E_II / (2.0 * t**2)
            + self.G_II
        )

    def G_m(self, T: float) -> float:
        return self.H_m(T) - self.S_m(T) * T


def K_Fe3O4(T: float) -> float:
    fe3o4 = Fe3O4Solid()
    pb_l = PbLiquid()
    pbo = PbOYellowSolid()
    fe_solid = FeSolid()
    exponent = (
        fe3o4.G_m(T)
        - 4.0 * (pbo.G_m(T) - pb_l.G_m(T))
        - 3.0 * fe_solid.G_m(T)
    ) / (R * T)
    return math.exp(exponent) * C_O_s(T) ** 4 * C_Fe_s(T) ** 3


def K_Fe(T: float) -> float:
    return 1.22e-4 * math.exp(-9418.0 / (R * T)) / C_Fe_s(T)


def compute_c_fe_bulk(T: float, c_o_bulk: float) -> float:
    c_fe_sat = C_Fe_s(T)
    if c_o_bulk <= 0.0:
        return c_fe_sat
    c_fe_bulk = (K_Fe3O4(T) / c_o_bulk**4) ** (1.0 / 3.0)
    return min(c_fe_bulk, c_fe_sat)


def mol_to_mass_percent(concentration_mol_m3: np.ndarray | float, molar_mass: float, rho_lead: float) -> np.ndarray | float:
    return concentration_mol_m3 * molar_mass / rho_lead * 100.0


@dataclass
class SimulationResults:
    z: np.ndarray
    time: np.ndarray
    time_profiles: np.ndarray
    C_O_profiles_masspct: np.ndarray
    C_Fe_profiles_masspct: np.ndarray
    Fe3O4_flux_profiles: np.ndarray
    left_boundary_O_masspct: np.ndarray
    left_boundary_Fe_masspct: np.ndarray
    integral_Fe3O4_flux: np.ndarray
    steady_C_O_masspct: np.ndarray
    steady_C_Fe_masspct: np.ndarray
    steady_Fe3O4_flux: np.ndarray
    steady_time: float


def project_to_magnetite_stability(
    C_O_old: np.ndarray,
    C_Fe_old: np.ndarray,
    K_eq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C_O_new = C_O_old.copy()
    C_Fe_new = C_Fe_old.copy()
    Fe3O4_removed = np.zeros_like(C_O_old)

    oversat = (C_O_old**4) * (C_Fe_old**3) > K_eq
    oversat &= (C_O_old > 0.0) & (C_Fe_old > 0.0)

    for idx in np.where(oversat)[0]:
        co = float(C_O_old[idx])
        cfe = float(C_Fe_old[idx])
        alpha_max = min(co / 4.0, cfe / 3.0)
        if alpha_max <= 0.0:
            continue

        def f(alpha: float) -> float:
            return (co - 4.0 * alpha) ** 4 * (cfe - 3.0 * alpha) ** 3 - K_eq

        left = 0.0
        right = alpha_max * (1.0 - 1.0e-14)
        if f(left) <= 0.0:
            continue
        if f(right) > 0.0:
            alpha = alpha_max
        else:
            for _ in range(80):
                mid = 0.5 * (left + right)
                if f(mid) > 0.0:
                    left = mid
                else:
                    right = mid
            alpha = 0.5 * (left + right)

        C_O_new[idx] = max(co - 4.0 * alpha, 0.0)
        C_Fe_new[idx] = max(cfe - 3.0 * alpha, 0.0)
        Fe3O4_removed[idx] = alpha

    return C_O_new, C_Fe_new, Fe3O4_removed


def run_simulation() -> SimulationResults:
    rho_lead = rho_pb(T_K)
    D_O = D_O_Pb(T_K)
    D_Fe = D_Fe_Pb(T_K)
    c_o_sat = C_O_s(T_K)
    c_o_bulk = C_O_BULK_FACTOR * c_o_sat
    c_o_init = INITIAL_C_O_FACTOR * c_o_sat
    c_fe_bulk = compute_c_fe_bulk(T_K, c_o_bulk)
    c_fe_init = INITIAL_C_FE_FACTOR * c_fe_bulk
    c_fe_sat = C_Fe_s(T_K)
    k_fe = K_Fe(T_K)
    k_eq = K_Fe3O4(T_K)

    dx = L / NX
    mesh = Grid1D(nx=NX, dx=dx)
    z = np.array(mesh.cellCenters.value[0])

    C_O = CellVariable(name="C_O", mesh=mesh, value=c_o_init, hasOld=True)
    C_Fe = CellVariable(name="C_Fe", mesh=mesh, value=c_fe_init, hasOld=True)

    C_O.constrain(c_o_bulk, where=mesh.facesRight)
    C_Fe.constrain(c_fe_bulk, where=mesh.facesRight)
    C_O.faceGrad.constrain([0.0], where=mesh.facesLeft)

    eq_O = TransientTerm(var=C_O) == DiffusionTerm(coeff=D_O, var=C_O)
    eq_Fe = TransientTerm(var=C_Fe) == DiffusionTerm(coeff=D_Fe, var=C_Fe)

    history_time = []
    history_left_O = []
    history_left_Fe = []
    history_integral_flux = []

    saved_times = []
    saved_O_profiles = []
    saved_Fe_profiles = []
    saved_flux_profiles = []

    save_index = 0
    current_time = 0.0
    last_fe3o4_flux_local_kg_m2_s = np.zeros(NX)

    progress = tqdm(total=MAX_STEPS, desc="FiPy simulation", unit="step") if tqdm is not None else None

    for step in range(1, MAX_STEPS + 1):
        c_o_prev = np.array(C_O.value, dtype=float).copy()
        c_fe_prev = np.array(C_Fe.value, dtype=float).copy()

        C_O.updateOld()
        C_Fe.updateOld()

        left_cfe = float(c_fe_prev[0])
        j_fe_left = k_fe * (c_fe_sat - left_cfe)
        grad_left_fe = -j_fe_left / D_Fe
        C_Fe.faceGrad.constrain([grad_left_fe], where=mesh.facesLeft)

        eq_O.solve(var=C_O, dt=DT)
        eq_Fe.solve(var=C_Fe, dt=DT)

        c_o_arr = np.maximum(np.array(C_O.value, dtype=float), 0.0)
        c_fe_arr = np.maximum(np.array(C_Fe.value, dtype=float), 0.0)

        c_o_arr, c_fe_arr, fe3o4_removed_mol_m3 = project_to_magnetite_stability(
            c_o_arr, c_fe_arr, k_eq
        )

        C_O.setValue(c_o_arr)
        C_Fe.setValue(c_fe_arr)

        current_time += DT

        fe3o4_source_kg_m3_s = fe3o4_removed_mol_m3 * M_Fe3O4 / DT
        fe3o4_flux_local_kg_m2_s = fe3o4_source_kg_m3_s * dx
        fe3o4_integral_flux_kg_m2_s = float(np.sum(fe3o4_source_kg_m3_s) * dx)
        last_fe3o4_flux_local_kg_m2_s = fe3o4_flux_local_kg_m2_s.copy()

        history_time.append(current_time)
        history_left_O.append(mol_to_mass_percent(c_o_arr[0], m_O, rho_lead))
        history_left_Fe.append(mol_to_mass_percent(c_fe_arr[0], m_Fe, rho_lead))
        history_integral_flux.append(fe3o4_integral_flux_kg_m2_s)

        while save_index < len(SAVE_PROFILE_TIMES) and current_time >= SAVE_PROFILE_TIMES[save_index]:
            saved_times.append(current_time)
            saved_O_profiles.append(mol_to_mass_percent(c_o_arr.copy(), m_O, rho_lead))
            saved_Fe_profiles.append(mol_to_mass_percent(c_fe_arr.copy(), m_Fe, rho_lead))
            saved_flux_profiles.append(fe3o4_flux_local_kg_m2_s.copy())
            save_index += 1

        delta_o = np.max(np.abs(c_o_arr - c_o_prev))
        delta_fe = np.max(np.abs(c_fe_arr - c_fe_prev))
        scale_o = max(float(np.max(np.abs(c_o_arr))), c_o_bulk, 1.0e-12 * c_o_sat, MACHINE_EPS)
        scale_fe = max(float(np.max(np.abs(c_fe_arr))), c_fe_bulk, 1.0e-12 * c_fe_sat, MACHINE_EPS)
        steady_metric_raw = max(delta_o / scale_o, delta_fe / scale_fe)
        steady_metric = max(steady_metric_raw, MACHINE_EPS)

        if step % 2000 == 0:
            print(
                f"step={step:7d}, t={current_time:12.3f} s, "
                f"steady_metric={steady_metric:.3e}, dt={DT:.3e} s, "
                f"J_Fe(left)={j_fe_left * m_Fe:.3e} kg/(m^2 s), "
                f"Fe3O4 integral={fe3o4_integral_flux_kg_m2_s:.3e} kg/(m^2 s)"
            )

        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                sim_time_s=f"{current_time:.2e}",
                steady=f"{steady_metric:.2e}",
                left_Fe=f"{history_left_Fe[-1]:.2e}",
                left_O=f"{history_left_O[-1]:.2e}",
                Fe3O4_int=f"{fe3o4_integral_flux_kg_m2_s:.2e}",
            )
        elif step % 500 == 0:
            print(
                f"[progress] step={step}/{MAX_STEPS}, "
                f"t={current_time:.3e} s, dt={DT:.3e} s, "
                f"steady_metric={steady_metric:.3e}"
            )

        if step >= MIN_STEPS_BEFORE_STEADY and steady_metric < STEADY_TOL:
            print(f"Стационарное состояние достигнуто при t = {current_time:.3f} s")
            break
    else:
        print("Предупреждение: достигнут MAX_STEPS до выполнения критерия стационарности.")

    if progress is not None:
        progress.close()

    c_o_steady = np.array(C_O.value, dtype=float)
    c_fe_steady = np.array(C_Fe.value, dtype=float)
    steady_flux = last_fe3o4_flux_local_kg_m2_s.copy()

    if not saved_times or saved_times[-1] < current_time:
        saved_times.append(current_time)
        saved_O_profiles.append(mol_to_mass_percent(c_o_steady.copy(), m_O, rho_lead))
        saved_Fe_profiles.append(mol_to_mass_percent(c_fe_steady.copy(), m_Fe, rho_lead))
        saved_flux_profiles.append(steady_flux.copy())

    return SimulationResults(
        z=z,
        time=np.array(history_time),
        time_profiles=np.array(saved_times),
        C_O_profiles_masspct=np.array(saved_O_profiles),
        C_Fe_profiles_masspct=np.array(saved_Fe_profiles),
        Fe3O4_flux_profiles=np.array(saved_flux_profiles),
        left_boundary_O_masspct=np.array(history_left_O),
        left_boundary_Fe_masspct=np.array(history_left_Fe),
        integral_Fe3O4_flux=np.array(history_integral_flux),
        steady_C_O_masspct=mol_to_mass_percent(c_o_steady, m_O, rho_lead),
        steady_C_Fe_masspct=mol_to_mass_percent(c_fe_steady, m_Fe, rho_lead),
        steady_Fe3O4_flux=steady_flux,
        steady_time=current_time,
    )


def save_results(results: SimulationResults) -> None:
    c_o_s_val = C_O_s(T_K)
    c_o_bulk_val = C_O_BULK_FACTOR * c_o_s_val
    c_fe_bulk_val = compute_c_fe_bulk(T_K, c_o_bulk_val)
    c_o_init_val = INITIAL_C_O_FACTOR * c_o_s_val
    c_fe_init_val = INITIAL_C_FE_FACTOR * c_fe_bulk_val

    np.savez(
        RESULTS_FILE,
        z=results.z,
        time=results.time,
        time_profiles=results.time_profiles,
        C_O_profiles_masspct=results.C_O_profiles_masspct,
        C_Fe_profiles_masspct=results.C_Fe_profiles_masspct,
        Fe3O4_flux_profiles=results.Fe3O4_flux_profiles,
        left_boundary_O_masspct=results.left_boundary_O_masspct,
        left_boundary_Fe_masspct=results.left_boundary_Fe_masspct,
        integral_Fe3O4_flux=results.integral_Fe3O4_flux,
        steady_C_O_masspct=results.steady_C_O_masspct,
        steady_C_Fe_masspct=results.steady_C_Fe_masspct,
        steady_Fe3O4_flux=results.steady_Fe3O4_flux,
        steady_time=np.array([results.steady_time]),
        T_C=np.array([T_C]),
        T_K=np.array([T_K]),
        L=np.array([L]),
        NX=np.array([NX]),
        DT=np.array([DT]),
        C_O_BULK_FACTOR=np.array([C_O_BULK_FACTOR]),
        INITIAL_C_O_FACTOR=np.array([INITIAL_C_O_FACTOR]),
        INITIAL_C_FE_FACTOR=np.array([INITIAL_C_FE_FACTOR]),
        C_O_s=np.array([c_o_s_val]),
        C_O_bulk=np.array([c_o_bulk_val]),
        C_O_init=np.array([c_o_init_val]),
        C_Fe_s=np.array([C_Fe_s(T_K)]),
        C_Fe_bulk=np.array([c_fe_bulk_val]),
        C_Fe_init=np.array([c_fe_init_val]),
        D_O=np.array([D_O_Pb(T_K)]),
        D_Fe=np.array([D_Fe_Pb(T_K)]),
        K_Fe=np.array([K_Fe(T_K)]),
        K_Fe3O4=np.array([K_Fe3O4(T_K)]),
    )
    print(f"Результаты сохранены в {RESULTS_FILE}")


def print_summary(results: SimulationResults) -> None:
    print("\nИтог расчета")
    print("=" * 60)
    print(f"T = {T_K:.2f} K")
    print(f"L = {L:.4f} m")
    print(f"NX = {NX}")
    print(f"Стационарность достигнута при t = {results.steady_time:.3f} s")
    print(f"C_O^s = {C_O_s(T_K):.6e} mol/m^3")
    print(f"C_O^bulk = {C_O_BULK_FACTOR * C_O_s(T_K):.6e} mol/m^3")
    print(f"C_O^init = {INITIAL_C_O_FACTOR * C_O_s(T_K):.6e} mol/m^3")
    print(f"C_Fe^bulk = {compute_c_fe_bulk(T_K, C_O_BULK_FACTOR * C_O_s(T_K)):.6e} mol/m^3")
    print(f"C_Fe^init = {INITIAL_C_FE_FACTOR * compute_c_fe_bulk(T_K, C_O_BULK_FACTOR * C_O_s(T_K)):.6e} mol/m^3")
    print(f"D_O = {D_O_Pb(T_K):.6e} m^2/s")
    print(f"D_Fe = {D_Fe_Pb(T_K):.6e} m^2/s")
    print(f"K_Fe = {K_Fe(T_K):.6e} m/s")
    print("=" * 60)


def main() -> None:
    results = run_simulation()
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
