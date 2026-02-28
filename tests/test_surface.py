from __future__ import annotations

import math

from data_processing_plotting.surface import (
    AMU_TO_KG,
    ANGSTROM_TO_M,
    C_CM_PER_S,
    HARTREE_TO_J,
    MU1,
    MU2,
    analyze_surface,
)


def _gaussian_text(r: float, theta_deg: float, energy: float) -> str:
    theta = math.radians(theta_deg)
    h1 = (0.0, 0.0, r)
    h2 = (r * math.sin(theta), 0.0, r * math.cos(theta))
    return f"""
 Entering Gaussian System
 Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
 1          8           0        0.000000    0.000000    0.000000
 2          1           0        {h1[0]:.6f}    {h1[1]:.6f}    {h1[2]:.6f}
 3          1           0        {h2[0]:.6f}    {h2[1]:.6f}    {h2[2]:.6f}
 ---------------------------------------------------------------------
 SCF Done:  E(RB3LYP) =  {energy:.12f}     A.U. after   10 cycles
 Normal termination of Gaussian
 """


def _expected_wavenumbers(kr_hartree_per_a2: float, ktheta_hartree_per_rad2: float, r_eq: float) -> tuple[float, float]:
    k_r_si = kr_hartree_per_a2 * HARTREE_TO_J / (ANGSTROM_TO_M**2)
    k_theta_si = ktheta_hartree_per_rad2 * HARTREE_TO_J
    r_eq_m = r_eq * ANGSTROM_TO_M

    nu1_hz = (1.0 / (2.0 * math.pi)) * math.sqrt(k_r_si / MU1)
    nu2_hz = (1.0 / (2.0 * math.pi)) * math.sqrt(k_theta_si / (r_eq_m**2 * MU2))
    return nu1_hz / C_CM_PER_S, nu2_hz / C_CM_PER_S


def test_analyze_surface_synthetic(tmp_path):
    r0 = 1.00
    t0 = 104.5
    e0 = -76.0
    kr = 0.60
    kth = 0.04

    rs = [0.95, 1.00, 1.05]
    ts = [100.0, 104.5, 109.0]
    idx = 0
    for r in rs:
        for t in ts:
            dt = math.radians(t - t0)
            dr = r - r0
            energy = e0 + 0.5 * kr * dr * dr + 0.5 * kth * dt * dt
            (tmp_path / f"pt_{idx:03d}.out").write_text(_gaussian_text(r, t, energy), encoding="utf-8")
            idx += 1

    result = analyze_surface(str(tmp_path), str(tmp_path / "surface.png"))
    exp_nu1, exp_nu2 = _expected_wavenumbers(kr, kth, r0)

    assert abs(result["equilibrium_energy_hartree"] - e0) < 1e-8
    assert abs(result["equilibrium_r_angstrom"] - r0) < 1e-6
    assert abs(result["equilibrium_theta_deg"] - t0) < 1e-4
    assert abs(result["nu1_cm^-1"] - exp_nu1) < 0.2
    assert abs(result["nu2_cm^-1"] - exp_nu2) < 0.2
    assert (tmp_path / "surface.png").exists()

