from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARTREE_TO_J = 4.3597447222071e-18
ANGSTROM_TO_M = 1e-10
AMU_TO_KG = 1.66053906660e-27
C_CM_PER_S = 2.99792458e10

MU1 = 2.0 * AMU_TO_KG
MU2 = 0.5 * AMU_TO_KG


@dataclass(frozen=True)
class SurfacePoint:
    file_name: str
    r_angstrom: float
    theta_deg: float
    energy_hartree: float


class OutLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and ".out" in href:
            self.links.append(href)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1n = np.linalg.norm(v1)
    v2n = np.linalg.norm(v2)
    if v1n == 0.0 or v2n == 0.0:
        raise ValueError("Zero-length vector encountered while computing angle.")
    cosang = float(np.dot(v1, v2) / (v1n * v2n))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def parse_gaussian_output(text: str, file_name: str = "<memory>") -> SurfacePoint:
    energy_match = re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+\.\d+)", text)
    if not energy_match:
        raise ValueError(f"Could not find SCF energy in {file_name}.")
    energy_hartree = float(energy_match[-1])

    lines = text.splitlines()
    atoms: list[tuple[int, np.ndarray]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "Standard orientation:" not in line and "Input orientation:" not in line:
            i += 1
            continue

        j = i + 5
        candidate: list[tuple[int, np.ndarray]] = []
        while j < len(lines):
            row = lines[j].strip()
            if not row:
                j += 1
                continue
            if row.startswith("----"):
                break
            parts = row.split()
            if len(parts) < 6:
                break
            atomic_number = int(parts[1])
            xyz = np.array([float(parts[3]), float(parts[4]), float(parts[5])], dtype=float)
            candidate.append((atomic_number, xyz))
            j += 1
        if candidate:
            atoms = candidate
        i += 1

    if len(atoms) != 3:
        raise ValueError(f"Expected triatomic geometry in {file_name}, got {len(atoms)} atoms.")

    h_indices = [idx for idx, (z, _) in enumerate(atoms) if z == 1]
    if len(h_indices) == 2:
        c_idx = [idx for idx in range(3) if idx not in h_indices][0]
        i1, i2 = h_indices
    else:
        coords = [xyz for _, xyz in atoms]
        central_scores = [
            _distance(coords[idx], coords[(idx + 1) % 3]) + _distance(coords[idx], coords[(idx + 2) % 3])
            for idx in range(3)
        ]
        c_idx = int(np.argmin(central_scores))
        others = [idx for idx in range(3) if idx != c_idx]
        i1, i2 = others[0], others[1]

    c = atoms[c_idx][1]
    a = atoms[i1][1]
    b = atoms[i2][1]

    r_angstrom = 0.5 * (_distance(c, a) + _distance(c, b))
    theta_deg = _angle_deg(a - c, b - c)

    return SurfacePoint(
        file_name=file_name,
        r_angstrom=r_angstrom,
        theta_deg=theta_deg,
        energy_hartree=energy_hartree,
    )


def _read_url_text(url: str) -> str:
    with urlopen(url) as response:  # nosec B310
        return response.read().decode("utf-8", errors="replace")


def _discover_out_urls(base_url: str) -> list[str]:
    html = _read_url_text(base_url)
    parser = OutLinkParser()
    parser.feed(html)

    candidates: set[str] = set()
    for href in parser.links:
        if href.startswith("mailto:"):
            continue
        if ".out" not in href:
            continue
        url = urljoin(base_url, href)
        if "/-/blob/" in url:
            url = url.replace("/-/blob/", "/-/raw/")
        candidates.add(url)

    for match in re.findall(r"https?://[^\s\"'<>]+\.out", html):
        url = match
        if "/-/blob/" in url:
            url = url.replace("/-/blob/", "/-/raw/")
        candidates.add(url)

    return sorted(candidates)


def _load_source(source: str) -> list[tuple[str, str]]:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        if source.lower().endswith(".out"):
            return [(Path(parsed.path).name or "input.out", _read_url_text(source))]

        urls = _discover_out_urls(source)
        if not urls:
            raise ValueError(
                "No .out files discovered at URL. Provide a direct file URL or a page listing .out links."
            )
        return [(Path(urlparse(u).path).name, _read_url_text(u)) for u in urls]

    root = Path(source)
    if not root.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if root.is_file():
        if root.suffix.lower() != ".out":
            raise ValueError(f"File is not a Gaussian .out file: {source}")
        return [(root.name, root.read_text(encoding="utf-8", errors="replace"))]

    out_files = sorted(root.glob("*.out"))
    if not out_files:
        raise ValueError(f"No .out files found in directory: {source}")
    return [(p.name, p.read_text(encoding="utf-8", errors="replace")) for p in out_files]


def load_surface_points(source: str) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    for file_name, content in _load_source(source):
        points.append(parse_gaussian_output(content, file_name=file_name))
    if len(points) < 3:
        raise ValueError("Need at least 3 points to analyze the potential energy surface.")
    return points


def find_equilibrium(points: Iterable[SurfacePoint]) -> SurfacePoint:
    return min(points, key=lambda p: p.energy_hartree)


def _fit_local_quadratic(points: list[SurfacePoint], eq: SurfacePoint) -> tuple[float, float]:
    """
    Fit E(dr, dtheta) = c0 + c1*dr + c2*dtheta + c3*dr^2 + c4*dtheta^2 + c5*dr*dtheta
    around the minimum and return (k_r, k_theta) where:
      k_r = d2E/dr2  (Hartree / Angstrom^2)
      k_theta = d2E/dtheta2 (Hartree / rad^2)
    """
    if len(points) < 20:
        raise ValueError("Need at least 20 points to estimate frequencies robustly.")

    r = np.array([p.r_angstrom for p in points], dtype=float)
    theta_rad = np.radians(np.array([p.theta_deg for p in points], dtype=float))
    e = np.array([p.energy_hartree for p in points], dtype=float)

    dr = r - eq.r_angstrom
    dtheta = theta_rad - math.radians(eq.theta_deg)

    # Bias the fit toward points near equilibrium where harmonic approximation is valid.
    dist = np.sqrt((dr / 0.06) ** 2 + (dtheta / 0.12) ** 2)
    n_fit = min(len(points), max(120, len(points) // 8))
    idx = np.argsort(dist)[:n_fit]

    drf = dr[idx]
    dthf = dtheta[idx]
    ef = e[idx]

    x = np.column_stack(
        [
            np.ones_like(drf),
            drf,
            dthf,
            drf * drf,
            dthf * dthf,
            drf * dthf,
        ]
    )
    coeffs, *_ = np.linalg.lstsq(x, ef, rcond=None)
    c3 = float(coeffs[3])
    c4 = float(coeffs[4])

    k_r = 2.0 * c3
    k_theta = 2.0 * c4
    if k_r <= 0.0 or k_theta <= 0.0:
        raise ValueError("Failed to obtain positive harmonic curvatures near equilibrium.")
    return k_r, k_theta


def estimate_frequencies(points: list[SurfacePoint]) -> dict[str, float]:
    eq = find_equilibrium(points)
    k_r_hartree_per_a2, k_theta_hartree_per_rad2 = _fit_local_quadratic(points, eq)

    k_r_si = k_r_hartree_per_a2 * HARTREE_TO_J / (ANGSTROM_TO_M**2)
    k_theta_si = k_theta_hartree_per_rad2 * HARTREE_TO_J
    r_eq_m = eq.r_angstrom * ANGSTROM_TO_M

    nu1_hz = (1.0 / (2.0 * math.pi)) * math.sqrt(k_r_si / MU1)
    nu2_hz = (1.0 / (2.0 * math.pi)) * math.sqrt(k_theta_si / (r_eq_m**2 * MU2))

    return {
        "nu1_cm^-1": nu1_hz / C_CM_PER_S,
        "nu2_cm^-1": nu2_hz / C_CM_PER_S,
        "k_r_hartree_per_a2": k_r_hartree_per_a2,
        "k_theta_hartree_per_rad2": k_theta_hartree_per_rad2,
    }


def plot_surface(points: list[SurfacePoint], output_path: str) -> None:
    r = np.array([p.r_angstrom for p in points], dtype=float)
    theta = np.array([p.theta_deg for p in points], dtype=float)
    e = np.array([p.energy_hartree for p in points], dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(r, theta, e, cmap="viridis", linewidth=0.2, antialiased=True, alpha=0.95)
    ax.set_xlabel("r (Angstrom)")
    ax.set_ylabel("theta (deg)")
    ax.set_zlabel("Energy (Hartree)")
    ax.set_title("Potential Energy Surface")

    e_min = float(np.min(e))
    e_max = float(np.max(e))
    margin = 0.05 * (e_max - e_min if e_max > e_min else 1.0)
    ax.set_zlim(e_min - margin, e_max + margin)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_surface(source: str, plot_path: str) -> dict[str, float]:
    points = load_surface_points(source)
    eq = find_equilibrium(points)
    freqs = estimate_frequencies(points)
    plot_surface(points, plot_path)

    return {
        "n_points": float(len(points)),
        "equilibrium_energy_hartree": eq.energy_hartree,
        "equilibrium_r_angstrom": eq.r_angstrom,
        "equilibrium_theta_deg": eq.theta_deg,
        "nu1_cm^-1": freqs["nu1_cm^-1"],
        "nu2_cm^-1": freqs["nu2_cm^-1"],
    }
