from __future__ import annotations

import argparse
import json
from pathlib import Path

from .surface import analyze_surface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise 3.2: parse Gaussian outputs, plot PES, and estimate vibrational frequencies."
    )
    parser.add_argument("source", help="Path to directory/file with .out files, or URL.")
    parser.add_argument(
        "--plot",
        default="energy_surface.png",
        help="Output PNG filename for the surface plot (default: energy_surface.png).",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default="",
        help="Optional JSON output path for computed summary values.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = analyze_surface(args.source, args.plot)

    print(f"Processed points: {int(result['n_points'])}")
    print(f"Equilibrium energy (Hartree): {result['equilibrium_energy_hartree']:.10f}")
    print(f"Equilibrium bond length r (Angstrom): {result['equilibrium_r_angstrom']:.6f}")
    print(f"Equilibrium bond angle theta (deg): {result['equilibrium_theta_deg']:.6f}")
    print(f"Symmetric stretch frequency nu1 (cm^-1): {result['nu1_cm^-1']:.2f}")
    print(f"Bending frequency nu2 (cm^-1): {result['nu2_cm^-1']:.2f}")
    print(f"Surface plot written to: {args.plot}")

    if args.json_path:
        out = Path(args.json_path)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"JSON summary written to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

