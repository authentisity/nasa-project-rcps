"""
Cross-check the steady FLOWUnsteady coefficients against AeroSandbox's vortex
lattice method on the same wing and reference quantities
(docs/phase1_validation.md, section 3).

The wing is FLOWVLM's `simpleWing`: straight leading and trailing edges
between the root chord (twist 0) and the tip chord (twist_tip, rotated about
its leading edge). On a tapered wing the local twist angle is therefore not
linear in span, so the VLM wing is built from sections along those edges.
The VLM runs with one chordwise panel (the lifting line of FLOWUnsteady's
actuator line model) and with eight (a lifting surface).

Usage (needs `pip install aerosandbox`):
    python src/validation/vlm_check.py --data data/processed/wing_steady.pt
"""

import argparse
from pathlib import Path

import aerosandbox as asb
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
B = 2.489  # (m) span, fixed in wing_timeseries_sweep.jl
INPUTS = ["AOA", "ar", "tr", "lambda", "gamma", "twist_tip", "magVinf"]


def vlm(AOA, ar, tr, lam, gam, twist_tip, magVinf, chordwise, n_sections=13):
    """CL and Cm (MAC quarter chord, nose-up +) of the VLM on the sweep's wing."""
    t = lambda a: np.tan(np.radians(a))
    c_tip = B / ar
    c_root = c_tip / tr
    S = B * (c_root + c_tip) / 2
    mac = 2 / 3 * c_root * (1 + tr + tr**2) / (1 + tr)
    y_mac = B / 6 * (1 + 2 * tr) / (1 + tr)

    tip_le = np.array([B / 2 * t(lam), B / 2, B / 2 * t(gam)])
    tip_te = tip_le + c_tip * np.array([np.cos(np.radians(twist_tip)), 0, -np.sin(np.radians(twist_tip))])
    xsecs = []
    for f in np.linspace(0, 1, n_sections):
        le = f * tip_le
        chord = (1 - f) * np.array([c_root, 0, 0]) + f * tip_te - le
        xsecs.append(asb.WingXSec(xyz_le=le, chord=np.hypot(chord[0], chord[2]),
                                  twist=np.degrees(np.arctan2(-chord[2], chord[0])),
                                  airfoil=asb.Airfoil("naca0012")))
    plane = asb.Airplane(wings=[asb.Wing(symmetric=True, xsecs=xsecs)],
                         xyz_ref=[y_mac * t(lam) + mac / 4, 0, y_mac * t(gam)],
                         s_ref=S, c_ref=mac, b_ref=B)
    r = asb.VortexLatticeMethod(plane, asb.OperatingPoint(velocity=magVinf, alpha=AOA),
                                spanwise_resolution=2, chordwise_resolution=chordwise).run()
    return r["CL"], r["Cm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data/processed/wing_steady.pt")
    args = parser.parse_args()

    data = torch.load(args.data, weights_only=False)
    X = data["x_raw"].numpy()[:, [data["input_columns"].index(c) for c in INPUTS]]
    Y, targets = data["y"].numpy(), data["target_columns"]
    AOA, tr, twist = X[:, 0], X[:, 2], X[:, 5]
    print(f"{len(X)} designs, targets {targets}")

    for chordwise in (1, 8):
        CL, Cm = np.array([vlm(*x, chordwise=chordwise) for x in X]).T
        CL_fu = Y[:, targets.index("CL")]
        m = np.abs(CL) > 0.05
        q = np.percentile(CL_fu[m] / CL[m], [5, 50, 95])
        print(f"\n{chordwise} chordwise panel(s)\n  CL: r {np.corrcoef(CL_fu, CL)[0, 1]:.4f}, "
              f"FLOWUnsteady / VLM median {q[1]:.3f} (5-95% {q[0]:.3f} - {q[2]:.3f}), "
              f"residual std {np.std(CL_fu - CL):.4f}")
        print("  dCL/dtwist_tip, FLOWUnsteady / VLM, by taper ratio (fit CL ~ AOA + twist_tip):")
        for lo, hi in [(0.3, 0.45), (0.45, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.0)]:
            b = (tr >= lo) & (tr < hi)
            A = np.column_stack([AOA[b], twist[b], np.ones(b.sum())])
            k_fu, k_vlm = (np.linalg.lstsq(A, v[b], rcond=None)[0] for v in (CL_fu, CL))
            print(f"    tr {lo:.2f}-{hi:.2f} (n {b.sum():3d}): {k_fu[1] / k_vlm[1]:.2f}")
        if "Cm" in targets:
            Cm_fu = Y[:, targets.index("Cm")]
            slope, offset = np.polyfit(Cm, Cm_fu, 1)
            print(f"  Cm: r {np.corrcoef(Cm_fu, Cm)[0, 1]:.3f}, FLOWUnsteady = {slope:.3f} VLM "
                  f"{offset:+.4f}, MAE {np.abs(Cm_fu - Cm).mean():.4f}")


if __name__ == "__main__":
    main()
