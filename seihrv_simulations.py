"""
SEIHRV numerical simulations for the paper.

Run:
    python generate_seihrv_figures.py

Creates:
    fig_timeseries.pdf
    fig_phaseportraits.pdf
    fig_bifurcation.pdf

Also creates PNG copies for preview.
"""

from pathlib import Path
import numpy as np

# Use a non-interactive backend so the script works from terminal/LaTeX builds.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import solve_ivp
from numpy.linalg import eigvals


OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def baseline_params():
    """COVID-inspired baseline parameter set; time unit = days."""
    mu = 1.0 / 200.0
    return dict(
        Lambda=mu * 1.0,  # S0 = Lambda/mu = 1
        beta1=0.30,
        beta2=0.10,
        beta3=0.03,
        sigma=1.0 / 5.2,
        gamma=1.0 / 7.0,
        delta=0.05,
        rho=1.0 / 10.0,
        alpha=0.20,
        eta=1.0 / 3.0,
        mu=mu,
    )


def R0_value(p):
    kE = p["sigma"] + p["mu"]
    kI = p["gamma"] + p["delta"] + p["mu"]
    S0 = p["Lambda"] / p["mu"]
    RE = p["beta2"] * S0 / kE
    RI = p["beta1"] * p["sigma"] * S0 / (kE * kI)
    RV = p["beta3"] * p["sigma"] * p["alpha"] * S0 / (kE * kI * p["eta"])
    return RE + RI + RV, (RE, RI, RV)


def seihrv_rhs(t, y, p):
    S, E, I, H, R, V = y
    lam = p["beta1"] * I + p["beta2"] * E + p["beta3"] * V
    dS = p["Lambda"] - lam * S - p["mu"] * S
    dE = lam * S - (p["sigma"] + p["mu"]) * E
    dI = p["sigma"] * E - (p["gamma"] + p["delta"] + p["mu"]) * I
    dH = p["delta"] * I - (p["rho"] + p["mu"]) * H
    dR = p["gamma"] * I + p["rho"] * H - p["mu"] * R
    dV = p["alpha"] * I - p["eta"] * V
    return np.array([dS, dE, dI, dH, dR, dV], dtype=float)


def initial_conditions(p, seed=1e-4):
    S0 = p["Lambda"] / p["mu"]
    return np.array([S0 - seed, 0.0, seed, 0.0, 0.0, 0.0], dtype=float)


def savefig(fig, name):
    pdf = OUTDIR / f"{name}.pdf"
    png = OUTDIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf.resolve()}")
    return pdf


def fig_timeseries():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), constrained_layout=True)

    # (a) R0 < 1
    p = baseline_params()
    p.update(beta1=0.08, beta2=0.03, beta3=0.005)
    R0, _ = R0_value(p)
    sol = solve_ivp(
        seihrv_rhs, [0, 400], initial_conditions(p, seed=1e-3),
        args=(p,), t_eval=np.linspace(0, 400, 4001),
        rtol=1e-9, atol=1e-12, method="LSODA"
    )
    axes[0].plot(sol.t, sol.y[2], lw=1.4, label=r"$I(t)$")
    axes[0].plot(sol.t, sol.y[5], "--", lw=1.1, label=r"$V(t)$")
    axes[0].set_title(rf"(a) Extinction: $\mathcal{{R}}_0={R0:.2f}<1$")
    axes[0].set_xlabel(r"Time $t$ (days)")
    axes[0].set_ylabel("Population / pathogen load")
    axes[0].set_xlim(0, 200)
    axes[0].legend(frameon=False)

    # (b) damped oscillation
    p = baseline_params()
    p["alpha"] = 0.10
    R0, _ = R0_value(p)
    sol = solve_ivp(
        seihrv_rhs, [0, 1500], initial_conditions(p, seed=1e-3),
        args=(p,), t_eval=np.linspace(0, 1500, 6001),
        rtol=1e-9, atol=1e-12, method="LSODA"
    )
    axes[1].plot(sol.t, sol.y[2], lw=1.2, label=r"$I(t)$")
    axes[1].plot(sol.t, sol.y[5], "--", lw=1.0, label=r"$V(t)$")
    axes[1].set_title(rf"(b) Damped: $\mathcal{{R}}_0={R0:.2f}$, $\alpha=0.10$")
    axes[1].set_xlabel(r"Time $t$ (days)")
    axes[1].legend(frameon=False)

    # (c) weakly damped transient
    p = baseline_params()
    p.update(alpha=2.0, beta3=0.05, eta=0.15)
    R0, _ = R0_value(p)
    sol = solve_ivp(
        seihrv_rhs, [0, 3000], initial_conditions(p, seed=1e-3),
        args=(p,), t_eval=np.linspace(0, 3000, 12001),
        rtol=1e-10, atol=1e-13, method="LSODA"
    )
    V_rescaled = sol.y[5] / max(np.max(sol.y[5]), 1e-15) * np.max(sol.y[2])
    axes[2].plot(sol.t, sol.y[2], lw=1.1, label=r"$I(t)$")
    axes[2].plot(sol.t, V_rescaled, "--", lw=0.9, label=r"$V(t)$ rescaled")
    axes[2].set_title(rf"(c) Weakly damped: $\mathcal{{R}}_0={R0:.2f}$, $\alpha=2.0$")
    axes[2].set_xlabel(r"Time $t$ (days)")
    axes[2].legend(frameon=False)

    return savefig(fig, "fig_timeseries")


def fig_phaseportraits():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    # Panel (a)
    p = baseline_params()
    p["alpha"] = 0.10
    R0, _ = R0_value(p)
    last_sol = None
    for seed in [2e-3, 5e-3, 1e-2]:
        sol = solve_ivp(
            seihrv_rhs, [0, 1500], initial_conditions(p, seed=seed),
            args=(p,), t_eval=np.linspace(0, 1500, 6001),
            rtol=1e-10, atol=1e-13, method="LSODA"
        )
        last_sol = sol
        axes[0].plot(sol.y[2], sol.y[1], lw=0.85, alpha=0.85)
    Iend = np.mean(last_sol.y[2, -500:])
    Eend = np.mean(last_sol.y[1, -500:])
    axes[0].plot(Iend, Eend, "ko", markersize=5, label=r"$\mathcal{E}^*$ stable")
    axes[0].set_xlabel(r"Infectious $I(t)$")
    axes[0].set_ylabel(r"Exposed $E(t)$")
    axes[0].set_title(rf"(a) $\alpha=0.10$, $\mathcal{{R}}_0={R0:.2f}$")
    axes[0].legend(frameon=False)

    # Panel (b)
    p = baseline_params()
    p.update(alpha=2.0, beta3=0.05, eta=0.15)
    R0, _ = R0_value(p)
    sol = solve_ivp(
        seihrv_rhs, [0, 3000], initial_conditions(p, seed=1e-3),
        args=(p,), t_eval=np.linspace(0, 3000, 12001),
        rtol=1e-10, atol=1e-13, method="LSODA"
    )
    axes[1].plot(sol.y[2], sol.y[1], lw=0.65, alpha=0.8)
    Iend = np.mean(sol.y[2, -500:])
    Eend = np.mean(sol.y[1, -500:])
    axes[1].plot(Iend, Eend, "ko", markersize=5, label=r"$\mathcal{E}^*$ stable")
    axes[1].set_xlabel(r"Infectious $I(t)$")
    axes[1].set_ylabel(r"Exposed $E(t)$")
    axes[1].set_title(rf"(b) $\alpha=2.0$, $\mathcal{{R}}_0={R0:.2f}$")
    axes[1].legend(frameon=False)

    return savefig(fig, "fig_phaseportraits")


def numerical_jacobian(y0, p, eps=1e-7):
    y0 = np.asarray(y0, dtype=float)
    f0 = seihrv_rhs(0.0, y0, p)
    J = np.zeros((6, 6))
    for j in range(6):
        yj = y0.copy()
        yj[j] += eps
        J[:, j] = (seihrv_rhs(0.0, yj, p) - f0) / eps
    return J


def fig_bifurcation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4), constrained_layout=True)

    alphas = np.linspace(0.05, 3.0, 45)
    real_parts, imag_parts = [], []

    for alpha in alphas:
        p = baseline_params()
        p.update(alpha=float(alpha), beta3=0.05, eta=0.15)

        sol = solve_ivp(
            seihrv_rhs, [0, 3000], initial_conditions(p, seed=1e-3),
            args=(p,), t_eval=[3000],
            rtol=1e-10, atol=1e-13, method="LSODA"
        )
        y_eq = sol.y[:, -1]
        J = numerical_jacobian(y_eq, p)
        ev = eigvals(J)
        complex_evs = [z for z in ev if abs(z.imag) > 1e-7]
        if complex_evs:
            lead = max(complex_evs, key=lambda z: z.real)
            real_parts.append(lead.real)
            imag_parts.append(abs(lead.imag))
        else:
            lead = max(ev, key=lambda z: z.real)
            real_parts.append(lead.real)
            imag_parts.append(0.0)

    ax1.plot(alphas, real_parts, lw=1.5, label=r"$\mathrm{Re}\,\lambda$")
    ax1.axhline(0, color="k", ls=":", lw=0.8)
    ax1.set_xlabel(r"Shedding rate $\alpha$")
    ax1.set_ylabel(r"$\mathrm{Re}\,\lambda$")
    ax1.set_title(r"(a) Dominant complex eigenvalue")
    ax1.legend(frameon=False)

    ax2.plot(alphas, imag_parts, lw=1.5, label=r"$|\mathrm{Im}\,\lambda|$")
    ax2.set_xlabel(r"Shedding rate $\alpha$")
    ax2.set_ylabel(r"$|\mathrm{Im}\,\lambda|$")
    ax2.set_title(r"(b) Oscillation frequency")
    ax2.legend(frameon=False)

    return savefig(fig, "fig_bifurcation")


def main():
    p = baseline_params()
    R0, comps = R0_value(p)
    print(f"Baseline R0 = {R0:.3f} (RE={comps[0]:.3f}, RI={comps[1]:.3f}, RV={comps[2]:.3f})")
    outputs = [fig_timeseries(), fig_phaseportraits(), fig_bifurcation()]
    print("\nAll figures generated:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
