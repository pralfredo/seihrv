"""
SEIHRV simulation suite — reproduction code for the paper
"High-Dimensional Epidemic Dynamics, Stability, Oscillations, and
 Probabilistic Outcomes" (Upreti).

This single script reproduces all numerical results in Sections 13 and 14:

  Figure 2 (fig_timeseries.pdf):
      I(t) and V(t) in three regimes (extinction, damped, weakly damped).

  Figure 3 (fig_phaseportraits.pdf):
      (I, E) phase-plane trajectories at small and large alpha.

  Figure 4 (fig_bifurcation.pdf):
      Real and imaginary parts of the dominant complex eigenvalue of J(E*)
      as alpha varies (1D sweep).

  Figure 5 (fig_bif2d.pdf  +  bif2d_data.csv):
      2D bifurcation diagram: classification of (beta_1, alpha) grid points
      by stability of the endemic equilibrium.

Run with:    python seihrv_simulations.py
Outputs are written to a ./figures folder next to this script as both PDF and PNG files.
"""

import csv
from collections import Counter

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import solve_ivp
from numpy.linalg import eigvals

# ────────────────────────────────────────────────────────────────────────────
# Plot style
# ────────────────────────────────────────────────────────────────────────────
rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ────────────────────────────────────────────────────────────────────────────
# Output helpers: create real image files in ./figures
# ────────────────────────────────────────────────────────────────────────────
OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)

def save_figure(fig, name):
    """Save a Matplotlib figure as both PDF and PNG in ./figures."""
    pdf_path = OUTDIR / f"{name}.pdf"
    png_path = OUTDIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"  saved {pdf_path}")
    print(f"  saved {png_path}")
    return pdf_path, png_path


# ────────────────────────────────────────────────────────────────────────────
# SEIHRV model
# ────────────────────────────────────────────────────────────────────────────
# Time unit: days. Population is normalized so that the disease-free susceptible
# population S0 = Lambda/mu = 1 (frequency-dependent transmission convention).
# 1/mu ~ 200 days is a compressed demographic timescale for visualization, a
# standard simplification when demographic and disease timescales must be
# comparable (see Hethcote 2000).

def baseline_params():
    mu = 1.0 / 200.0
    return dict(
        Lambda = mu * 1.0,   # so Lambda/mu = 1
        beta1  = 0.30,       # I -> S transmission
        beta2  = 0.10,       # E -> S presymptomatic transmission
        beta3  = 0.03,       # V -> S environmental transmission
        sigma  = 1.0 / 5.2,  # 1 / incubation period
        gamma  = 1.0 / 7.0,  # 1 / infectious period
        delta  = 0.05,       # hospitalization rate from I
        rho    = 1.0 / 10.0, # hospital recovery (10 days)
        alpha  = 0.20,       # viral shedding into V
        eta    = 1.0 / 3.0,  # environmental decay
        mu     = mu,         # natural mortality
    )


def R0_value(p):
    kE = p['sigma'] + p['mu']
    kI = p['gamma'] + p['delta'] + p['mu']
    S0 = p['Lambda'] / p['mu']
    RE = p['beta2'] * S0 / kE
    RI = p['beta1'] * p['sigma'] * S0 / (kE * kI)
    RV = p['beta3'] * p['sigma'] * p['alpha'] * S0 / (kE * kI * p['eta'])
    return RE + RI + RV, (RE, RI, RV)


def seihrv_rhs(t, y, p):
    S, E, I, H, R, V = y
    lam = p['beta1'] * I + p['beta2'] * E + p['beta3'] * V
    return [
        p['Lambda'] - lam * S - p['mu'] * S,
        lam * S - (p['sigma'] + p['mu']) * E,
        p['sigma'] * E - (p['gamma'] + p['delta'] + p['mu']) * I,
        p['delta'] * I - (p['rho'] + p['mu']) * H,
        p['gamma'] * I + p['rho'] * H - p['mu'] * R,
        p['alpha'] * I - p['eta'] * V,
    ]


def initial_conditions(p, seed=1e-3):
    S0 = p['Lambda'] / p['mu']
    return [S0 - seed, 0.0, seed, 0.0, 0.0, 0.0]


def jacobian_at_equilibrium(p, t_eq=5000):
    """Integrate to long-time equilibrium and return the finite-difference Jacobian."""
    sol = solve_ivp(seihrv_rhs, [0, t_eq], initial_conditions(p), args=(p,),
                    t_eval=[t_eq], rtol=1e-10, atol=1e-13, method='LSODA')
    y_eq = sol.y[:, -1]
    eps = 1e-7
    f0 = np.array(seihrv_rhs(0, y_eq, p))
    J = np.zeros((6, 6))
    for j in range(6):
        yj = y_eq.copy()
        yj[j] += eps
        J[:, j] = (np.array(seihrv_rhs(0, yj, p)) - f0) / eps
    return J, y_eq


def leading_complex_eigenvalue(J):
    """Return the complex eigenvalue with largest real part (preferring complex pairs)."""
    evals = eigvals(J)
    complex_pairs = [e for e in evals if abs(e.imag) > 1e-6]
    if complex_pairs:
        return max(complex_pairs, key=lambda e: e.real)
    return max(evals, key=lambda e: e.real)


# ────────────────────────────────────────────────────────────────────────────
# Figure 2: time series in three regimes
# ────────────────────────────────────────────────────────────────────────────
def fig_timeseries():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    t_eval = np.linspace(0, 400, 4001)

    # (a) Extinction: scaled-down transmission so R0 < 1
    p = baseline_params()
    p['beta1'] = 0.08; p['beta2'] = 0.03; p['beta3'] = 0.005
    R0a, _ = R0_value(p)
    sol = solve_ivp(seihrv_rhs, [0, 400], initial_conditions(p, seed=1e-3),
                    args=(p,), t_eval=t_eval, rtol=1e-9, atol=1e-12)
    axes[0].plot(sol.t, sol.y[2], 'b-', lw=1.5, label=r'$I(t)$')
    axes[0].plot(sol.t, sol.y[5], 'g--', lw=1.2, label=r'$V(t)$')
    axes[0].set_title(rf'(a) Extinction: $\mathcal{{R}}_0 = {R0a:.2f} < 1$')
    axes[0].set_xlabel(r'Time $t$ (days)')
    axes[0].set_ylabel(r'Population / pathogen load')
    axes[0].legend(loc='upper right', frameon=False)
    axes[0].set_xlim(0, 200)

    # (b) Damped oscillation to endemic equilibrium
    p = baseline_params()
    p['alpha'] = 0.10
    R0b, _ = R0_value(p)
    sol = solve_ivp(seihrv_rhs, [0, 1500], initial_conditions(p, seed=1e-3),
                    args=(p,), t_eval=np.linspace(0, 1500, 6001),
                    rtol=1e-9, atol=1e-12)
    axes[1].plot(sol.t, sol.y[2], 'b-', lw=1.2, label=r'$I(t)$')
    axes[1].plot(sol.t, sol.y[5], 'g--', lw=1.0, label=r'$V(t)$')
    axes[1].set_title(rf'(b) Damped oscillation: $\mathcal{{R}}_0 = {R0b:.2f}$, $\alpha = 0.10$')
    axes[1].set_xlabel(r'Time $t$ (days)')
    axes[1].legend(loc='upper right', frameon=False)

    # (c) Weakly damped oscillation (large alpha)
    p = baseline_params()
    p['alpha'] = 2.0; p['beta3'] = 0.05; p['eta'] = 0.15
    R0c, _ = R0_value(p)
    sol = solve_ivp(seihrv_rhs, [0, 3000], initial_conditions(p, seed=1e-3),
                    args=(p,), t_eval=np.linspace(0, 3000, 12001),
                    rtol=1e-10, atol=1e-13, method='LSODA')
    axes[2].plot(sol.t, sol.y[2], 'r-', lw=1.0, label=r'$I(t)$')
    axes[2].plot(sol.t,
                 sol.y[5] / np.max(sol.y[5]) * np.max(sol.y[2]),
                 'g--', lw=0.8, label=r'$V(t)$ (rescaled)')
    axes[2].set_title(rf'(c) Weakly damped: $\mathcal{{R}}_0 = {R0c:.2f}$, $\alpha = 2.0$')
    axes[2].set_xlabel(r'Time $t$ (days)')
    axes[2].legend(loc='upper right', frameon=False)

    plt.tight_layout()
    save_figure(fig, 'fig_timeseries')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Figure 3: phase portraits in (I, E)
# ────────────────────────────────────────────────────────────────────────────
def fig_phaseportraits():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # (a) Spiral into endemic equilibrium at small alpha
    p = baseline_params()
    p['alpha'] = 0.10
    R0a, _ = R0_value(p)
    for seed in [5e-3, 2e-3, 1e-2]:
        sol = solve_ivp(seihrv_rhs, [0, 1500],
                        initial_conditions(p, seed=seed),
                        args=(p,), t_eval=np.linspace(0, 1500, 6001),
                        rtol=1e-10, atol=1e-13)
        axes[0].plot(sol.y[2], sol.y[1], lw=0.8, alpha=0.8)
    Iend = np.mean(sol.y[2, -500:])
    Eend = np.mean(sol.y[1, -500:])
    axes[0].plot(Iend, Eend, 'ko', markersize=6, label=r'$\mathcal{E}^*$ (stable)')
    axes[0].set_xlabel(r'Infectious $I(t)$')
    axes[0].set_ylabel(r'Exposed $E(t)$')
    axes[0].set_title(rf'(a) $\alpha = 0.10$, $\mathcal{{R}}_0 = {R0a:.2f}$')
    axes[0].legend(frameon=False)

    # (b) Weakly damped spiral at large alpha
    p = baseline_params()
    p['alpha'] = 2.0; p['beta3'] = 0.05; p['eta'] = 0.15
    R0b, _ = R0_value(p)
    sol = solve_ivp(seihrv_rhs, [0, 3000], initial_conditions(p, seed=1e-3),
                    args=(p,), t_eval=np.linspace(0, 3000, 12001),
                    rtol=1e-10, atol=1e-13, method='LSODA')
    axes[1].plot(sol.y[2], sol.y[1], 'r-', lw=0.5, alpha=0.7)
    Iend = np.mean(sol.y[2, -500:])
    Eend = np.mean(sol.y[1, -500:])
    axes[1].plot(Iend, Eend, 'ko', markersize=6,
                 label=r'$\mathcal{E}^*$ (stable, weakly attracting)')
    axes[1].set_xlabel(r'Infectious $I(t)$')
    axes[1].set_ylabel(r'Exposed $E(t)$')
    axes[1].set_title(rf'(b) $\alpha = 2.0$, $\mathcal{{R}}_0 = {R0b:.2f}$')
    axes[1].legend(frameon=False)

    plt.tight_layout()
    save_figure(fig, 'fig_phaseportraits')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Figure 4: 1D bifurcation -- Re(lambda) and |Im(lambda)| vs alpha
# ────────────────────────────────────────────────────────────────────────────
def fig_bifurcation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    alphas = np.linspace(0.05, 3.0, 40)
    reals, imags = [], []
    for alpha in alphas:
        p = baseline_params()
        p['alpha'] = alpha; p['beta3'] = 0.05; p['eta'] = 0.15
        J, _ = jacobian_at_equilibrium(p, t_eq=3000)
        lead = leading_complex_eigenvalue(J)
        reals.append(lead.real)
        imags.append(abs(lead.imag))

    ax1.plot(alphas, reals, 'b-', lw=1.5, label=r'$\mathrm{Re}\,\lambda$')
    ax1.axhline(0, color='k', ls=':', lw=0.8)
    ax1.set_xlabel(r'Shedding rate $\alpha$')
    ax1.set_ylabel(r'Re $\lambda$ (dominant complex eigenvalue)')
    ax1.set_title(r'(a) Leading eigenvalue of $J(\mathcal{E}^*)$ vs $\alpha$')
    ax1.legend(frameon=False)
    ax1.set_xlim(alphas[0], alphas[-1])

    ax2.plot(alphas, imags, 'r-', lw=1.5, label=r'$|\mathrm{Im}\,\lambda|$')
    ax2.set_xlabel(r'Shedding rate $\alpha$')
    ax2.set_ylabel(r'$|\mathrm{Im}\,\lambda|$ (oscillation frequency)')
    ax2.set_title(r'(b) Oscillation frequency vs $\alpha$')
    ax2.legend(frameon=False)
    ax2.set_xlim(alphas[0], alphas[-1])

    plt.tight_layout()
    save_figure(fig, 'fig_bifurcation')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Figure 5: 2D bifurcation sweep over (beta_1, alpha)
# ────────────────────────────────────────────────────────────────────────────
def classify_point(p):
    """Classify one parameter point by eigenvalue analysis at its equilibrium."""
    R0, _ = R0_value(p)
    if R0 < 1:
        return ('df_stable', R0, None, None)
    try:
        J, _ = jacobian_at_equilibrium(p, t_eq=5000)
    except Exception:
        return ('error', R0, None, None)
    lead = leading_complex_eigenvalue(J)
    cls = 'unstable' if lead.real > 0 else (
        'endemic_spiral' if abs(lead.imag) > 1e-6 else 'endemic_real')
    return (cls, R0, lead.real, abs(lead.imag))


def fig_bif2d():
    """Sweep (beta1, alpha) and produce both a CSV and a heatmap PDF."""
    beta1s = np.linspace(0.06, 0.60, 28)
    alphas = np.linspace(0.05, 4.0, 24)
    results = []
    for b1 in beta1s:
        for a in alphas:
            p = baseline_params()
            p['beta1'] = b1
            p['alpha'] = a
            p['beta3'] = 0.05
            p['eta']   = 0.15
            cls, R0, mre, im = classify_point(p)
            results.append((b1, a, R0, cls, mre, im))

    classes = [r[3] for r in results]
    print(f'  classification counts: {dict(Counter(classes))}')
    print(f'  total points: {len(results)}')

    # Write CSV
    with open(OUTDIR / 'bif2d_data.csv', 'w') as f:
        f.write('beta1,alpha,R0,class,max_re,max_im\n')
        for b1, a, R0, cls, mre, im in results:
            f.write(f'{b1},{a},'
                    f'{R0 if R0 is not None else ""},'
                    f'{cls},'
                    f'{mre if mre is not None else ""},'
                    f'{im if im is not None else ""}\n')
    print(f'  saved {OUTDIR / "bif2d_data.csv"}')

    # Heatmap PDF
    endemic = [r for r in results if r[3] in ('endemic_spiral', 'endemic_real')]
    df_pts  = [r for r in results if r[3] == 'df_stable']
    unstable = [r for r in results if r[3] == 'unstable']

    fig, ax = plt.subplots(figsize=(8, 5))
    if endemic:
        R0s   = [r[2] for r in endemic]
        alphs = [r[1] for r in endemic]
        res   = [r[4] for r in endemic]
        sc = ax.scatter(R0s, alphs, c=res, s=60, marker='s',
                        cmap='RdYlGn_r', edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(r'$\mathrm{Re}\,\lambda$  (dominant complex eigenvalue)')
    if df_pts:
        ax.scatter([r[2] for r in df_pts], [r[1] for r in df_pts],
                   facecolors='none', edgecolors='gray', s=60,
                   label=r'$\mathcal{R}_0 < 1$ (disease-free)')
    if unstable:
        ax.scatter([r[2] for r in unstable], [r[1] for r in unstable],
                   c='red', marker='x', s=80, label='unstable')
    ax.axvline(1.0, color='k', ls='--', lw=1, label=r'$\mathcal{R}_0 = 1$')
    ax.set_xlabel(r'Basic reproduction number $\mathcal{R}_0$')
    ax.set_ylabel(r'Shedding rate $\alpha$')
    ax.set_title(r'Stability of the endemic equilibrium across $(\mathcal{R}_0, \alpha)$')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    plt.tight_layout()
    save_figure(fig, 'fig_bif2d')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = baseline_params()
    R0, comps = R0_value(p)
    print(f'Baseline R0 = {R0:.3f}  '
          f'(RE={comps[0]:.3f}, RI={comps[1]:.3f}, RV={comps[2]:.3f})')

    print('\nFigure 2 (time series):')
    fig_timeseries()

    print('\nFigure 3 (phase portraits):')
    fig_phaseportraits()

    print('\nFigure 4 (1D bifurcation):')
    fig_bifurcation()

    print('\nFigure 5 (2D bifurcation sweep):')
    fig_bif2d()

    print('\nAll figures generated.')
