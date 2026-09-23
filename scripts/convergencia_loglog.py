"""
Log-log convergence study: RMSE vs. N for standard and antithetic MC.

Theoretical prediction (unbiased, i.i.d. samples):
    RMSE_std(N)  = sigma_g / sqrt(N)
    RMSE_anti(N) = sigma_g * sqrt(1 + rho) / sqrt(N)

Both are slope -1/2 lines in log-log, separated by 0.5*log10(1+rho) decades,
constant in N. A flattening curve at large N indicates bias, not variance.
Full design rationale (N spacing, R sizing, seed choice, referencing against
the exact price) is in apuntes_scripts.pdf, section on this script.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bs_pricer.black_scholes import call_price
from bs_pricer.monte_carlo import (
    simulate_ST,
    mc_pricer,
    simulate_ST_antithetic,
    mc_pricer_antithetic,
)


def rmse_contra_referencia(precios, referencia):
    # Against the exact Black-Scholes price, not the replica mean: this way
    # RMSE captures bias + variance. See PDF.
    return np.sqrt(np.mean((precios - referencia) ** 2))


def main():
    S0 = 100.0
    K = 80.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    tipo_opcion = "call"

    precio_bs = call_price(S0, K, r, sigma, T)

    # [1] Pilot run for sigma_g and rho, independent of the sweep below.
    N_piloto = 50_000
    seed_piloto = 100

    ST = simulate_ST(S0, r, sigma, T, N_piloto, seed=seed_piloto)
    _, sigma_g, _ = mc_pricer(ST, K, r, T, tipo_opcion)

    ST1, ST2 = simulate_ST_antithetic(S0, r, sigma, T, N_piloto, seed=seed_piloto)
    _, _, rho, factor_predicho = mc_pricer_antithetic(ST1, ST2, K, r, T, tipo_opcion)

    # [2] Sweep over N with R replicas per point. 10 geometric points in
    # [1e2, 1e5], rounded to even (antithetic uses M = N/2 pairs). R=100.
    # Independent seeds per (N, replica): common seeds would nest the
    # samples and invalidate the least-squares slope fit. See PDF.
    N_valores = np.unique(
        2 * np.round(np.geomspace(100, 100_000, 10) / 2).astype(int)
    )
    R = 100

    rmse_std = np.empty(len(N_valores))
    rmse_anti = np.empty(len(N_valores))

    for j, N in enumerate(N_valores):
        N = int(N)
        precios_std = np.empty(R)
        precios_anti = np.empty(R)

        for i in range(R):
            seed = 1_000_000 * (j + 1) + i

            ST_i = simulate_ST(S0, r, sigma, T, N, seed=seed)
            precios_std[i], _, _ = mc_pricer(ST_i, K, r, T, tipo_opcion)

            ST1_i, ST2_i = simulate_ST_antithetic(S0, r, sigma, T, N, seed=seed)
            precios_anti[i], _, _, _ = mc_pricer_antithetic(
                ST1_i, ST2_i, K, r, T, tipo_opcion
            )

        rmse_std[j] = rmse_contra_referencia(precios_std, precio_bs)
        rmse_anti[j] = rmse_contra_referencia(precios_anti, precio_bs)

    # [3] Slope fit and offset. Compatibility: |slope + 0.5| < 3*SE.
    x = np.log10(N_valores)

    (pend_std, _), cov_std = np.polyfit(x, np.log10(rmse_std), 1, cov=True)
    (pend_anti, _), cov_anti = np.polyfit(x, np.log10(rmse_anti), 1, cov=True)
    se_std = np.sqrt(cov_std[0, 0])
    se_anti = np.sqrt(cov_anti[0, 0])

    sep_medida = np.mean(np.log10(rmse_anti) - np.log10(rmse_std))
    sep_predicha = 0.5 * np.log10(factor_predicho)

    def veredicto(p, se):
        return "COMPATIBLE" if abs(p + 0.5) < 3 * se else "NO COMPATIBLE"

    print(f"Precio analítico (Black-Scholes): {precio_bs:.4f}")
    print(f"Piloto: sigma_g = {sigma_g:.4f}, rho = {rho:.4f}")
    print(f"\nBarrido: {len(N_valores)} valores de N en [{N_valores[0]}, {N_valores[-1]}], R = {R}")
    print(f"\n{'N':>8} {'RMSE std':>12} {'RMSE anti':>12} {'cociente':>10}")
    for N, a, b in zip(N_valores, rmse_std, rmse_anti):
        print(f"{N:>8} {a:>12.5f} {b:>12.5f} {b / a:>10.4f}")

    print(f"\nPendiente estándar:   {pend_std:+.4f} ± {se_std:.4f}   {veredicto(pend_std, se_std)} con -0.5")
    print(f"Pendiente antitético: {pend_anti:+.4f} ± {se_anti:.4f}   {veredicto(pend_anti, se_anti)} con -0.5")
    print(f"\nSeparación vertical (décadas log10):")
    print(f"    Predicha  0.5*log10(1+rho): {sep_predicha:+.4f}")
    print(f"    Medida    (media):          {sep_medida:+.4f}")

    # [4] Figure -- English labels, goes into the README
    N_linea = np.geomspace(N_valores[0], N_valores[-1], 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(N_valores, rmse_std, "o", color="C0", label="Standard MC (measured)")
    ax.loglog(N_valores, rmse_anti, "s", color="C1", label="Antithetic variates (measured)")
    ax.loglog(N_linea, sigma_g / np.sqrt(N_linea), "--", color="C0",
              label=r"Theory: $\sigma_g/\sqrt{N}$")
    ax.loglog(N_linea, sigma_g * np.sqrt(factor_predicho) / np.sqrt(N_linea), "--",
              color="C1", label=r"Theory: $\sigma_g\sqrt{1+\rho}/\sqrt{N}$")

    ax.set_xlabel(r"Number of payoff evaluations $N$")
    ax.set_ylabel("RMSE vs. Black-Scholes price")
    ax.set_title(
        rf"European call ($S_0$={S0:g}, $K$={K:g}, $\sigma$={sigma:g}, $T$={T:g}), "
        f"{R} replicas per point"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    carpeta = Path(__file__).resolve().parent.parent / "figures"
    carpeta.mkdir(exist_ok=True)
    fig.savefig(carpeta / "convergencia_loglog.png", dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada en {carpeta / 'convergencia_loglog.png'}")

    plt.show()


if __name__ == "__main__":
    main()