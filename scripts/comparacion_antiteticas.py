"""
Standard vs. antithetic MC at equal payoff-evaluation budget.

Contrasts the theoretical variance reduction (1 + rho) against a
replication-based empirical measurement. Design rationale (why replication
instead of comparing SE directly, seed choice, tolerance derivation) is in
apuntes_scripts.pdf, section on this script.
"""

import numpy as np

from bs_pricer.black_scholes import call_price
from bs_pricer.monte_carlo import (
    simulate_ST,
    mc_pricer,
    simulate_ST_antithetic,
    mc_pricer_antithetic,
)


def main():
    S0 = 100.0
    K = 80.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    tipo_opcion = "call"

    # [1] Single-run sanity check
    N_total = 50_000
    seed_check = 100

    precio_bs = call_price(S0, K, r, sigma, T)

    ST_std = simulate_ST(S0, r, sigma, T, N_total, seed=seed_check)
    precio_std, _, se_std = mc_pricer(ST_std, K, r, T, tipo_opcion)

    ST1, ST2 = simulate_ST_antithetic(S0, r, sigma, T, N_total, seed=seed_check)
    precio_anti, se_anti, rho, factor_predicho = mc_pricer_antithetic(
        ST1, ST2, K, r, T, tipo_opcion
    )

    print(f"Precio analítico (Black-Scholes): {precio_bs:.4f}")
    print(f"\n[1] MC ESTÁNDAR (N = {N_total})")
    print(f"    Precio: {precio_std:.4f}   SE: {se_std:.6f}")
    print(f"\n[2] ANTITÉTICAS (M = {N_total // 2} parejas, mismo presupuesto)")
    print(f"    Precio: {precio_anti:.4f}   SE: {se_anti:.6f}")
    print(f"    rho(g1, g2) = {rho:.4f}")
    print(f"    Factor de reducción predicho (1 + rho) = {factor_predicho:.4f}")

    # [2] Empirical factor by replication (independent of the TCL formula
    # being validated; not comparable to a single-run SE ratio, which is
    # circular). R=1000, partial pairing across methods -- see PDF.
    R = 1000
    N_rep = 20_000

    precios_std = np.empty(R)
    precios_anti = np.empty(R)

    for i in range(R):
        ST_i = simulate_ST(S0, r, sigma, T, N_rep, seed=i)
        precios_std[i], _, _ = mc_pricer(ST_i, K, r, T, tipo_opcion)

        ST1_i, ST2_i = simulate_ST_antithetic(S0, r, sigma, T, N_rep, seed=i)
        precios_anti[i], _, _, _ = mc_pricer_antithetic(
            ST1_i, ST2_i, K, r, T, tipo_opcion
        )

    var_std = precios_std.var(ddof=1)
    var_anti = precios_anti.var(ddof=1)
    factor_empirico = var_anti / var_std

    # [3] Theory vs. empirical contrast. Tolerance from expected noise
    # 2/sqrt(R) ~ 6.3% (R=1000), never from the observed error. See PDF.
    error_rel = abs(factor_empirico - factor_predicho) / factor_predicho
    TOL_REL = 0.20

    print(f"\n[3] CONTRASTE (R = {R} réplicas, N = {N_rep} por réplica)")
    print(f"    Factor predicho  (1 + rho):        {factor_predicho:.4f}")
    print(f"    Factor empírico  (Var_a / Var_s):  {factor_empirico:.4f}")
    print(f"    Error relativo:                    {error_rel:.2%}")
    print(f"    {'COINCIDE' if error_rel < TOL_REL else 'NO COINCIDE'} (tol = {TOL_REL:.0%})")


if __name__ == "__main__":
    main()