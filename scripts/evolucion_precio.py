from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bs_pricer.black_scholes import call_price
from bs_pricer.monte_carlo import simulate_ST, simulate_ST_antithetic


def payoff_descontado(ST, K, r, T, tipo_opcion):
    """Discounted payoff array. Duplicates the formula inside mc_pricer:
    that function only returns the aggregate, not the raw array needed
    here to build a cumulative trajectory."""
    if tipo_opcion == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif tipo_opcion == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"tipo_opcion debe ser 'call' o 'put', no {tipo_opcion!r}")
    return np.exp(-r * T) * payoff


def trayectoria(payoffs, n_puntos=1000):
    """Cumulative mean and standard error, vectorized (O(çN) total).

    Uses running sums of X and X^2 instead of recomputing np.std on each
    prefix, which would be O(N * n_puntos).
    """
    n = len(payoffs)
    idx_min = 100  # menos de 100 muestras da un SE demasiado ruidoso para pintar
    paso = max(1, (n - idx_min) // n_puntos)
    indices = np.arange(idx_min, n, paso)

    cum_sum = np.cumsum(payoffs)
    cum_sq_sum = np.cumsum(payoffs**2)

    n_arr = indices.astype(float)
    media = cum_sum[indices - 1] / n_arr
    # Var muestral = (sum(x^2) - n*media^2) / (n-1)
    var = (cum_sq_sum[indices - 1] - n_arr * media**2) / (n_arr - 1)
    se = np.sqrt(var / n_arr)

    return indices, media, se


def main():
    S0, K, r, sigma, T = 100.0, 80.0, 0.05, 0.2, 1.0
    tipo_opcion = "call"
    n_evals = 100_000
    seed = 7
    k = 3.891  # mismo criterio de cobertura que en los tests

    precio_bs = call_price(S0, K, r, sigma, T)

    # Estandar: n_evals evaluaciones, cada indice i cuesta i evaluaciones.
    ST = simulate_ST(S0, r, sigma, T, n_evals, seed=seed)
    payoffs_std = payoff_descontado(ST, K, r, T, tipo_opcion)
    idx_std, media_std, se_std = trayectoria(payoffs_std)
    x_std = idx_std  # 1 punto = 1 evaluacion

    # Antitetico: n_evals evaluaciones = n_evals/2 pares. Cada indice i de
    # la trayectoria de PARES cuesta 2*i evaluaciones -> eje X escalado
    # por 2 para que sea comparable en la misma unidad que el estandar.
    ST1, ST2 = simulate_ST_antithetic(S0, r, sigma, T, n_evals, seed=seed)
    Y = 0.5 * (payoff_descontado(ST1, K, r, T, tipo_opcion)
               + payoff_descontado(ST2, K, r, T, tipo_opcion))
    idx_anti, media_anti, se_anti = trayectoria(Y)
    x_anti = idx_anti * 2

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.axhline(precio_bs, color="black", linestyle="--", linewidth=1.5,
               label=f"Analytical (${precio_bs:.4f})")

    ax.plot(x_std, media_std, color="C0", lw=1.3, label="Standard MC")
    ax.fill_between(x_std, media_std - k * se_std, media_std + k * se_std,
                     color="C0", alpha=0.15)

    ax.plot(x_anti, media_anti, color="C1", lw=1.3, label="Antithetic variates")
    ax.fill_between(x_anti, media_anti - k * se_anti, media_anti + k * se_anti,
                     color="C1", alpha=0.15)

    ax.set_xlabel("Payoff evaluations")
    ax.set_ylabel("Estimated price ($)")
    ax.set_title(
        f"Price convergence ({k}-sigma band), single run\n"
        rf"European call ($S_0$={S0:g}, $K$={K:g}, $\sigma$={sigma:g}, $T$={T:g})"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    carpeta = Path(__file__).resolve().parent.parent / "figures"
    carpeta.mkdir(exist_ok=True)
    fig.savefig(carpeta / "evolucion_precio.png", dpi=150, bbox_inches="tight")
    print(f"Figura guardada en {carpeta / 'evolucion_precio.png'}")
    print(f"Precio analitico: {precio_bs:.4f}")
    print(f"Precio final estandar:   {media_std[-1]:.4f} (SE={se_std[-1]:.6f})")
    print(f"Precio final antitetico: {media_anti[-1]:.4f} (SE={se_anti[-1]:.6f})")

    plt.show()


if __name__ == "__main__":
    main()