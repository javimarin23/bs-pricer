"""
Estudio de convergencia log-log: RMSE frente a N para MC estándar y antitético.

Predicción teórica (sin sesgo, muestras i.i.d.):
    RMSE_std(N)  = sigma_g / sqrt(N)
    RMSE_anti(N) = sigma_g * sqrt(1 + rho) / sqrt(N)

Ambas son rectas de pendiente -1/2 en log-log, separadas verticalmente por
0.5 * log10(1 + rho) décadas. La separación es constante en N porque rho es
una propiedad del payoff, no del tamaño de muestra. Un aplanamiento de la
curva a N grande indicaría sesgo (suelo de error), no varianza.
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
    # Contra el precio exacto de Black-Scholes, NO contra la media de las
    # réplicas: así el RMSE recoge sesgo + varianza. Contra la media solo
    # se vería la varianza y un sesgo sistemático quedaría invisible.
    return np.sqrt(np.mean((precios - referencia) ** 2))


def main():
    S0 = 100.0
    K = 80.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    tipo_opcion = "call"

    precio_bs = call_price(S0, K, r, sigma, T)

    # ---------------------------------------------------------------
    # [1] Piloto: sigma_g y rho para las rectas teóricas
    #
    # Misma corrida que el bloque [1] de comparacion_antiteticas.py
    # (N = 50.000, seed = 100), así que reproduce rho = -0.8837.
    # Es independiente del barrido: sus semillas no se solapan (ver [2]).
    # ---------------------------------------------------------------
    N_piloto = 50_000
    seed_piloto = 100

    ST = simulate_ST(S0, r, sigma, T, N_piloto, seed=seed_piloto)
    _, sigma_g, _ = mc_pricer(ST, K, r, T, tipo_opcion)

    ST1, ST2 = simulate_ST_antithetic(S0, r, sigma, T, N_piloto, seed=seed_piloto)
    _, _, rho, factor_predicho = mc_pricer_antithetic(ST1, ST2, K, r, T, tipo_opcion)

    # ---------------------------------------------------------------
    # [2] Barrido en N con R réplicas por punto
    #
    # N: 10 puntos geométricos en [1e2, 1e5] (~3 por década, tres décadas
    # de brazo de palanca para la pendiente). Redondeados a par porque el
    # antitético usa M = N/2 parejas.
    #
    # R = 100: error relativo por punto ~ 1/sqrt(2R) = 7.1% (0.031 décadas);
    # con 10 puntos en 3 décadas, SE(pendiente) ~ 0.01.
    #
    # Semillas independientes por cada par (N, réplica). Con semillas
    # comunes a todos los N las muestras quedarían anidadas, los errores
    # de puntos vecinos correlacionados, y la curva saldría artificialmente
    # suave e invalidaría el ajuste por mínimos cuadrados. Los dos métodos
    # sí comparten semilla dentro de cada (N, réplica), igual que en el
    # contraste: eso no afecta a la pendiente de cada curva.
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # [3] Ajuste de pendientes y separación
    #
    # Criterio de compatibilidad: |pendiente + 0.5| < 3 SE, mismo espíritu
    # que k = 3.891 en los tests (fallo espurio despreciable).
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # [4] Figura
    # Etiquetas en inglés: la figura va al README.
    # ---------------------------------------------------------------
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
