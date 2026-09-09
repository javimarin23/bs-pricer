import numpy as np
from bs_pricer.black_scholes import call_price, put_price

# Simula las trayectorias de S_T
def simulate_ST(S, r, sigma, T, N, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST_samples = S * np.exp(drift + diffusion)
    return ST_samples


# Calcula el precio estimado y estadísticos vía Monte Carlo
def mc_pricer(ST_samples, K, r, T, tipo_opcion):
    N = len(ST_samples)

    if tipo_opcion == "call":
        payoff = np.maximum(ST_samples - K, 0.0)
    elif tipo_opcion == "put":
        payoff = np.maximum(K - ST_samples, 0.0)
    else:
        raise ValueError("El tipo de opción debe ser 'call' o 'put'")

    discounted_payoff = np.exp(-r * T) * payoff

    precio_estimado = np.mean(discounted_payoff)
    stdev_muestral = np.std(discounted_payoff, ddof=1)
    error_estandar = stdev_muestral / np.sqrt(N)

    return precio_estimado, stdev_muestral, error_estandar


def simulate_ST_antithetic(S0, r, sigma, T, n_evals, seed=None):
    """
    Simula precios terminales bajo Q por pares antitéticos.

    n_evals es el presupuesto de evaluaciones de payoff, no el número de
    pares: se generan M = n_evals // 2 normales, y cada una se usa dos
    veces (Z y -Z). Así la comparación contra mc_pricer con el mismo
    n_evals es a igualdad de coste.

    Devuelve (ST1, ST2), ambos de longitud M. ST1[i] y ST2[i] son las dos
    ramas del mismo par: NO son muestras independientes.
    """
    rng = np.random.default_rng(seed)
    M = n_evals // 2

    Z = rng.standard_normal(M)

    drift = (r - 0.5 * sigma**2) * T
    difusion = sigma * np.sqrt(T)

    ST1 = S0 * np.exp(drift + difusion * Z)
    ST2 = S0 * np.exp(drift - difusion * Z)

    return ST1, ST2


def mc_pricer_antithetic(ST1, ST2, K, r, T, tipo_opcion):
    """
    Valora por Monte Carlo antitético a partir de un par (ST1, ST2) ya simulado.

    Devuelve (precio, error_estandar, rho, factor_predicho), donde rho es la
    correlación muestral entre las dos ramas y factor_predicho = 1 + rho es la
    reducción de varianza que predice la teoría frente al MC estándar a igual
    presupuesto.
    """
    if tipo_opcion == "call":
        payoff1 = np.maximum(ST1 - K, 0.0)
        payoff2 = np.maximum(ST2 - K, 0.0)
    elif tipo_opcion == "put":
        payoff1 = np.maximum(K - ST1, 0.0)
        payoff2 = np.maximum(K - ST2, 0.0)
    else:
        raise ValueError(f"tipo_opcion debe ser 'call' o 'put', no {tipo_opcion!r}")

    descuento = np.exp(-r * T)
    g1 = descuento * payoff1
    g2 = descuento * payoff2

    # La muestra independiente es el promedio del par, no cada rama por separado.
    Y = 0.5 * (g1 + g2)
    M = Y.size

    precio = Y.mean()
    sigma_Y = Y.std(ddof=1)
    error_estandar = sigma_Y / np.sqrt(M)

    rho = np.corrcoef(g1, g2)[0, 1]

    return precio, error_estandar, rho, 1.0 + rho