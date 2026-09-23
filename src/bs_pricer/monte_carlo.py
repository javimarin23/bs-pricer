import numpy as np


def simulate_ST(S0, r, sigma, T, n_evals, seed=None):
    """Simulate terminal prices S_T under Q using the exact GBM transition."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_evals)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    return S0 * np.exp(drift + diffusion)


def mc_pricer(ST_samples, K, r, T, tipo_opcion):
    """Standard Monte Carlo price, sample stdev and standard error."""
    n_evals = len(ST_samples)

    if tipo_opcion == "call":
        payoff = np.maximum(ST_samples - K, 0.0)
    elif tipo_opcion == "put":
        payoff = np.maximum(K - ST_samples, 0.0)
    else:
        raise ValueError(f"tipo_opcion debe ser 'call' o 'put', no {tipo_opcion!r}")

    discounted_payoff = np.exp(-r * T) * payoff

    precio_estimado = np.mean(discounted_payoff)
    stdev_muestral = np.std(discounted_payoff, ddof=1)
    error_estandar = stdev_muestral / np.sqrt(n_evals)

    return precio_estimado, stdev_muestral, error_estandar


def simulate_ST_antithetic(S0, r, sigma, T, n_evals, seed=None):
    """Simulate terminal prices in antithetic pairs.

    n_evals is the payoff-evaluation budget, not the number of pairs: M =
    n_evals // 2 normals are drawn, each used twice (Z and -Z). This makes a
    comparison against simulate_ST with the same n_evals an equal-cost one.

    Returns (ST1, ST2), each of length M. ST1[i] and ST2[i] are the two
    branches of the same pair, NOT independent samples.
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
    """Antithetic Monte Carlo price from an already-simulated pair (ST1, ST2).

    Returns (precio, error_estandar, rho, factor_predicho), where rho is the
    sample correlation between the two branches and factor_predicho = 1 + rho
    is the variance reduction predicted by theory against standard MC at
    equal budget.
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

    # The independent sample unit is the pair average, not each branch alone.
    Y = 0.5 * (g1 + g2)
    M = Y.size

    precio = Y.mean()
    sigma_Y = Y.std(ddof=1)
    error_estandar = sigma_Y / np.sqrt(M)

    rho = np.corrcoef(g1, g2)[0, 1]

    return precio, error_estandar, rho, 1.0 + rho