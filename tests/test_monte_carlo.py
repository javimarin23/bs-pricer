import pytest
import numpy as np
from bs_pricer.black_scholes import call_price
from bs_pricer.monte_carlo import simulate_ST, mc_pricer


def test_monte_carlo_call_convergencia():
    """Standard MC price falls within a k-sigma band of the analytical price."""
    S = 100.0
    K = 80.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    tipo_opcion = "call"

    # Design parameters, justified
    k = 3.891    # 99.99% coverage
    N = 400_000  # sized from a 50k pilot for < 0.5% relative margin
    seed = 50

    precio_analitico = call_price(S, K, r, sigma, T)

    muestras_ST = simulate_ST(S, r, sigma, T, N, seed)
    precio_mc, _, error_estandar = mc_pricer(muestras_ST, K, r, T, tipo_opcion)

    margen_error = k * error_estandar

    assert np.abs(precio_mc - precio_analitico) <= margen_error, \
        f"Monte Carlo ({precio_mc:.4f}) diferirá del analítico ({precio_analitico:.4f}) más de {margen_error:.4f}"