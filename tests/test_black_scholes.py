import numpy as np
import pytest
from bs_pricer.black_scholes import call_price, put_price, griegas

# Base parameters for all tests
S, K, r, sigma, T = 100.0, 80.0, 0.05, 0.20, 1.0

# Third derivative (Speed) estimated numerically ~ -5.675e-4 at these
# parameters -> justifies atol=1e-6 for finite-difference tests below.
# See apuntes_tolerancias_numericas / apuntes_antiteticas_convergencia.


def test_paridad_put_call():
    """Put-call parity: C_0 - P_0 = S_0 - K * e^{-rT}."""
    C_0 = call_price(S, K, r, sigma, T)
    P_0 = put_price(S, K, r, sigma, T)

    izq = C_0 - P_0
    der = S - K * np.exp(-r * T)

    assert np.isclose(izq, der, atol=1e-12)


def test_delta_diferencia_finita():
    """Analytical Delta vs. central finite difference of call_price."""
    eps = 1e-5

    griegas_calc = griegas(S, K, r, sigma, T, "call")
    delta_analitica = griegas_calc["Delta"]

    c_up = call_price(S + eps, K, r, sigma, T)
    c_down = call_price(S - eps, K, r, sigma, T)
    delta_numerica = (c_up - c_down) / (2 * eps)

    assert np.isclose(delta_analitica, delta_numerica, atol=1e-6)


def test_vega_diferencia_finita():
    """Analytical Vega vs. central finite difference of call_price."""
    eps = 1e-5

    griegas_calc = griegas(S, K, r, sigma, T, "call")
    vega_analitica = griegas_calc["Vega"]

    c_up = call_price(S, K, r, sigma + eps, T)
    c_down = call_price(S, K, r, sigma - eps, T)
    vega_numerica = (c_up - c_down) / (2 * eps)

    assert np.isclose(vega_analitica, vega_numerica, atol=1e-6)


def test_limite_volatilidad_cero():
    """As sigma -> 0, the call price converges to the deterministic payoff."""
    sigma_cero = 1e-9

    C_0 = call_price(S, K, r, sigma_cero, T)
    valor_determinista = np.exp(-r * T) * max(S * np.exp(r * T) - K, 0)

    assert np.isclose(C_0, valor_determinista, atol=1e-9)


def test_limite_precio_infinito():
    """As S -> infinity, Delta -> 1 and the call behaves like a forward."""
    S_infinito = 1e6

    C_0 = call_price(S_infinito, K, r, sigma, T)
    griegas_calc = griegas(S_infinito, K, r, sigma, T, "call")

    assert np.isclose(griegas_calc["Delta"], 1.0, atol=1e-9)
    assert np.isclose(C_0, S_infinito - K * np.exp(-r * T), rtol=1e-9, atol=0)