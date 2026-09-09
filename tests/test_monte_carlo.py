import pytest
import numpy as np
from bs_pricer.black_scholes import call_price, put_price
from bs_pricer.monte_carlo import simulate_ST, mc_pricer

def test_monte_carlo_call_convergencia():
    # 1. Parámetros del test
    S = 100.0
    K = 80.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    tipo_opcion = "call"
    
    # Parámetros de diseño justificados
    k = 3.891           # Cobertura 99.99%
    N = 400000          # Dimensionado con piloto (50k) para error relativo < 0.5%
    seed = 50     
    
    # 2. Precio analítico exacto (Black-Scholes)
    precio_analitico = call_price(S, K, r, sigma, T)
    
    # 3. Simulación Monte Carlo
    muestras_ST = simulate_ST(S, r, sigma, T, N, seed)
    precio_mc, _, error_estandar = mc_pricer(muestras_ST, K, r, T, tipo_opcion)
    
    # 4. Construcción del margen IC
    margen_error = k * error_estandar
    
    # 5. Afirmación del test: el valor real debe estar dentro del intervalo
    assert np.abs(precio_mc - precio_analitico) <= margen_error, \
        f"Monte Carlo ({precio_mc:.4f}) diferirá del analítico ({precio_analitico:.4f}) más de {margen_error:.4f}"