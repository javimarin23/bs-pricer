import numpy as np
import pytest
from bs_pricer.black_scholes import call_price, put_price, griegas

# Parámetros base para los tests
S, K, r, sigma, T = 100.0, 80.0, 0.05, 0.20, 1.0

# Calculamos la tercera derivada para saber que tolerancia de error utilizar
 # Speed ≈ -5.675e-4 en S=100,K=80,... → justifica atol=1e-6, ver apuntes_tolerancias_numericas#
# eps = 1e-5 

# griegas_calc_up = griegas(S+eps, K, r, sigma, T, "call")
# g_up = griegas_calc_up["Gamma"]
# griegas_calc_down = griegas(S-eps, K, r, sigma, T, "call")
# g_down = griegas_calc_down["Gamma"]

# tercera_derivada_numerica = (g_up - g_down) / (2 * eps)
# print(tercera_derivada_numerica)

# TEST 1: Paridad Put-Call
# C_0 - P_0 = S_0 - K * e^{-rT}
def test_paridad_put_call():
    C_0 = call_price(S, K, r, sigma, T)
    P_0 = put_price(S, K, r, sigma, T)
    
    izq = C_0 - P_0
    der = S - K * np.exp(-r * T)
    
    
    assert np.isclose(izq, der, atol=1e-12)

# TEST 2: Diferencias Finitas contra Griegas Analíticas
# V(x+eps) - V(x-eps) / (2*eps) comparado con la fórmula de las griegas
def test_delta_diferencia_finita():
    eps = 1e-5 
    
    # 1. Delta Matemática
    griegas_calc = griegas(S, K, r, sigma, T, "call")
    delta_analitica = griegas_calc["Delta"]
    
    # 2. Delta Numérica 
    c_up = call_price(S + eps, K, r, sigma, T)
    c_down = call_price(S - eps, K, r, sigma, T)
    delta_numerica = (c_up - c_down) / (2 * eps)
    
    assert np.isclose(delta_analitica, delta_numerica, atol=1e-6)

def test_vega_diferencia_finita():
    eps = 1e-5 
    
    griegas_calc = griegas(S, K, r, sigma, T, "call")
    vega_analitica = griegas_calc["Vega"]
    
    # Diferencia Central moviendo la volatilidad 
    c_up = call_price(S, K, r, sigma + eps, T)
    c_down = call_price(S, K, r, sigma - eps, T)
    vega_numerica = (c_up - c_down) / (2 * eps)
    
    assert np.isclose(vega_analitica, vega_numerica, atol=1e-6)

# TEST 3: Límites Conocidos
def test_limite_volatilidad_cero():
    sigma_cero = 1e-9 
    
    C_0 = call_price(S, K, r, sigma_cero, T)
    valor_determinista = np.exp(-r * T) * max(S * np.exp(r * T) - K, 0)
    
    assert np.isclose(C_0, valor_determinista, atol=1e-9)

def test_limite_precio_infinito():
    S_infinito = 1e6 
    
    C_0 = call_price(S_infinito, K, r, sigma, T)
    griegas_calc = griegas(S_infinito, K, r, sigma, T, "call")
    
    assert np.isclose(griegas_calc["Delta"], 1.0, atol=1e-9)
    assert np.isclose(C_0, S_infinito - K * np.exp(-r * T), rtol=1e-9, atol=0)
    
