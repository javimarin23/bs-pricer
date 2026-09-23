"""
Tests del estimador antitético.

Dos propiedades independientes:
  1. Insesgadez: converge al precio analítico de Black-Scholes.
  2. Reducción de varianza: el factor empírico coincide con la predicción 1 + rho.

Semilla fija en ambos. Un test debe ser determinista: con semilla aleatoria,
un fallo de 1 entre 1000 es irreproducible y erosiona la confianza en la suite.
La exploración sobre semillas se hace aparte, no dentro del test.
"""

import numpy as np

from bs_pricer import (
    call_price,
    simulate_ST,
    mc_pricer,
    simulate_ST_antithetic,
    mc_pricer_antithetic,
)

S0, K, r, SIGMA, T = 100.0, 80.0, 0.05, 0.2, 1.0
TIPO = "call"


def test_antithetic_insesgado():
    """El estimador antitético converge al precio analítico.

    Detecta errores en el propio estimador: descuento mal aplicado, deriva
    incorrecta bajo Q, promedio de las parejas mal hecho.

    k = 3.891 (cobertura 99.99%), el mismo criterio que
    test_monte_carlo_call_convergencia. Con semilla fija el resultado es
    determinista, pero la k sigue fijando el margen ante cambios de N, de
    parámetros o de versión de NumPy: una k pequeña haría el test frágil.
    """
    N = 200_000
    SEED = 2024
    K_SIGMAS = 3.891

    precio_bs = call_price(S0, K, r, SIGMA, T)

    ST1, ST2 = simulate_ST_antithetic(S0, r, SIGMA, T, N, seed=SEED)
    precio, se, _, _ = mc_pricer_antithetic(ST1, ST2, K, r, T, TIPO)

    assert abs(precio - precio_bs) < K_SIGMAS * se


def test_antithetic_factor_varianza():
    """La reducción de varianza medida coincide con la predicción 1 + rho.

    El factor empírico se mide como cociente de varianzas ENTRE réplicas, no
    comparando los errores típicos de una sola corrida: eso último es circular,
    porque SE_anti/SE_std es idénticamente 1 + rho_gorro por la fórmula del TCL.

    R = 1000: ruido del cociente de dos varianzas ~ 2/sqrt(R) = 6.3%.
    TOL_REL = 0.20 son ~3 desviaciones de ese ruido, mismo criterio que la
    k del test anterior. La tolerancia sale del ruido ESPERADO del diseño,
    nunca del error observado en una corrida.

    N = 2000 por réplica: el factor 1 + rho no depende de N, así que se usa el
    N más pequeño que mantenga el coste del test por debajo de un segundo.
    """
    R = 1000
    N = 2000
    TOL_REL = 0.20

    precios_std = np.empty(R)
    precios_anti = np.empty(R)
    factores = np.empty(R)

    for i in range(R):
        ST = simulate_ST(S0, r, SIGMA, T, N, seed=i)
        precios_std[i], _, _ = mc_pricer(ST, K, r, T, TIPO)

        ST1, ST2 = simulate_ST_antithetic(S0, r, SIGMA, T, N, seed=i)
        precios_anti[i], _, _, factores[i] = mc_pricer_antithetic(
            ST1, ST2, K, r, T, TIPO
        )

    factor_predicho = factores.mean()
    factor_empirico = precios_anti.var(ddof=1) / precios_std.var(ddof=1)

    # Sin esta línea el test pasaría con parejas NO antitéticas: dos flujos
    # independientes dan rho ~ 0, factor ~ 1, y predicho y empírico coinciden
    # en 1 sin que haya reducción ninguna. Verificado por mutación.
    assert factor_predicho < 1.0

    assert abs(factor_empirico - factor_predicho) / factor_predicho < TOL_REL


def test_antithetic_se_reportado():
    """El error típico que reporta el código coincide con la dispersión real.

    Los dos tests anteriores no tocan el SE: el de varianza no lo usa, y en el
    de insesgadez un SE inflado solo ensancha el intervalo, es decir, lo hace
    más permisivo. Este test cierra ese hueco comparando la fórmula del TCL
    contra una medición independiente.

    Detecta, entre otros, el error de tratar las 2M evaluaciones como muestras
    independientes en vez de M parejas: eso ignora rho e infla el SE en un
    factor 1/sqrt(1+rho) ~ 2.9. Verificado por mutación.

    TOL_REL = 0.10: el SE empírico es una desviación típica estimada con R
    puntos, con ruido relativo ~ 1/sqrt(2R) = 2.2%. La tolerancia son ~4.5
    desviaciones de ese ruido, con margen extra porque a N = 2000 el error del
    estimador conserva algo de asimetría (cola derecha del payoff) y la
    hipótesis de normalidad detrás de 1/sqrt(2R) no es exacta.
    """
    R = 1000
    N = 2000
    TOL_REL = 0.10

    precios = np.empty(R)
    ses = np.empty(R)

    for i in range(R):
        ST1, ST2 = simulate_ST_antithetic(S0, r, SIGMA, T, N, seed=i)
        precios[i], ses[i], _, _ = mc_pricer_antithetic(ST1, ST2, K, r, T, TIPO)

    se_reportado = ses.mean()
    se_empirico = precios.std(ddof=1)

    assert abs(se_reportado - se_empirico) / se_empirico < TOL_REL