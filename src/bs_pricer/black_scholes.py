import numpy as np
from scipy.stats import norm
from bs_pricer.utils import _d1_d2


def call_price(S0, K, r, sigma, T):
    """Closed-form Black-Scholes price of a European call."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)

    termino_A = S0 * norm.cdf(d1)
    termino_B = K * np.exp(-r * T) * norm.cdf(d2)

    return termino_A - termino_B


def put_price(S0, K, r, sigma, T):
    """Closed-form Black-Scholes price of a European put."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)

    termino_A_prima = S0 * norm.cdf(-d1)
    termino_B_prima = K * np.exp(-r * T) * norm.cdf(-d2)

    return termino_B_prima - termino_A_prima


def griegas(S0, K, r, sigma, T, tipo_opcion):
    """Closed-form Greeks (Delta, Gamma, Theta, Vega, Rho) for a European call or put."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)

    # Gamma and Vega are identical for calls and puts
    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)

    # Delta, Theta and Rho differ by option type
    if tipo_opcion == "call":
        delta = norm.cdf(d1)
        theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    elif tipo_opcion == "put":
        delta = norm.cdf(d1) - 1
        theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    else:
        raise ValueError(f"tipo_opcion debe ser 'call' o 'put', no {tipo_opcion!r}")

    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}


if __name__ == "__main__":

    S_0 = 100.0   # Precio actual de la acción (S)
    K = 80.0      # Strike (Precio en el Kink)
    r = 0.05      # Tasa libre de riesgo (5%)
    sigma = 0.2   # Volatilidad (20%)
    T = 1.0       # Tiempo al vencimiento (1 año)
    tipo_opcion = "call"  # call o put

    if tipo_opcion == "call":
        precio = call_price(S_0, K, r, sigma, T)
        print(f"--- ANÁLISIS DE LA OPCIÓN CALL ---")
        print(f"Precio justo (Prima): {precio:.2f} €")
    elif tipo_opcion == "put":
        precio = put_price(S_0, K, r, sigma, T)
        print(f"--- ANÁLISIS DE LA OPCIÓN PUT ---")
        print(f"Precio justo (Prima): {precio:.2f} €")
    else:
        raise ValueError("El tipo de opción debe ser 'call' o 'put'")

    griegas_call = griegas(S_0, K, r, sigma, T, tipo_opcion)

    print("\nGriegas:")
    for nombre, valor in griegas_call.items():
        if nombre in ["Vega", "Rho"]:
            print(f"{nombre}: {valor/100:.4f} (por 1% de cambio)")
        elif nombre == "Theta":
            print(f"{nombre}: {valor/365:.4f} (pérdida de valor diaria)")
        else:
            print(f"{nombre}: {valor:.4f}")