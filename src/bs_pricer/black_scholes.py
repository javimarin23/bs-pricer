import numpy as np
from scipy.stats import norm # Importamos la distribución normal
from bs_pricer.utils import _d1_d2

# Calcula C_0
def call_price(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    
    termino_A = S * norm.cdf(d1)
    termino_B = K * np.exp(-r * T) * norm.cdf(d2)
    
    return termino_A - termino_B

# Calcula P_0
def put_price(S, K, r, sigma, T):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    
    termino_A_prima = S * norm.cdf(-d1)
    termino_B_prima = K * np.exp(-r * T) * norm.cdf(-d2)
    
    return termino_B_prima - termino_A_prima

# Calculo de las 5 griegas
def griegas(S, K, r, sigma, T, tipo_opcion):
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    
    # GAMMA y VEGA son idénticas para Calls y Puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) 
    
    # DELTA, THETA y RHO
    if (tipo_opcion == "call"):
        delta = norm.cdf(d1)
        
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        
    elif (tipo_opcion == "put"):
        delta = norm.cdf(d1) - 1
        
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("El tipo de opción debe ser 'call' o 'put'")
    
        
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}
    
if __name__ == "__main__":

    S_0 = 100.0   # Precio actual de la acción (S)
    K = 80.0     # Strike (Precio en el Kink)
    r = 0.05      # Tasa libre de riesgo (5%)
    sigma = 0.2  # Volatilidad (20%)
    T = 1.0       # Tiempo al vencimiento (1 año)
    tipo_opcion = "call"  # call o put
    
    
        
    # Calculamos el precio de la Call/Put
    if (tipo_opcion == "call"):
        precio = call_price(S_0, K, r, sigma, T)
        print(f"--- ANÁLISIS DE LA OPCIÓN CALL ---")
        print(f"Precio justo (Prima): {precio:.2f} €")
    elif (tipo_opcion == "put"):
        precio = put_price(S_0, K, r, sigma, T)
        print(f"--- ANÁLISIS DE LA OPCIÓN PUT ---")
        print(f"Precio justo (Prima): {precio:.2f} €")
    else:
        raise ValueError("El tipo de opción debe ser 'call' o 'put'")
        
    # 2. Calculamos las Griegas
    griegas_call = griegas(S_0, K, r, sigma, T, tipo_opcion)
    
    print("\nSensibilidades (Griegas):")
    for nombre, valor in griegas_call.items():
        # Dividimos Vega y Rho por 100 para ver el impacto de un cambio del 1%
        if nombre in ["Vega", "Rho"]:
            print(f"{nombre}: {valor/100:.4f} (por 1% de cambio)")
        # Dividimos Theta por 365 para ver el impacto diario
        elif nombre == "Theta":
            print(f"{nombre}: {valor/365:.4f} (pérdida de valor diaria)")
        else:
            print(f"{nombre}: {valor:.4f}")