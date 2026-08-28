import numpy as np

# Calcula d1 and d2
def _d1_d2(S, K, r, sigma, T):
    
    if (T<=0):
        raise ValueError("T debe ser mayor que cero")
    
    if (sigma==0):
        raise ValueError("sigma debe ser distinta de cero")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return (d1, d2)