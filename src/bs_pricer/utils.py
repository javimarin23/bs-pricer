import numpy as np


def _d1_d2(S0, K, r, sigma, T):
    """Compute d1 and d2 for the Black-Scholes formula."""
    if T <= 0:
        raise ValueError("T debe ser mayor que cero")

    if sigma == 0:
        raise ValueError("sigma debe ser distinta de cero")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return (d1, d2)