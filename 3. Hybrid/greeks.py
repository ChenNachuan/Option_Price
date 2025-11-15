# greeks.py

import numpy as np
from scipy.stats import norm
from typing import Tuple

# 导入我们的 Black-76/BSM 公式
from financial_utils import EPSILON

def vectorized_calculate_vega(
    S: np.ndarray, 
    K: np.ndarray, 
    T: np.ndarray, 
    r: float, 
    q: float, 
    sigma: np.ndarray
) -> np.ndarray:
    """
    Vega 的矢量化版本，用于快速批量计算。
    """
    def norm_pdf(x):
        return (1.0 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * x**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        vega = S * np.exp(-q * T) * norm_pdf(d1) * np.sqrt(T)
        
        vega[T < EPSILON] = 0.0
        vega[sigma < EPSILON] = 0.0
        vega[~np.isfinite(vega)] = 0.0
    
    return vega / 100.0 # 每 1% 波动率