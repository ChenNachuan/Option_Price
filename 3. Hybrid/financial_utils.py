# financial_utils.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# 导入官方 API 定义的数据类
try:
    from cfe_fin_math_api import InstrumentStaticData
except ImportError:
    print("警告: cfe_fin_math_api 未找到。")
    from dataclasses import dataclass
    @dataclass
    class InstrumentStaticData:
        trading_day: str
        instrument_id: str
        expire_day: str
        strike_price: Optional[float] = None
        option_type: Optional[str] = None

EPSILON = 1e-9

def bsm_price(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    q: float, 
    sigma: float, 
    option_type: str = 'call'
) -> float:
    """
    计算 Black-Scholes-Merton (BSM) 期权价格。
    Black-76 用法: S=F, q=r
    """
    if T < EPSILON or sigma < EPSILON:
        if option_type == 'call':
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        elif option_type == 'put':
            return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        else:
            return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
    else:
        raise ValueError("option_type 必须是 'call' 或 'put'")
        
    return price

def implied_volatility(
    market_price: float, 
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    q: float, 
    option_type: str = 'call',
    vol_min: float = 1e-4, 
    vol_max: float = 5.0
) -> float:
    """
    使用 brentq (Brent's method) 数值求解隐含波动率 (IV)。
    """
    def objective_function(sigma):
        if sigma < EPSILON:
            sigma = EPSILON
        return bsm_price(S, K, T, r, q, sigma, option_type) - market_price

    min_price = 0.0
    if option_type == 'call':
        min_price = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        min_price = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price < min_price - EPSILON:
        return np.nan 

    try:
        f_min = objective_function(vol_min)
        f_max = objective_function(vol_max)
        if f_min * f_max > 0:
            return np.nan 
        iv = brentq(objective_function, vol_min, vol_max, xtol=1e-6)
        return iv
    except (ValueError, RuntimeError):
        return np.nan