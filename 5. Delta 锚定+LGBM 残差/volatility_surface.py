import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

# --- SVI 模型 (GARCH 代理) ---

def svi_raw_formula(k, params):
    """
    SVI 原始公式
    k: log-moneyness
    params: [a, b, rho, m, sigma]
    """
    a, b, rho, m, sigma = params
    
    # 确保 SVI 参数在有效范围内
    b = max(0, b)
    sigma = max(1e-6, sigma)
    rho = max(-0.9999, min(0.9999, rho))
    
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def objective_function(params, k_market, w_market):
    """
    SVI 拟合的目标函数 (最小二乘)
    """
    w_model = svi_raw_formula(k_market, params)
    error = (w_model - w_market)**2
    return np.sum(error)

def fit_svi_surface(moneyness_points: list, w_points: list):
    """
    拟合SVI波动率曲面 (用于 Put 定价)
    """
    df = pd.DataFrame({'k': moneyness_points, 'w': w_points})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df_agg = df.groupby('k')['w'].mean().reset_index().sort_values('k')
    
    if len(df_agg) < 5: # SVI 有5个参数
        print(f"警告: SVI 拟合点不足 ({len(df_agg)})，无法拟合。")
        return None

    k_market = df_agg['k'].values
    w_market = df_agg['w'].values

    initial_a = np.min(w_market)
    initial_b = 0.1
    initial_rho = -0.5
    initial_m = np.mean(k_market)
    initial_sigma = 0.1
    params_0 = [initial_a, initial_b, initial_rho, initial_m, initial_sigma]

    bounds = (
        (1e-6, np.inf),  # a >= 0
        (1e-6, np.inf),  # b >= 0
        (-0.9999, 0.9999), # rho in (-1, 1)
        (np.min(k_market), np.max(k_market)), # m
        (1e-6, np.inf)   # sigma > 0
    )

    try:
        result = minimize(
            objective_function, x0=params_0, args=(k_market, w_market),
            method='L-BFGS-B', bounds=bounds
        )
        if result.success:
            return result.x  # 返回 [a, b, rho, m, sigma]
        else:
            return None
    except Exception:
        return None

def get_vol_from_svi_params(svi_params, k, T):
    """
    从 SVI 参数中查询波动率
    """
    if svi_params is None or np.isnan(k) or T <= 1e-9:
        return 0.20  # 返回默认波动率
    
    w = svi_raw_formula(k, svi_params)
    sigma = np.sqrt(max(w, 1e-9) / T)
    return max(0.05, min(sigma, 1.5))

# --- Cubic Spline 模型 (ML 代理) ---

def fit_spline_surface(moneyness_points: list, iv_points: list):
    """
    使用三次样条插值 (Cubic Spline) 拟合波动率曲面 (用于 Call 定价)
    """
    df = pd.DataFrame({'x': moneyness_points, 'y': iv_points})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df_agg = df.groupby('x')['y'].mean().reset_index().sort_values('x')

    x = df_agg['x'].values
    y = df_agg['y'].values

    if len(x) < 4:
        print(f"警告: Spline 拟合点不足 ({len(x)})，波动率曲面可能不可靠。")
        if len(x) < 2:
            return None # 无法拟合
        return np.poly1d(np.polyfit(x, y, 1)) # 退化为线性

    return CubicSpline(x, y, extrapolate=True)

def get_vol_from_spline(spline_model, moneyness: float) -> float:
    """
    从 Spline 模型中查询波动率
    """
    if spline_model is None or np.isnan(moneyness):
        return 0.20  # 返回默认波动率
    
    vol = float(spline_model(moneyness))
    return max(0.05, min(vol, 1.5))