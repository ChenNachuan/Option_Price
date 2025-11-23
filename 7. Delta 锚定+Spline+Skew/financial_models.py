import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# 为Black76模型定义一个微小的T值下限
T_MIN = 1e-9

def black76(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    计算Black76期权价格 (适用于标的为期货的期权)
    """
    T = max(T, T_MIN)
    sigma = max(sigma, 1e-6)

    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'C':
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == 'P':
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    else:
        raise ValueError("option_type 必须是 'C' 或 'P'")
    
    return price

def delta_black76(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    计算Black76模型的Delta
    """
    T = max(T, T_MIN)
    sigma = max(sigma, 1e-6)
    
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'C':
        delta = np.exp(-r * T) * norm.cdf(d1)
    elif option_type == 'P':
        delta = np.exp(-r * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type 必须是 'C' 或 'P'")
        
    return delta

def gamma_black76(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    计算Black76模型的Gamma
    """
    T = max(T, T_MIN)
    sigma = max(sigma, 1e-6)
    
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # N'(d1) (norm.pdf)
    pdf_d1 = norm.pdf(d1)
    
    gamma = np.exp(-r * T) * pdf_d1 / (F * sigma * np.sqrt(T))
    return gamma


def theta_black76(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    计算Black76模型的Theta (每年)
    """
    T = max(T, T_MIN)
    sigma = max(sigma, 1e-6)
    
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d1 = norm.cdf(-d1)
    cdf_neg_d2 = norm.cdf(-d2)

    # (Theta 的标准公式)
    term1 = -(F * np.exp(-r * T) * pdf_d1 * sigma) / (2 * np.sqrt(T))
    
    if option_type == 'C':
        term2 = -r * K * np.exp(-r * T) * cdf_d2
        term3 = r * F * np.exp(-r * T) * cdf_d1
        theta = term1 + term2 + term3
    elif option_type == 'P':
        term2 = r * K * np.exp(-r * T) * cdf_neg_d2
        term3 = -r * F * np.exp(-r * T) * cdf_neg_d1
        theta = term1 + term2 + term3
    else:
        raise ValueError("option_type 必须是 'C' 或 'P'")
    
    # (返回 *每年* 的衰减)
    return theta

def implied_volatility(
    market_price: float, 
    F: float, 
    K: float, 
    T: float, 
    r: float, 
    option_type: str
) -> float:
    """
    使用数值方法 (brentq) 反推隐含波动率
    """
    T = max(T, T_MIN)

    # 目标函数：模型价格 - 市场价格
    def objective_func(sigma):
        return black76(F, K, T, r, sigma, option_type) - market_price

    try:
        # 在一个合理的波动率范围内 (0.1% to 200%) 搜索解
        iv = brentq(objective_func, 1e-3, 2.0, rtol=1e-6)
        return iv
    except ValueError:
        # 如果在该区间内无解
        return np.nan
    
def vega_black76(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    计算 Black76 模型的 Vega (对波动率的敏感度)
    注意：通常 Vega 衡量的是 sigma 变化 1% (0.01) 带来的价格变化，
    但这里我们返回原始导数 dV/dSigma，后续计算时需注意单位。
    """
    T = max(T, T_MIN)
    sigma = max(sigma, 1e-6)
    
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Vega = F * e^(-rT) * n(d1) * sqrt(T)
    vega = F * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
    
    return vega