# financial_utils.py

"""
金融数学辅助函数库 (Financial Mathematics Utility Library)
为期权定价项目提供所有必需的数学公式和计算。
包括：
- BSM (Merton) 定价公式
- 隐含波动率 (IV) 求解器
- 基于期货价格的 S (现货) 和 q (股息率) 估算
- 精确的到期时间 (T) 计算
- 合约信息解析
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# 定义一个极小值，用于防止除零错误
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
   
    
    参数:
    S: 标的资产现货价格
    K: 行权价
    T: 剩余到期时间 (年化)
    r: 无风险利率
    q: 连续股息率
    sigma: 隐含波动率
    option_type: 'call' 或 'put'
    
    返回:
    期权的理论价格
    """
    
    # 防止 T=0 或 sigma=0 导致的除零错误
    if T < EPSILON or sigma < EPSILON:
        if option_type == 'call':
            return max(S - K, 0.0)
        elif option_type == 'put':
            return max(K - S, 0.0)
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
   
    
    参数:
    market_price: 期权的市场价格 (midprice)
    ... (其他 BSM 参数) ...
    vol_min: 波动率搜索下限
    vol_max: 波动率搜索上限
    
    返回:
    隐含波动率 (sigma)，如果无解则返回 np.nan
    """

    # 1. 定义目标函数，其根即为隐含波动率
    def objective_function(sigma):
        if sigma < EPSILON:
            sigma = EPSILON
        
        return bsm_price(S, K, T, r, q, sigma, option_type) - market_price

    # 2. 检查无套利边界
    # 价格不能低于其内涵价值的折现
    min_price = 0.0
    if option_type == 'call':
        min_price = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        min_price = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    if market_price < min_price - EPSILON:
        return np.nan # 价格低于无套利下限，无解

    # 3. 使用 brentq 求解
    try:
        # 检查搜索区间的两端是否异号
        # brentq 需要 f(a) 和 f(b) 符号相反
        f_min = objective_function(vol_min)
        f_max = objective_function(vol_max)

        if f_min * f_max > 0:
            # 如果同号，说明市场价格可能超出了合理波动率范围
            # 尝试扩大搜索范围
            if f_min > 0: # 市场价比 vol_min 对应的价格还低
                return np.nan # 无法定价
            else: # 市场价比 vol_max 对应的价格还高
                return np.nan # 波动率可能高于 500%
                
        iv = brentq(objective_function, vol_min, vol_max, xtol=1e-6)
        
        return iv
    
    except (ValueError, RuntimeError):
        # brentq 在找不到根时会抛出 ValueError
        return np.nan

def calculate_time_to_expiry(
    current_datetime: datetime, 
    expiry_datetime: datetime
) -> float:
    """
    精确计算以年为单位的剩余到期时间 T。
    使用 365.25 天来平均闰年。
    """
    time_delta = expiry_datetime - current_datetime
    
    # 转换为秒，然后除以一年的总秒数
    total_seconds = time_delta.total_seconds()
    
    # 如果已过期，返回一个极小值
    if total_seconds <= 0:
        return EPSILON
        
    # 一年的秒数 (365.25 * 24 * 60 * 60)
    seconds_in_year = 31_557_600 
    
    return total_seconds / seconds_in_year

def estimate_S_and_q(
    F1: float, T1: float, 
    F2: float, T2: float, 
    r: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    根据两份期货合约价格 (F1, F2) 及其到期时间 (T1, T2)，
    反向估算 S (现货价格) 和 q (股息率)。
    
    基于公式: F = S * exp((r - q) * T)
    """
    try:
        # T2 必须大于 T1
        time_diff = T2 - T1
        if time_diff < EPSILON:
            # 时间差太小，无法稳定求解
            return None, None
            
        # 1. 求解 r - q
        # (r - q) = ln(F2 / F1) / (T2 - T1)
        r_minus_q = np.log(F2 / F1) / time_diff
        
        # 2. 求解 q
        q = r - r_minus_q
        
        # 3. 求解 S
        # S = F1 / exp((r - q) * T1)
        S = F1 / np.exp(r_minus_q * T1)
        
        # 避免返回极端值
        if not np.isfinite(S) or not np.isfinite(q):
            return None, None
            
        return S, q
        
    except (ValueError, OverflowError, ZeroDivisionError):
        # 捕获可能的数学错误 (例如 log(0))
        return None, None

def parse_option_id(instrument_id: str) -> Tuple[float, str]:
    """
    从合约ID (例如 'IO2507-C-3800') 中解析出行权价和类型。
    """
    try:
        parts = instrument_id.split('-')
        strike = float(parts[2])
        option_type_char = parts[1]
        
        if option_type_char == 'C':
            option_type = 'call'
        elif option_type_char == 'P':
            option_type = 'put'
        else:
            option_type = 'unknown'
            
        return strike, option_type
        
    except (IndexError, ValueError):
        # 如果ID格式不正确
        return 0.0, 'unknown'

def get_static_data_map(
    static_data_list: List[Any], 
    is_option: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    将官方API的静态数据列表转换为一个高效的字典 (hash map)，
    以便在循环中快速查找。
    
    返回: 
    { 
        'IO2507-C-3800': {
            'strike': 3800.0, 
            'type': 'call', 
            'expiry': datetime_object(2025-07-18 15:00:00)
        },
        ... 
    }
    """
    data_map = {}
    
    for static_data in static_data_list:
        instrument_id = static_data.instrument_id
        
        # 期权合约的到期日是固定的，但时间我们假定为收盘时 15:00
        # 这是计算T (到期时间) 的关键
        try:
            expiry_date = datetime.strptime(static_data.expire_day, '%Y%m%d')
            # 设置为当天交易结束时间
            expiry_datetime = expiry_date.replace(hour=15, minute=0, second=0, microsecond=0)
            
            item_data = {'expiry': expiry_datetime}
            
            if is_option:
                strike, option_type = parse_option_id(instrument_id)
                item_data['strike'] = strike
                item_data['type'] = option_type
            
            data_map[instrument_id] = item_data
            
        except ValueError:
            print(f"警告: 无法解析 {instrument_id} 的日期 {static_data.expire_day}")
            continue
            
    return data_map