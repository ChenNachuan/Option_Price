# feature_utils.py

import numpy as np
from datetime import datetime, time
from typing import List, Dict, Any, Tuple, Optional

try:
    from cfe_fin_math_api import InstrumentStaticData
except ImportError:
    print("警告: 无法导入 CfeFinMathApi。")
    from dataclasses import dataclass
    @dataclass
    class InstrumentStaticData:
        trading_day: str
        instrument_id: str
        expire_day: str
        strike_price: Optional[float] = None
        option_type: Optional[str] = None

EPSILON = 1e-9

def calculate_time_to_expiry(
    current_datetime: datetime, 
    expiry_datetime: datetime
) -> float:
    """
    精确计算以年为单位的剩余到期时间 T。
    """
    time_delta = expiry_datetime - current_datetime
    total_seconds = time_delta.total_seconds()
    
    if total_seconds <= 0:
        return EPSILON 
        
    seconds_in_year = 365.25 * 24 * 60 * 60
    
    return total_seconds / seconds_in_year

def calculate_log_moneyness(K: float, F: float) -> float:
    """
    计算对数价态 (Log-Moneyness)。
    """
    if F <= 0 or K <= 0:
        return 0.0
    return np.log(K / F)

def parse_option_id(instrument_id: str) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """
    从合约ID (例如 'IO2507-C-3800') 中解析。
    返回: (strike, is_call (1/0), month_id_str ('IO2507'))
    """
    try:
        parts = instrument_id.split('-')
        if len(parts) < 3:
            return None, None, None
        month_id_str = parts[0]
        strike = float(parts[2])
        is_call = 1 if parts[1] == 'C' else 0
        return strike, is_call, month_id_str
    except (IndexError, ValueError, TypeError):
        return None, None, None

def get_static_data_map(
    opt_static_list: List[InstrumentStaticData],
    fut_static_list: List[InstrumentStaticData]
) -> Dict[str, Dict[str, Any]]:
    """
    将官方API的静态数据列表 转换为一个高效的字典 (hash map)，
    包含从期权到对应期货的映射。
    """
    
    fut_month_to_id_map = {}
    for static_data in fut_static_list:
        instrument_id = static_data.instrument_id
        if instrument_id.startswith('IF'):
            month_str = instrument_id[2:6] 
            fut_month_to_id_map[month_str] = instrument_id

    data_map = {}
    for static_data in opt_static_list:
        instrument_id = static_data.instrument_id
        
        # 确保 strike_price 存在，这是一个期权
        if static_data.strike_price is None:
            continue
            
        strike, is_call, month_id_str = parse_option_id(instrument_id)
        
        if strike is not None:
            try:
                # *** 修正：确保 expire_day 是 str ***
                # API 保证了 expire_day 是 str
                expiry_date = datetime.strptime(static_data.expire_day, '%Y%m%d')
                expiry_datetime = expiry_date.replace(hour=15, minute=0)
                
                option_month_str = month_id_str[2:6] 
                corresponding_future_id = fut_month_to_id_map.get(option_month_str)

                if corresponding_future_id:
                    data_map[instrument_id] = {
                        'K': strike,
                        'is_call': is_call,
                        'type': 'call' if is_call == 1 else 'put',
                        'month_id': month_id_str, 
                        'expiry': expiry_datetime,
                        'future_id': corresponding_future_id
                    }
            except ValueError:
                continue
            
    return data_map