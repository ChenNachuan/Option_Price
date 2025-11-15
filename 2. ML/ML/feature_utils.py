# feature_utils.py

"""
计划二 (Plan 2)：在线特征工程库

职责：
- 提供 main.py 在 500ms 循环内快速构建特征所需的所有函数。
- 提供 train_model.py 在离线训练时构建特征所需的基础函数。
- 不包含任何 BSM 或 IV 计算。
"""

import numpy as np
from datetime import datetime, time
from typing import List, Dict, Any, Tuple, Optional

# 导入官方 API 定义的数据类 (Data Classes)
# 这对于类型提示是安全的，并且允许我们处理传入的对象
try:
    from cfe_fin_math_api import InstrumentStaticData
except ImportError:
    # 这是一个备用方案，以防在没有API的环境中（如纯pandas训练）导入
    print("警告: 无法导入 CfeFinMathApi。将使用模拟数据类。")
    from dataclasses import dataclass
    
    @dataclass
    class InstrumentStaticData:
        trading_day: str
        instrument_id: str
        expire_day: str
        strike_price: Optional[float]
        option_type: Optional[str]
        # ... 其他字段在这里，但我们只关心上面的

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
    
    # 如果已过期或非常接近，返回 0
    if total_seconds <= 0:
        return 0.0
        
    # 一年的秒数 (365.25 * 24 * 60 * 60)
    seconds_in_year = 31_557_600 
    
    return total_seconds / seconds_in_year

def parse_option_id(instrument_id: str) -> Tuple[Optional[float], Optional[int]]:
    """
    从合约ID (例如 'IO2507-C-3800') 中解析出行权价和类型。
    
    返回: 
        (strike, is_call)
        is_call: 1 代表 'C' (Call), 0 代表 'P' (Put), -1 代表未知
    """
    try:
        parts = instrument_id.split('-')
        if len(parts) < 3:
            return None, None
            
        strike = float(parts[2])
        option_type_char = parts[1]
        
        if option_type_char == 'C':
            is_call = 1
        elif option_type_char == 'P':
            is_call = 0
        else:
            is_call = -1 # 不是有效的期权类型
            
        return strike, is_call
        
    except (IndexError, ValueError, TypeError):
        # 如果ID格式不正确
        return None, None

def get_static_data_map(
    static_data_list: List[InstrumentStaticData]
) -> Dict[str, Dict[str, Any]]:
    """
    将官方API的静态数据列表 转换为一个高效的字典 (hash map)，
    以便在循环中快速查找。
    
    返回: 
    { 
        'IO2507-C-3800': {
            'K': 3800.0, 
            'is_call': 1, 
            'expiry': datetime_object(2025-07-18 15:00:00)
        },
        ... 
    }
    """
    data_map = {}
    
    for static_data in static_data_list:
        instrument_id = static_data.instrument_id
        
        # 我们只关心期权
        if static_data.strike_price is None:
            continue
            
        try:
            # 1. 解析 K 和 Type
            strike, is_call = parse_option_id(instrument_id)
            
            if strike is None:
                continue # 不是有效的期权ID

            # 2. 解析到期日
            expiry_date = datetime.strptime(static_data.expire_day, '%Y%m%d')
            # 假设所有合约在交易日下午 3:00 (15:00) 到期
            expiry_datetime = expiry_date.replace(hour=15, minute=0, second=0, microsecond=0)
            
            # 3. 存储到 map
            data_map[instrument_id] = {
                'K': strike,
                'is_call': is_call,
                'expiry': expiry_datetime
            }
            
        except ValueError:
            print(f"警告: 无法解析静态数据 {instrument_id} 的日期 {static_data.expire_day}")
            continue
            
    return data_map