import numpy as np
from datetime import datetime, time

# 交易结束时间 (15:00:00)
TRADE_END_TIME = time(15, 0, 0)
SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60 # (使用 365.25 更精确)
EPSILON = 1e-9 # 用于防止除以零

def parse_date_time(day_str: str, sec_str: str, msec: int) -> datetime:
    """将API的日期和时间字符串合并为datetime对象"""
    base_dt = datetime.strptime(f"{day_str} {sec_str}", "%Y%m%d %H:%M:%S")
    return base_dt.replace(microsecond=msec * 1000)

def calculate_T(current_dt: datetime, expire_day: str) -> float:
    """
    (非向量化) 计算以年为单位的精确剩余到期时间 (T)
    """
    # expire_day 是 YYYYMMDD 字符串, current_dt 是 datetime 对象
    try:
        expire_dt = datetime.strptime(expire_day, '%Y%m%d').replace(hour=15, minute=0)
    except Exception as e:
        print(f"解析到期日 {expire_day} 失败: {e}")
        return EPSILON
        
    time_diff_seconds = (expire_dt - current_dt).total_seconds()
    
    # 保证T是一个小的正数
    return max(time_diff_seconds / SECONDS_IN_YEAR, EPSILON)

def get_mid_price(bid_p: float, bid_v: int, ask_p: float, ask_v: int) -> float | None:
    """
    计算一个有效的中间价。
    """
    if bid_v > 0 and ask_v > 0 and ask_p > bid_p:
        return (bid_p + ask_p) / 2.0
    elif bid_v > 0 and ask_v == 0:
        return bid_p
    elif bid_v == 0 and ask_v > 0:
        return ask_p
    return None

# *** 修正：添加 T 作为参数，与训练脚本保持一致 ***
def get_log_moneyness(F: float, K: float, T: float) -> float:
    """
    (非向量化) 计算对数超额收益
    """
    if F <= 0 or K <= 0 or T <= 0:
        return np.nan
    return np.log(K / F)