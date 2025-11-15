import numpy as np
from datetime import datetime, time

# 交易结束时间 (15:00:00)
TRADE_END_TIME = time(15, 0, 0)
SECONDS_IN_YEAR = 365.0 * 24 * 60 * 60

def parse_date_time(day_str: str, sec_str: str, msec: int) -> datetime:
    """将API的日期和时间字符串合并为datetime对象"""
    base_dt = datetime.strptime(f"{day_str} {sec_str}", "%Y%m%d %H:%M:%S")
    return base_dt.replace(microsecond=msec * 1000)

def calculate_T(current_dt: datetime, expire_day: str) -> float:
    """
    计算以年为单位的精确剩余到期时间 (T)
    到期日通常在当天交易结束时 (15:00) 结算
    """
    expire_dt = datetime.strptime(expire_day, "%Y%m%d")
    expire_dt = datetime.combine(expire_dt.date(), TRADE_END_TIME)
    
    time_diff_seconds = (expire_dt - current_dt).total_seconds()
    
    # 保证T是一个小的正数，避免除零错误
    return max(time_diff_seconds / SECONDS_IN_YEAR, 1e-9)

def get_mid_price(bid_p: float, bid_v: int, ask_p: float, ask_v: int) -> float | None:
    """
    计算一个有效的中间价。
    如果买卖盘都有效，返回中间价。
    如果市场无效（如0价），返回None。
    """
    if bid_v > 0 and ask_v > 0 and ask_p > bid_p:
        return (bid_p + ask_p) / 2.0
    elif bid_v > 0 and ask_v == 0:
        return bid_p  # 市场只有买盘
    elif bid_v == 0 and ask_v > 0:
        return ask_p  # 市场只有卖盘
    return None

def get_log_moneyness(F: float, K: float) -> float:
    """
    计算对数超额收益
    F: 标的期货价格
    K: 期权行权价
    """
    # *** 修正：防止 log(0) 或 log(负数) ***
    if F <= 0 or K <= 0:
        return np.nan
    return np.log(K / F)