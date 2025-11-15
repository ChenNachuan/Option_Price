# main.py

"""
期权定价项目 - 计划一：动态曲面拟合
主执行文件 (已集成 financial_utils 和 volatility_models)

职责：
1. 初始化 API 和数据。
2. 运行 iter_test 循环。
3. (每500ms) 调用 financial_utils 估算 S 和 q。
4. (每10min) 当 option_md 出现时：
    a. 调用 financial_utils 反解隐含波动率 (IV)。
    b. 调用 volatility_models 拟合新的曲面。
    c. 更新波动率模型缓存。
5. (每500ms) 使用缓存中的模型进行预测：
    a. 从缓存获取对应月份的模型。
    b. 调用 model.predict() 获取 sigma。
    c. 调用 financial_utils.bsm_price() 计算 theoprice。
6. 提交预测结果。
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- 1. 导入官方 API ---
try:
    from cfe_fin_math_api import (
        CfeFinMathApi, 
        InstrumentStaticData, 
        MarketData, 
        SamplePrediction
    )
except ImportError:
    print("错误: 无法导入 'cfe_fin_math_api'。请确保 cfe_fin_math_api.py 在同一目录下。")
    sys.exit(1)

# --- 2. 导入我们的辅助库 ---
try:
    import financial_utils as fin_utils
    import volatility_models as vol_models
except ImportError:
    print("错误: 无法导入 'financial_utils.py' 或 'volatility_models.py'。")
    print("请确保这两个文件与 main.py 在同一目录下。")
    sys.exit(1)

# --- 3. 全局常量和配置 ---
RISK_FREE_RATE = 0.0394  # 假设固定无风险利率为 2.5%
VOL_FIT_METHOD = 'spline' # 选择 'spline' (三次样条) 或 'poly' (二阶多项式)
DEFAULT_S = 3800.0      # 标的现货价的冷启动默认值
DEFAULT_Q = 0.027       # 股息率的冷启动默认值

# --- 4. 主执行函数 ---
def main():
    """
    主执行函数
    """
    
    # --- 4.1 初始化 ---
    print("正在初始化 API 和静态数据...")
    try:
        api: CfeFinMathApi = CfeFinMathApi()
        opt_static_list: List[InstrumentStaticData] = api.get_option_static_md()
        fut_static_list: List[InstrumentStaticData] = api.get_future_static_md()
    except Exception as e:
        print(f"API 初始化失败: {e}")
        return

    # 将静态数据列表转换为高效的字典映射 (Hash Map)
    opt_static_map = fin_utils.get_static_data_map(opt_static_list, is_option=True)
    fut_static_map = fin_utils.get_static_data_map(fut_static_list, is_option=False)
    
    # 确定用于估算 S 和 q 的近月和次月期货
    sorted_futures = sorted(fut_static_map.items(), key=lambda item: item[1]['expiry'])
    
    if len(sorted_futures) < 2:
        print("错误：需要至少两个期货合约来估算 S 和 q。")
        return
        
    NEAR_FUT_ID = sorted_futures[0][0]
    NEAR_FUT_EXPIRY = sorted_futures[0][1]['expiry']
    NEXT_FUT_ID = sorted_futures[1][0]
    NEXT_FUT_EXPIRY = sorted_futures[1][1]['expiry']
    
    print(f"将使用 {NEAR_FUT_ID} 和 {NEXT_FUT_ID} 来估算 S 和 q。")

    # --- 4.2 初始化状态和缓存 ---
    
    # 波动率曲面缓存：{ 'IO2507': fitted_model, 'IO2508': fitted_model, ... }
    vol_surface_cache: Dict[str, vol_models.BaseVolatilityModel] = {}
    
    # 实时状态变量
    current_S: float = DEFAULT_S
    current_q: float = DEFAULT_Q
    
    # 预测价格的目标列表
    target_prices = [0.0] * len(opt_static_list)
    
    # --- 4.3 获取 API 迭代器 ---
    iter_test = api.iter_test()
    
    print("初始化完成，开始进入主循环...")

    # --- 4.4 主循环 ---
    for future_md, option_md, predict in iter_test:
        
        try:
            # --- 步骤 1: (每500ms) 估算 S 和 q ---
            
            # 1.1 获取当前时间
            # *** 这是被修正的行 ***
            current_time_str = f"{future_md.trading_day} {future_md.update_sec}.{future_md.update_msec:03d}"
            # *********************
            
            current_datetime = datetime.strptime(current_time_str, '%Y%m%d %H:%M:%S.%f')

            # 1.2 获取最新的期货价格
            latest_futures_prices: Dict[str, float] = {
                inst: price for inst, price in zip(future_md.instrument_id, future_md.last_price)
            }
            
            F1 = latest_futures_prices.get(NEAR_FUT_ID)
            F2 = latest_futures_prices.get(NEXT_FUT_ID)
            
            # 1.3 计算期货到期时间
            T1 = fin_utils.calculate_time_to_expiry(current_datetime, NEAR_FUT_EXPIRY)
            T2 = fin_utils.calculate_time_to_expiry(current_datetime, NEXT_FUT_EXPIRY)
            
            # 1.4 估算 S 和 q
            if F1 and F2 and T1 > fin_utils.EPSILON and T2 > fin_utils.EPSILON:
                S_est, q_est = fin_utils.estimate_S_and_q(F1, T1, F2, T2, RISK_FREE_RATE)
                
                # 更新全局状态
                if S_est is not None and q_est is not None:
                    current_S = S_est
                    current_q = q_est
                    
            # --- 步骤 2: (每10min) "学习" - 拟合波动率曲面 ---
            if option_md:
                
                # 2.1 收集所有合约的 (ID, IV) 数据
                iv_data_for_fitter: Dict[str, float] = {}
                
                for i in range(len(option_md.instrument_id)):
                    instrument_id = option_md.instrument_id[i]
                    
                    # 2.2 计算 midprice，并处理无效价格
                    bid = option_md.bid_p1[i]
                    ask = option_md.ask_p1[i]
                    
                    if np.isnan(bid) or bid <= 0:
                        if np.isnan(ask) or ask <= 0:
                            continue # 没有有效价格
                        midprice = ask
                    elif np.isnan(ask) or ask <= 0:
                        midprice = bid
                    else:
                        midprice = (bid + ask) * 0.5
                    
                    # 2.3 获取合约静态信息
                    static_info = opt_static_map.get(instrument_id)
                    if not static_info:
                        continue # 找不到静态数据
                    
                    K = static_info['strike']
                    opt_type = static_info['type']
                    T_opt = fin_utils.calculate_time_to_expiry(current_datetime, static_info['expiry'])
                    
                    if T_opt < fin_utils.EPSILON:
                        continue # 合约已过期
                        
                    # 2.4 反解隐含波动率 (IV)
                    iv = fin_utils.implied_volatility(
                        midprice, current_S, K, T_opt, RISK_FREE_RATE, current_q, opt_type
                    )
                    
                    # 仅当IV有效时才将其用于拟合
                    if iv is not None and np.isfinite(iv) and iv > 0.01: # 至少 1%
                        iv_data_for_fitter[instrument_id] = iv
                
                # 2.5 按月份拟合曲面
                if iv_data_for_fitter:
                    fitted_models = vol_models.fit_surfaces_by_month(
                        iv_data_for_fitter,
                        opt_static_map,
                        method=VOL_FIT_METHOD
                    )
                    
                    # 2.6 *** 核心步骤: 更新全局缓存 ***
                    vol_surface_cache.update(fitted_models)

            # --- 步骤 3: (每500ms) "预测" - 计算理论价格 ---
            
            for i in range(len(opt_static_list)):
                instrument_id = opt_static_list[i].instrument_id
                month_id = instrument_id.split('-')[0]
                
                static_info = opt_static_map.get(instrument_id)
                if not static_info:
                    target_prices[i] = 0.0
                    continue
                    
                K = static_info['strike']
                opt_type = static_info['type']
                T_opt = fin_utils.calculate_time_to_expiry(current_datetime, static_info['expiry'])

                if T_opt < fin_utils.EPSILON:
                    target_prices[i] = 0.0
                    continue

                # 3.1 从缓存中获取该月份的模型
                model = vol_surface_cache.get(month_id)
                
                # 3.2 预测 sigma
                if model and model.is_fitted:
                    predicted_sigma = model.predict(K)
                else:
                    predicted_sigma = vol_models.DEFAULT_VOLATILITY
                    
                # 3.3 计算 BSM 价格
                theoprice = fin_utils.bsm_price(
                    current_S, K, T_opt, RISK_FREE_RATE, current_q, predicted_sigma, opt_type
                )
                
                if not np.isfinite(theoprice):
                    theoprice = 0.0
                    
                target_prices[i] = theoprice
            
            # --- 步骤 4: (每500ms) 提交预测 ---
            predict.target = target_prices
            api.predict(predict)

        except Exception as e:
            print(f"主循环出错: {e}")
            try:
                predict.target = target_prices 
                api.predict(predict)
            except:
                pass 

# --- 5. 运行主函数 (标准 Python 入口) ---
if __name__ == "__main__":
    main()