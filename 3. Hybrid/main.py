# main.py
# 适用于计划三：混合波动率预测 (ML 预测 IV, BSM/Black76 定价)

"""
期权定价项目 - 计划三：混合波动率预测
主执行文件 (提交用)

职责：
1. 初始化 API, 加载 IV 预测模型 'model.lgb'。
2. 加载 financial_utils (用于 BSM) 和 feature_utils (用于特征)。
3. 运行 iter_test 循环。
4. (每10min) 当 option_md 出现时：更新 Volume 和 OpenInterest 的缓存。
5. (每500ms) 
    a. 获取最新的期货价格 F。
    b. 构建特征矩阵 (Moneyness, T, volume, oi, ...)。
    c. 调用 model.predict() 预测 sigma。
    d. 调用 bsm_price() (Black-76 模式) 计算 ahtoprice。
6. 提交预测结果。
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import lightgbm as lgb

# --- 1. 导入官方 API ---
try:
    from cfe_fin_math_api import (
        CfeFinMathApi, 
        InstrumentStaticData, 
        MarketData, 
        SamplePrediction
    )
except ImportError:
    print("错误: 无法导入 'cfe_fin_math_api'。")
    sys.exit(1)

# --- 2. 导入我们的辅助库 ---
try:
    import feature_utils as feat_utils
    import financial_utils as fin_utils
except ImportError:
    print("错误: 无法导入 'feature_utils.py' 或 'financial_utils.py'。")
    sys.exit(1)

# --- 3. 全局常量和配置 ---
MODEL_PATH = 'model.lgb'
RISK_FREE_RATE = 0.0165     # (1.65%) 我们的固定利率

# --- 4. 主执行函数 ---
def main():
    
    # --- 4.1 初始化 API 和静态数据 ---
    print("正在初始化 API 和静态数据...")
    try:
        api: CfeFinMathApi = CfeFinMathApi()
        opt_static_list: List[InstrumentStaticData] = api.get_option_static_md()
        fut_static_list: List[InstrumentStaticData] = api.get_future_static_md()
    except Exception as e:
        print(f"API 初始化失败: {e}")
        return

    # 4.2 加载训练好的模型
    print(f"正在加载模型: {MODEL_PATH}...")
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
        TRAINED_FEATURES = model.feature_name()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
        
    print(f"模型加载成功。需要的特征: {TRAINED_FEATURES}")

    # 4.3 转换静态数据 (*** 包含 IO -> IF 映射 ***)
    opt_static_map = feat_utils.get_static_data_map(opt_static_list, fut_static_list)
    
    # 4.4 初始化状态和缓存
    latest_volume_cache: Dict[str, int] = {inst_id: 0 for inst_id in opt_static_map}
    latest_oi_cache: Dict[str, int] = {inst_id: 0 for inst_id in opt_static_map}
    latest_futures_prices: Dict[str, float] = {} # { 'IF2507': 3800.0, ... }
    
    target_prices = [0.0] * len(opt_static_list)
    
    # --- 4.5 获取 API 迭代器 ---
    iter_test = api.iter_test()
    
    print("初始化完成，开始进入主循环...")

    # --- 4.6 主循环 ---
    for future_md, option_md, predict in iter_test:
        
        try:
            # --- 步骤 1: (每500ms) 获取基础特征 (F, T) ---
            current_time_str = f"{future_md.trading_day} {future_md.update_sec}.{future_md.update_msec:03d}"
            current_datetime = datetime.strptime(current_time_str, '%Y%m%d %H:%M:%S.%f')

            # 1.2 更新所有期货的最新价格到缓存
            for i, inst_id in enumerate(future_md.instrument_id):
                latest_futures_prices[inst_id] = future_md.last_price[i]
                    
            # --- 步骤 2: (每10min) "学习" - 更新特征缓存 ---
            if option_md:
                for i in range(len(option_md.instrument_id)):
                    inst_id = option_md.instrument_id[i]
                    if inst_id in latest_volume_cache:
                        latest_volume_cache[inst_id] = option_md.volume[i]
                        latest_oi_cache[inst_id] = option_md.open_interest[i]

            # --- 步骤 3: (每500ms) "预测" - 构建特征矩阵并预测 Sigma ---
            
            features_list = []       # 存储特征
            pricing_info_list = [] # 存储用于BSM定价的参数
            
            for i in range(len(opt_static_list)):
                instrument_id = opt_static_list[i].instrument_id
                static_info = opt_static_map.get(instrument_id)
                
                # 如果合约没有对应的期货 (e.g., IO2603 vs IF2512)，我们无法用此模型定价
                if not static_info:
                    features_list.append(None) # 占位
                    pricing_info_list.append(None)
                    continue
                
                # 3.1 收集定价参数
                K = static_info['K']
                opt_type = static_info['type']
                T_opt = feat_utils.calculate_time_to_expiry(current_datetime, static_info['expiry'])
                F_price = latest_futures_prices.get(static_info['future_id'], np.nan)
                
                if T_opt < fin_utils.EPSILON or np.isnan(F_price) or F_price <= 0:
                    features_list.append(None) # 占位
                    pricing_info_list.append({'valid': False})
                    continue
                    
                # 3.2 收集ML特征
                log_moneyness = feat_utils.calculate_log_moneyness(K, F_price)
                
                features_list.append([
                    log_moneyness,                      # log_moneyness
                    T_opt,                              # T
                    latest_volume_cache[instrument_id], # volume
                    latest_oi_cache[instrument_id],     # open_interest
                    static_info['is_call'],             # is_call
                    RISK_FREE_RATE                      # r
                ])
                
                # 3.3 存储定价所需信息
                pricing_info_list.append({
                    'valid': True,
                    'F': F_price,
                    'K': K,
                    'T': T_opt,
                    'r': RISK_FREE_RATE,
                    'type': opt_type
                })

            # 3.4 批量预测 Sigma
            # 构建 DataFrame，确保特征顺序正确
            X_predict = pd.DataFrame(features_list, columns=TRAINED_FEATURES)
            # 预测
            predicted_sigmas = model.predict(X_predict)
            
            # --- 步骤 4: (每500ms) "定价" - 计算 Theoprice ---
            
            for i in range(len(opt_static_list)):
                info = pricing_info_list[i]
                
                # 检查是否为有效合约
                if not info or not info['valid']:
                    target_prices[i] = 0.0
                    continue
                
                sigma = predicted_sigmas[i]
                
                # 防止无效 sigma
                if not np.isfinite(sigma) or sigma <= 0.01:
                    sigma = 0.01 # 最小波动率
                
                # 4.1 调用 Black-76 (S=F, q=r)
                theoprice = fin_utils.bsm_price(
                    S=info['F'], 
                    K=info['K'], 
                    T=info['T'], 
                    r=info['r'], 
                    q=info['r'], # Black-76
                    sigma=sigma, 
                    option_type=info['type']
                )
                
                target_prices[i] = max(0.0, theoprice) # 价格不能为负
            
            # --- 步骤 5: (每500ms) 提交预测 ---
            predict.target = target_prices
            api.predict(predict)

        except Exception as e:
            print(f"主循环出错: {e}")
            try:
                predict.target = target_prices 
                api.predict(predict)
            except:
                pass 

if __name__ == "__main__":
    main()