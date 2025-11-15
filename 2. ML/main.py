# main.py

"""
期权定价项目 - 计划二：直接机器学习定价 (LightGBM)
主执行文件 (提交用)

职责：
1. 初始化 API, 加载训练好的模型 'model.lgb'。
2. 运行 iter_test 循环。
3. (每10min) 当 option_md 出现时：更新 Volume 和 OpenInterest 的缓存。
4. (每500ms) 
    a. 获取最新的期货价格 F。
    b. 调用 feature_utils 构建特征矩阵。
    c. 调用 model.predict() 批量预测。
5. 提交预测结果。
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import lightgbm as lgb # 导入 LightGBM

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

# --- 2. 导入我们的特征库 ---
try:
    import feature_utils as feat_utils
except ImportError:
    print("错误: 无法导入 'feature_utils.py'。请确保它与 main.py 在同一目录下。")
    sys.exit(1)

# --- 3. 全局常量和配置 ---
MODEL_PATH = 'model.lgb'
RISK_FREE_RATE = 0.0165     # (1.65%) 作为一个固定特征输入
NEAR_FUT_ID = 'IF2507'    # 我们用近月期货价格作为 F

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
        # 获取训练时的特征名称
        TRAINED_FEATURES = model.feature_name()
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保 'model.lgb' 文件与 main.py 在同一目录下。")
        print("你是否已经运行了 'train_model.py' ?")
        return
        
    print(f"模型加载成功。需要的特征: {TRAINED_FEATURES}")

    # 4.3 转换静态数据
    opt_static_map = feat_utils.get_static_data_map(opt_static_list)
    
    # 4.4 初始化状态和缓存
    
    # *** 这是 plan 2 的核心缓存 ***
    # 我们需要缓存 Volume 和 OpenInterest，因为它们只在 10 分钟 tick 中出现
    # 但 500ms 的预测需要它们
    
    # 用 0 初始化
    latest_volume_cache: Dict[str, int] = {inst_id: 0 for inst_id in opt_static_map}
    latest_oi_cache: Dict[str, int] = {inst_id: 0 for inst_id in opt_static_map}
    
    # 预测价格的目标列表
    target_prices = [0.0] * len(opt_static_list)
    
    # --- 4.5 获取 API 迭代器 ---
    iter_test = api.iter_test()
    
    print("初始化完成，开始进入主循环...")

    # --- 4.6 主循环 ---
    for future_md, option_md, predict in iter_test:
        
        try:
            # --- 步骤 1: (每500ms) 获取基础特征 (F, T) ---
            
            # 1.1 获取当前时间
            current_time_str = f"{future_md.trading_day} {future_md.update_sec}.{future_md.update_msec:03d}"
            current_datetime = datetime.strptime(current_time_str, '%Y%m%d %H:%M:%S.%f')

            # 1.2 获取最新的期货价格 (F)
            F_price = np.nan
            for i, inst_id in enumerate(future_md.instrument_id):
                if inst_id == NEAR_FUT_ID:
                    F_price = future_md.last_price[i]
                    break
            
            if np.isnan(F_price):
                # 如果近月期货价格无效，跳过本次预测
                api.predict(predict) # 提交上一次的预测
                continue
                    
            # --- 步骤 2: (每10min) "学习" - 更新特征缓存 ---
            # 这个模块的*唯一*目的就是更新我们的缓存
            
            if option_md:
                for i in range(len(option_md.instrument_id)):
                    inst_id = option_md.instrument_id[i]
                    if inst_id in latest_volume_cache:
                        latest_volume_cache[inst_id] = option_md.volume[i]
                        latest_oi_cache[inst_id] = option_md.open_interest[i]

            # --- 步骤 3: (每500ms) "预测" - 构建特征矩阵并预测 ---
            
            # 3.1 准备一个列表来收集所有合约的特征
            features_list = []
            
            for i in range(len(opt_static_list)):
                instrument_id = opt_static_list[i].instrument_id
                static_info = opt_static_map.get(instrument_id)
                
                if not static_info:
                    features_list.append(None) # 占位
                    continue
                
                # 计算 T
                T_opt = feat_utils.calculate_time_to_expiry(current_datetime, static_info['expiry'])
                
                # 3.2 收集所有特征
                features_list.append([
                    F_price,                      # F_price
                    static_info['K'],             # K
                    T_opt,                        # T
                    static_info['is_call'],       # is_call
                    latest_volume_cache[instrument_id], # volume
                    latest_oi_cache[instrument_id],     # open_interest
                    RISK_FREE_RATE                # r
                ])

            # 3.3 批量预测
            # 将列表转换为 numpy 数组，并确保特征顺序正确
            X_predict = pd.DataFrame(features_list, columns=TRAINED_FEATURES)
            
            predicted_prices = model.predict(X_predict)
            
            # 清理无效预测
            predicted_prices[np.isnan(predicted_prices)] = 0.0
            predicted_prices[predicted_prices < 0] = 0.0
            
            target_prices = predicted_prices.tolist()

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

if __name__ == "__main__":
    main()