# train_drift_model.py
# 最终优化（方向七）：离线训练一个 "ML-Drift" (残差) 模型
# (*** 关键修正：统一使用 'mid_price' 来计算期货价格，以匹配 'pricing_engine.py' ***)

import os
import sys
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, time
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split

# --- 全局常量 ---
EPSILON = 1e-9 
DATA_ROOT_DIR = '../data' # 确保这指向您数据文件夹的父目录
MODEL_SAVE_PATH = 'model_drift.lgb'
RISK_FREE_RATE = 0.03

FEATURE_NAMES = [
    'S_diff', 
    'S_diff_sq', 
    'T', 
    'K', 
    'log_moneyness', 
    'r', 
    'option_type'
]

def load_all_data(data_root: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    """
    加载所有15天的数据并合并。
    """
    all_future_md = []
    all_option_md = []
    all_static_data = {} 

    if not os.path.exists(data_root):
        print(f"错误: 找不到数据目录 '{data_root}'。")
        print(f"请确保你的15天数据存放在: {os.path.abspath(data_root)}")
        return None

    date_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith('2025')]
    if not date_folders:
        print(f"错误: 在 '{data_root}' 中未找到任何日期文件夹 (如 '20250623')。")
        return None
    print(f"找到 {len(date_folders)} 天的数据...")

    for date in date_folders:
        day_path = os.path.join(data_root, date)
        
        fut_file = os.path.join(day_path, f"IF_{date}.csv")
        if os.path.exists(fut_file):
            try:
                all_future_md.append(pd.read_csv(fut_file))
            except Exception as e:
                print(f"读取 {fut_file} 失败: {e}")

        opt_files = glob.glob(os.path.join(day_path, f"IO*_{date}.csv"))
        for opt_file in opt_files:
            try:
                all_option_md.append(pd.read_csv(opt_file))
            except Exception as e:
                print(f"读取 {opt_file} 失败: {e}")

        static_files = glob.glob(os.path.join(day_path, f"*_{date}_static.csv"))
        for static_file in static_files:
            try:
                df = pd.read_csv(static_file)
                for _, row in df.iterrows():
                    all_static_data[row['instrument_id']] = row.to_dict()
            except Exception as e:
                print(f"读取静态文件 {static_file} 失败: {e}")
                
    if not all_future_md or not all_option_md:
        print("错误: 数据加载不完整，期货或期权行情列表为空。")
        return None

    return pd.concat(all_future_md), pd.concat(all_option_md), all_static_data

def create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 trading_day, update_sec, update_msec 合并为 datetime 对象
    """
    try:
        time_str = df['trading_day'].astype(str) + " " + df['update_sec'].astype(str)
        df['datetime'] = pd.to_datetime(time_str, format='%Y%m%d %H:%M:%S')
        df['datetime'] = df['datetime'] + pd.to_timedelta(df['update_msec'], unit='ms')
        return df
    except Exception as e:
        print(f"创建时间戳时出错: {e}")
        return pd.DataFrame()

def calculate_time_to_expiry_local(current_dt_series: pd.Series, expiry_dt_series: pd.Series) -> pd.Series:
    """
    (向量化) 计算以年为单位的剩余到期时间 (T)
    """
    time_delta = expiry_dt_series - current_dt_series
    total_seconds = time_delta.dt.total_seconds()
    total_seconds[total_seconds <= 0] = EPSILON
    return total_seconds / (365.25 * 24 * 60 * 60)

def get_log_moneyness_local(F_series: pd.Series, K_series: pd.Series, T_series: pd.Series) -> pd.Series:
    """
    (向量化) 计算对数超额收益
    """
    result = pd.Series(np.nan, index=F_series.index)
    valid_mask = (F_series > 0) & (K_series > 0) & (T_series > 0)
    result[valid_mask] = np.log(K_series[valid_mask] / F_series[valid_mask])
    return result

# train_drift_model.py
def calculate_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    (向量化) 计算 'midprice' - (*** 修正：匹配 utils.py 中的逻辑 ***)
    """
    bid_p = df['bid_p1']
    ask_p = df['ask_p1']
    bid_v = df['bid_v1'] # 假设你已加载
    ask_v = df['ask_v1'] # 假设你已加载

    # 条件1：买卖盘都存在
    cond1 = (bid_v > 0) & (ask_v > 0) & (ask_p > bid_p)
    price1 = (bid_p + ask_p) / 2.0
    
    # 条件2：只有买盘
    cond2 = (bid_v > 0) & (ask_v == 0)
    price2 = bid_p
    
    # 条件3：只有卖盘
    cond3 = (bid_v == 0) & (ask_v > 0)
    price3 = ask_p
    
    # 使用 np.select 来应用逻辑
    # 默认值 (cond1,2,3 都不满足) 设为 NaN
    mid_price = np.select(
        [cond1, cond2, cond3], 
        [price1, price2, price3], 
        default=np.nan
    )
    
    # 回退到 last_price (如果 mid_price 仍是 NaN)
    # (注意：原始 utils.py 是返回 None，但训练时你不能有 None)
    # 也许更好的回退是 last_price
    mid_price_series = pd.Series(mid_price, index=df.index)
    mid_price_series = mid_price_series.fillna(df['last_price']) 
    # (确保你的CSV也加载了 'last_price')
    
    return mid_price_series

def main_train():
    """
    主训练函数 (方向七：ML-Drift)
    """
    print("开始离线训练 (方向七：ML-Drift)...")
    
    # 1. 加载数据
    loaded_data = load_all_data(DATA_ROOT_DIR)
    if loaded_data is None: return 
    fut_df, opt_df, static_data_raw = loaded_data
    print(f"总共加载: {len(fut_df)} 条期货行情, {len(opt_df)} 条期权行情")

    # 2. 处理静态数据
    print("正在处理静态数据...")
    static_info_processed = {}
    fut_month_to_id_map = {}
    for inst_id, data in static_data_raw.items():
        try:
            expire_day_str = str(data['expire_day']).split('.')[0] 
            expiry_dt = datetime.strptime(expire_day_str, '%Y%m%d').replace(hour=15, minute=0)
            if 'IF' in inst_id:
                month_str = inst_id[2:6] 
                fut_month_to_id_map[month_str] = inst_id
            elif 'IO' in inst_id:
                option_type = data.get('option_type')
                strike_price = data.get('strike_price')
                is_call = 1 if option_type == 'C' else (0 if option_type == 'P' else -1)
                if is_call != -1 and strike_price is not None and pd.notna(strike_price):
                    static_info_processed[inst_id] = {
                        'expiry': expiry_dt,
                        'K': float(strike_price),
                        'option_type': is_call, # 1 for Call, 0 for Put
                    }
        except Exception: continue
            
    for inst_id, data in static_info_processed.items():
        month_str = inst_id[2:6] 
        static_info_processed[inst_id]['future_id'] = fut_month_to_id_map.get(month_str)

    static_df = pd.DataFrame.from_dict(static_info_processed, orient='index')
    static_df.index.name = 'instrument_id'

    # 3. 处理时间戳和价格
    print("正在处理时间戳和中间价...")
    fut_df = create_datetime_index(fut_df)
    opt_df = create_datetime_index(opt_df)
    if fut_df.empty or opt_df.empty: return

    # *** 关键修正：在 pivot 之前计算期货的 mid_price ***
    fut_df['mid_price'] = calculate_mid_price(fut_df)
    opt_df['midprice'] = calculate_mid_price(opt_df)
    # *** 修正结束 ***

    # 4. 合并数据
    print("正在合并期货、期权和静态数据...")
    # (*** 关键修正：Pivot 'mid_price' 而不是 'last_price' ***)
    fut_df_pivot = fut_df.pivot(index='datetime', columns='instrument_id', values='mid_price')
    fut_df_pivot = fut_df_pivot.sort_index().ffill()
    
    opt_df = opt_df.sort_values('datetime')
    data = pd.merge_asof(opt_df, fut_df_pivot, on='datetime', direction='backward')
    data = pd.merge(data, static_df, left_on='instrument_id', right_index=True, how='left')

    # 5. 构建特征和目标 (Y)
    print("正在构建特征和目标 (Y = Price Drift)...")
    
    # 清理
    data = data.dropna(subset=['midprice', 'future_id', 'K', 'expiry', 'option_type'])
    data = data[data['midprice'] > 0]

    # 匹配对应的期货价格 (现在是 'mid_price')
    data['F_price'] = data.apply(lambda row: row[row['future_id']], axis=1)
    data = data.dropna(subset=['F_price'])
    data = data[data['F_price'] > 0]
    
    data['T'] = calculate_time_to_expiry_local(data['datetime'], data['expiry'])
    data = data[data['T'] > EPSILON]
    
    # 6. *** 关键步骤: 识别和合并锚点 ***
    print("正在识别和合并锚点数据...")
    data['min'] = data['datetime'].dt.minute
    data['sec'] = data['datetime'].dt.second
    data['msec'] = data['datetime'].dt.microsecond / 1000
    
    is_10min = (data['min'] % 10 == 0) & (data['sec'] == 0) & (data['msec'] == 0)
    is_929 = (data['datetime'].dt.time == time(9, 29, 0))
    data['is_anchor'] = is_10min | is_929

    anchor_data = data[data['is_anchor'] == True].copy()
    anchor_data = anchor_data[[
        'datetime', 'instrument_id', 'midprice', 'F_price', 'volume', 'open_interest'
    ]]
    anchor_data = anchor_data.rename(columns={
        'midprice': 'price_anchor', 
        'F_price': 'S_anchor', 
        'volume': 'anchor_volume', 
        'open_interest': 'anchor_open_interest'
    })

    data = pd.merge_asof(data, anchor_data, on='datetime', by='instrument_id', direction='backward')
    data = data.dropna(subset=['price_anchor', 'S_anchor'])

    # 7. *** 最终定义 X 和 Y ***
    print("正在创建最终的 X 和 Y 训练集...")
    
    # 目标 Y = 价格漂移量
    Y = data['midprice'] - data['price_anchor']
    
    # 特征 X
    data['S_diff'] = data['F_price'] - data['S_anchor']
    data['S_diff_sq'] = data['S_diff'] ** 2
    data['log_moneyness'] = get_log_moneyness_local(data['F_price'], data['K'], data['T'])
    data['r'] = RISK_FREE_RATE
    
    X = data[FEATURE_NAMES]
    
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    Y = Y.fillna(0)
    
    if X.empty:
        print("错误：没有构建出任何有效的训练数据。")
        return

    print(f"训练集构建完毕，特征数量: {len(X)}")

    # 8. 训练模型
    print("正在按时间分割训练集和验证集 (后 20% 作为验证)...")

    # 确保数据是按时间排序的 (merge_asof 应该已经保证了，但这里再确认一下)
    # 假设 'data' DataFrame 仍然可用且与 X, Y 索引对齐
    data_sorted = data.sort_values('datetime')
    X = X.loc[data_sorted.index]
    Y = Y.loc[data_sorted.index]

    split_index = int(len(X) * 0.8) # 按 80/20 分割
    # split_date = data_sorted['datetime'].iloc[split_index] # 也可以按日期分割
    # print(f"时间分割点: {split_date}")

    X_train = X.iloc[:split_index]
    y_train = Y.iloc[:split_index]
    X_val = X.iloc[split_index:]
    y_val = Y.iloc[split_index:]

    print(f"训练集: {len(X_train)}， 验证集: {len(X_val)}")
    print("正在训练 LightGBM 模型 (优化目标: MAE)...")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        
        # --- 核心正则化 ---
        # 1. 大幅降低 num_leaves。63 几乎肯定过拟合了。
        #    从 31 (2^5) 或 15 (2^4) 开始尝试。这是最重要的参数。
        num_leaves=15, 
        
        # 2. 增加 L1/L2 正则化
        reg_alpha=0.1,  # L1
        reg_lambda=0.1, # L2

        # 3. 引入随机性 (Bagging)
        subsample=0.8, # (或 bagging_fraction) 只用 80% 的数据
        colsample_bytree=0.8, # (或 feature_fraction) 只用 80% 的特征

        # --- 其他参数保持不变 ---
        n_jobs=-1,
        random_state=42,
        objective='regression_l1', 
        metric='mae'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], 
        eval_metric='mae',
        callbacks=[lgb.early_stopping(50, verbose=True)] # 早停
    )
    
    print("\n模型训练完成。")
    print(f"最佳迭代次数: {model.best_iteration_}")
    
    val_mae = 0.0
    if model.best_score_ and 'valid_0' in model.best_score_:
        val_mae = model.best_score_['valid_0'].get('l1', model.best_score_['valid_0'].get('mae'))
    print(f"验证集 MAE (Drift): {val_mae}") 
    
    print("正在使用全量数据重新训练最终模型...")
    final_model = lgb.LGBMRegressor(
        n_estimators=model.best_iteration_ if model.best_iteration_ else 500, 
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        random_state=42,
        objective='regression_l1'
    )
    final_model.fit(X, Y) 

    # 9. 保存模型
    final_model.booster_.save_model(MODEL_SAVE_PATH)
    print(f"最终模型训练完成，已保存到: {MODEL_SAVE_PATH}")
    print("\n特征重要性:")
    for f_name, f_importance in zip(FEATURE_NAMES, final_model.feature_importances_):
        print(f"{f_name}: {f_importance}")

# --- 关键的入口 ---
if __name__ == "__main__":
    try:
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        import scipy
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}")
        print("请运行: pip install pandas numpy lightgbm scipy scikit-learn pyarrow")
        sys.exit(1)
        
    main_train()