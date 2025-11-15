# train_model.py
# 适用于计划三：混合波动率预测 (ML 预测 IV)
# (最终修正版：使用 Plan 2 的数据加载逻辑 + Plan 3 的训练目标 + MAE 优化)

import os
import sys
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, time
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split 

# 导入我们自己的库
import feature_utils as feat_utils
import financial_utils as fin_utils
import greeks as greeks

# 从 scipy.stats 导入 norm
from scipy.stats import norm

# --- 全局常量 ---
EPSILON = 1e-9 # 定义一个极小值

# --- 配置 ---
DATA_ROOT_DIR = '../data' 
# (***) 修改：加载所有月份
OPTION_MONTH_PREFIXES = ['IO2507', 'IO2508', 'IO2509', 'IO2512', 'IO2603', 'IO2606']
FUTURE_PREFIXES = ['IF2507', 'IF2508', 'IF2509', 'IF2512']

MODEL_SAVE_PATH = 'model.lgb'
RISK_FREE_RATE = 0.0165 # (1.65%) 

def load_all_data(data_root: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    """
    加载所有15天的数据并合并。(采纳自 Plan 2)
    """
    all_future_md = []
    all_option_md = []
    all_static_data = {} 

    if not os.path.exists(data_root):
        print(f"错误: 找不到数据目录 '{data_root}'。")
        print(f"请确保你的15天数据存放在: {os.path.abspath(data_root)}")
        return None

    date_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith('2025')]
    print(f"找到 {len(date_folders)} 天的数据...")

    for date in date_folders:
        day_path = os.path.join(data_root, date)
        
        # 1. 加载期货行情
        fut_file = os.path.join(day_path, f"IF_{date}.csv")
        if os.path.exists(fut_file):
            try:
                all_future_md.append(pd.read_csv(fut_file, engine='pyarrow'))
            except Exception as e:
                print(f"读取 {fut_file} 失败: {e}")

        # 2. 加载所有期权行情 (*** 已修改为加载所有月份 ***)
        for prefix in OPTION_MONTH_PREFIXES:
            opt_file_glob = glob.glob(os.path.join(day_path, f"{prefix}*_{date}.csv"))
            for opt_file in opt_file_glob:
                try:
                    all_option_md.append(pd.read_csv(opt_file, engine='pyarrow'))
                except Exception as e:
                    print(f"读取 {opt_file} 失败: {e}")

        # 3. 加载静态数据 (*** 已修改为加载所有月份 ***)
        fut_static_file = os.path.join(day_path, f"IF_{date}_static.csv")
        if os.path.exists(fut_static_file):
            try:
                df = pd.read_csv(fut_static_file, engine='pyarrow')
                for _, row in df.iterrows():
                    all_static_data[row['instrument_id']] = row.to_dict()
            except Exception: pass

        for prefix in OPTION_MONTH_PREFIXES:
            opt_static_file_glob = glob.glob(os.path.join(day_path, f"{prefix}*_{date}_static.csv"))
            for opt_static_file in opt_static_file_glob:
                try:
                    df = pd.read_csv(opt_static_file, engine='pyarrow')
                    for _, row in df.iterrows():
                        all_static_data[row['instrument_id']] = row.to_dict()
                except Exception: pass
                
    if not all_future_md or not all_option_md:
        print("错误: 数据加载不完整，期货或期权行情列表为空。")
        return None

    return pd.concat(all_future_md), pd.concat(all_option_md), all_static_data

def create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 trading_day, update_sec, update_msec 合并为 datetime 对象
    """
    try:
        # 修正 TypeError: can only concatenate str (not "datetime.time")
        time_str = df['trading_day'].astype(str) + " " + df['update_sec'].astype(str)
        df['datetime'] = pd.to_datetime(time_str, format='%Y%m%d %H:%M:%S')
        df['datetime'] = df['datetime'] + pd.to_timedelta(df['update_msec'], unit='ms')
        return df
    except Exception as e:
        print(f"创建时间戳时出错: {e}")
        return pd.DataFrame() # 返回空
        
# (内部) 矢量化 bsm_price
def v_bsm_price(S, K, T, r, q, sigma, option_type):
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
        
        price = np.where(option_type == 'call', call_price, put_price)
        price[T < EPSILON] = np.nan
        return price

def vectorized_iv_approximation(data: pd.DataFrame, r: float) -> pd.DataFrame:
    """
    使用一步牛顿法 (One-Step Newton-Raphson) 矢量化地近似 IV。
    """
    print("开始矢量化近似 IV...")
    
    S = data['F_price'].values
    K = data['K'].values
    T = data['T'].values
    market_price = data['midprice'].values
    option_type = data['type'].values
    
    sigma_guess = np.full_like(S, 0.30) 
    r_vec = np.full_like(S, r)
    
    price_at_guess = v_bsm_price(S, K, T, r_vec, r_vec, sigma_guess, option_type)
    vega_at_guess = greeks.vectorized_calculate_vega(S, K, T, r, r, sigma_guess)

    price_error = price_at_guess - market_price
    vega_at_guess[vega_at_guess == 0] = EPSILON
    
    iv_approx = sigma_guess - (price_error / (vega_at_guess * 100)) 
    
    iv_approx[~np.isfinite(iv_approx)] = np.nan 
    iv_approx[iv_approx < 0.01] = np.nan
    iv_approx[iv_approx > 1.5] = np.nan
    
    print("矢量化 IV 近似完成。")
    data['IV'] = iv_approx
    return data


def main_train():
    """
    主训练函数 (Plan 3 最终版)
    """
    print("开始离线训练 (计划三：IV 预测器)...")
    
    # 1. 加载所有数据 (使用 Plan 2 的方法)
    loaded_data = load_all_data(DATA_ROOT_DIR)
    if loaded_data is None: return 
    fut_df, opt_df, static_data_raw = loaded_data
    print(f"总共加载: {len(fut_df)} 条期货行情, {len(opt_df)} 条期权行情")

    # 2. 准备静态数据 (使用 Plan 2 的方法, 修复 strptime bug)
    print("正在转换静态数据...")
    static_info_processed = {}
    fut_month_to_id_map = {}

    for inst_id, data in static_data_raw.items():
        try:
            # (***) 修正 strptime TypeError (***)
            expire_day_str = str(data['expire_day']).split('.')[0] 
            expiry_dt = datetime.strptime(expire_day_str, '%Y%m%d').replace(hour=15, minute=0)
            
            if 'IF' in inst_id:
                month_str = inst_id[2:6] 
                fut_month_to_id_map[month_str] = inst_id
            elif 'IO' in inst_id:
                is_call = 1 if data.get('option_type') == 'C' else (0 if data.get('option_type') == 'P' else -1)
                if is_call != -1 and 'strike_price' in data and pd.notna(data['strike_price']):
                    static_info_processed[inst_id] = {
                        'expiry': expiry_dt,
                        'K': float(data['strike_price']),
                        'is_call': is_call,
                        'type': 'call' if is_call == 1 else 'put'
                    }
        except Exception:
            continue
            
    for inst_id, data in static_info_processed.items():
        month_str = inst_id[2:6] 
        static_info_processed[inst_id]['future_id'] = fut_month_to_id_map.get(month_str)

    # *** 修正：将字典转换为 DataFrame，并确保索引名为 'instrument_id' ***
    static_df = pd.DataFrame.from_dict(static_info_processed, orient='index')
    static_df.index.name = 'instrument_id'


    # 3. 数据预处理和对齐
    print("正在处理时间戳...")
    fut_df = create_datetime_index(fut_df)
    opt_df = create_datetime_index(opt_df)
    if fut_df.empty or opt_df.empty: return

    fut_df_pivot = fut_df.pivot(index='datetime', columns='instrument_id', values='last_price')
    fut_df_pivot = fut_df_pivot.sort_index().ffill()
    opt_df = opt_df.sort_values('datetime')

    print("正在合并期权与期货数据...")
    # *** 修正：使用 pd.merge 替换 .join，修复 KeyError ***
    data = pd.merge(merged_df, static_df, left_on='instrument_id', right_index=True, how='left')
    
    print("正在构建特征和目标 (IV)...")
    data['midprice'] = (data['bid_p1'] + data['ask_p1']) / 2
    data['midprice'] = data['midprice'].fillna(data['ask_p1']).fillna(data['bid_p1'])
    
    # 这里的 dropna 现在可以正确工作了
    data = data.dropna(subset=['midprice', 'future_id', 'K', 'expiry', 'type'])
    data = data[data['midprice'] > 0]

    data['F_price'] = data.apply(lambda row: row[row['future_id']], axis=1)
    data = data.dropna(subset=['F_price'])
    data = data[data['F_price'] > 0]
    
    data['T'] = data.apply(lambda row: feat_utils.calculate_time_to_expiry(row['datetime'], row['expiry']), axis=1)
    data = data[data['T'] > 0]
    
    data['log_moneyness'] = data.apply(lambda row: feat_utils.calculate_log_moneyness(row['K'], row['F_price']), axis=1)
    
    feature_names = [
        'log_moneyness', 'T', 'volume', 'open_interest', 'is_call',
    ]
    data['r'] = RISK_FREE_RATE 
    feature_names.append('r')

    # 4.6 *** 构建目标 (y = IV) - 使用矢量化近似 ***
    data = vectorized_iv_approximation(data, RISK_FREE_RATE)
    
    # 4.7 最终清洗
    final_data = data.dropna(subset=feature_names + ['IV'])
    
    X = final_data[feature_names]
    y = final_data['IV']
    
    if X.empty:
        print("错误：没有构建出任何有效的训练数据。")
        return

    print(f"训练集构建完毕，特征数量: {len(X)}")

    # 5. 改进的模型训练 (按时间分割)
    print("正在按时间分割训练集和验证集...")
    
    unique_days = final_data['trading_day'].unique()
    unique_days.sort()
    
    if len(unique_days) < 2:
        print("错误：数据量太少，无法进行时间序列分割（至少需要2天）。")
        return
        
    split_point = int(len(unique_days) * 0.8) # 80/20 分割
    train_days = unique_days[:split_point]
    val_days = unique_days[split_point:]

    print(f"训练日: {len(train_days)} 天, 验证日: {len(val_days)} 天")

    train_idx = final_data['trading_day'].isin(train_days)
    val_idx = final_data['trading_day'].isin(val_days)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    if X_train.empty or X_val.empty:
        print("错误：分割数据集失败，训练集或验证集为空。")
        return

    print(f"训练集: {len(X_train)}， 验证集: {len(X_val)}")
    print("正在训练 LightGBM 模型 (优化目标: MAE)...")

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        random_state=42,
        objective='mae', 
        metric='mae'      
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=True)] 
    )
    
    print("\n模型训练完成。")
    print(f"最佳迭代次数: {model.best_iteration_}")
    
    val_mae = 0.0
    if model.best_score_ and 'valid_0' in model.best_score_:
        # (*** 修正 KeyError ***)
        if 'l1' in model.best_score_['valid_0']:
            val_mae = model.best_score_['valid_0']['l1']
        elif 'mae' in model.best_score_['valid_0']:
            val_mae = model.best_score_['valid_0']['mae']
    print(f"验证集 MAE (IV): {val_mae}")
    
    print("正在使用全量数据重新训练最终模型...")
    final_model = lgb.LGBMRegressor(
        n_estimators=model.best_iteration_ or 500, 
        learning_rate=0.05,
        num_leaves=63,
        n_jobs=-1,
        random_state=42,
        objective='mae'
    )
    final_model.fit(X, y) 

    # 7. 保存模型
    final_model.booster_.save_model(MODEL_SAVE_PATH)
    print(f"最终模型训练完成，已保存到: {MODEL_SAVE_PATH}")

# --- 关键的入口 ---
if __name__ == "__main__":
    try:
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        import scipy
        from scipy.stats import norm 
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}")
        print("请运行: pip install pandas numpy lightgbm scipy scikit-learn pyarrow")
        sys.exit(1)
        
    main_train()