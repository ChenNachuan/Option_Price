# train_model.py
# 适用于计划三：混合波动率预测 (ML 预测 IV)
# (修正：KeyError 'mae' -> 'l1')

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

# 导入官方 API 数据类
try:
    from cfe_fin_math_api import InstrumentStaticData, InstrumentType, OptionType
except ImportError:
    print("错误: 找不到 cfe_fin_math_api.py。")
    sys.exit(1)

# 定义一个极小值，用于防止除零错误
EPSILON = 1e-9

# --- 配置 ---
DATA_ROOT_DIR = '../data' 
OPTION_MONTH_PREFIXES = ['IO2507', 'IO2508', 'IO2509', 'IO2512', 'IO2603', 'IO2606']
FUTURE_PREFIXES = ['IF2507', 'IF2508', 'IF2509', 'IF2512']

MODEL_SAVE_PATH = 'model.lgb'
RISK_FREE_RATE = 0.0165 

# (load_all_data 和 create_datetime_index 函数保持不变)
def load_all_data(data_root: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    加载所有15天的数据并合并。
    """
    all_future_md = []
    all_option_md = []
    all_future_static = []
    all_option_static = []

    if not os.path.exists(data_root):
        print(f"错误: 找不到数据目录 '{data_root}'。")
        print(f"请确保你的15天数据存放在: {os.path.abspath(data_root)}")
        return None

    date_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith('2025')]
    
    if not date_folders:
        print(f"错误: 在 '{data_root}' 中没有找到任何日期文件夹 (例如 '20250623')。")
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

        for prefix in OPTION_MONTH_PREFIXES:
            opt_file_glob = glob.glob(os.path.join(day_path, f"{prefix}*_{date}.csv"))
            for opt_file in opt_file_glob:
                try:
                    all_option_md.append(pd.read_csv(opt_file))
                except Exception as e:
                    print(f"读取 {opt_file} 失败: {e}")

        fut_static_file = os.path.join(day_path, f"IF_{date}_static.csv")
        if os.path.exists(fut_static_file):
            try:
                all_future_static.append(pd.read_csv(fut_static_file))
            except Exception: pass

        for prefix in OPTION_MONTH_PREFIXES:
            opt_static_file_glob = glob.glob(os.path.join(day_path, f"{prefix}*_{date}_static.csv"))
            for opt_static_file in opt_static_file_glob:
                try:
                    all_option_static.append(pd.read_csv(opt_static_file))
                except Exception: pass

    if not all_future_md or not all_option_md:
        print("错误: 数据加载不完整，期货或期权行情列表为空。")
        return None

    opt_static_concat = pd.concat(all_option_static).drop_duplicates(subset=['instrument_id']).reset_index(drop=True)
    fut_static_concat = pd.concat(all_future_static).drop_duplicates(subset=['instrument_id']).reset_index(drop=True)

    return (pd.concat(all_future_md), pd.concat(all_option_md),
            fut_static_concat, opt_static_concat)

def create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        time_str = df['trading_day'].astype(str) + " " + df['update_sec']
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
    主训练函数 (Plan 3 改进版)
    """
    print("开始离线训练 (计划三：IV 预测器)...")
    
    loaded_data = load_all_data(DATA_ROOT_DIR)
    if loaded_data is None: return 
    fut_df, opt_df, fut_static_df, opt_static_df = loaded_data
    print(f"总共加载: {len(fut_df)} 条期货行情, {len(opt_df)} 条期权行情")

    print("正在转换静态数据 (手动添加 instrument_type)...")
    valid_static_keys = {f.name for f in InstrumentStaticData.__dataclass_fields__.values()}
    opt_static_list = []
    for row in opt_static_df.to_dict('records'):
        row['instrument_type'] = InstrumentType.OPTION
        if row.get('option_type') == 'C': row['option_type'] = OptionType.CALL
        elif row.get('option_type') == 'P': row['option_type'] = OptionType.PUT
        else: row['option_type'] = None
        row['strike_price'] = float(row['strike_price']) if pd.notna(row.get('strike_price')) else None
        cleaned_row = {k: v for k, v in row.items() if k in valid_static_keys and pd.notna(v)}
        try:
            opt_static_list.append(InstrumentStaticData(**cleaned_row))
        except Exception: pass

    fut_static_list = []
    for row in fut_static_df.to_dict('records'):
        row['instrument_type'] = InstrumentType.FUTURE
        row['option_type'] = None
        row['strike_price'] = None
        cleaned_row = {k: v for k, v in row.items() if k in valid_static_keys and pd.notna(v)}
        try:
            fut_static_list.append(InstrumentStaticData(**cleaned_row))
        except Exception: pass
            
    static_map = feat_utils.get_static_data_map(opt_static_list, fut_static_list)
    static_df = pd.DataFrame.from_dict(static_map, orient='index')

    print("正在处理时间戳...")
    fut_df = create_datetime_index(fut_df)
    opt_df = create_datetime_index(opt_df)
    if fut_df.empty or opt_df.empty: return

    fut_df_pivot = fut_df.pivot(index='datetime', columns='instrument_id', values='last_price')
    fut_df_pivot = fut_df_pivot.sort_index().ffill()
    opt_df = opt_df.sort_values('datetime')

    print("正在合并期权与期货数据...")
    merged_df = pd.merge_asof(opt_df, fut_df_pivot, on='datetime', direction='backward')
    
    print("正在构建特征和目标 (IV)...")
    data = merged_df.join(static_df, on='instrument_id')
    data['midprice'] = (data['bid_p1'] + data['ask_p1']) / 2
    data['midprice'] = data['midprice'].fillna(data['ask_p1']).fillna(data['bid_p1'])
    
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

    data = vectorized_iv_approximation(data, RISK_FREE_RATE)
    
    final_data = data.dropna(subset=feature_names + ['IV'])
    
    X = final_data[feature_names]
    y = final_data['IV']
    
    if X.empty:
        print("错误：没有构建出任何有效的训练数据。")
        return

    print(f"训练集构建完毕，特征数量: {len(X)}")

    print("正在将数据 80/20 分割为训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
    
    # --- (!!!) 这是被修正的行 (!!!) ---
    # 错误日志显示 metric 被命名为 'l1'
    print(f"验证集 MAE: {model.best_score_['valid_0']['l1']}")
    # --- (修正结束) ---

    
    print("正在使用全量数据重新训练最终模型...")
    final_model = lgb.LGBMRegressor(
        n_estimators=model.best_iteration_ or 500, # 使用早停找到的最佳次数
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
        print("请运行: pip install pandas numpy lightgbm scipy scikit-learn")
        sys.exit(1)
        
    main_train()