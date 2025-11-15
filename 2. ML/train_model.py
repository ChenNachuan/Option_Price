# train_model.py
# 适用于计划二：直接 ML 定价
# (*** 修正：TypeError，使用 pd.merge_asof 替换 pd.merge ***)

import os
import sys
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, time
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import train_test_split #<-- “计划二”使用随机分割

# --- 全局常量 ---
EPSILON = 1e-9

# --- 配置 ---
DATA_ROOT_DIR = '../data' 
# (***) 加载所有月份
OPTION_MONTH_PREFIXES = ['IO2507', 'IO2508', 'IO2509', 'IO2512', 'IO2603', 'IO2606']
FUTURE_PREFIXES = ['IF2507', 'IF2508', 'IF2509', 'IF2512']

MODEL_SAVE_PATH = 'model.lgb'
RISK_FREE_RATE = 0.0165 

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

        # 2. 加载所有期权行情
        for prefix in OPTION_MONTH_PREFIXES:
            opt_file_glob = glob.glob(os.path.join(day_path, f"{prefix}*_{date}.csv"))
            for opt_file in opt_file_glob:
                try:
                    all_option_md.append(pd.read_csv(opt_file, engine='pyarrow'))
                except Exception as e:
                    print(f"读取 {opt_file} 失败: {e}")

        # 3. 加载静态数据
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

def calculate_time_to_expiry_local(current_dt, expiry_dt):
    if pd.isna(current_dt) or pd.isna(expiry_dt):
        return np.nan
    time_delta = expiry_dt - current_dt
    total_seconds = time_delta.total_seconds()
    if total_seconds <= 0:
        return 0.0
    return total_seconds / (365.25 * 24 * 60 * 60)

def main_train():
    """
    主训练函数 (Plan 2)
    """
    print("开始离线训练 (计划二：直接价格预测)...")
    
    loaded_data = load_all_data(DATA_ROOT_DIR)
    if loaded_data is None: return 
    fut_df, opt_df, static_data_raw = loaded_data
    print(f"总共加载: {len(fut_df)} 条期货行情, {len(opt_df)} 条期权行情")

    print("正在转换静态数据...")
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
                is_call = 1 if data.get('option_type') == 'C' else (0 if data.get('option_type') == 'P' else -1)
                if is_call != -1 and 'strike_price' in data and pd.notna(data['strike_price']):
                    static_info_processed[inst_id] = {
                        'expiry': expiry_dt,
                        'K': float(data['strike_price']),
                        'is_call': is_call,
                    }
        except Exception: continue
            
    for inst_id, data in static_info_processed.items():
        month_str = inst_id[2:6] 
        static_info_processed[inst_id]['future_id'] = fut_month_to_id_map.get(month_str)

    static_df = pd.DataFrame.from_dict(static_info_processed, orient='index')
    static_df.index.name = 'instrument_id'

    print("正在处理时间戳...")
    fut_df = create_datetime_index(fut_df)
    opt_df = create_datetime_index(opt_df)
    if fut_df.empty or opt_df.empty: return

    fut_df_pivot = fut_df.pivot(index='datetime', columns='instrument_id', values='last_price')
    fut_df_pivot = fut_df_pivot.sort_index().ffill()
    opt_df = opt_df.sort_values('datetime')

    print("正在合并期权与期货数据...")
    
    # --- (!!!) 这是被修正的行 (!!!) ---
    # 使用 pd.merge_asof 替换 pd.merge
    data = pd.merge_asof(
        opt_df, 
        fut_df_pivot, 
        on='datetime', 
        direction='backward'
    )
    # --- (修正结束) ---
    
    print("正在构建特征和目标 (Price)...")
    data = pd.merge(data, static_df, left_on='instrument_id', right_index=True, how='left')
    
    data['midprice'] = (data['bid_p1'] + data['ask_p1']) / 2
    data['midprice'] = data['midprice'].fillna(data['ask_p1']).fillna(data['bid_p1'])
    
    data = data.dropna(subset=['midprice', 'future_id', 'K', 'expiry'])
    data = data[data['midprice'] > 0]

    data['F_price'] = data.apply(lambda row: row[row['future_id']], axis=1)
    data = data.dropna(subset=['F_price'])
    data = data[data['F_price'] > 0]
    
    data['T'] = data.apply(lambda row: calculate_time_to_expiry_local(row['datetime'], row['expiry']), axis=1)
    data = data[data['T'] > 0]
    
    feature_names = [
        'F_price', 'K', 'T', 'is_call', 'volume', 'open_interest',
    ]
    data['r'] = RISK_FREE_RATE 
    feature_names.append('r')

    final_data = data.dropna(subset=feature_names + ['midprice', 'trading_day'])
    
    X = final_data[feature_names]
    y = final_data['midprice']
    
    if X.empty:
        print("错误：没有构建出任何有效的训练数据。")
        return

    print(f"训练集构建完毕，特征数量: {len(X)}")

    # --- (!!!) 警告：这是随机分割，会导致过拟合 (!!!) ---
    # --- (但我们先修复 bug，再解决过拟合) ---
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
    
    val_mae = 0.0
    if model.best_score_ and 'valid_0' in model.best_score_:
        if 'l1' in model.best_score_['valid_0']:
            val_mae = model.best_score_['valid_0']['l1']
        elif 'mae' in model.best_score_['valid_0']:
            val_mae = model.best_score_['valid_0']['mae']
    print(f"验证集 MAE (Price): {val_mae}") 
    
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
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"错误: 缺少必要的库: {e}")
        print("请运行: pip install pandas numpy lightgbm scipy scikit-learn pyarrow")
        sys.exit(1)
        
    main_train()