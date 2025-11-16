# train_residual_model.py
# (方案二：训练残差模型)
# (!!!) 警告：此脚本需要大量计算资源 (!!!)
# (它会在 15 天数据中的每个 10 分钟锚点上拟合 SVI/Spline)

import os
import sys
import glob
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, time
from typing import Dict, Any, List, Tuple, Optional

# --- 导入你项目中的模块 ---
import utils
import financial_models
import volatility_surface

# --- 全局常量 ---
EPSILON = 1e-9 
DATA_ROOT_DIR = '../data' # 确保这指向您数据文件夹的父目录
MODEL_SAVE_PATH = 'model_residual.lgb'
RISK_FREE_RATE = 0.03 # (必须与 pricing_engine.py 一致)
FUT_CONTRACT = "IF2507"
OPT_PREFIX = "IO2507"

LOG_MONEYNESS_FILTER_LOW = -0.3
LOG_MONEYNESS_FILTER_HIGH = 0.3

# --- 关键：残差模型的特征 ---
# 我们给模型所有它需要的信息来预测基线模型的 *错误*
FEATURE_NAMES = [
    # 1. 状态 (合约在哪里?)
    'log_moneyness', 
    'T', 
    'is_call', 
    'r', # r 也可以保留
    # 2. 修正目标 (时间过去了多久?)
    'time_since_anchor' 
]


# --- 1. 数据加载 (与你其他训练脚本类似) ---

def load_all_data(data_root: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]]:
    all_future_md, all_option_md, all_static_data = [], [], {}
    if not os.path.exists(data_root):
        print(f"错误: 找不到数据目录 '{data_root}'。")
        return None
    date_folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith('2025')]
    print(f"找到 {len(date_folders)} 天的数据...")

    for date in date_folders:
        day_path = os.path.join(data_root, date)
        fut_file = os.path.join(day_path, f"IF_{date}.csv")
        if os.path.exists(fut_file):
            all_future_md.append(pd.read_csv(fut_file, engine='pyarrow'))
            
        opt_files = glob.glob(os.path.join(day_path, f"{OPT_PREFIX}*_{date}.csv"))
        for opt_file in opt_files:
            all_option_md.append(pd.read_csv(opt_file, engine='pyarrow'))
            
        static_files = glob.glob(os.path.join(day_path, f"*{OPT_PREFIX}*_{date}_static.csv"))
        for static_file in static_files:
            df = pd.read_csv(static_file, engine='pyarrow')
            for _, row in df.iterrows(): all_static_data[row['instrument_id']] = row.to_dict()
        
        # (也加载期货静态数据)
        fut_static_file = os.path.join(day_path, f"IF_{date}_static.csv")
        if os.path.exists(fut_static_file):
            df = pd.read_csv(fut_static_file, engine='pyarrow')
            for _, row in df.iterrows(): all_static_data[row['instrument_id']] = row.to_dict()

    if not all_future_md or not all_option_md:
        print("错误: 数据加载不完整。")
        return None
    return pd.concat(all_future_md), pd.concat(all_option_md), all_static_data

def create_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        time_str = df['trading_day'].astype(str) + " " + df['update_sec'].astype(str)
        df['datetime'] = pd.to_datetime(time_str, format='%Y%m%d %H:%M:%S')
        df['datetime'] = df['datetime'] + pd.to_timedelta(df['update_msec'], unit='ms')
        return df
    except Exception as e: return pd.DataFrame()

def calculate_mid_price_local(df: pd.DataFrame) -> pd.Series:
    bid_p, ask_p = df['bid_p1'], df['ask_p1']
    bid_v, ask_v = df.get('bid_v1', 1), df.get('ask_v1', 1)
    cond1 = (bid_v > 0) & (ask_v > 0) & (ask_p > bid_p); price1 = (bid_p + ask_p) / 2.0
    cond2 = (bid_v > 0) & (ask_v == 0); price2 = bid_p
    cond3 = (bid_v == 0) & (ask_v > 0); price3 = ask_p
    mid_price = np.select([cond1, cond2, cond3], [price1, price2, price3], default=np.nan)
    mid_price_series = pd.Series(mid_price, index=df.index).fillna(df['last_price'])
    return mid_price_series

# --- 2. 核心：离线复现 Calibrate 和 Predict 逻辑 ---

def calculate_greeks_at_anchor(anchor_batch: pd.DataFrame) -> pd.DataFrame:
    """
    (!! 关键函数 !!)
    在单个 10 分钟锚点上，复现 pricing_engine.calibrate 的全部逻辑
    """
    if anchor_batch.empty:
        return pd.DataFrame()

    # 1. 获取 F 和 T
    F_anchor = anchor_batch[anchor_batch['instrument_id'] == FUT_CONTRACT]['mid_price'].values[0]
    current_dt_anchor = anchor_batch.name
    
    if pd.isna(F_anchor) or F_anchor <= 0:
        return pd.DataFrame()

    # 2. 准备 IV 拟合数据
    moneyness_points, w_points, iv_points = [], [], []
    iv_data_list = []

    for _, row in anchor_batch.iterrows():
        if not row['instrument_id'].startswith(OPT_PREFIX):
            continue
            
        market_price = row['mid_price']
        if market_price is None or market_price <= 0.1:
            continue
            
        K = row['K']
        opt_type = 'C' if row['is_call'] == 1 else 'P'
        T = utils.calculate_T(current_dt_anchor, str(row['expire_day']))
        log_moneyness = utils.get_log_moneyness(F_anchor, K)
        
        if not (LOG_MONEYNESS_FILTER_LOW <= log_moneyness <= LOG_MONEYNESS_FILTER_HIGH):
            continue
            
        iv = financial_models.implied_volatility(market_price, F_anchor, K, T, RISK_FREE_RATE, opt_type)
        
        if iv is not np.nan and 0.05 < iv < 1.5:
            moneyness_points.append(log_moneyness)
            w_points.append(iv**2 * T)
            iv_points.append(iv)
        
        iv_data_list.append({'instrument_id': row['instrument_id'], 'T': T, 'log_moneyness': log_moneyness})

    # 3. 拟合 SVI 和 Spline
    svi_params, spline_model = None, None
    if len(moneyness_points) > 5:
        svi_params = volatility_surface.fit_svi_surface(moneyness_points, w_points)
        spline_model = volatility_surface.fit_spline_surface(moneyness_points, iv_points)
        
    if svi_params is None or spline_model is None:
        return pd.DataFrame() # 拟合失败，丢弃这个锚点

    # 4. 计算 Greeks
    results = []
    for data in iv_data_list:
        opt_id = data['instrument_id']
        static_info = anchor_batch[anchor_batch['instrument_id'] == opt_id].iloc[0]
        K = static_info['K']
        opt_type = 'C' if static_info['is_call'] == 1 else 'P'
        T = data['T']
        k = data['log_moneyness']
        
        if opt_type == 'P':
            sigma = volatility_surface.get_vol_from_svi_params(svi_params, k, T)
        else: # 'C'
            sigma = volatility_surface.get_vol_from_spline(spline_model, k)
            
        delta = financial_models.delta_black76(F_anchor, K, T, RISK_FREE_RATE, sigma, opt_type)
        gamma = financial_models.gamma_black76(F_anchor, K, T, RISK_FREE_RATE, sigma)
        
        results.append({
            'instrument_id': opt_id,
            'datetime_anchor': current_dt_anchor, # (!!!) 修正：重命名锚点时间 (!!!)
            'market_price_old': static_info['mid_price'],
            'S_old': F_anchor,
            'delta': delta,
            'gamma': gamma
        })

    return pd.DataFrame(results)

def main_train():
    print("开始离线训练 (方案二：残差模型)...")
    
    # 1. 加载数据
    loaded_data = load_all_data(DATA_ROOT_DIR)
    if loaded_data is None: return 
    fut_df, opt_df, static_data_raw = loaded_data
    print(f"总共加载: {len(fut_df)} 条期货行情, {len(opt_df)} 条期权行情")

    # 2. 处理静态数据
    print("正在处理静态数据...")
    static_info_processed = {}
    for inst_id, data in static_data_raw.items():
        try:
            expire_day = str(data['expire_day']).split('.')[0]
            if inst_id.startswith(OPT_PREFIX):
                static_info_processed[inst_id] = {
                    'expire_day': expire_day,
                    'K': float(data['strike_price']),
                    'is_call': 1 if data.get('option_type') == 'C' else 0,
                }
        except Exception: continue
    static_df = pd.DataFrame.from_dict(static_info_processed, orient='index')
    static_df.index.name = 'instrument_id'

    # 3. 处理时间戳和价格
    print("正在处理时间戳和中间价...")
    fut_df = create_datetime_index(fut_df)
    opt_df = create_datetime_index(opt_df)
    fut_df['mid_price'] = calculate_mid_price_local(fut_df)
    opt_df['mid_price'] = calculate_mid_price_local(opt_df)

    # 4. (!!!) 识别 10 分钟锚点
    print("正在识别 10 分钟锚点...")
    opt_df['min'] = opt_df['datetime'].dt.minute
    opt_df['sec'] = opt_df['datetime'].dt.second
    opt_df['msec'] = opt_df['datetime'].dt.microsecond / 1000
    # (假设 10min 快照在 0 秒 0 毫秒)
    is_anchor = (opt_df['min'] % 10 == 0) & (opt_df['sec'] == 0) & (opt_df['msec'] == 0)
    opt_df['is_anchor'] = is_anchor
    
    # (也标记期货锚点)
    fut_df['min'] = fut_df['datetime'].dt.minute
    fut_df['sec'] = fut_df['datetime'].dt.second
    fut_df['msec'] = fut_df['datetime'].dt.microsecond / 1000
    fut_df['is_anchor'] = (fut_df['min'] % 10 == 0) & (fut_df['sec'] == 0) & (fut_df['msec'] == 0)

    anchor_opt_df = opt_df[opt_df['is_anchor'] == True]
    anchor_fut_df = fut_df[fut_df['is_anchor'] == True]
    
    # (合并期权和期货的锚点数据，用于校准)
    anchor_all_df = pd.concat([
        anchor_opt_df, 
        anchor_fut_df[anchor_fut_df['instrument_id'] == FUT_CONTRACT]
    ])
    anchor_all_df = pd.merge(anchor_all_df, static_df, left_on='instrument_id', right_index=True, how='left')
    anchor_all_df = anchor_all_df.sort_values('datetime')
    
    # 5. (!!!) 离线计算所有锚点的 Greeks
    print("正在离线计算所有锚点的 Greeks (这可能需要几分钟)...")
    all_greeks_df = anchor_all_df.groupby('datetime').apply(calculate_greeks_at_anchor)
    all_greeks_df = all_greeks_df.reset_index(drop=True)
    if all_greeks_df.empty:
        print("错误：未能计算任何 Greeks。检查锚点匹配逻辑。")
        return
    print(f"Greeks 计算完成，共 {len(all_greeks_df)} 条记录。")

    # 6. 合并高频数据和锚点数据
    print("正在合并高频数据和锚点 Greeks...")
    fut_df_pivot = fut_df.pivot(index='datetime', columns='instrument_id', values='mid_price')
    fut_df_pivot = fut_df_pivot.sort_index().ffill()
    
    data = opt_df.sort_values('datetime')
    data = pd.merge(data, static_df, left_on='instrument_id', right_index=True, how='left')
    data = pd.merge_asof(data, fut_df_pivot[[FUT_CONTRACT]], on='datetime', direction='backward')
    data = data.rename(columns={FUT_CONTRACT: 'F_price'})
    
    # (!!!) 关键合并：将锚点 Greeks 合并到所有 500ms tick 上
    all_greeks_df = all_greeks_df.sort_values('datetime_anchor')
    data = pd.merge_asof(
        data, 
        all_greeks_df, 
        left_on='datetime',       # (!!!) 修正：使用 left_on (高频时间)
        right_on='datetime_anchor', # (!!!) 修正：使用 right_on (锚点时间)
        by='instrument_id', 
        direction='backward'
        # (!!!) 修正：不再需要 suffixes
    )
    
    # 7. (!!!) 计算 X 和 Y (残差)
    print("正在计算 Baseline, Residual(Y) 和 Features(X)...")
    data = data.dropna(subset=['F_price', 'K', 'expire_day', 'S_old', 'delta', 'mid_price'])
    data = data[data['F_price'] > 0]
    
    # X: 计算 S_diff
    data['S_diff'] = data['F_price'] - data['S_old']
    data['S_diff_sq'] = data['S_diff'] ** 2
    
    # Y: 计算 Baseline 和 Residual
    data['price_baseline'] = data['market_price_old'] + \
                             data['delta'] * data['S_diff'] + \
                             0.5 * data['gamma'] * data['S_diff_sq']
                             
    Y = data['mid_price'] - data['price_baseline']
    Y.name = 'Y_residual'
    data['Y_residual'] = Y 
    
    # X: 计算 T 和其他特征
    data['T'] = data.apply(lambda row: utils.calculate_T(row['datetime'], str(row['expire_day'])), axis=1)
    
    # (!!!) --- 修正：内联实现向量化 --- (!!!)
    F_series = data['F_price']
    K_series = data['K']
    result = pd.Series(np.nan, index=F_series.index) # 默认值为 NaN
    
    # 仅在 F > 0 且 K > 0 时计算
    valid_mask = (F_series > 0) & (K_series > 0)
    result[valid_mask] = np.log(K_series[valid_mask] / F_series[valid_mask])
    
    data['log_moneyness'] = result
    data['r'] = RISK_FREE_RATE
    data['time_since_anchor'] = (data['datetime'] - data['datetime_anchor']).dt.total_seconds()
    
    # 8. 准备训练
    final_data = data.dropna(subset=FEATURE_NAMES + [Y.name])
    X = final_data[FEATURE_NAMES]
    Y = Y.loc[final_data.index]
    
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    Y = Y.fillna(0)
    
    if X.empty:
        print("错误：没有构建出任何有效的训练数据。")
        return
        
    print(f"训练集构建完毕，特征数量: {len(X)}")

    # 9. 训练模型 (使用时间序列分割)
    print("正在按时间分割训练集和验证集 (后 20% 作为验证)...")
    split_index = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_index], Y.iloc[:split_index]
    X_val, y_val = X.iloc[split_index:], Y.iloc[split_index:]

    print(f"训练集: {len(X_train)}， 验证集: {len(X_val)}")
    print("正在训练 LightGBM 残差模型 (优化目标: MAE)...")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31, # (正则化)
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='regression_l1', # (MAE)
        metric='mae'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)], 
        eval_metric='mae',
        callbacks=[lgb.early_stopping(50, verbose=True)] 
    )
    
    print("\n模型训练完成。")
    print(f"最佳迭代次数: {model.best_iteration_}")
    
    val_mae = model.best_score_['valid_0'].get('l1', 0.0) if model.best_score_ else 0.0
    print(f"验证集 MAE (Residual): {val_mae}") 
    
    print("正在使用全量数据重新训练最终模型...")
    final_model = lgb.LGBMRegressor(
        n_estimators=model.best_iteration_ or 200, # 使用最佳迭代
        learning_rate=0.05,
        num_leaves=31,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        objective='regression_l1'
    )
    final_model.fit(X, Y) 

    final_model.booster_.save_model(MODEL_SAVE_PATH)
    print(f"最终残差模型训练完成，已保存到: {MODEL_SAVE_PATH}")
    
    print("\n特征重要性:")
    for f_name, f_importance in zip(FEATURE_NAMES, final_model.feature_importances_):
        print(f"{f_name}: {f_importance}")

# --- 入口 ---
if __name__ == "__main__":
    main_train()