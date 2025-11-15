# pricing_engine.py
# 最终优化（方向七）：在线预测 "ML-Drift" (残差) 模型
# (*** 修正：调用 utils.py 中正确的函数名 ***)

from cfe_fin_math_api import CfeFinMathApi, InstrumentStaticData, MarketData, SamplePrediction
import utils
import numpy as np
import lightgbm as lgb

# --- 全局常量 ---
FUT_CONTRACT = "IF2507"
OPT_PREFIX = "IO2507"

# (***) 这是我们的9个特征。这个列表必须与 train_drift_model.py 中的列表完全一致
FEATURE_NAMES = [
    'S_diff', 
    'S_diff_sq', 
    'T', 
    'K', 
    'log_moneyness', 
    'r', 
    'option_type', 
    'anchor_volume', 
    'anchor_open_interest'
]
MODEL_DRIFT_PATH = "model_drift.lgb" # 确保此文件与 main.py 在同一目录

class PricingEngine:
    """
    期权定价引擎 (最终优化版 - 方向七：ML-Drift 残差模型)
    """
    def __init__(self, api: CfeFinMathApi):
        print("正在初始化定价引擎 (最终优化版 - ML-Drift法)...")
        
        self.opt_static_list: list[InstrumentStaticData] = api.get_option_static_md()
        self.fut_static_list: list[InstrumentStaticData] = api.get_future_static_md()
        
        self.trading_day: str = api.date
        self.r: float = 0.03  # 利率 'r' 仍然是一个有用的特征

        # 方向七：加载预先训练好的 ML 漂移模型
        try:
            self.model_drift = lgb.Booster(model_file=MODEL_DRIFT_PATH)
            self.expected_features = self.model_drift.num_feature()
            print(f"成功加载预训练的漂移模型 '{MODEL_DRIFT_PATH}'。")
            
            if self.expected_features != len(FEATURE_NAMES):
                print(f"*** 严重警告：特征数量不匹配！***")
                print(f"模型 '{MODEL_DRIFT_PATH}' 期望 {self.expected_features} 个特征,")
                print(f"但 pricing_engine.py 中定义了 {len(FEATURE_NAMES)} 个特征。")
            else:
                 print(f"模型期望 {self.expected_features} 个特征, 与定义相符。")

        except lgb.basic.LightGBMError:
            print(f"错误：无法加载模型 '{MODEL_DRIFT_PATH}'。")
            print("请确保您已经运行了离线训练脚本 (train_drift_model.py)。")
            raise
            
        self.contract_cache = {}
        
        for s in self.opt_static_list:
            if not s.instrument_id.startswith(OPT_PREFIX):
                continue
            
            try:
                parts = s.instrument_id.split('-')
                option_type = parts[1] # 'C' or 'P'
                strike_price = float(parts[2]) # K
                
                self.contract_cache[s.instrument_id] = {
                    "K": strike_price,
                    "expire_day": s.expire_day,
                    "option_type": 1 if option_type == 'C' else 0, # 0/1 特征
                    
                    "price_anchor": 0.0,
                    "S_anchor": 0.0,
                    
                    "anchor_volume": 0,
                    "anchor_open_interest": 0
                }
            except Exception as e:
                print(f"解析合约 {s.instrument_id} 静态数据失败: {e}")

        self.fut_static_info = next(
            (s for s in self.fut_static_list if s.instrument_id == FUT_CONTRACT), None
        )
        if self.fut_static_info is None:
            raise RuntimeError(f"未能在静态数据中找到标的期货 {FUT_CONTRACT}")
            
        print(f"引擎初始化完毕。共加载 {len(self.contract_cache)} 份 {OPT_PREFIX} 期权合约。")

    def _get_future_price(self, future_md: MarketData) -> float | None:
        try:
            fut_idx = future_md.instrument_id.index(FUT_CONTRACT)
        except ValueError:
            return None
        F_mid = utils.get_mid_price(
            future_md.bid_p1[fut_idx], future_md.bid_v1[fut_idx], 
            future_md.ask_p1[fut_idx], future_md.ask_v1[fut_idx]
        )
        return F_mid if F_mid is not None else future_md.last_price[fut_idx]

    def calibrate(self, option_md: MarketData, future_md: MarketData):
        """
        校准阶段 (每10分钟): 仅更新锚点值
        """
        print(f"--- 正在 {option_md.update_sec} 更新市场锚点 ---")
        
        F_anchor = self._get_future_price(future_md)
        if F_anchor is None:
            print("锚点更新失败：无法获取标的期货价格。")
            return

        for i, opt_id in enumerate(option_md.instrument_id):
            cache_item = self.contract_cache.get(opt_id)
            if not cache_item: continue 

            market_price = utils.get_mid_price(
                option_md.bid_p1[i], option_md.bid_v1[i], 
                option_md.ask_p1[i], option_md.ask_v1[i]
            )
            
            if market_price is not None:
                cache_item["price_anchor"] = market_price
                cache_item["S_anchor"] = F_anchor
                cache_item["anchor_volume"] = option_md.volume[i]
                cache_item["anchor_open_interest"] = option_md.open_interest[i]


    def predict(self, future_md: MarketData, predict: SamplePrediction):
        """
        预测阶段 (每500毫秒): 使用预训练的 ML-Drift 模型
        """
        
        # 1. 获取当前市场快照
        F_new = self._get_future_price(future_md)
        current_dt = utils.parse_date_time(
            self.trading_day, future_md.update_sec, future_md.update_msec
        )
        
        # 2. 如果主要数据缺失，无法预测
        if F_new is None or F_new <= 0:
            # 如果我们无法获取标的价格，只能重用锚点价格
            # (更好的做法可能是返回一个全0或全锚点价的列表)
            final_prices = []
            for s in self.opt_static_list:
                cache_item = self.contract_cache.get(s.instrument_id)
                if cache_item:
                    final_prices.append(cache_item["price_anchor"])
                else:
                    final_prices.append(0.0) # 理论上不应发生
            predict.target = final_prices
            return

        # 3. 准备特征矩阵
        # 我们需要按 self.opt_static_list 的顺序构建预测
        feature_list = []
        price_anchor_list = []
        
        for s in self.opt_static_list:
            cache_item = self.contract_cache.get(s.instrument_id)
            
            # 如果合约不在我们的缓存中 (例如，不是 IO2507)
            if not cache_item:
                feature_list.append([0.0] * len(FEATURE_NAMES)) # 填充0
                price_anchor_list.append(0.0)
                continue

            # 从缓存中提取锚点和静态数据
            S_anchor = cache_item['S_anchor']
            K = cache_item['K']
            option_type = cache_item['option_type']
            expire_day = cache_item['expire_day']
            anchor_volume = cache_item['anchor_volume']
            anchor_open_interest = cache_item['anchor_open_interest']
            
            # 存储锚点价格，用于后续计算
            price_anchor_list.append(cache_item['price_anchor'])

            # 4. 严格按照训练顺序构建特征
            # (*** 这必须与 train_drift_model.py 完全匹配 ***)
            
            # 1. S_diff
            S_diff = F_new - S_anchor
            # 2. S_diff_sq
            S_diff_sq = S_diff ** 2
            # 3. T
            T = utils.calculate_T(current_dt, expire_day)
            # 4. K
            # (K 变量已在上面获取)
            # 5. log_moneyness
            # (*** 修正：确保使用 utils.py 中的正确函数 ***)
            log_moneyness = utils.get_log_moneyness(F_new, K, T)
            # 6. r
            r = self.r
            # 7. option_type
            # (option_type 变量已在上面获取)
            # 8. anchor_volume
            # (anchor_volume 变量已在上面获取)
            # 9. anchor_open_interest
            # (anchor_open_interest 变量已在上面获取)
            
            current_features = [
                S_diff,
                S_diff_sq,
                T,
                K,
                log_moneyness,
                r,
                option_type,
                anchor_volume,
                anchor_open_interest
            ]
            feature_list.append(current_features)

        # 5. 批量预测
        # 转换为 numpy 数组以进行批量推理
        X_pred = np.array(feature_list)
        
        # 填充 NaN/Inf (安全起见，尽管训练时已处理)
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        predicted_drifts = self.model_drift.predict(X_pred)
        
        # 6. 计算最终理论价格
        # TheoPrice = AnchorPrice + PredictedDrift
        final_prices = np.array(price_anchor_list) + predicted_drifts
        
        # 确保价格不为负
        final_prices[final_prices < 0] = 0.0

        # 7. 提交结果
        predict.target = final_prices.tolist()