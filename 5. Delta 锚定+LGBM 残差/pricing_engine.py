# pricing_engine.py (修改版 - 方案二)

from cfe_fin_math_api import CfeFinMathApi, InstrumentStaticData, MarketData, SamplePrediction
import utils
import financial_models
import volatility_surface
import numpy as np
import pandas as pd # (!!!) 新增导入
import lightgbm as lgb # (!!!) 新增导入
from datetime import datetime # (!!!) 新增导入

# ... (FUT_CONTRACT, OPT_PREFIX, 过滤器常量不变) ...
FUT_CONTRACT = "IF2507"
OPT_PREFIX = "IO2507"
LOG_MONEYNESS_FILTER_LOW = -0.3
LOG_MONEYNESS_FILTER_HIGH = 0.3
RESIDUAL_MODEL_PATH = 'model_residual.lgb'

class PricingEngine:
    def __init__(self, api: CfeFinMathApi):
        print("正在初始化定价引擎 (方案二：Delta 锚定 + ML 残差)...")
        
        self.opt_static_list: list[InstrumentStaticData] = api.get_option_static_md()
        self.fut_static_list: list[InstrumentStaticData] = api.get_future_static_md()
        
        self.trading_day: str = api.date
        self.r: float = 0.03  # (必须与训练时一致)

        self.svi_params = None    
        self.spline_model = None  
        
        # (!!!) 新增：加载残差模型
        try:
            self.model_residual = lgb.Booster(model_file=RESIDUAL_MODEL_PATH)
            self.model_features = self.model_residual.feature_name()
            print(f"残差模型 '{RESIDUAL_MODEL_PATH}' 加载成功。")
            print(f"需要特征: {self.model_features}")
        except Exception as e:
            print(f"*** 严重错误: 无法加载残差模型 '{RESIDUAL_MODEL_PATH}' ***: {e}")
            self.model_residual = None

        # (!!!) 新增：缓存校准时间
        self.last_calibrate_time = None 

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
                    "type": option_type,
                    "is_call": 1 if option_type == 'C' else 0, # (!!!) 新增
                    "T": 0.0,
                    "market_price_old": 0.0, 
                    "delta": 0.0,
                    "gamma": 0.0, 
                    "S_old": 0.0
                }
            except Exception as e:
                print(f"解析合约 {s.instrument_id} 静态数据失败: {e}")

        # ... (fut_static_info 加载不变) ...
        self.fut_static_info = next(
            (s for s in self.fut_static_list if s.instrument_id == FUT_CONTRACT), None
        )
        if self.fut_static_info is None:
            raise RuntimeError(f"未能在静态数据中找到标的期货 {FUT_CONTRACT}")
            
        print(f"引擎初始化完毕。共加载 {len(self.contract_cache)} 份 {OPT_PREFIX} 期权合约。")

    def _get_future_price(self, future_md: MarketData) -> float | None:
        # ... (此函数不变) ...
        try:
            fut_idx = future_md.instrument_id.index(FUT_CONTRACT)
        except ValueError:
            return None
        F_mid = utils.get_mid_price(
            future_md.bid_p1[fut_idx], 
            future_md.bid_v1[fut_idx], 
            future_md.ask_p1[fut_idx], 
            future_md.ask_v1[fut_idx]
        )
        return F_mid if F_mid is not None else future_md.last_price[fut_idx]

    def calibrate(self, option_md: MarketData, future_md: MarketData):
        print(f"--- 正在 {option_md.update_sec} 进行非对称曲面校准 ---")
        
        current_dt = utils.parse_date_time(
            self.trading_day, option_md.update_sec, option_md.update_msec
        )
        
        # (!!!) 新增：缓存校准时间
        self.last_calibrate_time = current_dt 
        
        F = self._get_future_price(future_md)
        if F is None:
            print("校准失败：无法获取标的期货价格。")
            return

        # ... (IV, SVI, Spline 的拟合逻辑完全不变) ...
        moneyness_points = []
        w_points = [] # 用于 SVI (Put)
        iv_points = [] # 用于 Spline (Call)
        for i, opt_id in enumerate(option_md.instrument_id):
            static_data = self.contract_cache.get(opt_id)
            if not static_data: continue 
            market_price = utils.get_mid_price(
                option_md.bid_p1[i], option_md.bid_v1[i], 
                option_md.ask_p1[i], option_md.ask_v1[i]
            )
            if market_price is not None:
                static_data["market_price_old"] = market_price
            if market_price is None or market_price <= 0.1: continue
            K = static_data["K"]
            opt_type = static_data["type"]
            T = utils.calculate_T(current_dt, static_data["expire_day"])
            log_moneyness = utils.get_log_moneyness(F, K)
            if not (LOG_MONEYNESS_FILTER_LOW <= log_moneyness <= LOG_MONEYNESS_FILTER_HIGH):
                continue
            iv = financial_models.implied_volatility(
                market_price, F, K, T, self.r, opt_type
            )
            if iv is not np.nan and 0.05 < iv < 1.5:
                moneyness_points.append(log_moneyness)
                w_points.append(iv**2 * T)
                iv_points.append(iv)
        
        if len(moneyness_points) > 5:
            fitted_svi = volatility_surface.fit_svi_surface(moneyness_points, w_points)
            if fitted_svi is not None:
                self.svi_params = fitted_svi
                print(f"SVI (Put) 校准成功: 使用 {len(moneyness_points)} 个点。")
            fitted_spline = volatility_surface.fit_spline_surface(moneyness_points, iv_points)
            if fitted_spline is not None:
                self.spline_model = fitted_spline
                print(f"Spline (Call) 校准成功: 使用 {len(moneyness_points)} 个点。")
        else:
            print(f"校准警告: 有效IV点不足 ({len(moneyness_points)})，将沿用旧曲面。")
        if self.svi_params is None or self.spline_model is None:
            return 
            
        # ... (Greeks 缓存逻辑完全不变) ...
        for opt_id, cache_item in self.contract_cache.items():
            K = cache_item["K"]
            opt_type = cache_item["type"]
            T = utils.calculate_T(current_dt, cache_item["expire_day"])
            k = utils.get_log_moneyness(F, K)
            if opt_type == 'P':
                sigma = volatility_surface.get_vol_from_svi_params(self.svi_params, k, T)
            else: # 'C'
                sigma = volatility_surface.get_vol_from_spline(self.spline_model, k)
            cache_item["T"] = T
            cache_item["delta"] = financial_models.delta_black76(F, K, T, self.r, sigma, opt_type)
            cache_item["gamma"] = financial_models.gamma_black76(F, K, T, self.r, sigma)
            cache_item["S_old"] = F

    def predict(self, future_md: MarketData, predict: SamplePrediction):
        """
        预测阶段 (每500毫秒):
        Price(t) = (Market_Price(t_calib) + Delta_Change) + ML_Residual
        """
        
        # (!!!) 如果模型加载失败或未校准，回退到 纯 Delta 锚定
        use_ml_residual = self.model_residual is not None and \
                          self.svi_params is not None and \
                          self.spline_model is not None and \
                          self.last_calibrate_time is not None

        F_new = self._get_future_price(future_md)
        current_dt = utils.parse_date_time( # (!!!) 获取当前时间
            self.trading_day, future_md.update_sec, future_md.update_msec
        )

        if F_new is None:
            # 回退：保持上一次的锚点市场价
            target_prices = []
            for opt_static in self.opt_static_list:
                cache_item = self.contract_cache.get(opt_static.instrument_id)
                target_prices.append(cache_item["market_price_old"] if cache_item else 0.0)
            predict.target = target_prices
            return
        
        # (!!!) --- 批量预测逻辑 --- (!!!)
        
        # 1. 准备容器
        baseline_prices = [0.0] * len(self.opt_static_list)
        features_list = []
        valid_indices = [] # 记录需要 ML 预测的行索引

        for i, opt_static in enumerate(self.opt_static_list):
            cache_item = self.contract_cache.get(opt_static.instrument_id)
            if not cache_item:
                continue

            # 2. 计算基线价格 (Delta-Gamma 锚定)
            market_price_old = cache_item["market_price_old"]
            S_old = cache_item["S_old"]
            delta = cache_item["delta"]
            gamma = cache_item["gamma"]
            S_diff = F_new - S_old
            price_baseline = market_price_old + delta * S_diff + 0.5 * gamma * (S_diff ** 2)
            
            baseline_prices[i] = price_baseline # 存储基线价格
            
            # 3. 如果不使用 ML，就此打住
            if not use_ml_residual:
                continue
                
            # 4. 收集 ML 特征
            try:
                K = cache_item["K"]
                T_new = utils.calculate_T(current_dt, cache_item["expire_day"])
                if T_new <= 1e-9: continue # 合约已到期
                    
                is_call = cache_item["is_call"]
                log_moneyness = utils.get_log_moneyness(F_new, K)
                time_since_anchor = (current_dt - self.last_calibrate_time).total_seconds()

                features = [
                    log_moneyness,
                    T_new,
                    is_call,
                    self.r,
                    time_since_anchor
                ]
                
                features_list.append(features)
                valid_indices.append(i) # 记录这一行
            except Exception:
                continue # 特征计算失败

        # 5. 批量预测残差
        final_prices = np.array(baseline_prices)
        
        if use_ml_residual and features_list:
            X_predict = pd.DataFrame(features_list, columns=self.model_features)
            predicted_residuals = self.model_residual.predict(X_predict)
            
            # 6. 合并: Final = Baseline + (Residual * Weight)
            valid_baselines = final_prices[valid_indices]
            
            # (!!!) --- 修正：给 ML 模型降权 --- (!!!)
            # 我们只相信 ML 模型 10% 或 20%
            # 这是一个超参数，你可以先从 0.1 尝试
            SHRINKAGE_FACTOR = 1
            weighted_residuals = predicted_residuals * SHRINKAGE_FACTOR
            
            valid_final_prices = valid_baselines + weighted_residuals
            final_prices[valid_indices] = valid_final_prices

        # 7. 清理并提交
        final_prices[final_prices < 0] = 0.0
        final_prices[np.isnan(final_prices)] = 0.0
        
        predict.target = final_prices.tolist()