# pricing_engine.py

from cfe_fin_math_api import CfeFinMathApi, InstrumentStaticData, MarketData, SamplePrediction
import utils
import financial_models
import volatility_surface
import numpy as np
from datetime import datetime

# --- 全局配置 ---
FUT_CONTRACT = "IF2507"
OPT_PREFIX = "IO2507"

# 宽范围过滤器：让 Spline 看到更多数据，捕捉完整的波动率微笑
LOG_MONEYNESS_FILTER_LOW = -0.35
LOG_MONEYNESS_FILTER_HIGH = 0.35

# 时间常数
SECONDS_IN_YEAR_PRECISE = 365.25 * 24 * 60 * 60

class PricingEngine:
    """
    期权定价引擎 (纯净版：Spline + Skew Delta / Vanna 修正)
    """
    
    # (!!!) 修复：确保 __init__ 接收 api 参数
    def __init__(self, api: CfeFinMathApi):
        print("正在初始化定价引擎 (纯净版：Spline + Skew Delta)...")
        
        # 使用传入的 api 获取静态数据
        self.opt_static_list: list[InstrumentStaticData] = api.get_option_static_md()
        self.fut_static_list: list[InstrumentStaticData] = api.get_future_static_md()
        
        self.trading_day: str = api.date
        self.r: float = 0.03

        self.spline_model = None
        self.last_calibrate_time = None

        self.contract_cache = {}
        
        for s in self.opt_static_list:
            if not s.instrument_id.startswith(OPT_PREFIX):
                continue
            
            try:
                parts = s.instrument_id.split('-')
                option_type = parts[1] 
                strike_price = float(parts[2]) 
                
                self.contract_cache[s.instrument_id] = {
                    "K": strike_price,
                    "expire_day": s.expire_day,
                    "type": option_type,
                    # 缓存所有 Greeks
                    "market_price_old": 0.0, 
                    "S_old": 0.0,
                    "delta": 0.0,
                    "gamma": 0.0, 
                    "theta": 0.0, 
                    "vega": 0.0,  # 用于 Skew 修正
                    "slope": 0.0  # 用于 Skew 修正
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
            future_md.bid_p1[fut_idx], 
            future_md.bid_v1[fut_idx], 
            future_md.ask_p1[fut_idx], 
            future_md.ask_v1[fut_idx]
        )
        return F_mid if F_mid is not None else future_md.last_price[fut_idx]

    def calibrate(self, option_md: MarketData, future_md: MarketData):
        """
        校准阶段：拟合统一 Spline，不做过度过滤
        """
        print(f"--- 正在 {option_md.update_sec} 进行 (Spline+Skew) 曲面校准 ---")
        
        current_dt = utils.parse_date_time(
            self.trading_day, option_md.update_sec, option_md.update_msec
        )
        self.last_calibrate_time = current_dt
        
        F = self._get_future_price(future_md)
        if F is None:
            print("校准失败：无法获取标的期货价格。")
            return

        moneyness_points = []
        iv_points = []
        weights = [] # 如果 volatility_surface 支持权重，这里收集 Vega

        # --- 步骤 1: 数据收集 ---
        for i, opt_id in enumerate(option_md.instrument_id):
            static_data = self.contract_cache.get(opt_id)
            if not static_data: continue 

            # 更新锚点价格 (不做 Spread 过滤，保留原始市场信息)
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
                iv_points.append(iv)
                
                # 尝试计算 Vega 作为权重 (可选优化)
                try:
                    vega_w = financial_models.vega_black76(F, K, T, self.r, iv)
                    weights.append(max(vega_w, 1e-4))
                except:
                    weights.append(1.0)
        
        # --- 步骤 2: 拟合平滑样条 ---
        if len(moneyness_points) > 5:
            # 尝试传递 weights (取决于你的 volatility_surface.py 是否已更新支持它)
            # 如果没更新，fit_spline_surface 会忽略多余参数或报错，这里做个简单的兼容性处理
            try:
                self.spline_model = volatility_surface.fit_spline_surface(
                    moneyness_points, iv_points, weights=weights
                )
            except TypeError:
                # 如果 fit_spline_surface 不接受 weights，则回退到旧调用
                self.spline_model = volatility_surface.fit_spline_surface(
                    moneyness_points, iv_points
                )
                
            if self.spline_model:
                print(f"Spline 校准成功: 使用 {len(moneyness_points)} 个点。")
            else:
                print("Spline 拟合失败。")
        else:
            print(f"校准警告: 有效IV点不足 ({len(moneyness_points)})，将沿用旧曲面。")

        if self.spline_model is None:
            return 

        # --- 步骤 3: 计算并缓存 Greeks (含 Vega 和 Slope) ---
        print("Greeks 缓存：正在计算 Delta, Gamma, Theta, Vega 和 Slope...")
        
        for opt_id, cache_item in self.contract_cache.items():
            K = cache_item["K"]
            opt_type = cache_item["type"]
            T = utils.calculate_T(current_dt, cache_item["expire_day"])
            k = utils.get_log_moneyness(F, K)
            
            # 获取 Sigma 和 Slope
            sigma, slope = volatility_surface.get_vol_and_slope_from_spline(
                self.spline_model, k
            )
            
            cache_item["delta"] = financial_models.delta_black76(F, K, T, self.r, sigma, opt_type)
            cache_item["gamma"] = financial_models.gamma_black76(F, K, T, self.r, sigma)
            cache_item["theta"] = financial_models.theta_black76(F, K, T, self.r, sigma, opt_type)
            
            # 缓存 Skew 相关参数
            cache_item["vega"] = financial_models.vega_black76(F, K, T, self.r, sigma)
            cache_item["slope"] = slope
            
            cache_item["S_old"] = F


    def predict(self, future_md: MarketData, predict: SamplePrediction):
        """
        预测阶段：Delta + Gamma + Theta + SkewDelta (Vanna)
        """
        if self.spline_model is None or self.last_calibrate_time is None:
            predict.target = [0.0] * len(self.opt_static_list)
            return

        F_new = self._get_future_price(future_md)
        current_dt = utils.parse_date_time(
            self.trading_day, future_md.update_sec, future_md.update_msec
        )

        if F_new is None:
            target_prices = []
            for opt_static in self.opt_static_list:
                cache_item = self.contract_cache.get(opt_static.instrument_id)
                target_prices.append(cache_item["market_price_old"] if cache_item else 0.0)
            predict.target = target_prices
            return
        
        time_diff_seconds = (current_dt - self.last_calibrate_time).total_seconds()
        dt = max(0, time_diff_seconds / SECONDS_IN_YEAR_PRECISE)
        
        target_prices = []
        for opt_static in self.opt_static_list:
            cache_item = self.contract_cache.get(opt_static.instrument_id)
            if not cache_item:
                target_prices.append(0.0); continue

            market_price_old = cache_item["market_price_old"]
            S_old = cache_item["S_old"]
            delta = cache_item["delta"]
            gamma = cache_item["gamma"]
            theta = cache_item["theta"]
            vega = cache_item["vega"]
            slope = cache_item["slope"]
            
            S_diff = F_new - S_old
            
            # Skew 修正 (Vanna)
            # 捕捉因标的价格变化导致的波动率沿曲面滑动带来的价格影响
            skew_correction = 0.0
            if S_old > 1e-6:
                skew_correction = vega * slope * (-S_diff / S_old)

            # 汇总
            price_change = (delta * S_diff) + \
                           (0.5 * gamma * (S_diff ** 2)) + \
                           (theta * dt) + \
                           skew_correction
            
            price_new = market_price_old + price_change
            target_prices.append(max(0, price_new))

        predict.target = target_prices