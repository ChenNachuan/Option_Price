# pricing_engine.py

from cfe_fin_math_api import CfeFinMathApi, InstrumentStaticData, MarketData, SamplePrediction
import utils
import financial_models
import volatility_surface
import numpy as np
from datetime import datetime

# 项目说明中指定的标的期货和期权前缀
FUT_CONTRACT = "IF2507" # (!!!) 警告：你的日志显示 IO2507，请确保这里匹配！
OPT_PREFIX = "IO2507"   # (!!!) 警告：你的日志显示 IO2507，请确保这里匹配！

# 优化 3: 数据过滤设置
LOG_MONEYNESS_FILTER_LOW = -0.25
LOG_MONEYNESS_FILTER_HIGH = 0.25
SECONDS_IN_YEAR_PRECISE = 365.25 * 24 * 60 * 60

class PricingEngine:
    """
    期权定价引擎 (最终方案 优化版 - 纯 Cubic Spline 锚定模型)
    """
    def __init__(self, api: CfeFinMathApi):
        print("正在初始化定价引擎 (最终方案 优化版 - 纯 Spline 锚定)...")
        
        self.opt_static_list: list[InstrumentStaticData] = api.get_option_static_md()
        self.fut_static_list: list[InstrumentStaticData] = api.get_future_static_md()
        
        self.trading_day: str = api.date
        self.r: float = 0.03  # 假设无风险利率为 3%

        # (!!!) 修正：我们不再需要 SVI
        # self.svi_params = None
        self.spline_model = None  # (!!!) 只使用 Spline
        
        self.last_calibrate_time = None

        # HPC 缓存 (不变)
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
                    "T": 0.0,
                    "market_price_old": 0.0, 
                    "delta": 0.0,
                    "gamma": 0.0, 
                    "theta": 0.0, 
                    "S_old": 0.0
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
        # (此函数不变)
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
        print(f"--- 正在 {option_md.update_sec} 进行 (纯 Spline) 曲面校准 ---")
        
        current_dt = utils.parse_date_time(
            self.trading_day, option_md.update_sec, option_md.update_msec
        )
        self.last_calibrate_time = current_dt
        
        F = self._get_future_price(future_md)
        if F is None:
            print("校准失败：无法获取标的期货价格。")
            return

        moneyness_points = []
        # w_points = [] # (!!!) 移除 SVI
        iv_points = [] # (!!!) 保留 Spline

        # 步骤 1: 收集 IV 数据 (不变)
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
                # (!!!) 关键：无论 C/P，都加入 iv_points
                moneyness_points.append(log_moneyness)
                # w_points.append(iv**2 * T) # (!!!) 移除 SVI
                iv_points.append(iv)
        
        # 步骤 2: (!!!) 只拟合 Spline
        if len(moneyness_points) > 5: # (Spline 最好 > 4 个点)
            # (!!!) 移除 SVI
            # fitted_svi = volatility_surface.fit_svi_surface(moneyness_points, w_points)
            # if fitted_svi is not None:
            #     self.svi_params = fitted_svi
            
            fitted_spline = volatility_surface.fit_spline_surface(moneyness_points, iv_points)
            if fitted_spline is not None:
                self.spline_model = fitted_spline
                print(f"纯 Spline (Call+Put) 校准成功: 使用 {len(moneyness_points)} 个点。")
        else:
            print(f"校准警告: 有效IV点不足 ({len(moneyness_points)})，将沿用旧曲面。")

        if self.spline_model is None: # (!!!) 只检查 Spline
            return # 沿用旧曲面

        # 步骤 3: (!!!) 缓存 Greeks (只使用 Spline)
        print("Greeks 缓存：正在计算 Delta, Gamma, 和 Theta...")
        for opt_id, cache_item in self.contract_cache.items():
            K = cache_item["K"]
            opt_type = cache_item["type"]
            T = utils.calculate_T(current_dt, cache_item["expire_day"])
            k = utils.get_log_moneyness(F, K)
            
            # (!!!) 关键修正：所有期权 (C 和 P) 都使用 Spline
            sigma = volatility_surface.get_vol_from_spline(self.spline_model, k)
            
            # (缓存 Greeks 的代码不变)
            cache_item["T"] = T
            cache_item["delta"] = financial_models.delta_black76(F, K, T, self.r, sigma, opt_type)
            cache_item["gamma"] = financial_models.gamma_black76(F, K, T, self.r, sigma)
            cache_item["theta"] = financial_models.theta_black76(F, K, T, self.r, sigma, opt_type)
            cache_item["S_old"] = F


    def predict(self, future_md: MarketData, predict: SamplePrediction):
        """
        预测阶段 (每500毫秒):
        使用 "Delta-Gamma-Theta 锚定" 模型 (基于纯 Spline 的 Greeks)
        """
        
        # (!!!) 修正：只检查 Spline
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
        
        # (Delta-Gamma-Theta 锚定逻辑完全不变)
        
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
            S_diff = F_new - S_old
            
            price_change = (delta * S_diff) + \
                           (0.5 * gamma * (S_diff ** 2)) + \
                           (theta * dt)
            
            price_new = market_price_old + price_change
            
            target_prices.append(max(0, price_new))

        predict.target = target_prices