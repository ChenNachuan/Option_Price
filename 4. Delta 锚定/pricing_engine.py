from cfe_fin_math_api import CfeFinMathApi, InstrumentStaticData, MarketData, SamplePrediction
import utils
import financial_models
import volatility_surface
import numpy as np

# 项目说明中指定的标的期货和期权前缀
FUT_CONTRACT = "IF2507"
OPT_PREFIX = "IO2507"

# 优化 3: 数据过滤设置
LOG_MONEYNESS_FILTER_LOW = -0.3
LOG_MONEYNESS_FILTER_HIGH = 0.3

class PricingEngine:
    """
    期权定价引擎 (最终优化版 - Delta 锚定模型)
    """
    def __init__(self, api: CfeFinMathApi):
        print("正在初始化定价引擎 (最终优化版 - Delta 锚定)...")
        
        self.opt_static_list: list[InstrumentStaticData] = api.get_option_static_md()
        self.fut_static_list: list[InstrumentStaticData] = api.get_future_static_md()
        
        self.trading_day: str = api.date
        self.r: float = 0.03  # 假设无风险利率为 3%

        # 非对称波动率模型
        self.svi_params = None    # 用于 Put (GARCH 代理)
        self.spline_model = None  # 用于 Call (ML 代理)

        # HPC 缓存 - 现在存储 市场价格 而非 理论价格
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
                    # *** 核心变更：我们现在缓存 市场价格 作为锚点 ***
                    "market_price_old": 0.0, 
                    "delta": 0.0,
                    "gamma": 0.0, 
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
        F = self._get_future_price(future_md)
        if F is None:
            print("校准失败：无法获取标的期货价格。")
            return

        moneyness_points = []
        w_points = [] # 用于 SVI (Put)
        iv_points = [] # 用于 Spline (Call)

        # *** 核心变更：在收集IV数据的同时，缓存 市场价格 ***
        for i, opt_id in enumerate(option_md.instrument_id):
            static_data = self.contract_cache.get(opt_id)
            if not static_data: continue 

            market_price = utils.get_mid_price(
                option_md.bid_p1[i], option_md.bid_v1[i], 
                option_md.ask_p1[i], option_md.ask_v1[i]
            )
            
            # *** 缓存 市场价格 作为锚点 ***
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
                w_points.append(iv**2 * T) # 收集 SVI 数据
                iv_points.append(iv)       # 收集 Spline 数据
        
        # 拟合两个模型
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

        # 计算并缓存所有合约的 Delta, Gamma
        for opt_id, cache_item in self.contract_cache.items():
            K = cache_item["K"]
            opt_type = cache_item["type"]
            T = utils.calculate_T(current_dt, cache_item["expire_day"])
            k = utils.get_log_moneyness(F, K)
            
            if opt_type == 'P':
                sigma = volatility_surface.get_vol_from_svi_params(self.svi_params, k, T)
            else: # 'C'
                sigma = volatility_surface.get_vol_from_spline(self.spline_model, k)
            
            # 缓存
            cache_item["T"] = T
            cache_item["delta"] = financial_models.delta_black76(F, K, T, self.r, sigma, opt_type)
            cache_item["gamma"] = financial_models.gamma_black76(F, K, T, self.r, sigma)
            cache_item["S_old"] = F
            # 注意：我们不再缓存 "theo_price"，而是依赖 "market_price_old"


    def predict(self, future_md: MarketData, predict: SamplePrediction):
        """
        预测阶段 (每500毫秒):
        使用 "Delta 锚定" 模型
        Price(t) = Market_Price(t_calib) + Delta_Change
        """
        
        if self.svi_params is None or self.spline_model is None:
            predict.target = [0.0] * len(self.opt_static_list)
            return

        F_new = self._get_future_price(future_md)
        if F_new is None:
            # 如果无法获取期货价格，我们回退到 "基线Demo" 逻辑：
            # 保持上一次的预测值不变
            target_prices = []
            for opt_static in self.opt_static_list:
                cache_item = self.contract_cache.get(opt_static.instrument_id)
                target_prices.append(cache_item["market_price_old"] if cache_item else 0.0)
            predict.target = target_prices
            return
        
        target_prices = []
        for opt_static in self.opt_static_list:
            cache_item = self.contract_cache.get(opt_static.instrument_id)
            if not cache_item:
                target_prices.append(0.0); continue

            # *** 核心变更：实现 Delta 锚定 ***

            # 1. 获取锚点 (来自10分钟前的 真实市场)
            market_price_old = cache_item["market_price_old"]
            S_old = cache_item["S_old"]
            
            # 2. 获取Greeks (来自10分钟前的 理论模型)
            delta = cache_item["delta"]
            gamma = cache_item["gamma"]
            
            # 3. 计算标的物的变动
            S_diff = F_new - S_old
            
            # 4. 计算期权价格的 *变动*
            price_change = delta * S_diff + 0.5 * gamma * (S_diff ** 2)
            
            # 5. 新价格 = 旧的市场价格 + 理论的价格变动
            price_new = market_price_old + price_change
            
            target_prices.append(max(0, price_new))

        predict.target = target_prices