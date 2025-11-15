# volatility_models.py

"""
波动率曲面模型库 (Volatility Surface Model Library)
为期权定价项目提供动态拟合波动率曲面的模型。
包括：
- BaseVolatilityModel (抽象基类)
- CubicSplineModel (三次样条插值)
- PolynomialModel (多项式拟合，类 Wing 模型)
- fit_surfaces_by_month (核心管理函数)
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# 当拟合失败或数据不足时返回的默认波动率
DEFAULT_VOLATILITY = 0.20

def _prepare_data(K_array: np.ndarray, IV_array: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    一个内部辅助函数，用于清洗和准备用于拟合的数据。
    1. 移除 NaN
    2. 按 K 排序
    3. 移除 K 的重复值 (取 IV 的平均值)
    4. 检查数据点是否足够
    """
    try:
        # 使用 pandas 快速清洗数据
        df = pd.DataFrame({'K': K_array, 'IV': IV_array})
        
        # 1. 移除 NaN
        df = df.dropna()
        
        if df.empty:
            return None
            
        # 2. 按 K 排序，并对 K 重复的值取 IV 平均值
        df_clean = df.groupby('K')['IV'].mean().reset_index()
        
        # 3. 确保至少有3个点来进行有意义的拟合
        if len(df_clean) < 3:
            return None
            
        return df_clean['K'].values, df_clean['IV'].values
        
    except Exception:
        return None

class BaseVolatilityModel(ABC):
    """
    波动率模型抽象基类 (接口)。
    所有模型都必须实现 fit 和 predict 方法。
    """
    def __init__(self):
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, K_array: np.ndarray, IV_array: np.ndarray):
        """
        使用 (K, IV) 数据点拟合模型。
        """
        pass
    
    @abstractmethod
    def predict(self, K: float) -> float:
        """
        预测给定 K 值的波动率。
        """
        pass

class CubicSplineModel(BaseVolatilityModel):
    """
    使用三次样条插值拟合波动率微笑。
    这是一种非参数方法，非常灵活。
    """
    def __init__(self):
        super().__init__()
        self.spline_model: Optional[CubicSpline] = None
        
    def fit(self, K_array: np.ndarray, IV_array: np.ndarray):
        prepared_data = _prepare_data(K_array, IV_array)
        
        if prepared_data is None:
            self.is_fitted = False
            return
            
        K_clean, IV_clean = prepared_data
        
        try:
            # 拟合三次样条，设置 extrapolate=True 以允许外推
            self.spline_model = CubicSpline(K_clean, IV_clean, extrapolate=True)
            self.is_fitted = True
        except ValueError as e:
            # print(f"三次样条拟合失败: {e}")
            self.is_fitted = False
            
    def predict(self, K: float) -> float:
        if not self.is_fitted or self.spline_model is None:
            return DEFAULT_VOLATILITY
        
        # 预测 sigma 值
        sigma = float(self.spline_model(K))
        
        # 防止外推时出现负波动率或极端值
        return max(0.01, min(sigma, 5.0)) # 限制在 1% 到 500%

class PolynomialModel(BaseVolatilityModel):
    """
    使用二阶多项式 (抛物线) 拟合波动率微笑。
    这是一种参数化方法，类似于简化的 "Wing 模型"。
    """
    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree
        self.poly_model: Optional[np.poly1d] = None

    def fit(self, K_array: np.ndarray, IV_array: np.ndarray):
        prepared_data = _prepare_data(K_array, IV_array)
        
        if prepared_data is None:
            self.is_fitted = False
            return
            
        K_clean, IV_clean = prepared_data
        
        # 确保数据点足够进行多项式拟合
        if len(K_clean) <= self.degree:
            self.is_fitted = False
            return
            
        try:
            # 拟合 N 阶多项式
            coeffs = np.polyfit(K_clean, IV_clean, self.degree)
            self.poly_model = np.poly1d(coeffs)
            self.is_fitted = True
        except (np.linalg.LinAlgError, ValueError) as e:
            # print(f"多项式拟合失败: {e}")
            self.is_fitted = False
            
    def predict(self, K: float) -> float:
        if not self.is_fitted or self.poly_model is None:
            return DEFAULT_VOLATILITY
            
        sigma = float(self.poly_model(K))
        
        # 防止拟合的抛物线出现负值
        return max(0.01, min(sigma, 5.0)) # 限制在 1% 到 500%

def fit_surfaces_by_month(
    iv_data: Dict[str, float],
    static_data_map: Dict[str, Dict[str, Any]],
    method: str = 'spline'
) -> Dict[str, BaseVolatilityModel]:
    """
    高级管理函数。
    接收所有合约的 IV 数据，按月份分组，并为每个月份拟合一个波动率模型。
    
    参数:
    iv_data: { 'IO2507-C-3800': 0.22, 'IO2507-P-3700': 0.24, ... }
    static_data_map: 从 financial_utils.get_static_data_map() 来的字典
    method: 'spline' 或 'poly'
    
    返回:
    { 'IO2507': CubicSplineModel(fitted), 'IO2508': CubicSplineModel(fitted), ... }
    """
    
    # 1. 按月份聚合数据
    monthly_data = {}
    
    for instrument_id, iv in iv_data.items():
        if not np.isfinite(iv):
            continue
            
        # 从合约ID中提取月份 (例如 'IO2507')
        month_id = instrument_id.split('-')[0]
        
        # 从 static_data_map 获取行权价
        strike = static_data_map.get(instrument_id, {}).get('strike')
        
        if strike is None:
            continue
            
        # 初始化月份的列表
        if month_id not in monthly_data:
            monthly_data[month_id] = {'K': [], 'IV': []}
            
        # 添加数据
        monthly_data[month_id]['K'].append(strike)
        monthly_data[month_id]['IV'].append(iv)
        
    # 2. 为每个月份拟合模型
    fitted_surfaces = {}
    
    for month_id, data in monthly_data.items():
        K_array = np.array(data['K'])
        IV_array = np.array(data['IV'])
        
        # 选择模型
        if method.lower() == 'spline':
            model = CubicSplineModel()
        elif method.lower() == 'poly':
            model = PolynomialModel(degree=2)
        else:
            raise ValueError("未知的拟合方法。请选择 'spline' 或 'poly'。")
            
        # 拟合模型
        model.fit(K_array, IV_array)
        
        # 存入字典
        fitted_surfaces[month_id] = model
        
    return fitted_surfaces