import numpy as np
import pandas as pd
from arch import arch_model

def calculate_hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    计算 Hurst 指数。
    Hurst 指数 < 0.5: 均值回归
    Hurst 指数 = 0.5: 随机游走
    Hurst 指数 > 0.5: 趋势增强
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

def calculate_realized_volatility(log_returns: np.ndarray) -> float:
    """
    计算年化已实现波动率。
    """
    return np.sqrt(np.sum(log_returns**2)) * np.sqrt(252 / len(log_returns))

def detect_jumps_bipower(log_returns: np.ndarray, threshold_multiplier: float = 3.0) -> tuple[float, float, float]:
    """
    使用简化的阈值法检测跳跃。
    更严谨的方法是 Bipower Variation，但这里为了快速实现，采用标准差阈值。
    """
    if len(log_returns) < 2:
        return 0.0, 0.0, 0.0
        
    std_dev = np.std(log_returns)
    threshold = threshold_multiplier * std_dev
    
    jumps = log_returns[np.abs(log_returns) > threshold]
    
    jump_intensity = len(jumps) / len(log_returns) if len(log_returns) > 0 else 0.0
    jump_mean = np.mean(jumps) if len(jumps) > 0 else 0.0
    jump_vol = np.std(jumps) if len(jumps) > 0 else 0.0
    
    return jump_intensity, jump_mean, jump_vol

def fit_garch_params(log_returns: np.ndarray) -> dict:
    """
    拟合 GARCH(1,1) 模型并返回其关键参数。
    GARCH 模型用于描述波动率聚类现象。
    """
    if np.var(log_returns) < 1e-12:
        return {'omega': 0, 'alpha': 0, 'beta': 0}
        
    garch_model = arch_model(log_returns * 100, vol='Garch', p=1, q=1, dist='Normal')
    
    try:
        res = garch_model.fit(disp='off')
        params = res.params
        return {
            'omega': params.get('omega', 0),      # 长期方差的常数项
            'alpha': params.get('alpha[1]', 0),   # ARCH 项系数 (前期误差)
            'beta': params.get('beta[1]', 0)      # GARCH 项系数 (前期方差)
        }
    except Exception:
        # 如果模型无法收敛，返回默认值
        return {'omega': 0, 'alpha': 0, 'beta': 0}