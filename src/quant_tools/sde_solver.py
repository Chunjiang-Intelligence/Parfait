import numpy as np
import torch

def generate_sde_path(
    sde_params: dict,
    initial_price: float,
    horizon: int,
    dt: float = 1/252
) -> np.ndarray:
    prices = np.zeros(horizon)
    prices[0] = initial_price
    
    mu = sde_params.get("annualized_drift", 0.0)
    sigma = sde_params.get("realized_volatility", 0.2)
    jump_intensity = sde_params.get("jump_intensity", 0.0)
    jump_mean = sde_params.get("jump_mean", 0.0)

    for t in range(1, horizon):
        # 扩散项
        diffusion = sigma * np.sqrt(dt) * np.random.randn()
        # 漂移项
        drift = mu * dt
        # 跳跃项
        jump = 0
        if np.random.rand() < jump_intensity:
             jump = jump_mean
        
        # 价格更新
        prices[t] = prices[t-1] * np.exp(drift - 0.5 * sigma**2 * dt + diffusion + jump)
        
    return prices

def normalize_path(path: np.ndarray) -> np.ndarray:
    return path / path[0]