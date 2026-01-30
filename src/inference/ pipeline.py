import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from ..models.llm_agent import LLMAgent
from ..models.ts_backbone import TransformerEncoder
from ..models.diffusion_unet import ConditionalUNet1D
from ..quant_tools.sde_solver import generate_sde_path, normalize_path

def get_ddpm_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02):
    """
    生成 DDPM 所需的 variance schedule
    """
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1. - alphas_cumprod),
        "posterior_variance": betas * (1. - torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])) / (1. - alphas_cumprod)
    }

class InferencePipeline:
    def __init__(self, llm_agent: LLMAgent, ts_backbone: TransformerEncoder, diffusion_model: ConditionalUNet1D, config: Dict, device="cuda"):
        self.llm = llm_agent
        self.backbone = ts_backbone
        self.diffusion = diffusion_model
        self.config = config
        self.device = device
        
        self.llm.model.to(device)
        self.backbone.to(device)
        self.diffusion.to(device)

        self.num_timesteps = 1000
        self.schedule = {k: v.to(device) for k, v in get_ddpm_schedule(self.num_timesteps).items()}

    def run(self, instruction: str, history_data: np.ndarray, total_simulations: int = 100):
        self.llm.model.eval()
        self.backbone.eval()
        self.diffusion.eval()
        
        with torch.no_grad():
            print("Step 1: LLM generating multi-modal scenarios...")
            scenarios = self.llm.generate_params(instruction)
            print(f"Generated {len(scenarios)} scenarios.")
            for s in scenarios: print(f"  - Scenario '{s.get('scenario', 'N/A')}': Probability {s.get('probability', 0):.2f}")

            all_guide_curves = []
            all_style_vectors = []
            
            print("Step 2: Preparing guide curves and context for all scenarios...")
            history_tensor = torch.from_numpy(history_data).float().unsqueeze(0).to(self.device)
            base_style_vector = self.backbone(history_tensor)
            initial_price = history_data[-1, self.config['data']['features'].index('close_qfq')]

            for scenario in scenarios:
                prob = scenario.get('probability', 0)
                sde_params = scenario.get('params', {})
                num_paths = max(1, round(prob * total_simulations))
                for _ in range(num_paths):
                    guide_path = generate_sde_path(sde_params, initial_price, self.config['data']['future_horizon'])
                    all_guide_curves.append(normalize_path(guide_path))
                    all_style_vectors.append(base_style_vector)

            if not all_guide_curves:
                print("No valid scenarios generated. Aborting.")
                return None

            num_total_paths = len(all_guide_curves)
            guide_curves_tensor = torch.from_numpy(np.array(all_guide_curves)).float().unsqueeze(1).to(self.device)
            style_vectors_tensor = torch.cat(all_style_vectors, dim=0)

            print(f"Step 3: Starting parallel diffusion sampling for {num_total_paths} paths...")
            shape = (num_total_paths, len(self.config['data']['features']), self.config['data']['future_horizon'])
            sample = torch.randn(shape, device=self.device)

            for t in tqdm(reversed(range(self.num_timesteps)), desc="DDPM Sampling"):
                time_tensor = torch.full((num_total_paths,), t, device=self.device, dtype=torch.long)
                
                predicted_noise = self.diffusion(sample, time_tensor, style_vectors_tensor, guide_curves_tensor)
                
                # DDPM 采样步骤
                alpha_t_cumprod = self.schedule['alphas_cumprod'][t]
                sqrt_one_minus_alpha_t_cumprod = self.schedule['sqrt_one_minus_alphas_cumprod'][t]
                beta_t = self.schedule['betas'][t]
                
                model_mean = (1.0 / torch.sqrt(1.0 - beta_t)) * (sample - beta_t * predicted_noise / sqrt_one_minus_alpha_t_cumprod)
                
                if t > 0:
                    posterior_variance = self.schedule['posterior_variance'][t]
                    noise = torch.randn_like(sample)
                    sample = model_mean + torch.sqrt(posterior_variance) * noise
                else:
                    sample = model_mean

        print("Simulation finished.")
        simulations = sample.cpu().numpy()
        close_price_index = self.config['data']['features'].index('close_qfq')
        simulations[:, close_price_index, :] *= initial_price

        return simulations # Shape: [num_simulations, Features, Horizon]