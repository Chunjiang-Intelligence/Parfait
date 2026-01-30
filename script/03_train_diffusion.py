import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.ts_backbone import TransformerEncoder
from src.models.diffusion_unet import ConditionalUNet1D

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_cumprod = self.alphas_cumprod[timesteps].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[timesteps]).sqrt()
        
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()
        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
            
        return sqrt_alpha_cumprod * original_samples + sqrt_one_minus_alpha_cumprod * noise

class DiffusionDataset(Dataset):
    def __init__(self, data_list, features_idx):
        self.data = data_list
        self.features_idx = features_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 历史序列 [HistoryWindow, Features] -> [Features, HistoryWindow]
        history = torch.tensor(item['history']).permute(1, 0)
        
        # 2. 未来真值序列 [FutureHorizon, Features] -> [Features, FutureHorizon]
        future = torch.tensor(item['future_gt']).permute(1, 0)
        
        close_idx = self.features_idx.index('close_qfq')
        guide = future[close_idx:close_idx+1, :].clone()
        guide = guide / (guide[:, 0:1] + 1e-8) # 以后预测窗口第一天为基准1.0
        
        return {
            "history": history,
            "future": future,
            "guide": guide
        }

def train():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['training']['seed'])

    os.makedirs(config['diffusion']['output_dir'], exist_ok=True)

    data_path = os.path.join(config['data']['processed_path'], "dataset_v1.pkl")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Run script 01 first.")
        return

    raw_data_list = pd.read_pickle(data_path)
    dataset = DiffusionDataset(raw_data_list, config['data']['features'])
    dataloader = DataLoader(dataset, batch_size=config['diffusion']['batch_size'], shuffle=True, num_workers=4)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    backbone = TransformerEncoder(
        input_dim=config['data']['features_num'], 
        model_dim=config['diffusion']['context_dim'], 
        num_heads=8, 
        num_layers=3
    ).to(device)

    diffusion_model = ConditionalUNet1D(
        input_channels=config['data']['features_num'], 
        guide_channels=1, 
        model_channels=config['diffusion']['model_channels'], 
        context_dim=config['diffusion']['context_dim']
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(diffusion_model.parameters()), 
        lr=float(config['diffusion']['learning_rate']),
        weight_decay=1e-5
    )

    criterion = nn.MSELoss()

    print(f"Starting training on {device}...")
    
    for epoch in range(config['diffusion']['num_epochs']):
        backbone.train()
        diffusion_model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # 数据搬运
            # history: [B, Features, Hist_Len] -> 需要转回 [B, Hist_Len, Features] 给 Transformer
            hist = batch['history'].permute(0, 2, 1).to(device)
            future = batch['future'].to(device) # [B, Features, Future_Len]
            guide = batch['guide'].to(device)   # [B, 1, Future_Len]
            style_context = backbone(hist) # [B, ContextDim]

            noise = torch.randn_like(future)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (future.shape[0],), device=device).long()

            noisy_future = scheduler.add_noise(future, noise, timesteps)

            noise_pred = diffusion_model(noisy_future, timesteps, style_context, guide)

            loss = criterion(noise_pred, noise)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(diffusion_model.parameters()), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(backbone.state_dict(), os.path.join(config['diffusion']['output_dir'], f"backbone_ep{epoch}.pt"))
            torch.save(diffusion_model.state_dict(), os.path.join(config['diffusion']['output_dir'], f"unet_ep{epoch}.pt"))

    torch.save(backbone.state_dict(), os.path.join(config['diffusion']['output_dir'], "backbone_final.pt"))
    torch.save(diffusion_model.state_dict(), os.path.join(config['diffusion']['output_dir'], "unet_final.pt"))
    print("Training Completed.")

if __name__ == "__main__":
    train()