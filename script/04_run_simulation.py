import torch
import yaml
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.llm_agent import LLMAgent
from src.models.ts_backbone import TransformerEncoder
from src.models.diffusion_unet import ConditionalUNet1D
from src.inference.pipeline import InferencePipeline

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = LLMAgent(config['llm']['model_name'], use_lora=True)
    
    backbone = TransformerEncoder(input_dim=5, model_dim=256, num_heads=8, num_layers=3)
    backbone.load_state_dict(torch.load(os.path.join(config['diffusion']['output_dir'], "backbone.pt")))
    
    diffusion = ConditionalUNet1D(
        input_channels=5, guide_channels=1, 
        model_channels=config['diffusion']['model_channels'], 
        context_dim=config['diffusion']['context_dim']
    )
    diffusion.load_state_dict(torch.load(os.path.join(config['diffusion']['output_dir'], "unet.pt")))

    pipeline = InferencePipeline(agent, backbone, diffusion, config)

    sample_df = pd.read_csv(os.path.join(config['data']['raw_path'], os.listdir(config['data']['raw_path'])[0]))
    history_data = sample_df[config['data']['features']].iloc[:60].values

    instruction = "如果现在央行意外加息，且市场出现恐慌性抛售，但随后有国家队入场救市"
    simulated_paths = pipeline.run(instruction, history_data, total_simulations=50)

    np.save("simulated_results.npy", simulated_paths)
    print(f"Successfully generated 50 paths based on instruction. Saved to simulated_results.npy")

if __name__ == "__main__":
    main()