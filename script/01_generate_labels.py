import os
import sys
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.loader import load_stock_data, create_sliding_windows
from src.data_processing.labeler import generate_labels_for_window

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config['data']['raw_path']
    processed_dir = config['data']['processed_path']
    os.makedirs(processed_dir, exist_ok=True)

    all_windows = []
    
    raw_full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), raw_dir)
    stock_files = [f for f in os.listdir(raw_full_path) if f.endswith('.csv')]

    print(f"Processing {len(stock_files)} stock files from {raw_full_path}...")
    
    history_window = config['data']['history_window']
    future_horizon = config['data']['future_horizon']
    features = config['data']['features']
    
    for file in tqdm(stock_files):
        file_path = os.path.join(raw_full_path, file)
        df = load_stock_data(file_path, features)
        if df.empty: continue

        for hist_df, fut_df in create_sliding_windows(df, history_window, future_horizon):
            labels = generate_labels_for_window(fut_df)
            if not labels: continue
            
            window_data = {
                "history": hist_df[features].values.astype(np.float32),
                "future_gt": fut_df[features].values.astype(np.float32),
                "labels": labels,
                "stock_code": file.replace(".csv", "")
            }
            all_windows.append(window_data)

    pd.to_pickle(all_windows, os.path.join(processed_dir, "dataset_v1.pkl"))
    print(f"\nDataset generated with {len(all_windows)} windows, saved to {processed_dir}.")

if __name__ == "__main__":
    main()
