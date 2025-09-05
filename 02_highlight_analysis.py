# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:16:52 2025

@author: aa36
"""

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_data_and_embeddings(config):
    """Load original data and UMAP embeddings"""
    # Load original data
    df = pd.read_csv(config['paths']['input_data'], sep=config['paths']['separator'])
    
    # Load the "all features" embedding
    embedding_path = (
        Path(config['paths']['base_dir']) / 
        f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}" /
        "embeddings" /
        f"umap_2d_all_mindist{config['umap']['min_dist']}_nn{config['umap']['n_neighbors']}_coloredby_{config['columns']['color_column'].replace(' ', '_')}.csv"
    )
    
    embedding = pd.read_csv(embedding_path)
    return df, embedding

def create_highlight_visualization(df, embedding, config, save_path):
    """Create the combined file and visualizations"""
    # Combine data - THIS CREATES THE FILE YOU NEED
    combined_df = pd.concat([
        embedding[['UMAP1', 'UMAP2']].reset_index(drop=True),
        df.reset_index(drop=True)
    ], axis=1)
    
    # Save combined data
    combined_path = Path(save_path) / "combined_embedding_with_metadata.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined data to: {combined_path}")
    
    return combined_df

def main():
    config = load_config()
    
    # Setup output directory
    output_dir = (
        Path(config['paths']['base_dir']) / 
        f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}" /
        "highlight_analysis"
    )
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df, embedding = load_data_and_embeddings(config)
    
    # Create and save combined data
    combined_df = create_highlight_visualization(df, embedding, config, output_dir)

if __name__ == "__main__":
    main()