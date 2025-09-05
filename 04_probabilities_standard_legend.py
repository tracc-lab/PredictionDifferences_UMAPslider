import yaml
import pandas as pd
import numpy as np
import ast
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_reference_coordinate_ranges(config):
    """Get coordinate ranges from the 'All' dataset for consistent scaling"""
    base_dir = Path(config['paths']['base_dir']) / f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}"
    
    all_embedding_path = (
        base_dir / "embeddings" / 
        f"umap_2d_all_mindist{config['umap']['min_dist']}_nn{config['umap']['n_neighbors']}_coloredby_{config['columns']['color_column'].replace(' ', '_')}.csv"
    )
    
    if all_embedding_path.exists():
        df = pd.read_csv(all_embedding_path)
        
        # Get ranges with some padding
        x_min, x_max = df['UMAP1'].min(), df['UMAP1'].max()
        y_min, y_max = df['UMAP2'].min(), df['UMAP2'].max()
        
        # Add padding (10% of range)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        return {
            'x_min': x_min - x_padding,
            'x_max': x_max + x_padding,
            'y_min': y_min - y_padding,
            'y_max': y_max + y_padding
        }
    else:
        print(f"Warning: All embedding not found at {all_embedding_path}")
        return {'x_min': -15, 'x_max': 15, 'y_min': -15, 'y_max': 15}

def get_standardized_legend_layout():
    """Return standardized legend configuration for consistent positioning"""
    return dict(
        x=0.85,  # Fixed X position
        y=0.95,  # Fixed Y position  
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.3)",
        borderwidth=1,
        font=dict(size=12),
        itemsizing="constant",
        itemwidth=30,  # Fixed width
        tracegroupgap=5  # Fixed spacing
    )

def expand_prediction_arrays(df):
    """Convert string arrays to lists and explode into rows"""
    for col in ['Predictions', 'Probabilities', 'y_true']:
        if df[col].dtype == object and isinstance(df[col].iloc[0], str):
            df[col] = df[col].apply(ast.literal_eval)
    
    return df.explode(['Predictions', 'Probabilities', 'y_true']).reset_index(drop=True)

def create_probability_plots_standard_legend(config):
    # Get reference coordinate ranges
    coord_ranges = get_reference_coordinate_ranges(config)
    print(f"Using coordinate ranges: {coord_ranges}")
    
    # Load data
    highlight_path = (
        Path(config['paths']['base_dir']) / 
        f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}" /
        "highlight_analysis" /
        "combined_embedding_with_metadata.csv"
    )
    base_df = pd.read_csv(highlight_path)
    
    # Load and process results
    results_df = expand_prediction_arrays(pd.read_csv(config['paths']['detailed_results']))
    
    # Create output directory
    output_dir = (
        Path(config['paths']['base_dir']) / 
        f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}" /
        "probability_intensity_standard_legend")
    output_dir.mkdir(exist_ok=True)
    
    # Process each feature selection method
    for method in results_df['Feature_Selection_Method'].unique():
        exploded_results = results_df[results_df['Feature_Selection_Method'] == method].copy()
        
        # Special handling for "All" method to ensure exact coordinate alignment
        if method.lower() == 'all':
            merged_df = base_df.copy()
            
            if len(exploded_results) == len(base_df):
                merged_df['Predictions'] = exploded_results['Predictions'].values
                merged_df['Probabilities'] = exploded_results['Probabilities'].values
                merged_df['y_true'] = exploded_results['y_true'].values
            else:
                print(f"Warning: Length mismatch for All method. Base: {len(base_df)}, Results: {len(exploded_results)}")
                merged_df = pd.concat([
                    base_df.iloc[:len(exploded_results)].reset_index(drop=True),
                    exploded_results.reset_index(drop=True)
                ], axis=1)
        else:
            # For subset methods, use the original logic
            merged_df = base_df.iloc[:len(exploded_results)].copy().reset_index(drop=True)
            merged_df = pd.concat([merged_df, exploded_results], axis=1)
        
        # **KEY FIX: Clean data to remove NaN/Inf values**
        print(f"Before cleaning: {len(merged_df)} rows")
        merged_df = merged_df.dropna(subset=['UMAP1', 'UMAP2'])
        merged_df = merged_df[np.isfinite(merged_df['UMAP1']) & np.isfinite(merged_df['UMAP2'])]
        print(f"After cleaning: {len(merged_df)} rows")
        
        # Debug information
        print(f"\n--- Processing {method} ---")
        print(f"Exploded results length: {len(exploded_results)}")
        print(f"Merged dataframe length: {len(merged_df)}")
        print(f"UMAP1 range: [{merged_df['UMAP1'].min():.2f}, {merged_df['UMAP1'].max():.2f}]")
        print(f"UMAP2 range: [{merged_df['UMAP2'].min():.2f}, {merged_df['UMAP2'].max():.2f}]")
        
        # Create highlight mask
        highlight_mask = merged_df[config['highlight']['id_column']].isin(config['highlight']['ids'])
        
        # Create standardized legend categories
        merged_df['Legend_Category'] = 'Regular Points'
        
        # Create probability intensity categories for legend consistency
        high_prob_mask = highlight_mask & (merged_df['Probabilities'] >= 0.7)
        med_prob_mask = highlight_mask & (merged_df['Probabilities'] >= 0.3) & (merged_df['Probabilities'] < 0.7)
        low_prob_mask = highlight_mask & (merged_df['Probabilities'] < 0.3)
        
        merged_df.loc[high_prob_mask, 'Legend_Category'] = 'High Probability (≥0.7)'
        merged_df.loc[med_prob_mask, 'Legend_Category'] = 'Med Probability (0.3-0.7)'
        merged_df.loc[low_prob_mask, 'Legend_Category'] = 'Low Probability (<0.3)'
        
        # Color mapping for probability intensity
        color_map = {
            'Regular Points': '#7f7f7f',           # Gray
            'High Probability (≥0.7)': '#006400',  # Dark Green
            'Med Probability (0.3-0.7)': '#FFD700', # Gold
            'Low Probability (<0.3)': '#FF4500'     # Red-Orange
        }
        
        # Create plot with standardized legend
        fig = px.scatter(
            merged_df,
            x='UMAP1',
            y='UMAP2',
            color='Legend_Category',
            color_discrete_map=color_map,
            title=f"UMAP: Prediction Probability Intensity ({method})"
        )
        
        # Apply standardized layout (SAME as base image)
        fig.update_layout(
            # Fixed coordinate ranges
            xaxis=dict(
                range=[coord_ranges['x_min'], coord_ranges['x_max']],
                constrain='domain',
                title="UMAP1"
            ),
            yaxis=dict(
                range=[coord_ranges['y_min'], coord_ranges['y_max']],
                scaleanchor="x",
                scaleratio=1,
                constrain='domain',
                title="UMAP2"
            ),
            
            # **IDENTICAL LEGEND LAYOUT** as base image
            legend=get_standardized_legend_layout(),
            showlegend=True,
            
            # Fixed size (same as base)
            width=1600,
            height=1200,
            autosize=False,
            
            # Same margins as base
            margin=dict(l=80, r=200, t=100, b=80)
        )
        
        # Customize markers (same as base)
        fig.update_traces(
            marker=dict(
                size=14, 
                opacity=0.8,
                line=dict(width=0.5, color='white')
            )
        )
        
        # Create method-specific output directory
        method_dir = output_dir / method.lower()
        method_dir.mkdir(exist_ok=True)
        
        # Save plot with standardized legend
        img_path = method_dir / "probability_intensity.png"
        fig.write_image(
            str(img_path),
            width=1600,
            height=1200,
            scale=2
        )
        
        # Save data for debugging
        data_path = method_dir / "probability_data.csv"
        merged_df.to_csv(data_path, index=False)
        
        print(f"Saved {method} probability plot (standard legend): {img_path}")
        print(f"Saved {method} probability data: {data_path}")
    
    print(f"\nAll probability plots (standard legend) saved to: {output_dir}")

if __name__ == "__main__":
    config = load_config()
    create_probability_plots_standard_legend(config)
