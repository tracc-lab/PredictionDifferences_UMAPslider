import yaml
import pandas as pd
import numpy as np
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

def create_base_image_with_standard_legend(config):
    """Create base highlighted subjects image with standardized legend"""
    
    # Get base directory
    base_dir = Path(config['paths']['base_dir']) / f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}"
    
    # Load combined data
    combined_path = base_dir / "highlight_analysis/combined_embedding_with_metadata.csv"
    if not combined_path.exists():
        print("Error: Combined data not found. Run 02_highlight_analysis.py first.")
        return
    
    df = pd.read_csv(combined_path)
    
    # Clean data to remove NaN/Inf values (same as overlays)
    print(f"Before cleaning: {len(df)} rows")
    df = df.dropna(subset=['UMAP1', 'UMAP2'])
    df = df[np.isfinite(df['UMAP1']) & np.isfinite(df['UMAP2'])]
    print(f"After cleaning: {len(df)} rows")
    
    # Get coordinate ranges
    coord_ranges = get_reference_coordinate_ranges(config)
    
    # Create highlight mask
    highlight_mask = df[config['highlight']['id_column']].isin(config['highlight']['ids'])
    
    # Create categorical color column for consistent legend
    df['Legend_Category'] = 'Regular Points'
    df.loc[highlight_mask, 'Legend_Category'] = 'Highlighted Subjects'
    
    print(f"Highlighted points: {highlight_mask.sum()}")
    
    # Create plot with standardized legend
    fig = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='Legend_Category',
        color_discrete_map={
            'Regular Points': config['highlight']['default_color'],
            'Highlighted Subjects': config['highlight']['highlight_color']
        },
        title="UMAP: Highlighted Subjects"
    )
    
    # Apply standardized layout
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
        
        # **STANDARDIZED LEGEND** - same position/size for all images
        legend=get_standardized_legend_layout(),
        showlegend=True,
        
        # Fixed size
        width=1600,
        height=1200,
        autosize=False,
        
        # Standard margins to accommodate legend
        margin=dict(l=80, r=200, t=100, b=80)
    )
    
    # Customize markers
    fig.update_traces(
        marker=dict(
            size=14, 
            opacity=0.7,
            line=dict(width=0.5, color='white')
        )
    )
    
    # Create output directory
    output_dir = base_dir / "highlight_analysis_standard_legend"
    output_dir.mkdir(exist_ok=True)
    
    # Save base image with standardized legend
    base_path = output_dir / "highlighted_subjects.png"
    fig.write_image(
        str(base_path),
        width=1600,
        height=1200,
        scale=2
    )
    
    print(f"Created base image (standard legend): {base_path}")
    print(f"Image dimensions: 3200x2400 pixels (1600x1200 @ scale=2)")
    print(f"Coordinate ranges: X=[{coord_ranges['x_min']:.2f}, {coord_ranges['x_max']:.2f}], Y=[{coord_ranges['y_min']:.2f}, {coord_ranges['y_max']:.2f}]")
    
    return base_path

if __name__ == "__main__":
    config = load_config()
    create_base_image_with_standard_legend(config)
