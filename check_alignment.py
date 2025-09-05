import yaml
import pandas as pd
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def check_coordinate_alignment():
    """Check coordinate alignment between base image and overlays"""
    config = load_config()
    base_dir = Path(config['paths']['base_dir']) / f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}"
    
    print("=== Coordinate Alignment Check ===\n")
    
    # Load base coordinates
    combined_path = base_dir / "highlight_analysis/combined_embedding_with_metadata.csv"
    if combined_path.exists():
        base_df = pd.read_csv(combined_path)
        print(f"Base image coordinates:")
        print(f"  Shape: {base_df.shape}")
        print(f"  UMAP1 range: [{base_df['UMAP1'].min():.3f}, {base_df['UMAP1'].max():.3f}]")
        print(f"  UMAP2 range: [{base_df['UMAP2'].min():.3f}, {base_df['UMAP2'].max():.3f}]")
        print(f"  UMAP1 mean: {base_df['UMAP1'].mean():.3f}")
        print(f"  UMAP2 mean: {base_df['UMAP2'].mean():.3f}")
        
        # Check highlighted points
        highlight_ids = config['highlight']['ids']
        id_col = config['highlight']['id_column']
        base_highlighted = base_df[base_df[id_col].isin(highlight_ids)]
        print(f"  Highlighted points: {len(base_highlighted)}")
        if len(base_highlighted) > 0:
            print(f"  Highlighted UMAP1 mean: {base_highlighted['UMAP1'].mean():.3f}")
            print(f"  Highlighted UMAP2 mean: {base_highlighted['UMAP2'].mean():.3f}")
    else:
        print(f"Base coordinates not found: {combined_path}")
        return
    
    print()
    
    # Check overlay coordinates (skip 'all' since it doesn't exist as accuracy overlay)
    for method in ['pt10', 'cv10', 'intersection']:
        print(f"{method.upper()} accuracy overlay:")
        
        acc_path = base_dir / "prediction_accuracy_aligned" / method / "accuracy_data.csv"
        if acc_path.exists():
            overlay_df = pd.read_csv(acc_path)
            print(f"  Shape: {overlay_df.shape}")
            print(f"  UMAP1 range: [{overlay_df['UMAP1'].min():.3f}, {overlay_df['UMAP1'].max():.3f}]")
            print(f"  UMAP2 range: [{overlay_df['UMAP2'].min():.3f}, {overlay_df['UMAP2'].max():.3f}]")
            print(f"  UMAP1 mean: {overlay_df['UMAP1'].mean():.3f}")
            print(f"  UMAP2 mean: {overlay_df['UMAP2'].mean():.3f}")
            
            # Calculate offset from base
            offset_x = base_df['UMAP1'].mean() - overlay_df['UMAP1'].mean()
            offset_y = base_df['UMAP2'].mean() - overlay_df['UMAP2'].mean()
            print(f"  Offset from base: X={offset_x:.3f}, Y={offset_y:.3f}")
            
            # Check highlighted points
            overlay_highlighted = overlay_df[overlay_df[id_col].isin(highlight_ids)]
            print(f"  Highlighted points: {len(overlay_highlighted)}")
            if len(overlay_highlighted) > 0 and len(base_highlighted) > 0:
                h_offset_x = base_highlighted['UMAP1'].mean() - overlay_highlighted['UMAP1'].mean()
                h_offset_y = base_highlighted['UMAP2'].mean() - overlay_highlighted['UMAP2'].mean()
                print(f"  Highlighted offset: X={h_offset_x:.3f}, Y={h_offset_y:.3f}")
                
                # This is the key measurement for alignment!
                if abs(h_offset_x) > 0.1 or abs(h_offset_y) > 0.1:
                    print(f"  *** SIGNIFICANT MISALIGNMENT DETECTED ***")
                    print(f"  *** Suggested slider adjustment: X={h_offset_x:.3f}, Y={h_offset_y:.3f} ***")
        else:
            print(f"  Not found: {acc_path}")
        
        print()

if __name__ == "__main__":
    check_coordinate_alignment()
