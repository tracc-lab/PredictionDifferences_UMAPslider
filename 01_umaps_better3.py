import os
import yaml
import pandas as pd
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
import plotly.express as px
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Load and validate configuration file"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Convert paths to Path objects
    config['paths']['base_dir'] = Path(config['paths']['base_dir'])
    config['paths']['input_data'] = Path(config['paths']['input_data'])
    
    return config

def setup_directory_structure(base_dir, min_dist, n_neighbors):
    """Create folder structure based on UMAP parameters"""
    param_dir = f"min_dist_{min_dist}_nn_{n_neighbors}"
    paths = {
        'base': base_dir / param_dir,
        '2d': base_dir / param_dir / '2d',
        '3d': base_dir / param_dir / '3d',
        'clusters': base_dir / param_dir / 'clusters',
        'clusters_2d': base_dir / param_dir / 'clusters' / '2d',
        'clusters_3d': base_dir / param_dir / 'clusters' / '3d',
        'embeddings': base_dir / param_dir / 'embeddings'
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths

def create_2d_umap(data, array_name, color_values, save_path, config):
    """Create and save 2D UMAP visualization"""
    umap_2d = UMAP(
        n_components=2,
        n_neighbors=config['umap']['n_neighbors'],
        min_dist=config['umap']['min_dist'],
        random_state=config['umap']['random_state']
    )
    embedding_2d = umap_2d.fit_transform(data)
    
    plt.figure(figsize=config['visualization']['figsize'])
    scatter = plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=color_values,
        cmap=config['visualization']['cmap'],
        alpha=config['visualization']['point_alpha']
    )
    
    UMAP.plot.connectivity(embedding_2d, edge_bundling='hammer')
    
    if pd.api.types.is_numeric_dtype(color_values):
        plt.colorbar(scatter, label=config['columns']['color_column'])
    else:
        plt.legend(*scatter.legend_elements(), title=config['columns']['color_column'])
    
    plt.title(f'2D UMAP of {array_name}\nColored by {config["columns"]["color_column"]}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    base_filename = f"umap_2d_{array_name.lower().replace(' ', '_')}_coloredby_{config['columns']['color_column'].replace(' ', '_')}"
    plt.savefig(
        save_path['2d'] / f"{base_filename}.png",
        transparent=True, dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2']).to_csv(
        save_path['embeddings'] / f"{base_filename}.csv", index=False
    )
    
    return embedding_2d

def create_3d_umap(data, array_name, color_values, save_path, config):
    """Create and save 3D UMAP visualization"""
    umap_3d = UMAP(
        n_components=3,
        n_neighbors=config['umap']['n_neighbors'],
        min_dist=config['umap']['min_dist'],
        random_state=config['umap']['random_state']
    )
    embedding_3d = umap_3d.fit_transform(data)
    
    fig = px.scatter_3d(
        x=embedding_3d[:, 0], y=embedding_3d[:, 1], z=embedding_3d[:, 2],
        color=color_values,
        title=f"3D UMAP of {array_name}",
        labels={'color': config['columns']['color_column']}
    )
    
    base_filename = f"umap_3d_{array_name.lower().replace(' ', '_')}_coloredby_{config['columns']['color_column'].replace(' ', '_')}"
    fig.write_html(str(save_path['3d'] / f"{base_filename}.html"))
    fig.write_image(str(save_path['3d'] / f"{base_filename}.png"))
    
    pd.DataFrame(embedding_3d, columns=['UMAP1', 'UMAP2', 'UMAP3']).to_csv(
        save_path['embeddings'] / f"{base_filename}.csv", index=False
    )
    
    return embedding_3d

def compute_gini_impurity(cluster_labels, true_labels):
    """Compute mean Gini impurity across clusters"""
    clusters = np.unique(cluster_labels)
    gini_scores = []
    
    for cluster in clusters:
        if cluster == -1:
            continue
        mask = (cluster_labels == cluster)
        class_counts = np.bincount(true_labels[mask])
        proportions = class_counts / np.sum(class_counts)
        gini = 1 - np.sum(proportions ** 2)
        gini_scores.append(gini)
    
    return np.mean(gini_scores) if gini_scores else np.nan

def cluster_analysis(embedding, array_name, save_path, config, plot_type='2d', true_labels=None):
    """Perform clustering and save results"""
    clusterer = HDBSCAN(
        min_cluster_size=config['clustering']['min_cluster_size'],
        min_samples=config['clustering']['min_samples'],
        cluster_selection_epsilon=config['clustering']['cluster_selection_epsilon'],
        metric=config['clustering']['metric']
    )
    clusters = clusterer.fit_predict(embedding)
    
    metrics = {
        'n_clusters': len(np.unique(clusters)) - (1 if -1 in clusters else 0),
        'n_noise': np.sum(clusters == -1),
        'plot_type': plot_type,
        'silhouette': silhouette_score(embedding, clusters) if len(np.unique(clusters)) > 1 else np.nan
    }
    
    if true_labels is not None:
        metrics.update({
            'ARI': adjusted_rand_score(true_labels, clusters),
            'mean_gini': compute_gini_impurity(clusters, true_labels)
        })
    
    base_name = f"clusters_{plot_type}_{array_name.lower().replace(' ', '_')}"
    pd.DataFrame([metrics]).to_csv(
        save_path[f'clusters_{plot_type}'] / f"{base_name}_metrics.csv",
        index=False
    )
    
    labels_df = pd.DataFrame({'cluster': clusters})
    if true_labels is not None:
        labels_df['true_label'] = true_labels
    labels_df.to_csv(
        save_path[f'clusters_{plot_type}'] / f"{base_name}_labels.csv",
        index=False
    )
    
    if plot_type == '2d':
        fig = px.scatter(
            x=embedding[:, 0], y=embedding[:, 1],
            color=clusters,
            title=f"2D Clusters - {array_name}",
            labels={'color': 'Cluster'},
            color_continuous_scale=px.colors.qualitative.Alphabet
        )
    else:
        fig = px.scatter_3d(
            x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2],
            color=clusters,
            title=f"3D Clusters - {array_name}",
            labels={'color': 'Cluster'},
            color_continuous_scale=px.colors.qualitative.Alphabet
        )
        fig.update_layout(scene=dict(
            xaxis_title='UMAP1',
            yaxis_title='UMAP2',
            zaxis_title='UMAP3'
        ))
    
    fig.write_html(str(save_path[f'clusters_{plot_type}'] / f"{base_name}.html"))
    fig.write_image(str(save_path[f'clusters_{plot_type}'] / f"{base_name}.png"))
    
    return clusters, metrics

def main():
    config = load_config()
    
    # Load data
    df = pd.read_csv(
        config['paths']['input_data'],
        sep=config['paths']['separator']
    )
    color_values = df[config['columns']['color_column']]
    alt_color_values = df[config['columns']['alt_color_column']]
    
    # Create arrays from configured columns
    arrays = {
        name: df[cols].values
        for name, cols in config['columns']['features'].items()
    }
    
    # Setup directory structure
    paths = setup_directory_structure(
        config['paths']['base_dir'],
        config['umap']['min_dist'],
        config['umap']['n_neighbors']
    )
    
    # Process each array
    results = {}
    for name, data in arrays.items():
        print(f"\nProcessing {name}...")
        identifier = f"{name}_mindist{config['umap']['min_dist']}_nn{config['umap']['n_neighbors']}"
        
        # Generate UMAP projections
        umap_2d = create_2d_umap(data, identifier, color_values, paths, config)
        umap_3d = create_3d_umap(data, identifier, color_values, paths, config)
        
        # Cluster analysis
        clusters_2d, metrics_2d = cluster_analysis(
            umap_2d, identifier, paths, config, '2d', color_values
        )
        clusters_3d, metrics_3d = cluster_analysis(
            umap_3d, identifier, paths, config, '3d', color_values
        )
        
        print("2D Metrics:", metrics_2d)
        print("3D Metrics:", metrics_3d)
        
        # Save combined results
        cluster_df = pd.DataFrame({
            'UMAP1_2D': umap_2d[:, 0],
            'UMAP2_2D': umap_2d[:, 1],
            'Cluster_2D': clusters_2d,
            'UMAP1_3D': umap_3d[:, 0],
            'UMAP2_3D': umap_3d[:, 1],
            'UMAP3_3D': umap_3d[:, 2],
            'Cluster_3D': clusters_3d,
            config['columns']['color_column']: color_values,
            config['columns']['alt_color_column']: alt_color_values
        })
        cluster_df.to_csv(
            paths['clusters'] / f"cluster_assignments_{identifier}.csv",
            index=False
        )
        
        results[name] = {
            'embeddings': {'2d': umap_2d, '3d': umap_3d},
            'clusters': {'2d': clusters_2d, '3d': clusters_3d},
            'metrics': {'2d': metrics_2d, '3d': metrics_3d}
        }
    
    print(f"\nAnalysis complete. Results saved to: {paths['base']}")

if __name__ == "__main__":
    main()