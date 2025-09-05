# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 23:11:21 2025

@author: Asus
"""
import os
import yaml
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button
from PIL import Image
from pathlib import Path

# Configuration
IMAGE_SIZE = (1200, 900)
DPI = 100

class UMAPViewer:
    def __init__(self, show_alignment_controls=False):
        self.show_alignment_controls = show_alignment_controls
        self.config = load_config()
        self.base_dir = self.get_base_dir()
        self.fig, self.ax = plt.subplots(figsize=(12, 9), dpi=DPI)
        self.overlays = {}
        self.active_overlays = []
        
        # UI Setup
        self.setup_interface()
        self.load_images()
        self.create_controls()
        
    def get_image_extent(self):
        """Get consistent image extent for proper alignment"""
        return [0, IMAGE_SIZE[0], IMAGE_SIZE[1], 0]  # [left, right, bottom, top]
    
    def get_base_dir(self):
        umap = self.config['umap']
        return Path(self.config['paths']['base_dir']) / \
            f"min_dist_{umap['min_dist']}_nn_{umap['n_neighbors']}"
    
    def setup_interface(self):
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Set consistent axis limits for better alignment
        self.ax.set_xlim(0, IMAGE_SIZE[0])
        self.ax.set_ylim(IMAGE_SIZE[1], 0)  # Invert Y axis for image coordinates
        
        # Ensure aspect ratio is equal to prevent distortion
        self.ax.set_aspect('equal', adjustable='box')
    
    def load_images(self):
        # Verify image alignment before loading
        self.verify_image_alignment()
        
        # Load base image
        base_path = self.ensure_base_image_exists()
        self.base_image_array = load_image(base_path)
        extent = self.get_image_extent()
        self.base_img = self.ax.imshow(
            self.base_image_array, 
            extent=extent,
            interpolation='nearest',  # Prevent interpolation artifacts
            origin='upper'
        )
        
        # Load all overlays (skip 'All' for accuracy since it doesn't exist)
        for feature_set in ['PT10', 'CV10', 'Intersection']:
            self.load_feature_set(feature_set)
        
        # Load All only for intensity overlays if they exist
        self.load_feature_set_intensity_only('All')
    
    def load_feature_set(self, feature_set):
        """Load both accuracy and intensity for a feature set"""
        # Try standardized legend versions first, then aligned, then original
        set_dir_acc = self.base_dir / "prediction_accuracy_standard_legend" / feature_set.lower()
        set_dir_prob = self.base_dir / "probability_intensity_standard_legend" / feature_set.lower()
        
        # Fall back to aligned versions
        if not set_dir_acc.exists():
            set_dir_acc = self.base_dir / "prediction_accuracy_aligned" / feature_set.lower()
        if not set_dir_prob.exists():
            set_dir_prob = self.base_dir / "probability_intensity_aligned" / feature_set.lower()
        
        # Fall back to original versions
        if not set_dir_acc.exists():
            set_dir_acc = self.base_dir / "prediction_accuracy" / feature_set.lower()
        if not set_dir_prob.exists():
            set_dir_prob = self.base_dir / "probability_intensity" / feature_set.lower()

        print("The searched directory is: ", set_dir_acc)
        
        # Get consistent extent for alignment
        extent = self.get_image_extent()
        
        # Accuracy overlay
        acc_path = set_dir_acc / "prediction_accuracy.png"
        if acc_path.exists():
            acc_img_array = load_image(acc_path)
            acc_img = self.ax.imshow(
                acc_img_array, 
                alpha=0, 
                extent=extent,
                interpolation='nearest',
                origin='upper'
            )
            self.overlays[f"{feature_set}_accuracy"] = {
                'type': 'accuracy',
                'feature_set': feature_set,
                'artist': acc_img
            }
        
        # Intensity overlay
        int_path = set_dir_prob / "probability_intensity.png"
        if int_path.exists():
            int_img_array = load_image(int_path)
            int_img = self.ax.imshow(
                int_img_array, 
                alpha=0, 
                extent=extent,
                interpolation='nearest',
                origin='upper'
            )
            self.overlays[f"{feature_set}_intensity"] = {
                'type': 'intensity', 
                'feature_set': feature_set,
                'artist': int_img
            }
        
        # Debugging
        print(f"Loading overlay for {feature_set}")
        print(f"Accuracy path: {acc_path.exists()}, Intensity path: {int_path.exists()}")
    
    def load_feature_set_intensity_only(self, feature_set):
        """Load only intensity overlay for a feature set (for 'All' which has no accuracy overlay)"""
        # Try standardized legend versions first, then aligned, then original
        set_dir_prob = self.base_dir / "probability_intensity_standard_legend" / feature_set.lower()
        
        # Fall back to aligned versions
        if not set_dir_prob.exists():
            set_dir_prob = self.base_dir / "probability_intensity_aligned" / feature_set.lower()
        
        # Fall back to original versions  
        if not set_dir_prob.exists():
            set_dir_prob = self.base_dir / "probability_intensity" / feature_set.lower()

        print("The searched intensity directory is: ", set_dir_prob)
        
        # Get consistent extent for alignment
        extent = self.get_image_extent()
        
        # Intensity overlay only
        int_path = set_dir_prob / "probability_intensity.png"
        if int_path.exists():
            int_img_array = load_image(int_path)
            int_img = self.ax.imshow(
                int_img_array, 
                alpha=0, 
                extent=extent,
                interpolation='nearest',
                origin='upper'
            )
            self.overlays[f"{feature_set}_intensity"] = {
                'type': 'intensity', 
                'feature_set': feature_set,
                'artist': int_img
            }
            print(f"Loaded intensity overlay for {feature_set}")
        else:
            print(f"Intensity overlay not found: {int_path}")
    
    def create_controls(self):
        # Feature Set Selector
        self.feature_ax = plt.axes([0.01, 0.4, 0.15, 0.15])
        self.feature_selector = CheckButtons(
            self.feature_ax,
            ['All', 'PT10', 'CV10', 'Intersection'],
            actives=[True, False, False, False]
        )
        
        # Overlay Type Selector
        #[    ,     dist between the 2 relative to the plot, how squished the boxes are, ]
        self.type_ax = plt.axes([0.01, 0.2, 0.1, 0.15])
        self.type_selector = RadioButtons(
            self.type_ax,
            ('Accuracy', 'Intensity'),
            active=0
        )
        
        # Alpha Slider
        self.slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(
            self.slider_ax,
            'Overlay Opacity',
            0, 1,
            valinit=0.5
        )
        
        # Connect events
        self.feature_selector.on_clicked(self.update_feature_sets)
        self.type_selector.on_clicked(self.update_overlay_type)
        self.slider.on_changed(self.update_alpha)
        
        # Add alignment controls (optional)
        self.add_alignment_controls()
        
        # Initial state
        self.current_type = 'accuracy'
        self.update_display()
    
    def update_feature_sets(self, label):
        self.update_display()
    
    def update_overlay_type(self, label):
        self.current_type = label.lower()
        self.update_display()
    
    def update_alpha(self, val):
        for overlay in self.active_overlays:
            overlay['artist'].set_alpha(val)
        self.fig.canvas.draw_idle()
    
    def update_display(self):
        # Get selected feature sets
        selected_features = [
            label for label, active in zip(
                ['All', 'PT10', 'CV10', 'Intersection'],
                self.feature_selector.get_status()
            ) if active
        ]
        
        # Hide all overlays first
        for overlay in self.overlays.values():
            overlay['artist'].set_alpha(0)
        
        # Show selected ones
        self.active_overlays = [
            overlay for key, overlay in self.overlays.items()
            if (overlay['feature_set'] in selected_features and
                overlay['type'] == self.current_type)
        ]
        
        # Apply current alpha
        self.update_alpha(self.slider.val)
        
        # Update title
        self.ax.set_title(
            f"UMAP Viewer\n"
            f"Showing: {self.current_type.capitalize()} "
            f"for {', '.join(selected_features)}",
            pad=20
        )

        # Debugging
        print(f"\n--- Updating Display ---")
        print(f"Selected type: {self.current_type}")
        print(f"Selected features: {selected_features}")
        print(f"Active overlays: {[o['feature_set'] + '_' + o['type'] for o in self.active_overlays]}")
    
    def ensure_base_image_exists(self):
        """Ensure the base highlighted subjects image exists with proper alignment"""
        # Try standardized legend version first
        base_path = self.base_dir / "highlight_analysis_standard_legend/highlighted_subjects.png"
        
        # Check if we need to regenerate for standardized legend alignment
        regenerate = False
        if not base_path.exists():
            regenerate = True
            print("Standardized legend base image not found, creating it...")
        else:
            # Check if standardized legend overlays exist - if they do, we need standardized legend base too
            standard_acc_dir = self.base_dir / "prediction_accuracy_standard_legend"
            if standard_acc_dir.exists():
                print("Standardized legend overlays detected, using standardized legend base image...")
        
        if regenerate:
            try:
                return self.create_base_highlight_image_standard_legend()
            except Exception as e:
                print(f"Failed to create standardized legend base image: {e}")
                # Fall back to regular base image
                fallback_path = self.base_dir / "highlight_analysis/highlighted_subjects.png"
                if fallback_path.exists():
                    print("Using regular base image as fallback")
                    return fallback_path
                else:
                    raise FileNotFoundError(f"Base image not found and couldn't be created: {base_path}")
        return base_path
    
    def create_base_highlight_image_standard_legend(self):
        """Create base highlighted subjects image with standardized legend matching overlays"""
        base_path = self.base_dir / "highlight_analysis_standard_legend/highlighted_subjects.png"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load combined data
        combined_path = self.base_dir / "highlight_analysis/combined_embedding_with_metadata.csv"
        if not combined_path.exists():
            print("Combined data not found, running highlight analysis...")
            import subprocess
            import sys
            subprocess.run([sys.executable, "02_highlight_analysis.py"], check=True)
        
        # Create the highlight plot with standardized legend
        import subprocess
        import sys
        subprocess.run([sys.executable, "02_highlight_standard_legend.py"], check=True)
        
        print(f"Created standardized legend base image: {base_path}")
        return base_path
    
    def create_base_highlight_image(self):
        """Create the base highlighted subjects image if it doesn't exist"""
        base_path = self.base_dir / "highlight_analysis/highlighted_subjects.png"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load combined data
        combined_path = self.base_dir / "highlight_analysis/combined_embedding_with_metadata.csv"
        if not combined_path.exists():
            print("Combined data not found, running highlight analysis...")
            import subprocess
            import sys
            subprocess.run([sys.executable, "02_highlight_analysis.py"], check=True)
        
        # Create the highlight plot with consistent coordinate ranges
        import pandas as pd
        import plotly.express as px
        
        df = pd.read_csv(combined_path)
        
        # Get coordinate ranges from the 'All' dataset
        coord_ranges = self.get_umap_coordinate_ranges()
        
        # Create highlight mask
        highlight_mask = df[self.config['highlight']['id_column']].isin(self.config['highlight']['ids'])
        df['color'] = self.config['highlight']['default_color']
        df.loc[highlight_mask, 'color'] = self.config['highlight']['highlight_color']
        
        # Create plot
        fig = px.scatter(
            df,
            x='UMAP1',
            y='UMAP2',
            color='color',
            color_discrete_map="identity",
            title="Highlighted Subjects"
        )
        
        # **KEY FIX: Set the same coordinate ranges as overlays**
        fig.update_layout(
            xaxis=dict(
                range=[coord_ranges['x_min'], coord_ranges['x_max']],
                constrain='domain'
            ),
            yaxis=dict(
                range=[coord_ranges['y_min'], coord_ranges['y_max']],
                scaleanchor="x",
                scaleratio=1,
                constrain='domain'
            ),
            width=1600,
            height=1200,
            autosize=False
        )
        
        # Customize and save
        fig.update_traces(marker=dict(size=14, opacity=0.7))
        fig.write_image(
            str(base_path),
            width=1600,
            height=1200,
            scale=2
        )
        print(f"Created base image with aligned coordinates: {base_path}")
        return base_path
    
    def verify_image_alignment(self):
        """Verify that all images have consistent dimensions and can be aligned"""
        # First, ensure aligned images exist
        self.ensure_aligned_images_exist()
        
        base_path = self.ensure_base_image_exists()
        base_img = Image.open(base_path)
        print(f"Base image size: {base_img.size}")
        
        # Check overlay images (prefer aligned versions)
        for feature_set in ['All', 'PT10', 'CV10', 'Intersection']:
            for img_type in ['accuracy', 'intensity']:
                # Try aligned version first
                if img_type == 'accuracy':
                    img_path = self.base_dir / "prediction_accuracy_aligned" / feature_set.lower() / "prediction_accuracy.png"
                    if not img_path.exists():
                        img_path = self.base_dir / "prediction_accuracy" / feature_set.lower() / "prediction_accuracy.png"
                else:
                    img_path = self.base_dir / "probability_intensity_aligned" / feature_set.lower() / "probability_intensity.png"
                    if not img_path.exists():
                        img_path = self.base_dir / "probability_intensity" / feature_set.lower() / "probability_intensity.png"
                
                if img_path.exists():
                    overlay_img = Image.open(img_path)
                    if overlay_img.size != base_img.size:
                        print(f"Warning: Size mismatch for {img_path}")
                        print(f"  Base: {base_img.size}, Overlay: {overlay_img.size}")
                    else:
                        print(f"✓ Size match for {feature_set} {img_type}: {overlay_img.size}")
                else:
                    print(f"Missing image: {img_path}")

    def add_alignment_controls(self):
        """Add fine-tuning controls for manual alignment adjustment"""
        # Add offset controls if needed
        self.offset_x = 0
        self.offset_y = 0
        
        # Detect automatic offset
        detected_offset = self.detect_coordinate_offset()
        
        # Fine alignment sliders (optional, for debugging)
        if self.show_alignment_controls:
            self.x_offset_ax = plt.axes([0.01, 0.05, 0.1, 0.02])
            self.y_offset_ax = plt.axes([0.01, 0.02, 0.1, 0.02])
            
            # Auto-apply detected offset as initial values
            initial_x = detected_offset['x'] if detected_offset else 0
            initial_y = detected_offset['y'] if detected_offset else 0
            
            self.x_slider = Slider(self.x_offset_ax, 'X Offset', -20, 20, valinit=initial_x)
            self.y_slider = Slider(self.y_offset_ax, 'Y Offset', -20, 20, valinit=initial_y)
            
            self.x_slider.on_changed(self.adjust_x_offset)
            self.y_slider.on_changed(self.adjust_y_offset)
            
            # Apply initial offset if detected
            if detected_offset:
                self.offset_x = initial_x
                self.offset_y = initial_y
                self.update_overlay_positions()
                print(f"Auto-applied detected offset: X={initial_x:.3f}, Y={initial_y:.3f}")
                
            # Add auto-align button
            self.auto_align_ax = plt.axes([0.12, 0.03, 0.05, 0.03])
            self.auto_align_button = plt.Button(self.auto_align_ax, 'Auto\nAlign')
            self.auto_align_button.on_clicked(self.auto_align_overlays)
    
    def adjust_x_offset(self, val):
        """Adjust X offset for overlay alignment"""
        self.offset_x = val
        self.update_overlay_positions()
    
    def adjust_y_offset(self, val):
        """Adjust Y offset for overlay alignment"""
        self.offset_y = val
        self.update_overlay_positions()
    
    def update_overlay_positions(self):
        """Update overlay positions with current offsets"""
        extent = self.get_image_extent()
        adjusted_extent = [
            extent[0] + self.offset_x,
            extent[1] + self.offset_x,
            extent[2] + self.offset_y,
            extent[3] + self.offset_y
        ]
        
        for overlay in self.overlays.values():
            overlay['artist'].set_extent(adjusted_extent)
        
        self.fig.canvas.draw_idle()
    
    def get_umap_coordinate_ranges(self):
        """Get the coordinate ranges from the 'All' dataset to use as reference"""
        # Load the 'All' dataset embedding to get coordinate ranges
        all_embedding_path = (
            self.base_dir / "embeddings" / 
            f"umap_2d_all_mindist{self.config['umap']['min_dist']}_nn{self.config['umap']['n_neighbors']}_coloredby_{self.config['columns']['color_column'].replace(' ', '_')}.csv"
        )
        
        if all_embedding_path.exists():
            import pandas as pd
            df = pd.read_csv(all_embedding_path)
            
            # Get ranges with some padding
            x_min, x_max = df['UMAP1'].min(), df['UMAP1'].max()
            y_min, y_max = df['UMAP2'].min(), df['UMAP2'].max()
            
            # Add padding (10% of range)
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            
            ranges = {
                'x_min': x_min - x_padding,
                'x_max': x_max + x_padding,
                'y_min': y_min - y_padding,
                'y_max': y_max + y_padding
            }
            
            print(f"UMAP coordinate ranges: X=[{ranges['x_min']:.2f}, {ranges['x_max']:.2f}], Y=[{ranges['y_min']:.2f}, {ranges['y_max']:.2f}]")
            return ranges
        else:
            print(f"Warning: All embedding not found at {all_embedding_path}")
            # Return default ranges
            return {'x_min': -15, 'x_max': 15, 'y_min': -15, 'y_max': 15}
    
    def ensure_aligned_images_exist(self):
        """Generate aligned images if they don't exist"""
        aligned_acc_dir = self.base_dir / "prediction_accuracy_aligned"
        aligned_prob_dir = self.base_dir / "probability_intensity_aligned"
        
        if not aligned_acc_dir.exists() or not aligned_prob_dir.exists():
            print("Aligned images not found, generating them...")
            try:
                # Generate aligned accuracy plots
                import subprocess
                import sys
                
                print("Generating aligned accuracy plots...")
                subprocess.run([sys.executable, "03_pred_acc_aligned.py"], check=True)
                
                print("Generating aligned probability plots...")
                subprocess.run([sys.executable, "04_probabilities_aligned.py"], check=True)
                
                print("Aligned images generated successfully!")
                
            except Exception as e:
                print(f"Failed to generate aligned images: {e}")
                print("Falling back to original images...")
    
    def detect_coordinate_offset(self):
        """Detect coordinate offset between base image and overlays"""
        try:
            # Load base coordinates
            combined_path = self.base_dir / "highlight_analysis/combined_embedding_with_metadata.csv"
            if not combined_path.exists():
                return None
                
            import pandas as pd
            base_df = pd.read_csv(combined_path)
            
            # Check PT10 as representative (since All accuracy doesn't exist)
            pt10_acc_path = self.base_dir / "prediction_accuracy_aligned/pt10/accuracy_data.csv"
            if not pt10_acc_path.exists():
                print("PT10 accuracy data not found for offset detection")
                return None
                
            overlay_df = pd.read_csv(pt10_acc_path)
            
            # Focus on highlighted points for better alignment detection
            highlight_ids = self.config['highlight']['ids']
            id_col = self.config['highlight']['id_column']
            
            # Get highlighted points from both datasets
            base_highlighted = base_df[base_df[id_col].isin(highlight_ids)]
            overlay_highlighted = overlay_df[overlay_df[id_col].isin(highlight_ids)]
            
            if len(base_highlighted) > 0 and len(overlay_highlighted) > 0:
                # Calculate offset using highlighted points (more accurate)
                offset_x = base_highlighted['UMAP1'].mean() - overlay_highlighted['UMAP1'].mean()
                offset_y = base_highlighted['UMAP2'].mean() - overlay_highlighted['UMAP2'].mean()
                
                print(f"Detected coordinate offset (using {len(base_highlighted)} highlighted points from PT10):")
                print(f"  X={offset_x:.3f}, Y={offset_y:.3f}")
            else:
                # Fallback to all points
                offset_x = base_df['UMAP1'].mean() - overlay_df['UMAP1'].mean()
                offset_y = base_df['UMAP2'].mean() - overlay_df['UMAP2'].mean()
                
                print(f"Detected coordinate offset (using all points from PT10):")
                print(f"  X={offset_x:.3f}, Y={offset_y:.3f}")
            
            if abs(offset_x) > 0.05 or abs(offset_y) > 0.05:
                print("Significant offset detected - will auto-apply")
                return {'x': offset_x, 'y': offset_y}
            else:
                print("No significant offset detected")
                return None
                
        except Exception as e:
            print(f"Could not detect coordinate offset: {e}")
            return None
    
    def auto_align_overlays(self, event):
        """Auto-align overlays using detected coordinate offset"""
        detected_offset = self.detect_coordinate_offset()
        if detected_offset:
            # Update sliders
            self.x_slider.set_val(detected_offset['x'])
            self.y_slider.set_val(detected_offset['y'])
            # The slider change events will automatically call adjust_x_offset and adjust_y_offset
            print(f"Auto-aligned overlays with offset: X={detected_offset['x']:.3f}, Y={detected_offset['y']:.3f}")
        else:
            print("No significant offset detected - no adjustment needed")
    
    # ...existing code...
def load_image(path):
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGBA")
    print(f"Loaded image {path} with original size: {img.size}")
    
    # Use high-quality resampling and ensure consistent dimensions
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Ensure consistent data type
    img_array = img_array.astype(np.uint8)
    
    print(f"Resized image {path} to: {IMAGE_SIZE}")
    return img_array

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Set show_alignment_controls=True if you need fine-tuning controls for alignment
    viewer = UMAPViewer(show_alignment_controls=True)  # Enable alignment controls for testing
    plt.show()