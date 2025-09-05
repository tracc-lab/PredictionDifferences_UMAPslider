# UMAP Overlay Alignment Pipeline
# Complete workflow for generating perfectly aligned UMAP visualizations

## Prerequisites
- Python environment with: plotly, pandas, numpy, matplotlib, PIL, yaml
- config.yaml file with proper paths and highlight settings
- detailed_results_run_XX.csv file with prediction results

## Full Pipeline Steps

### 1. Initial Data Processing
```bash
# Generate initial UMAP embeddings (if needed)
python 01_umaps_better3.py
```

### 2. Base Image Generation
```bash
# Create base highlighted subjects analysis
python 02_highlight_analysis.py

# Generate base image with standardized legend for perfect alignment
python 02_highlight_standard_legend.py
```

### 3. Overlay Generation (Standardized Legends)
```bash
# Generate accuracy overlays with standardized legends
python 03_pred_acc_standard_legend.py

# Generate probability intensity overlays with standardized legends  
python 04_probabilities_standard_legend.py
```

### 4. Quality Control & Validation
```bash
# Check coordinate alignment across all images
python check_alignment.py

# Verify image dimensions match
python check_image_sizes.py
```

### 5. Interactive Viewer
```bash
# Launch the enhanced slider viewer
python 05_enhanced_slider_viewer.py
```

## Alternative: No-Legend Pipeline (if you prefer minimal overlays)
```bash
# Base image without legend
python 02_highlight_no_legend.py

# Overlays without legends
python 03_pred_acc_no_legend.py
python 04_probabilities_no_legend.py

# Launch viewer
python 05_enhanced_slider_viewer.py
```

## Automated Full Pipeline

```bash
# Run complete standardized legend pipeline
test_standard_legends.bat
```
Note: this file does not include the generation of initial UMAP embeddings. That is because each dataset has different requirements and different parameter values may be needed. This is a trial and error process, unlike the automated pipeline in the batch file.

## Key Files Generated:
- `highlight_analysis_standard_legend/highlighted_subjects.png` - Base image
- `prediction_accuracy_standard_legend/[method]/prediction_accuracy.png` - Accuracy overlays
- `probability_intensity_standard_legend/[method]/probability_intensity.png` - Probability overlays
- `*_data.csv` files - Coordinate data for debugging

## Expected Results:
✅ Perfect pixel-level alignment across all overlays
✅ Consistent legend positioning and sizing
✅ Clean data (no NaN/Inf values causing density issues)
✅ Interactive slider viewer with smooth overlay switching
✅ Auto-alignment detection with manual fine-tuning controls

## Troubleshooting:
- If alignment issues persist: Run `check_alignment.py` to verify coordinate consistency
- If images are different sizes: Run `check_image_sizes.py` to identify dimension mismatches
- For manual adjustment: Enable alignment controls in viewer with `show_alignment_controls=True`
