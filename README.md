# UMAP Interactive Overlay Viewer 🎯

A comprehensive toolkit for creating perfectly aligned, interactive UMAP visualizations with prediction accuracy and probability intensity overlays. This tool addresses the critical challenge of visual misalignment in multi-layer UMAP plots, ensuring pixel-perfect overlay accuracy for scientific analysis.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/plotly-5.0+-green.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🚀 Key Strengths

### Perfect Alignment Technology
- **Zero-offset coordinate alignment** - Eliminates visual drift between base and overlay images
- **Standardized legend positioning** - Ensures identical layout across all visualizations
- **Data cleaning pipeline** - Removes NaN/Inf values that cause density artifacts
- **Automatic dimension validation** - Guarantees consistent image sizing

### Interactive Analysis
- **Real-time overlay switching** - Seamlessly toggle between accuracy and probability views
- **Multi-method comparison** - Compare PT10, CV10, Intersection, and All feature selection methods
- **Adjustable transparency** - Fine-tune overlay opacity for optimal visualization
- **Manual alignment controls** - Optional fine-tuning sliders for precision adjustment

### Scientific Rigor
- **Coordinate system validation** - Built-in checks ensure mathematical consistency
- **Quality control metrics** - Automated validation of alignment accuracy
- **Reproducible pipeline** - Standardized workflow for consistent results
- **Debug-friendly outputs** - Comprehensive logging and diagnostic tools

## 📋 Prerequisites

### Required Python Packages
```bash
pip install plotly pandas numpy matplotlib pillow pyyaml umap-learn
```
or
```bash
pip install requirements.txt
```

## 📁 Required Input Files

### Essential Files
1. **`config.yaml`** - Configuration file with paths and highlight settings
   ```yaml
   paths:
     base_dir: "umap_results_2"
     detailed_results: "detailed_results_run_11.csv"
   
   umap:
     min_dist: 0.1
     n_neighbors: 100
   
   columns:
     color_column: "event"
   
   highlight:
     id_column: "id"
     ids: [363, 364, 365, ...]  # IDs to highlight
     highlight_color: "#FF0000"
     default_color: "#1f77b4"
   ```

2. **`detailed_results_run_XX.csv`** - Prediction results with columns:
   - `Feature_Selection_Method`: Method name (All, PT10, CV10, Intersection)
   - `Predictions`: Array of predictions per sample
   - `Probabilities`: Array of prediction probabilities
   - `y_true`: Array of true labels

3. **UMAP embedding data** - Pre-computed UMAP coordinates
   - Before the very first run of the slider, you can run 01_umaps_better3.py
   - You can change the hyperparameter values of UMAP from the config file

## 🔄 Complete Workflow

### Option 1: Automated Pipeline (Recommended)
```bash
# Run complete pipeline from scratch
./run_full_pipeline.bat
```

### Option 2: Manual Step-by-Step
```bash
# 1. Generate base analysis
python 02_highlight_analysis.py

# 2. Create standardized base image
python 02_highlight_standard_legend.py

# 3. Generate accuracy overlays
python 03_pred_acc_standard_legend.py

# 4. Generate probability overlays
python 04_probabilities_standard_legend.py

# 5. Validate alignment
python check_alignment.py

# 6. Launch interactive viewer
python 05_enhanced_slider_viewer.py
```

## 🎮 Interactive Viewer Features

### Controls
- **Feature Selection Checkboxes**: Toggle between All, PT10, CV10, Intersection methods
- **Overlay Type Radio Buttons**: Switch between Accuracy and Intensity visualizations
- **Opacity Slider**: Adjust overlay transparency (0-100%)
- **Alignment Controls**: Optional manual fine-tuning (enable with `show_alignment_controls=True`)

### Visualizations
- **Base Layer**: Highlighted subjects on UMAP embedding
- **Accuracy Overlay**: Correct (blue) vs Incorrect (red) predictions for highlighted subjects
- **Intensity Overlay**: Probability intensity with color-coded confidence levels

## 📊 Output Structure

```
umap_results_2/min_dist_0.1_nn_100/
├── highlight_analysis_standard_legend/
│   ├── highlighted_subjects.png          # Base image with standardized legend
│   └── combined_embedding_with_metadata.csv
├── prediction_accuracy_standard_legend/
│   ├── pt10/
│   │   ├── prediction_accuracy.png       # PT10 accuracy overlay
│   │   └── accuracy_data.csv            # Coordinate data for debugging
│   ├── cv10/prediction_accuracy.png
│   └── intersection/prediction_accuracy.png
└── probability_intensity_standard_legend/
    ├── all/probability_intensity.png     # Probability overlays
    ├── pt10/probability_intensity.png
    ├── cv10/probability_intensity.png
    └── intersection/probability_intensity.png
```

## 🔧 Quality Control & Validation

### Automated Checks
- **Coordinate Alignment**: Verifies 0.000 offset between all datasets
- **Dimension Consistency**: Ensures all images are 3200×2400 pixels
- **Data Integrity**: Validates same sample counts across methods
- **Legend Standardization**: Confirms identical legend positioning

### Diagnostic Tools
```bash
# Check coordinate alignment (should show 0.000 offset)
python check_alignment.py

# Verify image dimensions match
python check_image_sizes.py

# Generate visual comparison plots
python visual_alignment_check.py
```

## 🎯 Use Cases & Applications

### Machine Learning Model Evaluation
- **Multi-method comparison**: Compare feature selection strategies side-by-side
- **Prediction confidence analysis**: Visualize model uncertainty across embedding space
- **Error pattern identification**: Identify regions of systematic prediction errors

### Biomedical Research
- **Patient outcome prediction**: Visualize ICU mortality predictions with clinical context
- **Feature importance analysis**: Compare different clinical feature sets (PT10, CV10, etc.)
- **Cohort analysis**: Highlight specific patient populations for detailed study

### General Data Science
- **Dimensionality reduction QA**: Validate UMAP embeddings maintain prediction patterns
- **Model interpretability**: Understand spatial patterns in high-dimensional predictions
- **Publication-ready visualizations**: Generate aligned plots for scientific manuscripts

## 🛠️ Troubleshooting

### Common Issues & Solutions

**Alignment Problems**
```bash
# Check coordinate consistency
python check_alignment.py
# Expected: All offsets should be 0.000
```

**Size Mismatches**
```bash
# Verify image dimensions
python check_image_sizes.py
# Expected: All images should be same size
```

**Missing Overlays**
- Ensure `detailed_results_run_XX.csv` contains all required methods
- Check `config.yaml` paths are correct
- Verify highlight IDs exist in the dataset

**Performance Issues**
- For large datasets (>10k points), consider subsampling
- Enable GPU acceleration for UMAP computation
- Increase system RAM for better performance

### Debug Mode
Enable detailed logging and alignment controls:
```python
viewer = UMAPViewer(show_alignment_controls=True)
```

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional overlay types (confusion matrices, feature importance)
- 3D UMAP support
- Custom color schemes and themes
- Performance optimizations for large datasets

### Development Setup
```bash
git clone https://github.com/your-repo/umap-interactive-overlay
cd umap-interactive-overlay
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the excellent [UMAP](https://umap-learn.readthedocs.io/) dimensionality reduction library
- Interactive visualizations powered by [Plotly](https://plotly.com/)
- Cluster Analysis in the early steps with [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)
- Inspired by the need for rigorous alignment in multi-layer scientific visualizations

## 📞 Documentation

- **Documentation**: See `PIPELINE_README.md` for detailed technical documentation

---

