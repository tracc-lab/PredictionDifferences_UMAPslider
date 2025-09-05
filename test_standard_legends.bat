@echo off
echo ===================================
echo  TESTING STANDARDIZED LEGEND ALIGNMENT
echo ===================================
echo.
echo This version keeps the legends but makes them identical
echo across all images so the plot areas align perfectly.
echo.
echo Generating standardized legend images...
echo.

echo 1. Creating base image with standard legend...
python 02_highlight_standard_legend.py
echo.

echo 2. Creating accuracy overlays with standard legends...
python 03_pred_acc_standard_legend.py  
echo.

echo 3. Creating probability overlays with standard legends...
python 04_probabilities_standard_legend.py
echo.

echo 4. Checking alignment...
python check_alignment.py
echo.

echo 5. Starting enhanced viewer with standard legend images...
echo   - All legends should be identical in size and position
echo   - Plot areas should align perfectly
echo   - No more "left shift" or border issues
echo.
python 05_enhanced_slider_viewer.py

pause
