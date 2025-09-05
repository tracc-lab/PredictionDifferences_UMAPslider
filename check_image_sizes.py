import yaml
from pathlib import Path
from PIL import Image
import os

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def check_image_sizes():
    """Check the actual dimensions of all image files"""
    config = load_config()
    base_dir = Path(config['paths']['base_dir']) / f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}"
    
    print("=== Image Size Check ===\n")
    
    # Check base image
    base_path = base_dir / "highlight_analysis/highlighted_subjects.png"
    if base_path.exists():
        with Image.open(base_path) as img:
            print(f"Base image: {img.size} ({img.size[0]}x{img.size[1]}) - {os.path.getsize(base_path)/1024:.1f}KB")
            base_size = img.size
    else:
        print(f"Base image not found: {base_path}")
        return
    
    print()
    
    # Check accuracy overlays
    print("ACCURACY OVERLAYS:")
    for method in ['all', 'pt10', 'cv10', 'intersection']:
        # Check aligned version first
        aligned_path = base_dir / "prediction_accuracy_aligned" / method / "prediction_accuracy.png"
        original_path = base_dir / "prediction_accuracy" / method / "prediction_accuracy.png"
        
        for version, path in [("aligned", aligned_path), ("original", original_path)]:
            if path.exists():
                with Image.open(path) as img:
                    size_match = "✓" if img.size == base_size else "✗"
                    print(f"  {method.upper()} ({version}): {img.size} {size_match} - {os.path.getsize(path)/1024:.1f}KB")
                    if img.size != base_size:
                        print(f"    Size difference: {img.size[0] - base_size[0]}x{img.size[1] - base_size[1]}")
            else:
                print(f"  {method.upper()} ({version}): NOT FOUND")
    
    print()
    
    # Check probability overlays
    print("PROBABILITY OVERLAYS:")
    for method in ['all', 'pt10', 'cv10', 'intersection']:
        # Check aligned version first
        aligned_path = base_dir / "probability_intensity_aligned" / method / "probability_intensity.png"
        original_path = base_dir / "probability_intensity" / method / "probability_intensity.png"
        
        for version, path in [("aligned", aligned_path), ("original", original_path)]:
            if path.exists():
                with Image.open(path) as img:
                    size_match = "✓" if img.size == base_size else "✗"
                    print(f"  {method.upper()} ({version}): {img.size} {size_match} - {os.path.getsize(path)/1024:.1f}KB")
                    if img.size != base_size:
                        print(f"    Size difference: {img.size[0] - base_size[0]}x{img.size[1] - base_size[1]}")
            else:
                print(f"  {method.upper()} ({version}): NOT FOUND")
    
    print()
    print("Legend: ✓ = Same size as base, ✗ = Different size")

def resize_all_images_to_match():
    """Resize all images to match the base image size"""
    config = load_config()
    base_dir = Path(config['paths']['base_dir']) / f"min_dist_{config['umap']['min_dist']}_nn_{config['umap']['n_neighbors']}"
    
    # Get base image size
    base_path = base_dir / "highlight_analysis/highlighted_subjects.png"
    if not base_path.exists():
        print("Base image not found!")
        return
    
    with Image.open(base_path) as base_img:
        target_size = base_img.size
        print(f"Target size: {target_size}")
    
    # Resize all overlay images
    image_paths = []
    
    # Add accuracy overlays
    for method in ['all', 'pt10', 'cv10', 'intersection']:
        for folder in ["prediction_accuracy_aligned", "prediction_accuracy"]:
            path = base_dir / folder / method / "prediction_accuracy.png"
            if path.exists():
                image_paths.append(path)
    
    # Add probability overlays
    for method in ['all', 'pt10', 'cv10', 'intersection']:
        for folder in ["probability_intensity_aligned", "probability_intensity"]:
            path = base_dir / folder / method / "probability_intensity.png"
            if path.exists():
                image_paths.append(path)
    
    print(f"\nResizing {len(image_paths)} images to {target_size}...")
    
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                if img.size != target_size:
                    print(f"Resizing {img_path.name}: {img.size} -> {target_size}")
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized_img.save(img_path)
                else:
                    print(f"Already correct size: {img_path.name}")
        except Exception as e:
            print(f"Error resizing {img_path}: {e}")
    
    print("Resize complete!")

if __name__ == "__main__":
    print("1. Checking current image sizes...")
    check_image_sizes()
    
    print("\n" + "="*50)
    response = input("\nDo you want to resize all images to match the base image? (y/n): ")
    
    if response.lower() == 'y':
        print("\n2. Resizing images...")
        resize_all_images_to_match()
        print("\n3. Checking sizes after resize...")
        check_image_sizes()
    else:
        print("Skipping resize.")
