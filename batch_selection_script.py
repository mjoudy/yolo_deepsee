#!/usr/bin/env python3
"""
Batch Selection Script for YOLO Dataset Preparation
Self-contained script for Google Colab usage with Google Drive
"""

import os
import shutil
import random
import pandas as pd
from PIL import Image
import yaml

# === DEFAULT CONFIGURATION ===
# These are defaults - you can override them when calling functions
DEFAULT_BASE_PATH = "/content/drive/MyDrive/DeepSeaProject/dataset_seanoe_101899"
DEFAULT_IMAGES_FOLDER = "images/Images"
DEFAULT_CSV_NAME = "raw-dataset.csv"
DEFAULT_N = "all"
DEFAULT_SELECTED_CLASSES = ["Buccinid snail"]
DEFAULT_RANDOM_SEED = 42



# === END CONFIGURATION ===



def create_batch_dataset(N_images=None, classes=None, output_suffix=None, 
                        base_path=None, images_folder=None, csv_name=None, random_seed=None):
    """
    Create a batch dataset by randomly selecting N images and their annotations.
    
    Args:
        N_images: Number of images to select (uses DEFAULT_N if None)
        classes: List of class names to include (uses DEFAULT_SELECTED_CLASSES if None)
        output_suffix: Suffix for output folders (uses N_images if None)
        base_path: Base path for data (uses DEFAULT_BASE_PATH if None)
        images_folder: Images folder name (uses DEFAULT_IMAGES_FOLDER if None)
        csv_name: CSV file name (uses DEFAULT_CSV_NAME if None)
        random_seed: Random seed (uses DEFAULT_RANDOM_SEED if None)
    """
    # Use defaults if not provided
    if N_images is None:
        N_images = DEFAULT_N
    if classes is None:
        classes = DEFAULT_SELECTED_CLASSES
    if base_path is None:
        base_path = DEFAULT_BASE_PATH
    if images_folder is None:
        images_folder = DEFAULT_IMAGES_FOLDER
    if csv_name is None:
        csv_name = DEFAULT_CSV_NAME
    if random_seed is None:
        random_seed = DEFAULT_RANDOM_SEED
    if output_suffix is None:
        output_suffix = str(N_images)
    
    # Build full paths
    images_dir = os.path.join(base_path, images_folder)
    csv_path = os.path.join(base_path, csv_name)
    
    # Set random seed
    random.seed(random_seed)
    
    # Setup output folders (in base_path)
    output_images_folder = os.path.join(base_path, f"images_{output_suffix}")
    output_labels_folder = os.path.join(base_path, f"labels_{output_suffix}")
    
    # Create output directories
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    
    print(f"🔧 Configuration:")
    print(f"   - Base path: {base_path}")
    print(f"   - Images folder: {images_dir}")
    print(f"   - CSV file: {csv_path}")
    print(f"   - Classes: {classes}")
    print(f"   - Number of images: {N_images}")
    print(f"   - Output folders: {output_images_folder}, {output_labels_folder}")
    print()
    
    # Load annotations
    try:
        df = pd.read_csv(csv_path, sep=';', low_memory=False)
        print(f"✅ Loaded {len(df)} annotations from {csv_path}")
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {csv_path}")
        print(f"   Please check if the path is correct and the file exists.")
        return
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Filter by classes
    df_filtered = df[df['name_sp'].isin(classes)]
    print(f"✅ Filtered to {len(df_filtered)} annotations for classes: {classes}")
    
    # Get unique images
    all_images = df_filtered['name_img'].unique().tolist()
    print(f"✅ Found {len(all_images)} unique images with annotations")
    
    # Select images
    if N_images == "all" or len(all_images) <= N_images:
        print(f"📋 Selecting all {len(all_images)} images")
        selected_images = all_images
    else:
        print(f"🎯 Randomly selecting {N_images} out of {len(all_images)} images")
        selected_images = random.sample(all_images, N_images)
    
    print(f"🎯 Selected {len(selected_images)} images")
    print()
    
    # Process each selected image
    processed_count = 0
    for i, img_name in enumerate(selected_images, 1):
        src_img_path = os.path.join(images_dir, img_name)
        dst_img_path = os.path.join(output_images_folder, img_name)
        
        if not os.path.exists(src_img_path):
            print(f"⚠️  Warning: Image {img_name} not found at {src_img_path}")
            continue
        
        # Copy image
        shutil.copy(src_img_path, dst_img_path)
        
        # Get image dimensions
        try:
            with Image.open(src_img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"⚠️  Warning: Could not open image {img_name}: {e}")
            continue
        
        # Filter annotations for this image
        annots = df_filtered[df_filtered['name_img'] == img_name]
        
        # Write YOLO-format txt
        txt_filename = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(output_labels_folder, txt_filename)
        
        with open(txt_path, "w") as f:
            for _, row in annots.iterrows():
                class_id = classes.index(row['name_sp'])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                if any(pd.isnull([x1, y1, x2, y2])):
                    continue  # skip invalid rows
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = abs(x2 - x1) / img_width
                height = abs(y2 - y1) / img_height
                
                # Write to txt file
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        processed_count += 1
        if i % 10 == 0:
            print(f"📁 Processed {i}/{len(selected_images)} images...")
    
    print(f"\n✅ Successfully processed {processed_count} images and annotations!")
    print(f"📂 Output folders:")
    print(f"   - Images: {output_images_folder}/")
    print(f"   - Labels: {output_labels_folder}/")
    
    return {
        'images_folder': output_images_folder,
        'labels_folder': output_labels_folder,
        'dataset_yaml': os.path.join(base_path, f"{output_suffix}_dataset.yaml"),
        'processed_count': processed_count,
        'classes': classes
    }

def create_dataset_yaml(output_suffix, classes, base_path=None):
    """Create YOLO dataset configuration file."""
    if base_path is None:
        base_path = DEFAULT_BASE_PATH
        
    dataset_config = {
        'path': base_path,  # Use base_path as the dataset root
        'train': f'images_{output_suffix}',
        'val': f'images_{output_suffix}',  # Using same data for train/val for now
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = os.path.join(base_path, f"{output_suffix}_dataset.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"📄 Created dataset config: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    # Create dataset and get outputs with default settings
    results = create_batch_dataset()
    
    if results:
        # Create dataset YAML file
        create_dataset_yaml(str(DEFAULT_N), results['classes'], DEFAULT_BASE_PATH) 