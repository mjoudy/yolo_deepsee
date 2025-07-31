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

# === CONFIGURATION - MODIFY THESE FOR YOUR SETUP ===
# For Google Colab with Google Drive
BASE_PATH = "/content/drive/MyDrive/DeepSeaProject/dataset_seanoe_101899"  # Your Google Drive path
IMAGES_FOLDER = "images/Images"  # Relative to BASE_PATH
CSV_NAME = "raw-dataset.csv"  # Relative to BASE_PATH

# Dataset parameters
N = "all"  # Number of images to select - use "all" for all images, or a number like 100
SELECTED_CLASSES = ["Buccinid snail"]  # Classes to include
RANDOM_SEED = 42  # For reproducible results



# === END CONFIGURATION ===

# Build full paths
IMAGES_DIR = os.path.join(BASE_PATH, IMAGES_FOLDER)
CSV_PATH = os.path.join(BASE_PATH, CSV_NAME)

def create_batch_dataset(N_images=None, classes=None, output_suffix=None):
    """
    Create a batch dataset by randomly selecting N images and their annotations.
    
    Args:
        N_images (int): Number of images to select (uses global N if None)
        classes (list): List of class names to include (uses global SELECTED_CLASSES if None)
        output_suffix (str): Suffix for output folders (uses N if None)
    """
    # Use global defaults if not provided
    if N_images is None:
        N_images = N
    if classes is None:
        classes = SELECTED_CLASSES
    if output_suffix is None:
        output_suffix = str(N_images)
    
    # Setup output folders
    output_images_folder = f"images_{output_suffix}"
    output_labels_folder = f"labels_{output_suffix}"
    
    # Create output directories
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    
    print(f"🔧 Configuration:")
    print(f"   - Base path: {BASE_PATH}")
    print(f"   - Images folder: {IMAGES_DIR}")
    print(f"   - CSV file: {CSV_PATH}")
    print(f"   - Classes: {classes}")
    print(f"   - Number of images: {N_images}")
    print(f"   - Output folders: {output_images_folder}, {output_labels_folder}")
    print()
    
    # Load annotations
    try:
        df = pd.read_csv(CSV_PATH, sep=';', low_memory=False)
        print(f"✅ Loaded {len(df)} annotations from {CSV_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
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
        src_img_path = os.path.join(IMAGES_DIR, img_name)
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
        'dataset_yaml': f"{output_suffix}_dataset.yaml",
        'processed_count': processed_count,
        'classes': classes
    }

def create_dataset_yaml(output_suffix, classes):
    """Create YOLO dataset configuration file."""
    dataset_config = {
        'path': '.',
        'train': f'images_{output_suffix}',
        'val': f'images_{output_suffix}',  # Using same data for train/val for now
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = f"{output_suffix}_dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"📄 Created dataset config: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Create dataset and get outputs
    results = create_batch_dataset()
    
    if results:
        # Create dataset YAML file
        create_dataset_yaml(str(N), results['classes']) 