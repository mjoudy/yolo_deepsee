# config.py

import os

# === Shared Parameters ===
BASE_PATH = "/content/drive/MyDrive/DeepSeaProject"  # or use os.getcwd() locally

# Select species/classes to include
SELECTED_CLASSES = ["Bythograeid crab", "Other fish"]
SAFE_CLASS_NAMES = "_".join(cls.replace(" ", "_") for cls in SELECTED_CLASSES)

# Paths (dynamically generated)
DATASET_DIR = os.path.join(BASE_PATH, f"yolo_dataset_{SAFE_CLASS_NAMES}")
OUTPUT_DIR = os.path.join(BASE_PATH, f"yolo_output_{SAFE_CLASS_NAMES}")
CSV_NAME = "raw-dataset.csv"
CSV_PATH = os.path.join(BASE_PATH, CSV_NAME)
IMAGES_FOLDER = "images/Images"
IMAGES_DIR = os.path.join(BASE_PATH, IMAGES_FOLDER)

# Constants
VAL_RATIO = 0.2
RANDOM_SEED = 42
