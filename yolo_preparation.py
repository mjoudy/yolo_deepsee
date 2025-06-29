import os
import shutil
import cv2
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def read_annotation_csv(csv_path):
    """Read annotation CSV with ; separator and skip bad lines."""
    return pd.read_csv(csv_path, delimiter=';', on_bad_lines='skip', engine='python')

def setup_output_structure(output_dir):
    """Create folder structure for YOLO: images/train, images/val, labels/train, labels/val."""
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

def filter_and_split_dataset(df, selected_classes, val_ratio=0.2, random_seed=42):
    """Filter dataset for selected classes and split using multi-label stratification."""
    df_filtered = df[df["name_sp"].isin(selected_classes)]
    classes = sorted(df_filtered["name_sp"].unique())
    class2id = {cls: i for i, cls in enumerate(classes)}
    grouped = df_filtered.groupby("name_img")

    img_to_species = df_filtered.groupby("name_img")["name_sp"].unique()
    image_names = img_to_species.index.tolist()

    if len(classes) > 1:
        # MULTI-CLASS → use multilabel stratified split
        multilabel_matrix = pd.DataFrame(0, index=image_names, columns=classes)
        for img_name, species_list in img_to_species.items():
            for sp in species_list:
                if sp in multilabel_matrix.columns:
                    multilabel_matrix.loc[img_name, sp] = 1

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_seed)
        train_idx, val_idx = next(msss.split(multilabel_matrix.values, multilabel_matrix.values))
        image_names = multilabel_matrix.index.to_list()
        train_imgs = [image_names[i] for i in train_idx]
        val_imgs = [image_names[i] for i in val_idx]
    else:
        # SINGLE-CLASS → fallback to regular stratified split
        image_names = np.array(image_names)
        y = [1] * len(image_names)  # dummy labels for stratify
        train_imgs, val_imgs = train_test_split(
            image_names,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=y
        )

    return grouped, class2id, train_imgs, val_imgs, classes

def convert_to_yolo_format(anns, class2id, w, h):
    """Convert annotations to YOLO format (class_id, x_center, y_center, width, height)."""
    yolo_lines = []
    for _, row in anns.iterrows():
        try:
            x_c = (row["x1"] + row["x2"]) / 2 / w
            y_c = (row["y1"] + row["y2"]) / 2 / h
            bw = abs(row["x2"] - row["x1"]) / w
            bh = abs(row["y2"] - row["y1"]) / h
            class_id = class2id[row["name_sp"]]
            yolo_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
        except Exception as e:
            print(f"⚠️ Skipping bad annotation: {e}")
    return yolo_lines

def process_split(image_list, split_name, images_dir, output_dir, grouped, class2id):
    """Process and convert each image and its annotations into YOLO format."""
    for img_name in tqdm(image_list, desc=f"Processing {split_name}"):
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Cannot read image: {img_path}")
            continue
        h, w = img.shape[:2]

        # Copy image
        shutil.copy(img_path, os.path.join(output_dir, "images", split_name, img_name))

        try:
            anns = grouped.get_group(img_name)
        except KeyError:
            print(f"⚠️ No annotations for: {img_name}")
            continue

        # Create YOLO label file
        yolo_lines = convert_to_yolo_format(anns, class2id, w, h)
        if yolo_lines:
            label_file = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(output_dir, "labels", split_name, label_file)
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))

def create_data_yaml(output_dir, class_names, yaml_path="data.yaml"):
    """Create YOLO-compatible data.yaml config file."""
    yaml_dict = {
        "path": output_dir,
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names
    }
    with open(os.path.join(output_dir, yaml_path), "w") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)
