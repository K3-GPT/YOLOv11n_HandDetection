import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


def convert_coco_to_yolo(coco_json_path, images_dir, labels_output_dir, target_class_names=None):
    """
    Converts COCO format annotations to YOLO format label files.

    Args:
        coco_json_path (str or Path): Path to the COCO format annotation JSON file (e.g., instances_train2017.json).
        images_dir (str or Path): Path to the directory containing the corresponding images.
        labels_output_dir (str or Path): Path to the directory where YOLO format .txt labels will be saved.
        target_class_names (list, optional): List of class names to include. E.g., ['hand']. If None, includes all.
    """
    # Ensure directories exist
    os.makedirs(labels_output_dir, exist_ok=True)

    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create mapping from category id to index (YOLO class id)
    # Find categories matching target names (if provided) or use all
    categories = coco_data['categories']
    category_mapping = {}
    yolo_class_ids = []
    yolo_class_names = []

    if target_class_names is None:
        # Include all categories
        for cat in categories:
            category_mapping[cat['id']] = len(category_mapping)
            yolo_class_ids.append(len(category_mapping) - 1)
            yolo_class_names.append(cat['name'])
    else:
        # Only include specified target classes
        for i, tgt_name in enumerate(target_class_names):
            found = False
            for cat in categories:
                if cat['name'].lower() == tgt_name.lower():  # Case-insensitive match
                    category_mapping[cat['id']] = i
                    yolo_class_ids.append(i)
                    yolo_class_names.append(tgt_name)
                    found = True
                    break
            if not found:
                print(
                    f"Warning: Target class '{tgt_name}' not found in COCO categories: {[c['name'] for c in categories]}")
                # Optionally, raise an error if a target class is missing
                # raise ValueError(f"Target class '{tgt_name}' not found in COCO categories.")

    print(f"Mapping COCO categories to YOLO IDs: {category_mapping}")
    print(f"Yolo class names: {yolo_class_names}")

    # Create a lookup for image id to image info (width, height, filename)
    image_info_lookup = {img['id']: img for img in coco_data['images']}

    # Process each annotation
    for ann in coco_data['annotations']:
        category_id = ann['category_id']

        # Skip if category is not in target list (if specified)
        if target_class_names is not None and category_id not in [cat['id'] for cat in categories if
                                                                  cat['name'] in target_class_names]:
            continue

            # Map COCO category id to YOLO class id
        if category_id not in category_mapping:
            # This can happen if target_class_names was used and an unexpected category appears, or if mapping is wrong
            print(f"Warning: Annotation with unexpected category_id {category_id} found. Skipping.")
            continue
        yolo_class_id = category_mapping[category_id]

        # Get image info for normalization
        image_id = ann['image_id']
        img_info = image_info_lookup.get(image_id)
        if not img_info:
            print(f"Warning: No image info found for annotation image_id {image_id}. Skipping.")
            continue

        img_width = img_info['width']
        img_height = img_info['height']

        # Extract bounding box coordinates (x_min, y_min, width, height)
        bbox = ann['bbox']
        x_min, y_min, width, height = bbox

        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (x_min + width / 2.0) / img_width
        y_center = (y_min + height / 2.0) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # Ensure coordinates are within bounds [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_width = max(0.0, min(1.0, norm_width))
        norm_height = max(0.0, min(1.0, norm_height))

        # Determine the output label filename (same as image name, change extension to .txt)
        img_filename = img_info['file_name']
        # Handle potential nested paths in file_name
        img_path_obj = Path(img_filename)
        label_filename = img_path_obj.with_suffix('.txt').name
        label_path = os.path.join(labels_output_dir, label_filename)

        # Append the label line to the corresponding .txt file
        # Format: <class_id> <x_center> <y_center> <width> <height>
        label_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"

        # Use 'a' mode to append if multiple objects exist in one image
        with open(label_path, 'a') as f_label:
            f_label.write(label_line)

    print(f"Conversion complete for {coco_json_path}. Labels saved to: {labels_output_dir}")


def main():
    # --- Configuration ---
    # Point to the base directory containing 'annotations', 'train2017', 'val2017'
    base_data_dir = Path(r"D:\Python_Files\Personal_projects\YOLOv8\hand_detection_dataset")

    # Path to COCO annotation files (relative to base_data_dir)
    train_annotations_path = base_data_dir / "annotations" / "instances_train2017.json"
    val_annotations_path = base_data_dir / "annotations" / "instances_val2017.json"

    # Paths to COCO image directories (relative to base_data_dir)
    train_images_dir = base_data_dir / "train2017"
    val_images_dir = base_data_dir / "val2017"

    # Output directories for YOLO format labels and images (relative to base_data_dir or absolute)
    # We'll create a new folder structure: converted_yolo/train/images, converted_yolo/train/labels, etc.
    output_base_dir = base_data_dir.parent / "hand_detection_dataset_converted"  # Or wherever you prefer
    train_labels_output_dir = output_base_dir / "train" / "labels"
    val_labels_output_dir = output_base_dir / "validation" / "labels"
    train_images_output_dir = output_base_dir / "train" / "images"
    val_images_output_dir = output_base_dir / "validation" / "images"

    # Specify the target class name(s) from COCO annotations that you want to keep.
    # Check your instances_*.json file under "categories".
    target_class_names = ["hand"]  # Modify this if the category name in your JSON is different

    print("Starting COCO to YOLO conversion...")
    print(f"Output will be saved to: {output_base_dir}")

    # Create output directories
    os.makedirs(train_labels_output_dir, exist_ok=True)
    os.makedirs(val_labels_output_dir, exist_ok=True)
    os.makedirs(train_images_output_dir, exist_ok=True)
    os.makedirs(val_images_output_dir, exist_ok=True)

    # Convert Training Annotations
    print(f"Processing training annotations: {train_annotations_path}")
    convert_coco_to_yolo(
        coco_json_path=train_annotations_path,
        images_dir=train_images_dir,
        labels_output_dir=train_labels_output_dir,
        target_class_names=target_class_names
    )

    # Convert Validation Annotations
    print(f"Processing validation annotations: {val_annotations_path}")
    convert_coco_to_yolo(
        coco_json_path=val_annotations_path,
        images_dir=val_images_dir,
        labels_output_dir=val_labels_output_dir,
        target_class_names=target_class_names
    )

    # Copy Images (this part handles moving the images to the expected YOLO structure)
    import shutil
    print("\nCopying images to the new YOLO structure...")
    print(f"Copying train images from {train_images_dir} to {train_images_output_dir}")
    for img_file in train_images_dir.glob('*'):
        if img_file.is_file():
            shutil.copy2(img_file, train_images_output_dir / img_file.name)

    print(f"Copying val images from {val_images_dir} to {val_images_output_dir}")
    for img_file in val_images_dir.glob('*'):
        if img_file.is_file():
            shutil.copy2(img_file, val_images_output_dir / img_file.name)

    print("\n--- Conversion and Copying Summary ---")
    print(f"Converted labels and copied images are saved in: {output_base_dir}")
    print("Final directory structure:")
    print(f"  {output_base_dir}")
    print(f"  ├── train/")
    print(f"  │   ├── images -> Contains images copied from {train_images_dir}")
    print(f"  │   └── labels -> Contains .txt files converted from {train_annotations_path}")
    print(f"  └── validation/")
    print(f"      ├── images -> Contains images copied from {val_images_dir}")
    print(f"      └── labels -> Contains .txt files converted from {val_annotations_path}")
    print("\nThis structure is ready for the training script.")


if __name__ == "__main__":
    main()