import os
import sys
import time
from datetime import timedelta
from ultralytics import YOLO
import yaml

class TeeLogger:
    """
    A class that redirects writes to both stdout and a log file.
    """
    def __init__(self, filename, stream=sys.stdout):
        self.file = open(filename, 'w', encoding='utf-8') # Open log file with utf-8 encoding
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
        # Ensure data is written immediately to the file
        self.file.flush()
        self.stream.flush()

    def flush(self):
        # Required for compatibility with file-like objects
        self.file.flush()
        self.stream.flush()

    def close(self):
        self.file.close()


def create_dataset_yaml(data_dir, yaml_path):
    """
    Creates the dataset YAML configuration file required by YOLOv8.

    Args:
        data_dir (str): Path to the root directory containing 'images' and 'labels' subdirectories.
                        Expected structure:
                        data_dir/
                        ├── train/
                        │   ├── images/
                        │   └── labels/
                        ├── val/ (or validation/)
                        │   ├── images/
                        │   └── labels/
                        └── test/ (optional)
                            ├── images/
                            └── labels/
        yaml_path (str): Path where the dataset.yaml file will be saved.
    """
    dataset_config = {
        'path': data_dir,  # Root path to the dataset
        'train': 'train/images',  # Path to training images (relative to 'path')
        'val': 'validation/images',  # Path to validation images (relative to 'path')
        'test': '',  # Optional, leave empty if no separate test set
        'nc': 1,  # Number of classes
        'names': ['hand']  # Class names, ordered by index (index 0 corresponds to 'hand')
    }

    with open(yaml_path, 'w', encoding='utf-8') as f: # Use utf-8 encoding for YAML
        yaml.dump(dataset_config, f, default_flow_style=False)
    print(f"Dataset YAML configuration saved to: {yaml_path}")


def main():
    # --- Configuration ---
    # 1. Define paths
    dataset_root_dir = "D:/Python_Files/Personal_projects/YOLOv8/hand_detection_dataset_converted"


    dataset_yaml_path = "hand_detection_dataset.yaml"

    # Name for the model checkpoint to save during training
    model_save_name = "yolo11n_hand_detect.pt"

    # Log file path
    log_file_path = "终端.txt"

    # --- Setup Logging ---
    # Redirect both stdout and stderr to the log file AND the console
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    tee_logger = TeeLogger(log_file_path, original_stdout)
    tee_logger_err = TeeLogger(log_file_path, original_stderr) # Also log errors

    sys.stdout = tee_logger
    sys.stderr = tee_logger_err

    start_time = time.time() # Record the start time of the entire process

    try:
        # 2. Create the dataset YAML file (if not already created)
        # IMPORTANT: Make sure your data is in YOLO format before running this!
        # Expected folders: dataset_root_dir/train/images, dataset_root_dir/train/labels,
        #                   dataset_root_dir/validation/images, dataset_root_dir/validation/labels
        create_dataset_yaml(dataset_root_dir, dataset_yaml_path)

        # --- Training Setup ---
        # 3. Load a pre-trained YOLO11n model (recommended for transfer learning)
        #    This loads general features from COCO dataset, helping converge faster.
        print("Loading pre-trained YOLO11n model...")
        model = YOLO("yolo11n.pt") # Downloads if not present

        # --- Training Execution ---
        # 4. Start training
        print("Starting training on hand detection dataset...")
        print(f"Training log will be saved to: {log_file_path}")
        train_start_time = time.time() # Record start time of training specifically
        train_results = model.train(
            data=dataset_yaml_path,      # Path to your dataset YAML file
            epochs=100,                  # Number of training epochs. Adjust based on results.
            imgsz=640,                   # Input image size (you can try 320 for faster training with potential accuracy trade-off)
            batch=16,                    # Batch size. Adjust according to your GPU memory. RTX 4060 8GB might handle 16 okay, but monitor.
                                         # If you encounter Out Of Memory errors, reduce this (e.g., 8, 4).
            device="0",                  # Use GPU 0. Change to 'cpu' if needed (very slow).
            name=model_save_name,        # Name for saving checkpoints and results in runs/detect/
            cache=False,                 # Cache images in memory (True can speed up but uses more RAM/GPU memory).
            optimizer='auto',            # Use auto optimizer selection (usually AdamW for YOLO11)
            # resume=False,              # Set to True to resume training from the last saved checkpoint
            # freeze=[0, 9],             # Example: Freeze first 10 layers (0-9). Useful for feature extraction.
                                         # Uncomment and adjust if you want to freeze early layers initially.

            # Optimizer settings (often default is fine, but can be tuned)
            lr0=0.01,                    # Initial learning rate (default for YOLO11 is often around 0.01)
            lrf=0.01,                    # Final learning rate (at epoch end) (default for YOLO11 is often around 0.01)

            # Data Augmentation (often default is good, but can be adjusted)
            # fliplr=0.5,                # Horizontal flip probability (default is often 0.5)
            # mosaic=1.0,                # Mosaic augmentation probability (default is often 1.0 for first 3 epochs, then off)
            # mixup=0.0,                 # MixUp augmentation probability (default is often 0.0)
            # copy_paste=0.0,            # Copy-paste augmentation probability (default is often 0.0)

            # Overfitting Prevention (often default is fine, but can be tuned)
            # dropout=0.0,               # Dropout rate (applied to the classification head usually)

            # Validation settings
            val=True,                    # Enable validation during training (default is True)
            save_period=10,              # Save checkpoint every N epochs (set to -1 to disable periodic saves)
            plots=True,                  # Generate training plots (default is True)
            # patience=100,              # Epochs to wait for no observable improvement for early stopping (default is 100)

            # Advanced
            # half=False,                # Use FP16 half precision (can speed up training and reduce memory usage slightly)
            # amp=True,                  # Automatic Mixed Precision (AMP) training (default is True, usually beneficial)
            # workers=8,                 # Number of worker threads for data loading (default is usually 8)
        )
        train_end_time = time.time() # Record end time of training
        total_train_duration = train_end_time - train_start_time

        print("Training completed successfully!")
        print(f"Trained model saved as: {model_save_name}")
        print(f"Training logs and results saved in: runs/detect/{model_save_name}")
        print(f"Total training duration: {timedelta(seconds=int(total_train_duration))}")
        # Training results (losses, metrics over epochs) are stored in train_results object
        # You can access them if needed, e.g., train_results.results_dict

        # --- Optional: Validate the final model ---
        print("\nValidating the final model on the validation set...")
        # The model is already loaded with the best weights from training
        metrics = model.val() # Uses the validation set defined in the dataset YAML
        print("Final validation metrics:")
        print(metrics.results_dict)

        # --- Optional: Run a quick prediction example (Uncomment if needed) ---
        # Replace 'path/to/your/test_image.jpg' with an actual image path from your validation set or elsewhere.
        # test_image_path = "path/to/your/test_image.jpg"
        # if os.path.exists(test_image_path):
        #     print(f"\nRunning prediction on example image: {test_image_path}...")
        #     results = model(test_image_path)
        #     # Display the result (opens a window)
        #     results[0].show()
        #     # Or save the result with a specific filename
        #     results[0].save(filename=f'prediction_example_{os.path.basename(test_image_path)}')
        #     print(f"Prediction result saved.")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout) # Print traceback to both console and log
        sys.stderr.write(f"\nAn error occurred during execution: {e}\n")
        traceback.print_exc(file=sys.stderr) # Also print traceback to stderr log
    finally:
        # --- Restore Original Streams and Close Logger ---
        # Always restore the original stdout/stderr, even if an error occurs
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_logger.close()
        tee_logger_err.close()
        end_time = time.time() # Record the end time of the entire process
        total_duration = end_time - start_time
        print(f"\nScript execution finished.")
        print(f"Total script runtime (including setup): {timedelta(seconds=int(total_duration))}")
        # Write final summary to the log file as well (though it's already printed due to tee)
        # original_stdout.write(f"\nScript execution finished.\n")
        # original_stdout.write(f"Total script runtime (including setup): {timedelta(seconds=int(total_duration))}\n")


if __name__ == "__main__":
    main()