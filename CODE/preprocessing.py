import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_data(input_dir, output_dir, target_size=(160, 160)):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for sub_dir in os.listdir(input_dir):
        sub_dir_path = os.path.join(input_dir, sub_dir)
        output_sub_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        for file_name in tqdm(os.listdir(sub_dir_path), desc=f"Processing {sub_dir}"):
            file_path = os.path.join(sub_dir_path, file_name)

            if file_name.endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize image to the target size
                img = cv2.resize(img, target_size)

                # Normalize pixel values to the range [0, 1]
                img = img / 255.0

                # Save the preprocessed image
                output_file_path = os.path.join(output_sub_dir, file_name)
                cv2.imwrite(output_file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    input_directory = "datasets/augmented_data"  # Update with your input directory
    output_directory = "datasets/preprocessed_data"
    preprocess_data(input_directory, output_directory)
