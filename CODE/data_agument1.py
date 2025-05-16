import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def augment_data(input_dir, output_dir, augmentation_factor=10):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create an ImageDataGenerator with extended augmentation settings
    datagen = ImageDataGenerator(
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        width_shift_range=0.2,
        height_shift_range=0.2,
        preprocessing_function=lambda x: cv2.GaussianBlur(x, (5, 5), 0),  # Add blur
    )

    for sub_dir in os.listdir(input_dir):
        sub_dir_path = os.path.join(input_dir, sub_dir)
        output_sub_dir = os.path.join(output_dir, sub_dir)
        os.makedirs(output_sub_dir, exist_ok=True)

        for file_name in os.listdir(sub_dir_path):
            file_path = os.path.join(sub_dir_path, file_name)

            if file_name.endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.reshape((1,) + img.shape)  # Add batch dimension

                # Generate augmented images and save them
                for i, augmented_img in enumerate(datagen.flow(img, batch_size=1)):
                    augmented_img = augmented_img[0].astype(np.uint8)
                    augmented_file_path = os.path.join(output_sub_dir, f"augmented_{i}_{file_name}")
                    cv2.imwrite(augmented_file_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))

                    # Check if reached the desired augmentation factor
                    if i + 1 == augmentation_factor:
                        break

if __name__ == "__main__":
    input_directory = "datasets/mydata"
    output_directory = "datasets/augmented_data"
    augment_data(input_directory, output_directory)
