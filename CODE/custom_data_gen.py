import os
import cv2
import numpy as np
from keras.utils import Sequence, to_categorical

class YourDataGenerator(Sequence):
    def __init__(self, data_dir, batch_size=32, image_size=(160, 160), num_classes=10):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_labels = sorted(os.listdir(data_dir))
        self.class_to_index = {label: i for i, label in enumerate(self.class_labels)}
        self.index_to_class = {i: label for i, label in enumerate(self.class_labels)}
        self.image_files = []
        self.labels = []
        
        for label in self.class_labels:
            label_dir = os.path.join(data_dir, label)
            for filename in os.listdir(label_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    self.image_files.append(os.path.join(label_dir, filename))
                    self.labels.append(label)

        self.num_samples = len(self.image_files)
        self.indexes = np.arange(self.num_samples)
        self.steps_per_epoch = self.num_samples // self.batch_size

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start:end]
        
        batch_anchor = []
        batch_positive = []
        batch_negative = []
        batch_labels = []  # Added line to include labels

        for i in batch_indexes:
            anchor_class = self.labels[i]
            anchor_image_path = self.image_files[i]

            # Anchor image
            img_anchor = cv2.imread(anchor_image_path)
            img_anchor = cv2.cvtColor(img_anchor, cv2.COLOR_BGR2RGB)
            img_anchor = cv2.resize(img_anchor, self.image_size)
            img_anchor = img_anchor / 255.0  # Normalize pixel values to [0, 1]

            # Positive image (image of the same class)
            positive_candidates = [idx for idx, label in enumerate(self.labels) if label == anchor_class]
            positive_index = np.random.choice(positive_candidates)
            img_positive = cv2.imread(self.image_files[positive_index])
            img_positive = cv2.cvtColor(img_positive, cv2.COLOR_BGR2RGB)
            img_positive = cv2.resize(img_positive, self.image_size)
            img_positive = img_positive / 255.0

            # Negative image (image of a different class)
            negative_candidates = [idx for idx, label in enumerate(self.labels) if label != anchor_class]
            negative_index = np.random.choice(negative_candidates)
            img_negative = cv2.imread(self.image_files[negative_index])
            img_negative = cv2.cvtColor(img_negative, cv2.COLOR_BGR2RGB)
            img_negative = cv2.resize(img_negative, self.image_size)
            img_negative = img_negative / 255.0

            batch_anchor.append(img_anchor)
            batch_positive.append(img_positive)
            batch_negative.append(img_negative)
            batch_labels.append(self.class_to_index[anchor_class])  # Added line to include labels

        return [np.array(batch_anchor), np.array(batch_positive), np.array(batch_negative)], np.array(batch_labels)  # Updated line to return labels
