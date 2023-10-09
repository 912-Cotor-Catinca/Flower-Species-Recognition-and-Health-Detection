import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


training_data = []
# Path to the PlantVillage dataset directory
dataset_dir = "C:/Users/Catinca/Documents/TomatoDataset/"


def load_dataset():
    # Load images and extract labels from directory names
    class_num = 0
    for label in os.listdir(dataset_dir):
        count = 0
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                count += 1
        training_data.append([class_num, count])
        print(count)
        class_num += 1


# Load the dataset
load_dataset()
print(len(training_data))
total = 0
n_samples = 18160
classes = 10
weights = []
for x, y in training_data:
    print(y)
for x, y in training_data:
    weights.append(n_samples/(classes*y))

print(weights)
