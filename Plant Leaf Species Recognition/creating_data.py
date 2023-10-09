import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


training_data = []
src_path = "C:/Users/Catinca/Documents/DataSet/"
dirlist = sorted_alphanumeric(os.listdir(src_path))


def create_training_data():
    for sclass in dirlist:
        path = os.path.join(src_path, sclass)
        class_num = dirlist.index(sclass)
        print(class_num)
        for img in tqdm(os.listdir(path)):
            ext = os.path.splitext(img)[-1].lower()
            if ext == ".tif":
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (224, 224))
                training_data.append([new_array, class_num])
            else:
                new_path = path + "/" + img
                # print(new_path)
                for file in os.listdir(new_path):
                    # print(os.path.join(new_path, file))
                    img_array = cv2.imread(os.path.join(new_path, file), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (224, 224))
                    training_data.append([new_array, class_num])


create_training_data()
print(len(training_data))
print(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 224, 224, 1)
# Convert training_data to numpy arrays
# X = np.array([data[0] for data in training_data])
# y = np.array([data[1] for data in training_data])

# Preprocessing data and pickling them for future use
# Commented out so that preprocessing doesnt repeat on testing of various models

pickle_out = open("X_Augumented_Grayscale", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_Augumented_Grayscale", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
