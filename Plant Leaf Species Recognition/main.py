# Testing predictions on model
from tensorflow import keras
import keras.utils as image
import os
from PIL import Image
from keras.applications.densenet import DenseNet121, preprocess_input
import numpy as np
import cv2

model = keras.models.load_model("CNN-DenseNet")


def apply_mask(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    range1 = (36, 0, 0)
    range2 = (86, 255, 255)
    mask = cv2.inRange(hsv, range1, range2)
    # res = cv2.bitwise_and(img, img, mask=mask)
    result = img.copy()
    result[mask == 0] = (255, 255, 255)
    cv2.imwrite("result.jpg", result)
    return result


def prediction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)

    img = img.reshape(-1, 224, 224, 3)
    img = img.astype('float32')
    img = img / 255.0
    print(img.shape)
    # ---
    lst = model.predict(img)
    print(lst)
    index = np.argmax(lst, axis=-1)[0]
    return index, lst[0][index]


labels = ["Ulmus carpinifolia", "Acer", "Salix aurita", "Quercus", "Alnus incana", "Betula pubescens",
          "Salix alba 'Sericea",
          "Populus tremula", "Ulmus glabra", "Sorbus aucuparia", "Salix sinerea", "Populus", "Tilia",
          "Sorbus intermedia", "Fagus silvatica"]


# path = "samples/acer.jpg"
# for img in os.listdir("samples/"):
#     img_path = os.path.join("samples/", img)
#     print(img_path)
#     result, acc = prediction(img_path)
#     print(labels[result])
#     print(acc)

import matplotlib.pyplot as plt

# Data
models = ['DenseNet', 'Custom Model']
training_accuracy = [98, 94]
validation_accuracy = [97, 92]

# Plotting
x = range(len(models))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, training_accuracy, width, label='Training Accuracy')
rects2 = ax.bar([i + width for i in x], validation_accuracy, width, label='Validation Accuracy')

# Add labels, title, and legend
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Training and Validation Accuracy')
ax.set_xticks([i + width/2 for i in x])
ax.set_xticklabels(models)
ax.legend()


# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
ax.margins(y=0.25)
plt.show()

