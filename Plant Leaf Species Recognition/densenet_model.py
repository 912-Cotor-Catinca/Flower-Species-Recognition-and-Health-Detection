import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle

# Model 4

import tensorflow as tf
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense

base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Freeze the weights of the pre-trained layers
# for layer in base_model.layers:
#     layer.trainable = False

# Create a new model by adding a new classification layer on top of the pre-trained base model

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(15, activation='softmax'))


pickle_in = open("X_Augumented_Grayscale1", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y_Augumented_Grayscale1", "rb")
y = pickle.load(pickle_in)

NAME = "DenseNet_Model"

tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.80)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train_rgb = np.stack((X_train,) * 3, axis=-1)
X_val_rgb = np.stack((X_test,) * 3, axis=-1)

# Verify the new shapes
print("X_train_rgb shape:", X_train_rgb.shape)
print("X_val_rgb shape:", X_val_rgb.shape)

# Normalize pixel values to the range [0, 1]
X_train_rgb = X_train_rgb / 255.0
X_val_rgb = X_val_rgb / 255.0

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on your dataset
history = model.fit(X_train_rgb, y_train, epochs=10, batch_size=1, validation_split=0.2, callbacks=[tensorboard])


# Plot Accuracy
plt.title('Model Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='green', label='test')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

plt.title('Model Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='green', label='test')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

model.summary()

model.save("CNN-DenseNet")
model = keras.models.load_model("CNN-DenseNet")
print(y_test)
# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(X_val_rgb, y_test, verbose=0)
# Predict the classes for the test dataset
y_pred = model.predict(X_val_rgb)
y_pred_classes = np.argmax(y_pred, axis=1)

print(y_pred_classes.shape)
# Generate the classification report
report = classification_report(y_test, y_pred_classes)
# Print the accuracy and classification report
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:")
print(report)
# Print the confusion matrix using Matplotlib
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_classes)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print("Generate a prediction")
prediction = model.predict(X_val_rgb[:1])
print("prediction shape:", prediction.shape)
