import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
from sklearn.model_selection import train_test_split

# Model 2

pickle_in = open("X_Augumented_Grayscale", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y_Augumented_Grayscale", "rb")
y = pickle.load(pickle_in)
X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.80)
print(len(X_train))
print(len(X_test))

NAME = "Model_2"
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

# print(X_train)

model = Sequential()

model.add(Conv2D(100, (5, 5), padding="same", strides=(2, 2), activation="relu", input_shape=(224, 224, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5)))
model.add(Dropout(0.2))

model.add(Conv2D(250, (5, 5), padding="same", strides=(2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(15))
model.add(Activation('softmax'))

# model.summary()


tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=1, validation_split=0.2, callbacks=[tensorboard])

# Plot Accuracy
plt.title('Model Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='green', label='test')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

# Plot Loss
plt.title('Model Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='green', label='test')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


model.save("CNN-model2")

model = keras.models.load_model("CNN-model2")
print(y_test)
# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# Predict the classes for the test dataset
y_pred = model.predict(X_test)
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
prediction = model.predict(X_test[:1])
print("prediction shape:", prediction.shape)
