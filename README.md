# Flower-Species-Recognition-and-Health-Detection
Bachelor Degree
![Screenshot 2023-05-11 203000](https://github.com/912-Cotor-Catinca/Plant-Species-Recognition-and-Health-Detection/assets/72121526/6bdf6cb4-50d3-41c9-ad43-0e0ac3710b6c)


# Plant Species Recognition
## Dataset
* [Swedish Leaf Dataset](https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/) 
  * 15 classes with images with only one leaf and a clean background
## Preprocessing
* The Swedish Leaf Dataset, containing 15 classes with 75 images per class was transformed using augmantation transformations such as rotation, crop, flip and skewing.
* The final dataset contains 15 classes with 408 images per class, incresing the initial dataset.
## Plant Species Classification Model
* First model is a custom model build from scrath using Keras Framework. The model contains 2 convolutional layers, each one followed by a Max Pooling and Droupout layers.
  * To complete the model, I added one or more Dense layers for classification after the last output tensor from the convolutional base.
* Second model is a DenseNet-121 pre-trained model.
## Classification
* Plant leaf species classification is a multi-class classification problem, meaning that there are more than two classes to be predicted.
* Loss function: **sparse categorical crossentropy**; optimizer: **Adam**; evaluation metrics: **accuracy metrics**.
* Sparse categorical crossentropy loss function is usually applied when there is a multi-class classification problem and the labels are integers.
## Transfer Learning
* DenseNet-121 specifically refers to the architecture with 121 layers.
* To adapt the pre-trained model to the new classification task, the last layer of the pre-trained model is removed, and new layers are added on top.
* The code freezes the weights of the pre-trained layers, this indicates that weights of those layers will remain fixed during training. This ensures that during training, only the weights of the newly added layers will be updated, while the pre-trained weights remain fixed.
## Results
* Custom model: 92% accuracy.
* DenseNet-121 pre trained model: 98% accuracy.

# Heatlh Detection
## Dataset
* [Tomato Plant Leaf Disease Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)
  * Consists of around 18,000 images
  * 9 distinct types of plant leaf diseases, each with varying numbers of samples per class
## Preprocessing
* The Tomato Plant Leaf Disease Dataset, which is an imbalanced dataset was transformed using augmantation transformations such as rotation, crop, elastic transformation and resize.
* Undersampling was used in order to get a more balanced dataset.
* The final dataset contains 9 classes with 1816 images per class.
## Plant Leaf Disease Recognition Model
* First model is a custom model build from scrath using PyTorch Framework. The model contains 3 convolutional layers, each one followed by a Max Pooling. The output of the final pooling layer is then flattened.
  * To map the high-level features to the desired output classes, 3 Fully Connected layers were applied.
  * To complete the model, I added one Dropout layer in order to mitigate the overfitting.
* Second model is a ResNet-18 pre-trained model.
## Classification
* The training dataset comprised 80% of the entire dataset, while the remaining 20% was allocated for testing purposes.
* DataLoader class from the PyTorch framework was utilized for efficiently handle the dataset during training.
* Plant leaf disease classification is a **multi-class classification problem**, meaning that there are more than two classes to be predicted.
* The loss function chose for this classification problem is **crossentropy** and the optimizer is **Adam** with a **learning rate equal to 0.001**, that determines the size of the steps the optimizer takes.
* The training loop was build from scratch: for each epoch the model is switch to training mode; for each train batch : clear gradients -> forward pass -> compute loss -> compute gradients -> adjust learnable parameters -> compute accuracy -> compute total loss
## Transfer Learning
* ResNet-18 - skip connections, which enable the model to effectively train very deep neural networks.
* By freezing the pretrained layers and adding a new classification layer, the model can be fine-tuned on a new dataset specific to the current classification taskcates that weights of those layers will remain fixed during training. This ensures that during training, only the weights of the newly added layers will be updated, while the pre-trained weights remain fixed.
## Results
* Custom model: 92% accuracy.
* ResNet-121 pre trained model: 93% accuracy.

