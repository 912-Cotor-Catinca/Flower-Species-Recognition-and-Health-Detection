import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from sklearn.metrics import f1_score, precision_score

# Define a custom dataset
# Custom dataset class
from torchsummary import summary
from CustomModel import Model
from CustomDataset import CustomDataset, load_dataset


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
num_classes = 10
batch_size = 64
num_epochs = 15

# Create the model and move it to the device
model = Model(input_shape=(3, 224, 224), num_classes=num_classes).to(device)
summary(model, (3, 224, 224))

# Define the loss function and optimizer
# weights = [0.853, 1.816, 1.141, 0.951, 1.907, 1.025, 1.083, 1.293, 4.868, 0.338]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(255),
                                transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                               )
train_accu = []
train_losses = []
precision_scores = []
f1_scores = []
val_accu = []
val_loss = []


def test():
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        v_loss = 0
        total_batches = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Compute the validation loss
            loss_v = criterion(outputs, labels)
            v_loss += loss_v.item()
            total_batches += 1

        accuracy = correct / total
        val_accu.append(accuracy)
        avg_val_loss = v_loss / total_batches
        val_loss.append(avg_val_loss)
        print(f"Test Accuracy: {accuracy} - Validation loss {avg_val_loss}")


def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0
        correct = 0
        total = 0
        train_loss = 0.0
        predicted_labels = []
        true_labels = []
        start = time.time()
        print("Epoch: " + str(epoch) + " started")
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Print predictions and labels for each step
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        end = time.time()
        # train_loss = running_loss / len(train_loader)
        train_loss /= len(train_loader)
        accu = correct / total
        train_accu.append(accu)
        train_losses.append(train_loss)
        precision = precision_score(true_labels, predicted_labels, average='macro')
        precision_scores.append(precision)
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        f1_scores.append(f1)
        print("Epoch: " + str(epoch) + " finished in:" + str(end - start) +
              " Train loss: " + str(train_loss) +
              " Accuracy: " + str(accu) +
              " Precision: " + str(precision) +
              " F1 Score: " + str(f1))
        test()


if __name__ == '__main__':
    data_dir = []
    X = []
    y = []
    # Path to the PlantVillage dataset directory
    load_dataset(data_dir)
    # Create the dataset and dataloaders
    for features, labels in data_dir:
        X.append(features)
        y.append(labels)
    dataset = CustomDataset(X, y, transform=transform)

    train_size = int(0.8 * len(data_dir))
    print(train_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    total_step = len(train_loader)
    print(total_step)
    total_iterations = num_epochs * total_step
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=total_iterations)
    lrs = []

    # Training loop
    train()
    torch.save(model.state_dict(), 'plant_disease_model_1.pt')
    # test()
    # Evaluation
    # Training loop and metric calculations
    # Plotting
    epochs = list(range(1, 16))
    plt.plot(epochs, train_accu, label='Train Accuracy')
    plt.plot(epochs, val_accu, label='Validation Accuracy')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy vs Validation Accuracy')

    # Adding a legend
    plt.legend()
    # Displaying the plot
    plt.show()
    # Plotting accuracy
    plt.plot(train_accu)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.show()

    # Plotting
    epochs = list(range(1, 16))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss vs Validation Loss')

    # Adding a legend
    plt.legend()
    # Displaying the plot
    plt.show()

    # Plotting precision
    plt.plot(precision_scores)
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision')
    plt.show()

    # Plotting F1 score
    plt.plot(f1_scores)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.show()
