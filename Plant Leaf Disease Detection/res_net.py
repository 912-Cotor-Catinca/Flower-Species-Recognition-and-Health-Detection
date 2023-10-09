import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, f1_score
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms, datasets

# Set device (GPU if available, otherwise CPU)
from CustomDataset import load_dataset, CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations for training and testing sets
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(255),
                                transforms.CenterCrop(224), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                               )

# Define hyperparameters
number_classes = 10
batch_size = 64
num_epochs = 10

# Load the data
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

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 10)

model_conv = model_conv.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.fc.parameters(), lr=0.001)

# Training loop
number_epochs = 15
train_accu = []
val_accu = []
val_loss = []
train_losses = []
precision_scores = []
f1_scores = []

for epoch in range(number_epochs):
    model_conv.train()  # Set the model to training mode
    running_loss = 0.0
    running_corrects = 0
    total_train = 0
    predicted_labels = []
    true_labels = []
    start = time.time()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model_conv(images)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_train += labels.size(0)
        predicted_labels.extend(predicted.tolist())
        true_labels.extend(labels.tolist())
        running_corrects += predicted.eq(labels).sum().item()
    end = time.time()
    running_loss /= len(train_loader)
    accu = running_corrects / total_train
    train_accu.append(accu)
    train_losses.append(running_loss)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    precision_scores.append(precision)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    f1_scores.append(f1)
    # Print training loss for every epoch
    print(
        f"Epoch {epoch + 1}/{number_epochs} - Finished in{end - start} - Training loss: {running_loss} "
        f"- Epoch Accuracy:{accu} - Precision{precision} - f1score{f1}")

    # Validation loop
    model_conv.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    v_loss = 0
    total_batches = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model_conv(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Compute the validation loss
            loss_v = criterion(outputs, labels)
            v_loss += loss_v.item()
            total_batches += 1

    # Print validation accuracy for every epoch
    accuracy = correct / total
    val_accu.append(accuracy)
    avg_val_loss = v_loss / total_batches
    val_loss.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Validation accuracy: {accuracy}% Validation loss{avg_val_loss}")

# Save the trained model
torch.save(model_conv.state_dict(), 'resnet_model.pth')
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