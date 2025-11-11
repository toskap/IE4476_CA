import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ------------Data set up-------------------
data_dir = "data" 

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),   
    transforms.ToTensor(),         
])

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
print(f"Found {len(dataset)} images across {len(class_names)} classes: {class_names}")

# --------------Convert to NumPy---------

# Arrays
X = []
y = []

print("Loading images into memory.")
for img, label in tqdm(dataset):
    X.append(img.numpy())   
    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Loaded X: {X.shape}, y: {y.shape}")


X_flat = X.reshape(X.shape[0], -1)
print(f"Flattened shape: {X_flat.shape}")

# ----------------Standardise data-----------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)
print("Features standardised.")


# -----------------Split dataset into training and testing--------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Normalize features
mean = X_train.mean(dim=0)
std = X_train.std(dim=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# ----------------Linear classifier----------------
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=X_train.shape[1], out_features=len(class_names)),
    torch.nn.Softmax(dim=1)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 1000
for epoch in range(num_epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# ----------------Evaluation----------------
with torch.no_grad():
    y_pred_probs = model(X_test)
    y_pred_classes = torch.argmax(y_pred_probs, dim=1)

    accuracy = (y_pred_classes == y_test).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')

    # Convert to NumPy for sklearn
    y_pred_classes = y_pred_classes.cpu().numpy()
    y_true = y_test.cpu().numpy()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8,8), constrained_layout=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("True", fontsize=10)
    plt.title("Confusion Matrix: Linear Classification without PCA", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    # plt.tight_layout()
    plt.show()


print("Finished with linear classifier without PCA.")
