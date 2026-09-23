import os
import pickle
import numpy as np


# Path to the extracted CIFAR-10 dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "cifar-10-batches-py")


def load_batch(file_path):
    """Load one CIFAR-10 batch."""
    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    # CIFAR-10 format: (10000, 3072)
    # Convert to: (10000, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, np.array(labels)


# Load training batches
x_train_list = []
y_train_list = []

for i in range(1, 6):
    file_path = os.path.join(DATA_DIR, f"data_batch_{i}")

    images, labels = load_batch(file_path)

    x_train_list.append(images)
    y_train_list.append(labels)


# Combine all training batches
x_train = np.concatenate(x_train_list)
y_train = np.concatenate(y_train_list)


# Load test batch
test_path = os.path.join(DATA_DIR, "test_batch")
x_test, y_test = load_batch(test_path)


print("CIFAR-10 loaded successfully!")
print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)
# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

print("\nNumber of classes:", len(class_names))
print("Classes:", class_names)
print("First training label:", y_train[0])
print("First training image shape:", x_train[0].shape)