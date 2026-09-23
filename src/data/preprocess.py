import numpy as np

from dataset_loader import x_train, x_test


# Normalize image pixel values
# Original range: 0-255
# New range: 0-1

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0


print("Normalization completed!")

print("Training images shape:", x_train.shape)
print("Test images shape:", x_test.shape)

print("Training pixel minimum:", x_train.min())
print("Training pixel maximum:", x_train.max())

print("Test pixel minimum:", x_test.min())
print("Test pixel maximum:", x_test.max())