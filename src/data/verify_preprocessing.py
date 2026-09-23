import matplotlib.pyplot as plt

from preprocess import x_train
from dataset_loader import y_train, class_names


# Verify data type
print("Data type:", x_train.dtype)

# Verify pixel range
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())

# Display a normalized image
plt.figure(figsize=(4, 4))

plt.imshow(x_train[0])
plt.title(f"Class: {class_names[y_train[0]]}")
plt.axis("off")

plt.show()