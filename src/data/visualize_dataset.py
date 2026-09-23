import matplotlib.pyplot as plt
import numpy as np

from dataset_loader import x_train, y_train, class_names


# Select 9 random images
random_indices = np.random.choice(len(x_train), 9, replace=False)


# Display random images
plt.figure(figsize=(10, 10))

for i, index in enumerate(random_indices):
    plt.subplot(3, 3, i + 1)

    plt.imshow(x_train[index])
    plt.title(class_names[y_train[index]])
    plt.axis("off")

plt.tight_layout()
plt.show()