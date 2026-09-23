import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess import x_train
from augmentation import data_augmentation


# Select 4 images
sample_images = x_train[:4]


# Convert images to TensorFlow tensor
sample_images = tf.convert_to_tensor(sample_images)


# Apply augmentation
augmented_images = data_augmentation(
    sample_images,
    training=True
)


# Display original and augmented images
plt.figure(figsize=(10, 5))

for i in range(4):

    # Original image
    plt.subplot(2, 4, i + 1)
    plt.imshow(sample_images[i])
    plt.title("Original")
    plt.axis("off")

    # Augmented image
    plt.subplot(2, 4, i + 5)
    plt.imshow(augmented_images[i])
    plt.title("Augmented")
    plt.axis("off")


plt.tight_layout()
plt.show()