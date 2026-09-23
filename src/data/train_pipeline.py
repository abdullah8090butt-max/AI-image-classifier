import tensorflow as tf

from preprocess import x_train
from dataset_loader import y_train
from augmentation import data_augmentation


# Create TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)


# Shuffle training data
train_dataset = train_dataset.shuffle(
    buffer_size=1000
)


# Create batches
train_dataset = train_dataset.batch(32)


# Apply data augmentation
train_dataset = train_dataset.map(
    lambda images, labels: (
        data_augmentation(images, training=True),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)


# Prefetch
train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


print("Training dataset pipeline created successfully!")
print("Batch size: 32")


# Test one batch
for images, labels in train_dataset.take(1):

    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)