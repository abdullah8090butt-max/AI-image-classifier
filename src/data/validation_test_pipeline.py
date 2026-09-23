import tensorflow as tf

from preprocess import x_train, x_test
from dataset_loader import y_train, y_test


# Split training data into training and validation sets
x_validation = x_train[45000:]
y_validation = y_train[45000:]

x_training = x_train[:45000]
y_training = y_train[:45000]


# Create validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (x_validation, y_validation)
)

validation_dataset = validation_dataset.batch(32)
validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)


# Create test dataset
test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)
)

test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# Display dataset information
print("Validation and test pipelines created successfully!")

print("Training images:", x_training.shape)
print("Validation images:", x_validation.shape)
print("Test images:", x_test.shape)

print("Validation batch size: 32")
print("Test batch size: 32")


# Test validation batch
for images, labels in validation_dataset.take(1):
    print("Validation batch images shape:", images.shape)
    print("Validation batch labels shape:", labels.shape)


# Test test batch
for images, labels in test_dataset.take(1):
    print("Test batch images shape:", images.shape)
    print("Test batch labels shape:", labels.shape)