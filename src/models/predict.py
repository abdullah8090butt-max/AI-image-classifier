import sys
import os
import numpy as np
import tensorflow as tf


# Add data folder to Python path
sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

from validation_test_pipeline import test_dataset


# Load the trained CNN model
model_path = os.path.join(
    os.path.dirname(__file__),
    "cnn_model.keras"
)

model = tf.keras.models.load_model(model_path)

print("Saved CNN model loaded successfully!")


# Generate predictions from one test batch
for test_images, test_labels in test_dataset.take(1):

    predictions = model.predict(
        test_images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    actual_classes = test_labels.numpy().flatten()


print("\nPredictions generated successfully!")

print("\nFirst 10 Predictions:")
print("Predicted:", predicted_classes[:10])
print("Actual:   ", actual_classes[:10])