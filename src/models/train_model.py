import sys
import os
import numpy as np
import tensorflow as tf


# Add data folder to Python path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data"
    )
)


from cnn_model import model

from train_pipeline import train_dataset

from validation_test_pipeline import (
    validation_dataset,
    test_dataset
)


print("Model and datasets loaded successfully!")
print("Training dataset connected.")
print("Validation dataset connected.")
print("Test dataset connected.")


# ==========================================
# Train Model
# ==========================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20
)


print("\nCNN training completed successfully!")


# ==========================================
# Save Model
# ==========================================

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


model_path = os.path.join(
    project_root,
    "cnn_model.keras"
)


model.save(model_path)


print("\nCNN model saved successfully!")
print("Model location:")
print(model_path)


# ==========================================
# Evaluate Test Dataset
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# Generate Predictions
# ==========================================

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