import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ==========================================
# Add data folder to Python path
# ==========================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "data"
    )
)


from validation_test_pipeline import test_dataset
from dataset_loader import x_test, y_test


# ==========================================
# Load Saved Model
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


model = tf.keras.models.load_model(
    model_path
)


print("Saved CNN model loaded successfully!")

print("Model location:")
print(model_path)


# ==========================================
# CIFAR-10 Class Names
# ==========================================

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


# ==========================================
# Select Sample Images
# ==========================================

num_samples = 12


sample_images = x_test[:num_samples]

sample_labels = y_test[:num_samples]


# ==========================================
# Generate Predictions
# ==========================================

print("\nGenerating predictions...")


predictions = model.predict(
    sample_images,
    verbose=0
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


print("Predictions generated successfully!")


# ==========================================
# Display Predictions
# ==========================================

plt.figure(
    figsize=(12, 9)
)


for i in range(num_samples):

    plt.subplot(
        3,
        4,
        i + 1
    )


    plt.imshow(
        sample_images[i]
    )


    actual_class = class_names[
        int(sample_labels[i])
    ]


    predicted_class = class_names[
        int(predicted_classes[i])
    ]


    plt.title(
        f"Actual: {actual_class}\n"
        f"Predicted: {predicted_class}"
    )


    plt.axis("off")


plt.suptitle(
    "CIFAR-10 CNN Sample Predictions",
    fontsize=16
)


plt.tight_layout()


plt.show()


# ==========================================
# Print Prediction Results
# ==========================================

print("\n========== SAMPLE PREDICTIONS ==========")


for i in range(num_samples):

    actual_class = class_names[
        int(sample_labels[i])
    ]


    predicted_class = class_names[
        int(predicted_classes[i])
    ]


    result = (
        "Correct"
        if actual_class == predicted_class
        else "Incorrect"
    )


    print(
        f"{i + 1:02d}. "
        f"Actual: {actual_class:12s} | "
        f"Predicted: {predicted_class:12s} | "
        f"{result}"
    )


print("\nStep 4.6 completed successfully!")

print("\nPhase 4 completed successfully!")