import tensorflow as tf


# Stop training when validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# Save the best model during training
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_cnn_model.keras",
    monitor="val_loss",
    save_best_only=True
)


# Store all callbacks together
callbacks = [
    early_stopping,
    model_checkpoint
]


print("Training callbacks created successfully!")
print("EarlyStopping: patience = 5")
print("ModelCheckpoint: best model will be saved")