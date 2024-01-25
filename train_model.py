# train_model.py
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the  dataset
# (You may need to adapt this code based on the dataset structure)
# ...

# Split the dataset into training and testing sets
# ...
train_image_path = "data/cancer_detection/train/"
test_image_path = "data/cancer_detection/test/"
# Build a CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Log the model with MLflow
mlflow.start_run()
mlflow.tensorflow.log_model(tf_saved_model_dir="saved_model", artifact_path="model")

# Evaluate the model
y_pred = model.predict(test_images)
accuracy = accuracy_score(test_labels, (y_pred > 0.5).astype(int))

# Log metrics with MLflow
mlflow.log_metric("accuracy", accuracy)

mlflow.end_run()
