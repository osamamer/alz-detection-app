import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple model
def create_dummy_model():
    model = models.Sequential([
        layers.Input(shape=(64, 64, 3)),  # Input layer for 64x64 RGB images
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 output classes
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Instantiate and save the model
dummy_model = create_dummy_model()

# Save the model in .keras format
dummy_model.save('models/dummy_model.keras')

print("Dummy model saved as 'dummy_model.keras'")
