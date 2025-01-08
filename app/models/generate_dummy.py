from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Input

# Define a dummy Keras model
def create_dummy_model():
    model = Sequential([
        Input(shape=(64, 64, 3)),  # Dummy input shape
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes for demonstration
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Generate and save the model
def save_dummy_model(filepath):
    model = create_dummy_model()
    model.save(filepath, save_format='h5')
    print(f"Dummy model saved at: {filepath}")

# Specify the path to save the model
save_path = '/home/osama/AlzieDet/App/app/models/dummy_model.h5'

if __name__ == '__main__':
    save_dummy_model(save_path)

