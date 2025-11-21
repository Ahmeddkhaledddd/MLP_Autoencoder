import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Paths to datasets
train_zip_path = '/content/drive/My Drive/train_actor_faces.zip'
validate_zip_path = '/content/drive/My Drive/validate_actor_faces.zip'
test_zip_path = '/content/drive/My Drive/test_actor_faces.zip'

# Unzipping datasets
for zip_path, output_dir in [(train_zip_path, 'train_dataset'),
                             (validate_zip_path, 'validate_dataset'),
                             (test_zip_path, 'test_dataset')]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

# Directory paths
train_dir = 'train_dataset/train_actor_faces'
validate_dir = 'validate_dataset/validate_actor_faces'
test_dir = 'test_dataset/test_actor_faces'

# Data preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validate_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)

# Define Encoder
def build_encoder(input_shape, latent_dim):
    encoder = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(latent_dim, activation='relu'),
    ])
    return encoder

# Define Decoder
def build_decoder(latent_dim, output_shape):
    decoder = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(np.prod(output_shape), activation='sigmoid'),
        layers.Reshape(output_shape),
    ])
    return decoder

# Combine Encoder and Decoder into Autoencoder
input_shape = (64, 64, 3)  # Example input shape
latent_dim = 128
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)

input_image = layers.Input(shape=input_shape)
latent_vector = encoder(input_image)
reconstructed_image = decoder(latent_vector)

autoencoder = models.Model(inputs=input_image, outputs=reconstructed_image)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Prepare data for autoencoder
X_train = np.concatenate([train_generator[i][0] for i in range(len(train_generator))])
y_train = np.concatenate([train_generator[i][1] for i in range(len(train_generator))])

X_val = np.concatenate([validate_generator[i][0] for i in range(len(validate_generator))])
y_val = np.concatenate([validate_generator[i][1] for i in range(len(validate_generator))])

X_test = np.concatenate([test_generator[i][0] for i in range(len(test_generator))])
y_test = np.concatenate([test_generator[i][1] for i in range(len(test_generator))])

# Custom Callback for Training Feedback
class ReconstructionAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_data, val_data):
        super(ReconstructionAccuracyCallback, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.train_accuracies = []
        self.val_accuracies = []

    def calculate_reconstruction_accuracy(self, original, reconstructed):
        mse = np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))
        return 100 - (mse.mean() * 100)

    def on_epoch_end(self, epoch, logs=None):
        train_reconstructed = self.model.predict(self.train_data)
        val_reconstructed = self.model.predict(self.val_data)

        train_accuracy = self.calculate_reconstruction_accuracy(self.train_data, train_reconstructed)
        val_accuracy = self.calculate_reconstruction_accuracy(self.val_data, val_reconstructed)

        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1} - Train Reconstruction Accuracy: {train_accuracy:.2f}%, Validation Reconstruction Accuracy: {val_accuracy:.2f}%")

# Train Autoencoder
callback = ReconstructionAccuracyCallback(X_train, X_val)
history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=20,
    batch_size=32,
    shuffle=True,
    callbacks=[callback]
)

# Plot Training and Validation Loss and Accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(callback.train_accuracies, label='Train Reconstruction Accuracy')
plt.plot(callback.val_accuracies, label='Validation Reconstruction Accuracy')
plt.title('Reconstruction Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()

# Final Reconstruction Accuracy
final_train_acc = callback.train_accuracies[-1]
final_val_acc = callback.val_accuracies[-1]

print(f"Final Reconstruction Accuracy - Train: {final_train_acc:.2f}%, Validation: {final_val_acc:.2f}%")

# Use Encoder for Feature Extraction
train_features = encoder.predict(X_train)
val_features = encoder.predict(X_val)
test_features = encoder.predict(X_test)
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# SVM with GridSearchCV
param_grid = {
    'C': [10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=2, verbose=2)
grid_search.fit(train_features, np.argmax(y_train, axis=1))

# Best Parameters
best_params = grid_search.best_params_
print(f"Best Parameters from GridSearchCV: {best_params}")

# Evaluate on Train, Validation, and Test Data
y_train_pred = grid_search.best_estimator_.predict(train_features)
y_val_pred = grid_search.best_estimator_.predict(val_features)
y_test_pred = grid_search.best_estimator_.predict(test_features)

train_accuracy = accuracy_score(np.argmax(y_train, axis=1), y_train_pred)
val_accuracy = accuracy_score(np.argmax(y_val, axis=1), y_val_pred)
test_accuracy = accuracy_score(np.argmax(y_test, axis=1), y_test_pred)

print(f"Recognition Accuracy - Validation: {val_accuracy * 100:.2f}%, Test: {test_accuracy * 100:.2f}%")

# Plot SVM Recognition Accuracy
plt.figure(figsize=(6, 4))
plt.bar(['Validation', 'Test'], [val_accuracy * 100, test_accuracy * 100], color=['orange', 'green'])
plt.title('SVM Recognition Accuracy')
plt.ylabel('Accuracy (%)')
plt.show()

