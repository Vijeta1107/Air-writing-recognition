import os
import cv2
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf

ORIGINAL_ZIP_PATH = "/content/dataset1.zip"
DATASET_PATH = "/content/dataset1"

# 1. Extract Dataset
if not os.path.exists(DATASET_PATH):
    print(f"Extracting {ORIGINAL_ZIP_PATH}...")
    with zipfile.ZipFile(ORIGINAL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("/content/")
    print("Extraction complete.")

# 2. Load Dataset
X = []
y = []
classes = sorted(os.listdir(DATASET_PATH))
print("Classes:", classes)
label_map = {cls:i for i,cls in enumerate(classes)}

for cls in classes:
    folder = os.path.join(DATASET_PATH, cls)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Normalize to 0.0 - 1.0
        img = img / 255.0
        X.append(img)
        y.append(label_map[cls])

# NOTE: No transposition is done here. The model learns upright images.
X = np.array(X).reshape(-1, 28, 28, 1)
y = np.array(y)
print("Dataset shape:", X.shape)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# 4. Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15
)
datagen.fit(X_train)

# 5. Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train Model
print("Training Model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=40,
    validation_data=(X_test, y_test)
)

# 7. Convert and Save as TFLite (Float32 for reliability)
print("Converting to TFLite (Float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("air_model_optimized.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved optimized model as air_model_optimized.tflite! You can download this from Colab and put it in your Android assets.")
