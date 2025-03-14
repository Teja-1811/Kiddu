import tensorflow as tf
import os
import tarfile
import wget
import shutil
import scipy  # Ensure scipy is available
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 37  # Adjust based on dataset (Stanford Dogs or Oxford Pets)
EPOCHS = 10

# Function to validate images
def is_valid_image(file_path):
    try:
        img = load_img(file_path)  # Try loading the image
        return True
    except:
        return False

# Function to clean corrupt images
def clean_corrupt_images(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if not is_valid_image(file_path):
                print(f"Removing corrupt image: {file_path}")
                os.remove(file_path)

# Ensure dataset folder exists without deleting manually added images
if not os.path.exists("dataset"):
    os.makedirs("dataset", exist_ok=True)
    print("Dataset folder created. Please add images manually.")

# Remove corrupt images before training
clean_corrupt_images("dataset")

print("Images successfully organized into breed folders!")

# Load EfficientNetB3 without top layers
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

# Add custom classification head
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation and dataset loading
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

def safe_flow_from_directory(datagen, directory, subset):
    return datagen.flow_from_directory(directory, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                       class_mode='categorical', subset=subset)

train_generator = safe_flow_from_directory(datagen, "dataset", subset='training')
val_generator = safe_flow_from_directory(datagen, "dataset", subset='validation')

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save the fine-tuned model
model.save('model/breed_classifier.h5')
