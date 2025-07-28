import numpy as np
import pandas as pd # Although not strictly needed for eval, it's in your notebook
import cv2
import os
from os import listdir
from PIL import Image # For array_to_img later, if needed
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model # Corrected import for load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # Corrected import

print(tf.__version__)

# --- Configuration matching your notebook ---
DATASET_ROOT = r"C:\Users\Dell INSPIRON\Plant-Disease-Detection\Dataset"
IMAGE_SIZE = (256, 256)
NORMALIZATION_FACTOR = 225.0 # Note: Your notebook uses 225.0, not 255.0
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 10
ALL_LABELS = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
BINARY_LABELS = [0, 1, 2]

# --- Function to convert image to array (from your notebook) ---
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, IMAGE_SIZE)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error converting image {image_dir}: {e}")
        return None

# --- Load and preprocess data (from your notebook) ---
print("Loading and preprocessing data...")
root_dir = listdir(DATASET_ROOT)
image_list, label_list = [], []
temp = -1

for directory in root_dir:
    plant_image_list = listdir(f"{DATASET_ROOT}/{directory}")
    temp += 1
    for files in plant_image_list:
        image_path = f"{DATASET_ROOT}/{directory}/{files}"
        img_array = convert_image_to_array(image_path)
        if img_array is not None and img_array.size > 0:
            image_list.append(img_array)
            label_list.append(BINARY_LABELS[temp])

label_list = np.array(label_list)

# Split data into training and test sets
# Note: x_train is further split into train/val in your notebook, but for model evaluation,
# we need the original x_test and y_test after the first split.
x_train_full, x_test_raw, y_train_full, y_test_raw = train_test_split(
    image_list, label_list, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE
)

# Normalize and reshape test data
x_test = np.array(x_test_raw, dtype=np.float16) / NORMALIZATION_FACTOR
x_test = x_test.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
y_test = to_categorical(y_test_raw, num_classes=len(ALL_LABELS)) # Ensure num_classes matches your problem

print(f"Loaded X_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# --- Evaluate Model Function ---
def evaluate_model(model_path, X_test_data, y_test_data):
    print(f"\nEvaluating model: {model_path}")
    try:
        # Important: Your model was saved at C:\Plant-Disease-Detection\Model\
        # Adjust path if you move the model
        full_model_path = os.path.join(r"C:\Plant-Disease-Detection\Model", model_path)
        
        model = load_model(full_model_path) # Use keras.models.load_model
        
        loss, accuracy = model.evaluate(X_test_data, y_test_data, verbose=1)
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        return accuracy
    except Exception as e:
        print(f"  Could not load or evaluate {full_model_path}: {e}")
        print(f"  Please ensure the model path is correct and the model file '{model_path}' exists at 'C:\\Plant-Disease-Detection\\Model\\'.")
        return -1 # Indicate failure

# --- Evaluate both models ---
model1_name = 'plant_disease_model.h5'
model2_name = 'plant_disease_model_1.h5' # Assuming this is the name of your other model

acc1 = evaluate_model(model1_name, x_test, y_test)
acc2 = evaluate_model(model2_name, x_test, y_test)

# --- Comparison Results ---
print("\n--- Comparison Results ---")
if acc1 > acc2:
    print(f"'{model1_name}' performed better with accuracy: {acc1:.4f}")
    print(f"Consider using '{model1_name}' in your main_app.py.")
elif acc2 > acc1:
    print(f"'{model2_name}' performed better with accuracy: {acc2:.4f}")
    print(f"Consider using '{model2_name}' in your main_app.py.")
elif acc1 == -1 or acc2 == -1:
    print("One or both models could not be evaluated. Please check paths and data.")
else:
    print(f"Both models performed similarly (accuracy: {acc1:.4f}). You can choose either.")