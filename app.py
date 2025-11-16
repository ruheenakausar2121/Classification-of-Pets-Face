import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import tensorflow as tf
# We use try/except block for image loading to skip corrupt files safely
# Try multiple import locations to support different TF/Keras versions; fallback to PIL if needed.
try:
    # Preferred: tensorflow.keras.preprocessing (older TF) or tensorflow.keras.utils (newer TF)
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
    except Exception:
        raise ImportError("load_img and img_to_array not found in tensorflow.keras.preprocessing or tensorflow.keras.utils")
except Exception:
    try:
        # Fallback to standalone keras package if available
        from keras.preprocessing.image import load_img, img_to_array
    except Exception:
        # Final fallback: implement simple load_img/img_to_array using PIL
        from PIL import Image
        import numpy as np
        def load_img(path, target_size=None):
            img = Image.open(path).convert('RGB')
            if target_size is not None:
                img = img.resize((target_size[0], target_size[1]))
            return img
        def img_to_array(img):
            return np.array(img, dtype='float32')

# --- Configuration ---
# Your project folder is C:\Users\Ruheena\...\NEW project.
# main_archive_fp points to the 'archive' folder inside your project.
main_archive_fp = "./archive" 
IMAGE_SIZE = (96, 96) # Resize all images to 96x96 pixels

# --- 1. Dynamic Label Setup & File Discovery ---
label_map = {}
encoded_labels = 0
all_image_paths = []

# CRITICAL FIX: The deeper path is required due to the nested folder structure.
# This path is: ./archive/cat-breeds/cat-breeds/*
deeper_path = os.path.join(main_archive_fp, 'cat-breeds', 'cat-breeds', '*')

# Use glob to find all directories (which are the breed names) at this deepest level
breed_directories = [d for d in glob.glob(deeper_path) if os.path.isdir(d)]

# Dynamically create the label map
for breed_path in breed_directories:
    breed_name = os.path.basename(breed_path)
    
    # Assign a unique integer to each breed
    if breed_name not in label_map:
        label_map[breed_name] = encoded_labels
        encoded_labels += 1
    
    # Get all .jpg images within this breed folder 
    paths = glob.glob(os.path.join(breed_path, '*.jpg'))
    all_image_paths.extend(paths)

# Print the dynamic map for verification
print("--- Dynamic Label Map Created ---")
for breed, code in label_map.items():
    print(f"Breed: {breed}, Code: {code}")
print(f"Total breeds detected: {len(label_map)}")
print(f"Total images found: {len(all_image_paths)}")
print("-----------------------------------")

# --- 2. Label Encoding Function (using the dynamic map) ---
def label_encode(label):
    return label_map.get(label, -1) # Returns -1 if label not found

# --- 3. Main Data Loading Loop ---
features = [] # To store the image data (X)
labels = []   # To store the encoded labels (y)

for path in all_image_paths:
    # 1. Extract the string label (breed name) from the PARENT directory of the image
    label_str = os.path.basename(os.path.dirname(path))
    
    # 2. Encode the string label into an integer
    encoded_label = label_encode(label_str)
    
    if encoded_label != -1:
        try:
            # 3. Load the image and resize it
            img = load_img(path, target_size=IMAGE_SIZE)
            
            # 4. Convert to NumPy array
            img_array = img_to_array(img)
            
            # 5. Add the processed data
            features.append(img_array)
            labels.append(encoded_label)
            
        except Exception as e:
            # Skip corrupted or unreadable files
            print(f"Skipping error loading image {os.path.basename(path)}: {e}")
            
# --- 4. Final Preparation ---
# Convert the lists of data into NumPy arrays for machine learning
X = np.array(features, dtype='float32')
y = np.array(labels)

# Normalize the image data to a 0-1 range (0-255 is standard image range)
X = X / 255.0

print("\n--- Processing Complete ---")
print(f"Shape of Features (X): {X.shape}") # (Num_Images, 224, 224, 3)
print(f"Shape of Labels (y): {y.shape}")
from sklearn.model_selection import train_test_split
try:
    from tensorflow.keras.utils import to_categorical # type: ignore
except ImportError:
    from keras.utils import to_categorical

# --- Data Splitting and Encoding ---

# 1. Convert integer labels (y) to one-hot encoded format (e.g., 5 -> [0, 0, 0, 0, 0, 1, 0, ...])
# This is necessary for multi-class classification models.
num_classes = len(label_map) # 66 classes
y_one_hot = to_categorical(y, num_classes=num_classes)

# 2. Split the features (X) and one-hot labels (y_one_hot)
# We will use 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_one_hot, 
    test_size=0.2, # 20% for testing
    random_state=42, # Ensures the split is the same every time you run it
    shuffle=True,    # Randomly shuffles data before splitting
    stratify=y_one_hot # Ensures each set has roughly the same proportion of each cat breed
)

# 3. Print the shapes of the new arrays for verification
print("\n--- Data Split Complete ---")
print(f"Training Features (X_train) shape: {X_train.shape}")
print(f"Testing Features (X_test) shape: {X_test.shape}")
print(f"Training Labels (y_train) shape: {y_train.shape}")
print(f"Testing Labels (y_test) shape: {y_test.shape}")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
except ImportError:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# --- Model Configuration ---
# You can get these variables from the shapes printed previously
INPUT_SHAPE = (96, 96, 3) # The size of your input images
NUM_CLASSES = len(label_map) # 66

# --- 1. Define the CNN Model ---
def create_model():
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten the 3D output to 1D vector for the Dense layers
        Flatten(),
        
        # Fully Connected (Dense) Layers
        Dense(512, activation='relu'),
        Dropout(0.5), # Helps prevent overfitting
        
        # Output Layer: 66 units, one for each class. 
        # 'softmax' ensures the outputs are probabilities that sum to 1.
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# --- 2. Create and Compile the Model ---
cat_model = create_model()

# Compile the model with the appropriate settings for classification
cat_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Used for one-hot encoded labels
    metrics=['accuracy']
)

# 3. Print the model summary to confirm structure
print("\n--- Model Summary ---")
cat_model.summary()

# --- 4. Define Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 10 # You can increase this later for better results

# --- 5. Train the Model ---
print("\n--- Starting Model Training ---")
history = cat_model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

print("\n--- Training Complete ---")

import os

# Define the path to save the model file on your Desktop
# Note: This path points to the Desktop/NEW project folder.
model_save_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), # Base directory of app.py
    'cat_breed_classifier.keras'
)

# Save the model
try:
    cat_model.save(model_save_path)
    print(f"\nModel successfully saved to: {model_save_path}")
except Exception as e:
    print(f"\nError saving model: {e}")

    import matplotlib.pyplot as plt

# 1. Plot Training & Validation Accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 2. Plot Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot to an image file
plot_save_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'training_history_plot.png'
)
plt.savefig(plot_save_path)
print(f"Training history plot saved to: {plot_save_path}")

# --- Model Evaluation ---
loss, accuracy = cat_model.evaluate(X_test, y_test, verbose=1)

print(f"\nFinal Test Loss: {loss:.4f}")
print(f"Final Test Accuracy: {accuracy*100:.2f}%")