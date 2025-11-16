import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Configuration (MUST MATCH TRAINING) ---
IMAGE_SIZE = (96, 96) 

# --- IMPORTANT: Label Mapping ---
# This list MUST be in the same order as your 'Code' output (0 through 65).
# I created this based on the alphabetical order found in your training log.
LABEL_NAMES = [
    'abyssinian', 'american_bobtail', 'american_curl', 'american_shorthair', 
    'american_wirehair', 'balinese', 'bengal', 'birman', 'bombay', 
    'british_shorthair', 'burmese', 'chartreux', 'chausie', 'cornish_rex', 
    'cymric', 'cyprus', 'devon_rex', 'donskoy', 'egyptian_mau', 
    'european_shorthair', 'exotic_shorthair', 'german_rex', 'havana_brown', 
    'himalayan', 'japanese_bobtail', 'karelian_bobtail', 'khao_manee', 
    'korat', 'korean_bobtail', 'kurilian_bobtail', 'laperm', 'lykoi', 
    'maine_coon', 'manx', 'mekong_bobtail', 'munchkin', 'nebelung', 
    'norwegian_forest_cat', 'ocicat', 'oregon_rex', 'oriental_shorthair', 
    'persian', 'peterbald', 'pixie_bob', 'ragamuffin', 'ragdoll', 
    'russian_blue', 'safari', 'savannah', 'scottish_fold', 'selkirk_rex', 
    'serengeti', 'siamese', 'siberian', 'singapura', 'sokoke', 
    'somali', 'sphynx', 'thai', 'tonkinese', 'toyger', 
    'turkish_angora', 'turkish_van', 'ukrainian_levkoy', 'ural_rex', 'vankedisi'
]

# --- 1. Load the Model ---
# Assumes the model is saved in the same directory as this script.
model_path = os.path.join(os.path.dirname(__file__), 'cat_breed_classifier.keras')

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure 'cat_breed_classifier.keras' is in the same folder.")
    exit()

# --- 2. Preprocess and Predict Function ---
def predict_new_image(image_path):
    print(f"\nProcessing image: {image_path}")
    
    # 2.1 Load and Resize Image
    img = load_img(image_path, target_size=IMAGE_SIZE)
    
    # 2.2 Convert to Array and Normalize (0-1 range)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    
    # 2.3 Add batch dimension (model expects batch of images)
    # Shape changes from (96, 96, 3) to (1, 96, 96, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 2.4 Make Prediction
    predictions = model.predict(img_array)
    
    # 2.5 Find the best prediction
    # predictions is an array of 66 probabilities. argmax finds the index (code).
    predicted_code = np.argmax(predictions[0])
    predicted_breed = LABEL_NAMES[predicted_code]
    confidence = predictions[0][predicted_code] * 100
    
    print("\n--- Prediction Result ---")
    print(f"Predicted Breed: {predicted_breed.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Optional: print top 3
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    print("\nTop 3 Predictions:")
    for i in top_3_indices:
        breed = LABEL_NAMES[i]
        conf = predictions[0][i] * 100
        print(f" - {breed.capitalize()}: {conf:.2f}%")

# --- 3. Example Usage ---
# CRITICAL: Place a new JPG image (e.g., 'new_cat.jpg') in your NEW project folder
# Then, replace 'example_cat.jpg' with the actual filename.
new_image_file = 'example_cat.jpg' 

# Make sure the image exists before running!
if os.path.exists(new_image_file):
    predict_new_image(new_image_file)
else:
    print("\n!!! ERROR !!!")
    print(f"Please place an image named '{new_image_file}' in your folder and try again.")