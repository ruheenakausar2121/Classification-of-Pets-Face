
😻 Cat Breed Classifier: Deep Learning Solution

Automated classification is crucial in a world flooded with digital content. This project delivers a robust solution for fine-grained image recognition, capable of identifying 66 distinct cat breeds from raw image data using a Convolutional Neural Network (CNN).

🌟 Project Overview

This repository contains a self-contained Python solution for:

Data preprocessing: resizing, normalization, and organization.

Model training: building a CNN from scratch with TensorFlow/Keras.

Prediction: inference on new images using the trained model.

The dataset comprises 11,276 images across 66 unique cat breeds.

⚙️ Technologies Used

Python 3.x

TensorFlow / Keras – CNN modeling and training

NumPy – efficient array operations

Scikit-learn – train/test splitting

Matplotlib – visualizing training performance

🖼 Dataset Details

Source: External image archive

Total Images: 11,276

Preprocessing:

Resized to 96x96 pixels

Normalized to a 0-1 range

🏗 Model Architecture

Custom sequential CNN designed for multi-class classification:

Three Conv2D layers with MaxPooling (feature extraction)

Flatten layer

Dense layer (512 units) with Dropout (0.5) to prevent overfitting

Output Dense layer (66 units) with Softmax activation

🎯 Performance
Metric	Result
Classes	66
Final Test Accuracy	13.79%
Training Accuracy	64%

⚠️ Observation: The model overfits. Adding data augmentation and more training data is recommended for production-level accuracy.

💻 How to Run
1. Install Dependencies
pip install numpy pandas matplotlib tensorflow pillow scikit-learn

2. Prepare Data

Organize your dataset as:

project_root/
│
├─ app.py
└─ archive/
   └─ cat-breeds/
      └─ [BreedName]/
         └─ image1.jpg
         └─ image2.jpg
         ...

3. Execute Script
python app.py


The script handles:

Data loading & preprocessing

Model definition & training

Saving the trained model

🏆 Results & Outputs

After training, the following will be generated:

cat_breed_classifier.keras – Trained model ready for inference

training_history_plot.png – Accuracy & loss plot over epochs

predict_breed.py – Optional script for testing new images

📈 Future Improvements

Implement data augmentation to reduce overfitting

Increase dataset size for better generalization

Experiment with transfer learning using pre-trained models
