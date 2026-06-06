# Exercise Posture Classification System

## Project Overview

This project develops an Exercise Posture Classification System using Deep Learning techniques. The goal is to classify exercise images into different workout categories using Convolutional Neural Networks (CNNs) and Transfer Learning models.

The project was completed as part of the Machine Learning course in the MSc Data Science program.

---

## Dataset

Dataset Name:
Workout / Exercise Images Dataset

Source:
Kaggle

Dataset Link:
https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images

### Dataset Statistics

* Total Images: 13,853
* Total Classes: 22
* Image Type: RGB Images

Classes include:

* Barbell Biceps Curl
* Bench Press
* Chest Fly Machine
* Deadlift
* Hammer Curl
* Hip Thrust
* Incline Bench Press
* Lat Pulldown
* Lateral Raises
* Leg Extension
* Leg Raises
* Plank
* Pull Up
* Push Up
* Romanian Deadlift
* Russian Twist
* Shoulder Press
* Squat
* T-Bar Row
* Tricep Dips
* Tricep Pushdown
* Decline Bench Press

---

## Project Objectives

* Build a Custom CNN model for exercise classification.
* Compare CNN performance with Transfer Learning models.
* Evaluate model performance using multiple metrics.
* Visualize results using confusion matrices and learning curves.
* Develop a Streamlit application for image prediction.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* PIL
* Streamlit

---

## Models Implemented

### Custom CNN

Custom architecture consisting of:

* Convolution Layers
* MaxPooling Layers
* Dense Layers
* Dropout Layer
* Softmax Output Layer

### MobileNetV2

Transfer Learning model using pretrained ImageNet weights.

### ResNet50

Transfer Learning model using pretrained ImageNet weights.

---

## Model Performance

| Model       | Test Accuracy |
| ----------- | ------------- |
| Custom CNN  | 96.44%        |
| MobileNetV2 | 93.51%        |
| ResNet50    | 97.55%        |

Best Performing Model:
**ResNet50**

---

## Generated Outputs

The project generates:

* Class Distribution Chart
* Accuracy Curves
* Loss Curves
* Confusion Matrices
* Classification Reports
* Model Comparison Table
* Grad-CAM Visualizations

---

## Streamlit Application

The project includes a Streamlit application that allows users to:

* Upload exercise images
* Predict exercise category
* Display confidence scores
* View Top-5 predictions

Run the application:

```bash
streamlit run app.py
```

---

## Repository Structure

```text
Exercise-Posture-Classification/

│
├── exercise-posture-classification.ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── exercise_posture_best_model.h5
│
├── exercise_posture_custom_cnn.keras
│
├── class_names.json
│
├── outputs/
│   ├── confusion_matrix.pdf
│   ├── accuracy_curve.pdf
│   ├── loss_curve.pdf
│   ├── gradcam.pdf

```

---

## How to Run

### Step 1

Clone the repository:

```bash
git clone YOUR_GITHUB_LINK
```

### Step 2

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 3

Run the notebook or Streamlit application.

Notebook:

```bash
jupyter notebook
```

Streamlit:

```bash
streamlit run app.py
```

---

## Author

Aswathy Kariyeri

MSc Data Science

Machine Learning Project

Exercise Posture Classification System
