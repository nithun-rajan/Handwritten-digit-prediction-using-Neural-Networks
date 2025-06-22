# 🧠 Handwritten Digit Recognition using Neural Networks

This project demonstrates a deep learning pipeline to recognize handwritten digits using the MNIST dataset. Leveraging TensorFlow and Keras, the model classifies 28x28 grayscale images of digits (0–9) with high accuracy.

---

## 📌 Objective

To build, train, evaluate, and visualize a neural network that can accurately classify handwritten digits from the MNIST dataset using TensorFlow/Keras.

---

## 🗃️ Dataset

- **MNIST Dataset**  
  - 60,000 training images  
  - 10,000 testing images  
  - Each image: 28x28 grayscale  
  - Digit labels: 0 to 9  

This dataset is widely used in computer vision benchmarks for evaluating image classification models.

---

## 🧠 Model Architecture

The model is built using Keras’ `Sequential` API:

- Input layer: `Flatten` 28x28 images  
- Hidden layers: `Dense`, `ReLU` activations  
- Output layer: `Dense` with 10 neurons (softmax activation)  

**Loss function:** `sparse_categorical_crossentropy`  
**Optimizer:** `adam`  
**Metric:** `accuracy`

---

## 📈 Performance Summary

- **Test Accuracy:** ~97.7%
- **Confusion Matrix:**  
  The confusion matrix shows strong classification across most digits, with minor misclassifications particularly in digits like `5` and `3`.

![Confusion Matrix](https://raw.githubusercontent.com/nithun-rajan/Handwritten-digit-prediction-using-Neural-Networks/main/Figure_2.png)

---

## 🔍 Sample Predictions

### 🖼️ Predicted Digit - '5'
![Digit 5 Prediction](https://raw.githubusercontent.com/nithun-rajan/Handwritten-digit-prediction-using-Neural-Networks/main/Figure_1.png)

### 🖼️ True Digit - '3' from Dataset
![Digit 3 Image](https://raw.githubusercontent.com/nithun-rajan/Handwritten-digit-prediction-using-Neural-Networks/main/ok.png)

---

## 🗂 File Structure

