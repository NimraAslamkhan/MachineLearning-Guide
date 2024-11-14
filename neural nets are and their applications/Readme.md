# Artificial Neural Networks (ANNs)
![image](https://github.com/user-attachments/assets/38bef2ad-85e6-41c2-9b11-d070c26eb8f7)


## Introduction
Artificial neural networks (ANNs) are inspired by the brain’s structure, with networks of biological neurons as a model. However, ANNs have evolved beyond their biological inspiration. They are the foundation of deep learning, excelling at tasks like image classification, speech recognition, and complex games such as Go.

This document provides an overview of ANN development, followed by an introduction to Multilayer Perceptrons (MLP) using TensorFlow's Keras API.

## Historical Background
- **1943**: First ANN model introduced by Warren McCulloch and Walter Pitts, modeling how neurons might work in animal brains.
- **1960s**: Initial progress led to overestimated expectations about intelligent machines.
- **1980s**: Renewed interest with new architectures and training methods.
- **1990s**: Shifted focus to other machine learning methods.
- **Modern Day**: Increased data availability, computing power, and improved algorithms have made ANNs highly effective and scalable.

## Biological Neural Networks (BNNs)
- Neurons, the building blocks of BNNs, form complex networks by connecting through synapses.
- BNNs are often organized in layers, which perform basic logical operations (AND, OR, NOT).
- These networks are capable of intricate computations due to their highly interconnected structure.

## Artificial Neural Networks (ANNs) Fundamentals
### Perceptron
- The **Perceptron** is a basic ANN unit introduced by Frank Rosenblatt in 1957.
- **Linear Combination of Inputs**: Computes a weighted sum of inputs and applies a step function.
- **Binary Classification**: Useful for separating data into two classes.
- **Architecture**: Single-layer perceptrons form dense layers, ideal for binary and multiclass classification tasks.

### Training the Perceptron
- The perceptron adjusts weights to reduce errors, following **Hebbian learning** principles.
- **Perceptron Convergence Theorem**: If data is linearly separable, the algorithm will converge.

## Limitations and Multilayer Perceptrons (MLPs)
- Perceptrons cannot solve non-linear problems like XOR. This limitation led to the development of **Multilayer Perceptrons (MLPs)**.
- **MLPs** can solve non-linear problems by adding hidden layers, enabling complex decision boundaries and forming the basis of modern deep learning models.

## Multilayer Perceptron (MLP) Overview
- **Architecture**:
  - Input Layer: Receives input data.
  - Hidden Layers: Uses TLUs with non-linear activation functions.
  - Output Layer: Provides predictions.
- **Deep Neural Networks (DNNs)**: MLPs with multiple hidden layers are called DNNs.

## Key Concepts in MLPs
1. **Backpropagation**: Trains the network by computing gradients and performing updates.
   - **Reverse-Mode Automatic Differentiation**: Efficiently calculates gradients through forward and backward passes.
2. **Activation Functions**: Add non-linearity, allowing for complex patterns.
   - Common functions include **Sigmoid**, **Tanh**, and **ReLU**.

### MLP for Regression (MLPRegressor)
- **Regression Tasks**: MLPs can have one or more output neurons.
- **Preprocessing**: Normalization is essential.
- **Common Settings**: ReLU for hidden layers and Adam optimizer for gradient descent.

### MLPs for Classification
- **Binary Classification**: Single output neuron with sigmoid activation.
- **Multilabel Classification**: Each label has its own output neuron with sigmoid activation.
- **Multiclass Classification**: Multiple output neurons with softmax activation to ensure outputs sum to 1.
- **Loss Function**: Cross-entropy loss for improved probability outputs.

## Implementing MLPs with Keras
Keras provides an intuitive way to create and train neural networks. Here’s an example using Fashion MNIST for image classification:

### Dataset
- Fashion MNIST includes grayscale images representing clothing items.
- **Normalization**: Scale pixel values to 0–1 range.

### Building the Model
```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(300, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
Training the Model
```python
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

```
Evaluating the Model
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
```

MLPs are versatile neural networks adaptable for various tasks, with Keras simplifying their implementation. By understanding their structure, activation functions, and training methods, MLPs become valuable tools for solving both regression and classification problems in machine learning.


```python
