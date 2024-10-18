
# Machine Learning Overview

## What Is Machine Learning?

Machine learning is the **science (and art)** of programming computers so they can learn from data.

The part of a machine learning system that **learns and makes predictions** is called a **model**. 

**Neural Networks** and **Random Forests** are examples of models.

## Why Use Machine Learning?

### Automated Learning

ML-based spam filters automatically detect patterns in spam emails by comparing word frequencies in spam vs. ham (non-spam) emails. It adapts over time as new spam patterns emerge (e.g., detecting "For U" without manual intervention).

### Better Performance & Simplicity

ML models are **easier to maintain**, **shorter in code**, and **generally more accurate** because they continuously improve based on the data.

## Data Mining

ML models can be inspected to understand what features (e.g., certain words) are the best predictors for spam, often revealing new trends. This process of discovering hidden patterns in large datasets is called **data mining**.

## Examples of Applications

1. **Image Classification**
    - Typically performed using **Convolutional Neural Networks (CNNs)**.
    - *Example:* Classifying images of animals, cars, etc.

2. **Semantic Image Segmentation**
    - Each pixel in the image is classified to determine the exact location and shape of objects (e.g., tumors).
    - Typically uses **CNNs** or **Transformers**.
    - *Example:* Medical imaging to locate tumors.

3. **Natural Language Processing (NLP)**
    - Specifically **Text Classification**.
    - Can be tackled using **Recurrent Neural Networks (RNNs)**, **CNNs**, or **Transformers** for better performance.
    - *Example:* Sentiment analysis, spam detection.

## Types of Machine Learning Systems

### Based on Training Supervision

- **Supervised Learning**
    - Involves labeled data to train models for tasks like classification and regression.
    - *Example:* Spam detection, house price prediction.

- **Unsupervised Learning**
    - Deals with unlabeled data to find hidden patterns, such as clustering and dimensionality reduction.
    - *Example:* Customer segmentation, anomaly detection.

- **Semi-Supervised Learning**
    - Combines a small amount of labeled data with a large amount of unlabeled data.
    - *Example:* Image classification with limited labeled images.

- **Self-Supervised Learning**
    - Generates labels from the data itself to train models without external labeling.
    - *Example:* Predicting the next word in a sentence.

- **Reinforcement Learning**
    - Trains agents to make sequences of decisions by maximizing cumulative rewards.
    - *Example:* Game AI, robotic control.

### Based on Learning Incrementally

- **Online Learning**
    - Models learn continuously by updating with new data on the fly.
    - *Example:* Real-time recommendation systems.

- **Batch Learning**
    - Models are trained on a fixed dataset all at once and require retraining with new data.
    - *Example:* Traditional spam filters.

### Based on Learning Approach

- **Instance-Based Learning**
    - Models memorize training instances and make predictions by comparing new data to these instances.
    - *Example:* k-Nearest Neighbors (k-NN).

---

## Project Structure

Model-Based Learning: Models generalize patterns from the training data to make predictions (e.g., linear regression, neural networks).
