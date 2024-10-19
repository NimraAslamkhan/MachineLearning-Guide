# 📘 introduction to Machine Learning

## 📝 Chapter Overview
Welcome to the **Introduction to Machine Learning** chapter! This chapter introduces the basic principles of machine learning, covering the essential topics needed to understand and build machine learning models.
- [**/theory**](https://github.com/NimraAslamkhan/MachineLearning-Guide/blob/main/What%20machine%20learning%20is%2C%20problems%20it%20solves%2C%20and%20main%20categories/introduction%20to%20ML/theory.md) Contains theoretical notes, summaries of ML concepts such as generalization error, model tuning, and overfitting.
- - [**/coding**](https://github.com/NimraAslamkhan/MachineLearning-Guide/blob/main/What%20machine%20learning%20is%2C%20problems%20it%20solves%2C%20and%20main%20categories/introduction%20to%20ML/code_examples.ipynb): Contains the code examples, scripts for training, validation, and evaluation
## 📚 Topics Covered

### 🔍 1. What is Machine Learning?
Machine learning involves creating systems that **learn from data** instead of being explicitly programmed. The key focus is on improving system performance by learning from **training data**.

### 🧠 2. Types of Machine Learning Systems
- **Supervised Learning**: The model learns from labeled data and makes predictions.
- **Unsupervised Learning**: The model finds patterns or structures in data without labels.
- **Batch Learning**: The model is trained in one go using all data.
- **Online Learning**: The model is updated continuously as new data arrives.
- **Instance-based Learning**: The model memorizes examples and uses them for comparison.
- **Model-based Learning**: The model builds a general function that predicts outcomes based on data.

### 🧪 3. Training, Testing, and Validation
- **Training Set**: Used to fit the model.
- **Test Set**: Used to evaluate the model’s performance.
- **Validation Set**: Fine-tunes model parameters without overfitting.

### ⚖️ 4. Overfitting and Underfitting
- **Overfitting**: The model is too complex and learns noise, performing poorly on unseen data.
- **Underfitting**: The model is too simple, failing to capture patterns, resulting in poor performance on both training and unseen data.

### ⚙️ 5. Model Selection and Hyperparameter Tuning
- **Hyperparameter tuning** improves model performance by optimizing configurations like regularization, tree depth, and learning rates.
- **Example**: Tuning regularization in a linear model to reduce overfitting.

### 📉 6. Handling Data Mismatch
- **Data mismatch** occurs when training data doesn't represent real-world data. To handle this, you can create a **train-dev set** to better assess your model’s performance.

### 🍽 7. No Free Lunch Theorem
The **No Free Lunch (NFL) theorem** states that no single model works best for every problem. The ideal model depends on the specific dataset and assumptions.

## 🛠 Example: Model Selection Using Holdout Validation

Imagine you are building a model to predict housing prices. Here’s a breakdown of the process:

- **Data Split**: Divide 10,000 housing records into 80% training and 20% test data.
- **Model Training**: Train both a linear regression model and a decision tree model.
- **Validation**: Hold out 10% of the training data for validation.
- **Hyperparameter Tuning**: Adjust the decision tree depth to avoid overfitting.
- **Test Evaluation**: Test the final model on the unseen test data to estimate real-world performance.




