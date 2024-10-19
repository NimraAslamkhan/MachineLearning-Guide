# California Housing Prices Model

Welcome to the **California Housing Prices Prediction** project! In this project, we aim to predict the median housing prices of districts in California using census data, including metrics such as population, median income, and median house value. The model you develop will help a Machine Learning system assess the investment potential of different districts. 

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Framing](#problem-framing)
- [Data Pipeline](#data-pipeline)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [Model Deployment](#model-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Exercises](#exercises)

## Project Overview

This project uses California census data to predict the median housing price in districts across the state. The aim is to build a machine learning model that can predict housing prices based on various features such as:
- **Population**: Number of people in each district.
- **Median Income**: Average income of households.
- **Housing Features**: Including the number of rooms, age of houses, etc.

The model will be utilized to determine whether a specific area is worth investing in, which could impact revenue for the business.

## Problem Framing

The task is framed as a **supervised regression problem**, where:
- **Objective**: Predict the **median house price** in each district.
- **Input features**: Metrics such as median income, population, number of rooms, etc.
- **Output**: Predicted median housing price.
  
The data is relatively small, making this suitable for **batch learning**. We'll focus on building a **multiple regression model**, which will predict one value for each district. 

## Data Pipeline

The data preparation and transformation process is critical to building a robust model. Here is an overview of the pipeline:

1. **Data Loading**: Load the California housing dataset.
2. **Data Cleaning**: Handle missing values, outliers, and errors in the dataset.
3. **Feature Engineering**:
   - Creating new features (e.g., rooms per household, population per household).
   - Normalizing and scaling data.
   - Handling categorical data using one-hot encoding.
4. **Splitting the Data**: Divide the data into training and test sets.
5. **Transformations**: Apply transformations to ensure the data is in the right format for modeling (e.g., scaling numeric features).

This pipeline allows you to easily modify and expand the preprocessing steps without changing the core model code.

## Modeling and Evaluation

We experiment with various regression algorithms to find the best model for predicting housing prices:
- **Linear Regression**: A baseline model to predict house prices.
- **Decision Tree Regression**: A non-parametric model to capture non-linear relationships.
- **Random Forest**: An ensemble model that averages multiple decision trees.
- **Support Vector Regressor (SVR)**: Using different kernels and hyperparameters for performance tuning.

### Hyperparameter Tuning
For model optimization, we use:
- **GridSearchCV**: To tune hyperparameters and find the best model configuration.
- **RandomizedSearchCV**: To explore hyperparameter space efficiently.

### Model Evaluation
We evaluate the models based on metrics such as:
- **Mean Squared Error (MSE)**: To measure the average squared difference between predicted and actual values.
- **R² Score**: To assess how well the model explains variance in the data.
- **Cross-validation**: To validate the model's performance on different subsets of the data.

## Model Deployment

Several deployment strategies are explored:

1. **Web Application Integration**:
   - The model can be integrated into a web application where users input district data and receive predictions for housing prices.
   
2. **REST API**:
   - The model can be wrapped in a REST API, enabling easy integration into any front-end or mobile app.

3. **Cloud Deployment**:
   - Deploy the model on platforms like **Google Vertex AI**, which provides load balancing, scaling, and monitoring capabilities.
   - Use `joblib` to save the model and store it in Google Cloud Storage for easy retrieval and usage.

## Monitoring and Maintenance

After deployment, it’s important to monitor the model's performance regularly and ensure that it continues to perform well as data evolves.

### Monitoring Techniques:
- **Model Performance**: Track live performance metrics and compare them with the model's baseline performance.
- **Input Data Quality**: Monitor the data pipeline to detect issues in data quality (e.g., missing features or abnormal distributions).
- **Regular Retraining**: Automate the process of updating datasets and retraining models as new data becomes available.

### Backup and Rollback
- Backup every version of the dataset and model for quick rollback in case of issues with newly deployed models.

## Exercises

To further explore the project, here are some exercises you can try:
1. **Support Vector Regressor (SVR)**:
   - Try an SVR with different hyperparameters and evaluate its performance.
2. **Randomized Search**:
   - Use `RandomizedSearchCV` instead of `GridSearchCV` for hyperparameter tuning.
3. **Feature Selection**:
   - Add a `SelectFromModel` step in the pipeline to select the most important features.
4. **Custom Transformer**:
   - Build a custom transformer to use K-Nearest Neighbors (KNN) for predicting prices based on nearby districts.
5. **Experiment with Preprocessing Options**:
   - Use `GridSearchCV` to automate exploration of different preprocessing techniques.

## Conclusion

This project provides an end-to-end solution for predicting California housing prices using census data, including data preprocessing, model building, hyperparameter tuning, deployment, and monitoring. By following the steps outlined, you’ll have a robust machine learning pipeline capable of making accurate predictions in a real-world business scenario.

