![Mind Map of Chapter 4 Concepts](https://github.com/user-attachments/assets/2a08c2e4-8cf4-4d27-9924-e8166154d40b)

# Training models of  Machine Learning
<h2>Linear Regression Model</h2>
<p>This project demonstrates the implementation of a Linear Regression Model to predict a target variable based on input features using Python. Key methods such as the Normal Equation and Pseudoinverse Approach (SVD) are used to find the model parameters. The Mean Squared Error (MSE) is used as the cost function to evaluate the performance of the model.</p>

<h2>Overview</h2>
<p>Linear regression models the relationship between the features (independent variables) and the target (dependent variable) using a linear equation:</p>
<p class="formula">$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n$$</p>

<p>Where:</p>
<ul>
    <li><code>ŷ</code> is the predicted value.</li>
    <li><code>θ₀</code> is the bias (intercept).</li>
    <li><code>θ₁, ..., θₙ</code> are the feature weights (slopes).</li>
    <li><code>x₁, ..., xₙ</code> are the feature values.</li>
</ul>

<h2>Cost Function</h2>
<p>The Mean Squared Error (MSE) is used as the cost function to measure how well the model fits the training data:</p>
<p class="formula">$$MSE(θ) = \frac{1}{m} \sum_{i=1}^{m} \left( θ^T x^{(i)} - y^{(i)} \right)^2$$</p>

<p>Where:</p>
<ul>
    <li><code>m</code> is the number of training examples.</li>
    <li><code>x^{(i)}</code> is the feature vector for the <code>i</code>-th example.</li>
    <li><code>y^{(i)}</code> is the actual target value for the <code>i</code>-th example.</li>
    <li><code>θ</code> represents the model parameters.</li>
</ul>

<p>The goal of the model is to minimize this cost function by finding the optimal values of <code>θ</code>.</p>

<h2>Methods to Find Optimal Parameters</h2>

<h3>1. Normal Equation</h3>
<p>The Normal Equation provides a closed-form solution to find the optimal parameters <code>θ</code> by minimizing the cost function:</p>
<p class="formula">$$θ = (X^T X)^{-1} X^T y$$</p>

<p>Where:</p>
<ul>
    <li><code>X</code> is the matrix of features.</li>
    <li><code>y</code> is the vector of target values.</li>
</ul>

<p>The Normal Equation avoids the need for iterative methods like Gradient Descent but can be computationally expensive for large datasets.</p>

<h3>2. Pseudoinverse Approach (SVD)</h3>
<p>Scikit-Learn uses Singular Value Decomposition (SVD) to compute the pseudoinverse of the feature matrix <code>X</code>:</p>
<p class="formula">$$θ = X^+ y$$</p>

<p>This method is more numerically stable and is reliable when the matrix <code>X^T X</code> is singular or non-invertible.</p>

<h2>Gradient Descent Optimization</h2>

<h3>Introduction</h3>
<p>Gradient descent is an optimization algorithm used to minimize a cost function. It iteratively updates model parameters to find the minimum value of the cost function.</p>

<h3>Analogy</h3>
<p>Imagine being lost in the mountains: you feel the slope of the ground and move in the direction of the steepest decline until you reach the lowest point (the valley). Gradient descent follows a similar approach by calculating the gradient (slope) of the cost function with respect to parameters and moving downhill.</p>

<h3>Random Initialization</h3>
<p>The optimization process begins with random values for the parameters (<code>θ</code>).</p>

<h3>Learning Rate (η)</h3>
<p>The size of each step taken towards the minimum is controlled by the learning rate:</p>
<ul>
    <li>If <code>η</code> is too small, convergence is slow.</li>
    <li>If <code>η</code> is too large, the algorithm may diverge and fail to converge.</li>
</ul>



## Random Initialization

The optimization process begins with random values for the parameters (θ).

## Learning Rate (η)

The size of each step taken towards the minimum is controlled by the learning rate:
- If **η** is too small, convergence is slow.
- If **η** is too large, the algorithm may diverge and fail to converge.

## Challenges in Gradient Descent

- **Local Minima**: The algorithm may converge to a local minimum instead of the global minimum.
- **Plateaus and Irregular Terrain**: The algorithm may take a long time to converge if the cost function has flat regions.

## Convexity

For linear regression, the mean squared error (MSE) cost function is convex, ensuring that there is only one global minimum.

## Feature Scaling

To facilitate faster convergence, ensure that all features have similar scales.

## Batch Gradient Descent

Batch gradient descent computes the gradient of the cost function using the entire training dataset at each step:
- It can be slow for large datasets but is efficient for high-dimensional feature spaces.

## Gradient Calculation

The partial derivatives of the cost function are computed for each parameter, leading to the gradient vector:
\[
\nabla_\theta MSE(\theta) = \frac{2}{m} X^T (X\theta - y)
\]

## Gradient Update Step

The model parameters are updated using the formula:
\[
\theta_{\text{next step}} = \theta - \eta \nabla_\theta MSE(\theta)
\]



- **Batch Gradient Descent**: Uses the entire dataset to calculate the gradients.
- **Stochastic Gradient Descent**: Uses one data point at a time, making it faster but with more variance.
- **Mini-batch Gradient Descent**: Uses a subset of data points to balance speed and variance.

## 3. L1 Regularization (Lasso)
**L1 Regularization** adds a penalty to the absolute values of the coefficients, encouraging sparsity. This technique tends to shrink some coefficients entirely to zero, effectively removing features from the model.

## 4. L2 Regularization (Ridge)
**L2 Regularization** penalizes the square of the magnitude of the coefficients. It reduces the size of the coefficients but does not eliminate any features completely, making it a useful technique for reducing overfitting without feature selection.

## 5. Training Methods
There are two primary methods for training machine learning models:
- **Closed-form solution**: Directly computes the optimal solution using matrix operations (e.g., Normal Equation for linear regression).
- **Gradient Descent**: An iterative method that updates parameters to minimize the cost function.

## 01. Linear Regression
**Linear Regression** models the relationship between a dependent variable and one or more independent variables using a linear equation. It assumes that the relationship between the input variables and the output variable is linear.
# Linear Regression Model

This project demonstrates the implementation of a Linear Regression Model to predict a target variable based on input features using Python. Key methods such as the Normal Equation and Pseudoinverse Approach (SVD) are used to find the model parameters. The Mean Squared Error (MSE) is used as the cost function to evaluate the performance of the model.

## Overview

Linear regression models the relationship between the features (independent variables) and the target (dependent variable) using a linear equation:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n
$$

Where:

- $\hat{y}$ is the predicted value.
- $\theta_0$ is the bias (intercept).
- $\theta_1, \ldots, \theta_n$ are the feature weights (slopes).
- $x_1, \ldots, x_n$ are the feature values.

## Cost Function

The Mean Squared Error (MSE) is used as the cost function to measure how well the model fits the training data:

$$
MSE(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2
$$

Where:

- $m$ is the number of training examples.
- $x^{(i)}$ is the feature vector for the $i$-th example.
- $y^{(i)}$ is the actual target value for the $i$-th example.
- $\theta$ represents the model parameters.

The goal of the model is to minimize this cost function by finding the optimal values of $\theta$.

## Methods to Find Optimal Parameters

### 1. Normal Equation

The Normal Equation provides a closed-form solution to find the optimal parameters $\theta$ by minimizing the cost function:

$$
\theta = (X^T X)^{-1} X^T y
$$

Where:

- $X$ is the matrix of features.
- $y$ is the vector of target values.

The Normal Equation avoids the need for iterative methods like Gradient Descent, but can be computationally expensive for large datasets.

### 2. Pseudoinverse Approach (SVD)

Scikit-Learn uses Singular Value Decomposition (SVD) to compute the pseudoinverse of the feature matrix $X$:

$$
\theta = X^+ y
$$

This method is more numerically stable and is reliable when the matrix $X^T X$ is singular or non-invertible.


- **Coefficients**: These represent the weights assigned to the input features.
- **Intercept**: The constant term that determines where the line crosses the y-axis.

## 03. Polynomial Regression
# High-Degree Polynomial Regression

High-degree polynomial regression can closely fit training data but may severely overfit, as illustrated by a 300-degree polynomial model compared to simpler models (linear and quadratic). The goal is to find a balance in model complexity that provides good generalization performance.

## Overfitting vs. Underfitting

- **Overfitting**: The model performs well on training data but poorly on validation data.
- **Underfitting**: The model performs poorly on both training and validation data.

## Learning Curves

Learning curves provide insight into how a model learns. They show how training error and validation error change as more training data is used. Plotting training and validation errors against the number of training instances helps identify these issues.

### Scikit-Learn's `learning_curve` Function

The `learning_curve` function in Scikit-Learn is useful for generating learning curves and assessing model performance. It allows you to visualize the training and validation errors as the model is trained on increasing amounts of training data.

## Bias-Variance Trade-Off

Understanding the bias-variance trade-off is crucial for model evaluation:

- **Bias**: Error due to incorrect assumptions in the model (e.g., assuming linearity when data is quadratic). High bias leads to underfitting.
- **Variance**: Error due to excessive sensitivity to training data variations. High variance leads to overfitting.
- **Irreducible Error**: Error inherent to the data itself, which cannot be reduced by model complexity.

### Trade-Off Dynamics

Increasing model complexity typically increases variance and decreases bias, while decreasing complexity has the opposite effect. Finding the right balance is essential for achieving optimal model performance.




## 8. Logistic Regression and Softmax Regression
- **Logistic Regression**: Used for binary classification tasks, logistic regression models the probability that a given input belongs to a specific class (0 or 1). It uses the logistic function to map predictions to probabilities.
- **Softmax Regression**: A generalization of logistic regression used for multi-class classification. It computes probabilities for each class, ensuring the sum of the probabilities equals 1.

## 9. Handling Nonlinearity
To model nonlinear relationships, additional polynomial features can be added to a linear regression model. This allows the model to fit curves, capturing more complex relationships between features and the target variable.

## 10. Overfitting
**Overfitting** occurs when a model captures noise and overly complex patterns in the training data, resulting in poor generalization to new data. Regularization techniques like L1 and L2 help prevent overfitting by adding penalties to large model coefficients, encouraging simpler models.

---


