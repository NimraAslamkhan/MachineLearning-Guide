![Mind Map of Chapter 4 Concepts](https://github.com/user-attachments/assets/2a08c2e4-8cf4-4d27-9924-e8166154d40b)

# Training models of  Machine Learning


## 1. Closed-form Solution
A **closed-form solution** refers to calculating the optimal parameters directly using mathematical formulas, typically involving matrix operations. It is efficient for small datasets but becomes impractical for large datasets due to computational complexity.

## 2. Gradient Descent
**Gradient Descent** is an iterative optimization algorithm used to minimize the cost function by adjusting the model parameters. It computes the gradient (or slope) of the cost function and updates the parameters in the direction of the negative gradient to reduce the error.

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

## 6. Linear Regression
**Linear Regression** models the relationship between a dependent variable and one or more independent variables using a linear equation. It assumes that the relationship between the input variables and the output variable is linear.

- **Coefficients**: These represent the weights assigned to the input features.
- **Intercept**: The constant term that determines where the line crosses the y-axis.

## 7. Polynomial Regression
**Polynomial Regression** is an extension of linear regression that models nonlinear relationships by adding polynomial terms to the features. This allows the model to fit more complex curves and capture nonlinear patterns in the data.

## 8. Logistic Regression and Softmax Regression
- **Logistic Regression**: Used for binary classification tasks, logistic regression models the probability that a given input belongs to a specific class (0 or 1). It uses the logistic function to map predictions to probabilities.
- **Softmax Regression**: A generalization of logistic regression used for multi-class classification. It computes probabilities for each class, ensuring the sum of the probabilities equals 1.

## 9. Handling Nonlinearity
To model nonlinear relationships, additional polynomial features can be added to a linear regression model. This allows the model to fit curves, capturing more complex relationships between features and the target variable.

## 10. Overfitting
**Overfitting** occurs when a model captures noise and overly complex patterns in the training data, resulting in poor generalization to new data. Regularization techniques like L1 and L2 help prevent overfitting by adding penalties to large model coefficients, encouraging simpler models.

---


