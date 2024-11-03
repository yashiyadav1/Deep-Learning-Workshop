# 1. Initialize parameters (m, c) to 0 (or a small random value)
# 2. For each iteration:
#    a. Compute predicted values: y_pred = m*x + c
#    b. Calculate the cost (Mean Squared Error): cost = sum((y_pred - y_actual)^2) / N
#    c. Compute gradients: dm = sum(2*x*(y_pred - y_actual)) / N, dc = sum(2*(y_pred - y_actual)) / N
#    d. Update parameters: m = m - learning_rate * dm, c = c - learning_rate * dc
# 3. Repeat until convergence or for a fixed number of iterations.


#SOLUTION
import numpy as np
import matplotlib.pyplot as plt

# Simulating a dataset
np.random.seed(42) # Ensuring reproducibility
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Plotting the dataset
plt.scatter(X, y)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()

# Building the Linear Regression model
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
learning_rate = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients

print("Theta:", theta)

# Making predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta)

# Plotting predictions
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Predicted vs Actual Scores")
plt.show()



#GRADE ALLOTMENT
""" 
Data Understanding and Preprocessing (20 points): Proper visualization and data scaling.

Implementation of Linear Regression Equation (20 points): Correctly setting up the equation for linear regression.

Cost Function Calculation (20 points): Correctly calculating the mean squared error.

Gradient Descent Implementation (20 points): Proper implementation of the gradient descent algorithm including correct computation 
                                             of gradients and parameter updates.

Prediction and Model Evaluation (20 points): Using the final model to make predictions and evaluating its performance using a 
                                             test set (if provided) or through the final cost value. 
"""