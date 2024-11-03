# 1. Initialize weights (W) and bias (b) to 0 (or small random values)
# 2. For each iteration:
#    a. Compute weighted input: Z = W*X + b
#    b. Apply the sigmoid function: A = 1 / (1 + e^(-Z))
#    c. Calculate the cost (Cross-Entropy Loss): cost = -sum(y*log(A) + (1-y)*log(1-A)) / N
#    d. Compute gradients: dW = X*(A - y) / N, db = sum(A - y) / N
#    e. Update parameters: W = W - learning_rate * dW, b = b - learning_rate * db
# 3. Repeat until convergence or for a fixed number of iterations.


#SOLUTION
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('/..../..../health_metrics_dataset.csv')

# Define features and target variable
X = df[['Age', 'BMI', 'Blood_Pressure', 'Cholesterol']].values
y = df['Disease'].values

# Data Preprocessing
# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def logistic_regression(X, y, num_iterations, learning_rate):
    m, n = X.shape
    W = np.zeros(n)
    b = 0
    
    for i in range(num_iterations):
        z = np.dot(X, W) + b
        y_pred = sigmoid(z)
        loss = compute_loss(y, y_pred)
        
        dW = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        W -= learning_rate * dW
        b -= learning_rate * db
        
    return W, b

# Model training
num_iterations = 1000
learning_rate = 0.01
W, b = logistic_regression(X_train, y_train, num_iterations, learning_rate)

# Making predictions
z_test = np.dot(X_test, W) + b
predictions = sigmoid(z_test) >= 0.5

# Model Evaluation
print(f'Accuracy: {accuracy_score(y_test, predictions)}')
print(f'Precision: {precision_score(y_test, predictions)}')
print(f'Recall: {recall_score(y_test, predictions)}')
print(f'F1 Score: {f1_score(y_test, predictions)}')



#GRADE ALLOTMENT
""" 
Data Preprocessing (15 points): For correctly normalizing the features and splitting the dataset.

Model Implementation (20 points): For accurately implementing the logistic regression model and sigmoid function.

Cost Function and Optimization (25 points): For correctly calculating the cost function and applying gradient descent.

Prediction Accuracy (20 points): Based on the accuracy of the model on a test dataset.

Model Evaluation (20 points): For comprehensive evaluation using metrics such as accuracy, precision, recall, or F1 score. 
"""