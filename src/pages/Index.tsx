import React, { useState } from 'react';
import JupyterHeader from '@/components/JupyterHeader';
import JupyterMenu from '@/components/JupyterMenu';
import JupyterToolbar from '@/components/JupyterToolbar';
import CodeCell from '@/components/CodeCell';

interface CellData {
  id: number;
  code: string;
  cellType: 'code' | 'markdown';
}

const Index = () => {
  // Initial cell data
  const initialCells: CellData[] = [
    { id: 1, code: `<h1>EX -5 LogisiticRegression</h1>`, cellType: 'markdown' },
    { id: 2, code: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_wine

# Load wine dataset from sklearn
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Binary classification: Class 0 vs Rest
y_binary = (y == 0).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Rest', 'Class 0'], yticklabels=['Rest', 'Class 0'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Sigmoid Curve Visualization
z = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid, label='Sigmoid Curve')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.show()

-------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import load_wine

# Load wine dataset from sklearn
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Binary classification: Class 0 vs Rest
y_binary = (y == 0).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Sigmoid Curve Visualization
z = np.linspace(-10, 10, 1000)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid, label='Sigmoid Curve')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('Probability')
plt.grid(True)
plt.legend()
plt.show()

-----------------------------------------------------------------------------

from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Load wine dataset
win = datasets.load_wine()

# Data and target values
X = win.data
y = win.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

# Train logistic regression model
log_reg_model = linear_model.LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
log_reg_base_score = log_reg_model.score(X_test, y_test)
print("The score for the Logistic Regression Model is:", log_reg_base_score)

# Confusion Matrix
cm = confusion_matrix(y_test, log_reg_model.predict(X_test))
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(win.target_names))
plt.xticks(tick_marks, win.target_names, rotation=45)
plt.yticks(tick_marks, win.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Selecting two features for visualization
X = win.data[:, :2]
Y = win.target
log_reg_model.fit(X, Y)

# Create a mesh to plot decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = log_reg_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(1, figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel(win.feature_names[0])
plt.ylabel(win.feature_names[1])
plt.title('Logistic Regression Decision Boundary')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()
------------------------------------------------------------------------
#Sigmoid model of Logistic regression model
------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Sample liver dataset
# Assuming the dataset is loaded into a DataFrame
data = pd.read_csv(r"Copy your dataset path example downloads\Liver.csv")

#change accord to your Dataset Column name
# Encode 'Gender' column
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# Features and target
X = data.drop('Dataset', axis=1)
y = data['Dataset']

# Ensure all data is numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Predictions
y_pred = logreg.predict(X_test_scaled)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Sigmoid Curve Visualization using predicted probabilities
y_pred_proba = logreg.predict_proba(X_test_scaled)[:, 1]

plt.plot(np.sort(y_pred_proba), np.linspace(0, 1, len(y_pred_proba)), color='blue')
plt.xlabel('Predicted Probability (Sigmoid Output)')
plt.ylabel('Cumulative Probability')
plt.title('Sigmoid Output of Logistic Regression Model')
plt.grid(True)
plt.show()
------------------------------------------------------------
#Logistic Regression Decision Boundary 
-----------------------------------------------------------

from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Load wine dataset
win = datasets.load_wine()

# Data and target values
X = win.data
y = win.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

# Train logistic regression model
log_reg_model = linear_model.LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
log_reg_base_score = log_reg_model.score(X_test, y_test)
print("The score for the Logistic Regression Model is:", log_reg_base_score)

# Confusion Matrix
cm = confusion_matrix(y_test, log_reg_model.predict(X_test))
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(win.target_names))
plt.xticks(tick_marks, win.target_names, rotation=45)
plt.yticks(tick_marks, win.target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Selecting two features for visualization
X = win.data[:, :2]
Y = win.target
log_reg_model.fit(X, Y)

# Create a mesh to plot decision boundaries
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = log_reg_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(1, figsize=(8, 6))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel(win.feature_names[0])
plt.ylabel(win.feature_names[1])
plt.title('Logistic Regression Decision Boundary')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.show()

#END`, cellType: 'markdown' },
    { id: 3, code: `<h2>EX-6 Naive Bayes</h2>`, cellType: 'markdown' },
    { id: 4, code: `import pandas as pd
df = pd.read_csv(r"C:\Users\Sastra\Downloads\weather_forecast.csv")  
df


import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score  
import numpy as np  

# Load the dataset  
df = pd.read_csv(r"C:\Users\Sastra\YourDataset\weather_forecast.csv")  

# Check unique values before replacement  
print("Unique values before replacement:")  
for column in ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play']:  
    print(f"{column}: {df[column].unique()}")  

# Replace categorical features with numerical equivalents  
df = df.replace({  
    'Outlook': {'Sunny': 0, 'Overcast': 1, 'Rain': 2},  # Updated to include 'Rain'  
    'Temperature': {'Hot': 0, 'Mild': 1, 'Cool': 2},  
    'Humidity': {'High': 0, 'Normal': 1},  
    'Windy': {'Weak': 0, 'Strong': 1},  # Added mapping for 'Weak' and 'Strong'  
    'Play': {'Yes': 1, 'No': 0}  
})  

# Check unique values after replacement  
print("\nUnique values after replacement:")  
for column in ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play']:  
    print(f"{column}: {df[column].unique()}")  

# Define features and target variable  
X = df.iloc[:, :-1]  # Features  
y = df.iloc[:, -1]   # Target variable  

# Ensuring all features are of correct dtype  
X = X.astype(float)  

# Split the data into training and testing sets  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.2, random_state=42  
)  

# Create and train a Gaussian Naive Bayes classifier  
model = GaussianNB()  
model.fit(X_train, y_train)  

# Make predictions on the test set and evaluate the model  
y_pred = model.predict(X_test)  
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2f}")  

# Function to predict based on user input  
def make_prediction(model, num_features):  
    while True:  
        try:  
            test_input = []  
            print("Enter the features for prediction (separated by space):")  
            user_input = input().split()  

            if len(user_input) != num_features:  
                print(f"Please enter {num_features} values.")  
                continue  

            # Ensure input values are valid floats, if there's a conversion issue, catch it.  
            test_input = [float(val) for val in user_input]  

            # Predicting the class  
            predicted_class = model.predict([test_input])  
            print(f"Predicted class: {predicted_class[0]}")  

            another_prediction = input("Do you want to make another prediction? (yes/no): ")  
            if another_prediction.lower() != "yes":  
                break  

        except ValueError:  
            print("Invalid input. Please enter numeric values.")  
        except Exception as e:  
            print(f"An error occurred: {e}")  

# Call the prediction function  
make_prediction(model, len(X.columns))  

#END`, cellType: 'markdown' },
    { id: 5, code: `<h2>SVM & RVM (7 & 8)</h2>`, cellType: 'markdown' },
    { id: 6, code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
accuracies = []

for kernel in kernels:
    model = svm.SVC(kernel=kernel, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.bar(kernels, accuracies, color='skyblue')
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Kernel Types for SVM Classifier on Breast Cancer Dataset')
plt.ylim(0, 1)  
plt.grid(axis='y')
plt.show()

best_kernel = kernels[np.argmax(accuracies)]
print(f"Best Kernel: {best_kernel}")

final_model = svm.SVC(kernel=best_kernel, probability=True)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

y_prob = final_model.predict_proba(X_test)[:, 1]  
fpr, tpr, _ = roc_curve(y_test, y_prob)  
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title("SVM Decision Boundary (using first two features)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(final_model, X_train, y_train)`, cellType: 'markdown' },
    { id: 7, code: `<h2>SVM & RVM (7 & 8)</h2>`, cellType: 'markdown' },
    { id: 8, code: `import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X = X[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
accuracies = []

for kernel in kernels:
    model = svm.SVC(kernel=kernel, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.bar(kernels, accuracies, color='skyblue')
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Kernel Types for SVM Classifier on Breast Cancer Dataset')
plt.ylim(0, 1)  
plt.grid(axis='y')
plt.show()

best_kernel = kernels[np.argmax(accuracies)]
print(f"Best Kernel: {best_kernel}")

final_model = svm.SVC(kernel=best_kernel, probability=True)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

y_prob = final_model.predict_proba(X_test)[:, 1]  
fpr, tpr, _ = roc_curve(y_test, y_prob)  
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

def plot_decision_boundary(model, X, y):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.title("SVM Decision Boundary (using first two features)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(final_model, X_train, y_train)`, cellType: 'markdown' },

{ id: 9, code: `<h2>Feed Forward network (9 & 10)</h2>`, cellType: 'markdown' },
{ id: 10, code: `pip install tensorflow


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Load the dataset
data = pd.read_csv('D:/YourDataSet/winequality-red.csv', delimiter=';')

# Step 2: Preprocess the data
X = data.drop('quality', axis=1)  # Features
y = data['quality']               # Target

# Encode the target variable (convert quality to integer labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert quality to integer labels
y_categorical = to_categorical(y_encoded)   # One-hot encode the labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Build the feed-forward neural network
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Second hidden layer
model.add(Dense(32, activation='relu'))

# Output layer (softmax for multi-class classification)
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Step 5: Evaluate the model
# Predictions
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)  # Get the predicted class
y_true = np.argmax(y_test, axis=1)       # Get the true class

# Evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_true, y_pred)
class_report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(y_categorical.shape[1])])

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
-----------------------------------------------------------------
#Training model without regularization...
-----------------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('D:/YourDataset/winequality-red.csv', delimiter=';')

# Step 2: Preprocess the data
X = data.drop('quality', axis=1)  # Features
y = data['quality']               # Target

# Encode the target variable (convert quality to integer labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert quality to integer labels
y_categorical = to_categorical(y_encoded)   # One-hot encode the labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to build and train a model
def build_and_train_model(regularization=False):
    model = Sequential()

    if regularization:
        # Input layer and first hidden layer with L2 regularization and dropout
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))  # Dropout rate of 30%

        # Second hidden layer with L2 regularization and dropout
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.3))  # Dropout rate of 30%
    else:
        # Input layer and first hidden layer without regularization
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

        # Second hidden layer without regularization
        model.add(Dense(32, activation='relu'))

    # Output layer (softmax for multi-class classification)
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50, batch_size=32,
        validation_split=0.2, verbose=0
    )

    return model, history

# Step 3: Train models with and without regularization
print("Training model without regularization...")
model_no_reg, history_no_reg = build_and_train_model(regularization=False)

print("Training model with regularization...")
model_with_reg, history_with_reg = build_and_train_model(regularization=True)

# Step 4: Evaluate the models
# Predictions for the model without regularization
y_pred_prob_no_reg = model_no_reg.predict(X_test_scaled)
y_pred_no_reg = np.argmax(y_pred_prob_no_reg, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics for the model without regularization
accuracy_no_reg = accuracy_score(y_true, y_pred_no_reg)
precision_no_reg = precision_score(y_true, y_pred_no_reg, average='weighted')
recall_no_reg = recall_score(y_true, y_pred_no_reg, average='weighted')
f1_no_reg = f1_score(y_true, y_pred_no_reg, average='weighted')

# Predictions for the model with regularization
y_pred_prob_with_reg = model_with_reg.predict(X_test_scaled)
y_pred_with_reg = np.argmax(y_pred_prob_with_reg, axis=1)

# Metrics for the model with regularization
accuracy_with_reg = accuracy_score(y_true, y_pred_with_reg)
precision_with_reg = precision_score(y_true, y_pred_with_reg, average='weighted')
recall_with_reg = recall_score(y_true, y_pred_with_reg, average='weighted')
f1_with_reg = f1_score(y_true, y_pred_with_reg, average='weighted')

# Step 5: Compare metrics
metrics = {
    "Without Regularization": [accuracy_no_reg, precision_no_reg, recall_no_reg, f1_no_reg],
    "With Regularization": [accuracy_with_reg, precision_with_reg, recall_with_reg, f1_with_reg]
}
metrics_df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1 Score"])
print("\nComparison of Metrics:")
print(metrics_df)

# Step 6: Plot the comparison
x = np.arange(len(metrics_df.index))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, metrics_df["Without Regularization"], width, label='Without Regularization')
bars2 = ax.bar(x + width/2, metrics_df["With Regularization"], width, label='With Regularization')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Metrics: With vs Without Regularization')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df.index)
ax.legend()

# Add value labels on top of bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.show()
----------------------------------------------------------
#Training model without L2 regularization...
----------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv('D:/YourDataset/winequality-red.csv', delimiter=';')

# Step 2: Preprocess the data
X = data.drop('quality', axis=1)  # Features
y = data['quality']               # Target

# Encode the target variable (convert quality to integer labels)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert quality to integer labels
y_categorical = to_categorical(y_encoded)   # One-hot encode the labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to build and train a model
def build_and_train_model(l2_reg=False):
    model = Sequential()

    if l2_reg:
        # Input layer and first hidden layer with L2 regularization
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)))

        # Second hidden layer with L2 regularization
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    else:
        # Input layer and first hidden layer without regularization
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

        # Second hidden layer without regularization
        model.add(Dense(32, activation='relu'))

    # Output layer (softmax for multi-class classification)
    model.add(Dense(y_categorical.shape[1], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50, batch_size=32,
        validation_split=0.2, verbose=0
    )

    return model, history

# Step 3: Train models with and without L2 regularization
print("Training model without L2 regularization...")
model_no_reg, history_no_reg = build_and_train_model(l2_reg=False)

print("Training model with L2 regularization...")
model_with_l2, history_with_l2 = build_and_train_model(l2_reg=True)

# Step 4: Evaluate the models
# Predictions for the model without L2 regularization
y_pred_prob_no_reg = model_no_reg.predict(X_test_scaled)
y_pred_no_reg = np.argmax(y_pred_prob_no_reg, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics for the model without L2 regularization
accuracy_no_reg = accuracy_score(y_true, y_pred_no_reg)
precision_no_reg = precision_score(y_true, y_pred_no_reg, average='weighted')
recall_no_reg = recall_score(y_true, y_pred_no_reg, average='weighted')
f1_no_reg = f1_score(y_true, y_pred_no_reg, average='weighted')

# Predictions for the model with L2 regularization
y_pred_prob_with_l2 = model_with_l2.predict(X_test_scaled)
y_pred_with_l2 = np.argmax(y_pred_prob_with_l2, axis=1)

# Metrics for the model with L2 regularization
accuracy_with_l2 = accuracy_score(y_true, y_pred_with_l2)
precision_with_l2 = precision_score(y_true, y_pred_with_l2, average='weighted')
recall_with_l2 = recall_score(y_true, y_pred_with_l2, average='weighted')
f1_with_l2 = f1_score(y_true, y_pred_with_l2, average='weighted')

# Step 5: Compare metrics
metrics = {
    "Without L2 Regularization": [accuracy_no_reg, precision_no_reg, recall_no_reg, f1_no_reg],
    "With L2 Regularization": [accuracy_with_l2, precision_with_l2, recall_with_l2, f1_with_l2]
}
metrics_df = pd.DataFrame(metrics, index=["Accuracy", "Precision", "Recall", "F1 Score"])
print("\nComparison of Metrics:")
print(metrics_df)

# Step 6: Plot the comparison
x = np.arange(len(metrics_df.index))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, metrics_df["Without L2 Regularization"], width, label='Without L2 Regularization')
bars2 = ax.bar(x + width/2, metrics_df["With L2 Regularization"], width, label='With L2 Regularization')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Metrics: With vs Without L2 Regularization')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df.index)
ax.legend()

# Add value labels on top of bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.show()

#END`, cellType: 'markdown' },


{ id: 11, code: `<h2>EDA</h2>`, cellType: 'markdown' },
{ id: 12, code: `-------------------------------
1)Initial Data Overview
--------------------------------
# Load dataset
import pandas as pd
df = pd.read_csv('your_dataset.csv')

# Basic information
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nDataset Info:")
print(df.info())

--------------------------
2) Missing Values Analysis 
----------------------------
# Missing values summary
print("Missing Values Summary:")
print(df.isnull().sum())

# Visualizing missing values
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

-------------------------
3) Statistical Summary
-------------------------
# Descriptive statistics
print("Descriptive Statistics:")
print(df.describe(include='all'))

# For numerical columns
if len(df.select_dtypes(include=['int64','float64']).columns) > 0:
    print("\nNumerical Columns Statistics:")
    print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

-------------------------
4)  Univariate Analysis
-------------------------

# Numerical columns distribution
num_cols = df.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Categorical columns distribution
cat_cols = df.select_dtypes(include=['object','category','bool']).columns
for col in cat_cols:
    plt.figure(figsize=(8,4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

-------------------------------
5) Bivariate/Multivariate Analysis
---------------------------------

# Correlation matrix for numerical variables
if len(num_cols) > 1:
    plt.figure(figsize=(10,8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

# Pairplot for numerical variables (sample if dataset is large)
if len(num_cols) > 1:
    sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
    sns.pairplot(sample_df[num_cols])
    plt.show()

# Categorical vs Numerical analysis
if len(num_cols) > 0 and len(cat_cols) > 0:
    for num_col in num_cols:
        for cat_col in cat_cols:
            if len(df[cat_col].unique()) < 10:  # Avoid columns with too many categories
                plt.figure(figsize=(10,6))
                sns.boxplot(x=cat_col, y=num_col, data=df)
                plt.title(f'{num_col} by {cat_col}')
                plt.xticks(rotation=45)
                plt.show()

---------------------
6) Outlier Detection
---------------------

# Z-score method for numerical columns
from scipy import stats
import numpy as np

for col in num_cols:
    z = np.abs(stats.zscore(df[col]))
    print(f"Outliers in {col}: {len(np.where(z > 3)[0])}")
    
    # IQR method
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5*IQR)) | (df[col] > (Q3 + 1.5*IQR))]
    print(f"IQR outliers in {col}: {len(outliers)}")

--------------------------------------
7) Time Series Analysis (if applicable)
-------------------------------------
# Check for datetime columns
date_cols = df.select_dtypes(include=['datetime64']).columns
for col in date_cols:
    df[col] = pd.to_datetime(df[col])
    plt.figure(figsize=(12,6))
    df.set_index(col).resample('D').count().plot()  # Change resample frequency as needed
    plt.title(f'Trend over time ({col})')
    plt.show()

------------------------------------
8. Unique Values and Cardinality
----------------------------------
# Check unique values in each column
print("Unique Values Count:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")
    
# High cardinality columns
high_cardinality = [col for col in df.columns if df[col].nunique() > 50]
print("\nHigh Cardinality Columns (>50 unique values):", high_cardinality)

--------------------
9. Save EDA Report
--------------------

# Save basic statistics to CSV
df.describe(include='all').to_csv('eda_summary_statistics.csv')

# Save missing values info
df.isnull().sum().to_frame('missing_values').to_csv('missing_values_summary.csv')



`, cellType: 'markdown' }


  ];

  // State for cells and active cell
  const [cells, setCells] = useState<CellData[]>(initialCells);
  const [activeCellIndex, setActiveCellIndex] = useState<number | null>(null);

  // Handle cell focus
  const handleCellFocus = (index: number) => {
    setActiveCellIndex(index);
  };

  // Handle Alt+Enter to create a new cell
  const handleAltEnter = (index: number) => {
    const newCellId = Math.max(...cells.map(cell => cell.id)) + 1;
    const newCell: CellData = {
      id: newCellId,
      code: '',
      cellType: 'code'
    };
    
    // Insert the new cell after the current one
    const updatedCells = [
      ...cells.slice(0, index + 1),
      newCell,
      ...cells.slice(index + 1)
    ];
    
    setCells(updatedCells);
    // Set focus to the new cell
    setActiveCellIndex(index + 1);
  };

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <JupyterHeader />
      <JupyterMenu />
      <JupyterToolbar />
      
      <div className="flex-1 px-4 py-2 overflow-y-auto">
        {cells.map((cell, index) => (
          <CodeCell 
            key={cell.id}
            cellNumber={cell.cellType === 'code' ? cell.id : 0}
            code={cell.code} 
            cellType={cell.cellType} 
            isActive={activeCellIndex === index}
            onFocus={() => handleCellFocus(index)}
            onAltEnter={() => handleAltEnter(index)}
          />
        ))}
      </div>
    </div>
  );
};

export default Index;
