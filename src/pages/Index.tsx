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
---------------------------------------------------
==FEEDFORWARD WITH REGULARISATION CLASSIFICATION==
----------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the Titanic dataset from seaborn
data = sns.load_dataset("titanic")

# Drop irrelevant columns
data = data.drop(columns=["who", "deck", "embark_town", "alive", "adult_male", "class"], errors="ignore")

# Handle missing values
data["age"] = data["age"].fillna(data["age"].median())
data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
data["fare"] = data["fare"].fillna(data["fare"].median())

# Convert categorical variables
data = pd.get_dummies(data, columns=["sex", "embarked"], drop_first=True)

# Drop rows with missing target values
data = data.dropna(subset=["survived"])

# Define features and target
X = data.drop(columns=["survived"])
y = data["survived"]

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = ["age", "fare"]
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model functions
def build_model(l2_lambda=0.0, dropout_rate=0.0):
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda)),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate function
def train_and_evaluate(model_name, model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0
    )
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = (model.predict(X_test).flatten() >= 0.5).astype(int)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    return test_accuracy

# Train models and compare
models = {
    "Base Model": build_model(),
    "L2 Regularization": build_model(l2_lambda=0.01),
    "Dropout": build_model(dropout_rate=0.3),
    "Combined Regularization": build_model(l2_lambda=0.01, dropout_rate=0.3)
}

results = {}
for model_name, model in models.items():
    results[model_name] = train_and_evaluate(model_name, model)

# Compare results
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Model Type")
plt.ylabel("Test Accuracy")
plt.title("Model Comparison on Titanic Dataset")
plt.show()

---------------------------------------------------
== FEEDFORWARD WITH REGULARISATION REGRESSION==
----------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import EarlyStopping

# Load the inbuilt Wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = X["alcohol"]  # Predict alcohol content
X = X.drop(columns=["alcohol"])  # Remove alcohol from features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

# Baseline model
baseline_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])
baseline_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
baseline_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
baseline_metrics = evaluate_model(baseline_model, X_test, y_test)

# L1 Regularization model
l1_model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l1(0.01), input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu', kernel_regularizer=l1(0.01)),
    Dense(1)
])
l1_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
l1_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
l1_metrics = evaluate_model(l1_model, X_test, y_test)

# L2 Regularization model
l2_model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1)
])
l2_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
l2_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
l2_metrics = evaluate_model(l2_model, X_test, y_test)

# Dropout Regularization model
dropout_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(1)
])
dropout_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
dropout_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)
dropout_metrics = evaluate_model(dropout_model, X_test, y_test)

# Early Stopping model
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
early_stopping_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
early_stopping_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stopping_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
early_stopping_metrics = evaluate_model(early_stopping_model, X_test, y_test)

# Combine results
metrics_comparison = pd.DataFrame({
    "Baseline": baseline_metrics,
    "L1 Regularization": l1_metrics,
    "L2 Regularization": l2_metrics,
    "Dropout Regularization": dropout_metrics,
    "Early Stopping": early_stopping_metrics
})

print(metrics_comparison)

# Plot results
metrics_comparison.T.plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of Evaluation Metrics for Different Regularization Methods")
plt.ylabel("Metric Value")
plt.xlabel("Regularization Technique")
plt.xticks(rotation=45)
plt.legend(loc="upper right")
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

#Read Different File Formats and add Header
import pandas as pd

# Define column names
headers = ["Column1", "Column2", "Column3"]

# Read file and apply headers
df = pd.read_csv("file.txt", delimiter="\t", names=headers)

# Save to CSV with headers
df.to_csv("output.csv", index=False)



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
plt.show(

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
