# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer         # For imputing missing values
from sklearn.preprocessing import StandardScaler     # For scaling features
from sklearn.model_selection import train_test_split, GridSearchCV  # For data splitting and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Dataset
# It is assumed that the "heart_disease.csv" file is in the same directory as this script.
data = pd.read_csv("heart_disease.csv")
print("First 5 rows of the dataset:")
print(data.head())

# 2. Data Preprocessing

## 2.1 Check for Missing Values
print("\nMissing values in each column:")
print(data.isnull().sum())

## 2.2 Impute Missing Values
# We use the median imputation strategy to reduce the effect of outliers.
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Verify that there are no missing values after imputation
print("\nMissing values after imputation:")
print(data_imputed.isnull().sum())

## 2.3 Feature Scaling (Normalization)
# Features such as blood pressure and cholesterol have different scales.
# We exclude the 'target' column from scaling.
features = data_imputed.drop("target", axis=1)
target = data_imputed["target"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Convert the scaled features back into a DataFrame
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# 3. Model Selection and Training

## 3.1 Split the Data into Training and Testing Sets
# We use an 80-20 split. The stratify parameter ensures that the target distribution is preserved.
X_train, X_test, y_train, y_test = train_test_split(features_scaled_df, target, test_size=0.2, random_state=42, stratify=target)

## 3.2 Initialize the Random Forest Classifier
# Random Forest is chosen due to its ability to handle non-linear relationships and provide insights on feature importance.
rf = RandomForestClassifier(random_state=42)

## 3.3 Hyperparameter Tuning using GridSearchCV
# We define a grid of hyperparameters to search for the optimal combination.
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(grid_search.best_params_)

# Retrieve the best model from the grid search
best_rf = grid_search.best_estimator_

## 3.4 Train the Best Model on the Training Data
best_rf.fit(X_train, y_train)

# 4. Model Evaluation

## 4.1 Make Predictions on the Test Set
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

## 4.2 Print the Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

## 4.3 Calculate and Display the ROC AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score:", roc_auc)

## 4.4 Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix.png")  # Save the confusion matrix as an image
plt.show()

## 4.5 Plot the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line represents a random classifier
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")  # Save the ROC curve as an image
plt.show()

# 5. Reproducibility and Documentation
# This code is thoroughly commented to ensure each step is clear and can be replicated.
# All instructions for running the code, including library installations, are provided.
