import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
# Import the Tkinter GUI code
from animalgui import *

file_path = 'animal.csv'
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:")
print(missing_values)

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=['Symptom 1', 'Symptom 2', 'Symptom 3'])
print("Encoded Dataframe:")
print(df_encoded.head())

# Drop original categorical columns
df = df.drop(['Symptom 1', 'Symptom 2', 'Symptom 3'], axis=1)

# Exploratory Data Analysis (EDA) Plots
print("Plots for Exploratory Data Analysis:")
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.histplot(df['Age'], kde=True)
plt.title('Distribution of Age')

plt.subplot(2, 2, 2)
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')

plt.subplot(2, 2, 3)
sns.histplot(df['Temperature'], kde=True)
plt.title('Distribution of Temperature')

plt.subplot(2, 2, 4)
sns.boxplot(x=df['Temperature'])
plt.title('Boxplot of Temperature')

plt.tight_layout()
plt.show()

# Disease Distribution Plot
print("Distribution of Disease:")
sns.countplot(x='Disease', data=df)
plt.title('Distribution of Disease')
plt.show()

# Age Distribution by Disease Plot
print("Age Distribution by Disease:")
sns.boxplot(x='Disease', y='Age', data=df)
plt.title('Age Distribution by Disease')
plt.show()

# Temperature Distribution by Disease Plot
print("Temperature Distribution by Disease:")
sns.boxplot(x='Disease', y='Temperature', data=df)
plt.title('Temperature Distribution by Disease')
plt.show()

# Animal Distribution by Disease Plot
print("Animal Distribution by Disease:")
sns.boxplot(x='Disease', y='Animal', data=df)
plt.title('Animal Distribution by Disease')
plt.show()

# Correlation Matrix Plot (excluding non-numeric columns)
print("Correlation Matrix:")
numeric_columns = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Feature Engineering and Scaling
print("Performing Feature Engineering and Scaling:")
df = pd.get_dummies(df, columns=['Animal'], drop_first=True)
df['age_temperature_interaction'] = df['Age'] * df['Temperature']
df['age_bin'] = pd.cut(df['Age'], bins=[0, 5, 10, 15, 20], labels=['0-5', '6-10', '11-15', '16-20'])

scaler = MinMaxScaler()
df[['Age', 'Temperature']] = scaler.fit_transform(df[['Age', 'Temperature']])

# Encoding 'age_bin' using Label Encoding
le = LabelEncoder()
df['age_bin'] = le.fit_transform(df['age_bin'])

# Split the dataset
print("Splitting the dataset into training and testing sets")
x = df.drop('Disease', axis=1)
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#decision trees
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# Predictions
y_pred_tree = tree_model.predict(X_test)

# Evaluate the model
print("Decision Tree Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))

# Model Training using the best model from GridSearchCV
print("Training the best RandomForestClassifier model")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation
print("Evaluating the best model on the test set")
y_pred_best = model.predict(X_test)

# Display Accuracy, Classification Report, and Confusion Matrix
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print("Classification Report:\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

# Display Precision, Recall, and F1-score
print("Precision, Recall, and F1-score:")
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_best, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

# Hyperparameter Tuning
print("Performing GridSearchCV for hyperparameter tuning")
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters:", best_params)

# Feature Importance Plot
print("Plotting the top 10 important features")
feature_importance = pd.Series(best_model.feature_importances_, index=X_train.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()
