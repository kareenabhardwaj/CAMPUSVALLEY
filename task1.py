# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
file_path = r"D:\internship\tested.csv"  # Use raw string literal (r) to avoid issues with backslashes
titanic_df = pd.read_csv(file_path)



# Step 3: Explore the Data
print(titanic_df.head())  # Display first few rows of the dataset
print(titanic_df.info())  # Display information about the dataset
print(titanic_df.describe())  # Display summary statistics of numerical columns

# Step 4: Data Preprocessing
titanic_df.dropna(inplace=True)  # Handle missing values
titanic_df = pd.get_dummies(titanic_df, columns=['Sex', 'Embarked'])  # Convert categorical variables into numerical values

# Step 5: Select Features and Target Variable
X = titanic_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
y = titanic_df['Survived']

# Step 6: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)  # Calculate confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")  # Display classification report
print(classification_report(y_test, y_pred))

# Step 9: Visualize Results (optional)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.yticks([0, 1], ['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
