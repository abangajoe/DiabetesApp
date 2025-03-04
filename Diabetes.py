import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score,precision_score,confusion_matrix,recall_score,mean_squared_error
import pickle

# Load the dataset
diab_data = pd.read_csv("diabetes.csv")

# Check the dataset shape and summary statistics
print(diab_data.shape)
print(diab_data.describe())

# Prepare the feature matrix and target vector
x = diab_data.drop(columns='Outcome', axis=1)
y = diab_data['Outcome']

# Standardize the feature matrix
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Evaluate the model on the test set
x_test_pred = classifier.predict(x_test)
x_test_acc = accuracy_score(x_test_pred, y_test)

f1_score = f1_score(x_test_pred, y_test)
precision_score = precision_score(x_test_pred, y_test)
confusion_matrix = confusion_matrix(x_test_pred, y_test)
mean_absolute_error = mean_squared_error(x_test_pred, y_test)
recall_score = recall_score(x_test_pred, y_test)

print(f"The Accuracy of the model is: {x_test_acc * 100:.2f}%")
print(f"f1_score: {f1_score * 100:.2f}%")
print(f"precision_score: {precision_score * 100:.2f}%")
print(f"confusion_matrix: {confusion_matrix * 100}")
print(f"mean_absolute_error: {mean_absolute_error * 100:.2f}%")
print(f"recall_score: {recall_score * 100:.2f}%")


# Save the model using pickle
with open('diabetes.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Load the model to ensure it's saved correctly
with open('diabetes.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the scaler to ensure it's saved correctly
with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Define a function to make predictions
def predict_diabetes(input_data):
    input_data_to_array = np.asarray(input_data)
    input_data_reshaped = input_data_to_array.reshape(1, -1)
    standardized_data = scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(standardized_data)
    result = 'The person is non-diabetic' if prediction[0] == 0 else 'The person is diabetic'
    print("Predicted class for the given input data:", prediction[0])
    return result

# Test the prediction function with sample input data
input_data = (4,110,92,0,0,37.6,0.191,30)
result = predict_diabetes(input_data)
print(result)
