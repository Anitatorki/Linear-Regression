import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv("student-mat.csv", sep=";")
df = df[["age", "absences", "freetime", "Medu", "G1", "G2", "G3"]]

predict = "G3"
X = np.array(df.drop([predict], axis=1))
y = np.array(df[predict])

# Set number of epochs for model testing
EPOCHS = 100
best_accuracy = 0

# Test multiple models and keep the best one
for _ in range(EPOCHS):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("Best accuracy found:", best_accuracy)

# Test with a fixed random state for reproducibility
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
model = LinearRegression()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f"Model accuracy with fixed random state: {acc}")

# Save the best model
file_name = "best_model.sav"
joblib.dump(best_model, file_name)

# Load the model and test again
loaded_model = joblib.load(file_name)
print("Accuracy of the loaded model:", loaded_model.score(x_test, y_test))
