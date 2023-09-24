from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the dataset
data = pd.read_csv(r'C:\Users\user\Documents\ORE Project\framinghams.csv')

# Split the data into features (X) and target variable (y)
X = data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
y = data['TenYearCHD']

# Convert the numpy array to a pandas DataFrame
X = pd.DataFrame(X)

# Create an imputer object
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to your data
imputer.fit(X)

# Transform the data by replacing NaN values
X = imputer.transform(X)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the data
model.fit(X, y)

# Predict the chances of heart disease for new data
new_data = pd.DataFrame([[1, 50, 2, 1, 10, 0, 0, 1, 0, 200, 120, 80, 25, 70, 100]])
predicted_chances = model.predict_proba(new_data)[:, 1]

print('Predicted chances of heart disease:', predicted_chances)
