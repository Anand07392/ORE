import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv(r'C:\Users\user\Documents\ORE Project\Salarys.csv')

# Split the dataset into independent (X) and dependent (y) variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Train the model on the training set
regressor = LinearRegression()
regressor.fit(X, y)

# Predict the salary for a given number of years of experience
years_of_experience = 5.5
salary = regressor.predict([[years_of_experience]])
print(f"A person with {years_of_experience} years of experience is predicted to earn a salary of ${salary[0]:.2f}.")