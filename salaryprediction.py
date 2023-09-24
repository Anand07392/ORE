import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv(r'C:\Users\user\Documents\ORE Project\Salarys.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
regressor = LinearRegression()
regressor.fit(X, y)
years_of_experience = 5.5
salary = regressor.predict([[years_of_experience]])
print(f"A person with {years_of_experience} years of experience is predicted to earn a salary of ${salary[0]:.2f}.")
