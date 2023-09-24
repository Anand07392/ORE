from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import pandas as pd
data = pd.read_csv(r'C:\Users\user\Documents\ORE Project\framinghams.csv')
X = data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
y = data['TenYearCHD']
X = pd.DataFrame(X)
imputer = SimpleImputer(strategy='mean')
imputer.fit(X)
X = imputer.transform(X)
model = LogisticRegression()
model.fit(X, y)
new_data = pd.DataFrame([[1, 50, 2, 1, 10, 0, 0, 1, 0, 200, 120, 80, 25, 70, 100]])
predicted_chances = model.predict_proba(new_data)[:, 1]
print('Predicted chances of heart disease:', predicted_chances)
