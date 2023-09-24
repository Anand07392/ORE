import pandas as pd
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r'C:\Users\user\Documents\ORE Project\Prices.csv')
X = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']]
y = data['price']
model = LinearRegression()
model.fit(X, y)
new_data = pd.DataFrame([[3, 2, 1500, 5000, 1, 0, 1, 3, 1200, 300, 1990, 2005]], columns=X.columns)
predicted_prices = model.predict(new_data)
print('Predicted prices:', predicted_prices)
