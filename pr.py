# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample data (house prices)
data = {'size': [1000,1500,2000,2500], 
        'price': [300000,450000,600000,750000]}

df = pd.DataFrame(data)

X = df[['size']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Save model as pickle file
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved!")