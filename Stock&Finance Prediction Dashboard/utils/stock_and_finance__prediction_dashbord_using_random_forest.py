import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

!pip install yfinance

import yfinance as yf

Apple_data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

print(Apple_data.head())

Apple_data = Apple_data['Close']

df = Apple_data['AAPL']

Stock_Data = df.reset_index()

Stock_Data.head(1)

Stock_Data.info()

print(Stock_Data.size)
print(Stock_Data.shape)

Stock_Data['Date'] = pd.to_datetime(Stock_Data['Date'])

Stock_Data

Stock_Data.describe()

Stock_Data = Stock_Data.set_index("Date")

Stock_Data

df = Stock_Data.copy()

df

df["Close_t-1"] = df["AAPL"].shift(1)
df["MA_5"] = df["AAPL"].rolling(5).mean()
df["STD_5"] = df["AAPL"].rolling(5).std()
df["DayOfWeek"] = df.index.dayofweek
df["target"] = df["AAPL"].shift(-1)

df = df.dropna()

df

X = df[['Close_t-1', 'MA_5', 'STD_5', 'DayOfWeek']]
y = df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(mse)
print(r2)

!pip install joblib

import joblib

joblib.dump(model, 'RandomForestRegressor_model.pkl')

