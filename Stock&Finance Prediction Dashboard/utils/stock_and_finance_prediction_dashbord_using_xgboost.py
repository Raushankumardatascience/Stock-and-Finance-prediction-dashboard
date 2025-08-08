import pandas as pd
import numpy as np
import warnings
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings('ignore')

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# Keep only the 'Close' column
df = data[['Close']].copy()

# Feature Engineering
df["Close_t-1"] = df["Close"].shift(1)
df["MA_5"] = df["Close"].rolling(5).mean()
df["STD_5"] = df["Close"].rolling(5).std()
df["DayOfWeek"] = df.index.dayofweek
df["target"] = df["Close"].shift(-1)

# Drop missing values
df.dropna(inplace=True)

# Features and target
X = df[['Close_t-1', 'MA_5', 'STD_5', 'DayOfWeek']]
X.columns = [str(col).strip() for col in X.columns]  # 🧹 Clean column names
y = df['target']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
xgb = XGBRegressor(random_state=42)

# Define hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}

# Grid search
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("Tuned model performance:")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# Save model
joblib.dump(best_model, 'best_model.joblib')
print("✅ Model saved as best_model.joblib")
