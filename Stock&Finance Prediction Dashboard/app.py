from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from keras.models import load_model
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# ========== Helper Functions ==========

def download_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data[['Close']]

def predict_arima(data, forecast_days):
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast.tolist()

def predict_lstm(data, forecast_days):
    model = load_model("models/lstm_model.h5")
    scaler = joblib.load("models/lstm_scaler.pkl")

    scaled_data = scaler.transform(np.array(data).reshape(-1, 1))
    input_seq = scaled_data[-60:].reshape(1, 60, 1)

    predictions = []
    for _ in range(forecast_days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten().tolist()

def create_features(df):
    df = df.copy()
    df["Close_t-1"] = df["Close"].shift(1)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["STD_5"] = df["Close"].rolling(5).std()
    df["DayOfWeek"] = df.index.dayofweek
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()
    return df

def predict_random_forest(data, forecast_days):
    model = joblib.load("models/RandomForestRegressor_model.pkl")
    df = create_features(data)
    last_row = df.iloc[-1:]

    predictions = []
    for _ in range(forecast_days):
        X = last_row[['Close_t-1', 'MA_5', 'STD_5', 'DayOfWeek']]
        pred = model.predict(X)[0]
        predictions.append(pred)

        # Create new row for next prediction
        new_row = {
            "Close": pred,
            "Close_t-1": last_row["Close"].values[0],
            "MA_5": last_row["MA_5"].values[0],
            "STD_5": last_row["STD_5"].values[0],
            "DayOfWeek": (last_row["DayOfWeek"].values[0] + 1) % 7,
        }
        last_row = pd.DataFrame([new_row])

    return predictions

def predict_xgboost(data, forecast_days):
    model = joblib.load("models/best_model.joblib")
    df = create_features(data)
    df.columns = [str(col).strip() if isinstance(col, str) else str(col[0]) for col in df.columns]  # 🧹 Ensure no MultiIndex or extra spaces

    last_row = df.iloc[-1:]

    feature_columns = ['Close_t-1', 'MA_5', 'STD_5', 'DayOfWeek']
    last_row = last_row[feature_columns] 
    last_row.columns = feature_columns 
    predictions = []
    for _ in range(forecast_days):
        pred = model.predict(last_row)[0]
        predictions.append(pred)

        # Prepare next row
        new_row = {
            "Close": pred,
            "Close_t-1": last_row["Close_t-1"].values[0],
            "MA_5": last_row["MA_5"].values[0],
            "STD_5": last_row["STD_5"].values[0],
            "DayOfWeek": (last_row["DayOfWeek"].values[0] + 1) % 7,
        }
        last_row = pd.DataFrame([new_row])[feature_columns]  
        last_row.columns = feature_columns  

    return predictions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form['ticker']
        model_type = request.form['model']
        forecast_days = int(request.form.get('forecast_days', 7))

        data = download_data(ticker)
        close_data = data['Close']
        data.index = pd.to_datetime(data.index)

        if model_type == 'arima':
            forecast = predict_arima(close_data, forecast_days)
        elif model_type == 'lstm':
            forecast = predict_lstm(close_data, forecast_days)
        elif model_type == 'best':
            forecast = predict_xgboost(data, forecast_days)
        elif model_type == 'rf':
            forecast = predict_random_forest(data, forecast_days)
        else:
            return render_template('result.html', ticker=ticker, forecast=None, model_type="Error", error="Invalid model selected.")

        # Convert forecast to a list of floats
        forecast = [float(price) for price in forecast]

        return render_template('result.html', ticker=ticker, forecast=forecast, model_type=model_type.upper(), error=None)

    except Exception as e:
        return render_template('result.html', ticker=None, forecast=None, model_type="Error", error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
