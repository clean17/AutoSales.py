import os
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)

PREDICTION_PERIOD = 30
DATA_COLLECTION_PERIOD = 365  # 6 months

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

tickers = stock.get_market_ticker_list(market="KOSPI")

output_dir = 'D:\\stocks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def fetch_stock_data(ticker, fromdate, todate):
    ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
    fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker)
    fundamental['PER'] = fundamental['PER'].fillna(0)
    data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)
    return data

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

def create_model(input_shape):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(128),
        Dense(64),
        Dense(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

for ticker in tickers[:10]:  # Limit to 10 for demo purposes
    print(f"Processing {ticker}")
    data = fetch_stock_data(ticker, start_date, today)
    if data.empty or len(data) < 60:
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    X, Y = create_dataset(scaled_data, 60)

    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

    close_scaler = MinMaxScaler()
    close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

    predictions = model.predict(X[-PREDICTION_PERIOD:])
    predicted_prices = close_scaler.inverse_transform(predictions).flatten()

    extended_prices = np.concatenate((data['종가'].values, predicted_prices))
    extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))

    plt.figure(figsize=(26, 10))
    plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices')
    plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', linestyle='--')
    plt.title(f"{ticker} - Actual vs Predicted Prices")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    file_path = os.path.join(output_dir, f'{today}_{ticker}.png')
    plt.savefig(file_path)
    plt.close()
