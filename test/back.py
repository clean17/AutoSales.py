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

# 예측 기간
PREDICTION_PERIOD = 7
# 예측 성장률
EXPECTED_GROWTH_RATE = 1
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 480

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


tickers = stock.get_market_ticker_list(market="KOSPI")
# 종목 코드와 이름 딕셔너리 생성
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

output_dir = 'D:\\stocks'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 주식 데이터와 기본적인 재무 데이터를 가져온다
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

# LSTM 모델 학습 및 예측 함수 정의
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


def train_and_save_model(data, look_back=60, model_path='my_model_v1.Keras'):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, Y = create_dataset(scaled_data, look_back)
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, Y, batch_size=32, epochs=3, verbose=1, validation_split=0.1)
    model.save(model_path)
    return scaler

@tf.function
def predict(model, data):
    return model(data)

# 모델 불러오기 및 예측
def predict_stock_price_with_saved_model(data, scaler, model_path='my_model_v1.Keras', look_back=1, days_to_predict=PREDICTION_PERIOD):
    scaled_data = scaler.transform(data)

    def create_dataset(dataset, look_back=1):
        X = []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), :]
            X.append(a)
        return np.array(X)

    X = create_dataset(scaled_data, look_back)
    model = load_model(model_path)

    X_tensor = tf.convert_to_tensor(X[-days_to_predict:], dtype=tf.float32)

    predictions = predict(model, X_tensor)
    predictions = predictions.numpy()
    predictions = scaler.inverse_transform(
        np.concatenate((predictions, np.zeros((days_to_predict, data.shape[1] - 1))), axis=1)
    )[:, 0]

    # predictions = model.predict(X[-days_to_predict:])
    # predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((days_to_predict, data.shape[1] - 1))), axis=1))[:, 0]

    return predictions


predicted_stocks = {}

for ticker in tickers[:3]:
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
    print(f"Processing : {stock_name} {ticker}")
    data = fetch_stock_data(ticker, start_date, today)
    if data.empty or len(data) < 60: # 데이터가 충분하지 않으면 건너뜀
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    X, Y = create_dataset(scaled_data, 60)

    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, Y, epochs=3, batch_size=32, verbose=1, validation_split=0.1)

    close_scaler = MinMaxScaler()
    close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

    predictions = model.predict(X[-PREDICTION_PERIOD:])
    predicted_prices = close_scaler.inverse_transform(predictions).flatten()

    last_close = data['종가'].iloc[-1]
    future_return = (predicted_prices[-1] / last_close - 1) * 100

    # 성장률 이상만
    if future_return < EXPECTED_GROWTH_RATE:
        continue
        # predicted_stocks[ticker] = (future_return, data['종가'], predicted_prices, stock.get_market_ticker_name(ticker))

    extended_prices = np.concatenate((data['종가'].values, predicted_prices))
    extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))

    stock_name = stock.get_market_ticker_name(ticker)
    today = datetime.today().strftime('%Y%m%d')
    last_price = data['종가'].iloc[-1]

    plt.figure(figsize=(26, 10))
    plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices ({stock_name} {ticker})', color='blue')
    plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices ({stock_name} {ticker})',  color='red', linestyle='--')
    plt.title(f'{ticker} - Actual vs Predicted Prices {today} {stock_name} {ticker} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    # 레이블이 겹치지 않도록 하여 가독성 올린다
    plt.xticks(rotation=45)
    # plt.show()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ] {timestamp}.png')
    plt.savefig(file_path)
    plt.close()