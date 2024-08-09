import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)

# 예측 기간
PREDICTION_PERIOD = 7
# 예측 성장률
EXPECTED_GROWTH_RATE = 5
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 365

today = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')

# S&P 500 종목 리스트 가져오기
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_df = pd.read_html(url)[0]
tickers = sp500_df['Symbol'].tolist()

output_dir = 'D:\\sp500'
model_dir = 'sp_models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 주식 데이터를 가져오는 함수
def fetch_stock_data(ticker, fromdate, todate):
    stock_data = yf.download(ticker, start=fromdate, end=todate)
    if stock_data.empty:
        return pd.DataFrame()
    # 선택적인 컬럼만 추출하고 NaN 값을 0으로 채움
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
    stock_data['PER'] = 0  # FinanceDataReader의 PER 필드 대체
    return stock_data


def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

# LSTM 모델 학습 및 예측 함수 정의
def create_model(input_shape):
    model = tf.keras.Sequential([
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

count = 20

@tf.function(reduce_retracing=True)
def predict_model(model, data):
    return model(data)

# for ticker in tickers:
for ticker in tickers[count-1:]:
    count += 1
    print(f"Processing {count}/{len(tickers)} : {ticker}")
    data = fetch_stock_data(ticker, start_date, today)
    if data.empty or len(data) < 60:  # 데이터가 충분하지 않으면 건너뜀
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    X, Y = create_dataset(tf.convert_to_tensor(scaled_data), 60)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    model_file_path = os.path.join(model_dir, f'{ticker}_model_v1.Keras')
    if os.path.exists(model_file_path):
        model = tf.keras.models.load_model(model_file_path)
    else:
        model = create_model((X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,  # 10 에포크 동안 개선 없으면 종료
        verbose=1,
        mode='min',
        restore_best_weights=True  # 최적의 가중치를 복원
    )

    model.fit(X_train, Y_train, epochs=30, batch_size=32, verbose=0,
              validation_data=(X_val, Y_val), callbacks=[early_stopping])

    close_scaler = MinMaxScaler()
    close_prices_scaled = close_scaler.fit_transform(data[['Close']].values)

    predictions = predict_model(model, X[-PREDICTION_PERIOD:])
    predicted_prices = close_scaler.inverse_transform(predictions.numpy()).flatten()
    model.save(model_file_path)

    last_close = data['Close'].iloc[-1]
    future_return = (predicted_prices[-1] / last_close - 1) * 100

    if future_return < EXPECTED_GROWTH_RATE:
        continue

    extended_prices = np.concatenate((data['Close'].values, predicted_prices))
    extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
    last_price = data['Close'].iloc[-1]

    plt.figure(figsize=(26, 10))
    plt.plot(extended_dates[:len(data['Close'].values)], data['Close'].values, label='Actual Prices', color='blue')
    plt.plot(extended_dates[len(data['Close'].values)-1:], np.concatenate(([data['Close'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'{ticker} - Actual vs Predicted Prices {today} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {ticker} [ {last_price} ] {timestamp}.png')
    plt.savefig(file_path)
    plt.close()