import os
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import FinanceDataReader as fdr
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

# 미국 동부 시간대 설정
us_timezone = pytz.timezone('America/New_York')
now_us = datetime.now(us_timezone)
# 현재 시간 출력
today_us = now_us.strftime('%Y-%m-%d %H:%M:%S')
print("미국 동부 시간 기준 현재 시각:", today_us)
# 데이터 수집 시작일 계산
start_date_us = (now_us - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')
print("미국 동부 시간 기준 데이터 수집 시작일:", start_date_us)
today_us = datetime.today().strftime('%Y-%m-%d')

# Load the Wikipedia page
url = "https://en.wikipedia.org/wiki/Nasdaq-100"
# Read all tables from the webpage
tables = pd.read_html(url)

# Assuming the correct table index is found after manual inspection
nasdaq_100_table_index = 4  # You need to replace '4' with the correct index after confirming
nasdaq_100_df = tables[nasdaq_100_table_index]

# Print the first few rows to verify it is the correct table
print(nasdaq_100_df.head())

# If 'Symbol' or 'Ticker' is the correct column name
tickers = nasdaq_100_df['Ticker'].tolist()  # Replace 'Symbol' with the correct column name if different


output_dir = 'D:\\nasdaq'
model_dir = 'sp_models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 주식 데이터를 가져오는 함수
# def fetch_stock_data(ticker, fromdate, todate):
#     stock_data = yf.download(ticker, start=fromdate, end=todate)
#     print(stock_data.head())
#     if stock_data.empty:
#         return pd.DataFrame()
#     # 선택적인 컬럼만 추출하고 NaN 값을 0으로 채움
#     stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
#     stock_data['PER'] = 0  # FinanceDataReader의 PER 필드 대체
#     return stock_data

def fetch_stock_data(ticker, fromdate, todate):
    # stock_data = yf.download(ticker, start=fromdate, end=todate)
    stock_data = fdr.DataReader(ticker, start=fromdate, end=todate)
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


@tf.function(reduce_retracing=True)
def predict_model(model, data):
    return model(data)

count = 0

for ticker in tickers[count:]:
    print(f"Processing {count+1}/{len(tickers)} : {ticker}")
    count += 1
    data = fetch_stock_data(ticker, start_date_us, today_us)
    if data.empty or len(data) < 60:  # 데이터가 충분하지 않으면 건너뜀
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    # Convert the scaled_data to a TensorFlow tensor
    scaled_data_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)

    X, Y = create_dataset(scaled_data_tensor.numpy(), 60)  # numpy()로 변환하여 create_dataset 사용
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    model_file_path = os.path.join(model_dir, f'{ticker}_model_v1.Keras')
    if os.path.exists(model_file_path):
        model = tf.keras.models.load_model(model_file_path)
    else:
        model = create_model((X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # 10 에포크 동안 개선 없으면 종료
        verbose=1,
        mode='min',
        restore_best_weights=True  # 최적의 가중치를 복원
    )

    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1,
              validation_data=(X_val, Y_val), callbacks=[early_stopping])

    close_scaler = MinMaxScaler()
    close_prices_scaled = close_scaler.fit_transform(data[['Close']].values)

    # Make predictions with the model
    predictions = predict_model(model, tf.convert_to_tensor(X[-PREDICTION_PERIOD:], dtype=tf.float32))
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
    plt.title(f'{ticker} - Actual vs Predicted Prices {today_us} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_path = os.path.join(output_dir, f'{today_us} [ {future_return:.2f}% ] {ticker} [ {last_price} ] {timestamp}.png')
    plt.savefig(file_path)
    plt.close()
