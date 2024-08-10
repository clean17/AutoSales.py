import os
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set random seed for reproducibility
tf.random.set_seed(42)

# ticker = 'CRL'
ticker = 'EPAM'

# 예측 기간
PREDICTION_PERIOD = 7
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

today = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')

today_us = today
# start_date_us = start_date

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
    stock_data['PER'] = 0  # PER 열을 수동으로 추가, 실제 PER 값이 필요하다면 추가 계산 필요
    return stock_data


def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 3])  # 종가(Close) 예측
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

# 데이터 로드 및 스케일링
data = fetch_stock_data(ticker, start_date_us, today_us)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# 데이터셋 생성
X, Y = create_dataset(scaled_data, 60)

# 난수 데이터셋 분할
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

model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=1,
          validation_data=(X_val, Y_val), callbacks=[early_stopping])

close_scaler = MinMaxScaler()
close_prices_scaled = close_scaler.fit_transform(data[['Close']].values)

# 예측
predictions = model.predict(X[-PREDICTION_PERIOD:])
predicted_prices = close_scaler.inverse_transform(predictions).flatten()

last_close = data['Close'].iloc[-1]
future_return = (predicted_prices[-1] / last_close - 1) * 100

extended_prices = np.concatenate((data['Close'].values, predicted_prices))
extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))

plt.figure(figsize=(26, 10))
plt.plot(extended_dates[:len(data['Close'].values)], data['Close'].values, label='Actual Prices', color='blue')
plt.plot(extended_dates[len(data['Close'].values)-1:], np.concatenate(([data['Close'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
plt.title(f'{ticker} - Actual vs Predicted Prices {today_us} [Last Price: {last_close}] (Expected Return: {future_return:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()

model.save(model_file_path)
