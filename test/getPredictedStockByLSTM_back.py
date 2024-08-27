import os
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Set random seed for reproducibility
tf.random.set_seed(42)

# 예측 기간
PREDICTION_PERIOD = 7
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 365

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


output_dir = 'D:\\stocks'
# model_dir = os.path.join(output_dir, 'models')
model_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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

# def create_model(input_shape):
#     model = Sequential([
#         LSTM(256, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),  # 드롭아웃 추가
#         LSTM(128, return_sequences=True),
#         Dropout(0.2),  # 드롭아웃 추가
#         LSTM(64, return_sequences=False),
#         Dense(128),
#         Dropout(0.2),  # 드롭아웃 추가
#         Dense(64),
#         Dense(32),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model


count = 0
# ticker = '000150'
ticker = '002710'
stock_name = stock.get_market_ticker_name(ticker)

# 데이터 로드 및 스케일링
data = fetch_stock_data(ticker, start_date, today)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# 데이터셋 생성
X, Y = create_dataset(scaled_data, 60)

# 데이터셋 분할
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
Y_train, Y_val = Y[:train_size], Y[train_size:]

model_file_path = os.path.join(model_dir, f'{ticker}_model_v1.Keras')
if os.path.exists(model_file_path):
    model = tf.keras.models.load_model(model_file_path)
else:
    model = create_model((X_train.shape[1], X_train.shape[2]))
    #지금은 매번 학습할 예정이다
    # model.fit(X, Y, epochs=3, batch_size=32, verbose=1, validation_split=0.1)
    # model.save(model_file_path)

# 모델 생성
# model = create_model((X_train.shape[1], X_train.shape[2]))

# 체크포인트 설정
model_file_path = os.path.join(model_dir, f'{ticker}_best_model.h5')
checkpoint = ModelCheckpoint(
    model_file_path,
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# 입력 X에 대한 예측 Y 학습
# model.fit(X, Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1) # verbose=1 은 콘솔에 진척도
# 모델 학습
model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1,
          validation_data=(X_val, Y_val), callbacks=[checkpoint])

# model.save(model_file_path) # 체크포인트가 자동으로 최적의 상태를 저장한다

close_scaler = MinMaxScaler()
close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

# 예측, 입력 X만 필요하다
predictions = model.predict(X[-PREDICTION_PERIOD:])
predicted_prices = close_scaler.inverse_transform(predictions).flatten()

last_close = data['종가'].iloc[-1]
future_return = (predicted_prices[-1] / last_close - 1) * 100

extended_prices = np.concatenate((data['종가'].values, predicted_prices))
extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
last_price = data['종가'].iloc[-1]

plt.figure(figsize=(16, 8))
plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices', color='blue')
plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
plt.title(f'{today} {ticker} {stock_name} [ {last_price:.2f} ] (Expected Return: {future_return:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
# file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price:.2f} ] {timestamp}.png')
# plt.savefig(file_path)
# plt.close()
