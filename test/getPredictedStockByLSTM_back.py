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

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 예측 기간
PREDICTION_PERIOD = 3
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 100

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

output_dir = 'D:\\stocks'
# model_dir = os.path.join(output_dir, 'models')
model_dir = 'models'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 주식 데이터(시가, 고가, 저가, 종가, 거래량)와 재무 데이터(PER)를 가져온다
def fetch_stock_data(ticker, fromdate, todate):
    ohlcv = stock.get_market_ohlcv_by_date(fromdate=fromdate, todate=today, ticker=ticker)
    fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker)
    if 'PER' not in fundamental.columns:
        fundamental['PER'] = 0
    else:
        fundamental['PER'] = fundamental['PER'].fillna(0)
    data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)
    return data

# def create_dataset(dataset, look_back=60):
#     X, Y = [], []
#     for i in range(len(dataset) - look_back):
#         X.append(dataset[i:i+look_back])
#         Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
#     return np.array(X), np.array(Y)

def create_multistep_dataset(dataset, look_back=25, n_future=5):
    X, Y = [], []
    for i in range(len(dataset) - look_back - n_future + 1):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back:i+look_back+n_future, 3])
    return np.array(X), np.array(Y)

# LSTM 모델 학습 및 예측 함수 정의
def create_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(n_future)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



ticker = '002710'
stock_name = stock.get_market_ticker_name(ticker)

# 데이터 로드 및 스케일링
data = fetch_stock_data(ticker, start_date, today)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

n_future = 3
look_back = 15

# 데이터셋 생성
X, Y = create_multistep_dataset(scaled_data, look_back=look_back, n_future=n_future)

model = create_model((X.shape[1], X.shape[2]))

# model_file_path = os.path.join(model_dir, f'{ticker}_model_v4.h5')
# if os.path.exists(model_file_path):
#     model.load_weights(model_file_path)
#     print(f"{ticker}: 이전 가중치 로드")

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X, Y, batch_size=8, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop]) # fit; 학습 > 저장

# model.save_weights(model_file_path)
# print(f"{ticker}: 학습 후 가중치 저장됨 -> {model_file_path}")

# 종가 scaler fit (실제 데이터로)
close_scaler = MinMaxScaler()
close_prices = data['종가'].values.reshape(-1, 1)
close_scaler.fit(close_prices)

# 미래 예측
X_input = scaled_data[-look_back:].reshape(1, look_back, scaled_data.shape[1])
future_preds = model.predict(X_input, verbose=0).flatten()
future_prices = close_scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten()

# 실제 마지막 종가와 미래 예측 종가 합치기
all_prices = np.concatenate((data['종가'].values, future_prices))
extended_dates = pd.date_range(start=data.index[0], periods=len(all_prices), freq='B')
# print('extended_dates', extended_dates)
# print('all_prices', all_prices)

# 미래 수익률 계산
last_price = data['종가'].iloc[-1]
future_return = (future_prices[-1] / last_price - 1) * 100

# 그래프 그리기
plt.figure(figsize=(16, 8))
plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices')
plt.plot(
    extended_dates[len(data['종가'].values)-1:],
    np.concatenate(([data['종가'].values[-1]], future_prices)),
    label='Predicted Prices', color='orange', linestyle='--', marker='o'
)
plt.plot(
    [extended_dates[len(data['종가'].values)-1], extended_dates[len(data['종가'].values)]],
    [data['종가'].values[-1], future_prices[0]],
    linestyle='dashed', color='gray', linewidth=1.5, label='Actual-Predicted Bridge'
)
plt.title(f'{today} {ticker} {stock_name} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
# plt.xticks(rotation=45)
plt.grid(True)




# 이미지 저장
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ] {timestamp}.png')
plt.savefig(file_path)
plt.close()