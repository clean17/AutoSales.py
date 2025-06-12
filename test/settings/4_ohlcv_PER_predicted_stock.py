import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 데이터 수집
# today = datetime.today().strftime('%Y%m%d')
today = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=100)).strftime('%Y%m%d')
ticker = "000660"

# 주식 데이터((시가, 고가, 저가, 종가, 거래량))와 재무 데이터(PER)를 가져온다
def fetch_stock_data(ohlcv, ticker, fromdate, todate):
    fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker)
    fundamental['PER'] = fundamental['PER'].fillna(0)
    data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)
    return data

# ohlcv = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
# data = fetch_stock_data(ohlcv, ticker, last_year, today)
# data.to_pickle(f'{ticker}.pkl')

# data.to_csv(f'{ticker}.csv')  # 원하는 경로/이름으로 저장
# data = pd.read_csv(f'{ticker}.csv', index_col=0, parse_dates=True)

data = pd.read_pickle(f'{ticker}.pkl')

# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 시계열 데이터를 윈도우로 나누기
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

look_back = 15
X, Y = create_dataset(scaled_data, look_back)
print("X.shape:", X.shape) # X.shape: (0,) 데이터가 부족해서 슬라이딩 윈도우로 샘플이 만들어지지 않음
print("Y.shape:", Y.shape)

# 모델 생성 및 학습
# model = Sequential([
#     LSTM(32, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
#     Dropout(0.2),
#     LSTM(16, return_sequences=False),
#     Dropout(0.2),
#     Dense(16, activation='relu'),
#     Dense(8, activation='relu'),
#     Dense(1)
# ])

model = Sequential([
    LSTM(16, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(8, return_sequences=False),
    Dropout(0.2),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 기존 모델이 있으면 불러와서 이어서 학습
# model_path = 'models/save_model.h5'
# if os.path.exists(model_path):
#     model = tf.keras.models.load_model(model_path)
#     print(f"{ticker}: 이전 모델 로드")

# 콜백 설정
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # 10회 동안 개선없으면 종료, 최적의 가중치를 복원

# 이어서(혹은 처음부터) 학습
# batch_size 4 > var_loss 0.056
# batch_size 8 > var_loss 0.045
history = model.fit(X, Y, batch_size=8, epochs=200, validation_split=0.1, shuffle=False, verbose=1, callbacks=[early_stop])

# 학습 후 모델 저장
# model.save(model_path)
# print(f"{ticker}: 학습 후 모델 저장됨 -> {model_path}")



plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

#
#
# # 종가 데이터에 대한 스케일러 생성 및 훈련
# close_scaler = MinMaxScaler()
# close_prices = data['종가'].values.reshape(-1, 1)
# close_scaler.fit(close_prices)
#
# # 실제 값과 예측 값 비교를 위한 그래프
# actual_prices = data['종가'].values
#
# n_future = 3  # 예측 기간
# last_window = scaled_data[-look_back:]  # 최근 데이터
#
# future_preds = []
# current_window = last_window.copy()
#
# for _ in range(n_future):
#     pred = model.predict(current_window.reshape(1, look_back, scaled_data.shape[1]), verbose=0)
#     # pred shape: (1, 1) → 실제 종가만 추출
#     future_preds.append(pred[0, 0])
#
#     # 다음 입력을 위해 윈도우 업데이트 (맨 뒤에 새 예측 추가)
#     # pred 값은 전체 feature 중 종가 위치(3번 인덱스)에 넣어줘야 함
#     next_row = np.zeros(scaled_data.shape[1])
#     next_row[3] = pred  # 종가만 예측값, 나머지 피처는 0 (or 이전값 유지하고 싶으면 응용)
#     current_window = np.vstack([current_window[1:], next_row])
#
# # 종가 스케일 역변환 (미리 fit된 close_scaler 사용)
# future_preds_arr = np.array(future_preds).reshape(-1, 1)
# future_prices = close_scaler.inverse_transform(future_preds_arr).flatten()
#
# future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_future, freq='B')  # 'B'=business day
#
# print("예측 종가:", future_prices)
#
# plt.figure(figsize=(10, 5))
# plt.plot(data.index, actual_prices, label='Actual Prices')
# plt.plot(future_dates, future_prices, label='Future Predicted Prices', linestyle='--', marker='o', color='orange')
#
# plt.plot(
#     [data.index[-1], future_dates[0]],
#     [actual_prices[-1], future_prices[0]],
#     linestyle='dashed', color='gray', linewidth=1.5
# )
#
# plt.title(f'Actual and 5-day Predicted Prices for {ticker}')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.grid(True)
# # plt.show()
#
#
#
#
# output_dir = 'D:\\stocks'
# last_price = data['종가'].iloc[-1]
# future_return = (future_prices[-1] / last_price - 1) * 100
# stock_name = stock.get_market_ticker_name(ticker)
# timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
# file_path = os.path.join(output_dir, f'4 {today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ] {timestamp}.png')
# plt.savefig(file_path)
# plt.close()

'''
loss 가 가장 낮은 세팅
32레이어, 입력 15, 배치 8
'''