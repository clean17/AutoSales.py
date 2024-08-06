import yfinance as yf
import pandas as pd
from pykrx import stock
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt



# # 데이터 수집
# ticker = "005930.KS"  # 삼성전자
# data = yf.download(ticker, start="2021-01-01", end="2022-12-31")
#
# # 필요한 컬럼 선택 및 NaN 값 처리
# data = data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)



# 데이터 수집
ticker = '000150'  # 예시: 000150 종목 코드
data = stock.get_market_ohlcv_by_date("2020-01-01", "2023-09-20", ticker)

# 필요한 컬럼 선택 및 NaN 값 처리
data = data[['시가', '고가', '저가', '종가', '거래량']].fillna(0)

'''
pykrx 데이터가 훨씬 세세하다 (모델 예측이 더 잘나온다)
'''


# 데이터 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 시계열 데이터를 윈도우로 나누기
def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        Y.append(dataset[i + look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

look_back = 60
X, Y = create_dataset(scaled_data, look_back)

# 학습 데이터와 테스트 데이터 분리
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 모델 생성
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(look_back, X_train.shape[2])))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X_train, Y_train, batch_size=32, epochs=50, validation_data=(X_test, Y_test))

'''
Train Actual (훈련 실제 값):
- 훈련 데이터에 대한 실제 종가 값을 나타냅니다.
- 주식 데이터의 훈련 기간 동안의 실제 주가 변동을 보여줍니다.

Test Actual (테스트 실제 값):
- 테스트 데이터에 대한 실제 종가 값을 나타냅니다.
- 주식 데이터의 테스트 기간 동안의 실제 주가 변동을 보여줍니다.
- 그래프에서 Y_train_actual 데이터 뒤에 이어져서 나타납니다.

Train Predict (훈련 예측 값):
- 훈련 데이터에 대한 모델의 예측 값을 나타냅니다.
- 모델이 훈련 데이터로 학습한 결과를 보여줍니다.

Test Predict (테스트 예측 값):
- 테스트 데이터에 대한 모델의 예측 값을 나타냅니다.
- 모델이 새로운 데이터(테스트 데이터)에서 예측한 결과를 보여줍니다.
- 그래프에서 Y_train_actual 데이터 뒤에 이어져서 나타납니다.

Train Actual과 Train Predict과 매우 유사하면 잘 학습되었다.
'''
# 예측
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측값 역변환
train_predict = scaler.inverse_transform(np.concatenate((np.zeros((train_predict.shape[0], 3)), train_predict, np.zeros((train_predict.shape[0], 1))), axis=1))[:,3]
test_predict = scaler.inverse_transform(np.concatenate((np.zeros((test_predict.shape[0], 3)), test_predict, np.zeros((test_predict.shape[0], 1))), axis=1))[:,3]

Y_train_actual = scaler.inverse_transform(np.concatenate((np.zeros((Y_train.shape[0], 3)), Y_train.reshape(-1, 1), np.zeros((Y_train.shape[0], 1))), axis=1))[:,3]
Y_test_actual = scaler.inverse_transform(np.concatenate((np.zeros((Y_test.shape[0], 3)), Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], 1))), axis=1))[:,3]

# 시각화
plt.figure(figsize=(26, 12))
plt.plot(Y_train_actual, label='Train Actual')
plt.plot(range(len(Y_train_actual), len(Y_train_actual) + len(Y_test_actual)), Y_test_actual, label='Test Actual')
plt.plot(train_predict, label='Train Predict')
plt.plot(range(len(Y_train_actual), len(Y_train_actual) + len(test_predict)), test_predict, label='Test Predict')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
