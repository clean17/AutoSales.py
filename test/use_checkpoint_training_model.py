from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pykrx import stock
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# 데이터 수집
ticker = '000150'  # 예시: 000150 종목 코드
data = stock.get_market_ohlcv_by_date("2020-01-01", "2023-09-20", ticker)

# 필요한 컬럼 선택 및 NaN 값 처리
data = data[['시가', '고가', '저가', '종가', '거래량']].fillna(0)

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 시계열 데이터를 윈도우로 나누기
sequence_length = 60  # 예: 최근 60일 데이터를 기반으로 예측
X = []
y = []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length, 3])  # 종가(Close) 예측

X = np.array(X)
y = np.array(y)

# 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 정의 및 학습
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=False),
    Dense(128),
    Dense(64),
    Dense(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 체크포인트 설정
checkpoint = ModelCheckpoint('checkpoint.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), callbacks=[checkpoint])

# 베스트 모델 로드
model.load_weights('checkpoint.h5')

'''
Mean Squared Error: 0.0015389196947469608
Mean Absolute Error: 0.026108635373992564
Root Mean Squared Error: 0.039229066962482816
R-squared: 0.9625979185023417

'''
# 예측 값 (predictions)과 실제 값 (y_val)
predictions = model.predict(X_val)

# MSE
mse = mean_squared_error(y_val, predictions)
print("Mean Squared Error:", mse)

# MAE
mae = mean_absolute_error(y_val, predictions)
print("Mean Absolute Error:", mae)

# RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# R-squared
r2 = r2_score(y_val, predictions)
print("R-squared:", r2)

# 스케일러 객체 다시 생성
scaler_output = MinMaxScaler()
scaler_output.fit(data.iloc[:, :4])  # 첫 4개 열을 사용하여 스케일러 학습

# 예측 결과 복원
predictions = predictions.reshape(-1, 1)  # 예측 결과를 2차원 배열로 변환
original_predictions = scaler_output.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 3)), predictions), axis=1))[:, 3]

# 예측 결과를 데이터프레임으로 변환
predict_df = pd.DataFrame(original_predictions, columns=['Close'])
predict_df = predict_df.applymap(lambda x: int(x))

# 예측 결과의 날짜 생성
last_date = data.index[-1]
date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predict_df), freq='D')
predict_df.index = date_range

print(predict_df)
