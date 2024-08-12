from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pykrx import stock
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
sequence_length = 30  # 예: 최근 60일 데이터를 기반으로 예측
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
# model = Sequential([ # 첫번째 모델
#     LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     LSTM(128, return_sequences=True),
#     LSTM(64, return_sequences=False),
#     Dense(128),
#     Dense(64),
#     Dense(32),
#     Dense(1)
# ])

# model = Sequential([ # 두번째 모델
#     LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
#     Dropout(0.2),  # 드롭아웃 추가
#     LSTM(128, return_sequences=True),
#     Dropout(0.2),  # 드롭아웃 추가
#     LSTM(64, return_sequences=False),
#     Dense(128, activation='relu'),
#     Dropout(0.2),  # 드롭아웃 추가
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(1)
# ])

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model = Sequential()
# model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(LSTM(64, return_sequences=False))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 체크포인트 설정
# checkpoint = ModelCheckpoint('test/checkpoint.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])

# 베스트 모델 로드
# model.load_weights('test/checkpoint.h5')

# 예측 값 (predictions)과 실제 값 (y_val)
predictions = model.predict(X_val)

# MSE : 작을수록 좋다
mse = mean_squared_error(y_val, predictions)
print("Mean Squared Error:", mse)

# MAE : 작을수록 좋다
mae = mean_absolute_error(y_val, predictions)
print("Mean Absolute Error:", mae)

# RMSE : 작을수록 좋다
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# R-squared : 1에 가까울수록 좋다
r2 = r2_score(y_val, predictions)
print("R-squared:", r2)

'''
 # 첫번째 모델
Mean Squared Error: 0.0015389196947469608
Mean Absolute Error: 0.026108635373992564
Root Mean Squared Error: 0.039229066962482816
R-squared: 0.9625979185023417

Mean Squared Error: 0.0010377053046394288
Mean Absolute Error: 0.02196434952639314
Root Mean Squared Error: 0.03221343360524346
R-squared: 0.9747794907640986

Mean Squared Error: 0.0009223075914731872
Mean Absolute Error: 0.01909549573758318
Root Mean Squared Error: 0.03036951747185304
R-squared: 0.9775841300751816


 # 두번째 모델
Mean Squared Error: 0.001495403046561555
Mean Absolute Error: 0.025586915380050586
Root Mean Squared Error: 0.03867044150978309
R-squared: 0.9636555521316281

Mean Squared Error: 0.0014755793851861153
Mean Absolute Error: 0.02666502293425685
Root Mean Squared Error: 0.03841327095140578
R-squared: 0.9641373486807768

# 2레이어 128
Mean Squared Error: 0.0011847279113061883
Mean Absolute Error: 0.023152705614646822
Root Mean Squared Error: 0.03441987668929376
R-squared: 0.9735369339305066

과적합 방지를 제거하면 실제 데이터 예측이 잘 안될 수 도 있다. 성능 벤치마크에 목숨걸지 마라
'''

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
