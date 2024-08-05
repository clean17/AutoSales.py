from pykrx import stock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l2
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 데이터 수집
today = datetime.today().strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')
# ticker = "005930"  # 삼성전자
# ticker = "012750"  # 에스원
ticker = "248070"

ohlcv = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
fundamental = stock.get_market_fundamental_by_date(fromdate=last_year, todate=today, ticker=ticker)
data = pd.concat([ohlcv['종가'], fundamental['PER']], axis=1).dropna()

# 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X, Y = create_dataset(scaled_data, look_back)

# LSTM 모델 생성 및 학습
'''
유닛증가 - 50 > 100
정규화   - kernel_regularizer=l2(0.01) > L2정규화: 과적합 방지
 ㄴ 정규화를 하지 않으면 테스트 데이터에 치중, 새로운 데이터에 성능이 저하될 수 있다... 예상치가 너무 높아서 비활성화함
Dropout - 과적합방지, 일반화 향상
 ㄴ 없는 경우가 더 예측이 정확
ReLU    - 비선형 학습(복잡한 패턴)
 ㄴ 없는 경우가 더 예측이 정확
'''
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=False),
    Dense(128),
    Dense(64),
    Dense(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, batch_size=32, epochs=20) # epochs 학습 횟수 batch_size 학습에 사용할 샘플의 수

# 주가 예측
predictions = model.predict(X[-7:])

# 예측 결과 시각화
predicted_prices = scaler.inverse_transform(np.concatenate((predictions, np.zeros((7, X.shape[2] - 1))), axis=1))[:, 0]
actual_prices = data['종가'].values[-7:]

# 마지막 실제 가격 데이터 날짜 가져오기
last_date = ohlcv.index[-1]

# 예측된 가격을 위한 날짜 생성
prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=8)

# 예측 결과 시각화 수정
plt.figure(figsize=(26, 10))

# 실제 가격 데이터 플롯
plt.plot(ohlcv.index, ohlcv['종가'], label='Actual Prices', color='blue')

# 예측 시작 지점에 실제 데이터의 마지막 가격을 포함하여 연속적으로 만들기
predicted_prices_with_continuity = np.insert(predicted_prices, 0, ohlcv['종가'].iloc[-1])

# 수정된 예측 가격 데이터 플롯
plt.plot(prediction_dates, predicted_prices_with_continuity, label='Predicted Prices', color='red', linestyle='--')

plt.title('Actual Prices vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.show()


################### 모델 비교 ###################

'''
MSE가 낮을수록 실제와 동일하다

validation_split=0.1: 10% 검증 90% 학습

test_size=0.2, random_state=42 - 모델 학습 검증때만 사용

##############################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 정규화 사용 모델 학습 및 평가
model_regularized = Sequential([ # 0.021147689425332107
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model_regularized.compile(optimizer='adam', loss='mean_squared_error')
model_regularized.fit(X_train, Y_train, batch_size=32, epochs=50, validation_split=0.1)
predictions_regularized = model_regularized.predict(X_test)
mse_regularized = mean_squared_error(Y_test, predictions_regularized)

# 정규화 미사용 모델 학습 및 평가
model_non_regularized = Sequential([ # 0.005378332129577116
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model_non_regularized.compile(optimizer='adam', loss='mean_squared_error')
model_non_regularized.fit(X_train, Y_train, batch_size=32, epochs=50, validation_split=0.1)
predictions_non_regularized = model_non_regularized.predict(X_test)
mse_non_regularized = mean_squared_error(Y_test, predictions_non_regularized)

print(f'MSE with regularization: {mse_regularized}')
print(f'MSE without regularization: {mse_non_regularized}')
###########################################
'''


# # 모델 학습 후 저장
# model_64.save('my_model.h5')
#
# # 모델 불러오기
# model = load_model('my_model.h5')