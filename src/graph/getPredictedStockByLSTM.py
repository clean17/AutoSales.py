from pykrx import stock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

'''
정규화   - kernel_regularizer=l2(0.01) > L2정규화: 과적합 방지
Dropout - 과적합방지, 일반화 향상
ReLU    - 비선형 학습(복잡한 패턴)
3가지 옵션 모두 예측치가 엇나감
'''


def create_dataset(dataset, look_back=60):
    # 시계열 데이터를 윈도우로 나누기, 60: 최근 60일 데이터를 기반으로 예측
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)


# LSTM 모델 생성 및 학습
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


def train_and_save_model(data, look_back=60, model_path='my_model.keras'):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, Y = create_dataset(scaled_data, look_back)
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, Y, batch_size=32, epochs=50, verbose=0, validation_split=0.1)
    model.save(model_path, save_format='keras')
    return scaler


# 모델을 사용하여 예측 함수
def predict_stock_price_with_saved_model(data, scaler, model_path='my_model.keras', look_back=60, days_to_predict=7):
    scaled_data = scaler.transform(data)

    def create_prediction_dataset(dataset, look_back=60):
        X = []
        for i in range(len(dataset) - look_back + 1):
            X.append(dataset[i:(i + look_back), :])
        return np.array(X)

    X = create_prediction_dataset(scaled_data, look_back)
    model = load_model(model_path)

    # Ensure we have enough data to predict the requested days
    if len(X) < days_to_predict:
        raise ValueError(f"Not enough data to predict {days_to_predict} days. Available data points: {len(X)}")

    predictions = model.predict(X[-days_to_predict:])
    predictions = predictions.reshape(-1, 1)

    # Make sure the dimensions match for concatenation
    zeros_array = np.zeros((predictions.shape[0], data.shape[1] - 1))
    predictions_extended = np.concatenate((zeros_array, predictions), axis=1)

    original_predictions = scaler.inverse_transform(predictions_extended)[:, -1]

    return original_predictions






# 데이터 수집
today = datetime.today().strftime('%Y%m%d')
last_year = (datetime.today() - timedelta(days=365)).strftime('%Y%m%d')
# ticker = "005930"  # 삼성전자
# ticker = "012750"  # 에스원
ticker = "000150"


# 주요 피처(시가, 고가, 저가, 종가, 거래량) + 재무 지표(PER)
ohlcv = stock.get_market_ohlcv_by_date(fromdate=last_year, todate=today, ticker=ticker)
fundamental = stock.get_market_fundamental_by_date(fromdate=last_year, todate=today, ticker=ticker)

# PER 값이 없는 경우 대체 값 사용 (예: 0으로 채우기)
fundamental['PER'] = fundamental['PER'].fillna(0)

# 주가 데이터와 재무 지표 결합 및 NaN 값 처리
# data = pd.concat([ohlcv['종가'], fundamental['PER']], axis=1).dropna()
data = pd.concat([ohlcv, fundamental['PER']], axis=1).dropna()

# 필요한 컬럼 선택 및 NaN 값 처리
data = data[['시가', '고가', '저가', '종가', '거래량', 'PER']].fillna(0)

# 모델 학습 및 저장
my_model_path = 'new_model_origin.keras'
scaler = train_and_save_model(data, model_path=my_model_path)



# 주가 예측
prediction_period = 7 # 예측일
# predictions = model.predict(X[-7:])
predictions = predict_stock_price_with_saved_model(data, scaler, model_path=my_model_path, look_back=60, days_to_predict=prediction_period)



# 예측 결과 시각화
# predicted_prices = scaler.inverse_transform(np.concatenate((predictions, np.zeros((7, X.shape[2] - 1))), axis=1))[:, 0]
predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], data.shape[1] - 1)), predictions.reshape(-1, 1)), axis=1))[:, -1]
# predicted_prices = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((7, data.shape[1] - 1))), axis=1))[:, 0]
# actual_prices = data['종가'].values[-7:]

# 마지막 실제 가격 데이터 날짜 가져오기
last_date = ohlcv.index[-1]

# 예측된 가격을 위한 날짜 생성
prediction_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_period+1)

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
