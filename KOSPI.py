import os
import pandas as pd
import numpy as np
from pykrx import stock
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

# 시작 종목 인덱스 ( 중단된 경우 다시 시작용 )
count = 0
# 예측 기간
PREDICTION_PERIOD = 7
# 예측 성장률
EXPECTED_GROWTH_RATE = 5
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 365
# EarlyStopping
EARLYSTOPPING_PATIENCE = 8

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


tickers = stock.get_market_ticker_list(market="KOSPI")
# 종목 코드와 이름 딕셔너리 생성
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

output_dir = 'D:\\kospi_stocks'
# model_dir = os.path.join(output_dir, 'models')
model_dir = 'models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 주식 데이터와 기본적인 재무 데이터를 가져온다
# def fetch_stock_data(ticker, fromdate, todate):
#     ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
#     fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker)
#     stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
#
#     if 'PER' not in fundamental.columns:
#         print(f"PER data not available for {ticker} ({stock_name}). Filling with 0.")
#         fundamental['PER'] = 0  # PER 열이 없는 경우 0으로 채움
#
#     # PER 값이 NaN인 경우 0으로 채움
#     fundamental['PER'] = fundamental['PER'].fillna(0)
#     data = pd.concat([ohlcv, fundamental['PER']], axis=1).fillna(0)
#     return data

def fetch_stock_data(ticker, fromdate, todate):
    ohlcv = stock.get_market_ohlcv_by_date(fromdate, todate, ticker)
    daily_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker) # 기본 일별
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')

    # 'PER' 컬럼이 존재하는지 먼저 확인
    if 'PER' not in daily_fundamental.columns:
        # 일별 데이터에서 PER 정보가 없으면 월별 데이터 요청
        monthly_fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker, "m")
        if 'PER' in monthly_fundamental.columns:
            # 월별 PER 정보를 일별 데이터에 매핑
            daily_fundamental['PER'] = monthly_fundamental['PER'].reindex(daily_fundamental.index, method='ffill')
        else:
            # 월별 PER 정보도 없는 경우 0으로 처리
            daily_fundamental['PER'] = 0
    else:
        # 일별 PER 데이터 사용, NaN 값 0으로 채우기
        daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)

    # PER 데이터가 없으면 0으로 채우기
    if 'PER' not in daily_fundamental.columns or daily_fundamental['PER'].isnull().all():
        print(f"PER data not available for {ticker} ({stock_name}). Filling with 0.")
        daily_fundamental['PER'] = 0

    # PER 값이 NaN인 경우 0으로 채움
    daily_fundamental['PER'] = daily_fundamental['PER'].fillna(0)
    data = pd.concat([ohlcv, daily_fundamental[['PER']]], axis=1).fillna(0)
    return data

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    if len(dataset) < look_back:
        return np.array(X), np.array(Y)  # 빈 배열 반환
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

# 결과를 저장할 배열
saved_tickers = []

for ticker in tickers[count:]:
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} {ticker}")
    count += 1

    data = fetch_stock_data(ticker, start_date, today)
    if data.empty or len(data) < 60: # 데이터가 충분하지 않으면 건너뜀
        print(f"Not enough data for {ticker} to proceed.")
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    # X, Y = create_dataset(tf.convert_to_tensor(scaled_data), 60)

    # Python 객체 대신 TensorFlow 텐서를 사용
    # Convert the scaled_data to a TensorFlow tensor
    scaled_data_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)
    X, Y = create_dataset(scaled_data_tensor.numpy(), 60)  # numpy()로 변환하여 create_dataset 사용

    if len(X) == 0 or len(Y) == 0:
        print(f"Not enough samples for {ticker} to form a batch.")
        continue

    # 난수 데이터셋 분할
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    model_file_path = os.path.join(model_dir, f'{ticker}_model_v1.Keras')
    if os.path.exists(model_file_path):
        model = tf.keras.models.load_model(model_file_path)
    else:
        model = create_model((X_train.shape[1], X_train.shape[2]))
        # 지금은 매번 학습할 예정이다
        # model.fit(X, Y, epochs=3, batch_size=32, verbose=1, validation_split=0.1)
        # model.save(model_file_path)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLYSTOPPING_PATIENCE,  # 지정한 에포크 동안 개선 없으면 종료
        verbose=1,
        mode='min',
        restore_best_weights=True  # 최적의 가중치를 복원
    )

    # 체크포인트 설정
    # checkpoint = ModelCheckpoint(
    #     model_file_path,
    #     monitor='val_loss',
    #     save_best_only=True,
    #     mode='min',
    #     verbose=0
    # )

    # 입력 X에 대한 예측 Y 학습
    # model.fit(X, Y, epochs=50, batch_size=32, verbose=1, validation_split=0.1) # verbose=1 은 콘솔에 진척도
    # 모델 학습
    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1, # 충분히 모델링 되었으므로 20번만
              validation_data=(X_val, Y_val), callbacks=[early_stopping]) # 체크포인트 자동저장

    close_scaler = MinMaxScaler()
    close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)

    # 예측, 입력 X만 필요하다
    # predictions = model.predict(X[-PREDICTION_PERIOD:])
    # predicted_prices = close_scaler.inverse_transform(predictions).flatten()

    # 텐서 입력 사용하여 예측 실행 (권고)
    # predictions = predict_model(model, X[-PREDICTION_PERIOD:])
    # Make predictions with the model
    predictions = predict_model(model, tf.convert_to_tensor(X[-PREDICTION_PERIOD:], dtype=tf.float32))

    predicted_prices = close_scaler.inverse_transform(predictions.numpy()).flatten()
    model.save(model_file_path)

    last_close = data['종가'].iloc[-1]
    future_return = (predicted_prices[-1] / last_close - 1) * 100

    # 성장률 이상만
    if future_return < EXPECTED_GROWTH_RATE:
        continue

    extended_prices = np.concatenate((data['종가'].values, predicted_prices))
    extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
    last_price = data['종가'].iloc[-1]

    plt.figure(figsize=(26, 10))
    plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices', color='blue')
    plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'{ticker} - Actual vs Predicted Prices {today} {stock_name} [ {last_price} ] (Expected Return: {future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price} ] {timestamp}.png')
    plt.savefig(file_path)
    plt.close()

    saved_tickers.append(ticker)

print("Files were saved for the following tickers:")
print(saved_tickers)