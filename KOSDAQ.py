import os
import pandas as pd
import numpy as np
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)

# 시작 종목 인덱스 ( 중단된 경우 다시 시작용 )
count = 0
# 예측 기간
PREDICTION_PERIOD = 5
# 예측 성장률
EXPECTED_GROWTH_RATE = 6
# 데이터 수집 기간
DATA_COLLECTION_PERIOD = 365
# 과적합 방지
EARLYSTOPPING_PATIENCE = 10

LOOK_BACK = 30
# 반복 횟수
EPOCHS_SIZE = 150
BATCH_SIZE = 32

AVERAGE_VOLUME = 20000
AVERAGE_TRADING_VALUE = 1000000000

# 그래프 저장 경로
output_dir = 'D:\\kosdaq_stocks'
# 모델 저장 경로
# model_dir = os.path.join(output_dir, 'models')
model_dir = 'kosdaq_30_models'

today = datetime.today().strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


# tickers =  ['460930', '203450', '251970', '420570', '058450', '373200', '219550', '445090']
tickers = stock.get_market_ticker_list(market="KOSDAQ")

# 종목 코드와 이름 딕셔너리 생성
ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}



if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 주식 데이터와 기본적인 재무 데이터를 가져온다
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
    model = tf.keras.Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
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

    # print('# ============ debug ============ 1')
    data = fetch_stock_data(ticker, start_date, today)

    # 마지막 행의 데이터를 가져옴
    last_row = data.iloc[-1]
    # 종가가 0.0인지 확인
    if last_row['종가'] == 0.0:
        print("종가가 0 이므로 작업을 건너뜁니다.")
        continue

    # 데이터가 충분하지 않으면 건너뜀
    if data.empty or len(data) < LOOK_BACK:
        print(f"Not enough data for {ticker} to proceed.")
        continue

    average_volume = data['거래량'].mean() # volume
    if average_volume <= AVERAGE_VOLUME:
        print('average_volume', average_volume)
        continue

    trading_value = data['거래량'] * data['종가']
    average_trading_value = trading_value.mean()
    if average_trading_value <= AVERAGE_TRADING_VALUE:
        print('average_trading_value', average_trading_value)
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    # Python 객체 대신 TensorFlow 텐서를 사용
    # Convert the scaled_data to a TensorFlow tensor
    # print('# ============ debug ============ 2')
    scaled_data_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)
    X, Y = create_dataset(scaled_data_tensor.numpy(), LOOK_BACK)  # numpy()로 변환하여 create_dataset 사용

    if len(X) < 2 or len(Y) < 2:
        print(f"Not enough samples for {ticker} to split into train and test sets.")
        continue

    # 난수 데이터셋 분할
    # print('# ============ debug ============ 3')
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

    model_file_path = os.path.join(model_dir, f'{ticker}_model_v1.Keras')
    if os.path.exists(model_file_path):
        model = tf.keras.models.load_model(model_file_path)
    else:
        model = create_model((X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLYSTOPPING_PATIENCE,  # 10 에포크 동안 개선 없으면 종료
        verbose=0,
        mode='min',
        restore_best_weights=True  # 최적의 가중치를 복원
    )

    # 모델 학습
    model.fit(X_train, Y_train, epochs=EPOCHS_SIZE, batch_size=BATCH_SIZE, verbose=0, # 충분히 모델링 되었으므로 20번만
              validation_data=(X_val, Y_val), callbacks=[early_stopping]) # 체크포인트 자동저장

    close_scaler = MinMaxScaler()
    close_prices_scaled = close_scaler.fit_transform(data[['종가']].values)


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

    plt.figure(figsize=(16, 8))
    plt.plot(extended_dates[:len(data['종가'].values)], data['종가'].values, label='Actual Prices', color='blue')
    plt.plot(extended_dates[len(data['종가'].values)-1:], np.concatenate(([data['종가'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'{today} {ticker} {stock_name} [ {last_price:.2f} ] (Expected Return: {future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_path = os.path.join(output_dir, f'{today} [ {future_return:.2f}% ] {stock_name} {ticker} [ {last_price:.2f} ] {timestamp}.png')
    plt.savefig(file_path)
    plt.close()

    saved_tickers.append(ticker)

print("Files were saved for the following tickers:")
print(saved_tickers)