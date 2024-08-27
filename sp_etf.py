import matplotlib
matplotlib.use('Agg')  # Agg 백엔드를 사용하여 tkinter 관련 문제를 회피
import os
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash


count = 0
PREDICTION_PERIOD = 2
EXPECTED_GROWTH_RATE = 0
DATA_COLLECTION_PERIOD = 365
EARLYSTOPPING_PATIENCE = 30
LOOK_BACK = 30
EPOCHS_SIZE = 150
BATCH_SIZE = 32
DROPOUT = 0.3
tickers = ['SOXL', 'SOXS', 'TSLL', 'NVDL', 'NEON', 'CONL', 'LABU', 'TQQQ', 'SQQQ', 'FFIE', 'ASTS']

# 미국 동부 시간대 설정
us_timezone = pytz.timezone('America/New_York')
now_us = datetime.now(us_timezone)
# 현재 시간 출력
today_us = now_us.strftime('%Y-%m-%d %H:%M:%S')
print("미국 동부 시간 기준 현재 시각:", today_us)
# 데이터 수집 시작일 계산
start_date_us = (now_us - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')
print("미국 동부 시간 기준 데이터 수집 시작일:", start_date_us)

today_us = datetime.today().strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y%m%d')
# start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')

# ETF 종목을 가져오는 함수
def get_etf_tickers():
    etf_list = pd.read_html("https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds")[0]  # Wikipedia에서 ETF 목록 가져오기
    etf_tickers = etf_list['Ticker symbol'].tolist()  # 티커 리스트 추출
    return etf_tickers

# tickers = get_etf_tickers()

output_dir = 'D:\\sp500'
model_dir = 'sp_30(2)etf_models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def fetch_stock_data(ticker, fromdate, todate):
    stock_data = yf.download(ticker, start=fromdate, end=todate)
    if stock_data.empty:
        return pd.DataFrame()

    # yfinance를 통해 주식 정보 가져오기
    stock_info = yf.Ticker(ticker).info

    # PER 값을 info에서 추출, 없는 경우 0으로 처리
    per_value = stock_info.get('trailingPE', 0)  # trailingPE를 사용하거나 없으면 0

    # 주식 데이터에 PER 컬럼 추가
    # stock_data['PER'] = per_value

    # 선택적인 컬럼 추출 및 NaN 값 처리
    # stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'PER']].fillna(0)
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0)
    return stock_data

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back])
        Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

# LSTM 모델 학습 및 예측 함수 정의
def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add((LSTM(256, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(DROPOUT))

    # 두 번째 LSTM 층
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(DROPOUT))

    # 세 번째 LSTM 층
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(DROPOUT))

    # Dense 레이어
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(16, activation='relu'))

    # 출력 레이어
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


for file_name in os.listdir(output_dir):
    if file_name.startswith(today):
        # print(f"Sending to trash: {file_name}")
        send2trash(os.path.join(output_dir, file_name))

# @tf.function(reduce_retracing=True)
# def predict_model(model, data):
#     return model(data)

# 결과를 저장할 배열
saved_tickers = []

# for ticker in tickers:
for count, ticker in enumerate(tickers):
    print(f"Processing {count+1}/{len(tickers)} : {ticker}")
    data = fetch_stock_data(ticker, start_date_us, today_us)

    if data.empty or len(data) < LOOK_BACK:  # 데이터가 충분하지 않으면 건너뜀
        continue

    last_row = data.iloc[-1]
    if last_row['Close'] == 0.0:
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    X, Y = create_dataset(scaled_data, LOOK_BACK)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

    # Convert the scaled_data to a TensorFlow tensor
    # scaled_data_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)
    # X, Y = create_dataset(scaled_data_tensor.numpy(), LOOK_BACK)  # numpy()로 변환하여 create_dataset 사용
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


    model_file_path = os.path.join(model_dir, f'{ticker}_model_v2.Keras')
    if os.path.exists(model_file_path):
        model = tf.keras.models.load_model(model_file_path)
        if model.input_shape != (None, X_train.shape[1], X_train.shape[2]):
            print('Loaded model input shape does not match data input shape. Creating a new model.')
            model = create_model((X_train.shape[1], X_train.shape[2]))
    else:
        model = create_model((X_train.shape[1], X_train.shape[2]))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLYSTOPPING_PATIENCE,  # 10 에포크 동안 개선 없으면 종료
        verbose=0,
        mode='min',
        restore_best_weights=True  # 최적의 가중치를 복원
    )

    model.fit(X_train, Y_train, epochs=EPOCHS_SIZE, batch_size=BATCH_SIZE, verbose=0,
              validation_data=(X_val, Y_val), callbacks=[early_stopping])

    close_scaler = MinMaxScaler()
    close_prices_scaled = close_scaler.fit_transform(data[['Close']].values)

    # 예측, 입력 X만 필요하다
    predictions = model.predict(X[-PREDICTION_PERIOD:])
    predicted_prices = close_scaler.inverse_transform(predictions).flatten()

    # Make predictions with the model
    # predictions = predict_model(model, tf.convert_to_tensor(X[-PREDICTION_PERIOD:], dtype=tf.float32))
    # predicted_prices = close_scaler.inverse_transform(predictions.numpy()).flatten()
    model.save(model_file_path)

    last_close = data['Close'].iloc[-1]
    future_return = (predicted_prices[-1] / last_close - 1) * 100

    # if future_return < EXPECTED_GROWTH_RATE:
    #     continue

    extended_prices = np.concatenate((data['Close'].values, predicted_prices))
    extended_dates = pd.date_range(start=data.index[0], periods=len(extended_prices))
    last_price = data['Close'].iloc[-1]

    plt.figure(figsize=(16, 8))
    plt.plot(extended_dates[:len(data['Close'].values)], data['Close'].values, label='Actual Prices', color='blue')
    plt.plot(extended_dates[len(data['Close'].values)-1:], np.concatenate(([data['Close'].values[-1]], predicted_prices)), label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'{today_us} {ticker} [ {last_price:.2f} ] (Expected Return: {future_return:+.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    final_file_name = f'{today} [ {future_return:+.2f}% ] {ticker} [ {last_price:.2f} ].png'
    final_file_path = os.path.join(output_dir, final_file_name)
    print(final_file_name)
    plt.savefig(final_file_path)
    plt.close()

    saved_tickers.append(ticker)

print("Files were saved for the following tickers:")
print(saved_tickers)

for file_name in os.listdir(output_dir):
    if file_name.startswith(today):
        print(f"{file_name}")