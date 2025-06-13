import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash
import ast
from utils import create_dataset, create_multistep_dataset, get_safe_ticker_list, fetch_stock_data

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

output_dir = 'D:\\kospi_stocks'
os.makedirs(output_dir, exist_ok=True)

PREDICTION_PERIOD = 3
LOOK_BACK = 15
AVERAGE_VOLUME = 25000 # 평균거래량
AVERAGE_TRADING_VALUE = 3000000000 # 평균거래대금
MAX_ITERATIONS = 1
EXPECTED_GROWTH_RATE = 5
DATA_COLLECTION_PERIOD = 100

today = datetime.today().strftime('%Y%m%d')
today_us = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')


# tickers_kospi = get_safe_ticker_list(market="KOSPI")
# tickers_kosdaq = get_safe_ticker_list(market="KOSDAQ")
# tickers = tickers_kospi + tickers_kosdaq # 전체
tickers = ['014970']

ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}
# 성장률을 저장할 튜플
ticker_returns = {}

for file_name in os.listdir(output_dir):
    if file_name.startswith(today):
        # print(f"Sending to trash: {file_name}")
        send2trash(os.path.join(output_dir, file_name))


# 결과를 저장할 배열
results = []


for count, ticker in enumerate(tickers):
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

    data = fetch_stock_data(ticker, start_date, today)

    # 종가가 0.0이거나 500원 미만이면 건너뜀
    last_row = data.iloc[-1]
    if last_row['종가'] == 0.0 or last_row['종가'] < 500:
#         print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
        continue

    # 데이터가 부족하면 건너뜀
    if data.empty or len(data) < LOOK_BACK:
#         print(f"                                                        데이터가 부족하여 작업을 건너뜁니다")
        continue

    # 일일 평균 거래량/거래대금 체크
    average_volume = data['거래량'].mean()
    if average_volume <= AVERAGE_VOLUME:
#         print(f"                                                        평균 거래량({average_volume:.0f}주)이 부족하여 작업을 건너뜁니다.")
        continue

    trading_value = data['거래량'] * data['종가']
    average_trading_value = trading_value.mean()
    if average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_value = f"{average_trading_value / 100000000:.0f}억"
#         print(f"                                                        평균 거래액({formatted_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # 최근 한 달 거래액 체크
    recent_data = data.tail(20)
    recent_trading_value = recent_data['거래량'] * recent_data['종가']
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{recent_average_trading_value / 100000000:.0f}억"
#         print(f"                                                        최근 한 달 평균 거래액({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    X, Y = create_dataset(scaled_data, LOOK_BACK)
    if len(X) < 2 or len(Y) < 2:
        print(f"                                                        데이터셋이 부족하여 작업을 건너뜁니다.")
        continue

    # 모델 생성 및 학습
    model = create_model((X.shape[1], X.shape[2]))

    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, Y, batch_size=8, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])

    # 미래 예측 구간 반복
    close_scaler = MinMaxScaler()
    close_prices = data['종가'].values.reshape(-1, 1)
    close_scaler.fit(close_prices)

    last_window = scaled_data[-LOOK_BACK:]
    current_window = last_window.copy()
    future_preds = []

    for _ in range(PREDICTION_PERIOD):
        pred = model.predict(current_window.reshape(1, LOOK_BACK, scaled_data.shape[1]), verbose=0)
        future_preds.append(pred[0, 0]) # 예측 결과 저장
        next_row = np.zeros(scaled_data.shape[1])
        next_row[3] = pred  # 종가 인덱스에 예측값
        current_window = np.vstack([current_window[1:], next_row]) # 다음 윈도우 만들기: 뒤에 예측값 추가, 앞에서 하나 빼서 15개 유지

    # 예측값 스케일 역변환
    future_preds_arr = np.array(future_preds).reshape(-1, 1)
    predicted_prices = close_scaler.inverse_transform(future_preds_arr).flatten()
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=PREDICTION_PERIOD, freq='B')

    last_close = data['종가'].iloc[-1]
    avg_future_return = (np.mean(predicted_prices) / last_close - 1) * 100

    # 최고가 대비 현재가 하락률 계산
    max_close = np.max(data['종가'].values)
    last_close = data['종가'].iloc[-1]
    drop_pct = ((max_close - last_close) / max_close) * 100

    # 40% 이상 하락한 경우 건너뜀
    if drop_pct >= 40:
        continue

    # 기대 성장률 미만이면 건너뜀
    if avg_future_return < EXPECTED_GROWTH_RATE:
        continue

    # 결과 저장
    results.append((avg_future_return, stock_name))

    # 기존 파일 삭제
    for file_name in os.listdir(output_dir):
        if file_name.startswith(f"{today}") and stock_name in file_name and ticker in file_name:
            print(f"Deleting existing file: {file_name}")
            os.remove(os.path.join(output_dir, file_name))

    # 그래프 저장
    extended_prices = np.concatenate((data['종가'].values, predicted_prices))
    last_price = data['종가'].iloc[-1]

    plt.figure(figsize=(16, 8))
    # 실제 데이터
    plt.plot(data.index, data['종가'].values, label='Actual Prices')

    # 예측 데이터
    plt.plot(
        future_dates,
        predicted_prices,
        label='Predicted Prices',
        linestyle='--', marker='o', color='orange'
    )

    # 마지막 실제값과 첫 예측값을 점선으로 연결
    plt.plot(
        [data.index[-1], future_dates[0]],
        [data['종가'].values[-1], predicted_prices[0]],
        linestyle='dashed', color='gray', linewidth=1.5
    )

    plt.title(f'{today_us}   {stock_name} [ {last_price} ] (Expected Return: {avg_future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)

    final_file_name = f'{today} [ {avg_future_return:.2f}% ] {stock_name} [{ticker}].png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()

####################################

# 정렬 및 출력
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, stock_name in results:
    print(f"==== [ {avg_future_return:.2f}% ] {stock_name} ====")