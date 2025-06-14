import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash
import ast
from utils import create_model, create_multistep_dataset, get_safe_ticker_list, fetch_stock_data

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
EXPECTED_GROWTH_RATE = -10
DATA_COLLECTION_PERIOD = 100
window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수

today = datetime.today().strftime('%Y%m%d')
today_us = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

tickers_kospi = get_safe_ticker_list(market="KOSPI")
tickers_kosdaq = get_safe_ticker_list(market="KOSDAQ")
tickers = tickers_kospi + tickers_kosdaq # 전체

ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

# for file_name in os.listdir(output_dir):
#     if file_name.startswith(today):
#         send2trash(os.path.join(output_dir, file_name))



# 결과를 저장할 배열
results = []

for count, ticker in enumerate(tickers):
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

    data = fetch_stock_data(ticker, start_date, today)

    # 볼린저밴드
    data['MA20'] = data['종가'].rolling(window=window).mean()
    data['STD20'] = data['종가'].rolling(window=window).std()
    data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
    data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data.values)
    feature_cols = ['시가', '고가', '저가', '종가', '거래량', 'MA20', 'UpperBand', 'LowerBand', 'PER', 'PBR']
    X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
    scaled_data = scaler.fit_transform(X_for_model)
    X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD)

########################################################################

    # 데이터가 부족하면 건너뜀
    if data.empty or len(data) < LOOK_BACK:
#         print(f"                                                        데이터가 부족하여 작업을 건너뜁니다")
        continue

    if len(X) < 2 or len(Y) < 2:
        print(f"                                                        데이터셋이 부족하여 작업을 건너뜁니다 (신규 상장).")
        continue

    # 종가가 0.0이거나 500원 미만이면 건너뜀
    last_row = data.iloc[-1]
    if last_row['종가'] == 0.0 or last_row['종가'] < 500:
#         print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
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

    actual_prices = data['종가'].values # 최근 종가 배열
    last_close = actual_prices[-1]

    # 최고가 대비 현재가 하락률 계산
    max_close = np.max(actual_prices)
    drop_pct = ((max_close - last_close) / max_close) * 100

    # 40% 이상 하락한 경우 건너뜀
    if drop_pct >= 40:
        continue

########################################################################

    # 현재가
    last_close = data['종가'].iloc[-1]
    upper = data['UpperBand'].iloc[-1]
    lower = data['LowerBand'].iloc[-1]
    center = data['MA20'].iloc[-1]

    # 매수/매도 조건
    if last_close <= lower:
        print("                                                        과매도, 매수 신호!")
#     elif last_close >= upper:
#         print("과매수, 매도 신호!")
#     else:
#         print("중립(관망)")

    # 이동평균선이 하락중이면 제외
    data['MA_10'] = data['종가'].rolling(window=10).mean()
    ma_angle = data['MA_10'].iloc[-1] - data['MA_10'].iloc[-2] # 오늘의 이동평균선 방향

    if ma_angle > 0:
        # 상승 중인 종목만 예측/추천
        pass
    else:
        # 하락/횡보면 건너뜀
        print(f"                                                        이동평균선이 상승이 아니므로 건너뜁니다.")
        continue

########################################################################

    # 모델 생성 및 학습
    model = create_model((X.shape[1], X.shape[2]), PREDICTION_PERIOD)

    # 콜백 설정
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, Y, batch_size=8, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])

    # 종가 scaler fit (실제 데이터로)
    close_scaler = MinMaxScaler()
    close_prices = data['종가'].values.reshape(-1, 1)
    close_scaler.fit(close_prices)

    # X_input 생성 (마지막 구간)
    X_input = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, scaled_data.shape[1])
    future_preds = model.predict(X_input, verbose=0).flatten()
    predicted_prices = close_scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten() # (PREDICTION_PERIOD, )

    # 날짜 처리
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=PREDICTION_PERIOD, freq='B')
    avg_future_return = (np.mean(predicted_prices) / last_close - 1) * 100

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
    plt.plot(data.index, actual_prices, label='실제 가격')
    # 예측 데이터
    plt.plot(future_dates, predicted_prices, label='예측 가격', linestyle='--', marker='o', color='orange')

    if all(x in data.columns for x in ['MA20', 'UpperBand', 'LowerBand']):
        plt.plot(data.index, data['MA20'], label='20일 이동평균선') # MA20
        plt.plot(data.index, data['UpperBand'], label='볼린저밴드 상한선', linestyle='--') # Upper Band (2σ)
        plt.plot(data.index, data['LowerBand'], label='볼린저밴드 하한선', linestyle='--') # Lower Band (2σ)
        plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)

    # 마지막 실제값과 첫 번째 예측값을 점선으로 연결
    plt.plot(
        [data.index[-1], future_dates[0]],  # x축: 마지막 실제날짜와 첫 예측날짜
        [actual_prices[-1], predicted_prices[0]],  # y축: 마지막 실제종가와 첫 예측종가
        linestyle='dashed', color='gray', linewidth=1.5
    )

    plt.title(f'{today_us}   {stock_name} [ {last_price} ] (Expected Return: {avg_future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
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