import matplotlib
matplotlib.use('Agg')
import os
import sys
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from send2trash import send2trash
import ast
from utils import create_model, create_multistep_dataset, get_safe_ticker_list, fetch_stock_data, compute_rsi

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
AVERAGE_TRADING_VALUE = 2_500_000_000 # 평균거래대금 25억
EXPECTED_GROWTH_RATE = 5
DATA_COLLECTION_PERIOD = 300 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45
# DATA_COLLECTION_PERIOD = 120 # 샘플 수 = 81(100일 기준) - 20 - 4 + 1 = 58
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

    data = fetch_stock_data(ticker, start_date, today)
    # print('# 데이터 길이', len(data))

########################################################################

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 30:
        # print(f"                                                        데이터가 부족하여 작업을 건너뜁니다")
        continue

    # 500원 미만이면 패스
    last_row = data.iloc[-1]
    if last_row['종가'] < 500:
        # print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
        continue

    # 최근 한달 거래대금 중 4억 미만이 있으면 패스
    month_data = data.tail(20)
    month_trading_value = month_data['거래량'] * month_data['종가']
    # 하루라도 거래대금이 4억 미만이 있으면 제외
    if (month_trading_value < 400_000_000).any():
        # print(f"                                                        최근 4주 중 거래대금 4억 미만 발생 → 제외")
        continue

    # 일일 평균 거래량이 부족하면 패스
    # average_volume = data['거래량'].mean()
    # if average_volume <= AVERAGE_VOLUME:
    #     # print(f"                                                        평균 거래량({average_volume:.0f}주)이 부족하여 작업을 건너뜁니다.")
    #     continue

    # 전체 평균 거래대금이 기준치 이하면 패스
    # trading_value = data['거래량'] * data['종가']
    # average_trading_value = trading_value.mean()
    # if average_trading_value <= AVERAGE_TRADING_VALUE:
    #     formatted_value = f"{average_trading_value / 100_000_000:.0f}억"
    #     # print(f"                                                        평균 거래대금({formatted_value})이 부족하여 작업을 건너뜁니다.")
    #     continue

    # 최근 2주 거래대금이 기준치 이하면 패스
    recent_data = data.tail(10)
    recent_trading_value = recent_data['거래량'] * recent_data['종가']
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{recent_average_trading_value / 100_000_000:.0f}억"
        # print(f"                                                        최근 2주 평균 거래대금({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # 최고가 대비 현재가가 40% 이상 하락한 경우 건너뜀
    actual_prices = data['종가'].values # 종가 배열
    last_close = actual_prices[-1]
    max_close = np.max(actual_prices)
    drop_pct = ((max_close - last_close) / max_close) * 100
    if drop_pct >= 40:
        continue

    # 모든 4일 연속 구간에서 첫날 대비 마지막날 xx% 이상 급등하면 패스
    window_start = actual_prices[-10:-3]   # 0 ~ N-4
    window_end = actual_prices[-7:]      # 3 ~ N-1
    ratio = window_end / window_start   # numpy, pandas Series/DataFrame만 벡터화 연산 지원, ratio는 결과 리스트
    if np.any(ratio >= 1.6):
        print(f"                                                        최근 4일 연속 구간에서 첫날 대비 60% 이상 상승: 제외")
        continue

    # 현재 종가가 4일 전에 비해서 크게 하락하면 패스
    close_4days_ago = data['종가'].iloc[-5]
    rate = (last_close / close_4days_ago - 1) * 100
    if rate <= -18:
        print(f"                                                        4일 전 대비 {rate:.2f}% 하락 → 학습 제외")
        continue  # 또는 return

    # 최근 한달 동안의 변동률이 5%가 한번도 안되면 패스
    idx_list = [-5, -10, -15, -20]
    pass_flag = True
    for idx in idx_list:
        past_close = data['종가'].iloc[idx]
        change = abs(last_close / past_close - 1) * 100
        if change >= 5: # 기준치
            pass_flag = False
            break
    if pass_flag:
        # print(f"                                                        최근 4주간 가격변동 5% 미만 → 학습 pass")
        continue

########################################################################

    # 볼린저밴드
    data['MA5'] = data['종가'].rolling(window=5).mean()
    data['MA15'] = data['종가'].rolling(window=15).mean()
    data['MA20'] = data['종가'].rolling(window=window).mean()
    data['MA30'] = data['종가'].rolling(window=30).mean()
    data['MA60'] = data['종가'].rolling(window=60).mean()
    data['STD20'] = data['종가'].rolling(window=window).std()
    data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
    data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])
    # 이동평균 기울기(변화량)
    data['MA5_slope'] = data['MA5'].diff() # diff() 차이르 계산하는 함수
    data['MA15_slope'] = data['MA15'].diff()
    data['MA20_slope'] = data['MA20'].diff()
    data['MA60_slope'] = data['MA60'].diff()
    # 볼린저밴드 위치 (현재가가 상단/하단 어디쯤?)
    data['BB_perc'] = (data['종가'] - data['LowerBand']) / (data['UpperBand'] - data['LowerBand'] + 1e-9) # # 0~1 사이. 1이면 상단, 0이면 하단, 0.5면 중앙
    # 거래량 증감률 >> 거래량이 0이거나, 직전 거래량이 0일 때 문제 발생
    data['Volume_change'] = data['거래량'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    # RSI (14일)
    data['RSI14'] = compute_rsi(data['종가'])
    # 캔들패턴 (양봉/음봉, 장대양봉 등)
    data['is_bullish'] = (data['종가'] > data['시가']).astype(int) # 양봉이면 1, 음봉이면 0
    # 장대양봉(시가보다 종가가 2% 이상 상승)
    # data['long_bullish'] = ((data['종가'] - data['시가']) / data['시가'] > 0.02).astype(int)
    # 당일 변동폭 (고가-저가 비율)
    data['day_range_pct'] = (data['고가'] - data['저가']) / data['저가']

    # 볼린저밴드
    # last_close = data['종가'].iloc[-1]
    # upper = data['UpperBand'].iloc[-1]
    # lower = data['LowerBand'].iloc[-1]
    # center = data['MA20'].iloc[-1]

    # 매수/매도 조건
    # if last_close <= lower:
    #     print("                                                        과매도, 매수 신호!")
    # elif last_close >= upper:
    #     print("과매수, 매도 신호!")
    # else:
    #     print("중립(관망)")


    # 이동평균선이 하락중이면 제외
    ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
    # ma_angle_15 = data['MA15'].iloc[-1] - data['MA15'].iloc[-2]
    # ma_angle_20 = data['MA20'].iloc[-1] - data['MA20'].iloc[-2]

    # ma_cnt = 0
    # if ma_angle_5 > 0:
    #     ma_cnt = ma_cnt + 1
    # if ma_angle_15 > 0:
    #     ma_cnt = ma_cnt + 1
    # if ma_angle_20 > 0:
    #     ma_cnt = ma_cnt + 1

    # if ma_cnt >= 2:
    #     pass
    # else:
    #     # 하락/횡보면 건너뜀
    #     # print(f"                                                        이동평균선이 상승이 아니므로 건너뜁니다.")
    #     continue

    # # 이동평균선이 하락중이면 제외 (2가지 조건 비교)
    # ma_angle = data['MA30'].iloc[-1] - data['MA30'].iloc[-2] # 오늘의 이동평균선 방향
    #
    # if ma_angle > 0:
    #     pass
    # else:
    #     # print(f"                                                        이동평균선이 상승이 아니므로 건너뜁니다.")
    #     continue

    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    if data['MA5'].iloc[-1] < data['MA20'].iloc[-1] and ma_angle_5 < 0:
        # print(f"                                                        5일선이 20일선 보다 낮을 경우 : 제외")
        continue
        
########################################################################

    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data.values)
    # feature_cols = ['시가', '고가', '저가', '종가', '거래량', 'MA20', 'UpperBand', 'LowerBand', 'PER', 'PBR']
    feature_cols = [
       '종가', '거래량', 'MA15_slope', 'RSI14', 'BB_perc'
    ]

    X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
    # print(np.isfinite(X_for_model).all())  # True면 정상, False면 비정상
    # print(np.where(~np.isfinite(X_for_model)))  # 문제 있는 위치 확인
    scaled_data = scaler.fit_transform(X_for_model)
    X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD)
    print("X.shape:", X.shape) # X.shape: (0,) 데이터가 부족해서 슬라이딩 윈도우로 샘플이 만들어지지 않음
    print("Y.shape:", Y.shape)

    # 모델 생성 및 학습
    model = create_model((X.shape[1], X.shape[2]), PREDICTION_PERIOD)

    # 콜백 설정
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, Y, batch_size=16, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])

    # 종가 scaler fit (실제 데이터로)
    close_scaler = MinMaxScaler()
    close_prices = data['종가'].values.reshape(-1, 1) # DataFrame에서 ‘종가’(Close Price) 컬럼의 값을 1차원 배열로 꺼냄
    close_scaler.fit(close_prices) # close_prices 데이터에서 최소값/최대값을 학습(기억)함

    # X_input 생성 (마지막 구간)
    X_input = scaled_data[-LOOK_BACK:].reshape(1, LOOK_BACK, scaled_data.shape[1])
    future_preds = model.predict(X_input, verbose=0).flatten()
    predicted_prices = close_scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten() # (PREDICTION_PERIOD, )

    # 날짜 처리
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=PREDICTION_PERIOD, freq='B')
    avg_future_return = (np.mean(predicted_prices) / last_close - 1) * 100

    # 기대 성장률 미만이면 건너뜀
    if avg_future_return < EXPECTED_GROWTH_RATE:
        print(f"예상 : {avg_future_return:.2f}%")
        continue

    # 결과 저장
    results.append((avg_future_return, stock_name))

    # 기존 파일 삭제
    for file_name in os.listdir(output_dir):
        if file_name.startswith(f"{today}") and stock_name in file_name and ticker in file_name:
            print(f"Deleting existing file: {file_name}")
            os.remove(os.path.join(output_dir, file_name))




#######################################################################

    # 1. 조건별 색상 결정 (거래량 바 차트)
    up = data['종가'] > data['시가']
    down = data['종가'] < data['시가']
    bar_colors = np.where(up, 'red', np.where(down, 'blue', 'gray'))

    # 2. 인덱스 문자열 컬럼 추가 (x축 통일용)
    data_plot = data.copy()
    data_plot['date_str'] = data_plot.index.strftime('%Y-%m-%d')

    # data_plot.index가 DatetimeIndex라고 가정
    three_months_ago = data_plot.index.max() - pd.DateOffset(months=3)
    data_plot_recent = data_plot[data_plot.index >= three_months_ago].copy()

    # 3. 미래 날짜도 문자열로 변환
    future_dates_str = pd.to_datetime(future_dates).strftime('%Y-%m-%d')

    # 4. 그래프 (윗부분: 가격/지표, 아랫부분: 거래량)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- 상단: 가격 + 볼린저밴드 + 예측 ---
    # 실제 가격
    ax1.plot(data_plot_recent['date_str'], actual_prices, label='실제 가격', marker='s', markersize=6, markeredgecolor='white')

    # 예측 가격 (미래 날짜)
    ax1.plot(future_dates_str, predicted_prices, label='예측 가격', linestyle='--', marker='s', markersize=7, markeredgecolor='white', color='tomato')

    # 이동평균, 볼린저밴드, 영역 채우기
    if all(x in data_plot_recent.columns for x in ['MA20', 'UpperBand', 'LowerBand']):
        ax1.plot(data_plot_recent['date_str'], data_plot_recent['MA20'], label='20일 이동평균선', alpha=0.8)
        if 'MA5' in data_plot_recent.columns:
            ax1.plot(data_plot_recent['date_str'], data_plot_recent['MA5'], label='5일 이동평균선', alpha=0.8)
        ax1.plot(data_plot_recent['date_str'], data_plot_recent['UpperBand'], label='볼린저밴드 상한선', linestyle='--', alpha=0.8)
        ax1.plot(data_plot_recent['date_str'], data_plot_recent['LowerBand'], label='볼린저밴드 하한선', linestyle='--', alpha=0.8)
        ax1.fill_between(data_plot_recent['date_str'], data_plot_recent['UpperBand'], data_plot_recent['LowerBand'], color='gray', alpha=0.18)

    # 마지막 실제값과 첫 번째 예측값을 점선으로 연결
    ax1.plot(
        [data_plot_recent['date_str'].iloc[-1], future_dates_str[0]],
        [actual_prices[-1], predicted_prices[0]],
        linestyle='dashed', color='tomato', linewidth=1.5
    )

    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'{today_us}   {stock_name} [ {ticker} ] (Expected Return: {avg_future_return:.2f}%)')

    # --- 하단: 거래량 (양/음/동색 구분) ---
    ax2.bar(data_plot_recent['date_str'], data_plot_recent['거래량'], color=bar_colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.grid(True)

    # x축 라벨: 10일 단위만 표시 (과도한 라벨 겹침 방지)
    tick_idx = np.arange(0, len(data_plot_recent), 10)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(data_plot_recent['date_str'].iloc[tick_idx])

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 파일 저장 (옵션)
    final_file_name = f'{today} [ {avg_future_return:.2f}% ] {stock_name} [{ticker}].png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()

#######################################################################

# 정렬 및 출력
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, stock_name in results:
    print(f"==== [ {avg_future_return:.2f}% ] {stock_name} ====")







'''
    plt.figure(figsize=(16, 8))
    # 실제 데이터
    plt.plot(data.index, actual_prices, label='실제 가격')
    # 예측 데이터
    plt.plot(future_dates, predicted_prices, label='예측 가격', linestyle='--', marker='o', color='tomato')

    if all(x in data.columns for x in ['MA20', 'UpperBand', 'LowerBand']):
        plt.plot(data.index, data['MA20'], label='20일 이동평균선')
        plt.plot(data.index, data['MA5'], label='5일 이동평균선')
        plt.plot(data.index, data['UpperBand'], label='볼린저밴드 상한선', linestyle='--') # Upper Band (2σ)
        plt.plot(data.index, data['LowerBand'], label='볼린저밴드 하한선', linestyle='--') # Lower Band (2σ)
        plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)

    # 마지막 실제값과 첫 번째 예측값을 점선으로 연결
    plt.plot(
        [data.index[-1], future_dates[0]],  # x축: 마지막 실제날짜와 첫 예측날짜
        [actual_prices[-1], predicted_prices[0]],  # y축: 마지막 실제종가와 첫 예측종가
        linestyle='dashed', color='gray', linewidth=1.5
    )

    plt.title(f'{today_us}   {stock_name} [ {ticker} ] (Expected Return: {avg_future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    final_file_name = f'{today} [ {avg_future_return:.2f}% ] {stock_name} [{ticker}].png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()
'''