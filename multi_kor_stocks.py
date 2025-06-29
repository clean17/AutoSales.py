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
from utils import create_lstm_model, create_multistep_dataset, fetch_stock_data, add_technical_features, get_kor_ticker_list
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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
AVERAGE_TRADING_VALUE = 3_000_000_000 # 평균거래대금 30억
EXPECTED_GROWTH_RATE = 5
DATA_COLLECTION_PERIOD = 400 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45

today = datetime.today().strftime('%Y%m%d')
today_us = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')

# tickers = ['077970', '079160', '112610', '025540', '003530', '357880', '131970', '009450', '310210', '353200', '136150', '064350', '066575', '005880', '272290', '204270', '066570', '456040', '373220', '096770', '005490', '006650', '042700', '068240', '003280', '067160', '397030', '480370', '085660', '328130', '476040', '241710', '357780', '232140', '011170', '020180', '074600', '042000', '003350', '065350', '004490', '482630', '005420', '033100', '018880', '417200', '332570', '058970', '011790', '053800', '338220', '195870', '010950', '455900', '082740', '225570', '445090', '068760', '007070', '361610', '443060', '089850', '413640', '005850', '141080', '005380', '098460', '277810', '011780', '005810', '075580', '112040', '012510', '240810', '403870', '376900', '001740', '035420', '103140', '068270', '013990', '001450', '457190', '293580', '475150', '280360', '097950', '058820', '034220', '084370', '178320']
# tickers = ['480370']
tickers = get_kor_ticker_list()

ticker_to_name = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}

# for file_name in os.listdir(output_dir):
#     if file_name.startswith(today):
#         send2trash(os.path.join(output_dir, file_name))


# 결과를 저장할 배열
results = []

for count, ticker in enumerate(tickers):
    stock_name = ticker_to_name.get(ticker, 'Unknown Stock')
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")

    data = add_technical_features(fetch_stock_data(ticker, start_date, today))

########################################################################

    actual_prices = data['종가'].values # 종가 배열
    last_close = actual_prices[-1]

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 30:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 500원 미만이면 패스
    last_row = data.iloc[-1]
    if last_row['종가'] < 500:
        # print("                                                        종가가 0이거나 500원 미만 → pass")
        continue

    # 최근 한달 거래대금 중 4억 미만이 있으면 패스
    month_data = data.tail(20)
    month_trading_value = month_data['거래량'] * month_data['종가']
    # 하루라도 거래대금이 4억 미만이 있으면 제외
    if (month_trading_value < 400_000_000).any():
        # print(f"                                                        최근 4주 중 거래대금 4억 미만 발생 → pass")
        continue

    # 일일 평균 거래량이 부족하면 패스
    # average_volume = data['거래량'].mean()
    # if average_volume <= AVERAGE_VOLUME:
    #     # print(f"                                                        평균 거래량({average_volume:.0f}주)이 부족 → pass")
    #     continue

    # 전체 평균 거래대금이 기준치 이하면 패스
    # trading_value = data['거래량'] * data['종가']
    # average_trading_value = trading_value.mean()
    # if average_trading_value <= AVERAGE_TRADING_VALUE:
    #     formatted_value = f"{average_trading_value / 100_000_000:.0f}억"
    #     # print(f"                                                        평균 거래대금({formatted_value})이 부족 → pass")
    #     continue

    # 최근 2주 거래대금이 기준치 이하면 패스
    recent_data = data.tail(10)
    recent_trading_value = recent_data['거래량'] * recent_data['종가']
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{recent_average_trading_value / 100_000_000:.0f}억"
        # print(f"                                                        최근 2주 평균 거래대금({formatted_recent_value})이 부족 → pass")
        continue

    # 최고가 대비 현재가가 50% 이상 하락한 경우 건너뜀
    # max_close = np.max(actual_prices)
    # drop_pct = ((max_close - last_close) / max_close) * 100
    # if drop_pct >= 50:
    #     # print(f"                                                        최고가 대비 현재가가 50% 이상 하락한 경우 → pass : {drop_pct:.2f} %")
    #     # continue
    #     pass

    # 모든 4일 연속 구간에서 첫날 대비 마지막날 xx% 이상 급등하면 패스
    window_start = actual_prices[-10:-3]   # 0 ~ N-4
    window_end = actual_prices[-7:]      # 3 ~ N-1
    ratio = window_end / window_start   # numpy, pandas Series/DataFrame만 벡터화 연산 지원, ratio는 결과 리스트
    if np.any(ratio >= 1.6):
        print(f"                                                        최근 4일 연속 구간에서 첫날 대비 60% 이상 상승 → pass")
        continue

    # 현재 종가가 4일 전에 비해서 크게 하락하면 패스
    close_4days_ago = data['종가'].iloc[-5]
    rate = (last_close / close_4days_ago - 1) * 100
    if rate <= -18:
        print(f"                                                        4일 전 대비 {rate:.2f}% 하락 → pass")
        continue  # 또는 return

    # # 최근 한달 동안의 변동률이 5%가 한번도 안되면 패스
    # idx_list = [-5, -10, -15, -20]
    # pass_flag = True
    # for idx in idx_list:
    #     past_close = data['종가'].iloc[idx]
    #     change = abs(last_close / past_close - 1) * 100
    #     if change >= 3: # 기준치
    #         pass_flag = False
    #         break
    # if pass_flag:
    #     print(f"                                                        최근 4주간 가격변동 3% 미만 → pass")
    #     # continue
    #     pass

    # 최근 3일, 2달 평균 거래량 계산, 최근 3일 거래량이 최근 2달 거래량의 80% 안되면 패스
    recent_3_avg = data['거래량'][-3:].mean()
    recent_2months_avg = data['거래량'][-40:].mean()
    if recent_3_avg < recent_2months_avg * 0.25:
        temp = (recent_3_avg/recent_2months_avg * 100)
        print(f"                                                        최근 3일의 평균거래량이 최근 2달 평균거래량의 25% 미만 → pass : {temp:.2f} %")
        continue
        # pass


    # # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    # ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
    # if data['MA5'].iloc[-1] < data['MA20'].iloc[-1] and ma_angle_5 < 0:
    #     # print(f"                                                        5일선이 20일선 보다 낮을 경우 → pass")
    #     # continue
    #     pass

########################################################################


    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data.values)
    # feature_cols = [
    #     '종가', '고가', '저가', '거래량',
    # ]
    feature_cols = [
        '종가', '고가', 'PBR', '저가', '거래량', 'RSI14', 'ma10_gap',
    ]

    X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
    # print(np.isfinite(X_for_model).all())  # True면 정상, False면 비정상
    # print(np.where(~np.isfinite(X_for_model)))  # 문제 있는 위치 확인
    scaled_data = scaler.fit_transform(X_for_model) # fit 하면 그 데이터의 min/max만 기억
    # 슬라이딩 윈도우로 전체 데이터셋 생성
    X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD, 0)
    # print("X.shape:", X.shape) # X.shape: (0,) 데이터가 부족해서 슬라이딩 윈도우로 샘플이 만들어지지 않음
    # print("Y.shape:", Y.shape)

    # 머신러닝/딥러닝 모델 입력을 위해 학습/검증 분리
    # 3차원 유지: (n_samples, look_back, n_features)
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, shuffle=False)  # 시계열 데이터라면 shuffle=False 권장, random_state는 의미 없음(어차피 순서대로 나누니까)
    if X_train.shape[0] < 50:
        print("                                                        샘플 부족 : ", X_train.shape[0])
        continue

    # 모델 생성 및 학습
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]), PREDICTION_PERIOD,
                              lstm_units=[128,64], dense_units=[64,32])

    # 콜백 설정
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # history = model.fit(X, Y, batch_size=8, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])
    history = model.fit(X_train, y_train, batch_size=8, epochs=200,
                        validation_data=(X_val, y_val),
                        shuffle=False, verbose=0, callbacks=[early_stop])

    # 모델 평가
    # val_loss = model.evaluate(X_val, y_val, verbose=1)
    # print("Validation Loss :", val_loss)

    # X_val : 검증에 쓸 여러 시계열 구간의 집합
    # predictions : 검증셋 각 구간(윈도우)에 대해 미래 PREDICTION_PERIOD만큼의 예측치 반환
    # shape: (검증샘플수, 예측일수)
    predictions = model.predict(X_val, verbose=0)

    # 학습이 최소한으로 되었는지 확인 후 실제 예측을 시작
    # R-squared; (0=엉망, 1=완벽)
    r2 = r2_score(y_val, predictions)
    if r2 < 0.7:
        # print(f"                                                        R-squared 0.7 미만이면 패스 : ", {r2:.2f})
        continue


    # X_input 생성 (마지막 구간)
    X_input = X[-1:]
    future_preds = model.predict(X_input, verbose=0).flatten() # 1차원 벡터로 변환
    # print('future', future_preds)

    # 종가 scaler fit (실제 데이터로)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = data['종가'].values.reshape(-1, 1) # DataFrame에서 ‘종가’(Close Price) 컬럼의 값을 1차원 배열로 꺼냄
    # print(close_prices)
    close_scaler.fit(close_prices) # close_prices 데이터에서 최소값/최대값을 학습(기억)함 >>  예측값(정규화된 상태)을 실제 가격 단위로 되돌릴 때 필요

    # 모델 예측값(future_preds)은 정규화된 값임 >> scaler로 실제 가격 단위(원래 스케일)로 되돌림 (역정규화)
    predicted_prices = close_scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten() # (PREDICTION_PERIOD, )

    # 날짜 처리 : 예측 구간의 미래 날짜 리스트 생성, start는 마지막 날짜 다음 영업일(Business day)부터 시작
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=PREDICTION_PERIOD, freq='B')
    avg_future_return = (np.mean(predicted_prices) / last_close - 1) * 100

    # 기대 성장률 미만이면 건너뜀
    if avg_future_return < EXPECTED_GROWTH_RATE:
        # print(f"  예상 : {avg_future_return:.2f}%")
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
    three_months_ago = data_plot.index.max() - pd.DateOffset(months=4)
    data_plot_recent = data_plot[data_plot.index >= three_months_ago].copy()
    recent_n = len(data_plot_recent)

    # 3. 미래 날짜도 문자열로 변환
    future_dates_str = pd.to_datetime(future_dates).strftime('%Y-%m-%d')

    # 4. 그래프 (윗부분: 가격/지표, 아랫부분: 거래량)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- 상단: 가격 + 볼린저밴드 + 예측 ---
    # 실제 가격
    ax1.plot(data_plot_recent['date_str'], actual_prices[-recent_n:], label='실제 가격', marker='s', markersize=6, markeredgecolor='white')

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
        [actual_prices[-recent_n:][-1], predicted_prices[0]],
        linestyle='dashed', color='tomato', linewidth=1.5
    )

    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'{today_us}   {stock_name} [ {ticker} ] (Expected Return: {avg_future_return:.2f}%)')

    # --- 하단: 거래량 (양/음/동색 구분) ---
    ax2.bar(data_plot_recent['date_str'], data_plot_recent['거래량'], color=bar_colors, alpha=0.65)
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