import matplotlib # tkinter 충돌 방지, Agg 백엔드를 사용하여 GUI를 사용하지 않도록 한다
matplotlib.use('Agg')
import os
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from send2trash import send2trash
from utils import create_lstm_model, create_multistep_dataset, fetch_stock_data_us, get_nasdaq_symbols, extract_stock_code_from_filenames, get_usd_krw_rate, add_technical_features_us, check_column_types
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

exchangeRate = get_usd_krw_rate()
output_dir = 'D:\\sp500'
os.makedirs(output_dir, exist_ok=True)
rsi_flag = 0

PREDICTION_PERIOD = 5
EXPECTED_GROWTH_RATE = 5
DATA_COLLECTION_PERIOD = 400
LOOK_BACK = 25
AVERAGE_VOLUME = 30000
AVERAGE_TRADING_VALUE = 2_000_000 # 28억 쯤

# 미국 동부 시간대 설정
now_us = datetime.now(pytz.timezone('America/New_York'))
# 현재 시간 출력
print("미국 동부 시간 기준 현재 시각:", now_us.strftime('%Y-%m-%d %H:%M:%S'))
# 데이터 수집 시작일 계산
start_date_us = (now_us - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')
print("미국 동부 시간 기준 데이터 수집 시작일:", start_date_us)

end_date = datetime.today().strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y%m%d')


# tickers = get_nasdaq_symbols()
# tickers = extract_stock_code_from_filenames(output_dir)
tickers=['RKLB']


# 결과를 저장할 배열
results = []

for count, ticker in enumerate(tickers):
    print(f"Processing {count+1}/{len(tickers)} : {ticker}")

    data = add_technical_features_us(fetch_stock_data_us(ticker, start_date_us, end_date))
#     check_column_types(fetch_stock_data_us(ticker, start_date_us, end_date), ['Close', 'Open', 'High', 'Low', 'Volume', 'PER', 'PBR']) # 타입과 shape 확인 > Series 가 나와야 한다
#     continue

    ########################################################################

    actual_prices = data['Close'].values # 최근 종가 배열
    last_close = actual_prices[-1]

    if data.empty or len(data) < 30:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 종가가 0.0이거나 500원 미만이면 건너뜀
    last_row = data.iloc[-1]
    if last_row['Close'] == 0.0 or last_row['Close'] < 0.4:
        # print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
        continue

    # 최근 2 주
    recent_data = data.tail(10)
    recent_trading_value = recent_data['Volume'] * recent_data['Close']     # 최근 2주 거래대금 리스트
    # 하루라도 4억 미만이 있으면 제외
    if (recent_trading_value < 300_000).any(): # 30만 달러 === 4억
#         print(f"                                                        최근 2주 중 거래대금 4억 미만 발생 → 제외")
        continue

#     # 일일 평균 거래량/거래대금 체크
#     average_volume = data['Volume'].mean()
#     if average_volume <= AVERAGE_VOLUME:
# #         print(f"                                                        평균 거래량({average_volume:.0f}주)이 부족하여 작업을 건너뜁니다.")
#         continue

#     trading_value = data['Volume'] * data['Close']
#     average_trading_value = trading_value.mean()
#     if average_trading_value <= AVERAGE_TRADING_VALUE:
#         formatted_value = f"{(average_trading_value * exchangeRate) / 100_000_000:.0f}억"
# #         print(f"                                                        평균 거래액({formatted_value})이 부족하여 작업을 건너뜁니다.")
#         continue

    recent_trading_value = recent_data['Volume'] * recent_data['Close']
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{(recent_average_trading_value * exchangeRate)/ 100_000_000:.0f}억"
        print(f"                                                        최근 2주 평균 거래액({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # rolling window로 5일 전 대비 현재가 3배 이상 오른 지점 찾기
    rolling_min = data['Close'].rolling(window=5).min()    # 5일 중 최소가
    ratio = data['Close'] / rolling_min

    if np.any(ratio >= 2.8):
        print(f"                                                        어느 5일 구간이든 2.8배 급등: 제외")
        continue


    # 최고가 대비 현재가 하락률 계산
    max_close = np.max(actual_prices)
    drop_pct = ((max_close - last_close) / max_close) * 100

    # 40% 이상 하락한 경우 건너뜀
    if drop_pct >= 50:
        continue

    # 모든 4일 연속 구간에서 첫날 대비 마지막날 xx% 이상 급등
    window_start = actual_prices[:-3]   # 0 ~ N-4
    window_end = actual_prices[3:]      # 3 ~ N-1
    ratio = window_end / window_start   # numpy, pandas Series/DataFrame만 벡터화 연산 지원, ratio는 결과 리스트

    if np.any(ratio >= 1.6):
        print(f"                                                        어떤 4일 연속 구간에서 첫날 대비 60% 이상 상승: 제외")
        continue

    last_close = data['Close'].iloc[-1]
    close_4days_ago = data['Close'].iloc[-5]

    rate = (last_close / close_4days_ago - 1) * 100

    if rate <= -18:
        print(f"                                                        4일 전 대비 {rate:.2f}% 하락 → 학습 제외")
        continue  # 또는 return

#     idx_list = [-7, -14, -21, -28]
#     pass_flag = True
#
#     for idx in idx_list:
#         past_close = data['Close'].iloc[idx]
#         change = abs(last_close / past_close - 1) * 100
#         if change >= 5: # 기준치
#             pass_flag = False
#             break
#
#     if pass_flag:
#         # print(f"                                                        최근 4주간 가격변동 5% 미만 → 학습 pass")
#         continue  # 또는 return

    # 최근 3일, 2달 평균 거래량 계산, 최근 3일 거래량이 최근 2달 거래량의 25% 안되면 패스
    recent_3_avg = data['Volume'][-3:].mean()
    recent_2months_avg = data['Volume'][-40:].mean()
    if recent_3_avg < recent_2months_avg * 0.25:
        temp = (recent_3_avg/recent_2months_avg * 100)
        print(f"                                                        최근 3일의 평균거래량이 최근 2달 평균거래량의 25% 미만 → pass : {temp:.2f} %")
        continue
        # pass

    ########################################################################

    # 현재가
#     last_close = data['Close'].iloc[-1]
#     upper = data['UpperBand'].iloc[-1]
#     lower = data['LowerBand'].iloc[-1]
#     center = data['MA20'].iloc[-1]

    # 매수/매도 조건
    # if last_close <= lower:
    #     print("                                                        과매도, 매수 신호!")
    #     elif last_close >= upper:
    #         print("과매수, 매도 신호!")
    #     else:
    #         print("중립(관망)")


    # 이동평균선이 하락중이면 제외 (2가지 조건 비교)
#     ma_angle = data['MA30'].iloc[-1] - data['MA30'].iloc[-2] # 오늘의 이동평균선 방향
# #     ma_angle = data['MA20'].iloc[-1] - data['MA20'].iloc[-2] # 오늘의 이동평균선 방향
#
#     if ma_angle > 0:
#         # 상승 중인 종목만 예측/추천
#         pass
#     else:
#         # 하락/횡보면 건너뜀
#         # print(f"                                                        이동평균선이 상승이 아니므로 건너뜁니다.")
#         continue



#     # 이동평균선이 하락중이면 제외
#     ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
#     ma_angle_15 = data['MA15'].iloc[-1] - data['MA15'].iloc[-2]
#     ma_angle_20 = data['MA20'].iloc[-1] - data['MA20'].iloc[-2]
#
#     ma_cnt = 0
#     if ma_angle_5 > 0:
#         ma_cnt = ma_cnt + 1
#     if ma_angle_15 > 0:
#         ma_cnt = ma_cnt + 1
#     if ma_angle_20 > 0:
#         ma_cnt = ma_cnt + 1
#     if ma_cnt >= 2:
#         # 상승 중인 종목만 예측/추천
#         pass
#     else:
#         # 하락/횡보면 건너뜀
#         # print(f"                                                        이동평균선이 상승이 아니므로 건너뜁니다.")
#         continue

#     # (xx거래일 전)의 20일선과 현재 20일선 비교
#     if data['MA20'].iloc[-1] < data['MA20'].iloc[-20]:
#         # print(f"                                                        2달 전보다 20일선이 하락해 있으면 건너뜁니다.")
#         continue
#
#     if data['MA5'].iloc[-1] < data['MA20'].iloc[-1]:
#         # print(f"                                                        5일선이 20일선 보다 낮을 경우 : 제외")
#         # continue  # 조건에 맞지 않으면 건너뜀
#         pass

    ########################################################################

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data.values)
    feature_cols = [
        'Close', 'High', 'PBR', 'Low', 'Volume', 'RSI14', 'ma10_gap',
    ]

    X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
#     X_for_model = X_for_model.apply(pd.to_numeric, errors='coerce') # float64여야 함 (object라면 먼저 float 변환 필요)
#     X_for_model = X_for_model.replace([np.inf, -np.inf], 0) # # inf/-inf 값을 0으로(또는 np.nan으로) 대체
    scaled_data = scaler.fit_transform(X_for_model)
    # 슬라이딩 윈도우로 전체 데이터셋 생성
    X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD, 0)
    if count == 0:
        print("X.shape:", X.shape) # X.shape: (0,) 데이터가 부족해서 슬라이딩 윈도우로 샘플이 만들어지지 않음
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
        print(f"                                                        R-squared 0.7 미만이면 패스 : {r2:.2f}%")
        continue


    # X_input 생성 (마지막 구간)
    X_input = X[-1:]
    future_preds = model.predict(X_input, verbose=0).flatten() # 1차원 벡터로 변환
    # print('future', future_preds)

    # 종가 scaler fit (실제 데이터로)
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = data['Close'].values.reshape(-1, 1)
    close_scaler.fit(close_prices)

    # 모델 예측값(future_preds)은 정규화된 값임 >> scaler로 실제 가격 단위(원래 스케일)로 되돌림 (역정규화)
    predicted_prices = close_scaler.inverse_transform(future_preds.reshape(-1, 1)).flatten() # (PREDICTION_PERIOD, )

    # 날짜 처리
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=PREDICTION_PERIOD, freq='B')
    avg_future_return = (np.mean(predicted_prices) / last_close - 1) * 100

    # 기대 성장률 미만이면 건너뜀
    if avg_future_return < EXPECTED_GROWTH_RATE:
        print(f"예상 : {avg_future_return:.2f}%")
        # pass
        continue

    # 결과 저장
    results.append((avg_future_return, ticker))

    # 기존 파일 삭제
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        if os.path.isdir(file_path):
            continue
        if file_name.startswith(f"{today}") and ticker in file_name:
            print(f"Deleting existing file: {file_name}")
            send2trash(os.path.join(output_dir, file_name))




    # 1. 조건별 색상 결정 (거래량 바 차트)
    up = data['Close'] > data['Open']
    down = data['Close'] < data['Open']
    bar_colors = np.where(up, 'red', np.where(down, 'blue', 'gray'))

    # 2. 인덱스 문자열 컬럼 추가 (x축 통일용)
    data_plot = data.copy()
    data_plot['date_str'] = data_plot.index.strftime('%Y-%m-%d')

    # 3. 미래 날짜도 문자열로 변환
    future_dates_str = pd.to_datetime(future_dates).strftime('%Y-%m-%d')

    # 4. 그래프 (윗부분: 가격/지표, 아랫부분: 거래량)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- 상단: 가격 + 볼린저밴드 + 예측 ---
    # 실제 가격
    ax1.plot(data_plot['date_str'], actual_prices, label='실제 가격', marker='s', markersize=6, markeredgecolor='white')

    # 예측 가격 (미래 날짜)
    ax1.plot(future_dates_str, predicted_prices, label='예측 가격', linestyle='--', marker='s', markersize=7, markeredgecolor='white', color='tomato')

    # 이동평균, 볼린저밴드, 영역 채우기
    if all(x in data_plot.columns for x in ['MA20', 'UpperBand', 'LowerBand']):
        ax1.plot(data_plot['date_str'], data_plot['MA20'], label='20일 이동평균선', alpha=0.8)
        if 'MA5' in data_plot.columns:
            ax1.plot(data_plot['date_str'], data_plot['MA5'], label='5일 이동평균선', alpha=0.8)
        ax1.plot(data_plot['date_str'], data_plot['UpperBand'], label='볼린저밴드 상한선', linestyle='--', alpha=0.8)
        ax1.plot(data_plot['date_str'], data_plot['LowerBand'], label='볼린저밴드 하한선', linestyle='--', alpha=0.8)
        ax1.fill_between(data_plot['date_str'], data_plot['UpperBand'], data_plot['LowerBand'], color='gray', alpha=0.18)

    # 마지막 실제값과 첫 번째 예측값을 점선으로 연결
    ax1.plot(
        [data_plot['date_str'].iloc[-1], future_dates_str[0]],
        [actual_prices[-1], predicted_prices[0]],
        linestyle='dashed', color='gray', linewidth=1.5
    )

    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'{end_date}  {ticker} (Expected Return: {avg_future_return:.2f}%)')

    # --- 하단: 거래량 (양/음/동색 구분) ---
    ax2.bar(data_plot['date_str'], data_plot['Volume'], color=bar_colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.grid(True)

    # x축 라벨: 10일 단위만 표시 (과도한 라벨 겹침 방지)
    tick_idx = np.arange(0, len(data_plot), 10)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(data_plot['date_str'].iloc[tick_idx])

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 파일 저장 (옵션)
    final_file_name = f'{today} [ {avg_future_return:.2f}% ] {ticker}.png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()
    # plt.show()
    
    # plt.figure(figsize=(16, 8))
    # if rsi_flag:
    #     plt.subplot(2,1,1)
    # # 실제 데이터
    # plt.plot(data.index, actual_prices, label='실제 가격')
    # # 예측 데이터
    # plt.plot(future_dates, predicted_prices, label='예측 가격', linestyle='--', marker='o', color='tomato')
    # 
    # if all(x in data.columns for x in ['MA20', 'UpperBand', 'LowerBand']):
    #     plt.plot(data.index, data['MA20'], label='20일 이동평균선') # MA20
    #     plt.plot(data.index, data['MA5'], label='5일 이동평균선')
    #     plt.plot(data.index, data['UpperBand'], label='볼린저밴드 상한선', linestyle='--') # Upper Band (2σ)
    #     plt.plot(data.index, data['LowerBand'], label='볼린저밴드 하한선', linestyle='--') # Lower Band (2σ)
    #     plt.fill_between(data.index, data['UpperBand'], data['LowerBand'], color='gray', alpha=0.2)
    # 
    # # 마지막 실제값과 첫 번째 예측값을 점선으로 연결
    # plt.plot(
    #     [data.index[-1], future_dates[0]],  # x축: 마지막 실제날짜와 첫 예측날짜
    #     [actual_prices[-1], predicted_prices[0]],  # y축: 마지막 실제종가와 첫 예측종가
    #     linestyle='dashed', color='gray', linewidth=1.5
    # )
    # 
    # plt.title(f'{end_date}  {ticker} (Expected Return: {avg_future_return:.2f}%)')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(rotation=45)
    # 
    # if rsi_flag:
    #     data['RSI'] = compute_rsi(data['Close'])
    # 
    #     # RSI 차트 (하단)
    #     plt.subplot(2,1,2)
    #     plt.plot(data['RSI'], label='RSI(14)', color='purple')
    #     plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    #     plt.axhline(30, color='blue', linestyle='--', label='Oversold (30)')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.grid(True)
    # 
    # final_file_name = f'{today} [ {avg_future_return:.2f}% ] {ticker}.png'
    # final_file_path = os.path.join(output_dir, final_file_name)
    # plt.savefig(final_file_path)
    # plt.close()

####################################

# 정렬 및 출력
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, ticker in results:
    print(f"==== [ {avg_future_return:.2f}% ] [{ticker}] ====")

'''
Series
1차원 데이터

import pandas as pd
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a    10
# b    20
# c    30
# dtype: int64

특징
  1차원 벡터(배열) + 인덱스
  넘파이 배열에 “이름(인덱스)”이 붙은 것



DataFrame
2차원 데이터 (엑셀 표와 유사)

import pandas as pd
df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': [10, 20, 30]
}, index=['a', 'b', 'c'])
print(df)
#    col1  col2
# a     1    10
# b     2    20
# c     3    30

각 열이 Series임 (즉, df['col1']은 Series)
'''