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
import requests

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


output_dir = 'D:\\sp500'
os.makedirs(output_dir, exist_ok=True)

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle_us')

# pickle 폴더가 없으면 자동 생성 (이미 있으면 무시)
os.makedirs(pickle_dir, exist_ok=True)



PREDICTION_PERIOD = 3
EXPECTED_GROWTH_RATE = 3
DATA_COLLECTION_PERIOD = 400
LOOK_BACK = 15
KR_AVERAGE_TRADING_VALUE = 5_000_000_000

exchangeRate = get_usd_krw_rate()
if exchangeRate is None:
    print('#######################   exchangeRate is None   #######################')
else:
    print(f'#######################   exchangeRate is {exchangeRate}   #######################')

# 미국 동부 시간대 설정
now_us = datetime.now(pytz.timezone('America/New_York'))
# 현재 시간 출력
print("미국 동부 시간 기준 현재 시각:", now_us.strftime('%Y-%m-%d %H:%M:%S'))
# 데이터 수집 시작일 계산
start_date_us = (now_us - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')
start_five_date_us = (now_us - timedelta(days=5)).strftime('%Y-%m-%d')
print("미국 동부 시간 기준 데이터 수집 시작일:", start_date_us)

end_date = datetime.today().strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y%m%d')


# tickers = extract_stock_code_from_filenames(output_dir)
tickers = get_nasdaq_symbols()
# tickers=['RKLB']


# 결과를 저장할 배열
results = []
total_r2 = 0
total_cnt = 0
is_first_flag = True

for count, ticker in enumerate(tickers):
    print(f"Processing {count+1}/{len(tickers)} : {ticker}")


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data_us(ticker, start_five_date_us, end_date)
    else:
        df = pd.DataFrame()
        data = fetch_stock_data_us(ticker, start_date_us, end_date)

    # 중복 제거 & 새로운 날짜만 추가
    if not df.empty:
        # 기존 날짜 인덱스와 비교하여 새로운 행만 선택
        new_rows = data.loc[~data.index.isin(df.index)] # ~ (not) : 기존에 없는 날짜만 남김
        df = pd.concat([df, new_rows])
    else:
        df = data

    # 너무 먼 과거 데이터 버리기, 처음 272개
    if len(df) > 280:
        df = df.iloc[-280:]

    # 파일 저장
    df.to_pickle(filepath)
    # data = pd.read_pickle(filepath)
    data = df


    if data is None or 'Close' not in data.columns or data.empty:
        print(f"{ticker}: 데이터가 비었거나 'Close' 컬럼이 없습니다. pass.")
        continue

#     check_column_types(fetch_stock_data_us(ticker, start_date_us, end_date), ['Close', 'Open', 'High', 'Low', 'Volume', 'PER', 'PBR']) # 타입과 shape 확인 > Series 가 나와야 한다
#     continue

    ########################################################################

    actual_prices = data['Close'].values # 최근 종가 배열
    last_close = actual_prices[-1]

    if data.empty or len(data) < 50:
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
    if (recent_trading_value * exchangeRate < 400_000_000).any():
        # print(f"                                                        최근 2주 중 거래대금 4억 미만 발생 → 제외")
        continue

    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value * exchangeRate <= KR_AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{(recent_average_trading_value * exchangeRate)/ 100_000_000:.0f}억"
        print(f"                                                        최근 2주 평균 거래액({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # rolling window로 5일 전 대비 현재가 3배 이상 오른 지점 찾기
    rolling_min = data['Close'].rolling(window=5).min()    # 5일 중 최소가
    ratio = data['Close'] / rolling_min

    if np.any(ratio >= 2.0):
        print(f"                                                        어느 5일 구간이든 2배 급등: 제외")
        continue


    # # 최고가 대비 현재가 하락률 계산
    # max_close = np.max(actual_prices)
    # drop_pct = ((max_close - last_close) / max_close) * 100
    #
    # # 40% 이상 하락한 경우 건너뜀
    # if drop_pct >= 50:
    #     continue

    # # 모든 4일 연속 구간에서 첫날 대비 마지막날 xx% 이상 급등
    # window_start = actual_prices[:-3]   # 0 ~ N-4
    # window_end = actual_prices[3:]      # 3 ~ N-1
    # ratio = window_end / window_start   # numpy, pandas Series/DataFrame만 벡터화 연산 지원, ratio는 결과 리스트
    #
    # if np.any(ratio >= 1.6):
    #     print(f"                                                        어떤 4일 연속 구간에서 첫날 대비 60% 이상 상승: 제외")
    #     continue
    #
    # last_close = data['Close'].iloc[-1]
    # close_4days_ago = data['Close'].iloc[-5]
    #
    # rate = (last_close / close_4days_ago - 1) * 100
    #
    # if rate <= -18:
    #     print(f"                                                        4일 전 대비 {rate:.2f}% 하락 → 학습 제외")
    #     continue  # 또는 return


    # # 최근 3일, 2달 평균 거래량 계산, 최근 3일 거래량이 최근 2달 거래량의 25% 안되면 패스
    # recent_3_avg = data['Volume'][-3:].mean()
    # recent_2months_avg = data['Volume'][-40:].mean()
    # if recent_3_avg < recent_2months_avg * 0.15:
    #     temp = (recent_3_avg/recent_2months_avg * 100)
    #     # print(f"                                                        최근 3일의 평균거래량이 최근 2달 평균거래량의 25% 미만 → pass : {temp:.2f} %")
    #     # continue
    #     pass

    # 2차 생성 feature
    data = add_technical_features_us(data)

    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
    if data['MA5'].iloc[-1] < data['MA20'].iloc[-1] and ma_angle_5 < 0:
        # print(f"                                                        5일선이 20일선 보다 낮을 경우 → pass")
        continue
        # pass

    # 5일선이 너무 하락하면
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    change_rate = (ma5_today - ma5_yesterday) / ma5_yesterday
    if change_rate * 100 < -4:
        # print(f"어제 5일선의 변화율: {change_rate:.5f}")  # 소수점 5자리
        print(f"                                                        어제 5일선의 변화율: {change_rate * 100:.2f}% → pass")
        continue

    ########################################################################

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data.values)
    feature_cols = [
        'Close', 'High', 'PBR', 'Low', 'Volume',
        # 'RSI14',
        # 'ma10_gap',
    ]

    X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
#     X_for_model = X_for_model.apply(pd.to_numeric, errors='coerce') # float64여야 함 (object라면 먼저 float 변환 필요)
#     X_for_model = X_for_model.replace([np.inf, -np.inf], 0) # # inf/-inf 값을 0으로(또는 np.nan으로) 대체
    scaled_data = scaler.fit_transform(X_for_model)
    # 슬라이딩 윈도우로 전체 데이터셋 생성
    X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD, 0)
    if is_first_flag:
        is_first_flag = False
        print("X.shape:", X.shape) # X.shape: (0,) 데이터가 부족해서 슬라이딩 윈도우로 샘플이 만들어지지 않음
    # print("Y.shape:", Y.shape)

    if X.shape[0] < 50:
        print("                                                        샘플 부족 : ", X.shape[0])
        continue

    # 학습하기 직전에 요청을 보낸다
    percent = f'{round((count+1)/len(tickers)*100, 1):.1f}'
    try:
        requests.post(
            'http://localhost:8090/func/stocks/progress-update/nasdaq',
            json={
                "percent": percent,
                "count": count+1,
                "total_count": len(tickers),
                "ticker": ticker,
                "stock_name": "",
                "done": False,
            },
            timeout=5
        )
    except Exception as e:
        # logging.warning(f"progress-update 요청 실패: {e}")
        print(f"progress-update 요청 실패: {e}")
        pass  # 오류

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
    if r2 > 0:
        total_r2 += r2
        total_cnt += 1
    if r2 < 0.5:
        # print(f"                                                        R-squared 0.7 미만이면 패스 : {r2:.2f}%")
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
        if avg_future_return > 0:
            print(f"  예상 : {avg_future_return:.2f}%")
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
    # 인덱스를 명시적으로 DatetimeIndex로 변환
    data_plot.index = pd.to_datetime(data_plot.index)
    data_plot['date_str'] = data_plot.index.strftime('%Y-%m-%d')

    # data_plot.index가 DatetimeIndex라고 가정
    three_months_ago = data_plot.index.max() - pd.DateOffset(months=6)
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
    ax1.set_title(f'{end_date}  {ticker} (Expected Return: {avg_future_return:.2f}%)')

    # --- 하단: 거래량 (양/음/동색 구분) ---
    ax2.bar(data_plot_recent['date_str'], data_plot_recent['Volume'], color=bar_colors, alpha=0.65)
    ax2.set_ylabel('Volume')
    ax2.grid(True)

    # x축 라벨: 10일 단위만 표시 (과도한 라벨 겹침 방지)
    tick_idx = np.arange(0, len(data_plot_recent), 10)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels(data_plot_recent['date_str'].iloc[tick_idx])

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 파일 저장 (옵션)
    final_file_name = f'{today} [ {avg_future_return:.2f}% ] {ticker}.png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()

####################################

# 정렬 및 출력
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, ticker in results:
    print(f"==== [ {avg_future_return:.2f}% ] [{ticker}] ====")

try:
    requests.post(
        'http://localhost:8090/func/stocks/progress-update/nasdaq',
        json={"percent": 100, "done": True},
        timeout=5
    )
except Exception as e:
    # logging.warning(f"progress-update 요청 실패: {e}")
    print(f"progress-update 요청 실패: {e}")
    pass  # 오류

print('result_r2 : ', total_r2/total_cnt)
print('total_cnt : ', total_cnt)


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