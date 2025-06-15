import matplotlib # tkinter 충돌 방지, Agg 백엔드를 사용하여 GUI를 사용하지 않도록 한다
matplotlib.use('Agg')
import os
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
from send2trash import send2trash


from utils import create_model, create_multistep_dataset, get_safe_ticker_list, fetch_stock_data_us, get_nasdaq_symbols

output_dir = 'D:\\kospi_stocks'
os.makedirs(output_dir, exist_ok=True)

PREDICTION_PERIOD = 5
EXPECTED_GROWTH_RATE = 6
DATA_COLLECTION_PERIOD = 120
LOOK_BACK = 20
AVERAGE_VOLUME = 50000
AVERAGE_TRADING_VALUE = 2000000 # 28억 쯤
window = 20  # 이동평균 구간
num_std = 2  # 표준편차 배수

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
# tickers = tickers[515:]
tickers = ['CEP', 'CLPT', 'RKLB', 'PUBM', 'CBRL', 'NPCE', 'EVLV', 'NEXN', 'MBOT', 'JAMF', 'LQDA', 'SFIX', 'XMTR', 'RGC', 'TTAN', 'MNKD', 'INM', 'CERT', 'RR', 'REAL', 'LX', 'SDGR', 'UPXI', 'PPTA', 'PRTH', 'ETON', 'TOI', 'LULU', 'LOVE', 'WOOF', 'SATL', 'GRYP', 'GIII', 'CGNT', 'LTBR', 'VEON', 'BCAX', 'UTHR', 'CRWV', 'TEAM', 'SKWD', 'APLD', 'PCTY', 'DXPE', 'HLMN', 'VERX', 'MRX', 'SKYW', 'BRZE', 'ADTN', 'HNST', 'PDD', 'HNRG', 'BIGC', 'DLO', 'NIU', 'AAON', 'AMSC', 'PERI', 'MNDY', 'ATEC', 'ALKT', 'APP', 'RELY', 'TALK', 'NTGR', 'LFST', 'SNDX', 'IIIV', 'NUTX', 'UAL', 'BLFS', 'BTSG']

# tickers=['RKLB']


# 결과를 저장할 배열
results = []

for count, ticker in enumerate(tickers):
    print(f"Processing {count+1}/{len(tickers)} : {ticker}")

    data = fetch_stock_data_us(ticker, start_date_us, end_date)

    if data.empty:
        print(f"{ticker}: 데이터 없음, 건너뜀")
        continue

    #컬럼명을 명시적으로 지정해서 1차원 컬럼으로 변환
#     data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'PER', 'PBR']
    # 컬럼이 멀티인덱스거나 중복명이라면
    if isinstance(data.columns, pd.MultiIndex) or (len(data.columns) == 7 and not set(data.columns) == set(['Open', 'High', 'Low', 'Close', 'Volume', 'PER', 'PBR'])):
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'PER', 'PBR']

    # 볼린저밴드
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['STD20'] = data['Close'].rolling(window=window).std()
    data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
    data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])

    # 데이터셋 생성
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data.values)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'UpperBand', 'LowerBand', 'PER', 'PBR']

    X_for_model = data[feature_cols].fillna(0) # 모델 feature만 NaN을 0으로
    X_for_model = X_for_model.apply(pd.to_numeric, errors='coerce') # float64여야 함 (object라면 먼저 float 변환 필요)
    X_for_model = X_for_model.replace([np.inf, -np.inf], 0) # # inf/-inf 값을 0으로(또는 np.nan으로) 대체

    scaled_data = scaler.fit_transform(X_for_model)
    X, Y = create_multistep_dataset(scaled_data, LOOK_BACK, PREDICTION_PERIOD)
    # print("X.shape:", X.shape) # X.shape: (0,) 데이터가 부족해서 슬라이딩 윈도우로 샘플이 만들어지지 않음
    # print("Y.shape:", Y.shape)

    ########################################################################

    # 데이터가 부족하면 건너뜀
    if data.empty or len(data) < LOOK_BACK:
        # print(f"                                                        데이터가 부족하여 작업을 건너뜁니다")
        continue

    if len(X) < 2 or len(Y) < 2:
        print(f"                                                        데이터셋이 부족하여 작업을 건너뜁니다 (신규 상장).")
        continue

    # 종가가 0.0이거나 500원 미만이면 건너뜀
    last_row = data.iloc[-1]
    if last_row['Close'] == 0.0 or last_row['Close'] < 0.3:
        # print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
        continue

    # 일일 평균 거래량/거래대금 체크
    average_volume = data['Volume'].mean()
    if average_volume <= AVERAGE_VOLUME:
        print(f"                                                        평균 거래량({average_volume:.0f}주)이 부족하여 작업을 건너뜁니다.")
        continue

    trading_value = data['Volume'] * data['Close']
    average_trading_value = trading_value.mean()
    if average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_value = f"{average_trading_value / 140000:.0f}억"
        print(f"                                                        평균 거래액({formatted_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # 최근 한 달 거래액 체크
    recent_data = data.tail(20)
    recent_trading_value = recent_data['Volume'] * recent_data['Close']
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{recent_average_trading_value / 140000:.0f}억"
        print(f"                                                        최근 한 달 평균 거래액({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
        continue

    actual_prices = data['Close'].values # 최근 종가 배열
    last_close = actual_prices[-1]

    # 최고가 대비 현재가 하락률 계산
    max_close = np.max(actual_prices)
    drop_pct = ((max_close - last_close) / max_close) * 100

    # 40% 이상 하락한 경우 건너뜀
    if drop_pct >= 40:
        continue

    ########################################################################

    # 현재가
    last_close = data['Close'].iloc[-1]
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
    # data['MA_10'] = data['Close'].rolling(window=10).mean()
    # ma_angle = data['MA_10'].iloc[-1] - data['MA_10'].iloc[-2] # 오늘의 이동평균선 방향
    # ma_angle = data['MA20'].iloc[-1] - data['MA20'].iloc[-2] # 오늘의 이동평균선 방향
    #
    # if ma_angle > 0:
    #     # 상승 중인 종목만 예측/추천
    #     pass
    # else:
    #     # 하락/횡보면 건너뜀
    #     print(f"                                                        이동평균선이 상승이 아니므로 건너뜁니다.") # 20일선
    #     continue

    ########################################################################

    # 모델 생성 및 학습
    model = create_model((X.shape[1], X.shape[2]), PREDICTION_PERIOD)

    # 콜백 설정
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X, Y, batch_size=8, epochs=200, validation_split=0.1, shuffle=False, verbose=0, callbacks=[early_stop])

    # 종가 scaler fit (실제 데이터로)
    close_scaler = MinMaxScaler()
    close_prices = data['Close'].values.reshape(-1, 1)
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
    results.append((avg_future_return, ticker))

    # 기존 파일 삭제
    for file_name in os.listdir(output_dir):
        if file_name.startswith(f"{today}") and ticker in file_name:
            print(f"Deleting existing file: {file_name}")
            os.remove(os.path.join(output_dir, file_name))

    # 그래프 저장
    extended_prices = np.concatenate((data['Close'].values, predicted_prices))
    last_price = data['Close'].iloc[-1]

    plt.figure(figsize=(16, 8))
    # 실제 데이터
    plt.plot(data.index, actual_prices, label='실제 가격')
    # 예측 데이터
    plt.plot(future_dates, predicted_prices, label='예측 가격', linestyle='--', marker='o', color='tomato')

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

    plt.title(f'{end_date}  {ticker} (Expected Return: {avg_future_return:.2f}%)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    final_file_name = f'{today} [ {avg_future_return:.2f}% ] {ticker}.png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()

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