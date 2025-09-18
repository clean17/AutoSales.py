import matplotlib # tkinter 충돌 방지, Agg 백엔드를 사용하여 GUI를 사용하지 않도록 한다
matplotlib.use('Agg')
import os, sys
import pytz
import pandas as pd
from pykrx import stock
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from utils import create_lstm_model, create_multistep_dataset, fetch_stock_data_us, add_technical_features, \
    plot_candles_daily, plot_candles_weekly, drop_trading_halt_rows, regression_metrics, pass_filter, \
    get_usd_krw_rate, get_nasdaq_symbols, get_name_from_usa_ticker
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import time

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
KR_AVERAGE_TRADING_VALUE = 6_000_000_000

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
# tickers = ['MNKD']
tickers = tickers[:2]

def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en

# 결과를 저장할 배열
results = []
total_r2 = 0
total_cnt = 0
total_smape = 0

# print('==================================================')
# print("    RMSE, MAE → 예측 오차를 주가 단위(원) 그대로 해석 가능")
# print("    MAPE (%) → 평균적으로 몇 % 오차가 나는지 직관적")
# print("      0~10%: 매우 우수, 10~20%: 양호, 20~50%: 보통, 50% 이상: 부정확")
# print("    SMAPE (%) → SMAPE는 0% ~ 200% 범위를 가지지만, 보통은 0~100% 사이에서 해석")
# print("      0 ~ 10% → 매우 우수, 10 ~ 20% → 양호 (실사용 가능한 수준), 20 ~ 50% → 보통, 50% 이상 → 부정확")
# print("    R² → 모델 설명력, 0~1 범위에서 클수록 좋음")
# print("      0.6: 변동성의 약 60%를 설명")
# print('==================================================')

# 데이터 가져오는것만 1시간 걸리네
for count, ticker in enumerate(tickers):
    stock_name = get_name_from_usa_ticker(ticker)
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 학습하기 직전에 요청을 보낸다
    percent = f'{round((count+1)/len(tickers)*100, 1):.1f}'
    try:
        requests.post(
            'https://chickchick.shop/func/stocks/progress-update/nasdaq',
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


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data_us(ticker, start_five_date_us, end_date)
    else:
        df = pd.DataFrame()
        data = fetch_stock_data_us(ticker, start_date_us, end_date)

    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음
    else:
        df = data.copy()

    # 너무 먼 과거 데이터 버리기, 처음 272개
    if len(df) > 280:
        df = df.iloc[-280:]

    # 파일 저장
    df.to_pickle(filepath)
    # data = pd.read_pickle(filepath)
    data = df


    if data is None or 'Close' not in data.columns or data.empty:
        # print(f"{ticker}: 데이터가 비었거나 'Close' 컬럼이 없습니다. pass.")
        continue

    #     check_column_types(fetch_stock_data_us(ticker, start_date_us, end_date), ['Close', 'Open', 'High', 'Low', 'Volume', 'PBR']) # 타입과 shape 확인 > Series 가 나와야 한다
    #     continue

    ########################################################################

    actual_prices = data['Close'].values # 최근 종가 배열
    last_close = actual_prices[-1]

    # 0) 우선 거래정지/이상치 행 제거
    data, removed_idx = drop_trading_halt_rows(data)
    if len(removed_idx) > 0:
        # print(f"                                                        거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")
        pass

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 50:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 종가가 0.0이거나 500원 미만이면 건너뜀
    last_row = data.iloc[-1]
    if last_row['Close'] == 0.0 or last_row['Close'] * exchangeRate < 500:
        # print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
        continue

    # 한달 데이터
    month_data = data.tail(20)
    month_trading_value = month_data['Volume'] * month_data['Close']
    # 하루라도 거래대금이 5억 미만이 있으면 제외
    if (month_trading_value * exchangeRate < 500_000_000).any():
        # print(f"                                                        최근 4주 중 거래대금 5억 미만 발생 → pass")
        continue

    # 최근 2주 평균 거래대금 60억 미만 패스
    recent_data = data.tail(10)
    recent_trading_value = recent_data['Volume'] * recent_data['Close']     # 최근 2주 거래대금 리스트
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value * exchangeRate <= KR_AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{(recent_average_trading_value * exchangeRate)/ 100_000_000:.0f}억"
        print(f"                                                        최근 2주 평균 거래액({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # 투경 조건
    # 1. 당일의 종가가 3일 전날의 종가보다 100% 이상 상승
    if len(actual_prices) >= 4:
        close_3ago = actual_prices[-4]
        ratio = (last_close - close_3ago) / close_3ago * 100
        if last_close >= close_3ago * 2:  # 100% 이상 상승
            print(f"                                                        3일 전 대비 100% 이상 상승: {close_3ago} -> {last_close}  {ratio:.2f}% → pass")
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
    data = add_technical_features(data)

    threshold = 0.1  # 10%
    # isna() : pandas의 결측값(NA) 체크. NaN, None, NaT에 대해 True
    # mean() : 평균
    # isinf() : 무한대 체크
    cols_to_drop = [ # 결측치가 10% 이상인 칼럼
        col
        for col in data.columns
        if (~np.isfinite(pd.to_numeric(data[col], errors='coerce'))).mean() > threshold
    ]
    if len(cols_to_drop) > 0:
        # inplace=True : 반환 없이 입력을 그대로 수정
        # errors='ignore' : 목록에 없는 칼럼 지우면 에러지만 무시
        data.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        print("Drop candidates:", cols_to_drop)

    if 'MA20' not in data.columns:
        continue

    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    ma_angle_5 = data['MA5'].iloc[-1] - data['MA5'].iloc[-2]
    if data['MA5'].iloc[-1] < data['MA20'].iloc[-1] and ma_angle_5 < 0:
        # print(f"                                                        5일선이 20일선 보다 낮을 경우 → pass")
        continue
        # pass

    # 5일선이 너무 하락하면
    # ma5_today = data['MA5'].iloc[-1]
    # ma5_yesterday = data['MA5'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    # change_rate = (ma5_today - ma5_yesterday) / ma5_yesterday
    # if change_rate * 100 < -4:
    #     # print(f"어제 5일선의 변화율: {change_rate:.5f}")  # 소수점 5자리
    #     print(f"                                                        어제 5일선의 변화율: {change_rate * 100:.2f}% → pass")
    #     continue

    ########################################################################


    # 한국/영문 칼럼 자동 식별
    col_o = _col(df, '시가',   'Open')
    col_h = _col(df, '고가',   'High')
    col_l = _col(df, '저가',   'Low')
    col_c = _col(df, '종가',   'Close')
    col_v = _col(df, '거래량', 'Volume')

    # 학습에 쓸 피처
    feature_cols = [
        col_o, col_l, col_h, col_c,
        # col_v,
        'Vol_logdiff',
        # 'ma10_gap',
        'MA5_slope',
        # ** col_o, col_l, col_h, col_c, Vol_logdiff, MA5_slope
        # R-squared_avg :  0.5841392158480752
        # SMAPE :  42.22468885035757
    ]

    # 0) NaN/inf 정리
    data = data.replace([np.inf, -np.inf], np.nan)

    # 1) feature_cols만 남기고 dropna
    feature_cols = [c for c in feature_cols if c in data.columns]
    # print('feature_cols', feature_cols)
    X_df = data.dropna(subset=feature_cols).loc[:, feature_cols]


    # 종가 컬럼 이름/인덱스
    idx_close = feature_cols.index(col_c)

    # 2) 시계열 분리 후, train만 fit → val/전체 transform
    split = int(len(X_df) * 0.9)

    # (선택) 상수열 제거 (칼럼의 값이 동일한 경우)
    # const_cols = [c for c in X_df.columns if X_df[c].nunique() <= 1]
    # if const_cols:
    #     X_df = X_df.drop(columns=const_cols)
    #     feature_cols = [c for c in feature_cols if c not in const_cols]

    # 거의 상수 (99%) 제거
    # qconst_cols = []
    # for c in X_df.columns:
    #     top_ratio = X_df.iloc[:split][c].value_counts(normalize=True, dropna=False).iloc[0]
    #     if top_ratio >= 0.99:  # 99%가 같은 값
    #         qconst_cols.append(c)
    # if qconst_cols:
    #     X_df = X_df.drop(columns=const_cols)
    #     feature_cols = [c for c in feature_cols if c not in const_cols]

    # 필수 칼럼 빠지지 않게 가드
    # if col_c not in X_df.columns:
    #     raise ValueError(f"필수 컬럼 {col_c} 이(가) 상수로 판단되어 제거됨. 전처리 확인 필요")

    """
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_tr_2d = scaler.fit_transform(X_df.iloc[:split].values)   # fit은 train에만
        X_va_2d = scaler.transform(X_df.iloc[split:].values)
        X_all_2d = scaler.transform(X_df.values)                   # 전체 transform (누수 아님)
    """
    scaler = StandardScaler()                         # ← 변경: StandardScaler
    X_tr_2d = scaler.fit_transform(X_df.iloc[:split].values)   # fit: 스케일러가 데이터의 통계값을 학습         >> 훈련 세트: 훈련 데이터로만 통계(평균·최대/최소 등)를 학습 + 변환
    X_va_2d = scaler.transform(X_df.iloc[split:].values)       # transform: 이미 학습된 통계값을 써서 값을 변환 >> 검증/테스트 세트: 훈련에서 배운 통계로만 변환
    X_all_2d = scaler.transform(X_df.values)                   # 전체 transform (누수 아님)

    # 3) 윈도잉(블록별) → 학습/검증 세트
    X_train, Y_train = create_multistep_dataset(X_tr_2d, LOOK_BACK, PREDICTION_PERIOD, idx=idx_close)
    X_val,   Y_val   = create_multistep_dataset(X_va_2d, LOOK_BACK, PREDICTION_PERIOD, idx=idx_close)

    # 슬라이싱 정합성 체크, 실패하면 인덱스가 내림차순
    # X_val의 마지막 시점 종가 == 다음 샘플의 Y_val 첫 스텝 ?
    # (스케일된 값 기준)
    # 표준화/부동소수 오차 때문에 “완전 동일”이 아닐 수 있어 atol=1e-8 허용
    assert np.allclose(X_val[1:, -1, idx_close], Y_val[:-1, 0], atol=1e-8)


    # 4) 안전 체크, NaN이나 ±Inf가 섞여 있지 않은지를 한 번에 검사하는 정합성 체크
    for name, arr in [('X_train',X_train),('Y_train',Y_train),('X_val',X_val),('Y_val',Y_val)]:
        assert np.isfinite(arr).all(), f"{name} has NaN/inf: check preprocessing"

    # 5) 최소 샘플 수 확인
    if X_train.shape[0] < 50:
        continue

    #######################################################################

    # 6) 모델 생성/학습
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]), PREDICTION_PERIOD,
                              lstm_units=[128,64], dense_units=[64,32])

    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, Y_train, batch_size=8, epochs=200,
                        validation_data=(X_val, Y_val),
                        shuffle=False, verbose=0, callbacks=[early_stop])


    # 모델 평가
    # val_loss = model.evaluate(X_val, Y_val, verbose=1)
    # print("Validation Loss :", val_loss)

    # (선택) 검증 예측
    """
        X_val : 검증에 쓸 여러 시계열 구간의 집합
        preds : 검증셋 각 구간(윈도우)에 대해 미래 PREDICTION_PERIOD만큼의 예측치 반환
        shape: (검증샘플수, 예측일수)
    """
    preds = model.predict(X_val, verbose=0)

    #######################################################################

    from sklearn.metrics import r2_score
    # 모델 R²
    r2_model = r2_score(Y_val.ravel(), preds.ravel())

    for h in range(preds.shape[1]):
        print(f"h={h}, R2={r2_score(Y_val[:,h], preds[:,h])}")


    # 퍼시스턴스(naive) 베이스라인: 마지막 관측값을 모든 미래로 반복
    last_close = X_val[:, -1, idx_close]
    yhat_persist = np.tile(last_close[:, None], (1, preds.shape[1]))
    r2_persist = r2_score(Y_val.ravel(), yhat_persist.ravel())

    print("R2(model) =", r2_model, " / R2(persist) =", r2_persist)

    continue

    # ++ 지표 계산: scaled + restored(원 단위)
    metrics = regression_metrics(
        Y_val, preds,
        scaler=scaler,
        n_features=X_df.shape[1],
        idx_close=idx_close
    )

    # print("=== SCALED (표준화 공간) ===")
    for k,v in metrics["scaled"].items():
        if k == 'SMAPE (%)':
            total_smape += v
        if k == 'R2': # R-squared, (0=엉망, 1=완벽)
            # print(f"                                                        R-squared 0.6 미만이면 패스 : {r2}")
            total_r2 += v
        # print(f"    {k}: {v:.4f}")

    # if "restored" in metrics:
    #     print("\n=== RESTORED (원 단위) ===")
    #     for k,v in metrics["restored"].items():
    #         print(f"    {k}: {v:.4f}")

    ok = pass_filter(metrics, use_restored=False, r2_min=0.55, smape_max=35.0)  # 원 단위 기준으로 필터
    # print("                                                        PASS 1st filter?" , ok)
    if not ok:
        continue

    #######################################################################

    # 7) 마지막 윈도우 1개로 미래 H-step 예측
    if len(X_df) < LOOK_BACK:
        raise ValueError("LOOK_BACK보다 데이터가 짧습니다.")
    n_features = X_all_2d.shape[1]
    last_window = X_all_2d[-LOOK_BACK:].reshape(1, LOOK_BACK, n_features)
    future_scaled = model.predict(last_window, verbose=0)[0]        # shape=(H,)

    # 8) 스케일 역변환 (StandardScaler: x = z*scale_[idx] + mean_[idx])
    mu  = scaler.mean_[idx_close]
    std = scaler.scale_[idx_close]
    assert std > 0, "종가 표준편차가 0입니다(상수열?)."
    predicted_prices = future_scaled * std + mu                     # (H,)

    # 9) 미래 날짜(X_df 기준) + 참고지표, 예측 구간의 미래 날짜 리스트 생성, start는 마지막 날짜 다음 영업일(Business day)부터 시작
    future_dates = pd.bdate_range(start=X_df.index[-1] + pd.Timedelta(days=1), periods=PREDICTION_PERIOD, freq='B')
    last_close = X_df[col_c].iloc[-1]
    avg_future_return = (predicted_prices.mean() / last_close - 1.0) * 100

    # 기대 성장률 미만이면 건너뜀
    if avg_future_return < EXPECTED_GROWTH_RATE and avg_future_return < 20:
        if avg_future_return > 0:
            print(f"  예상 : {avg_future_return:.2f}%")
        # pass
        continue

    # 결과 저장
    results.append((avg_future_return, ticker, stock_name))

    # 기존 파일 삭제
    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)
        if os.path.isdir(file_path):
            continue
        if file_name.startswith(f"{today}") and ticker in file_name:
            print(f"Deleting existing file: {file_name}")
            send2trash(os.path.join(output_dir, file_name))



    #######################################################################

    # 10) 차트로 전달
    fig = plt.figure(figsize=(14, 16), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

    # sharex: 여러 서브플롯들이 x축(스케일/눈금/포맷)을 같이 쓸지 말지를 정하는 옵션
    ax_d_price = fig.add_subplot(gs[0, 0])
    ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
    ax_w_price = fig.add_subplot(gs[2, 0])
    ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

    daily_chart_title = f'{end_date}  {stock_name} [{ticker}] (예상 상승률: {avg_future_return:.2f}%)'
    plot_candles_daily(data, show_months=6  , title=daily_chart_title,
                       ax_price=ax_d_price, ax_volume=ax_d_vol,
                       future_dates=future_dates, predicted_prices=predicted_prices)

    plot_candles_weekly(data, show_months=12, title="Weekly Chart",
                        ax_price=ax_w_price, ax_volume=ax_w_vol)

    plt.tight_layout()

    # 파일 저장 (옵션)
    final_file_name = f'{today} [ {avg_future_return:.2f}% ] {stock_name} [{ticker}].png'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path)
    plt.close()

#######################################################################

# 정렬 및 출력
results.sort(reverse=True, key=lambda x: x[0])

for avg_future_return, ticker, stock_name in results:
    print(f"==== [ {avg_future_return:.2f}% ] {stock_name} [{ticker}] ====")

try:
    requests.post(
        'https://chickchick.shop/func/stocks/progress-update/nasdaq',
        json={"percent": 100, "done": True},
        timeout=5
    )
except Exception as e:
    # logging.warning(f"progress-update 요청 실패: {e}")
    print(f"progress-update 요청 실패: {e}")
    pass  # 오류

if total_cnt > 0:
    print('R-squared_avg : ', total_r2/total_cnt)
    print('SMAPE : ', total_smape/total_cnt)
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