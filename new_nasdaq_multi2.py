# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found / Skipping registering GPU devices... 안나오게
import os
# 1) GPU 완전 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2) C++ 백엔드 로그 레벨 낮추기 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib # tkinter 충돌 방지, Agg 백엔드를 사용하여 GUI를 사용하지 않도록 한다
matplotlib.use('Agg')
import os, sys
import pytz
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from utils import create_lstm_model, create_multistep_dataset, fetch_stock_data_us, add_technical_features, \
    plot_candles_daily, plot_candles_weekly, drop_trading_halt_rows, regression_metrics, pass_filter, \
    get_usd_krw_rate, get_nasdaq_symbols, get_name_from_usa_ticker, inverse_close_from_scaled, \
    classify_metrics_from_price, rolling_eval_3ahead, \
    inverse_close_matrix_fast, inverse_close_from_Xscale_fast, prices_from_logrets, log_returns_from_prices, \
    rmse, improve, smape, nrmse
import requests
import time

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

"""
피쳐를 어떤 조합으로 만드냐?
데이터셋의 크기(윈도우 크기)에 따른 차이
모델의 층을 얼마나 쌓냐?
필터링에 따른 차이
"""

output_dir = 'D:\\sp500'
os.makedirs(output_dir, exist_ok=True)

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle_us')
os.makedirs(pickle_dir, exist_ok=True) # 없으면 생성


PREDICTION_PERIOD = 3
EXPECTED_GROWTH_RATE = 3
DATA_COLLECTION_PERIOD = 500
LOOK_BACK = 15
KR_AVERAGE_TRADING_VALUE = 6_000_000_000


"""
AUC (ROC-AUC): 점수(=예측된 3일 수익률)를 기준으로 상승(≥3%) 케이스가 비상승보다 더 높은 점수를 받도록 잘 순위화했는가.
    해석: 0.5=무작위, 0.6~0.7=보통, 0.7+=좋음.
    장점: 컷오프(임계값) 를 정하지 않아도 성능을 볼 수 있음(임계에 덜 민감).
    주의: 검증 라벨에 양성/음성 둘 다 있어야 정의됨. 한쪽만 있으면 NaN → 그래서 single_class 가드가 있음.
    
F1@opt: 검증 구간에서 여러 컷오프를 훑어 F1이 최대가 되는 지점의 F1 점수. (Prec과 Recall의 조화평균)
    해석: 0~1. 불균형(양성이 적음)일 때 정확도보다 유용.
    장점: 실제로 컷오프를 정했을 때의 예측력 체감치를 준다.
    주의(중요): 검증셋에서 최적 컷을 직접 고르면 낙관적 편향이 생김. 더 엄밀히 하려면
               컷은 학습(또는 다른 검증 블록) 에서 정하고, 지금 검증엔 고정 컷으로 F1을 계산하거나,
               컷 없이 AUC를 주지표로 쓰고 F1은 보조로 확인.
               
R² (여기선 R2_GUARD): 레벨(가격) 회귀의 전반적 적합도. 1에 가까울수록 좋음, 0은 “평균만 예측” 수준, 음수도 가능(그보다 나쁨).
    사용 이유: 모델이 완전히 발산/붕괴한 케이스를 걸러냄.
    지표 공간: 선형 스케일 변환에 불변이라 scaled/restored가 동일. 보통 restored와 함께 보고 판단.
    
SMAPE (여기선 SMAPE_MAX): 가격 예측의 상대 오차(%). 낮을수록 좋음.    
    사용 이유: R²가 낮아도 절대 오차가 과도하게 크지 않은지 확인.
    지표 공간: 반드시 복원(restored, 원 단위) 로 계산해야 의미가 맞음.
"""
AUC_MIN    = 0.58      # 무작위(0.5)보단 확실히 낫다고 볼 최소선. (시장/종목에 맞게 0.55~0.65 사이 튜닝)
F1_MIN     = 0.35      # 불균형 데이터 기준 최소 체감 성능 가이드. (데이터에 따라 조정)
R2_GUARD   = 0.00      # 완전 발산 방지용, 평균 예측보다 나쁜 경우 컷
SMAPE_MAX  = 8.0       # 예측 가격의 상대 오차 상한. 너무 큰 오차면 제외.
MIN_POS    = 8            # 권장: 8~10
MIN_NEG    = 8
NAIVE      = 0.95
SPLIT      = 0.85



# 미국 동부 시간대 설정
now_us = datetime.now(pytz.timezone('America/New_York'))
print("미국 동부 시간 기준 현재 시각:", now_us.strftime('%Y-%m-%d %H:%M:%S'))
# 데이터 수집 시작일 계산
start_date_us = (now_us - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y-%m-%d')
start_five_date_us = (now_us - timedelta(days=5)).strftime('%Y-%m-%d')
print("미국 동부 시간 기준 데이터 수집 시작일:", start_date_us)

end_date = datetime.today().strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y%m%d')


# tickers = extract_stock_code_from_filenames(output_dir)
# tickers = get_nasdaq_symbols()
tickers = ['ESPR']
# tickers = ['TSLA']
# tickers = tickers[:200]

def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en






exchangeRate = get_usd_krw_rate()
if exchangeRate is None:
    print('#######################   exchangeRate is None   #######################')
else:
    print(f'#######################   exchangeRate is {exchangeRate}   #######################')


# print('==================================================')
# print("    RMSE, MAE → 예측 오차를 주가 단위(원) 그대로 해석 가능")
# print("    MAPE (%) → 평균적으로 몇 % 오차가 나는지 직관적")
# print("      0~10%: 매우 우수, 10~20%: 양호, 20~50%: 보통, 50% 이상: 부정확")
# print("    SMAPE (%) → SMAPE는 0% ~ 200% 범위를 가지지만, 보통은 0~100% 사이에서 해석")
# print("      0 ~ 10% → 매우 우수, 10 ~ 20% → 양호 (실사용 가능한 수준), 20 ~ 50% → 보통, 50% 이상 → 부정확")
# print("    R² → 모델 설명력, 0~1 범위에서 클수록 좋음")
# print("      0.6: 변동성의 약 60%를 설명")
# print('==================================================')

# 결과를 저장할 배열
results = []
total_r2 = 0
total_cnt = 0
total_smape = 0

total_r2_rest = 0
total_smape_rest = 0
total_auc = 0
total_f1 = 0

# 데이터 가져오는것만 1시간 걸리네
for count, ticker in enumerate(tickers):
    # stock_name = get_name_from_usa_ticker(ticker)
    # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")
    print(f"Processing {count+1}/{len(tickers)} : [{ticker}]")


    # 학습하기 직전에 요청을 보낸다
    # percent = f'{round((count+1)/len(tickers)*100, 1):.1f}'
    # try:
    #     requests.post(
    #         'https://chickchick.shop/func/stocks/progress-update/nasdaq',
    #         json={
    #             "percent": percent,
    #             "count": count+1,
    #             "total_count": len(tickers),
    #             "ticker": ticker,
    #             "stock_name": "",
    #             "done": False,
    #         },
    #         timeout=10
    #     )
    # except Exception as e:
    #     # logging.warning(f"progress-update 요청 실패: {e}")
    #     print(f"progress-update 요청 실패: {e}")
    #     pass  # 오류


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
    if len(df) > 500:
        df = df.iloc[-500:]

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
    # if len(removed_idx) > 0:
    #     print(f"                                                        거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")

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
        # print(f"                                                        최근 2주 평균 거래액({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
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


    # 최근 3일, 2달 평균 거래량 계산, 최근 3일 거래량이 최근 2달 거래량의 25% 안되면 패스
    recent_3_avg = data['Volume'][-3:].mean()
    recent_2months_avg = data['Volume'][-40:].mean()
    if recent_3_avg < recent_2months_avg * 0.15:
        temp = (recent_3_avg/recent_2months_avg * 100)
        # print(f"                                                        최근 3일의 평균거래량이 최근 2달 평균거래량의 25% 미만 → pass : {temp:.2f} %")
        # continue
        pass

    # 2차 생성 feature
    data = add_technical_features(data)
    # print(data)

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

    if 'MA20' not in data.columns or 'MA5' not in data.columns:
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
        col_o, col_l, col_h, col_c, 'Vol_logdiff',
    ]

    # 1) feature_cols만 남기고 dropna
    feature_cols = [c for c in feature_cols if c in data.columns]
    # print('feature_cols', feature_cols)
    X_df = data.dropna(subset=feature_cols).loc[:, feature_cols]


    # 종가 컬럼 이름/인덱스
    idx_close = feature_cols.index(col_c)

    # 2) 시계열 분리 후, train만 fit → val/전체 transform
    split = int(len(X_df) * SPLIT)

    scaler = StandardScaler()
    scaler.fit(X_df.iloc[:split])                   # ✅ 누수 방지
    X_all_2d = scaler.transform(X_df)        # 전체 transform (누수 아님)
    X_all, Y_all = create_multistep_dataset(X_all_2d, LOOK_BACK, PREDICTION_PERIOD, idx=idx_close)
    t0 = np.arange(len(X_all)) + LOOK_BACK          # 윈도우 끝 바로 다음 시점
    train_mask = (t0 + PREDICTION_PERIOD - 1) < split   # 타깃의 마지막 시점이 split '이전'에 끝나야 train
    val_mask   = (t0 >= split)          # 타깃의 첫 시점이 split '이후'여야 val
    X_train, Y_train = X_all[train_mask], Y_all[train_mask]
    X_val,   Y_val   = X_all[val_mask],   Y_all[val_mask]
    if count == 0:
        print('X_train', X_train.shape)      # X_train (218, 15, 6)
        print('X_val', X_val.shape)          # X_val (40, 15, 6), 단타 기준 35~50 적용


    assert np.allclose(X_val[1:, -1, idx_close], Y_val[:-1, 0], atol=1e-8)


    # 5) 최소 샘플 수 확인
    if X_train.shape[0] < 50:
        continue

    #######################################################################

    # 6) 모델 생성/학습
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]), PREDICTION_PERIOD,
                              lstm_units=[32], dropout=0.05, dense_units=[16])
                              # lstm_units=[48, 24] # 한층만 키우면, 드롭아웃은 0.0도 괜찮다 데이터셋이 너무 작아서


    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    """
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',   # 모니터링할 지표
        factor=0.5,           # 학습률을 절반으로 줄임
        patience=5,           # 5 epoch 동안 개선 없으면 실행
        min_lr=1e-6           # 학습률 하한선
    )
    """
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    history = model.fit(X_train, Y_train, batch_size=8, epochs=200,
                        validation_data=(X_val, Y_val),
                        shuffle=False, verbose=0, callbacks=[early_stop, rlrop])

    # (선택) 검증 예측
    """
        X_val : 검증에 쓸 여러 시계열 구간의 집합
        preds : 검증셋 각 구간(윈도우)에 대해 미래 PREDICTION_PERIOD만큼의 예측치 반환
        shape: (검증샘플수, 예측일수)
    """
    preds = model.predict(X_val, verbose=0)

    #######################################################################

    y_hist_end_va = X_val[:, -1, idx_close] # (N_va,) 각 윈도우의 마지막 타깃 관측값
    y_train_series = X_df.iloc[:split][col_c] # 종가만 1D 시리즈로 추출 (훈련 구간만)
    y_insample_for_mase = X_df.iloc[:split][col_c].astype(float).values # MASE 분모용 (스케일 전 원본 타깃 1D)

    """
        create_multistep_dataset는 가격 레벨(종가)을 타깃으로 뽑고 있음
        >> SMAPE는 restored 기준이 맞고, R²는 어느 쪽이든 같지만 restored로 같이 보는 걸 추천
        
        타겟이 가격이면 지표 비교는 RESTORED 기준
        3일 뒤 +3% 이상인가? >> returns 기반 분류 >> 복원된 가격으로 수익률을 계산해 임계값으로 분류 지표
        
        R²: 회귀 모델의 성능을 평가하는 대표 지표 >> 예측값이 실제값의 분산을 얼마나 설명하는가?를 수치로
    """
    # ++ 지표 계산: scaled + restored(원 단위)
    # metrics = regression_metrics(
    #     Y_val, preds,
    #     scaler=scaler,
    #     n_features=X_df.shape[1],
    #     idx_close=idx_close
    # )

    # if "scaled" in metrics:
    #     print("=== SCALED (표준화 공간) ===")
    #     for k,v in metrics["scaled"].items():
    #         if k == 'SMAPE (%)':
    #             total_smape += v
    #         if k == 'R2': # R-squared, (0=엉망, 1=완벽)
    #             # print(f"                                                        R-squared 0.6 미만이면 패스 : {r2}")
    #             total_r2 += v
    #         # print(f"    {k}: {v:.4f}")

    # if "restored" in metrics:
    #     print("\n=== RESTORED (원 단위) ===")
    #     for k,v in metrics["restored"].items():
    #         print(f"    {k}: {v:.4f}")

    """
        True → metrics["restored"](원 단위로 역변환된 값) 사용
            SMAPE/MAE/RMSE는 원 단위(복원) 에서만 의미가 제대로 남는다
        False → metrics["scaled"](표준화/정규화된 스케일 공간) 사용
            (returns 공간에서 R²/MAE 등)
    """
    # ok = pass_filter(metrics, use_restored=True, r2_min=0.5, smape_max=8.0)
    # # print("                                                        PASS 1st filter?" , ok)
    # if not ok:
    #     # continue
    #     pass

    # r2_rest   = metrics.get("restored", {}).get("R2", metrics["scaled"]["R2"])
    # smape_rest= metrics.get("restored", {}).get("SMAPE (%)", float("inf"))

    # 나이브 비교
    metrics, (y_true, y_pred, y_naive) = rolling_eval_3ahead(
        preds,
        Y_val,
        y_hist_end_va,                      # X_val 각 윈도우의 마지막 '종가' (스케일 공간이어도 됨)
        y_insample_for_mase=y_insample_for_mase, # ← 반드시 '원가격' 1D
        scaler=scaler,                      # ← 넘겨야 복원 수행
        n_features=X_df.shape[1],
        idx_close=idx_close
    )

    r2_rest     = metrics["model"]["R2"]         # 입력이 복원 가격이면 이게 곧 r2_rest
    smape_rest  = metrics["model"]["sMAPE"]      # 동일
    smape_naive = metrics["naive"]["sMAPE"]
    # print('smape_rest', smape_rest)

    """
        의사결정 성능(AUC/F1) 으로 1차 필터 → “+3%를 잘 골라내는가”를 직접 평가
        회귀 가드레일(R²/SMAPE) 로 2차 안전장치 → 완전 발산/품질 저하 방지
    
        튜닝 가이드    
            상승 케이스가 드물면 AUC_MIN 0.55~0.60, F1_MIN 0.25~0.40 정도에서 시작
            장세/종목군에 따라 컷을 넓히거나(완화) 좁히면(엄격) 됩니다.
            “내일만 판단”이면 h_idx=0 고정.
    """
    # print('metrics', metrics)

    """
    디버깅 코드
    h = min(2, preds.shape[1]-1)

    # 1) 복원 일관성 체크 (둘 다 복원)
    Yp = inverse_close_from_scaled(preds, scaler, X_df.shape[1], idx_close)
    Yt = inverse_close_from_scaled(Y_val, scaler, X_df.shape[1], idx_close)

    # 2) 같은 호라이즌 비교 범위/분포
    print("Yp(h) min/max/mean:", Yp[:,h].min(), Yp[:,h].max(), Yp[:,h].mean())
    print("Yt(h) min/max/mean:", Yt[:,h].min(), Yt[:,h].max(), Yt[:,h].mean())

    # 3) 현재가(분모) sanity
    cur = inverse_close_from_scaled(X_val[:, -1, idx_close][:,None], scaler, X_df.shape[1], idx_close)[:,0]
    print("cur min/max:", cur.min(), cur.max())

    # 4) 샘플 몇 개 직접 비교
    for i in range(3):
        print(i, "cur=", cur[i], " pred=", Yp[i,h], " true=", Yt[i,h],
              " rel_err=", abs(Yp[i,h]-Yt[i,h]) / ((abs(Yp[i,h])+abs(Yt[i,h]))/2))


    last_close = X_val[:, -1, idx_close]
    yhat_persist = np.tile(last_close[:, None], (1, preds.shape[1]))
    m_model   = regression_metrics(Y_val, preds, scaler=scaler, n_features=X_df.shape[1], idx_close=idx_close)
    m_naive   = regression_metrics(Y_val, yhat_persist, scaler=scaler, n_features=X_df.shape[1], idx_close=idx_close)
    print("SMAPE model vs naive:", m_model["restored"]["SMAPE (%)"], m_naive["restored"]["SMAPE (%)"])
    """



    # 2) 분류 지표 (3일 후 +3% 판단) — PREDICTION_PERIOD가 1이면 h=0
    h_idx = min(2, preds.shape[1]-1)   # 0:내일, 1:모레, 2:글피
    # print('h_idx', h_idx)
    cls = classify_metrics_from_price(
        preds, Y_val, X_val,
        scaler=scaler, n_features=X_df.shape[1], idx_close=idx_close,
        horizon_idx=h_idx, thresh_ret=0.03,
    )

    pos = cls["pos"]      # 검증셋에서 양성(라벨=1) 개수. 여기서 양성= “3일 누적 수익률 ≥ 3%”
    neg = cls["neg"]      # 검증셋에서 음성(라벨=0) 개수
    auc = cls.get("AUC", np.nan)   # 없으면 NaN으로
    f1  = cls['F1@opt']

    # 라벨 희소 여부 체크: pos_rate가 너무 낮으면(예: <5%) F1/AUC가 크게 출렁임
    pos_rate = cls["pos"] / (cls["pos"] + cls["neg"])
    # print('pos_rate', pos_rate)
    if pos_rate < 0.05:
        # print(f"    [VAL] pos={cls['pos']}, neg={cls['neg']}, pos_rate={pos_rate:.2%}, pos_rate가 너무 낮으면(예: <5%) F1/AUC가 크게 출렁임")
        continue

    def guard_regression(naive_compare=True):
        base = (r2_rest >= R2_GUARD) and (smape_rest <= SMAPE_MAX)
        if naive_compare:
            guard_ok = (smape_rest <= NAIVE * smape_naive) # 베이스라인 대비 5% 이상 개선
            base = base and guard_ok
        return base

    # print(f"    [METRICS] R2(rest)={r2_rest:.3f}  SMAPE(rest)={smape_rest:.2f}  "
    #       f"AUC={cls['AUC']:.3f}  F1@opt={cls['F1@opt']:.3f}  th_opt={cls['th_opt']:.4f}")

    # 3) 하이브리드 필터: 의사결정 성능 + 회귀 가드레일
    # 이벤트 존재여부를 검사하는 가드
    if cls.get("single_class", False):
        # 양/음 한쪽이 0 → PR/ROC 무의미 → 회귀 가드레일만
        ok = guard_regression(naive_compare=True)     # 분류 스킵
        # print(f"    [INFO] single_class (pos={pos}, neg={neg}) → skip AUC/F1")
        ok = False
    elif (pos < MIN_POS) or (neg < MIN_NEG):
        # 이벤트가 극소수 → 분류지표 신뢰도 낮음 → 스킵 또는 완화... 최소 품질선은 넘은 것 > 실거래 시 보수적
        ok = guard_regression(naive_compare=True)     # 분류 스킵(또는 완화)
        # print(f"    [INFO] few events (pos={pos}, neg={neg}) → skip/relax AUC/F1")
        ok = False
    else:
        print(f"    [METRICS] R2(rest)={r2_rest:.3f}  SMAPE(rest)={smape_rest:.2f}  "
              f"AUC={cls['AUC']:.3f}  F1@opt={cls['F1@opt']:.3f}  th_opt={cls['th_opt']:.4f}")
        ok = (np.isfinite(auc) and auc >= AUC_MIN) and (f1 >= F1_MIN) and guard_regression(naive_compare=True)

    if not ok:
        continue

    # if r2_rest > 0:
    #     total_r2_rest += r2_rest
    total_r2_rest += r2_rest
    total_smape_rest += smape_rest

    if np.isfinite(auc):          # 유한수만 합산
        total_auc += auc
    total_f1 += f1
    total_cnt += 1

    #######################################################################

    # =========================
    # 통과했으면 미래 예측/복원
    # =========================

    n_features = X_all_2d.shape[1] # n_features: 피쳐 칼럼 수
    last_window = X_all_2d[-LOOK_BACK:].reshape(1, LOOK_BACK, n_features)
    future_scaled = model.predict(last_window, verbose=0)[0]        # (H,)

    # 스케일러 종류 무관, 안전 복원
    predicted_prices = inverse_close_from_scaled(
        future_scaled.reshape(1, -1), scaler, n_features, idx_close
    ).ravel()  # (H,)

    # 9) 미래 날짜(X_df 기준) + 참고지표, 예측 구간의 미래 날짜 리스트 생성, start는 마지막 날짜 다음 영업일(Business day)부터 시작
    future_dates = pd.bdate_range(start=X_df.index[-1] + pd.Timedelta(days=1), periods=PREDICTION_PERIOD, freq='B')
    last_close = X_df[col_c].iloc[-1]
    avg_future_return = (predicted_prices.mean() / last_close - 1.0) * 100

    print(f"  predicted rate of increase : {avg_future_return:.2f}%")
    # 기대 성장률 미만이면 건너뜀
    if avg_future_return < EXPECTED_GROWTH_RATE and avg_future_return < 20:
        # if avg_future_return > 0:
        #     print(f"  predicted rate of increase : {avg_future_return:.2f}%")
        # pass
        continue

    # 결과 저장
    # results.append((avg_future_return, ticker, stock_name))
    results.append((avg_future_return, ticker))

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

    # daily_chart_title = f'{end_date}  {stock_name} [{ticker}] (예상 상승률: {avg_future_return:.2f}%)'
    daily_chart_title = f'{end_date}  [{ticker}] (예상 상승률: {avg_future_return:.2f}%)'
    plot_candles_daily(data, show_months=6  , title=daily_chart_title,
                       ax_price=ax_d_price, ax_volume=ax_d_vol,
                       future_dates=future_dates, predicted_prices=predicted_prices)

    plot_candles_weekly(data, show_months=12, title="Weekly Chart",
                        ax_price=ax_w_price, ax_volume=ax_w_vol)

    plt.tight_layout()

    # 파일 저장 (옵션)
    # final_file_name = f'{today} [ {avg_future_return:.2f}% ] {stock_name} [{ticker}].webp'
    final_file_name = f'{today} [ {avg_future_return:.2f}% ] [{ticker}].webp'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path, format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()





#######################################################################

# 정렬 및 출력
results.sort(reverse=True, key=lambda x: x[0])

# for avg_future_return, ticker, stock_name in results:
for avg_future_return, ticker in results:
    # print(f"==== [ {avg_future_return:.2f}% ] {stock_name} [{ticker}] ====")
    print(f"==== [ {avg_future_return:.2f}% ]  [{ticker}] ====")

try:
    requests.post(
        'https://chickchick.shop/func/stocks/progress-update/nasdaq',
        json={"percent": 100, "done": True},
        timeout=10
    )
except Exception as e:
    # logging.warning(f"progress-update 요청 실패: {e}")
    print(f"progress-update 요청 실패-nn: {e}")
    pass  # 오류

if total_cnt > 0:
    print('')
    print('R-squared_avg : ', total_r2_rest/total_cnt)
    print('SMAPE : ', total_smape_rest/total_cnt)
    print('AUC : ', total_auc/total_cnt)
    print('F1 : ', total_f1/total_cnt)
    print('total_cnt : ', total_cnt)
