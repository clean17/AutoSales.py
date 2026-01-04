import matplotlib # tkinter 충돌 방지, Agg 백엔드를 사용하여 GUI를 사용하지 않도록 한다
matplotlib.use('Agg')
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from send2trash import send2trash
from utils import create_lstm_model, create_multistep_dataset, fetch_stock_data_us, get_nasdaq_symbols, \
    extract_stock_code_from_filenames, get_usd_krw_rate, add_technical_features, check_column_types, \
    get_name_from_usa_ticker, plot_candles_daily, plot_candles_weekly, drop_trading_halt_rows, \
    inverse_close_matrix_fast, get_next_business_days, make_naive_preds, smape, pass_filter_v2, regression_metrics , \
    pass_filter, drop_sparse_columns
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
pickle_dir = os.path.join(root_dir, '../pickle_us')
os.makedirs(pickle_dir, exist_ok=True) # 없으면 생성



N_FUTURE = 3
LOOK_BACK = 15
EXPECTED_GROWTH_RATE = 4
DATA_COLLECTION_PERIOD = 700
KR_AVERAGE_TRADING_VALUE = 7_000_000_000
SPLIT      = 0.75

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


def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en

# 결과를 저장할 배열
results = []
total_r2 = 0
total_cnt = 0
total_smape = 0
is_first_flag = True

for count, ticker in enumerate(tickers):
    # stock_name = get_name_from_usa_ticker(ticker)
    # print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")
    if count % 100 == 0:
        print(f"Processing {count+1}/{len(tickers)} : {ticker}")

    # 학습하기 직전에 요청을 보낸다
    percent = f'{round((count+1)/len(tickers)*100, 1):.1f}'
    try:
        requests.post(
            'https://chickchick.shop/stocks/progress-update/nasdaq',
            json={
                "percent": percent,
                "count": count+1,
                "total_count": len(tickers),
                "ticker": ticker,
                "stock_name": "",
                "done": False,
            },
            timeout=10
        )
    except Exception as e:
        # logging.warning(f"progress-update 요청 실패: {e}")
        print(f"progress-update 요청 실패:-nn {e}")
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
    if len(df) > 700:
        df = df.iloc[-700:]

    # 파일 저장
    df.to_pickle(filepath)
    # data = pd.read_pickle(filepath)
    data = df


    # 한국/영문 칼럼 자동 식별
    col_o = _col(df, '시가',   'Open')
    col_h = _col(df, '고가',   'High')
    col_l = _col(df, '저가',   'Low')
    col_c = _col(df, '종가',   'Close')
    col_v = _col(df, '거래량', 'Volume')

    if data is None or col_c not in data.columns or data.empty:
        # print(f"{ticker}: 데이터가 비었거나 'Close' 컬럼이 없습니다. pass.")
        continue

#     check_column_types(fetch_stock_data_us(ticker, start_date_us, end_date), ['Close', 'Open', 'High', 'Low', 'Volume', 'PBR']) # 타입과 shape 확인 > Series 가 나와야 한다
#     continue

    ########################################################################

    actual_prices = data[col_c].values # 최근 종가 배열
    last_close = actual_prices[-1]

    # 0) 우선 거래정지/이상치 행 제거
    data, removed_idx = drop_trading_halt_rows(data)
    if len(removed_idx) > 0:
        # print(f"                                                        거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")
        pass

    if data.empty or len(data) < 70:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 종가가 0.0이거나 500원 미만이면 건너뜀
    last_row = data.iloc[-1]
    if last_row[col_c] == 0.0 or last_row[col_c] * exchangeRate < 500:
        # print("                                                        종가가 0이거나 500원 미만이므로 작업을 건너뜁니다.")
        continue

    # 한달 데이터
    month_data = data.tail(20)
    month_trading_value = month_data[col_v] * month_data[col_c]
    # 하루라도 거래대금이 5억 미만이 있으면 제외
    if (month_trading_value * exchangeRate < 500_000_000).any():
        # print(f"                                                        최근 4주 중 거래대금 5억 미만 발생 → pass")
        continue

    # 최근 2주 평균 거래대금 60억 미만 패스
    recent_data = data.tail(10)
    recent_trading_value = recent_data[col_v] * recent_data[col_c]     # 최근 2주 거래대금 리스트
    recent_average_trading_value = recent_trading_value.mean()
    if recent_average_trading_value * exchangeRate <= KR_AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{(recent_average_trading_value * exchangeRate)/ 100_000_000:.0f}억"
        # print(f"                                                        최근 2주 평균 거래액({formatted_recent_value})이 부족하여 작업을 건너뜁니다.")
        continue

    # ----- 투경 조건 -----
    # 1. 당일의 종가가 3일 전날의 종가보다 100% 이상 상승
    if len(actual_prices) >= 4:
        close_3ago = actual_prices[-4]
        ratio = (last_close - close_3ago) / close_3ago * 100
        if last_close >= close_3ago * 2:  # 100% 이상 상승
            # print(f"                                                        3일 전 대비 100% 이상 상승: {close_3ago} -> {last_close}  {ratio:.2f}% → pass")
            continue

    # rolling window로 5일 전 대비 현재가 3배 이상 오른 지점 찾기
    # rolling_min = data[col_c].rolling(window=5).min()    # 5일 중 최소가
    # ratio = data[col_c] / rolling_min
    #
    # if np.any(ratio >= 2.5):
    #     print(f"                                                        어느 5일 구간이든 2.5배 급등: 제외")
    #     continue


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
    # last_close = data[col_c].iloc[-1]
    # close_4days_ago = data[col_c].iloc[-5]
    #
    # rate = (last_close / close_4days_ago - 1) * 100
    #
    # if rate <= -18:
    #     print(f"                                                        4일 전 대비 {rate:.2f}% 하락 → 학습 제외")
    #     continue  # 또는 return


    # # 최근 3일, 2달 평균 거래량 계산, 최근 3일 거래량이 최근 2달 거래량의 25% 안되면 패스
    # recent_3_avg = data[col_v][-3:].mean()
    # recent_2months_avg = data[col_v][-40:].mean()
    # if recent_3_avg < recent_2months_avg * 0.15:
    #     temp = (recent_3_avg/recent_2months_avg * 100)
    #     # print(f"                                                        최근 3일의 평균거래량이 최근 2달 평균거래량의 25% 미만 → pass : {temp:.2f} %")
    #     # continue
    #     pass

    # 2차 생성 feature
    data = add_technical_features(data)

    # 결측 제거
    cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
    if len(cols_to_drop) > 0:
        pass
#         print("    Drop candidates:", cols_to_drop)
    data = cleaned

    if 'MA5' not in data.columns or 'MA20' not in data.columns:
        # print(f"                                                        이동평균선이 존재하지 않음 → pass")
        continue

    # 5일선이 너무 하락하면
    ma5_today = data['MA5'].iloc[-1]
    ma5_yesterday = data['MA5'].iloc[-2]

    # 변화율 계산 (퍼센트로 보려면 * 100)
    change_rate = (ma5_today - ma5_yesterday) / ma5_yesterday

    # 현재 5일선이 20일선보다 낮으면서 하락중이면 패스
    min_slope = -3
    if ma5_today < data['MA20'].iloc[-1] and change_rate * 100 < min_slope:
        # print(f"                                                        5일선이 20일선 보다 낮으면서 {min_slope}기울기보다 낮게 하락중[{change_rate * 100:.2f}] → pass")
        continue
        # pass


    ########################################################################

    # 학습에 쓸 피처
    feature_cols = [
        col_o, col_l, col_h, col_c, 'Vol_logdiff',
    ]

    cols = [c for c in feature_cols if c in data.columns]  # 순서 보존
    df = data.loc[:, cols].replace([np.inf, -np.inf], np.nan)
    X_df = df.dropna()  # X_df는 (정렬/결측처리된) 피처 데이터프레임, '종가' 컬럼 존재
    close_price = X_df[col_c].to_numpy()

    # 종가 컬럼 이름/인덱스
    idx_close = cols.index(col_c)

    # 2) 시계열 분리 후, train만 fit → val/전체 transform
    split = int(len(X_df) * SPLIT)
    scaler_X = StandardScaler().fit(X_df.iloc[:split])  # 원시 train 구간만, 중복 윈도우 때문에 같은 시점 행이 여러 번 들어가는 왜곡 방지
    X_all = scaler_X.transform(X_df)                    # 전체 변환 (누수 없음)

    # ↓ 여기서 X_all, Y_all을 '스케일된 X'로부터 만듦
    X_tmp, Y_xscale, t0 = create_multistep_dataset(X_all, LOOK_BACK, N_FUTURE, idx=idx_close, return_t0=True)
    t_end = t0 + LOOK_BACK - 1        # 윈도 끝 인덱스 (입력의 마지막 시점)
    t_y_end = t_end + N_FUTURE  # 타깃의 마지막 시점

    # 시점 마스크로 분리
    train_mask = (t_y_end < split)
    val_mask   = (t_y_end >= split)

    # 3) 윈도잉(블록별) → 학습/검증 세트
    X_train, Y_train = X_tmp[train_mask], Y_xscale[train_mask]
    X_val,   Y_val   = X_tmp[val_mask],   Y_xscale[val_mask]

    if is_first_flag:
        is_first_flag = False
        print("X_train", X_train.shape, "Y_train", Y_train.shape)
        print("X_val  ", X_val.shape,   " Y_val  ", Y_val.shape)

    X_val_tune, Y_val_tune = X_val[:-1], Y_val[:-1]    # 튜닝/early stopping 전용
    X_last,     Y_last     = X_val[-1:], Y_val[-1:]    # 마지막 1개(예측 입력 전용; 정답은 개발 중엔 보지 않음)

    # 5) 최소 샘플 수 확인
    if X_train.shape[0] < 50:
        print("                                                        샘플 부족 : ", X_train.shape[0])
        continue

    # ---- y 스케일링: Train으로만 fit ---- (타깃이 수익률이면 생략 가능)
    scaler_y = StandardScaler().fit(Y_train)
    y_train_scaled = scaler_y.transform(Y_train)
    y_val_tune_scaled = scaler_y.transform(Y_val_tune)

    #######################################################################

    # 6) 모델 생성/학습
    def make_huber_per_h(delta_vec, eps=1e-6):
        delta_vec = np.asarray(delta_vec, dtype="float32")
        delta_vec = np.maximum(delta_vec, eps)

        def huber_per_h(y_true, y_pred):
            err = tf.abs(y_true - y_pred)                  # (N,H)
            d   = tf.constant(delta_vec, dtype=err.dtype)  # (H,)
            quad = 0.5 * tf.square(err)
            lin  = d * err - 0.5 * tf.square(d)
            loss = tf.where(err <= d, quad, lin)           # (N,H)
            return tf.reduce_mean(loss)
        return huber_per_h

    stds = Y_train.std(axis=0).astype("float32")
    loss_fn = make_huber_per_h(2.0 * stds)

    # 6) 모델 생성/학습
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]), N_FUTURE,
                              lstm_units=[32], dropout=None, dense_units=[16], loss=loss_fn)

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    history = model.fit(
        X_train, y_train_scaled,
        batch_size=8, epochs=200, verbose=0, shuffle=False,
        validation_data=(X_val_tune, y_val_tune_scaled),
        callbacks=[early_stop, rlrop]
    )

    # 예측 (y-스케일)
    va_pred_s = model.predict(X_val_tune, verbose=0)
    # y-스케일 -> X-스케일
    va_pred_x = scaler_y.inverse_transform(va_pred_s)
    # X-스케일 -> 원 단위(종가만 역변환)
    pred_price = inverse_close_matrix_fast(va_pred_x, scaler_X, idx_close)

    #######################################################################

    # 밸리데이션 인덱스
    val_idx = np.flatnonzero(val_mask)
    # 튜닝용 밸리데이션 윈도우(마지막 1개 제외)
    val_tune_idx = val_idx[:-1]
    last_val_idx = val_idx[-1]    # 마지막 1개(예측용)
    split_tune_end = int(t_end[val_tune_idx].max())   # 원시 인덱스 (inclusive)
    # print('split_tune_end', split_tune_end)

    # 훈련+Val_tune 경계까지 사용(Last/Test 제외). split_tune_end는 직접 계산한 인덱스
    y_insample_price = close_price[:split_tune_end + 1]  # 길이 > m 필요
    # print('y_insample_price', y_insample_price)


    # ++ 지표 계산: scaled + restored(원 단위)
    metrics = regression_metrics(
        y_val_tune_scaled, va_pred_s,
        y_scaler=scaler_y,
        scaler=scaler_X,
        n_features=X_df.shape[1],
        idx_close=idx_close,
        y_insample_for_mase_restored=y_insample_price,
        m=1  # 예: 주기 5 영업일
    )

    # print("=== SCALED (표준화 공간) ===")
    for k,v in metrics["scaled"].items():
        if k == 'R2': # R-squared, (0=엉망, 1=완벽)
            # print(f"                                                        R-squared 0.6 미만이면 패스 : {r2}")
            total_r2 += v
            total_cnt += 1
        # print(f"    {k}: {v:.4f}")

    if "restored" in metrics:
        m_rest = metrics["restored"]
        mase_price = m_rest.get("MASE", np.nan)
        smape_price = m_rest.get("SMAPE (%)", np.nan)
        total_smape += smape_price

    # 가드: NaN/Inf는 탈락
    if not np.isfinite(mase_price):
        print("    MASE is NaN → fail")
        continue

    ok_first = pass_filter(metrics, use_restored=True, r2_min=None, smape_max=8.0)  # 원 단위 기준으로 필터, SMAPE 30으로 필터링하려면 무조건 restored 값으로
    if not ok_first:
        continue

    # ======================
    # 여기가 실제 사용 예
    # 전제:
    #   Y_val      : (N, H)  검증 실제값 (복원된 가격 또는 수익률)
    #   y_hist_end : (N,)    각 윈도우 마지막 실제값 (가격공간일 때 필요)
    #   y_pred_val : (N, H)  모델 예측 (있으면 hitrate 계산에 사용)
    #   use_restored=True라면 price space로 본다고 가정
    # ======================

    """
    나이브: 아주 단순한 규칙으로 만든 예측, 베이스 기준
    """
    # 2) 나이브: 먼저 X스케일에서 퍼시스턴스 행렬을 만들고 → 가격으로 복원
    # (A) 튜닝/평가: 마지막 1개 제외한 Val에 대해서만 지표 산출
    y_hist_end_x = X_val_tune[:, -1, idx_close]                  # (N,)  X-스케일
    H = Y_val_tune.shape[1] if Y_val_tune.ndim == 2 else 1
    y_naive_x = make_naive_preds(y_hist_end_x, horizon=H, mode="price")  # (N_tune, H) X-스케일 기준 naive

    # 가격으로 복원
    Y_val_tune_x  = scaler_y.inverse_transform(y_val_tune_scaled)     # (N,H)  y-스케일 -> X-스케일
    Y_val_tune_price = inverse_close_matrix_fast(Y_val_tune_x, scaler_X, idx_close)
    y_naive_price = inverse_close_matrix_fast(y_naive_x, scaler_X, idx_close)

    # 2) naive sMAPE
    smape_naive = smape(Y_val_tune_price, y_naive_price)
    smape_price = smape(Y_val_tune_price, pred_price)

    # === 2) 윈도우별 sMAPE 배열 ===
    def smape_rows(y_true, y_pred, eps=1e-12):
        num = np.abs(y_pred - y_true)
        den = (np.abs(y_true) + np.abs(y_pred) + eps) / 2.0
        # 윈도우별(행별) H-step 평균 sMAPE(%)를 반환
        return (100.0 * (num / den)).mean(axis=1)

    smape_model_rows = smape_rows(Y_val_tune_price, pred_price)    # (N_tune,)
    smape_naive_rows = smape_rows(Y_val_tune_price, y_naive_price) # (N_tune,)

    median_smape_model = np.nanmedian(smape_model_rows)
    median_smape_naive = np.nanmedian(smape_naive_rows)

    # 개선비율: 윈도우별로 모델 sMAPE가 나이브보다 작은 비율
    improved_window_ratio = float(np.mean(smape_model_rows < smape_naive_rows))

    # === 4) 컷오프 규칙에 반영 (예시) ===
    eps = 1e-9
    rel_ok = (smape_price <= smape_naive * (1 - 0.03))     # 상대 3% 개선
    abs_ok = ((smape_naive - smape_price) >= 0.2)          # 절대 0.2pp 개선
    mase_ok = (mase_price < 1.0 - eps)

    # 1) 베이스라인 격파
    base_ok = (rel_ok or abs_ok or mase_ok)
    # 2) 안정성
    stable_ok = (improved_window_ratio >= 0.60)
    # 3) (선택) 중앙값 보조
    median_ok = (median_smape_model <= median_smape_naive)

    # 기본 합격
    pass_rule = base_ok and stable_ok

    # 표본 수(Val_tune 윈도우 개수, X_val -1)
    n_tune = int(len(smape_model_rows))

    # 표본 적을 때 완화(옵션)
    if n_tune < 80:
        pass_rule = base_ok and (improved_window_ratio >= 0.55)

    # 타이브레이커(옵션)
    if not pass_rule and base_ok:
        tiny_margin = 0.05  # 0.05pp 낮으면 통과시겨줌, 마지막 허용선
        # median_ok를 빼면 타이브레이커가 느슨해지니 권장하지 않음
        if median_ok and ((smape_naive - smape_price) >= tiny_margin):
            pass_rule = True

    # 조건 통과 못함 -> pass
    # if not pass_rule:
    #     continue

    """
    어떤 예측을 쓸지(모델/블렌드/나이브) 를 정하기 위한 “안전장치”
    
    모델이 나이브보다 항상 좋지 않을 수 있으니, Val_tune에서 모델과 나이브를 가중합(α) 하여 오차가 최소가 되는 α*를 찾는다
    >> 리스크(큰 실수) 를 줄이고, 평균 성능을 끌어올릴 수 있다
    
    Val_tune에서 찾은 α*로 만든 혼합 예측(= blend)이 나이브보다 의미 있게 좋다(절대 0.2pp 또는 상대 3% 이상 개선)면, 
    운영/최종 예측에서 이 블렌딩을 사용하겠다는 결정
    """
    if pass_rule:
        decision = ("model", 1.0)     # 운영 예측은 모델 그대로
    else:
        alphas = np.linspace(0, 1, 21) # 0부터 1까지 21개 값(양 끝 포함)을 균등 간격으로 만든다
        best = (None, np.inf)
        for a in alphas:
            blend = a*pred_price + (1-a)*y_naive_price   # 가격 단위
            s = smape(Y_val_tune_price, blend)           # % 기준
            if s < best[1]:
                best = (a, s)
        alpha_star, smape_blend = best

        rel_thresh = smape_naive * (1 - 0.03)   # 상대 3% 개선
        abs_pp = 0.2 if smape_naive >= 3.0 else 0.1 # 나이브가 아주 작을 때는 0.2pp가 과도할 수 있으니
        abs_thresh = smape_naive - abs_pp

        accept_thresh = max(rel_thresh, abs_thresh)  # 더 느슨한(또는 넉넉한) 기준
        eps = 1e-6
        if smape_blend <= accept_thresh + eps:
            decision = ("blend", alpha_star)
        else:
            # decision = ("naive", 0.0)
            continue
    # print("decision:", decision, "alpha*:", alpha_star, "sMAPE_blend:", smape_blend)


    #######################################################################

    # 7) 마지막 윈도우 1개로 미래 H-step 예측
    n_features = X_all.shape[1]
    # X_all[-LOOK_BACK:] : 가장 최근 LOOK_BACK개 시점의 표준화된 피쳐들
    # last_window : 15일치의 피쳐들을 표준화한 블록
    last_window = X_last.reshape(1, LOOK_BACK, n_features)   # shape (1, L, F)
    # print('last_window', last_window)
    future_y_s  = model.predict(last_window, verbose=0)      # shape (1, H)  y-스케일

    # 8) y-표준화 → X-스케일
    future_y_x  = scaler_y.inverse_transform(future_y_s)     # y-스케일 -> X-스케일
    # print('future_y_x', future_y_x)

    y_pred_last_price = inverse_close_matrix_fast(
        future_y_x, scaler_X, idx_close
    ).ravel()  # shape=(H,) 최종 가격 단위

    # --- 나이브(퍼시스턴스) 마지막 윈도우 기준 ---
    last_close = X_df[col_c].iloc[-1]
    H = y_pred_last_price.shape[0]
    y_naive_last_price = np.repeat(last_close, H)

    # --- 운영 결정 반영 (val_tune에서 구한 decision 사용) ---
    # decision = ("model", 1.0) | ("blend", alpha_star) | ("naive", 0.0)
    kind, alpha = decision

    if kind == "model":
        predicted_prices = y_pred_last_price
    elif kind == "blend":
        predicted_prices = alpha * y_pred_last_price + (1 - alpha) * y_naive_last_price
    else:
        continue
    # print('predicted_prices', predicted_prices)

    # 9) 다음 영업일 가져오기
    future_dates = get_next_business_days()
    # print('future_dates', future_dates)
    avg_future_return = (predicted_prices.mean() / last_close - 1.0) * 100
    # print(f"    predicted rate of increase : {avg_future_return:.2f}%")

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
    daily_chart_title = f'{end_date}  {ticker} (예상 상승률: {avg_future_return:.2f}%)'
    plot_candles_daily(data, show_months=6  , title=daily_chart_title,
                       ax_price=ax_d_price, ax_volume=ax_d_vol,
                       future_dates=future_dates, predicted_prices=predicted_prices)

    plot_candles_weekly(data, show_months=12, title="Weekly Chart",
                        ax_price=ax_w_price, ax_volume=ax_w_vol)

    plt.tight_layout()

    # 파일 저장 (옵션)
    # final_file_name = f'{today} [ {avg_future_return:.2f}% ] {stock_name} [{ticker}].webp'
    final_file_name = f'{today} [ {avg_future_return:.2f}% ]  {ticker}.webp'
    final_file_path = os.path.join(output_dir, final_file_name)
    plt.savefig(final_file_path, format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close()

####################################

# 정렬 및 출력
results.sort(reverse=True, key=lambda x: x[0])

# for avg_future_return, ticker, stock_name in results:
for avg_future_return, ticker in results:
    # print(f"==== [ {avg_future_return:.2f}% ] {stock_name} [{ticker}] ====")
    print(f"==== [ {avg_future_return:.2f}% ]  {ticker} ====")

try:
    requests.post(
        'https://chickchick.shop/stocks/progress-update/nasdaq',
        json={"percent": 100, "done": True},
        timeout=10
    )
except Exception as e:
    # logging.warning(f"progress-update 요청 실패: {e}")
    print(f"progress-update 요청 실패-nn: {e}")
    pass  # 오류

if total_cnt > 0:
    print(f'R-squared_avg : {total_r2/total_cnt:.2f}')
    print(f'SMAPE : {total_smape/total_cnt:.2f}')
    print(f'total_cnt : {total_cnt}')


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