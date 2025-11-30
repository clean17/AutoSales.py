# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found / Skipping registering GPU devices... 안나오게
import os
# 1) GPU 완전 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2) C++ 백엔드 로그 레벨 낮추기 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import matplotlib
matplotlib.use('Agg')
import os, sys
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import create_lstm_model, create_multistep_dataset, fetch_stock_data, add_technical_features, \
    get_kor_ticker_dict_list, plot_candles_daily, plot_candles_weekly, drop_trading_halt_rows, regression_metrics, \
    pass_filter, inverse_close_matrix_fast, get_next_business_days, make_naive_preds, smape, pass_filter_v2, \
    drop_sparse_columns
import requests
import time

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


output_dir = 'D:\\kospi_stocks'
os.makedirs(output_dir, exist_ok=True)

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
os.makedirs(pickle_dir, exist_ok=True) # 없으면 생성

N_FUTURE = 3
LOOK_BACK = 15
EXPECTED_GROWTH_RATE = 4
DATA_COLLECTION_PERIOD = 700 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45
AVERAGE_TRADING_VALUE = 2_000_000_000 # 평균거래대금 20억
SPLIT      = 0.75

today = datetime.today().strftime('%Y%m%d')
today_us = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')
start_five_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')


# chickchick.com에서 종목 리스트 조회
tickers_dict = get_kor_ticker_dict_list()
tickers = list(tickers_dict.keys())
# tickers = ['204620'] # 글로벌 텍스프리
# tickers = tickers[1526:]

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

"""
SMAPE (%) → SMAPE는 0% ~ 200% 범위를 가지지만, 보통은 0~100% 사이에서 해석")
  0 ~ 10% → 매우 우수, 10 ~ 20% → 양호 (실사용 가능한 수준), 20 ~ 50% → 보통, 50% 이상 → 부정확
R² → 모델 설명력, 0~1 범위에서 클수록 좋음
  0.6: 변동성의 약 60%를 설명
"""

# 데이터 가져오는것만 1시간 걸리네
for count, ticker in enumerate(tickers):
    time.sleep(0.2)  # 200ms 대기
    stock_name = tickers_dict.get(ticker, 'Unknown Stock')
    # stock_name = '파인엠텍' # 테스트용
    print(f"Processing {count+1}/{len(tickers)} : {stock_name} [{ticker}]")


    # 학습하기전에 요청을 보낸다
    percent = f'{round((count+1)/len(tickers)*100, 1):.1f}'
    try:
        requests.post(
            'https://chickchick.shop/func/stocks/progress-update/kospi',
            json={
                "percent": percent,
                "count": count+1,
                "total_count": len(tickers),
                "ticker": ticker,
                "stock_name":stock_name,
                "done": False,
            },
            timeout=5
        )
    except Exception as e:
        # logging.warning(f"progress-update 요청 실패: {e}")
        print(f"progress-update 요청 실패-k: {e}")
        pass  # 오류


    # 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    if os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        data = fetch_stock_data(ticker, start_five_date, today)
    else:
        df = pd.DataFrame()
        data = fetch_stock_data(ticker, start_date, today)

    # 중복 제거 & 새로운 날짜만 추가 >> 덮어쓰는 방식으로 수정
    if not df.empty:
        # df와 data를 concat 후, data 값으로 덮어쓰기
        df = pd.concat([df, data])
        df = df[~df.index.duplicated(keep='last')]  # 같은 인덱스일 때 data가 남음
    else:
        df = data.copy()

    # 너무 먼 과거 데이터 버리기
    if len(df) > 500:
        df = df.iloc[-500:]

    # 파일 저장
    df.to_pickle(filepath)
    # continue # 데이터 저장용
    # df = pd.read_pickle(filepath)    # 디버깅용
    data = df

    # 한국/영문 칼럼 자동 식별
    col_o = _col(df, '시가',   'Open')
    col_h = _col(df, '고가',   'High')
    col_l = _col(df, '저가',   'Low')
    col_c = _col(df, '종가',   'Close')
    col_v = _col(df, '거래량', 'Volume')

    ########################################################################

    actual_prices = data[col_c].values # 종가 배열
    last_close = actual_prices[-1]

    # 거래정지/이상치 행 제거
    data, removed_idx = drop_trading_halt_rows(data)
    if len(removed_idx) > 0:
        # print(f"                                                        거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")
        pass

    # 데이터가 부족하면 패스
    if data.empty or len(data) < 70:
        # print(f"                                                        데이터 부족 → pass")
        continue

    # 500원 미만이면 패스
    last_row = data.iloc[-1]
    if last_row[col_c] < 500:
        # print("                                                        종가가 0이거나 500원 미만 → pass")
        continue

    # # 최근 한달 거래대금 중 3억 미만이 있으면 패스
    # month_data = data.tail(20)
    # month_trading_value = month_data[col_v] * month_data[col_c]
    # # 하루라도 거래대금이 3억 미만이 있으면 제외
    # if (month_trading_value < 300_000_000).any():
    #     # print(f"                                                        최근 4주 중 거래대금 3억 미만 발생 → pass")
    #     continue
    #
    # # 최근 2주 거래대금이 기준치 이하면 패스
    # recent_data = data.tail(10)
    # recent_trading_value = recent_data[col_v] * recent_data[col_c]
    # recent_average_trading_value = recent_trading_value.mean()
    # if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
    #     formatted_recent_value = f"{recent_average_trading_value / 100_000_000:.0f}억"
    #     # print(f"                                                        최근 2주 평균 거래대금({formatted_recent_value})이 부족 → pass")
    #     continue

    # 최근 3거래일 거래대금이 기준치 이하면 패스
    recent_5data = data.tail(3)
    recent_5trading_value = recent_5data[col_v] * recent_5data[col_c]
    recent_average_trading_value = recent_5trading_value.mean()
    if recent_average_trading_value <= AVERAGE_TRADING_VALUE:
        formatted_recent_value = f"{recent_average_trading_value / 100_000_000:.0f}억"
        # print(f"                                                        최근 3거래일 평균 거래대금({formatted_recent_value})이 부족 → pass")
        continue

    # 최고가 대비 현재가가 50% 이상 하락한 경우 건너뜀
    # max_close = np.max(actual_prices)
    # drop_pct = ((max_close - last_close) / max_close) * 100
    # if drop_pct >= 50:
    #     # print(f"                                                        최고가 대비 현재가가 50% 이상 하락한 경우 → pass : {drop_pct:.2f}%")
    #     # continue
    #     pass

    # ----- 투경 조건 -----
    # 1. 당일의 종가가 3일 전날의 종가보다 100% 이상 상승
    close_3ago = actual_prices[-4]
    ratio = (last_close - close_3ago) / close_3ago * 100
    if last_close >= close_3ago * 2:  # 100% 이상 상승
        print(f"                                                        3일 전 대비 100% 이상 상승: {close_3ago} -> {last_close}  {ratio:.2f}% → pass")
        continue

    # 2. 당일의 종가가 5일 전날의 종가보다 60% 이상 상승
    close_5ago = actual_prices[-6]
    ratio = (last_close - close_5ago) / close_5ago * 100
    if last_close >= close_5ago * 1.6:  # 60% 이상 상승
        print(f"                                                        5일 전 대비 60% 이상 상승: {close_5ago} -> {last_close}  {ratio:.2f}% → pass")
        continue

    # 3. 당일의 종가가 15일 전날의 종가보다 100% 이상 상승
    # close_15ago = actual_prices[-16]
    # ratio = (last_close - close_15ago) / close_15ago * 100
    # if last_close >= close_15ago * 2:  # 100% 이상 상승
    #     print(f"                                                        15일 전 대비 100% 이상 상승: {close_15ago} -> {last_close}  {ratio:.2f}% → pass")
    #     continue
    # --------------------

    # 현재 종가가 4일 전에 비해서 크게 하락하면 패스
    # close_4days_ago = actual_prices[-5]
    # rate = (last_close / close_4days_ago - 1) * 100 # 오늘 종가와 4일 전 종가의 상승/하락률(%)
    # if rate <= -18:
    #     print(f"                                                        4일 전 대비 {rate:.2f}% 하락 → pass")
    #     continue  # 또는 return


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
        print(f"                                                        5일선이 20일선 보다 낮으면서 {min_slope}기울기보다 낮게 하락중[{change_rate * 100:.2f}] → pass")
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

    if col_c not in X_df.columns:
        raise KeyError(f"'{col_c}' 컬럼이 없습니다.")

    # 종가 컬럼 이름/인덱스
    idx_close = cols.index(col_c)
    close_price = X_df[col_c].to_numpy()

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
        print('len(df)', len(df))
        print("X_train", X_train.shape, "Y_train", Y_train.shape)
        print("X_val  ", X_val.shape,   " Y_val  ", Y_val.shape)

    X_val_tune, Y_val_tune = X_val[:-1], Y_val[:-1]    # 튜닝/early stopping 전용
    X_last,     Y_last     = X_val[-1:], Y_val[-1:]    # 마지막 1개(예측 입력 전용; 정답은 개발 중엔 보지 않음)

    # 5) 최소 샘플 수 확인
    if X_train.shape[0] < 50:
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


    # 모델 평가
    # val_loss = model.evaluate(X_val, Y_val, verbose=1)
    # print("Validation Loss :", val_loss)

    # 예측 (y-스케일)
    va_pred_s = model.predict(X_val_tune, verbose=0)
    # y-스케일 -> X-스케일
    va_pred_x = scaler_y.inverse_transform(va_pred_s)
    # X-스케일 -> 원 단위(종가만 역변환)
    pred_price = inverse_close_matrix_fast(va_pred_x, scaler_X, idx_close)  # (N_val, H)

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
        y_insample_for_mase_restored=y_insample_price,  # ★ 여기!
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
    #     print("\n=== RESTORED (원 단위) ===")
    #     for k,v in metrics["restored"].items():
    #         print(f"    {k}: {v:.4f}")
        m_rest = metrics["restored"]
        mase_price = m_rest.get("MASE", np.nan)
        smape_price = m_rest.get("SMAPE (%)", np.nan)
        total_smape += smape_price
        # print("MASE(price)=", mase_price, "SMAPE(price)=", smape_price)

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
    # print('y_naive_price', y_naive_price)

    # 2) naive sMAPE
    smape_naive = smape(Y_val_tune_price, y_naive_price)
    smape_price = smape(Y_val_tune_price, pred_price)


    # 같은 공간을 사용하는지 체크 - 모델이 진짜로 나이브와 다른 예측을 하고 있나?
    # True면, 모델 예측이 나이브(내일=오늘) 와 사실상 같다는 뜻, 같은 공간이면 안됨
    # print("allclose(pred, naive):", np.allclose(pred_price, y_naive_price))
    # max |diff|가 0에 가깝다면 거의 동일 → 모델이 정보 추가를 못하고 있을 가능성
    # print("max |diff| :", np.max(np.abs(pred_price - y_naive_price)))
    # 첫 몇 개 샘플 비교
    # for i in range(3):
    #     print("win", i, "pred", pred_price[i], "naive", y_naive_price[i], "true", Y_val_tune_price[i])


    # 첫 스텝(h=1)이 입력 마지막 값과 같은지 건전성 체크, 둘의 값이 비슷해야함
    # last_obs = X_val_tune[:, -1, idx_close]
    # print("corr(pred[:,0], last_obs):", np.corrcoef(pred_price[:,0], last_obs)[0,1])
    # print("corr(true[:,0], last_obs):", np.corrcoef(Y_val_tune_price[:,0], last_obs)[0,1])


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

    # print(f"[ROW] median sMAPE model: {median_smape_model:.3f}  naive: {median_smape_naive:.3f}") # 낮아야 더 좋음

    # === 3) 윈도우별 MASE 배열(옵션) ===
    # MASE 분모: 인샘플(Train+Val_tune)에서 m-시즌 절대차 평균 (상수)
    # m_season = 1
    # den = np.mean(np.abs(y_insample_price[m_season:] - y_insample_price[:-m_season]))  # 인샘플 분모
    # mae_rows_model  = np.mean(np.abs(pred_price      - Y_val_tune_price), axis=1)      # ✅ 모델
    # mae_rows_naive  = np.mean(np.abs(y_naive_price   - Y_val_tune_price), axis=1)      # ✅ 나이브
    # mase_rows_model = mae_rows_model / (den + 1e-12)
    # mase_rows_naive = mae_rows_naive / (den + 1e-12)

    # median_mase_model = np.nanmedian(mase_rows_model)
    # median_mase_naive = np.nanmedian(mase_rows_naive)
    # print(f"[ROW] median MASE(model, m={m_season}): {median_mase_model:.3f}")
    # print(f"[ROW] median MASE(naive,  m={m_season}): {median_mase_naive:.3f}")

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



    # # 3) hit-rate (옵션) — 모델 예측이 있을 때만
    # try:
    #     # 4) (옵션) hit-rate도 가격공간 기준이면 베이스도 가격으로
    #     y_hist_end_price = inverse_close_from_Xscale_fast(y_hist_end_x, scaler_X, idx_close)
    #     hitrate_val = hit_rate(
    #         Y_val_tune_price, pred_price,
    #         y_base=y_hist_end_price,
    #         # use_horizon=1,   # h=1 기준; 전체 평균 쓰려면 "avg"
    #         use_horizon="avg",   # h=1 기준; 전체 평균 쓰려면 "avg"
    #         space="price"
    #     )
    # except NameError:
    #     hitrate_val = None

    # ctx = {
    #     "smape_naive": smape_naive,
    #     "n_val": int(Y_val.shape[0] if Y_val.ndim == 2 else len(Y_val)),
    #     "hitrate": None,
    # }

    # # 가격으로 복원 (N,)
    # y_base_price = y_hist_end_x * scaler_X.scale_[idx_close] + scaler_X.mean_[idx_close]
    #
    # # 1) 유효 시행 수(필요시 tiny-move 필터 포함) 계산
    # N_eff, _ = effective_trials_for_hitrate(
    #     Y_val_tune_price, pred_price, y_base=y_base_price,
    #     space="price", use_horizon=1, thr=0.003
    # )
    #
    # # 2) 유의수준(기본 5%)에 맞는 동적 임계치 계산
    # cut = min_sig_hitrate(N_eff, alpha=0.05)  # 예: N_eff=200이면 cut≈0.569
    #
    # # 3) 필터에 적용
    # # 고정 파라미터와 동적 컷을 함께 쓰고 싶다면 max()로 보수적으로:
    # dynamic_hitrate_min = cut

    # ok = pass_filter_v2(
    #     metrics, use_restored=True,
    #     r2_min=0.10, smape_max=10.0,
    #     require_naive_improve=True,
    #     naive_improve_min=0.05,
    #     samples_min=30,
    #     ctx=ctx
    # )
    # print("                                                        PASS 1st filter?" , ok)
    # if not ok:
    #     continue

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
    print(f"    predicted rate of increase : {avg_future_return:.2f}%")

    # 기대 성장률 미만이면 건너뜀
    if avg_future_return < EXPECTED_GROWTH_RATE and avg_future_return < 20:
        # if avg_future_return > 0:
        #     print(f"  predicted rate of increase : {avg_future_return:.2f}%")
        continue
        # pass

    # 결과 저장
    results.append((avg_future_return, stock_name, ticker))

    # 기존 파일 삭제
    for file_name in os.listdir(output_dir):
        if file_name.startswith(f"{today}") and stock_name in file_name and ticker in file_name:
            # print(f"                                                        Deleting existing file: {file_name}")
            os.remove(os.path.join(output_dir, file_name))



    #######################################################################

    # 10) 차트로 전달
    fig = plt.figure(figsize=(14, 16), dpi=150)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

    # sharex: 여러 서브플롯들이 x축(스케일/눈금/포맷)을 같이 쓸지 말지를 정하는 옵션
    ax_d_price = fig.add_subplot(gs[0, 0])
    ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
    ax_w_price = fig.add_subplot(gs[2, 0])
    ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

    daily_chart_title = f'{today_us}   {stock_name} [ {ticker} ] (예상 상승률: {avg_future_return:.2f}%) {predicted_prices.mean():1f}'
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

for avg_future_return, stock_name, ticker in results:
    print(f"==== [ {avg_future_return:.2f}% ] {stock_name} [{ticker}] ====")

try:
    requests.post(
        'https://chickchick.shop/func/stocks/progress-update/kospi',
        json={"percent": 100, "done": True},
        timeout=5
    )
except Exception as e:
    # logging.warning(f"progress-update 요청 실패: {e}")
    print(f"progress-update 요청 실패-k2: {e}")
    pass  # 오류

if total_cnt > 0:
    print(f'R-squared_avg : {total_r2/total_cnt:.2f}')
    print(f'SMAPE : {total_smape/total_cnt:.2f}')
    print(f'total_cnt : {total_cnt}')
