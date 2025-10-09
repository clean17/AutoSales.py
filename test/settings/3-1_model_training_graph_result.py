# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found / Skipping registering GPU devices... 안나오게
import os
# 1) GPU 완전 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2) C++ 백엔드 로그 레벨 낮추기 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# matplotlib.use("Agg") # plt.show() 에서 사용안함
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)
pickle_dir = os.path.join(BASE_DIR, 'pickle')

from utils import create_multistep_dataset, add_technical_features, create_lstm_model, drop_trading_halt_rows, \
    inverse_close_from_Xscale_fast, inverse_close_matrix_fast, rmse, improve, smape, nrmse




# 1. 데이터 수집
ticker = '000660' # 하이닉스
# tickers = ['000660', '008970', '006490', '042670', '023160', '006800', '323410', '009540', '034020', '358570', '000155', '035720', '00680K', '035420', '012510']
tickers = ['000660']

LOOK_BACK = 15
N_FUTURE = 3

for count, ticker in enumerate(tickers):
    print(f"Processing {count + 1}/{len(tickers)} : {ticker}")
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    data = pd.read_pickle(filepath)

    # 1-1. 우선 거래정지/이상치 행 제거
    data, removed_idx = drop_trading_halt_rows(data)
    if len(removed_idx) > 0:
        print(f"거래정지/이상치로 제거된 날짜 수: {len(removed_idx)}")

    # 2. 2차 생성 feature
    data = add_technical_features(data)
    # print(data.columns) # 칼럼 헤더 전체 출력

    # 3. 결측 제거
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


    # ---- 전처리: NaN/inf 제거 및 피처 선택 ----
    feature_cols = [
        '시가', '고가', '저가', '종가', 'Vol_logdiff',
        # 'RSI14',
    ]

    # 4. 피쳐, 무한대 필터링
    cols = [c for c in feature_cols if c in data.columns]  # 순서 보존
    df = data.loc[:, cols].replace([np.inf, -np.inf], np.nan)
    X_df = df.dropna() # X_df는 (정렬/결측처리된) 피처 데이터프레임, '종가' 컬럼 존재

    idx_close = feature_cols.index('종가')

    # 5. 스케일링, 시점 마스크
    split = int(len(X_df) * 0.85)
    scaler_X = StandardScaler().fit(X_df.iloc[:split])  # 원시 train 구간만, 중복 윈도우 때문에 같은 시점 행이 여러 번 들어가는 왜곡 방지
    X_all = scaler_X.transform(X_df)                    # 전체 변환 (누수 없음)


    # ↓ 여기서 X_all, Y_all을 '스케일된 X'로부터 만듦
    X_tmp, Y_xscale, t0 = create_multistep_dataset(X_all, LOOK_BACK, N_FUTURE, idx=idx_close, return_t0=True)
    t_end = t0 + LOOK_BACK - 1        # 윈도 끝 인덱스 (입력의 마지막 시점)
    t_y_end = t_end + (N_FUTURE - 1)  # 타깃의 마지막 시점

    # 시점 마스크로 분리
    train_mask = (t_y_end < split)
    val_mask   = (t_y_end >= split)

    X_train, Y_train = X_tmp[train_mask], Y_xscale[train_mask]
    X_val,   Y_val   = X_tmp[val_mask],   Y_xscale[val_mask]

    # ---- y 스케일링: Train으로만 fit ---- (타깃이 수익률이면 생략 가능)
    scaler_y = StandardScaler().fit(Y_train)
    y_train_scaled = scaler_y.transform(Y_train)
    y_test_scaled  = scaler_y.transform(Y_val)


    # ===== 공통 설정 =====
    input_shape = (X_train.shape[1], X_train.shape[2])
    early_stop  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlrop       = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # ===== 권장 실험 구성 =====
    # - LSTM32: 가벼운 모델 (작은 데이터/빠른 실험)
    # - LSTM64: 밸런스형 (권장 기본)
    # - LSTM128: 깊고 넓게 (데이터/패턴이 충분할 때만)
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

    variants = {
        "LSTM": {
            "lstm_units": [32],
            "dropout": None,
            "dense_units": [16],
            "lr": 5e-4, "loss": loss_fn, "delta": 1.0
        },
    }

    results = {}
    n_features = X_train.shape[2]

    for name, cfg in variants.items():
        print(f"[{name}] training...")
        model_v = create_lstm_model(
            input_shape, N_FUTURE,
            lstm_units=cfg["lstm_units"],
            dropout=cfg["dropout"],
            dense_units=cfg["dense_units"],
            lr=cfg["lr"], delta=cfg["delta"],
        )
        history = model_v.fit(
            X_train, y_train_scaled,
            batch_size=16, epochs=200, verbose=0, shuffle=False,
            validation_data=(X_val, y_test_scaled),
            callbacks=[early_stop, rlrop]
        )

        # 예측 (y-스케일)
        tr_pred_s = model_v.predict(X_train, verbose=0)
        va_pred_s = model_v.predict(X_val,   verbose=0)

        # y-스케일 -> X-스케일
        tr_pred_x = scaler_y.inverse_transform(tr_pred_s)
        va_pred_x = scaler_y.inverse_transform(va_pred_s)

        # X-스케일 -> 원 단위(종가만 역변환)
        tr_pred_p = inverse_close_matrix_fast(tr_pred_x, scaler_X, idx_close)  # (N_tr, H)
        va_pred_p = inverse_close_matrix_fast(va_pred_x, scaler_X, idx_close)  # (N_va, H)

        best_val = min(history.history.get('val_loss', [float('inf')]))
        results[name] = {
            "model": model_v,
            "train_pred_price": tr_pred_p,
            "val_pred_price":   va_pred_p,
            "best_val": best_val,
            "epochs_trained": len(history.history['loss']),
        }
        # print(f" -> best val_loss: {best_val:.6f} / epochs: {results[name]['epochs_trained']}")


    # ===== 최고 모델 선택 =====
    best_name = min(results.keys(), key=lambda k: results[k]["best_val"])
    # print(f"\n[Best] {best_name} (val_loss={results[best_name]['best_val']:.6f})")

    # 실제 정답(원 단위)도 준비 (이미 위에서 계산했으면 재사용)
    y_train_price = inverse_close_matrix_fast(Y_train, scaler_X, idx_close)
    y_val_price   = inverse_close_matrix_fast(Y_val,   scaler_X, idx_close)

    # ====== 평가 세트용 앵커(기준가격 C_t) ======
    n_features = X_train.shape[2]
    base_close_val_scaled = X_val[:, -1, idx_close]
    base_close_val = inverse_close_from_Xscale_fast(base_close_val_scaled, scaler_X, idx_close)

    # ====== 나이브 베이스라인 (C_{t+h} = C_t) ======
    naive_val = np.repeat(base_close_val[:, None], N_FUTURE, axis=1)

    print("\n=== Validation Metrics (원 단위) ===")
    for name, pack in results.items():
        y_true_p = y_val_price                     # (N,H) — 정답 가격
        y_pred_p = pack["val_pred_price"]          # (N,H) — 예측 가격

        # ------ ALL (가격 기준) ------
        r_all = rmse(y_true_p.reshape(-1), y_pred_p.reshape(-1))
        r_nv  = rmse(y_true_p.reshape(-1), naive_val.reshape(-1))
        s_all = smape(y_true_p.reshape(-1), y_pred_p.reshape(-1))
        nr_all = nrmse(y_true_p.reshape(-1), y_pred_p.reshape(-1))

        # R^2 (참고용)
        ybar = np.mean(y_true_p)
        sst  = np.sum((y_true_p - ybar)**2)
        sse  = np.sum((y_true_p - y_pred_p)**2)
        r2   = 1.0 - (sse / (sst + 1e-12))

        print(f"[{name}] ALL : RMSE={r_all:.2f} | naive={r_nv:.2f} | 개선={improve(r_all, r_nv):.1f}% "
              f"| sMAPE={s_all:.2f} | nRMSE={nr_all:.4f} | R^2={r2:.4f}")

        # ------ h별 (가격 기준) ------
        for h in range(N_FUTURE):
            r_h  = rmse(y_true_p[:,h], y_pred_p[:,h])
            r_hn = rmse(y_true_p[:,h], naive_val[:,h])
            s_h  = smape(y_true_p[:,h], y_pred_p[:,h])
            nr_h = nrmse(y_true_p[:,h], y_pred_p[:,h])

            # 방향성 적중률 (C_{t+h} - C_t)의 부호 일치 비율
            true_dir = np.sign(y_true_p[:,h] - base_close_val)
            pred_dir = np.sign(y_pred_p[:,h] - base_close_val)
            hit_rate = np.mean(true_dir == pred_dir)

            print(f"    h={h+1}: RMSE={r_h:.2f} | naive={r_hn:.2f} | ratio={r_h/r_hn:.3f} "
                  f"| 개선={improve(r_h, r_hn):.1f}% | sMAPE={s_h:.2f} | nRMSE={nr_h:.4f} "
                  f"| HitRate={hit_rate:.3f}")

        # ------ (선택) 로그수익률 기준 평가 ------
        # 가격 예측을 로그수익률로 변환해, 수익률 공간에서도 비교
        pred_log = np.log(np.clip(y_pred_p, 1e-12, None) / base_close_val[:, None])
        true_log = np.log(y_true_p / base_close_val[:, None])
        rlog_m = rmse(true_log.reshape(-1), pred_log.reshape(-1))
        rlog_n = rmse(true_log.reshape(-1), np.zeros_like(true_log).reshape(-1))  # 나이브(0수익률)
        print(f"    [log-returns] RMSE model={rlog_m:.6f} | naive={rlog_n:.6f} | 개선={improve(rlog_m, rlog_n):.1f}%")



    h = 0  # 0->t+1, 1->t+2, 2->t+3
    offset = len(y_train_price)

    plt.figure(figsize=(11,5))
    plt.plot(range(0, offset),                       y_train_price[:, h], label=f'Actual Train (h={h+1})', linewidth=1)
    plt.plot(range(offset, offset+len(y_val_price)), y_val_price[:,  h], label=f'Actual Val   (h={h+1})', linewidth=1, alpha=0.8)

    for name in results:
        tp = results[name]["train_pred_price"]
        vp = results[name]["val_pred_price"]
        plt.plot(range(0, offset),                       tp[:,h], label=f'{name} Pred Train')
        plt.plot(range(offset, offset+len(y_val_price)), vp[:,h], label=f'{name} Pred Val')

    plt.title(f'Comparison — LSTM units 32/64/128, look_back={LOOK_BACK}, h={h+1}')
    plt.xlabel('Sample index'); plt.ylabel('Price (KRW)'); plt.legend(); plt.tight_layout()
    plt.savefig(f'{ticker}_{best_name}.png', dpi=150)
    # plt.show()





"""
Keras LSTM은 입력 shape을 (batch, timesteps, features) 로 받는다
X가 바로 LSTM의 입력에 맞는 3D 텐서

"""