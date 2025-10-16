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
import tensorflow as tf

# 시드 고정 테스트
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(BASE_DIR)
pickle_dir = os.path.join(BASE_DIR, 'pickle')

from utils import create_multistep_dataset, add_technical_features, create_lstm_model, drop_trading_halt_rows, \
    inverse_close_matrix_fast, inverse_close_from_Xscale_fast, prices_from_logrets, log_returns_from_prices, \
    rmse, improve, smape, nrmse, drop_sparse_columns




# 1. 데이터 수집
ticker = '000660' # 하이닉스
tickers = ['095610', '000660', '008970', '006490', '042670', '023160', '006800', '323410', '009540', '034020', '358570', '000155', '035720', '00680K', '035420', '012510']
# tickers = ['095610']

LOOK_BACK = 15
N_FUTURE = 3

for count, ticker in enumerate(tickers):
    print(f"Processing {count + 1}/{len(tickers)} : {ticker}")
    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
    data = pd.read_pickle(filepath)

    # 1. 우선 거래정지/이상치 행 제거
    data, removed_idx = drop_trading_halt_rows(data)

    # 2. 2차 생성 feature
    data = add_technical_features(data)
    # print(data.columns) # 칼럼 헤더 전체 출력

    # 3. 결측 제거
    cleaned, cols_to_drop = drop_sparse_columns(data, threshold=0.10, check_inf=True, inplace=True)
    # print("    Drop candidates:", cols_to_drop)
    data = cleaned


    # ---- 전처리: NaN/inf 제거 및 피처 선택 ----
    feature_cols = [
        '시가', '고가', '저가', '종가', 'Vol_logdiff',
        # 'RSI14', # 빼는게 성능이 덜 튀고 안정적
    ]

    # 4. 피쳐, 무한대 필터링
    cols = [c for c in feature_cols if c in data.columns]  # 순서 보존
    df = data.loc[:, cols].replace([np.inf, -np.inf], np.nan)
    X_df = df.dropna()  # X_df는 (정렬/결측처리된) 피처 데이터프레임, '종가' 컬럼 존재

    if '종가' not in cols:
        raise KeyError("'종가' 컬럼이 데이터에 없습니다.")
    idx_close = cols.index('종가')

    # 5. 스케일링, 시점 마스크
    split = int(len(X_df) * 0.75)
    scaler_X = StandardScaler().fit(X_df.iloc[:split])  # 원시 train 구간만, 중복 윈도우 때문에 같은 시점 행이 여러 번 들어가는 왜곡 방지
    X_all = scaler_X.transform(X_df)                    # 전체 변환 (누수 없음)
    print(X_all.shape)

    # 원본 종가(스케일 전)를 1D ndarray로 확보 (X_df와 같은 인덱스/정렬/필터 상태)
    # close_raw = X_df.iloc[:, idx_close].to_numpy(dtype=float)
    close_raw = X_df['종가'].to_numpy(dtype=float)
    logret = log_returns_from_prices(close_raw)  # 인접한 두 가격으로 로그수익률을 만드니까 길이는 L-1, 

    # X_all(스케일) 종가 역변환 == close_raw(원본) 일치 여부 (전구간 체크)
    # recon_close = inverse_close_from_Xscale_fast(X_all[:, idx_close], scaler_X, idx_close)
    # X_all[:, idx_close]는 NumPy 2차원 배열 X_all에서 모든 행(:), 한 개의 열(idx_close 열 인덱스)을 선택해 1차원 벡터를 뽑는 슬라이싱
    # print("max|recon_close - close_raw| =", np.max(np.abs(recon_close - close_raw)))

    # ↓ 여기서 X_all, Y_all을 '스케일된 X'로부터 만듦
    # Y_xscale: X_all에서 ‘종가’ 열(idx_close)의 미래 H-step 값들을 그대로 뽑아 만든 타깃 행렬
    X_tmp, Y_xscale, t0 = create_multistep_dataset(X_all, LOOK_BACK, N_FUTURE, idx=idx_close, return_t0=True)
    # print('X_tmp', X_tmp.shape)         # (274, 15, 5)
    # print('Y_xscale', Y_xscale.shape)   # (274, 3)
    t_end = t0 + LOOK_BACK - 1  # 윈도 끝 인덱스 (입력의 마지막 시점) >> 전체 길이 291일 경우, t_end: 14~287, 288~290

    valid = (t_end + N_FUTURE) <= (len(close_raw)-1)
    X_tmp, t0, t_end = X_tmp[valid], t0[valid], t_end[valid]

    # 각 입력 윈도우의 마지막 시점(t_end) 이후 앞으로 N_FUTURE개의 로그수익률 벡터들을 쌓아 만든 2차원 배열 >> 윈도 끝 이후 1,2,...,N_FUTURE 기간의 로그수익률
    Y_log = np.stack([logret[t: t + N_FUTURE] for t in t_end], axis=0)
    # print(Y_log.shape)    # (274, 3)

    # minN = min(len(X_tmp), len(Y_log))
    # X_tmp     = X_tmp[:minN]      # (N, LOOK_BACK, F)
    # Y_log     = Y_log[:minN]      # (N, H)
    # # Y_xscale  = Y_xscale[:minN]
    # t0        = t0[:minN]
    # t_end     = t_end[:minN]
    # # print('t_end', t_end)

    # # 시점 마스크로 분리
    # train_mask = (t0 + N_FUTURE - 1) < split
    # val_mask   = (t0 >= split)
    #
    # # (현재 사용 중인) 마스크 합계/겹침/누락 체크
    # n_tr = int(np.sum(train_mask))
    # n_va = int(np.sum(val_mask))
    # overlap = int(np.sum(train_mask & val_mask))               # 둘 다 True인 곳
    # gaps    = np.where(~(train_mask | val_mask))[0]            # 둘 다 False인 곳 (누락)
    #
    # print(f"train True = {n_tr}")
    # print(f"valid True = {n_va}")
    # print(f"합계       = {n_tr + n_va}")
    # print(f"겹침 수    = {overlap} (0이어야 정상)")
    # print(f"누락 수    = {len(gaps)} (0이어야 정상)")

    # 가격 인덱스 기준으로 분할
    t_y_end = t_end + N_FUTURE   # 타깃의 마지막 시점
    # print('t_y_end', t_y_end)    # 마지막 인덱스 17 (15~17)

    # 시점 마스크로 분리
    train_mask = (t_y_end < split)
    val_mask   = (t_y_end >= split)

    # # 동일한 검증
    # n_tr2 = int(np.sum(train_mask))
    # n_va2 = int(np.sum(val_mask))
    # overlap2 = int(np.sum(train_mask & val_mask))
    # gaps2    = np.where(~(train_mask | val_mask))[0]
    #
    # print("\n[타깃 마지막 시점 기준]")
    # print(f"train True = {n_tr2}")
    # print(f"valid True = {n_va2}")
    # print(f"합계       = {n_tr2 + n_va2}")
    # print(f"겹침 수    = {overlap2} (0이어야 정상)")
    # print(f"누락 수    = {len(gaps2)} (0이어야 정상)")

    X_train, Y_train = X_tmp[train_mask], Y_log[train_mask]
    X_val,   Y_val   = X_tmp[val_mask],   Y_log[val_mask]
    # print('X_train', X_train.shape)    # (201, 15, 5)
    # print('Y_train', Y_train.shape)    # (201, 3)
    # print('X_val', X_val.shape)        # (73, 15, 5)
    # print('Y_val', Y_val.shape)        # (73, 3)

    # ---- (y_scaler) ----
    scaler_y_log = StandardScaler().fit(Y_train)  # 로그수익률에 대해
    Y_train_s = scaler_y_log.transform(Y_train)
    Y_val_s   = scaler_y_log.transform(Y_val)
    # --------------------

    # 1안 (종가 y스케일)
    # y_price_from_xscale_tr = inverse_close_matrix_fast(Y_xscale[train_mask], scaler_X, idx_close)
    # y_price_from_xscale_va = inverse_close_matrix_fast(Y_xscale[val_mask],   scaler_X, idx_close)

    # z1_tr = X_train[:, -1, idx_close]
    # z2_tr = X_all[t_end[train_mask], idx_close]
    # print("train max|z1 - z2| =", np.max(np.abs(z1_tr - z2_tr)))
    #
    # z1_va = X_val[:, -1, idx_close]
    # z2_va = X_all[t_end[val_mask], idx_close]
    # print("valid max|z1 - z2| =", np.max(np.abs(z1_va - z2_va)))

    # 윈도 마지막 시점 정합성 체크 (학습 데이터셋 vs 원본, 스케일 공간 체크)
    assert np.allclose(X_train[:, -1, idx_close], X_all[t_end[train_mask], idx_close])
    assert np.allclose(X_val[:, -1, idx_close],   X_all[t_end[val_mask],   idx_close])

    # 기준가격(원 단위): 각 샘플의 윈도 마지막 시점 t의 종가 (원 단위)
    base_close_train = inverse_close_from_Xscale_fast(X_all[t_end[train_mask], idx_close], scaler_X, idx_close)
    base_close_val   = inverse_close_from_Xscale_fast(X_all[t_end[val_mask],   idx_close], scaler_X, idx_close)
    # print('base_close_train', base_close_train.shape)
    # print('base_close_val', base_close_val.shape)
    # print('base_close_train', base_close_train)
    # print('base_close_val', base_close_val)

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
        delta_vec = np.maximum(delta_vec, eps)  # 0 방지

        def huber_per_h(y_true, y_pred):
            err = tf.abs(y_true - y_pred)          # (N,H)
            d   = tf.constant(delta_vec, err.dtype) # (H,)
            quad = 0.5 * tf.square(err)
            lin  = d * err - 0.5 * tf.square(d)
            loss = tf.where(err <= d, quad, lin)   # (N,H), broadcasting OK
            return tf.reduce_mean(loss)

        return huber_per_h


    # **스케일된 Y(Y_train_s)**로 학습하는 변형에도 Huber를 쓰려면 별도 델타를 만들어야
    stds = Y_train.std(axis=0)
    loss_fn = make_huber_per_h(2.0 * stds)
    stds_scaled = Y_train_s.std(axis=0).astype("float32")
    loss_fn_scaled = make_huber_per_h(2.0 * stds_scaled)
    variants = {
        # "LSTM32_mse": { # 낮게 예측함
        #     "lstm_units": [32],
        #     "dropout": 0.0,
        #     "dense_units": [16],
        #     "lr": 5e-4, "loss": "mse", "delta": 1.0,
        #     "batch_size": 8
        # },
        # "LSTM64_mse": {
        #     "lstm_units": [64, 32],
        #     "dropout": 0.1,
        #     "dense_units": [32, 16],
        #     "lr": 5e-4, "loss": "mse", "delta": 1.0,
        #     "batch_size": 8
        # },
        # "LSTM32": { # 높게 예측함
        #     "lstm_units": [32],
        #     "dropout": 0.0,
        #     "dense_units": [16],
        #     "lr": 5e-4, "loss": loss_fn, "delta": 1.0,
        #     "batch_size": 8
        # },
        # "LSTM64": {
        #     "lstm_units": [64, 32],
        #     "dropout": 0.1,
        #     "dense_units": [32, 16],
        #     "lr": 5e-4, "loss": loss_fn, "delta": 1.0,
        #     "batch_size": 8
        # },
        "Y_LSTM32": {
            "lstm_units": [32],
            "dropout": 0.0,
            "dense_units": [16],
            "lr": 5e-4,
            "loss": None,
            "delta": 1.0,
            "batch_size": 8,
        },
        "Y_LSTM64": {  # 채택.. 가장 안정적이고 평균 성능이 좋음
            "lstm_units": [64, 32],
            "dropout": 0.1,
            "dense_units": [32, 16],
            "lr": 5e-4,
            "loss": None,
            "delta": 1.0,
            "batch_size": 8,
        },
    }

    results = {}

    for name, cfg in variants.items():
        print(f"[{name}] training...")
        model = create_lstm_model(
            input_shape, N_FUTURE,
            lstm_units=cfg["lstm_units"],
            dropout=cfg["dropout"],
            dense_units=cfg["dense_units"],
            lr=cfg["lr"], delta=cfg["delta"],
            loss=cfg["loss"],
        )
        if name.startswith("Y"): # Y가 붙으면 로그수익률을 y-스케일링 했음
            history = model.fit(
                X_train, Y_train_s,  # (y_scaler)
                batch_size=cfg["batch_size"], epochs=200, verbose=0, shuffle=False,
                validation_data=(X_val, Y_val_s),  # (y_scaler)
                callbacks=[early_stop, rlrop]
            )
        else:
            history = model.fit(
                X_train, Y_train,
                batch_size=cfg["batch_size"], epochs=200, verbose=0, shuffle=False,
                validation_data=(X_val, Y_val),
                callbacks=[early_stop, rlrop]
            )


        # 예측(로그수익률), 예측 결과는 'y-스케일된 로그' or '로그'
        pred_tr_log = model.predict(X_train, verbose=0)
        pred_va_log = model.predict(X_val,   verbose=0)

        if name.startswith("Y"):
            # ---- (y_scaler) ----
            # 역표준화하여 '원-로그수익률'로 복원
            pred_tr_log = scaler_y_log.inverse_transform(pred_tr_log)
            pred_va_log = scaler_y_log.inverse_transform(pred_va_log)
            # ---------------------

        # 로그수익률 → 가격 복원
        price_pred_tr = prices_from_logrets(base_close_train, pred_tr_log)
        price_pred_va = prices_from_logrets(base_close_val,   pred_va_log)
        # 정답(로그수익률 Y)도 가격으로 변환
        price_true_tr = prices_from_logrets(base_close_train, Y_train)
        price_true_va = prices_from_logrets(base_close_val,   Y_val)


        # 두 방식이 거의 같은지 수치 확인, 0에 가까우면 동일하다고 판단(e-10으로 끝나면 사실상 0), 1이상 이상 차이가 나면 문제
        # print("[TRAIN] RMSE(true_xscale_vs_true_log) =", rmse(y_price_from_xscale_tr, price_true_tr))
        # print("[VALID] RMSE(true_xscale_vs_true_log) =", rmse(y_price_from_xscale_va, price_true_va))

        # 윈도 끝 시점 역변환 후 일치(RMSE), 정렬 어긋난는지 판단, 0 이면 이상없음; (1)+(2)를 합친 종합 체크
        # print("RMSE(C_t_from_X, C_t_from_raw) =", rmse(
        #     inverse_close_from_Xscale_fast(X_tmp[:, -1, idx_close], scaler_X, idx_close),
        #     close_raw[t_end]
        # ))
        assert np.allclose(
            inverse_close_from_Xscale_fast(X_tmp[:, -1, idx_close], scaler_X, idx_close),
            close_raw[t_end]
        )

        # (y_scaler) 역표준화 후 수치 확인 (e-10 > 사실상 0)
        # print("RMSE(true_xscale_vs_true_log) =",
        #       rmse(inverse_close_matrix(Y_xscale[val_mask], scaler_X, X_tmp.shape[2], idx_close),
        #            prices_from_logrets(base_close_val, Y_val)))

        best_val = min(history.history.get('val_loss', [float('inf')]))

        is_scaled = name.startswith("Y")
        results[name] = {
            "model": model,
            "is_scaled": is_scaled,
            "train_pred_price": price_pred_tr,
            "val_pred_price":   price_pred_va,
            "train_true_price": price_true_tr,
            "val_true_price":   price_true_va,
            "best_val": best_val,
            "epochs_trained": len(history.history['loss']),
        }
        # print(f" -> best val_loss: {best_val:.6f} / epochs: {results[name]['epochs_trained']}")


    # ===== 최고 모델 선택 =====
    best_name = min(results.keys(), key=lambda k: results[k]["best_val"])
    # print(f"\n[Best] {best_name} (val_loss={results[best_name]['best_val']:.6f})")

    # ===== 나이브 베이스라인(가격 고정: C_{t+h}=C_t) =====
    naive_val = np.repeat(base_close_val[:, None], N_FUTURE, axis=1)

    print("\n=== Validation Metrics (원 단위) ===")
    for name, pack in results.items():
        y_true_p = pack["val_true_price"]  # (N,H)
        y_pred_p = pack["val_pred_price"]  # (N,H)

        # ALL (가격 기준)
        r_all = rmse(y_true_p.reshape(-1), y_pred_p.reshape(-1))
        r_nv  = rmse(y_true_p.reshape(-1), naive_val.reshape(-1))
        s_all = smape(y_true_p.reshape(-1), y_pred_p.reshape(-1))
        nr_all = nrmse(y_true_p.reshape(-1), y_pred_p.reshape(-1))

        print(f"[{name}] ALL : RMSE={r_all:.2f} | naive={r_nv:.2f} | 개선={improve(r_all, r_nv):.1f}% | sMAPE={s_all:.2f} | nRMSE={nr_all:.4f}")

        # h별 (가격 기준)
        for h in range(N_FUTURE):
            r_h  = rmse(y_true_p[:,h], y_pred_p[:,h])
            r_hn = rmse(y_true_p[:,h], naive_val[:,h])
            s_h  = smape(y_true_p[:,h], y_pred_p[:,h])
            nr_h = nrmse(y_true_p[:,h], y_pred_p[:,h])
            print(f"    h={h+1}: RMSE={r_h:.2f} | naive={r_hn:.2f} | ratio={r_h/r_hn:.3f} | 개선={improve(r_h, r_hn):.1f}% | sMAPE={s_h:.2f} | nRMSE={nr_h:.4f}")

        # (선택) 로그수익률 기준도 하나 출력
        pred_va_raw = results[name]["model"].predict(X_val, verbose=0)
        if results[name]["is_scaled"]:
            yhat_log = scaler_y_log.inverse_transform(pred_va_raw)     # 스케일 → 원-로그수익률
        else:
            yhat_log = pred_va_raw                                     # 이미 원-로그수익률
        ytrue_log     = Y_val                                          # 원본(비표준화)
        ynaive_log    = np.zeros_like(ytrue_log)                       # 0수익률 베이스라인

        rlog_m = rmse(ytrue_log.reshape(-1), yhat_log.reshape(-1))
        rlog_n = rmse(ytrue_log.reshape(-1), ynaive_log.reshape(-1))
        print(f"[log-returns] RMSE model={rlog_m:.6f} | naive={rlog_n:.6f} | 개선={improve(rlog_m, rlog_n):.1f}%")

    # ===== 그래프 비교 (h=1 기본) =====
    import matplotlib.pyplot as plt

    h = 0  # 0->t+1, 1->t+2, 2->t+3
    y_train_price = results[next(iter(results))]["train_true_price"]   # 아무 모델에서 true만 재사용
    y_val_price   = results[next(iter(results))]["val_true_price"]
    offset = len(y_train_price)

    plt.figure(figsize=(11,5))
    plt.plot(range(0, offset),                       y_train_price[:, h], label=f'Actual Train (h={h+1})', linewidth=1)
    plt.plot(range(offset, offset+len(y_val_price)), y_val_price[:,  h], label=f'Actual Val   (h={h+1})', linewidth=1, alpha=0.8)

    for name, pack in results.items():
        tp = pack["train_pred_price"]; vp = pack["val_pred_price"]
        plt.plot(range(0, offset),                       tp[:,h], label=f'{name} Pred Train')
        plt.plot(range(offset, offset+len(y_val_price)), vp[:,h], label=f'{name} Pred Val')

    plt.title(f'Comparison — LSTM units 32/64/128 (log-returns), look_back={LOOK_BACK}, h={h+1}')
    plt.xlabel('Sample index'); plt.ylabel('Price (KRW)'); plt.legend(); plt.tight_layout()
    plt.savefig(f'{ticker}_{best_name}.png', dpi=150)
    # plt.show()

"""
Keras LSTM은 입력 shape을 (batch, timesteps, features) 로 받는다
X가 바로 LSTM의 입력에 맞는 3D 텐서

"""
