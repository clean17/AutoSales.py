# Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found / Skipping registering GPU devices... 안나오게
import os
# 1) GPU 완전 비활성화
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2) C++ 백엔드 로그 레벨 낮추기 (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime, timedelta
from pykrx import stock
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import requests
import yfinance as yf
import os
import re
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, precision_recall_curve, \
    f1_score
import warnings
import pandas_market_calendars as mcal
from pandas.api.types import is_numeric_dtype

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ----- 그래프 렌더링에 사용되는 상수들 -----
RED = "#E83030"
BLUE = "#195DE6"
ORANGE = '#FF9F1C'
GREEN = '#2E7D32'
GRAY1 = '#8E8E8E'
GRAY2 = '#9AA0A6'
today = datetime.today().strftime('%Y%m%d')

# ----- 공통 유틸 -----
def _col(df, ko: str, en: str):
    """한국/영문 칼럼 자동매핑: ko가 있으면 ko, 없으면 en을 반환"""
    if ko in df.columns: return ko
    return en

# 지수 가중 이동평균(EMA) 계열을 pandas로 계산하는 래퍼
def _ema(s: pd.Series, span: int) -> pd.Series:
    """
    EMA: α=2/(span+1) → 최신값 가중이 더 큼 → 빠른 반응.
    EMA: 단기 추세/크로스 등 빠른 신호에 유리.
    """
    return s.ewm(span=span, adjust=False).mean()

def _rma(s: pd.Series, length: int) -> pd.Series:
    """
    RMA(Wilder): α=1/length → 더 완만한 평활 → 지연(lag) 더 큼.
    RMA: RSI의 평균(상승/하락) 계산 등 Wilder 방식 필요할 때 정확한 매칭.
    """
    alpha = 1 / length
    return s.ewm(alpha=alpha, adjust=False).mean()




def rmse(a,b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a-b)**2)))

def improve(m, n, eps=1e-8):
    n = max(float(n), eps)     # 분모 하한
    return (1.0 - m/n) * 100.0

# (N, H) 형태로
def _to_2d(a):
    a = np.array(a)
    return a if a.ndim == 2 else a.reshape(-1, 1)

def _smape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def smape(y_true, y_pred, eps=1e-12):
    """
    y_true, y_pred: (N, H) 또는 (N,) 형태도 허용
    결측/무한값 제외 후 평균
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)).clip(min=eps)
    smape = 200.0 * num / den  # %
    mask = np.isfinite(smape)
    if not np.any(mask):
        return np.nan
    return np.mean(smape[mask])

def _mape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return 100.0 * np.mean(np.abs((y_true - y_pred) / denom))

def _mase(y_true, y_pred, y_insample, m: int = 1, eps: float = 1e-12):
    """
    y_true, y_pred : 평가 구간(검증/테스트)의 실제값과 예측값 (1D 또는 2D 가능)
    y_insample     : 분모 계산용 '훈련(in-sample) 구간'의 실제값(1D)
    m              : 계절 주기(비계절은 1)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_insample = np.asarray(y_insample).reshape(-1)

    # 분모: in-sample의 계절 차분 기반 MAE
    if len(y_insample) <= m:
        return np.nan
    denom = np.mean(np.abs(y_insample[m:] - y_insample[:-m]))
    denom = max(denom, eps)

    return np.mean(np.abs(y_true - y_pred)) / denom

def nrmse(y, yhat, eps=1e-8):
    return float(np.sqrt(np.mean((np.asarray(yhat)-np.asarray(y))**2)) / (np.mean(np.abs(y))+eps))


# --- 2) 유틸: naive 생성 ---
def make_naive_preds(y_hist_end, horizon, mode="price"):
    """
    y_hist_end: (N,) 각 윈도우의 마지막 실제값 y[t-1]; 각 검증(또는 학습) 윈도우의 마지막 입력 종가 y[t-1]들을 모은 1차원 벡터 (N,)
    horizon: H
    mode:
      - "price": persistence (내일도 오늘과 동일) -> 모든 h에 y[t-1] 반복
      - "return": 0 수익률(변화없음) 가정 -> 모든 h에 0
    return: (N, H)
    """
    y_hist_end = np.asarray(y_hist_end, dtype=float).reshape(-1, 1)
    if mode == "price":
        return np.repeat(y_hist_end, repeats=horizon, axis=1)
    elif mode == "return":
        return np.zeros((y_hist_end.shape[0], horizon), dtype=float)
    else:
        raise ValueError("mode must be 'price' or 'return'")

# --- 3) 유틸: hit-rate (방향 적중률) ---
def hit_rate(y_true, y_pred, *, y_base=None, use_horizon=1, space="price"):
    """
    y_true, y_pred: (N, H)
    y_base: (N,) 가격공간에서 방향을 볼 때 기준값(y[t-1]) 필요
    use_horizon: 정수 h(1-based) 또는 "avg" (모든 h 평균)
    space:
      - "price": 방향 = sign( y_true[:,h-1]-y_base ) vs sign( y_pred[:,h-1]-y_base )
      - "return": 방향 = sign( y_true[:,h-1] ) vs sign( y_pred[:,h-1] )
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    N, H = y_true.shape

    def _one_h(h_idx):
        if space == "price":
            if y_base is None:
                raise ValueError("price space hit-rate needs y_base (y_hist_end).")
            tgt = np.sign(y_true[:, h_idx] - y_base)
            est = np.sign(y_pred[:, h_idx] - y_base)
        else:  # return space
            tgt = np.sign(y_true[:, h_idx])
            est = np.sign(y_pred[:, h_idx])
        mask = np.isfinite(tgt) & np.isfinite(est)
        if not np.any(mask):
            return np.nan
        return np.mean((tgt[mask] == est[mask]).astype(float))

    if use_horizon == "avg":
        vals = []
        for h in range(H):
            vals.append(_one_h(h))
        vals = np.array(vals, dtype=float)
        return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
    else:
        h_idx = int(use_horizon) - 1
        return _one_h(h_idx)



# ----- 기본 지표 -----
def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs = _rma(up, length) / _rma(down, length)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr

def compute_atr(high, low, close, length=14):
    tr = _true_range(high, low, close)
    return _rma(tr, length)

def compute_di_adx(high, low, close, length=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=low.index)
    atr = compute_atr(high, low, close, length)
    plus_di = 100 * _rma(plus_dm, length) / atr
    minus_di = 100 * _rma(minus_dm, length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _rma(dx, length)
    return plus_di, minus_di, adx

def compute_cci(high, low, close, length=14):
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length, min_periods=length).mean()
    md = (tp - sma).abs().rolling(length, min_periods=length).mean()
    return (tp - sma) / (0.015 * md)

def compute_ultimate_osc(high, low, close, p1=7, p2=14, p3=28):
    prev_close = close.shift(1)
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
    def avg(n):
        return (bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum())
    uo = 100 * (4*avg(p1) + 2*avg(p2) + avg(p3)) / 7.0
    return uo

def compute_roc(close, length=12, pct=True):
    r = close.pct_change(length) if pct else close.diff(length)
    return r * 100 if pct else r




# 거래정지/이상치 행 제거
def drop_trading_halt_rows(data: pd.DataFrame):
    """
    변경사항:
      - 거래정지(거래량 <= 0)인 행은 제거하지 않음
      - 그 날의 시가/고가/종가는 '전일 종가'로 대체
      - 나머지 이상치(결측, 음수/0 가격, high<low 등)는 기존 규칙대로 제거
    반환:
      - 정제된 DataFrame, 제거된 인덱스 목록(기존과 동일 포맷)
    """
    d = data.copy()
    d = d.replace([np.inf, -np.inf], np.nan)

    # 한국/영문 칼럼 자동 식별
    col_o = _col(d, '시가',   'Open')
    col_h = _col(d, '고가',   'High')
    col_l = _col(d, '저가',   'Low')
    col_c = _col(d, '종가',   'Close')
    col_v = _col(d, '거래량', 'Volume')

    # 거래정지 마스크(거래량 <= 0 또는 결측을 0으로 간주)
    halt = d[col_v].fillna(0) <= 0

    # 전일 종가
    prev_close = d[col_c].shift(1)

    # 거래정지 날: 시가/고가/종가를 전일 종가로 대체(전일 종가가 NaN이면 그대로 NaN 유지)
    for col in (col_o, col_h, col_c):
        d.loc[halt, col] = prev_close.loc[halt]

    # (선택) 저가가 비정상/결측이면 전일 종가로 보정
    # - 저가가 없음, 0/음수, 혹은 저가 > 고가가 되어버린 경우
    fix_low = (
            halt &
            (
                    d[col_l].isna() |
                    (d[col_l] <= 0) |
                    (d[col_h].notna() & d[col_l].notna() & (d[col_l] > d[col_h]))
            )
    )
    d.loc[fix_low, col_l] = prev_close.loc[fix_low]

    # 유효성 체크 (거래량>0 조건은 제거: 거래정지 행을 살려야 하므로)
    notna = d[[col_o, col_h, col_l, col_c, col_v]].notna().all(axis=1)
    positive_price = d[[col_o, col_h, col_l, col_c]].gt(0).all(axis=1)
    hl_ok = d[col_h] >= d[col_l]

    valid = notna & positive_price & hl_ok
    removed_idx = d.index[~valid]

    return d.loc[valid].copy(), removed_idx


def get_safe_ticker_list(market="KOSPI"):
    def fetch_tickers_for_date(date):
        try:
            tickers = stock.get_market_ticker_list(market=market, date=date)
            # 데이터가 비어 있다면 예외를 발생시킴
            if not tickers:
                raise ValueError("Ticker list is empty")
            return tickers
        except (IndexError, ValueError) as e:
            return []

    # 현재 날짜로 시도
    today = datetime.now().strftime("%Y%m%d")
    tickers = fetch_tickers_for_date(today)

    # 첫 번째 시도가 실패한 경우 과거 날짜로 반복 시도
    if not tickers:
        print("데이터가 비어 있습니다. 가장 가까운 영업일로 재시도합니다.")
        for days_back in range(1, 7):
            previous_day = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            tickers = fetch_tickers_for_date(previous_day)
            if tickers:  # 성공적으로 데이터를 가져오면 반환
                return tickers

        print("영업일 데이터를 찾을 수 없습니다.")
        return []

    return tickers

# pykrx에 종목 리스트를 요청; 과도한 요청 시 IP차단 당한다
def get_kor_ticker_list_by_pykrx():
    tickers_kospi = get_safe_ticker_list(market="KOSPI")
    tickers_kosdaq = get_safe_ticker_list(market="KOSDAQ")
    tickers = tickers_kospi + tickers_kosdaq
    return tickers

# DB에 저장해놨던 종목 리스트 조회
def get_kor_ticker_list():
    # tickers_kospi = get_safe_ticker_list(market="KOSPI")
    # tickers_kosdaq = get_safe_ticker_list(market="KOSDAQ")
    # tickers = tickers_kospi + tickers_kosdaq

    url = "https://chickchick.shop/stocks/kor"
    res = requests.get(url)
    try:
        data = res.json()
    except ValueError:  # JSONDecodeError도 ValueError 하위
        data = {}
    tickers = [item["stock_code"] for item in data if "stock_code" in item]
    return tickers

def get_kor_ticker_dict_list():
    url = "https://chickchick.shop/stocks/kor"
    res = requests.get(url)
    try:
        data = res.json()
    except ValueError:  # JSONDecodeError도 ValueError 하위
        data = {}
    return {
        item["stock_code"]: item["stock_name"]
        for item in data
        if "stock_code" in item and "stock_name" in item
    }

def get_kor_summary_ticker_dict_list():
    days_ago_14 = (datetime.today() - timedelta(days=14)).strftime('%Y%m%d')
    res = requests.post(
        'https://chickchick.shop/stocks/interest/data/fire',
        json={
            "date": str(days_ago_14)
        },
        timeout=5
    )
    try:
        data = res.json()
    except ValueError:  # JSONDecodeError도 ValueError 하위
        data = {}
    return {
        item["stock_code"]: item["stock_name"]
        for item in data
        if "stock_code" in item and "stock_name" in item
    }

def get_favorite_ticker_dict_list():
    days_ago_14 = (datetime.today() - timedelta(days=14)).strftime('%Y%m%d')
    res = requests.post(
        'https://chickchick.shop/stocks/interest/data/favorite/schedule',
        json={
            "date": str(days_ago_14)
        },
        timeout=5
    )
    try:
        data = res.json()
    except ValueError:  # JSONDecodeError도 ValueError 하위
        data = {}
    return {
        item["stock_code"]: item["stock_name"]
        for item in data
        if "stock_code" in item and "stock_name" in item
    }

def get_kor_interest_ticker_dick_list():
    url = "https://chickchick.shop/stocks/interest/data/today"
    payload = {
        "date": today,
    }

    res = requests.post(url, json=payload)
    try:
        data = res.json()
    except ValueError:  # JSONDecodeError도 ValueError 하위
        data = {}
    return {
        item["stock_code"]: item["stock_name"]
        for item in data
        if "stock_code" in item and "stock_name" in item
    }



def get_nasdaq_symbols():
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&exchange=NASDAQ&limit=0"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Origin": "https://www.nasdaq.com"
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # 종목 목록 추출
    rows = data.get("data", {}).get("table", {}).get("rows", [])
    symbols = [row["symbol"] for row in rows if "symbol" in row]

    return symbols

# S&P 500 종목을 가져오는 함수
def get_sp500_tickers():
    sp500_constituents = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]  # Wikipedia에서 종목 가져오기
    sp500_tickers = sp500_constituents['Symbol'].tolist()  # 티커 리스트 추출
    return sp500_tickers

# 나스닥 100 종목을 가져오는 함수
def get_nasdaq100_tickers():
    nasdaq100_constituents = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]  # Wikipedia에서 종목 가져오기
    nasdaq100_tickers = nasdaq100_constituents['Ticker'].tolist()  # 티커 리스트 추출
    return nasdaq100_tickers

def get_russell1000_tickers():
    url = "https://en.wikipedia.org/wiki/Russell_1000_Index"
    dfs = pd.read_html(url)

    # 티커 테이블 찾기: 2개 이상 컬럼 포함 + Symbol/Ticker 포함
    target_df = None
    for df in dfs:
        cols = [str(c) for c in df.columns]
        if len(cols) >= 2 and any('Symbol' in c or 'Ticker' in c for c in cols):
            target_df = df.copy()
            target_df.columns = cols
            break

    if target_df is None:
        raise ValueError(f"러셀1000 티커 테이블을 찾을 수 없습니다. 페이지에서 테이블 개수: {len(dfs)}")

    # 티커 컬럼 자동 감지
    ticker_col = next((c for c in target_df.columns if 'Symbol' in c or 'Ticker' in c), None)
    tickers = target_df[ticker_col].tolist()
    return tickers



# 주식 데이터(시가, 고가, 저가, 종가, 거래량)와 재무 데이터(PER)를 가져온다 > pandas.DataFrame 객체
def fetch_stock_data(ticker, fromdate, todate):
    ohlcv = stock.get_market_ohlcv_by_date(fromdate=fromdate, todate=todate, ticker=ticker)
    fundamental = stock.get_market_fundamental_by_date(fromdate, todate, ticker)

    # 결측치 처리 (PER, PBR)
    for col in ['PER', 'PBR']:
        if col not in fundamental.columns:
            fundamental[col] = 0
        else:
            fundamental[col] = fundamental[col].fillna(0)

    # 두 컬럼 모두 DataFrame으로 합치기
    data = pd.concat([ohlcv, fundamental[['PER', 'PBR']]], axis=1)
    return data

# 미국 주식 데이터를 가져오는 함수
def fetch_stock_data_us(ticker, fromdate, todate):
    try:
        stock_data = yf.download(
            ticker,
            start=fromdate,
            end=todate,
            auto_adjust=True,
            progress=False,
            threads=False,   # 가끔 안정성 ↑
        )
    except Exception as e:
        print("download(start/end) failed:", e)

    if stock_data is None or stock_data.empty:
        return pd.DataFrame()

    # 컬럼 구조 확인
#     print("[DEBUG] stock_data.columns:", stock_data.columns)

    # MultiIndex 컬럼일 경우 평탄화
    if isinstance(stock_data.columns, pd.MultiIndex):
        # 어떤 레벨인지 확인 후 droplevel
        if ticker in stock_data.columns.get_level_values(1):
            stock_data.columns = stock_data.columns.droplevel(1)
        elif ticker in stock_data.columns.get_level_values(0):
            stock_data.columns = stock_data.columns.droplevel(0)

    # PER/PBR 정보 try-except로 예외처리
    per_value, pbr_value = 0, 0
    try:
        stock_info = yf.Ticker(ticker).info
        per_value = stock_info.get('trailingPE', 0)
        pbr_value = stock_info.get('priceToBook', 0)
    except Exception as e:
        print(f"[{ticker}] info 조회 오류: {e}")

    stock_data['PER'] = per_value
    stock_data['PBR'] = pbr_value

    # 이제 컬럼명만 남겼으니 정상적으로 슬라이싱 가능
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'PER', 'PBR']].fillna(0)
    return stock_data




# def create_dataset(dataset, look_back=30, idx=3):
#     X, Y = [], []
#     if len(dataset) < look_back:
#         return np.array(X), np.array(Y)  # 빈 배열 반환
#     for i in range(len(dataset) - look_back):
#         X.append(dataset[i:i+look_back, :])
#         Y.append(dataset[i+look_back, idx])  # 종가(Close) 예측
#     return np.array(X), np.array(Y)


# 시계열 데이터에서 딥러닝 모델 학습용 X, Y(입력, 정답) 세트를 만드는 전형적인 패턴
'''
- `dataset`: 2차원 numpy 배열 (shape: [전체 시점, 피처 수])
- `look_back`: 입력 구간 길이 (예: 15 → 과거 15일 데이터 사용)
- `n_future`: 예측할 미래 구간 길이 (예: 3 → 미래 3일 연속 예측)
- `X.shape`: [샘플 수, look_back, 피처 수]
- `Y.shape`: [샘플 수, n_future]
'''
def create_multistep_dataset(dataset, look_back, n_future, idx=3, return_t0=False):
    ds = np.asarray(dataset)
    if ds.ndim == 1:
        ds = ds.reshape(-1, 1)
    X, Y = [], []
    n = len(ds)
    t0_list = []    # 인덱스가 저장된 리스트
    for i in range(n - look_back - n_future + 1):
        X.append(ds[i:i+look_back])
        Y.append(ds[i+look_back:i+look_back+n_future, idx])
        if return_t0:
            t0_list.append(i)
    if return_t0:
        return np.array(X), np.array(Y), np.array(t0_list)
    return np.array(X), np.array(Y)


'''
model_32 = create_lstm_model(input_shape, n_future, lstm_units=[32,16], dense_units=[16,8])
model_64 = create_lstm_model(input_shape, n_future, lstm_units=[64,32], dense_units=[32,16])
model_128 = create_lstm_model(input_shape, n_future, lstm_units=[128,64], dense_units=[64,32])
model_256 = create_lstm_model(input_shape, n_future, lstm_units=[256,128,64], dense_units=[128,64,32])

LOOK_BACK = 18   # 과거 시점 수 (timesteps)
N_FEATURES = 10  # 입력 변수(피처) 수
input_shape = (LOOK_BACK, N_FEATURES)
'''
def create_lstm_model(input_shape, n_future,
                      lstm_units=[64, 32], dropout=0.2,
                      dense_units=[16, 8],
                      lr=5e-4, loss=None, delta=1.0):
    """
    input_shape: (look_back, n_features)
    n_future: 예측할 horizon 개수
    lstm_units: LSTM 레이어별 유닛 리스트
    dropout: 각 LSTM 뒤에 붙일 Dropout 비율(리스트로 주면 각각 다르게도 가능)
    dense_units: Dense 레이어별 유닛 리스트
    lr: Adam 학습률
    delta: Huber 손실 delta
    """
    model = Sequential()
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=return_seq))

        # 드롭아웃: 공통 float 또는 레이어별 리스트 지원
        if isinstance(dropout, (float, int)) and dropout > 0:
            model.add(Dropout(float(dropout)))
        elif isinstance(dropout, (list, tuple)) and i < len(dropout) and dropout[i] > 0:
            model.add(Dropout(float(dropout[i])))

    for units in dense_units:
        model.add(Dense(units, activation='relu'))

    model.add(Dense(n_future))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    if loss is None:
        # 기본은 표준 Huber(delta=스칼라)
        loss = tf.keras.losses.Huber(delta=delta)
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    return model


def create_model(input_shape, n_future):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(input_shape)),
        LSTM(16, return_sequences=False),
        Dense(16, activation='relu'),
        Dense(n_future)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 생성된 그래프중에서 해당 날짜의 국장 종목코드만 긁어오는 함수
def extract_numbers_from_filenames(directory, isToday):
    numbers = []
    today = datetime.today().strftime('%Y%m%d')

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            if isToday:
                if not filename.startswith(today):
                    continue
            # [ 앞의 6자리 숫자 추출
            # match = re.search(r'\s(\d{6})\s*\[', filename)

            # 마지막 대괄호 안의 6자리 숫자 추출
            # match = re.search(r'\[(\d{6})\]\.png$', filename)
            match = re.search(r'\[(\d{6})\]', filename)
            if match:
                numbers.append(match.group(1))

    # 중복제거
    seen = set()
    uniq = []
    for n in numbers:
        if n not in seen:
            seen.add(n)
            uniq.append(n)

    return uniq

def extract_stock_code_from_filenames(directory):
    stock_codes = []

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # .png 떼고 strip
            name_only = os.path.splitext(filename)[0].strip()
            # 만약 ']'로 끝나면 스킵 (삼륭물산 [014970] 처럼)
            if name_only.endswith(']'):
                continue
            # ']' 뒤에 나오는 영어/숫자만 추출 (공백 포함 가능)
            match = re.search(r'\]\s*([A-Za-z0-9]+)$', name_only)
            if match:
                stock_codes.append(match.group(1))

    return stock_codes

# 네이버에서 실시간 원달러 환율을 가져오는 함수
def get_usd_krw_rate():
    url="https://m.search.naver.com/p/csearch/content/qapirender.nhn?key=calculator&pkid=141&q=%ED%99%98%EC%9C%A8&where=m&u1=keb&u6=standardUnit&u7=0&u3=USD&u4=KRW&u8=down&u2=1"
    response = requests.get(url)
    data = response.json()
    # "country" 리스트에서 currencyUnit이 "원"인 항목 찾기
    for item in data.get("country", []):
        if item.get("currencyUnit") == "원":
            # 콤마(,) 제거 후 float 변환
            value_str = item.get("value", "0").replace(',', '')
            return float(value_str)
    # 못 찾으면 None 반환
    return None


# 예측 결과를 실제 값(주가)으로 복원
def inverse_close_from_scaled(scaled_2d, scaler, n_features, idx_close):
    """
    scaled_2d: (N,H) 또는 (1,H) 형태의 '스케일된 종가'들
    스케일러 종류와 무관하게(standard/minmax/robust) 안전하게 역변환
    """
    arr = np.atleast_2d(np.asarray(scaled_2d, float))
    N, H = arr.shape
    outs = []
    for h in range(H):
        dummy = np.zeros((N, n_features), float)
        dummy[:, idx_close] = arr[:, h]
        inv_full = scaler.inverse_transform(dummy)
        outs.append(inv_full[:, idx_close])
    return np.column_stack(outs)  # (N,H)

# ---- StandardScaler만 사용한다면 아래 계산식으로 원단위로 복원 가능 ----
def inverse_close_matrix_fast(Y_xscale, scaler_X, idx_close):
    """
    Y_xscale: (N, H) 형태의 스케일된 종가 행렬... y종가를 X스케일링 한 것
    scaler_X: StandardScaler 객체 (X에 대해 fit된 것)
    :return: 원 단위 종가 1D 벡터 반환
    """
    return Y_xscale * scaler_X.scale_[idx_close] + scaler_X.mean_[idx_close]

def inverse_close_from_Xscale_fast(close_scaled_1d, scaler_X, idx_close):
    """
    close_scaled_1d: (N,) 또는 (H,) 형태의 스케일된 종가 벡터
    :return: 원 단위 종가 1D 벡터 반환
    """
    return close_scaled_1d * scaler_X.scale_[idx_close] + scaler_X.mean_[idx_close]
# -----------------------------------




"""
지표 추가 함수

매수 판단:
  MACD·ADX·ROC 같이 추세 지표가 긍정적일 때 신뢰성 높음
  RSI나 STOCH이 중립이거나 과매수 직전일 때 진입 타이밍 좋음
  
진입(매수): MACD ↑, ADX > 25, ROC 양수, CCI +100 돌파  
보유: MACD·ADX 유지, Ultimate Oscillator 50 이상, ATR 안정  
매도: MACD 하락, ADX 25 이하, CCI·Ultimate Oscillator 꺾임


추세·모멘텀·변동성” 요약 피처
추세(Trend): EMA/MACD 기울기, MACD 히스토그램의 평균·합
모멘텀(Momentum): ROC 평균·최댓값, RSI 평균·상/하위 분위수, CCI 평균
변동성(Volatility/Risk): ATR 평균·최댓값, 수익률 표준편차, 고저폭 평균

모델이 윈도우 내 패턴을 “요약 벡터”로 바로 보게 되어 학습 안정성·일반화에 보통 도움이 된다
"""
def add_technical_features(data, window=20, num_std=2):
    data = data.sort_index()
    data = data.copy()

    # epsilon(엡실론), 0으로 나누거나, 분모가 0에 매우 가까워질 때 생기는 폭발을 막기 위한 “아주 작은 상수”
    eps = 1e-9

    # 한국/영문 칼럼 자동 식별
    col_o = _col(data, '시가',   'Open')
    col_h = _col(data, '고가',   'High')
    col_l = _col(data, '저가',   'Low')
    col_c = _col(data, '종가',   'Close')
    col_v = _col(data, '거래량', 'Volume')

    o, h, l, c, v = data[col_o], data[col_h], data[col_l], data[col_c], data[col_v]

    # 1) 타깃(로그수익률)
    ret1  = np.log(c).diff()

    """
    RSI(14) — 상대강도지수 (Relative Strength Index)
    70 이상이면 과매수, 30 이하이면 과매도
    """
    data['RSI14'] = compute_rsi(c)  # 사전에 정의 필요

    # 볼린저밴드 (MA20, STD20, 상단/하단 밴드)
    data['MA20'] =c.rolling(window=window).mean()
    data['STD20'] =c.rolling(window=window).std()
    data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
    data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])

    # 볼린저밴드 위치 (0~1)
    data['BB_perc'] = (data[col_c] - data['LowerBand']) / (data['UpperBand'] - data['LowerBand'] + 1e-9)

    # 이동평균선
    data['MA5'] =c.rolling(window=5).mean()
    data['MA10'] =c.rolling(window=10).mean()
    data['MA5_slope'] = data['MA5'].diff()
    data['MA10_slope'] = data['MA10'].diff()
    data['MA20_slope'] = data['MA20'].diff()

    # 거래량 증감률
    data['Volume_change'] = v.pct_change().replace([np.inf, -np.inf], 0).fillna(0) # 거래 중지등의 사건에 극단적 노이즈

    vol = v.replace(0, np.nan)              # 0은 로그 불가 → NaN
    data['Vol_logdiff'] = np.log(vol).diff()

    # 당일 변동폭 (고가-저가 비율)
    data['day_range_pct'] = (h - l) / (l + 1e-9)

    # 캔들패턴 (양봉/음봉, 장대양봉 등)
    # data['is_bullish'] = (data[col_c] > data['시가']).astype(int) # 양봉이면 1, 음봉이면 0
    # 장대양봉(시가보다 종가가 2% 이상 상승)
    # data['long_bullish'] = ((data[col_c] - data['시가']) / data['시가'] > 0.02).astype(int)

    # 최근 N일간 등락률
    # data['chg_5d'] = (data[col_c] /c.shift(5)) - 1
    # 현재가 vs 이동평균(MA) 괴리율
    data['ma10_gap'] = (data[col_c] - data['MA10']) / data['MA10']
    # 거래량 급증 신호
    data['volume_ratio'] = v / v.rolling(20).mean()


    # === 추가 지표 ===
    # MACD
    """
    단기(12일)·장기(26일) 이동평균 차이로 추세 방향을 측정. 0 이상이면 상승세, 0 이하이면 하락세
    MACD가 0선 위에서 골든크로스 유지 → 상승추세 지속 가능성 ↑
    
    단기 EMA(12)와 장기 EMA(26)의 차이로 추세 방향을 잡고, 시그널선(9 EMA)과 비교해 매수/매도 신호를 확인.
    MACD > Signal → 매수 우위
    MACD < Signal → 매도 우위
    0선 위에서 골든크로스 = 강한 매수 신호, 0선 아래에서 데드크로스 = 강한 매도 신호    
    """
    macd_line, macd_signal, macd_hist = compute_macd(c, 12, 26, 9)
    data['MACD'] = macd_line
    data['MACD_signal'] = macd_signal
    data['MACD_hist'] = macd_hist

    # +DI / -DI / ADX
    """
    ADX(14) — 평균 방향성 지수 (Average Directional Index)
    추세 강도만 측정 (방향 아님). 25 이상이면 강한 추세.
    30~40 이상이면 추세 신뢰 가능. 다만 너무 높으면(50 이상) 과열로 볼 수도 있음
    
    ADX는 추세 강도를 나타내고, +DI / -DI는 상승/하락 에너지를 비교.
    ADX > 25 → 뚜렷한 추세 존재
    +DI > -DI → 상승 추세, -DI > +DI → 하락 추세
    ADX가 높다고 해서 방향을 말해주진 않음 (강한 상승일 수도, 강한 하락일 수도 있음)
    """
    plus_di, minus_di, adx = compute_di_adx(h, l, c, 14)
    data['PlusDI'] = plus_di
    data['MinusDI'] = minus_di
    data['ADX14'] = adx

    # ATR
    """
    ATR(14) — 평균진폭범위 (Average True Range)
    변동성 지표. 값이 높을수록 변동성이 큼
    """
    data['ATR14'] = compute_atr(h, l, c, 14)

    """
    CCI(14) — 상품채널지수 (Commodity Channel Index)
    100 이상이면 과매수, 강세장 지속
    -100 이하 과매도, 약세장 진입 신호
    
    단기 모멘텀 변화를 잘 포착하지만, 노이즈가 많음
    """
    data['CCI14'] = compute_cci(h, l, c, 14)

    """
    Ultimate Oscillator
    
    단기·중기·장기 모멘텀을 종합한 지표. RSI/스토캐스틱보다 안정적.
    50 이상 → 매수 우위 
    50 이하 → 매도 고려
    70 이상 → 과매수 가능
    30 이하 → 과매도 가능
    """
    # data['UltimateOsc'] = compute_ultimate_osc(h, l, c, 7, 14, 28)

    # ROC (percent)
    """
    ROC (Rate of Change)
    매수세(황소)와 매도세(곰) 힘을 비교. 양수면 매수세 우위
    추세의 속도 확인
    """
    data['ROC12_pct'] = compute_roc(c, 12, pct=True)


    # ===== 추가(전환+7일용 추천 피쳐들) =====
    data['lower_wick_ratio'] = (np.minimum(o, c) - l) / (h - l + eps)  # 아래꼬리 비율
    data['close_pos'] = (c - l) / (h - l + eps)                        # 당일 range 내 종가 위치(0~1)

    data['bb_recover'] = (c > data['LowerBand']) & (c.shift(1) < data['LowerBand'].shift(1))  # 하단밴드 복귀 이벤트
    data['z20'] = (c - data['MA20']) / (data['STD20'] + eps)                                   # z-score

    data['macd_hist_chg'] = data['MACD_hist'].diff()  # MACD hist 가속

    # 안전 처리
    data.replace([np.inf, -np.inf], np.nan, inplace=True)


    return data


# 전일 대비 오늘 등락률['today_chg_rate'] 추가
def add_today_change_rate(df: pd.DataFrame,
                          close_col: str = "Close",
                          out_col: str = "today_chg_rate") -> pd.DataFrame:
    """
    전일 Close 대비 당일 Close 등락률(%) 컬럼 추가.
    today_chg_rate = (Close / prev_Close - 1) * 100

    - df: 인덱스는 날짜(정렬 가능한 형태)라고 가정
    - close_col: 종가 컬럼명
    - out_col: 결과 컬럼명
    """
    out = df.copy()

    # 날짜 오름차순 정렬(전일/당일 계산 정확히)
    out = out.sort_index()

    prev_close = out[close_col].shift(1)
    out[out_col] = (out[close_col] / prev_close - 1) * 100

    return out


# columns 각각에 대해 객체의 파이썬 타입과 shape를 출력
def check_column_types(data, columns):
    for col in columns:
        print(f"[{col}] type: {type(data[col])}, shape: {data[col].shape}")


# 미장 한글 종목명 요청
def get_name_from_usa_ticker(ticker: str) -> Optional[str]:
    TOSSINVEST_API_URL = "https://wts-info-api.tossinvest.com/api/v3/search-all/wts-auto-complete"

    payload = {
        "query":str(ticker),
        "sections":[
            {"type":"SCREENER"},
            {"type":"NEWS"},
            {"type":"PRODUCT","option":{"addIntegratedSearchResult":"true"}},
            {"type":"TICS"}
        ]
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    try:
        # timeout=(연결, 읽기)
        resp = requests.post(TOSSINVEST_API_URL, json=payload, headers=headers, timeout=(5, 15))
        resp.raise_for_status()                     # 4xx/5xx면 예외 발생

        data = resp.json()

        keywords = []
        for block in data.get("result", []):
            if block.get("type") == "PRODUCT":
                for item in block.get("data", {}).get("items", []):
                    if item.get("keyword"):
                        keywords.append(item.get("keyword"))


        first_keyword = keywords[0] if keywords else None
        return first_keyword

    except requests.exceptions.JSONDecodeError:
        print("응답이 JSON 형식이 아닙니다:", resp.text[:500])
    except requests.exceptions.Timeout:
        print("요청 타임아웃")
    except requests.exceptions.HTTPError as e:
        print("HTTP 에러:", e.response.status_code, e.response.text[:500])
    except requests.exceptions.RequestException as e:
        print("네트워크/요청 에러:", repr(e))




# 일봉 캔들 차트
def plot_candles_daily(
        data: pd.DataFrame,
        show_months: int = 5,
        title: str = "Daily — Bollinger Bands & Volume",
        ax_price=None,
        ax_volume=None,
        future_dates=None,           # iterable of datetime/date/str
        predicted_prices=None,       # iterable of float
        date_tick=10,
        today=None,
):
    # 입력 표준화
    df = pd.DataFrame(data).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 한국/영문 칼럼 자동 식별
    col_o = _col(df, '시가',   'Open')
    col_h = _col(df, '고가',   'High')
    col_l = _col(df, '저가',   'Low')
    col_c = _col(df, '종가',   'Close')
    col_v = _col(df, '거래량', 'Volume')

    # 최근 N개월 슬라이싱
    end = df.index.max()
    start = end - pd.DateOffset(months=show_months)
    df = df.loc[start:end].copy()

    # 숫자화 & 필수 결측 제거
    for col in [col_o, col_h, col_l, col_c, col_v]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col_o, col_h, col_l, col_c])

    # 축 준비
    if ax_price is None or ax_volume is None:
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios':[3,1]}
        )
    else:
        fig = ax_price.figure

    # 데이터 준비
    x = np.arange(len(df))
    o = df[col_o].to_numpy(); c = df[col_c].to_numpy()
    h = df[col_h].to_numpy(); l = df[col_l].to_numpy()
    up = c > o; down = c < o; same = ~(up | down)
    colors = np.where(up, RED, np.where(down, BLUE, GRAY1))

    # 윗 패널: 캔들 (고저선 + 몸통)
    ax_price.vlines(x[up],   l[up],   h[up],   color=RED,  linewidth=1.3, alpha=1, zorder=3)
    ax_price.vlines(x[down], l[down], h[down], color=BLUE, linewidth=1.3, alpha=1, zorder=3)
    ax_price.vlines(x[same], l[same], h[same], color=GRAY1, linewidth=1.3, alpha=1, zorder=3)

    width = 0.6
    for i in range(len(df)):
        bottom = min(o[i], c[i])
        height = max(abs(c[i]-o[i]), 1e-8)
        ax_price.add_patch(Rectangle((x[i]-width/2, bottom), width, height,
                                     facecolor=colors[i], edgecolor=colors[i],
                                     linewidth=1.0, alpha=1, zorder=2))

    # 지표(있을 때만)
    if 'MA20' in df.columns:
        ax_price.plot(x, df['MA20'].to_numpy(), label='20일 이동평균선', color=ORANGE, alpha=0.85, zorder=1)
    if 'MA5' in df.columns:
        ax_price.plot(x, df['MA5'].to_numpy(),  label='5일 이동평균선',  color=GREEN,  alpha=0.85, zorder=1)
    if {'UpperBand','LowerBand'}.issubset(df.columns):
        ub = pd.to_numeric(df['UpperBand'], errors='coerce').to_numpy()
        lb = pd.to_numeric(df['LowerBand'], errors='coerce').to_numpy()
        ax_price.plot(x, ub, '--', color=GRAY1, alpha=0.85, label='볼린저밴드 상한선', zorder=0)
        ax_price.plot(x, lb, '--', color=GRAY1, alpha=0.85, label='볼린저밴드 하한선', zorder=0)
        ax_price.fill_between(x, ub, lb, color=GRAY2, alpha=0.18, zorder=-1)

    # 예측(있을 때만)
    if future_dates is not None and predicted_prices is not None:
        fd = pd.to_datetime(pd.Index(future_dates))
        fd_str = fd.strftime('%Y-%m-%d').tolist()
        # x축이 인덱스 기반이므로, 예측 구간은 별도의 x2를 붙여서 연속되게 함
        x2 = np.arange(len(df), len(df) + len(fd_str))
        ax_price.plot(x2, predicted_prices, linestyle='--', marker='s', markersize=7,
                      markeredgecolor='white', color='tomato', label='예측 가격')
        # 마지막 실제값과 첫 예측값 연결 (있을 때)
        if len(x) > 0 and len(x2) > 0:
            ax_price.plot([x[-1], x2[0]], [c[-1], predicted_prices[0]],
                          linestyle='dashed', color='tomato', linewidth=1.5)


    # ax_price.tick_params(axis='x', which='both', labelbottom=False)  # 위 축 날짜 라벨 숨김
    ax_price.tick_params(axis='x', which='both', labelbottom=True)   # 윗 축 라벨 표시
    plt.setp(ax_price.get_xticklabels(), rotation=45, ha='right')    # 회전/정렬
    # plt.setp(ax_price.get_xticklabels(), rotation=0, ha='center')    # 회전/정렬
    ax_price.set_title(title, fontsize=14)
    ax_price.grid(True, alpha=0.25)
    handles, labels = ax_price.get_legend_handles_labels()
    if labels:
        ax_price.legend(loc='upper left')

    # 아랫 패널: 거래량
    ax_volume.bar(x, df[col_v].to_numpy(), color=colors, alpha=1)
    ax_volume.set_ylabel('Volume (D)')
    ax_volume.grid(True, alpha=0.25)

    # x축 눈금(날짜 라벨)
    ds = df.index.strftime('%Y-%m-%d')
    # tick_idx = np.arange(0, len(df), max(1, len(df)//10))
    tick_idx = np.arange(0, len(df), max(1, date_tick))
    ax_volume.set_xticks(tick_idx)
    ax_volume.set_xticklabels(ds[tick_idx], rotation=45, ha='right')
    # ax_volume.set_xticklabels(ds[tick_idx], rotation=0, ha='center')

    # ==============================
    # 오늘 캔들 위에 날짜 텍스트 표시
    # ==============================
    if today is not None:
        today_ts = pd.to_datetime(today)

        # df는 이미 show_months 만큼 슬라이싱된 데이터
        if today_ts in df.index:
            # x좌표: 오늘이 df 안에서 몇 번째 캔들인지 (0,1,2,...)
            x_pos = df.index.get_loc(today_ts)

            # y좌표: 오늘 캔들의 '고가' 기준으로 살짝 위
            high_price = df.loc[today_ts, col_h]   # '고가' 직통 말고 col_h 사용
            y_pos = high_price * 1.01  # 고가보다 1% 위에 글자

            ax_price.text(
                x_pos,
                y_pos,
                # today_ts.strftime('%m-%d'),
                '◀◀◀',
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90,
            )

    return fig, ax_price, ax_volume

# 주봉 캔들 차트
def plot_candles_weekly(
        data: pd.DataFrame,
        show_months: int = 12,
        title: str = "Weekly — Bollinger Bands & Volume",
        ax_price=None,
        ax_volume=None,
        date_tick=10,
):
    # 입력 표준화
    df = pd.DataFrame(data).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # 한국/영문 칼럼 자동 식별
    col_o = _col(df, '시가',   'Open')
    col_h = _col(df, '고가',   'High')
    col_l = _col(df, '저가',   'Low')
    col_c = _col(df, '종가',   'Close')
    col_v = _col(df, '거래량', 'Volume')

    # 최근 N개월 슬라이싱
    end = df.index.max()
    start = end - pd.DateOffset(months=show_months)
    df = df.loc[start:end].copy()

    # 숫자화 & 필수 결측 제거
    for col in [col_o, col_h, col_l, col_c, col_v]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col_o, col_h, col_l, col_c])

    # 주봉 리샘플
    w = df[[col_o,col_h,col_l,col_c,col_v]].resample('W-FRI').agg({
        col_o:'first',col_h:'max',col_l:'min',col_c:'last',col_v:'sum'
    }).dropna(subset=[col_o,col_h,col_l,col_c])

    # 지표(주봉 기준)
    w['MA5']  = w[col_c].rolling(5).mean()
    w['MA20'] = w[col_c].rolling(20).mean()
    std20 = w[col_c].rolling(20).std()
    w['UpperBand'] = w['MA20'] + 2*std20
    w['LowerBand'] = w['MA20'] - 2*std20

    # 축 준비
    if ax_price is None or ax_volume is None:
        fig, (ax_price, ax_volume) = plt.subplots(
            2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios':[3,1]}
        )
    else:
        fig = ax_price.figure

    # 데이터 준비
    x = np.arange(len(w))
    o = w[col_o].to_numpy(); c = w[col_c].to_numpy()
    h = w[col_h].to_numpy(); l = w[col_l].to_numpy()
    up = c > o; down = c < o; same = ~(up | down)
    colors = np.where(up, RED, np.where(down, BLUE, GRAY1))

    # 윗 패널: 캔들 (고저선 + 몸통)
    ax_price.vlines(x[up],   l[up],   h[up],   color=RED,  linewidth=1.3, alpha=1, zorder=3)
    ax_price.vlines(x[down], l[down], h[down], color=BLUE, linewidth=1.3, alpha=1, zorder=3)
    ax_price.vlines(x[same], l[same], h[same], color=GRAY1, linewidth=1.3, alpha=1, zorder=3)

    width = 0.6
    for i in range(len(w)):
        bottom = min(o[i], c[i])
        height = max(abs(c[i]-o[i]), 1e-8)
        ax_price.add_patch(Rectangle((x[i]-width/2, bottom), width, height,
                                     facecolor=colors[i], edgecolor=colors[i],
                                     linewidth=1.0, alpha=1, zorder=2))

    # 지표(있을 때만)
    if 'MA20' in w.columns:
        ax_price.plot(x, w['MA20'].to_numpy(), label='20일 이동평균선', color=ORANGE, alpha=0.85, zorder=1)
    if 'MA5' in w.columns:
        ax_price.plot(x, w['MA5'].to_numpy(),  label='5일 이동평균선',  color=GREEN,  alpha=0.85, zorder=1)
    if {'UpperBand','LowerBand'}.issubset(w.columns):
        ub = pd.to_numeric(w['UpperBand'], errors='coerce').to_numpy()
        lb = pd.to_numeric(w['LowerBand'], errors='coerce').to_numpy()
        ax_price.plot(x, ub, '--', color=GRAY1, alpha=0.85, label='볼린저밴드 상한선', zorder=0)
        ax_price.plot(x, lb, '--', color=GRAY1, alpha=0.85, label='볼린저밴드 하한선', zorder=0)
        ax_price.fill_between(x, ub, lb, color=GRAY2, alpha=0.18, zorder=-1)

    # ax_price.tick_params(axis='x', which='both', labelbottom=False)  # 위 축 날짜 라벨 숨김
    ax_price.tick_params(axis='x', which='both', labelbottom=True)   # 윗 축 라벨 표시
    # plt.setp(ax_price.get_xticklabels(), rotation=45, ha='right')    # 회전/정렬
    plt.setp(ax_price.get_xticklabels(), rotation=0, ha='center')    # 회전/정렬
    ax_price.set_title(title, fontsize=14)
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc='upper left')

    # 아랫 패널: 거래량
    ax_volume.bar(x, w[col_v].to_numpy(), color=colors, alpha=1)
    ax_volume.set_ylabel('Volume (W)')
    ax_volume.grid(True, alpha=0.25)

    # x축 눈금(날짜 라벨)
    dsw = w.index.strftime('%Y-%m-%d')
    # tick_idx = np.arange(0, len(w), max(1, len(w)//10))
    tick_idx = np.arange(0, len(w), max(1, date_tick))
    ax_volume.set_xticks(tick_idx)
    # ax_volume.set_xticklabels(dsw[tick_idx], rotation=45, ha='right')
    ax_volume.set_xticklabels(dsw[tick_idx], rotation=0, ha='center')

    return fig, ax_price, ax_volume








# 로그수익률을 원 단위로 복원
def prices_from_logrets(base_close_1d, logrets_2d):
    """
    이 함수는 **기준가(base price)**와 미래 구간의 로그수익률들로부터 각 구간의 가격 시퀀스를 복원
    :param base_close_1d: 각 샘플의 기준 시점 가격
    :param logrets_2d:    각 샘플의 미래 H-step 로그수익률
    :return:
    """
    base_close_1d = np.asarray(base_close_1d, dtype=float)
    logrets_2d    = np.asarray(logrets_2d, dtype=float)
    if base_close_1d.ndim != 1 or logrets_2d.ndim != 2:
        raise ValueError("shapes must be (N,), (N,H)")
    if len(base_close_1d) != logrets_2d.shape[0]:
        raise ValueError("N mismatch between base_close and logrets")
    return base_close_1d[:, None] * np.exp(np.cumsum(logrets_2d, axis=1))

# 로그수익률 시리즈 만들기
def log_returns_from_prices(close_1d: np.ndarray) -> np.ndarray:
    """
    close_1d: (L,) 원본 종가
    반환: (L-1,) g_1..g_{L-1},  g_t = log(C_t / C_{t-1})
    """
    return np.diff(np.log(close_1d))





def regression_metrics(
        y_true, y_pred, *,      # 정닶, 예측값
        y_scaler=None,          # 타깃 전용 스케일러가 있을 때 사용
        scaler=None,            # 전체 피처 스케일러(지금 케이스)
        n_features=None,
        idx_close=None,
        y_insample_for_mase_scaled=None,     # y-스케일(=현재 y_true/y_pred 공간)의 1D 시계열
        y_insample_for_mase_restored=None,   # 복원(가격) 공간의 1D 시계열
        m: int = 1                 # 계절 주기(없으면 1)
):
    y_true = _to_2d(y_true)
    y_pred = _to_2d(y_pred)

    # scaled-space
    """
    MSE / RMSE / MAE: 스케일 공간에서의 오차 크기
      표준화(평균0, 표준편차1)라면 RMSE≈1 이면 평균적인 오차, 0.1이면 매우 작음
      로그수익률 스케일이라면 “로그수익률 차이”의 크기
    MAPE / SMAPE(%): 스케일 공간의 상대오차. (표준화 스케일에선 해석이 직관적이지 않을 수 있음)
    R2: 스케일 공간에서의 결정계수. 1에 가까울수록 좋고, 0은 평균 예측과 비슷, 음수면 평균보다 못함
    """
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    # MAPE/SMAPE는 0 근처 값에 민감합니다(함수 내부에 eps 처리는 있지만 여전히 해석 주의)
    mape = _mape(y_true, y_pred)
    smape = _smape(y_true, y_pred)
    # r2는 y의 분산이 작거나 상수에 가까우면 의미가 희석될 수 있음
    r2   = r2_score(y_true, y_pred)

    out = {"scaled": {
        "MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae),
        "MAPE (%)": float(mape), "SMAPE (%)": float(smape), "R2": float(r2)
    }}

    # (선택) MASE
    if y_insample_for_mase_scaled is not None:
        out["scaled"]["MASE"] = float(_mase(
            y_true, y_pred,
            y_insample=np.asarray(y_insample_for_mase_scaled), m=m
        ))

    # restore if possible
    restored_true = restored_pred = None

    # --- restore to PRICE if possible (two-step) ---
    if (y_scaler is not None) and (scaler is not None) and (n_features is not None) and (idx_close is not None):
        # 1) y-스케일 → X-스케일
        true_x = y_scaler.inverse_transform(y_true)   # (N,H)
        pred_x = y_scaler.inverse_transform(y_pred)   # (N,H)

        # 2) X-스케일 → 가격 (close만 채워 역변환)
        # restored_true = inverse_close_from_scaled(true_x, scaler, n_features, idx_close)
        # restored_pred = inverse_close_from_scaled(pred_x, scaler, n_features, idx_close)

        # StandardScaler 전용 고속 버전
        restored_true = inverse_close_matrix_fast(true_x, scaler, idx_close)
        restored_pred = inverse_close_matrix_fast(pred_x,  scaler, idx_close)

    elif (scaler is not None) and (n_features is not None) and (idx_close is not None):
        def inv(part):
            part = np.array(part)
            if part.ndim == 2 and part.shape[1] > 1:
                cols = []
                for h in range(part.shape[1]):
                    dummy = np.zeros((part.shape[0], n_features), dtype=float)
                    dummy[:, idx_close] = part[:, h]
                    inv_full = scaler.inverse_transform(dummy)
                    cols.append(inv_full[:, idx_close])
                return np.column_stack(cols)
            else:
                dummy = np.zeros((part.shape[0], n_features), dtype=float)
                dummy[:, idx_close] = part.ravel()
                inv_full = scaler.inverse_transform(dummy)
                return inv_full[:, idx_close].reshape(-1, 1)

        restored_true = inv(y_true)
        restored_pred = inv(y_pred)

    if restored_true is not None and restored_pred is not None:
        """
        MSE / RMSE / MAE: 원 단위(₩) 오차
          예: RMSE=15,000원이면 1~3일 뒤 예측이 평균 1.5만원 정도 흔들린다는 감각
        MAPE / SMAPE(%): 가격 기준 상대오차(%). 종목/시점 간 비교가 쉬워 실무에 유용
          SMAPE는 0~200% 범위, 대칭적이라 극단값에 덜 민감
        R2: 원 가격 기준의 설명력
          0.5 이상이면 꽤 유의미, 0~0.3대면 예측이 쉽지 않거나 과소적합/데이터 한계 가능
        """
        r_mse  = mean_squared_error(restored_true, restored_pred)
        r_rmse = np.sqrt(r_mse)
        r_mae  = mean_absolute_error(restored_true, restored_pred)
        r_mape = _mape(restored_true, restored_pred)
        r_smape= _smape(restored_true, restored_pred)
        r_r2   = r2_score(restored_true, restored_pred)

        out["restored"] = {
            "MSE": float(r_mse), "RMSE": float(r_rmse), "MAE": float(r_mae),
            "MAPE (%)": float(r_mape), "SMAPE (%)": float(r_smape), "R2": float(r_r2)
        }
        # ★ 복원(가격) MASE 추가
        if y_insample_for_mase_restored is not None:
            out["restored"]["MASE"] = float(_mase(
                restored_true, restored_pred,
                y_insample=np.asarray(y_insample_for_mase_restored), m=m
            ))

    return out

"""
이 두 개만으로는 “나이브보다 낫다”/“과적합 아님”/“라벨 희소성 영향 없음”을 보장 못 한다

왜 불충분한가
  나이브 대비 우월성이 빠져있음 → 수준 미달 모델도 우연히 R² 0.6 넘을 수 있음
  표본 수/라벨 분포 미검증 → 분산 큰 구간/희소 라벨에선 지표가 흔들립니다
  시계열 검증 특수성(롤링/워크포워드) 미반영 → 단일 홀드아웃만 보면 낙관적
  
복원된(R², sMAPE) 는 “원 단위(가격)” 기준이라 실전 해석엔 꼭 필요
  R²: 변동성 설명력. 0에 가까우면 평균값과 비슷, 음수면 평균보다 나쁨. 가격예측은 본질적으로 어려워서 0.1~0.3도 흔함
  sMAPE: 상대오차(%)라 종목/구간 비교가 쉬움. 해석도 직관적.
  하지만 이 둘만 보면 과적합/누수/스케일 편향을 놓칠 수 있다
"""
def pass_filter(metrics, use_restored=True, r2_min=None, smape_max=30.0, mape_max=50.0):
    m = metrics["restored"] if (use_restored and "restored" in metrics) else metrics["scaled"]
    r2 = m["R2"]
    sm = m["SMAPE (%)"]
    # print(f"    DEBUG >> R2: {r2:.2f} SMAPE: {sm:.2f}") # <- 디버깅
    if not np.isfinite(sm):
        return False
    # SMAPE 상한 컷만 적용
    if sm > smape_max:
        return False
    # R² 컷 비활성화(기본) — 쓰려면 r2_min에 숫자 전달
    if (r2_min is not None) and (not np.isnan(r2)) and (r2 < r2_min):
        return False
    return True

"""
나이브 비교(예: C_{t+h}=C_t)**가 가장 현실적인 기준

복원 지표 (ALL / 각 h별):
  sMAPE ≤ 8%, R² ≥ 0.1 (데이터/종목에 따라 0.0~0.2 범위로 조정)
  RMSE_ratio ≤ 0.95 또는 sMAPE_ratio ≤ 0.95 (나이브 대비 ≥5% 개선)
  HitRate@h ≥ 0.52 (방향성 약간이라도 우위)
  
강건성:
  median sAPE와 P90 sAPE도 같이 보고, 허버 δ는 (타깃 표준편차 × 1~2) 근방으로 튜닝
  
이상치에 대비해 강건 지표/손실/클리핑도 함께 쓰도록
"""
def pass_filter_v2(
        metrics, *,              # regression_metrics() 결과
        use_restored=True,
        r2_min=0.10,             # 보수적으로 시작 (원단위 R²은 높이기 어려움), 0.0부터 0.2까지 테스트
        smape_max=8.0,           # 종목/호라이즌에 맞춰 튜닝
        # === 추가 가드 ===
        require_naive_improve=True,
        naive_improve_min=0.05,  # 나이브 대비 ≥5% 개선
        samples_min=50,          # 검증 샘플 최소 개수
        hitrate_min=None,        # 방향성 적중률 가드 (옵션, 예: 0.52)
        ctx=None,                # (옵션) y_true/y_pred/y_naive 등 추가 컨텍스트
):
    m = metrics["restored"] if (use_restored and "restored" in metrics) else metrics["scaled"]
    r2     = m["R2"]
    smape  = m["SMAPE (%)"]
    print(f"    DEBUG >> R2: {r2:.2f} SMAPE: {smape:.2f}") # <- 디버깅
    # print('    ctx["hitrate"]: ', ctx["hitrate"])

    # 0) 기본 컷
    if not np.isfinite(r2) or not np.isfinite(smape):
        return False
    if r2 < r2_min or smape > smape_max:
        return False

    # 1) 나이브 대비 개선
    if require_naive_improve and ctx is not None and "smape_naive" in ctx:
        smape_naive = ctx["smape_naive"]
        print(f'    min smape_naive: {smape_naive * (1 - naive_improve_min):.2f}')
        if not np.isfinite(smape_naive):
            return False
        # smape가 작을수록 좋음 → (모델 smape) ≤ (나이브 smape)*(1 - 개선율)
        eps = 1e-9
        if smape > smape_naive * (1 - naive_improve_min) + eps:
            return False

    # 2) 샘플 수 가드
    if ctx is not None and "n_val" in ctx:
        if ctx["n_val"] < samples_min:
            return False

    # 3) 방향성 가드(선택)
    if hitrate_min is not None and ctx is not None and "hitrate" in ctx:
        if ctx["hitrate"] < hitrate_min:
            return False

    return True


# 5일선이 20일선 아래에서 근접하게 다가옴
def near_bull_cross_signal(df: pd.DataFrame,
                           lookback: int = 5,
                           gap_bp: float = 0.004,    # MA20 대비 0.4% 이내
                           min_rise_bp: float = 0.002, # delta가 최근 lookback동안 최소 0.2%p 이상 상승, 근접 “속도”의 최소 요구치.
                           use_atr: bool = True
                           ) -> bool:
    """
    df: MA5, MA20 컬럼 포함. 인덱스는 날짜(오름차순).
    조건:
      - 현재 MA5 < MA20 (아직 역전 전)
      - 격차 |MA5-MA20| 가 MA20의 gap_bp 이내
      - 최근 lookback일 동안 delta가 유의미하게 상승(=붙는 중)
    """
    s_ma5 = pd.to_numeric(df['MA5'], errors='coerce')
    s_ma20 = pd.to_numeric(df['MA20'], errors='coerce')
    if s_ma5.isna().iloc[-lookback:].any() or s_ma20.isna().iloc[-lookback:].any():
        return False

    delta = s_ma5 - s_ma20
    # 아직 아래에 있음
    if not (delta.iloc[-1] < 0):
        return False

    # # 현재 격차가 충분히 좁은가? (MA20의 퍼센트 기준)
    # tight_now = abs(delta.iloc[-1]) <= (s_ma20.iloc[-1] * gap_bp)
    #
    # # 최근 lookback일 동안 delta가 유의미하게 상승(즉, 더 0에 가까워짐)
    # delta_rise = (delta.iloc[-1] - delta.iloc[-lookback]) >= (s_ma20.iloc[-1] * min_rise_bp)

    # 동적 갭 허용 (선택)
    if use_atr and {'High','Low','Close'}.issubset(df.columns):
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low']  - df['Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        if not pd.isna(atr.iloc[-1]):
            vol_bp = (atr / s_ma20).iloc[-1]
            gap_bp = max(gap_bp, float(vol_bp) * 0.8)

    tight_now = abs(delta.iloc[-1]) <= (s_ma20.iloc[-1] * gap_bp)

    # 갭이 크면 상승 요구치 가중
    gap_ratio = abs(delta.iloc[-1]) / s_ma20.iloc[-1]
    req_rise  = s_ma20.iloc[-1] * min_rise_bp * (1.5 if gap_ratio > gap_bp*0.5 else 1.0)
    delta_rise = (delta.iloc[-1] - delta.iloc[-lookback]) >= req_rise


    # 보조: MA5 기울기 양수, MA20 기울기 비양수(완화 가능)
    ma5_slope = s_ma5.iloc[-1] - s_ma5.iloc[-2]
    # ma20_slope = s_ma20.iloc[-1] - s_ma20.iloc[-2]
    # slope_ok = (ma5_slope > 0) and (ma20_slope <= 0)
    slope_ok   = (ma5_slope > 0)  # MA20 기울기 조건은 완화

    return bool(tight_now and delta_rise and slope_ok)


########## 필터링 추가 ##########


def _best_f1_threshold(y_true, scores):
    # 여러 컷을 훑어서 F1 최대가 되는 threshold 선택
    prec, rec, ths = precision_recall_curve(y_true, scores)
    f1s = 2*prec*rec / (prec + rec + 1e-12)
    if len(ths) == 0:
        # 전부 한 클래스일 때 등
        return 0.0, float(f1s.max() if len(f1s) else 0.0)
    i = int(np.nanargmax(f1s))
    th = ths[min(i, len(ths)-1)]
    return float(th), float(f1s[i])

def classify_metrics_from_price(preds_scaled, Y_val_scaled, X_val, *,
                                scaler, n_features, idx_close,
                                horizon_idx, thresh_ret=0.03):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # --- 복원 ---
    Yp = inverse_close_from_scaled(preds_scaled, scaler, n_features, idx_close)  # (N,H)
    Yt = inverse_close_from_scaled(Y_val_scaled, scaler, n_features, idx_close)  # (N,H)
    cur = inverse_close_from_scaled(X_val[:, -1, idx_close][:, None],
                                    scaler, n_features, idx_close)[:, 0]

    h = int(np.clip(horizon_idx, 0, Yp.shape[1]-1))
    ret_true = Yt[:, h] / cur - 1.0
    ret_pred = Yp[:, h] / cur - 1.0

    y_true = (ret_true >= thresh_ret).astype(int)
    scores = ret_pred.astype(float)

    pos = int(y_true.sum()); neg = int(y_true.size - pos)
    if pos == 0 or neg == 0:
        return {
            "AUC": np.nan, "F1@opt": 0.0, "th_opt": 0.0,
            "y_true": y_true, "scores": scores,
            "ret_true": ret_true, "ret_pred": ret_pred,
            "single_class": True, "pos": pos, "neg": neg,
        }

    # --- 분류 지표 ---
    auc = float(roc_auc_score(y_true, scores))
    th_opt, f1_opt = _best_f1_threshold(y_true, scores)

    return {
        "AUC": auc, "F1@opt": float(f1_opt), "th_opt": float(th_opt),
        "y_true": y_true, "scores": scores,
        "ret_true": ret_true, "ret_pred": ret_pred,
        "single_class": False, "pos": pos, "neg": neg,
    }


# def rolling_eval_3ahead(model, X_3d, Y_2d, y_hist_end, y_insample_for_mase):
def rolling_eval_3ahead(preds, Y_2d, y_hist_end, y_insample_for_mase,
                            *, scaler=None, n_features=None, idx_close=None):
    """
    Y_2d:  (N, 3)          # t+1, t+2, t+3의 실제값
    y_hist_end: (N,)       # 각 윈도우의 마지막 실제값 y[t-1]
    y_insample_for_mase: 1D, 훈련(in-sample) 구간의 실제값
    """
    y_pred = preds

    # 2) (선택) 복원: scaler가 주어지면 세 값 모두 원가격으로 변환
    if scaler is not None:
        Y_2d = inverse_close_from_scaled(Y_2d,  scaler, n_features, idx_close)
        y_pred = inverse_close_from_scaled(y_pred, scaler, n_features, idx_close)
        y_hist_end = inverse_close_from_scaled(
            y_hist_end.reshape(-1,1), scaler, n_features, idx_close
        )[:,0]
        # y_insample_for_mase도 원가격 1D로 전달되어야 함

    # 3) 나이브 베이스라인
    # 나이브: t+1..t+3 모두 y[t-1]로 고정
    y_naive = np.repeat(y_hist_end.reshape(-1,1), y_pred.shape[1], axis=1)
    def _metrics(y_t, y_p):
        mse  = mean_squared_error(y_t.reshape(-1), y_p.reshape(-1))
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_t, y_p)
        mape = _mape(y_t, y_p)
        smp  = _smape(y_t, y_p)          # 여기 값이 "원가격 기준"이 됨
        r2   = r2_score(y_t.reshape(-1), y_p.reshape(-1))
        ms   = _mase(y_t, y_p, y_insample=y_insample_for_mase, m=1)
        return dict(MAE=mae, RMSE=rmse, R2=r2, sMAPE=smp, MAPE=mape, MASE=ms)

    out = {"model": _metrics(Y_2d, y_pred), "naive": _metrics(Y_2d, y_naive)}
    return out, (Y_2d, y_pred, y_naive)

# 한국 영업일 가져오는 함수 (주말, 공휴일 제외)
def get_next_business_days():
    # (선택) pdc-market-calendars의 break 경고 억제
    warnings.filterwarnings(
        "ignore",
        message=r"\['break_start', 'break_end'\] are discontinued",
        category=UserWarning,
        module="pandas_market_calendars"
    )

    # 1) 한국거래소 캘린더
    cal = mcal.get_calendar('XKRX')

    # (선택) 단종된 휴장 타임 명시 제거
    if hasattr(cal, "remove_time"):
        for t in ("break_start", "break_end"):
            try:
                cal.remove_time(t)
            except Exception:
                pass

    # 2) 오늘(KST) 기준 스케줄 생성 구간
    today_kst = pd.Timestamp.today(tz='Asia/Seoul').normalize()
    start = today_kst - pd.Timedelta(days=14)
    end   = today_kst + pd.Timedelta(days=14)

    # 3) 스케줄(UTC 인덱스) → KST 변환
    schedule = cal.schedule(start_date=start, end_date=end)
    idx_utc = schedule.index
    if idx_utc.tz is None:
        idx_utc = idx_utc.tz_localize('UTC')
    sessions_kst = idx_utc.tz_convert('Asia/Seoul')

    # 4) 날짜(자정)로 정규화 + 타임존 제거 → 오늘 이후 3영업일
    bd_kst_dates = sessions_kst.normalize()               # still tz-aware (KST, 00:00)
    bd_kst_naive = bd_kst_dates.tz_localize(None)         # drop tz → tz-naive dates
    today_naive  = today_kst.tz_convert('Asia/Seoul').normalize().tz_localize(None)

    return bd_kst_naive[bd_kst_naive > today_naive][:3]

# 사용안함
def effective_trials_for_hitrate(y_true, y_pred, *, y_base=None, space="price", use_horizon=1, thr=None):
    """
    반환: (N_eff, mask)  # 그 지평에서 실제로 비교에 쓰인 시행 수
    - thr: tiny-move 필터(가격 변화율 절대값이 thr 미만이면 제외). 예: 0.003 (0.3%)
    """
    import numpy as np
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    N, H = y_true.shape
    h_idx = range(H) if use_horizon == "avg" else [int(use_horizon)-1]

    def _mask_for_h(h):
        if space == "price":
            if y_base is None: raise ValueError("price space needs y_base")
            # 변화율(참/예측) 계산해서 tiny-move 제거 옵션
            if thr is not None:
                chg_t = (y_true[:,h] - y_base) / y_base
                chg_p = (y_pred[:,h] - y_base) / y_base
                mask = np.isfinite(chg_t) & np.isfinite(chg_p) & (np.abs(chg_t) >= thr) & (np.abs(chg_p) >= thr)
            else:
                tgt = y_true[:,h] - y_base
                est = y_pred[:,h] - y_base
                mask = np.isfinite(tgt) & np.isfinite(est)
        else:  # return space
            if thr is not None:
                mask = np.isfinite(y_true[:,h]) & np.isfinite(y_pred[:,h]) & (np.abs(y_true[:,h]) >= thr) & (np.abs(y_pred[:,h]) >= thr)
            else:
                mask = np.isfinite(y_true[:,h]) & np.isfinite(y_pred[:,h])
        return mask

    masks = [_mask_for_h(h) for h in h_idx]
    N_eff = [int(m.sum()) for m in masks]
    return (sum(N_eff)/len(N_eff) if use_horizon == "avg" else N_eff[0]), masks

# 사용안함
def min_sig_hitrate(N_eff, alpha=0.05):
    z = 1.96 if alpha == 0.05 else 2.576  # 0.01
    return 0.5 + z * 0.5 / max(N_eff, 1)**0.5


def drop_sparse_columns(df: pd.DataFrame, threshold: float = 0.10, *, check_inf: bool = True, inplace: bool = False):
    """
    결측치 비율이 threshold(기본 10%)를 넘는 컬럼을 드롭.
    옵션으로 무한대(±inf) 비율도 같은 기준으로 드롭.

    check_inf : bool
        True면 숫자형 컬럼의 ±inf 비율도 검사하여 기준 초과 시 드롭
    inplace : bool
        True면 df를 직접 수정하고, False면 복사본을 반환
    """
    target = df if inplace else df.copy()
    dropped = []

    for col in list(target.columns):
        s = target[col]
        na_ratio = s.isna().mean()    # isna() : pandas의 결측값(NA) 체크. NaN, None, NaT에 대해 True

        inf_ratio = 0.0
        if check_inf and is_numeric_dtype(s):
            # 숫자형만 inf 체크 (문자열 등에선 불필요/오류 방지)
            inf_ratio = np.isinf(s.to_numpy()).mean()

        if (na_ratio > threshold) or (inf_ratio > threshold):
            target.drop(columns=[col], inplace=True, errors="ignore")
            dropped.append(col)

    return target, dropped


def signal_any_drop(data: pd.DataFrame,
                    days: int = 12,
                    up_thr: float = 3.0,
                    down_thr: float = -3.0,
                    today_chg_rate: str = "등락률") -> bool:
    """
    요구 조건:
      - 오늘 등락률(마지막 행) >= up_thr  (단위: %)
      - 어제부터 과거 days일 동안 등락률 <= down_thr 인 날이 '하루라도' 있음
      - 같은 기간(어제~과거 days일) 동안 MA5 < MA20 이 '항상' 성립
    컬럼 필요: '등락률', 'MA5', 'MA20'
    """

    # 안전 변환
    chg  = pd.to_numeric(data[today_chg_rate], errors='coerce')
    ma5  = pd.to_numeric(data['MA5'],   errors='coerce')
    ma20 = pd.to_numeric(data['MA20'],  errors='coerce')

    # 오늘 등락률(마지막 행)
    today_chg = chg.iloc[-1]

    # 어제~과거 days일 (총 days개): 마지막 행 제외한 꼬리 days개
    past_chg  = chg.iloc[-(days+1):-1]
    past_ma5  = ma5.iloc[-(days+1):-1]
    past_ma20 = ma20.iloc[-(days+1):-1]

    # 결측 있으면 보수적으로 False (원하면 dropna로 완화 가능)
    if past_chg.isna().any() or past_ma5.isna().any() or past_ma20.isna().any() or pd.isna(today_chg):
        return False

    cond_today        = (today_chg >= up_thr)
    cond_past_anydrop = past_chg.le(down_thr).any()     # 하루라도 down_thr 이하
    cond_ma_order     = past_ma5.lt(past_ma20).all()    # days기간 내내 MA5 < MA20

    return bool(cond_today and cond_past_anydrop and cond_ma_order)

def low_weekly_check(data: pd.DataFrame):
    # 인덱스가 날짜/시간이어야 함
    if not isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        data = data.copy()
        data.index = pd.to_datetime(data.index)

    # 한국/영문 칼럼 자동 식별
    col_o = _col(data, '시가',   'Open')
    col_h = _col(data, '고가',   'High')
    col_l = _col(data, '저가',   'Low')
    col_c = _col(data, '종가',   'Close')
    col_v = _col(data, '거래량', 'Volume')

    # 주봉 리샘플 (월~금 장 기준이면 W-FRI 권장)
    weekly = data.resample('W-FRI').agg({
        col_o: 'first',
        col_h: 'max',
        col_l: 'min',
        col_c: 'last',
        col_v: 'sum'
    }).dropna(subset=[col_c])  # 종가 없는 주 제거

    # 직전 2주 추출
    prev_close = weekly.iloc[-2][col_c]
    two_w_close = weekly.iloc[-3][col_c]
    three_w_close = weekly.iloc[-4][col_c]
    four_w_close = weekly.iloc[-5][col_c]
    this_close = weekly.iloc[-1][col_c]   # 마지막 주 종가
    first      = weekly.iloc[0][col_c]    # 첫번째 주 종가

    '''
    prev_close = 100
    this_close = 105
    one_w_ago_pct = (105 / 100) - 1   # 1.05 - 1 = 0.05    >> one_w_ago_pct = 0.05 (5% 상승)
    '''
    pct_from_first  = (this_close / first) - 1           # 이번 주 종가(this_close)가 첫 번째 주 종가(first) 대비 몇 % 변했는지
    one_w_ago_pct   = (this_close / prev_close) - 1      # 저번주 대비 이번주 증감률
    two_w_ago_pct   = (this_close / two_w_close) - 1     # 2주 대비 이번주 증감률
    three_w_ago_pct = (this_close / three_w_close) - 1   # 3주 대비 이번주 증감률
    four_w_ago_pct  = (this_close / four_w_close) - 1    # 4주 대비 이번주 증감률
    is_drop_over_1  = one_w_ago_pct < -0.01              # -1% 보다 더 하락했는가 // -0.005   # -0.5%

    return {
        "ok": True,
        # "this_week_close": float(this_close),
        # "last_week_close": float(prev_close),
        "pct_vs_lastweek": float(one_w_ago_pct*100),                    # 저번주 대비 이번주 증감률, 예: -0.0312 == -3.12%
        "pct_vs_last2week": float(two_w_ago_pct*100),                   # 2주 전 대비 이번주 증감률
        "pct_vs_last3week": float(three_w_ago_pct*100),                 # 3주 전 대비 이번주 증감률
        "pct_vs_last4week": float(four_w_ago_pct*100),                  # 4주 전 대비 이번주 증감률
        "is_drop_more_than_minus1pct": bool(is_drop_over_1),            # 주봉 증감률이 기준보다 하락했는지
        "pct_vs_firstweek": float(pct_from_first*100),                  # -0.22 -> -22% 하락
    }


# csv 데이터를 날짜순으로 내림차순
def sort_csv_by_today_desc(
        in_path: str,
        out_path: Optional[str] = None,
        date_col: str = "today",
        secondary_col: str = "ticker",
        encoding: str = "utf-8-sig",
) -> str:
    """
    CSV를 date_col(기본: today) 기준 내림차순(최신 먼저)으로 정렬해서 저장.
    - ticker처럼 앞자리 0 유지하려고 secondary_col은 문자열로 읽음.
    - out_path가 None이면 원본 파일명 뒤에 '_sorted'를 붙여 저장.
    반환: 저장된 out_path 문자열
    """
    if out_path is None:
        p = Path(in_path)
        out_path = str(p.with_name(p.stem + "_sorted" + p.suffix))

    df = pd.read_csv(in_path, dtype={secondary_col: str})
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.sort_values([date_col, secondary_col], ascending=[False, True])

    df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")

    df.to_csv(out_path, index=False, encoding=encoding)
    return out_path


# 프로세스 경쟁 방지
def safe_read_pickle(path):
    # 0바이트 또는 너무 작은 파일은 바로 제외
    try:
        size = os.path.getsize(path)
    except OSError:
        return None

    if size < 16:   # 임계치는 상황에 맞게 (피클 헤더 고려)
        return None

    try:
        return pd.read_pickle(path)
    except EOFError:
        return None
    except Exception:
        # 필요하면 로깅
        return None