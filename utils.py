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

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


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

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rma(s: pd.Series, length: int) -> pd.Series:
    # Wilder's moving average
    alpha = 1 / length
    return s.ewm(alpha=alpha, adjust=False).mean()




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




def drop_trading_halt_rows(data: pd.DataFrame):
    """OHLCV 기준으로 거래정지/비정상 바를 제거한 DataFrame과 제거된 인덱스를 반환"""
    d = data.copy()
    d = d.replace([np.inf, -np.inf], np.nan)

    # 한국/영문 칼럼 자동 식별
    col_o = _col(d, '시가',   'Open')
    col_h = _col(d, '고가',   'High')
    col_l = _col(d, '저가',   'Low')
    col_c = _col(d, '종가',   'Close')
    col_v = _col(d, '거래량', 'Volume')

    # 1) OHLCV 전부 존재
    notna = d[[col_o, col_h, col_l, col_c, col_v]].notna().all(axis=1)
    # 2) 가격이 0보다 큼(데이터 공급사에 따라 0으로 채워지는 경우 방지)
    positive_price = d[[col_o, col_h, col_l, col_c]].gt(0).all(axis=1)
    # 3) 거래량 > 0 (거래정지/휴장 등)
    nonzero_vol = d[col_v].fillna(0) > 0
    # 4) 고가 >= 저가 (이상치 방지)
    hl_ok = d[col_h] >= d[col_l]

    valid = notna & positive_price & nonzero_vol & hl_ok
    removed_idx = d.index[~valid]

    return d.loc[valid].copy(), removed_idx

def get_kor_ticker_list_by_pykrx():
    tickers_kospi = get_safe_ticker_list(market="KOSPI")
    tickers_kosdaq = get_safe_ticker_list(market="KOSDAQ")
    tickers = tickers_kospi + tickers_kosdaq
    return tickers

def get_kor_ticker_list():
    # tickers_kospi = get_safe_ticker_list(market="KOSPI")
    # tickers_kosdaq = get_safe_ticker_list(market="KOSDAQ")
    # tickers = tickers_kospi + tickers_kosdaq

    url = "https://chickchick.shop/func/stocks/kor"
    res = requests.get(url)
    data = res.json()
    tickers = [item["stock_code"] for item in data if "stock_code" in item]
    return tickers

def get_kor_ticker_dict_list():
    url = "https://chickchick.shop/func/stocks/kor"
    res = requests.get(url)
    data = res.json()
    return {
        item["stock_code"]: item["stock_name"]
        for item in data
        if "stock_code" in item and "stock_name" in item
    }

def get_kor_interest_ticker_dick_list():
    url = "https://chickchick.shop/func/stocks/interest/data"
    payload = {
        "date": today,
    }

    res = requests.post(url, json=payload)
    data = res.json()
    return {
        item["stock_code"]: item["stock_name"]
        for item in data
        if "stock_code" in item and "stock_name" in item
    }

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
    stock_data = yf.download(ticker, start=fromdate, end=todate, auto_adjust=True, progress=False)
    if stock_data.empty:
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




def create_dataset(dataset, look_back=30, idx=3):
    X, Y = [], []
    if len(dataset) < look_back:
        return np.array(X), np.array(Y)  # 빈 배열 반환
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, :])
        Y.append(dataset[i+look_back, idx])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

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
    t0_list = []
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
                      lr=5e-4, delta=1.0):
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
    model.compile(optimizer=Adam(lr), loss=Huber(delta=delta))
    return model


def create_model(input_shape, n_future):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(input_shape)),
        Dropout(0.2),
        LSTM(16, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(n_future)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


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
            match = re.search(r'\[(\d{6})\]\.png$', filename)
            if match:
                numbers.append(match.group(1))
    return numbers

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
def invert_scale(scaled_preds, scaler, feature_index=3):
    """
    scaled_preds: (샘플수, forecast_horizon) - 스케일된 종가 예측 결과
    scaler: 학습에 사용된 MinMaxScaler 객체
    feature_index: 종가 컬럼 인덱스(보통 3)
    """
    inv_preds = []
    for row in scaled_preds:
        temp = np.zeros((len(row), scaler.n_features_in_))
        temp[:, feature_index] = row  # 종가 위치에 예측값 할당
        inv = scaler.inverse_transform(temp)[:, feature_index]  # 역변환 후 종가만 추출
        inv_preds.append(inv)
    return np.array(inv_preds)


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

    # 한국/영문 칼럼 자동 식별
    col_o = _col(data, '시가',   'Open')
    col_h = _col(data, '고가',   'High')
    col_l = _col(data, '저가',   'Low')
    col_c = _col(data, '종가',   'Close')
    col_v = _col(data, '거래량', 'Volume')

    o, h, l, c, v = data[col_o], data[col_h], data[col_l], data[col_c], data[col_v]

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
    data['UltimateOsc'] = compute_ultimate_osc(h, l, c, 7, 14, 28)

    # ROC (percent)
    """
    ROC (Rate of Change)
    매수세(황소)와 매도세(곰) 힘을 비교. 양수면 매수세 우위
    추세의 속도 확인
    """
    data['ROC12_pct'] = compute_roc(c, 12, pct=True)

    # 안전 처리
    data.replace([np.inf, -np.inf], np.nan, inplace=True)


    return data


def check_column_types(data, columns):
    for col in columns:
        print(f"[{col}] type: {type(data[col])}, shape: {data[col].shape}")



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





def plot_candles_daily(
        data: pd.DataFrame,
        show_months: int = 5,
        title: str = "Daily — Bollinger Bands & Volume",
        ax_price=None,
        ax_volume=None,
        future_dates=None,           # iterable of datetime/date/str
        predicted_prices=None,       # iterable of float
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
    # plt.setp(ax_price.get_xticklabels(), rotation=45, ha='right')    # 회전/정렬
    plt.setp(ax_price.get_xticklabels(), rotation=0, ha='center')    # 회전/정렬
    ax_price.set_title(title, fontsize=14)
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc='upper left')

    # 아랫 패널: 거래량
    ax_volume.bar(x, df[col_v].to_numpy(), color=colors, alpha=1)
    ax_volume.set_ylabel('Volume (D)')
    ax_volume.grid(True, alpha=0.25)

    # x축 눈금(날짜 라벨)
    ds = df.index.strftime('%Y-%m-%d')
    tick_idx = np.arange(0, len(df), max(1, len(df)//10))
    ax_volume.set_xticks(tick_idx)
    # ax_volume.set_xticklabels(ds[tick_idx], rotation=45, ha='right')
    ax_volume.set_xticklabels(ds[tick_idx], rotation=0, ha='center')

    return fig, ax_price, ax_volume

def plot_candles_weekly(
        data: pd.DataFrame,
        show_months: int = 12,
        title: str = "Weekly — Bollinger Bands & Volume",
        ax_price=None,
        ax_volume=None,
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
    tick_idx = np.arange(0, len(w), max(1, len(w)//10))
    ax_volume.set_xticks(tick_idx)
    # ax_volume.set_xticklabels(dsw[tick_idx], rotation=45, ha='right')
    ax_volume.set_xticklabels(dsw[tick_idx], rotation=0, ha='center')

    return fig, ax_price, ax_volume

# def regression_metrics(y_true, y_pred):
#     # numpy array 변환
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#
#     # MSE, RMSE, MAE
#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#
#     # MAPE (0 division 방지)
#     mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
#
#     # SMAPE
#     smape = 100 * np.mean(
#         2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
#     )
#
#     # R²
#     r2 = r2_score(y_true, y_pred)
#
#     return {
#         "MSE": mse,
#         "RMSE": rmse,
#         "MAE": mae,
#         "MAPE (%)": mape,
#         "SMAPE (%)": smape,
#         "R2": r2
#     }



def _to_2d(a):
    a = np.array(a)
    return a if a.ndim == 2 else a.reshape(-1, 1)

def _smape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def _mape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return 100.0 * np.mean(np.abs((y_true - y_pred) / denom))

def regression_metrics(
        y_true, y_pred, *,
        y_scaler=None,          # 타깃 전용 스케일러가 있을 때 사용
        scaler=None,            # 전체 피처 스케일러(지금 케이스)
        n_features=None,
        idx_close=None
):
    y_true = _to_2d(y_true)
    y_pred = _to_2d(y_pred)

    # scaled-space
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = _mape(y_true, y_pred)
    smape = _smape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    out = {"scaled": {
        "MSE": float(mse), "RMSE": float(rmse), "MAE": float(mae),
        "MAPE (%)": float(mape), "SMAPE (%)": float(smape), "R2": float(r2)
    }}

    # restore if possible
    restored_true = restored_pred = None

    if y_scaler is not None:
        restored_true = y_scaler.inverse_transform(y_true)
        restored_pred = y_scaler.inverse_transform(y_pred)

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

    return out


def pass_filter(metrics, use_restored=True, r2_min=0.6, smape_max=30.0, mape_max=50.0):
    m = metrics["restored"] if (use_restored and "restored" in metrics) else metrics["scaled"]
    print("DEBUG >> R2:", m["R2"], "SMAPE:", m["SMAPE (%)"])  # <- 디버깅
    # return (m["R2"] >= r2_min) and (m["SMAPE (%)"] < smape_max) and (m["MAPE (%)"] < mape_max)
    return (m["R2"] >= r2_min) and (m["SMAPE (%)"] <= smape_max)



# 5일선이 20일선 아래에서 근접하게 다가옴
def near_bull_cross_signal(df: pd.DataFrame,
                           lookback: int = 5,
                           gap_bp: float = 0.004,    # MA20 대비 0.4% 이내
                           min_rise_bp: float = 0.002 # delta가 최근 lookback동안 최소 0.2%p 이상 상승
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

    # 현재 격차가 충분히 좁은가? (MA20의 퍼센트 기준)
    tight_now = abs(delta.iloc[-1]) <= (s_ma20.iloc[-1] * gap_bp)

    # 최근 lookback일 동안 delta가 유의미하게 상승(즉, 더 0에 가까워짐)
    delta_rise = (delta.iloc[-1] - delta.iloc[-lookback]) >= (s_ma20.iloc[-1] * min_rise_bp)

    # 보조: MA5 기울기 양수, MA20 기울기 비양수(완화 가능)
    ma5_slope = s_ma5.iloc[-1] - s_ma5.iloc[-2]
    ma20_slope = s_ma20.iloc[-1] - s_ma20.iloc[-2]
    slope_ok = (ma5_slope > 0) and (ma20_slope <= 0)

    return bool(tight_now and delta_rise and slope_ok)

