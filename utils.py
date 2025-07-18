from datetime import datetime, timedelta
from pykrx import stock
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import yfinance as yf
import os
import re
from bs4 import BeautifulSoup

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def get_kor_ticker_list():
    tickers_kospi = get_safe_ticker_list(market="KOSPI")
    tickers_kosdaq = get_safe_ticker_list(market="KOSDAQ")
    tickers = tickers_kospi + tickers_kosdaq
    return tickers

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
def create_multistep_dataset(dataset, look_back, n_future, idx=3):
    X, Y = [], []
    for i in range(len(dataset) - look_back - n_future + 1):
        X.append(dataset[i:i+look_back])
        # Y는 "종가" 인덱스만 n_future 길이로 슬라이싱!
        Y.append(dataset[i+look_back:i+look_back+n_future, idx]) # 3번 인덱스; 종가
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
                      dense_units=[16, 8]):
    """
    input_shape: (look_back, feature 수)
    n_future: 예측할 값 개수
    lstm_units: LSTM 레이어별 유닛 리스트 (예: [128, 64]면 LSTM 128, LSTM 64)
    dropout: 각 LSTM 뒤에 붙일 Dropout 비율(리스트로 주면 각각 다르게도 가능)
    dense_units: Dense 레이어별 유닛 리스트
    """
    model = Sequential()
    for i, units in enumerate(lstm_units):
        return_seq = (i < len(lstm_units) - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(LSTM(units, return_sequences=return_seq))
        # Dropout 리스트를 받을 수도 있고, 고정값을 쓸 수도 있음
        if dropout:
            model.add(Dropout(dropout if isinstance(dropout, float) else dropout[i]))
    for units in dense_units:
        model.add(Dense(units, activation='relu'))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# def create_model(input_shape, n_future):
#     model = Sequential([
#         LSTM(32, return_sequences=True, input_shape=(input_shape)),
#         Dropout(0.2),
#         LSTM(16, return_sequences=False),
#         Dropout(0.2),
#         Dense(16, activation='relu'),
#         Dense(8, activation='relu'),
#         Dense(n_future)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model


def compute_rsi(prices, period=14):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period, min_periods=1).mean()
    loss = down.rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

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


def add_technical_features(data, window=20, num_std=2):

    # RSI (14일)
    data['RSI14'] = compute_rsi(data['종가'])  # 사전에 정의 필요

    # 볼린저밴드 (MA20, STD20, 상단/하단 밴드)
    data['MA20'] = data['종가'].rolling(window=window).mean()
    data['STD20'] = data['종가'].rolling(window=window).std()
    data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
    data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])
    # 볼린저밴드 위치 (0~1)
    data['BB_perc'] = (data['종가'] - data['LowerBand']) / (data['UpperBand'] - data['LowerBand'] + 1e-9)

    # 이동평균선
    data['MA5'] = data['종가'].rolling(window=5).mean()
    data['MA10'] = data['종가'].rolling(window=10).mean()
    data['MA5_slope'] = data['MA5'].diff()
    data['MA10_slope'] = data['MA10'].diff()
    data['MA20_slope'] = data['MA20'].diff()

    # 거래량 증감률
    data['Volume_change'] = data['거래량'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    # 당일 변동폭 (고가-저가 비율)
    data['day_range_pct'] = (data['고가'] - data['저가']) / (data['저가'] + 1e-9)

    # 캔들패턴 (양봉/음봉, 장대양봉 등)
    # data['is_bullish'] = (data['종가'] > data['시가']).astype(int) # 양봉이면 1, 음봉이면 0
    # 장대양봉(시가보다 종가가 2% 이상 상승)
    # data['long_bullish'] = ((data['종가'] - data['시가']) / data['시가'] > 0.02).astype(int)

    # 최근 N일간 등락률
    data['chg_5d'] = (data['종가'] / data['종가'].shift(5)) - 1
    # 현재가 vs 이동평균(MA) 괴리율
    data['ma5_gap'] = (data['종가'] - data['MA5']) / data['MA5']
    data['ma10_gap'] = (data['종가'] - data['MA10']) / data['MA10']
    data['ma20_gap'] = (data['종가'] - data['MA20']) / data['MA20']
    # 거래량 급증 신호
    data['volume_ratio'] = data['거래량'] / data['거래량'].rolling(20).mean()

    return data

def add_technical_features_us(data, window=20, num_std=2):

    # RSI (14일)
    data['RSI14'] = compute_rsi(data['Close'])  # 사전에 정의 필요

    # 볼린저밴드 (MA20, STD20, 상단/하단 밴드)
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['STD20'] = data['Close'].rolling(window=window).std()
    data['UpperBand'] = data['MA20'] + (num_std * data['STD20'])
    data['LowerBand'] = data['MA20'] - (num_std * data['STD20'])
    # 볼린저밴드 위치 (0~1)
    data['BB_perc'] = (data['Close'] - data['LowerBand']) / (data['UpperBand'] - data['LowerBand'] + 1e-9)

    # 이동평균선
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA5_slope'] = data['MA5'].diff()
    data['MA10_slope'] = data['MA10'].diff()
    data['MA20_slope'] = data['MA20'].diff()

    # 거래량 증감률
    data['Volume_change'] = data['Volume'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    # 당일 변동폭 (고가-저가 비율)
    data['day_range_pct'] = (data['High'] - data['Low']) / (data['Low'] + 1e-9)

    # 캔들패턴 (양봉/음봉, 장대양봉 등)
    # data['is_bullish'] = (data['Close'] > data['Open']).astype(int) # 양봉이면 1, 음봉이면 0
    # 장대양봉(시가보다 종가가 2% 이상 상승)
    # data['long_bullish'] = ((data['Close'] - data['Open']) / data['Open'] > 0.02).astype(int)

    # 현재가 vs 이동평균(MA) 괴리율
    data['ma10_gap'] = (data['Close'] - data['MA10']) / data['MA10']

    return data

def check_column_types(data, columns):
    for col in columns:
        print(f"[{col}] type: {type(data[col])}, shape: {data[col].shape}")