from datetime import datetime, timedelta
from pykrx import stock
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
import yfinance as yf

# 시드 고정
import numpy as np, tensorflow as tf, random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

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



# 주식 데이터(시가, 고가, 저가, 종가, 거래량)와 재무 데이터(PER)를 가져온다
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

    # yfinance를 통해 주식 정보 가져오기
    stock_info = yf.Ticker(ticker).info

    # PER 값을 info에서 추출, 없는 경우 0으로 처리
    per_value = stock_info.get('trailingPE', 0)  # trailingPE를 사용하거나 없으면 0

    # PBR 값을 info에서 추출, 없는 경우 0으로 처리
    pbr_value = stock_info.get('priceToBook', 0)

    # 주식 데이터에 PER 컬럼 추가
    stock_data['PER'] = per_value
    stock_data['PBR'] = pbr_value

    # 선택적인 컬럼 추출 및 NaN 값 처리
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'PER', 'PBR']].fillna(0)
    return stock_data



def create_dataset(dataset, look_back=30):
    X, Y = [], []
    if len(dataset) < look_back:
        return np.array(X), np.array(Y)  # 빈 배열 반환
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, :])
        Y.append(dataset[i+look_back, 3])  # 종가(Close) 예측
    return np.array(X), np.array(Y)

def create_multistep_dataset(dataset, look_back, n_future):
    X, Y = [], []
    for i in range(len(dataset) - look_back - n_future + 1):
        X.append(dataset[i:i+look_back])
        # Y는 "종가" 인덱스만 n_future 길이로 슬라이싱!
        Y.append(dataset[i+look_back:i+look_back+n_future, 3]) # 3번 인덱스; 종가
    return np.array(X), np.array(Y)



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