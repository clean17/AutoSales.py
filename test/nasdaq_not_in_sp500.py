import yfinance as yf
import pandas as pd

# S&P 500 종목을 가져오는 함수
def get_sp500_tickers():
    sp500 = yf.Ticker("^GSPC")  # S&P 500 지수 티커
    sp500_constituents = sp500.history(period="1d")  # 가격 데이터를 가져옴
    sp500_constituents = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]  # Wikipedia에서 종목 가져오기
    sp500_tickers = sp500_constituents['Symbol'].tolist()  # 티커 리스트 추출
    return sp500_tickers

# 나스닥 100 종목을 가져오는 함수
def get_nasdaq100_tickers():
    nasdaq100 = yf.Ticker("^NDX")  # 나스닥 100 지수 티커
    nasdaq100_constituents = nasdaq100.history(period="1d")  # 가격 데이터를 가져옴
    nasdaq100_constituents = pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]  # Wikipedia에서 종목 가져오기
    nasdaq100_tickers = nasdaq100_constituents['Ticker'].tolist()  # 티커 리스트 추출
    return nasdaq100_tickers

# S&P 500 종목 가져오기
sp500_tickers = get_sp500_tickers()

# 나스닥 100 종목 가져오기
nasdaq100_tickers = get_nasdaq100_tickers()

# S&P 500에 속하지 않은 나스닥 100 종목 찾기
nasdaq100_not_in_sp500 = [ticker for ticker in nasdaq100_tickers if ticker not in sp500_tickers]

# 결과 출력
print("S&P 500에 속하지 않은 나스닥 100 종목:")
print(nasdaq100_not_in_sp500)
