import pandas as pd

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

# S&P 500 종목 가져오기
sp500_tickers = get_sp500_tickers() # 503 종목

# 나스닥 100 종목 가져오기
nasdaq100_tickers = get_nasdaq100_tickers() # 101 종목

us_set = set(sp500_tickers)
us_set.update(nasdaq100_tickers)

# nasdaq100_not_in_sp500 = [ticker for ticker in nasdaq100_tickers if ticker not in sp500_tickers]
# print("S&P 500에 속하지 않은 나스닥 100 종목:")
# print(nasdaq100_not_in_sp500)

print(list(us_set))