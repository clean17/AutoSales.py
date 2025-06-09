import pandas as pd
'''
pandas는 파이썬에서 데이터를 쉽게 다루고 분석할 수 있게 도와주는 라이브러리(패키지)
특히 표(엑셀, CSV 파일처럼 행과 열이 있는 데이터)를 다루는 데 아주 강력
'''

url = "https://en.wikipedia.org/wiki/Nasdaq-100"
tables = pd.read_html(url)

nasdaq_100_table_index = 4
nasdaq_100_df = tables[nasdaq_100_table_index]

# print(nasdaq_100_df.head())
# print(nasdaq_100_df)
'''
    Ticker                  Company             GICS Sector               GICS Sub-Industry
0     ADBE               Adobe Inc.  Information Technology            Application Software
1      AMD   Advanced Micro Devices  Information Technology                  Semiconductors
2     ABNB                   Airbnb  Consumer Discretionary  Hotels, Resorts & Cruise Lines
3    GOOGL  Alphabet Inc. (Class A)  Communication Services    Interactive Media & Services
4     GOOG  Alphabet Inc. (Class C)  Communication Services    Interactive Media & Services
..     ...                      ...                     ...                             ...
96    VRTX   Vertex Pharmaceuticals             Health Care                   Biotechnology
97     WBD   Warner Bros. Discovery  Communication Services                    Broadcasting
98    WDAY            Workday, Inc.  Information Technology            Application Software
99     XEL              Xcel Energy               Utilities                 Multi-Utilities
100     ZS                  Zscaler  Information Technology            Application Software

[101 rows x 4 columns]
'''


# If 'Symbol' or 'Ticker' is the correct column name
tickers = nasdaq_100_df['Ticker'].tolist()  # Replace 'Symbol' with the correct column name if different
print(tickers)
'''
['ADBE', 'AMD', 'ABNB', 'GOOGL', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'ANSS', 'AAPL', 'AMAT',
 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB', 'BKNG', 'AVGO',
 'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST',
 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GILD',
 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR',
 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT', 'MSTR', 'MDLZ', 'MNST', 'NFLX', 'NVDA', 'NXPI',
 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR', 'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN',
 'ROP', 'ROST', 'SHOP', 'SBUX', 'SNPS', 'TTWO', 'TMUS', 'TSLA', 'TXN', 'TTD', 'VRSK', 'VRTX',
 'WBD', 'WDAY', 'XEL', 'ZS']
'''

