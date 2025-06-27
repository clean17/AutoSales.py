from pykrx import stock
import pandas as pd
from datetime import datetime

# KOSPI와 KOSDAQ 시장의 종목 리스트 가져오기
tickers_kospi = stock.get_market_ticker_list(market="KOSPI")
tickers_kosdaq = stock.get_market_ticker_list(market="KOSDAQ")
tickers = tickers_kospi + tickers_kosdaq

# PBR이 없는 종목 리스트
no_pbr_tickers = []

# 기준 날짜 설정 (예: 최근 데이터)
today = datetime.today().strftime('%Y%m%d')

# 각 종목의 펀더멘탈 데이터에서 PBR 확인
for ticker in tickers:
    daily_fundamental = stock.get_market_fundamental_by_date(today, today, ticker)

    # PBR 데이터가 없는지 확인
    if 'PBR' not in daily_fundamental.columns or pd.isna(daily_fundamental['PBR'].values[0]):
        stock_name = stock.get_market_ticker_name(ticker)
        no_pbr_tickers.append((ticker, stock_name))

# PBR이 없는 종목 출력
print("PBR이 없는 종목:")
for ticker, name in no_pbr_tickers:
    print(f"{ticker}: {name}")

'''
PBR이 없는 종목:

365550: ESR켄달스퀘어리츠
415640: KB발해인프라
432320: KB스타리츠
400760: NH올원리츠
338100: NH프라임리츠
395400: SK리츠
377190: 디앤디플랫폼리츠
330590: 롯데리츠
357430: 마스턴프리미어리츠
088980: 맥쿼리인프라
094800: 맵스리얼티1
396690: 미래에셋글로벌리츠
357250: 미래에셋맵스리츠
448730: 삼성FN리츠
204210: 스타에스엠리츠
481850: 신한글로벌액티브리츠
404990: 신한서부티엔디리츠
900120: 씨엑스아이
900100: 애머릿지
950130: 엑세스바이오
900300: 오가닉티코스메틱
900340: 윙입푸드
900110: 이스트아시아홀딩스
950140: 잉글우드랩
900310: 컬러레이
950160: 코오롱티슈진
900250: 크리스탈신소재
900270: 헝셩그룹
'''