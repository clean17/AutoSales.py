from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta

# 날짜 설정
오늘 = datetime.today()
작년 = 오늘 - timedelta(days=365)
오늘 = 오늘.strftime('%Y%m%d')
작년 = 작년.strftime('%Y%m%d')

# 삼성전자의 종목 코드
종목_코드 = '005930'

# OHLCV 데이터 가져오기
ohlcv = stock.get_market_ohlcv_by_date(작년, 오늘, 종목_코드)

# 거래대금 정보 추가 (시가 * 거래량)
ohlcv['거래대금'] = ohlcv['시가'] * ohlcv['거래량']

# 등락률 정보 가져오기
등락률 = stock.get_market_price_change_by_ticker(작년, 오늘).loc[종목_코드, '등락률']

# 펀더멘탈 데이터 가져오기
펀더멘탈 = stock.get_market_fundamental_by_date(작년, 오늘, 종목_코드, freq='d')

# 데이터 병합 (여기서는 OHLCV 데이터에 펀더멘탈 데이터를 병합)
병합된_데이터 = pd.merge(ohlcv, 펀더멘탈, left_index=True, right_index=True)

# 등락률을 병합된 데이터에 추가합니다. 이는 일별 데이터가 아니기 때문에, 별도의 처리가 필요할 수 있습니다.
# 병합된_데이터['등락률'] = 등락률

# CSV 파일로 저장
병합된_데이터.to_csv('삼성전자_OHLCV_펀더멘탈_데이터.csv', encoding='utf-8-sig')

print("CSV 파일이 저장되었습니다.")