from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta
from openpyxl.utils import get_column_letter

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

# 펀더멘탈 데이터 가져오기
펀더멘탈 = stock.get_market_fundamental_by_date(작년, 오늘, 종목_코드, freq='d')

# 데이터 병합
병합된_데이터 = pd.merge(ohlcv, 펀더멘탈, left_index=True, right_index=True)

# Excel 파일로 저장
with pd.ExcelWriter('삼성전자_OHLCV_펀더멘탈_데이터.xlsx', engine='openpyxl') as excel_writer:
    병합된_데이터.to_excel(excel_writer, index=True)
    workbook = excel_writer.book
    worksheet = workbook.active
    for column_cells in worksheet.columns:
        length = max(len(str(cell.value)) for cell in column_cells)
        adjusted_width = length * 1.4  # 여기에서 너비에 40%를 추가합니다.
        worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = adjusted_width

print("Excel 파일이 저장되었습니다. 열 너비가 원래 길이의 140%로 조정되었습니다.")
