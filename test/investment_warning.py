from pykrx import stock
import datetime

def check_warning_release():
    today = datetime.datetime.today().strftime('%Y%m%d')

    # 현재 투자경고 종목 목록
    warnings = stock.get_market_invest_warning()
    codes = warnings.index.tolist()

    release_candidates = []

    for code in codes:
        # 최근 n일 데이터 조회
        df = stock.get_market_ohlcv_by_date(start, today, code)

        if not meets_warning_condition(df):     # 투자경고 조건 재충족 여부 검사
            release_candidates.append(code)

    return release_candidates


check_warning_release()