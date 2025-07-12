from pykrx import stock
from datetime import datetime, timedelta

def is_next_day_trading_day():
    today = datetime.today()
    tomorrow = today + timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y%m%d')

    # 내일 날짜가 실제로 영업일이면 반환값이 내일과 같음
    nearest_day = stock.get_nearest_business_day_in_a_week(tomorrow_str)
    return tomorrow_str == nearest_day

if __name__ == "__main__":
    if is_next_day_trading_day():
        print("내일은 영업일입니다.")
    else:
        print("내일은 휴장일입니다.")
