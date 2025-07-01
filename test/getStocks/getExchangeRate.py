import requests
from bs4 import BeautifulSoup

def get_usd_krw_rate(url="https://m.search.naver.com/p/csearch/content/qapirender.nhn?key=calculator&pkid=141&q=%ED%99%98%EC%9C%A8&where=m&u1=keb&u6=standardUnit&u7=0&u3=USD&u4=KRW&u8=down&u2=1"):
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


rate = get_usd_krw_rate()
print("현재 원/달러 환율:", rate)
