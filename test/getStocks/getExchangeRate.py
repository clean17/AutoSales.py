import requests
from bs4 import BeautifulSoup

def get_usd_krw_rate():
    url = "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    # 환율 값은 <span class="value">1380.50</span> 이런 식으로 나옴
    em_tag = soup.find('em', class_='no_down')
    # 모든 span 태그의 텍스트를 이어붙임
    rate_str = ''.join([span.get_text() for span in em_tag.find_all('span')])
    return float(rate_str.replace(',', ''))

rate = get_usd_krw_rate()
print("현재 원/달러 환율:", rate)

