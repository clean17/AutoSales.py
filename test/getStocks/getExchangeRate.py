import requests
from bs4 import BeautifulSoup

def get_usd_krw_rate():
    url = "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    target_em = soup.select_one('p.no_today em.no_up em.no_up')
    if target_em:
        rate_str = ''.join([span.get_text() for span in target_em.find_all('span')])
        # 쉼표 제거, float 변환
        return float(rate_str.replace(',', ''))
    else:
        print('em.no_up em.no_up이 없습니다.')

rate = get_usd_krw_rate()
print("현재 원/달러 환율:", rate)
