import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from datetime import datetime
from typing import List, Dict, Any, Iterable
import requests
import itertools
import time
import pandas as pd

KIND_LIST_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"

def fetch_kospi_kosdaq_payload_from_kind(url: str = KIND_LIST_URL) -> list:
    """
    KIND 상장법인목록.xls(실제는 HTML 테이블일 수 있음)에서
    회사명/시장구분/종목코드/업종을 가져와 payload 리스트로 반환.
    - 시장구분: '유가' -> stock_market='kospi', '코스닥' -> 'kosdaq'
    """

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://kind.krx.co.kr/",
    }

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    # KIND는 한글 인코딩이 cp949/euc-kr인 경우가 흔함
    # requests가 인코딩을 못 맞추는 경우가 있어서 보정
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = "cp949"

    html = r.text

    # 'xls'지만 실제로는 HTML table인 경우가 많아서 read_html 사용
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("KIND 응답에서 테이블을 찾지 못했습니다.")

    df = tables[0].copy()

    # 컬럼명 정리(공백/탭 등 제거)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["회사명", "시장구분", "종목코드", "업종"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"필수 컬럼이 없습니다: {missing}. 현재 컬럼: {list(df.columns)}")

    df = df[required].copy()

    # 시장구분 필터 및 매핑
    market_map = {"유가": "kospi", "코스닥": "kosdaq"}
    df = df[df["시장구분"].isin(market_map.keys())].copy()
    df["stock_market"] = df["시장구분"].map(market_map)

    # 종목코드 6자리로 보정 (숫자로 읽히면 0이 날아감)
    df["종목코드"] = df["종목코드"].astype(str).str.strip().str.zfill(6)

    payload = []
    for _, row in df.iterrows():
        payload.append({
            "nation": "kor",
            "stock_code": row["종목코드"],
            "stock_name": row["회사명"],
            "sector_code": None,
            "stock_market": row["stock_market"],   # kospi / kosdaq
            "category": row["업종"],
        })

    return payload


def chunked(iterable: List[Any], size: int) -> Iterable[List[Any]]:
    """size 단위로 분할"""
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, size))
        if not batch:
            break
        yield batch


def post_stocks_update(
        url: str,
        payload: List[Dict[str, Any]],
        batch_size: int = 500,
        timeout: float = 15.0,
        max_retries: int = 2,
        retry_backoff_sec: float = 1.0,
) -> None:
    """
    /stocks/update 엔드포인트로 배치 전송.
    Flask 쪽은 JSON 배열을 받아서 DTO 리스트로 파싱하는 구조.
    """
    with requests.Session() as s:
        headers = {"Content-Type": "application/json"}
        for idx, batch in enumerate(chunked(payload, batch_size), start=1):
            for attempt in range(1, max_retries + 2):  # 1 + retries
                try:
                    resp = s.post(url, json=batch, headers=headers, timeout=timeout)
                    resp.raise_for_status()
                    print(f"[{idx}] batch({len(batch)} rows) OK")
                    break
                except requests.RequestException as e:
                    if attempt <= max_retries:
                        wait = retry_backoff_sec * attempt
                        print(f"[{idx}] batch FAIL (attempt {attempt}): {e} -> retry in {wait}s")
                        time.sleep(wait)
                    else:
                        # 재시도 모두 실패
                        raise


if __name__ == "__main__":
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print('─────────────────────────────────────────────────────────────')
    print(f'{nowTime} - 🕒 update_kor_stocks_periodically')
    print('─────────────────────────────────────────────────────────────')
    isEmpty = False

    # 1) 페이로드 구성
    print('update korea stock market')
    payload = fetch_kospi_kosdaq_payload_from_kind()
    print(payload[:1])
    if len(payload) == 0:
        isEmpty = True
    # 2) 전송 (엔드포인트 주소 맞춰서 변경)
    API_URL = "https://chickchick.kr/stocks/update"
    post_stocks_update(API_URL, payload, batch_size=500)

    # 항상 post_stocks_update() 다음 /delisted-stock을 요청해야한다
    if isEmpty == False:
        print('delete delisted stock')
        payload = {}
        API_URL = "https://chickchick.kr/stocks/delisted-stock"
        with requests.Session() as s:
            headers = {"Content-Type": "application/json"}
            try:
                resp = s.post(API_URL, json={}, headers=headers, timeout=15.0)
                resp.raise_for_status()
            except requests.RequestException as e:
                raise