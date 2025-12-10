from datetime import datetime, timedelta
from typing import List, Dict, Any, Iterable
from pykrx import stock
import requests
import itertools
import time

def get_safe_ticker_list(market="KOSPI"):
    def fetch_tickers_for_date(date):
        try:
            tickers = stock.get_market_ticker_list(market=market, date=date)
            # 데이터가 비어 있다면 예외를 발생시킴
            if not tickers:
                raise ValueError("Ticker list is empty")
            return tickers
        except (IndexError, ValueError) as e:
            return []

    # 현재 날짜로 시도
    today = datetime.now().strftime("%Y%m%d")
    tickers = fetch_tickers_for_date(today)

    # 첫 번째 시도가 실패한 경우 과거 날짜로 반복 시도
    if not tickers:
        print("데이터가 비어 있습니다. 가장 가까운 영업일로 재시도합니다.")
        for days_back in range(1, 7):
            previous_day = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            tickers = fetch_tickers_for_date(previous_day)
            if tickers:  # 성공적으로 데이터를 가져오면 반환
                return tickers

        print("영업일 데이터를 찾을 수 없습니다.")
        return []

    return tickers

def unique_keep_order(seq: Iterable[str]) -> List[str]:
    """순서 유지하면서 중복 제거"""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_stock_payload(market: str = "KOSPI") -> List[Dict[str, Any]]:
    """
    tickers -> [{nation, stock_code, stock_name, sector_code, stock_market}, ...]
    """
    tickers = get_safe_ticker_list(market=market)  # ["005930", "000660", ...]
    tickers = unique_keep_order(tickers)

    # 티커명 매핑 (속도 위해 dict 미리 생성)
    ticker_to_name = {t: stock.get_market_ticker_name(t) for t in tickers}

    nation = "kor"
    stock_market = market.lower()
    payload = []
    for t in tickers:
        payload.append({
            "nation": nation,
            "stock_code": t,
            "stock_name": ticker_to_name.get(t) or "Unknown Stock",
            "sector_code": None,
            "stock_market": stock_market,
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
    # 1) 페이로드 구성
    print('update KOSPI')
    payload = build_stock_payload(market="KOSPI")
    # 2) 전송 (엔드포인트 주소 맞춰서 변경)
    API_URL = "https://chickchick.shop/func/stocks/update"
    post_stocks_update(API_URL, payload, batch_size=500)

    print('update KOSDAQ')
    payload = build_stock_payload(market="KOSDAQ")
    # 2) 전송 (엔드포인트 주소 맞춰서 변경)
    API_URL = "https://chickchick.shop/func/stocks/update"
    post_stocks_update(API_URL, payload, batch_size=500)

    # 항상 post_stocks_update() 다음 /delisted-stock을 요청해야한다
    print('delete delisted stock')
    payload = {}
    API_URL = "https://chickchick.shop/func/stocks/delisted-stock"
    with requests.Session() as s:
        headers = {"Content-Type": "application/json"}
        try:
            resp = s.post(API_URL, json={}, headers=headers, timeout=15.0)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise


