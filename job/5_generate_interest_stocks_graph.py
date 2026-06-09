"""
관심 종목의 오늘 그래프 생성
"""
import matplotlib
matplotlib.use("Agg")  # ✅ 비인터랙티브 백엔드 (창 안 띄움)
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import requests
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import add_technical_features, plot_candles_weekly, plot_candles_daily, drop_sparse_columns, \
    drop_trading_halt_rows, get_kor_summary_ticker_dict_list, get_favorite_ticker_dict_list, \
    get_stock_name, is_korean_stock_business_day, get_stock_created_at, extract_column, get_low_ticker_dict, \
    convert_to_yymmdd

# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root
data_dir = os.path.join(project_root, "data")
pickle_dir = os.path.join(data_dir, "pickle")

year = datetime.now().strftime("%Y")
month = datetime.now().strftime("%m")
day = datetime.now().strftime("%d")




def process_one(idx, ticker, tickers_dict, low_tickers_cdate_dict, low_tickers_graph_dict):
    stock_name = get_stock_name(tickers_dict, ticker)
    created_at_list = get_stock_created_at(low_tickers_cdate_dict, ticker)


    filepath = os.path.join(pickle_dir, f'{ticker}.pkl')

    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(filepath)

        if os.path.getsize(filepath) == 0:
            raise EOFError("⚠️ pickle 파일이 비어 있습니다.")

        df = pd.read_pickle(filepath)

    except (EOFError, FileNotFoundError) as e:
        print(f"⚠️ pickle 파일을 읽을 수 없습니다-5: {filepath}")
        print(e)
        return


    # 데이터가 부족하면 패스
    if df is None or df.empty or len(df) < 50:
        return

    # 거래정지/이상치 행 제거
    df, _ = drop_trading_halt_rows(df)

    # 2차 생성 feature
    df = add_technical_features(df)

    # 결측 제거
    df, _ = drop_sparse_columns(df, threshold=0.10, check_inf=True, inplace=True)

    # drop 이후 다시 생성
    df = add_technical_features(df)

    # 데이터가 부족하면 패스
    if df.empty or len(df) < 50:
        return


    data = df
    today = data.index[-1].strftime("%Y%m%d") # 마지막 인덱스

    today_str = str(today)

    if created_at_list:
        graph_file_name = low_tickers_graph_dict.get(ticker)
        final_file_name = graph_file_name
        title = graph_file_name.rsplit(".", 1)[0]
    else:
        final_file_name = f"{today_str} {stock_name} [{ticker}].webp"
        title = f"{today_str} {stock_name} [{ticker}] Daily Chart"

    year = today[:4]
    month = today[4:6]
    day = today[6:8]

    if created_at_list:
        output_dir = r"F:\5below20"
    else:
        output_dir = f"F:\\interest_stocks\\{year}\\{month}\\{day}"

    os.makedirs(output_dir, exist_ok=True)
    final_file_path = os.path.join(output_dir, final_file_name)

    # 그래프 그릴 때 필요한 것만 모아서 리턴
    plot_job = {
        "ticker": ticker,
        "origin": data,
        "today": today_str,
        "created_at_list": created_at_list,
        "title": title,
        "save_path": final_file_path,
    }


    if not created_at_list:
        try:
            requests.post(
                'https://chickchick.kr/stocks/interest/graph',
                json={
                    "nation": "kor",
                    "stock_code": str(ticker),
                    "graph_file": str(final_file_name),
                },
                timeout=10
            )
        except Exception as e:
            # logging.warning(f"progress-update 요청 실패: {e}")
            print(f"⚠️ progress-update 요청 실패-5-1: {e}")
            pass  # 오류
    else:
        for created_at in created_at_list:
            try:
                requests.post(
                    'https://chickchick.kr/stocks/low/graph',
                    json={
                        "nation": "kor",
                        "stock_code": str(ticker),
                        "created_at": str(created_at),
                        "graph_file": str(final_file_name),
                    },
                    timeout=10
                )
            except Exception as e:
                # logging.warning(f"progress-update 요청 실패: {e}")
                print(f"⚠️ progress-update 요청 실패-5-2: {e}")
                pass  # 오류


    return {
        "plot_job": plot_job,
    }



if __name__ == "__main__":
    if not is_korean_stock_business_day(verbose=False):
        # print("한국증시 영업일이 아니므로 실행하지 않습니다.")
        sys.exit(0)

    start = time.time()   # 시작 시간(초)
    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - 🕒 running 5_generate_interest_stocks_graph.py...')

    tickers_dict = get_kor_summary_ticker_dict_list()  # {code: name}
    fav_tickers_dict = get_favorite_ticker_dict_list()

    low_data_dict = get_low_ticker_dict("all")
    low_tickers_dict = extract_column(low_data_dict, "stock_name")
    # low_tickers_dict = get_low_ticker_dict("stock_name")

    tickers_dict.update(fav_tickers_dict)
    tickers_dict.update(low_tickers_dict)
    tickers = list(set(tickers_dict.keys()))  # | 는 합집합 연산자

    low_tickers_cdate_dict = {}
    for code, row in low_data_dict.items():
        if isinstance(row, dict) and row.get("created_at"):
            low_tickers_cdate_dict.setdefault(code, []).append(convert_to_yymmdd(row["created_at"]))
    low_tickers_graph_dict = extract_column(low_data_dict, "graph_file")


    # 테스트
    # tickers_dict = get_low_ticker_dict("stock_name")
    # low_tickers_cdate_dict = get_low_ticker_dict("created_at")
    # tickers = list(set(tickers_dict.keys()))

    plot_jobs = []

    origin_idx = idx = -1  # 오늘 // 3 (5일 전)
    # origin_idx = idx = 1
    workers = os.cpu_count()
    # with ThreadPoolExecutor(max_workers=workers) as executor:   # GIL(Global Interpreter Lock) >> I/O가 많은 경우
    with ProcessPoolExecutor(max_workers=workers-4) as executor:   # CPU를 진짜로 병렬로 돌리고 싶으면 >> CPU연산이 많은 경우
        futures = []

        while idx <= origin_idx:
            idx += 1
            for count, ticker in enumerate(tickers):
                futures.append(executor.submit(process_one, idx, ticker, tickers_dict, low_tickers_cdate_dict, low_tickers_graph_dict))

        # 완료된 것부터 하나씩 받아서 집계
        for f in as_completed(futures):
            try:
                res = f.result()
            except Exception as e:
                print("worker error:", e)
                continue

            if res is None:
                continue

            plot_job = res["plot_job"]
            plot_jobs.append(plot_job)


    # 싱글 스레드로 그래프 처리
    for job in plot_jobs:
        # 그래프 생성
        fig = plt.figure(figsize=(14, 16), dpi=150)
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[3, 1, 3, 1])

        ax_d_price = fig.add_subplot(gs[0, 0])
        ax_d_vol   = fig.add_subplot(gs[1, 0], sharex=ax_d_price)
        ax_w_price = fig.add_subplot(gs[2, 0])
        ax_w_vol   = fig.add_subplot(gs[3, 0], sharex=ax_w_price)

        created_at = None

        # created_at_list가 리스트인지 확인
        if isinstance(job["created_at_list"], list):
            # 여러 날짜가 들어있으면 반복
            for created_at in job["created_at_list"]:
                plot_candles_daily(
                    job["origin"],
                    show_months=4,
                    title=job["title"],
                    ax_price=ax_d_price,
                    ax_volume=ax_d_vol,
                    date_tick=5,
                    today=created_at,
                )

                plot_candles_weekly(
                    job["origin"],
                    show_months=12,
                    title="Weekly Chart",
                    ax_price=ax_w_price,
                    ax_volume=ax_w_vol,
                    date_tick=5,
                    today=created_at,
                )

        else:
            # 단일 값이면 한 번만 실행
            if job["created_at_list"] is not None:
                created_at = convert_to_yymmdd(job["created_at_list"])

            plot_candles_daily(
                job["origin"],
                show_months=4,
                title=job["title"],
                ax_price=ax_d_price,
                ax_volume=ax_d_vol,
                date_tick=5,
                today=created_at,
            )

            plot_candles_weekly(
                job["origin"],
                show_months=12,
                title="Weekly Chart",
                ax_price=ax_w_price,
                ax_volume=ax_w_vol,
                date_tick=5,
                today=created_at,
            )


        plt.tight_layout()
        # plt.show()

        # 파일 저장 (옵션)
        plt.savefig(job["save_path"], format="webp", dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    end = time.time()     # 끝 시간(초)
    elapsed = end - start

    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    nowTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    print(f'{nowTime} - Complete : 5_generate_interest_stocks_graph.py, 총 소요 시간: {hours}시간 {minutes}분 {seconds}초')


