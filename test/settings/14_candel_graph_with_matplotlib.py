from datetime import datetime, timedelta
import pandas as pd
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt


# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import fetch_stock_data, add_technical_features, plot_candles_weekly_standard, plot_candles_weekly, plot_candles_daily



# 데이터 수집
ticker = "007860"


DATA_COLLECTION_PERIOD = 400 # 샘플 수 = 68(100일 기준) - 20 - 4 + 1 = 45
# 현재 실행 파일 기준으로 루트 디렉토리 경로 잡기
root_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(=루트)
pickle_dir = os.path.join(root_dir, 'pickle')
start_five_date = (datetime.today() - timedelta(days=5)).strftime('%Y%m%d')
start_date = (datetime.today() - timedelta(days=DATA_COLLECTION_PERIOD)).strftime('%Y%m%d')
today = datetime.today().strftime('%Y%m%d')

# 데이터가 없으면 1년 데이터 요청, 있으면 5일 데이터 요청
filepath = os.path.join(pickle_dir, f'{ticker}.pkl')
if os.path.exists(filepath):
    df = pd.read_pickle(filepath)
    data = fetch_stock_data(ticker, start_five_date, today)
else:
    df = pd.DataFrame()
    data = fetch_stock_data(ticker, start_date, today)

# dropna; pandas DataFrame에서 결측값(NaN)이 있는 행을 모두 삭제; 받아오는 데이터가 영업일 기준이므로 할 필요가 없다
data = add_technical_features(data)
# 하나라도 결측이 있으면 행을 삭제
data = data.dropna(subset=['종가', '거래량'])




# 일봉 그래프
# plot_candles_standard(data, show_months=5, title='Bollinger Bands & Volume — Standard Candles')

# 주봉 그래프
# plot_candles_weekly_standard(data, show_months=12)


# sharex: 여러 서브플롯들이 x축(스케일/눈금/포맷)을 같이 쓸지 말지를 정하는 옵션
fig, (ax1, ax2, ax3, ax4) = plt.subplots(
    4, 1, figsize=(16, 20), sharex=False,
    gridspec_kw={'height_ratios':[3,1,3,1]},
    dpi=200
)

# 일봉 두 패널
plot_candles_daily(data, show_months=5, title="Daily — BB & Volume",
                   ax_price=ax1, ax_volume=ax2)

# 주봉 두 패널
plot_candles_weekly(data, show_months=12, title=" ",
                    ax_price=ax3, ax_volume=ax4)

plt.tight_layout()
# plt.show()

# 파일 저장 (옵션)
output_dir = 'D:\\stocks'
os.makedirs(output_dir, exist_ok=True)

final_file_name = f'{today} [{ticker}].png'
final_file_path = os.path.join(output_dir, final_file_name)
plt.savefig(final_file_path)
plt.close()