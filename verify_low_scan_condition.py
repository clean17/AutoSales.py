from utils import sort_csv_by_today_desc
import pandas as pd
from pathlib import Path

out_path = Path("lowscan_rules.py")

# df = pd.read_csv("csv/low_result_us_desc.csv")
df = pd.read_csv("csv/low_result_250513_desc.csv")

# 각 조건 정의
from lowscan_rules import build_conditions
conditions = build_conditions(df)


# 조건을 만족하는 행등만 골라서(sub) 확인
def test_condition(name, cond):
    """
    param cond: 조건식 결과
    """

    # 조건을 만족한 행들만 모인 DataFrame
    sub = df[cond]

    if len(sub) == 0:
        print(f"\n=== {name} ===")
        print("선택된 행이 없습니다.")
        return

    up_cnt = (sub["validation_chg_rate"] >= 7).sum()
    ratio = up_cnt / len(sub)

    if (ratio > 0.88):
        return

    if len(sub) > 19:
        return

    print(f"\n=== {name} ===")
    print(f"선택된 행 수: {len(sub)}")
    print(f"  validation_chg_rate >= 7 개수 : {up_cnt}")
    print(f"  Ratio (>=7)      : {ratio:.3f}")

    # validation_chg_rate >= 7 인 행들만 보기
    # sub_up = sub[sub["validation_chg_rate"] >= 7]
    # print("\n  ▶ validation_chg_rate >= 7 종목 목록")
    # print(sub_up[["ticker", "stock_name", "validation_chg_rate"]])   # 국장
    # print(sub_up[["ticker", "validation_chg_rate"]])   # 미장


# 모든 조건 테스트
for name, cond in conditions.items():
    test_condition(name, cond)


# saved = sort_csv_by_today_desc(
#     in_path=r"csv/low_result_us.csv",
#     out_path=r"csv/low_result_us_desc.csv",
# )
# print("saved:", saved)


