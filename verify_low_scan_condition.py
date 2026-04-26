"""
생성한 저점 매수 조건을 검증하는 스크립트
lowscan_rules.py > 새 CSV에서도 성능이 나오는지 검증
"""

from utils import sort_csv_by_today_desc
import pandas as pd
from pathlib import Path

out_path = Path("lowscan_rules.py")

# df = pd.read_csv("csv/low_result_us_desc.csv")
df = pd.read_csv("csv/low_result_7_desc.csv")

# 각 조건 정의
from lowscan_rules import build_conditions
conditions = build_conditions(df)

rows = []

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

    # up_cnt = (sub["validation_chg_rate"] >= 7).sum()   # 정적 기준
    up_cnt = (sub["is_success"] == 1).sum()            # 동적 기준 반영
    ratio = up_cnt / len(sub)

    rows.append({
        "rule": name,
        "count": len(sub),
        "success_cnt": up_cnt,
        "success_rate": ratio,
    })

    print(f"\n=== {name} ===")
    print(f"선택된 행 수: {len(sub)}")
    print(f"성공 개수   : {up_cnt}")
    print(f"성공률      : {ratio:.3f}")

    # validation_chg_rate >= 7 인 행들만 보기
    # sub_up = sub[sub["validation_chg_rate"] >= 7]
    # print("\n  ▶ validation_chg_rate >= 7 종목 목록")
    # print(sub_up[["ticker", "stock_name", "validation_chg_rate"]])   # 국장
    # print(sub_up[["ticker", "validation_chg_rate"]])   # 미장


# 모든 조건 테스트
for name, cond in conditions.items():
    test_condition(name, cond)

base_result = pd.DataFrame(rows)

print("\n===== BASE RULES =====")
print("룰 개수:", len(base_result))
print("총 선택 수:", base_result["count"].sum())
print("평균 성공률:", round(base_result["success_rate"].mean(), 2))
print("가중 성공률:", round(base_result["success_cnt"].sum() / base_result["count"].sum(), 2))


df = df.drop(columns=["pct_vs_lastweek"], errors="ignore")
df = df.drop(columns=["pct_vs_last4week"], errors="ignore")
df = df.drop(columns=["vol15"], errors="ignore")
df = df.drop(columns=["three_m_min_cur"], errors="ignore")
df = df.drop(columns=["mean_prev3"], errors="ignore")
df = df.drop(columns=["pos20_ratio"], errors="ignore")
df = df.drop(columns=["dist_to_ma20"], errors="ignore")
df.to_csv("csv/low_result_7_desc2.csv", index=False)

print("\n이제 룰 마이닝 스크립트에서 CSV만 아래로 바꿔서 다시 실행하세요:")
print('df = pd.read_csv("csv/low_result_7_desc2.csv")')
print('out_path = Path("low_result_7_desc2.py")')

