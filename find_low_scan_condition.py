import pandas as pd
import numpy as np
from pathlib import Path

# df = pd.read_csv("csv/low_result_250513_desc.csv")
df = pd.read_csv("csv/low_result_250507_desc.csv")
# df = pd.read_csv("csv/low_result_us_desc.csv")

TARGET_COL = "validation_chg_rate"
MIN_RATE = 0.82   # 미장 0.9
target = (df[TARGET_COL].to_numpy() >= 7)
out_path = Path("lowscan_rules.py")
# out_path = Path("lowscan_rules_us.py")

# 숫자 피처들(원하면 더/덜 제외 가능)
exclude = {"ticker", "stock_name", "predict_str", "today", TARGET_COL,
           "validation_chg_rate1",
           "validation_chg_rate2",
           "validation_chg_rate3",
           "validation_chg_rate4",
           "validation_chg_rate5",
           "validation_chg_rate6",
           "validation_chg_rate7",
           }
features = [c for c in df.columns if c not in exclude]

N = len(df)

# --- literal(원자 조건) 만들기: feature <= q or feature > q ---
literals = []
literal_masks = []

for f in features:
    col = df[f].astype(float).to_numpy()
    col_nonan = col[~np.isnan(col)]
    # 분위수 촘촘하게(원하면 0.05~0.95를 더 촘촘히)
    qs = np.unique(np.quantile(col_nonan, np.linspace(0.05, 0.95, 19)))

    for thr in qs:
        thr = float(thr)
        literals.append((f, "<=", thr))
        literal_masks.append(col <= thr)

        literals.append((f, ">", thr))
        literal_masks.append(col > thr)

def test_condition(name, cond, df, verbose=False):
    """
    param cond: 조건식 결과 (bool mask)
    return: 저장할 가치가 있으면 True, 아니면 False
    """
    sub = df[cond]

    if len(sub) == 0:
        if verbose:
            print(f"\n=== {name} ===")
            print("선택된 행이 없습니다.")
        return False

    up_cnt = (sub["validation_chg_rate"] >= 7).sum()
    ratio = up_cnt / len(sub)

    if ratio < MIN_RATE:
        return False

    if len(sub) < 19:
        return False

    if verbose:
        print(f"\n=== {name} ===")
        print(f"선택된 행 수: {len(sub)}")
        print(f"  validation_chg_rate >= 7 개수 : {up_cnt}")
        print(f"  Ratio (>=7)      : {ratio:.3f}")

    return True


def rule_to_code(name, conds, thr_round=3):
    lines = [f'    "{name}":']
    parts = []
    for f, op, thr in conds:
        thr = float(np.round(thr, thr_round))
        if op == "<=":
            parts.append(f'(df["{f}"] <= {thr})')
        else:
            parts.append(f'(df["{f}"] >= {thr})')  # ">": ge로 출력
    # 보기 좋게 줄바꿈
    joined = " &\n        ".join(parts)
    lines.append(f"        {joined},")
    return "\n".join(lines)

def mine_rules(
        min_ratio=0.7,
        min_count=20,
        max_depth=4,
        beam=400,            # 탐색을 어디까지/어떤 후보를 계속 확장할지, 단계별 상위 수량만 다음 뎁스로 가져간다
        expand_ratio=0.45,   # 중간 단계 확장용(너무 높이면 길이 막힘), 각 후보의 성능이 이 값보다 커야 다음 뎁스로 진행
):
    """(count, ratio, up_cnt, conds) 리스트 반환"""
    beams = [(np.ones(N, dtype=bool), [])]
    good = {}

    for _ in range(max_depth):
        new = []
        for base_mask, conds in beams:
            used = {c[0] for c in conds}  # feature 중복 방지
            for (lit, lmask) in zip(literals, literal_masks):
                if lit[0] in used:
                    continue

                m = base_mask & lmask
                cnt = int(m.sum())
                if cnt < min_count:
                    continue

                up = int((m & target).sum())
                ratio = up / cnt

                if ratio >= min_ratio:
                    key = tuple(sorted((c[0], c[1], round(float(c[2]), 6)) for c in (conds + [lit])))
                    # 같은 key면 더 큰 count/ratio 우선
                    prev = good.get(key)
                    if (prev is None) or (cnt > prev[0]) or (cnt == prev[0] and ratio > prev[1]):
                        good[key] = (cnt, ratio, up, conds + [lit])

                if ratio >= expand_ratio:
                    new.append((ratio, cnt, m, conds + [lit]))

        new.sort(key=lambda x: (-x[0], -x[1]))  # ratio 우선, 그 다음 count
        new = new[:beam]
        beams = [(m, conds) for _, _, m, conds in new]

    # count 큰 순 정렬
    out = sorted(good.values(), key=lambda x: (-x[0], -x[1], len(x[3])))
    return out

def make_name(conds):
    parts = []
    for f, op, thr in conds:
        opn = "le" if op == "<=" else "gt"
        thr_s = str(np.round(thr, 3)).replace("-", "m").replace(".", "_")
        parts.append(f"{f}_{opn}_{thr_s}")
    return "_and_".join(parts)

# ✅ 여기서 "최대한 많이" 얻고 싶으면 top_n 크게
rules = mine_rules(min_ratio=MIN_RATE, min_count=42, max_depth=7, beam=500, expand_ratio=0.45)
# 0.8, 38, 7, 500, 0.45 > 1000/1000 > 80.77%
# 0.8, 40, 7, 500, 0.45 > 992/1000 > 80.77%
# 0.8, 42, 7, 500, 0.45 > 168/182 > 82.61%
# 0.8, 44, 7, 500, 0.45 > 85/89 > 83.33%

# 0.82, 38, 7, 500, 0.45 > 1000/1000 > 89.47%
# 0.82, 39, 7, 500, 0.45 > 1000/1000 > 86%
# 0.82, 40, 7, 500, 0.45 > 8/8 > 86%

## 0.85, 30, 7, 500, 0.45 > 966/1000 > 83.8% 5/26
## 0.83, 40, 7, 500, 0.45 > 96/108   > 87.8% 4/29

## 0.82, 46, 7, 500, 0.45 > 96/96    > 90.32% 3/28
## 0.82, 44, 7, 500, 0.45 > 96/96    > 제로
## 0.82, 42, 7, 500, 0.45 > 159/163  > 88.4% 3/23
## 0.82, 40, 7, 500, 0.45 > 996/1000 > 88.9% 3/24
## 0.81, 46, 7, 500, 0.45 > 112/112  > 90.9% 2/20
## 0.81, 40, 7, 500, 0.45 > 942/1000 > 82.7% 5/24
## 0.81, 38, 7, 500, 0.45 > 998/1000 > 84.6% 6/33
## 0.81, 36, 7, 500, 0.45 > 997/1000 > 80%  5/20
## 0.81, 30, 7, 500, 0.45 > 962/1000 > 69.7% 13/30
## 0.8,  50, 7, 500, 0.45 > 890/897  > 87.18% 5/34
## 0.8,  40, 7, 500, 0.45 > 934/1000 > 82% 6/29
## 0.8,  30, 7, 500, 0.45 > 911/1000 > 70% 17/40



# 미장
# rules = mine_rules(min_ratio=MIN_RATE, min_count=50, max_depth=7, beam=500, expand_ratio=0.45)   # 0.85 > 82% // 0.9 > 87%

top_n = min(1000, len(rules))   # 필요하면 1000도 가능(조건 엄청 많아짐)
conditions = {}

selected = []  # (name, conds)만 저장해두고 파일로 씀

for i, (cnt, ratio, up, conds) in enumerate(rules[:top_n], start=1):
    name = f"rule_{i:03d}__n{cnt}__r{ratio:.3f}"

    # df로 mask 생성
    mask = np.ones(len(df), dtype=bool)
    for f, op, thr in conds:
        if op == "<=":
            mask &= (df[f] <= thr)
        else:
            # rule_to_code가 >=를 찍고 있으니 검증도 >=로 맞추기
            mask &= (df[f] >= thr)

    # ✅ 통과한 룰만 담기
    if test_condition(name, mask, df, verbose=False):
        selected.append((name, conds))

print(f"통과 룰 개수: {len(selected)} / {min(len(rules), top_n)}")


# dict 자료구조 >> 값: 불리언(boolean) 마스크
# df가 pandas DataFrame라면 >> 값은 pandas.Series (dtype=bool)
# pandas.Series[bool] (여러 값): True/False가 행 개수만큼 들어있는 1차원 배열 같은 것, 인덱스를 가진 1차원 배열
"""
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
index   value
a       10
b       20
c       30

# list
[1, 2, 3] + 1    # ❌ 에러

# Series
s + 1           # ⭕ 모든 원소에 +1

----------------------

mask = conditions["rule_001__n77__r0.805"]
mask.any()   # 하나라도 True면 True (단일 bool)
mask.all()   # 전부 True면 True (단일 bool)
mask.sum()   # True 개수 (int)
"""

# ✅ 통과 룰만 파일로 저장 (.py 모듈)
with out_path.open("w", encoding="utf-8") as f:
    f.write("# auto-generated: lowscan rules (filtered)\n")
    f.write("# usage:\n")
    f.write("#   from lowscan_rules import build_conditions, RULE_NAMES\n")
    f.write("#   find_conditions = build_conditions(df)\n\n")
    f.write("import numpy as np\n\n")

    # 이름 리스트도 같이 저장하면 편함
    f.write("RULE_NAMES = [\n")
    for name, _ in selected:
        f.write(f'    "{name}",\n')
    f.write("]\n\n")

    f.write("def build_conditions(df):\n")
    f.write("    conditions = {\n")

    for name, conds in selected:
        code = rule_to_code(name, conds)  # 기존 너 함수 그대로 활용
        lines = code.splitlines()
        f.write("        " + lines[0] + "\n")
        for line in lines[1:]:
            f.write("        " + line + "\n")

    f.write("    }\n")
    f.write("    return conditions\n")

print(f"saved to: {out_path.resolve()}")

# print("conditions = {")
# for i, (cnt, ratio, up, conds) in enumerate(rules[:top_n], start=1):
#     # name = f"rule_{i:03d}__n{cnt}__r{ratio:.3f}__" + make_name(conds)
#     name = f"rule_{i:03d}__n{cnt}__r{ratio:.3f}"
#     print(rule_to_code(name, conds))
#
#     mask = np.ones(len(df), dtype=bool)
#     for f, op, thr in conds:
#         if op == "<=":
#             mask &= (df[f] <= thr)
#         else:
#             mask &= (df[f] > thr)
#
#     conditions[name] = mask
# print("}")

# print(f"총 생성된 조건 수(top_n 기준): {len(conditions)}")
# # print("가장 큰 count 조건(상위 5개):")
# # for k in list(conditions.keys())[:5]:
# #     print(" -", k)
#
# # --- 네 test_condition 돌리기 ---
# def test_condition(name, cond):
#     sub = df[cond]
#     if len(sub) == 0:
#         return
#     up_cnt = (sub[TARGET_COL] >= 7).sum()
#     ratio = up_cnt / len(sub)
#
#     # ✅ 통과: ratio >= 0.8
#     if ratio < 0.9:
#         return
#
#     # print(f"\n=== {name} ===")
#     # print(f"선택된 행 수: {len(sub)}")
#     # print(f"  validation_chg_rate >= 7 개수 : {up_cnt}")
#     # print(f"  Ratio (>=7)      : {ratio:.3f}")
#
# for name, cond in conditions.items():
#     test_condition(name, cond)
