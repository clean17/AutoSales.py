import pandas as pd
import numpy as np
from pathlib import Path
import heapq
from itertools import count

# df = pd.read_csv("csv/low_result_250513_desc.csv")
# df = pd.read_csv("csv/low_result_250507_desc.csv")
df = pd.read_csv("csv/low_result_6_desc.csv")
# df = pd.read_csv("csv/low_result_us_desc.csv")

TARGET_COL = "validation_chg_rate"
MIN_RATE = 0.85   # 미장 0.9
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


"""
expand_ratio = 0.45 → beam 5k~10k도 그럭저럭
expand_ratio = 0.40 → beam 10k~30k 권장
expand_ratio = 0.35 → beam 30k~ 아니면 빔 컷이 너무 심해질 가능성 큼

beam : 단계별 상위 수량만 다음 뎁스로 가져간다, 안정성 목표(결과 흔들림 줄이기) 최소 5000 가능하면 10000~30000 쪽이 훨씬 안정적
expand_ratio : 각 후보의 성능이 이 값보다 커야 다음 뎁스로 진행
"""
import numpy as np
import heapq
from itertools import count

def mine_rules(
        min_ratio=0.7,
        min_count=20,
        max_depth=4,
        beam=400,
        expand_ratio=0.45,
        cnt_priority_ratio=0.85,   # <-- 여기부터 cnt를 더 중요하게 볼 임계치
        top_n=None,                # 원하면 최종 결과 상위 N개만 리턴
):
    """
    (count, ratio, up_cnt, conds) 리스트 반환

    beam(확장 후보) 선별 정책(유지):
    - ratio 우선
    - ratio 동률이면 cnt 큰 후보 우대

    최종 out 정렬 정책(변경):
    - ratio >= cnt_priority_ratio 이면 cnt 우선(그 다음 ratio)
    - ratio <  cnt_priority_ratio 이면 ratio 우선(그 다음 cnt)
    """

    beams = [(np.ones(N, dtype=bool), [])]
    good = {}

    print('beam', beam)
    print("expand_ratio", expand_ratio, "min_ratio", min_ratio, "min_count", min_count)
    print("cnt_priority_ratio", cnt_priority_ratio)

    for depth in range(max_depth):
        print('----------------------------------')
        print("depth", depth)

        heap = []
        uid = count()

        for base_mask, conds in beams:
            used = {c[0] for c in conds}

            for (lit, lmask) in zip(literals, literal_masks):
                if lit[0] in used:
                    continue

                m = base_mask & lmask
                cnt = int(m.sum())
                if cnt < min_count:
                    continue

                up = int((m & target).sum())
                ratio = up / cnt

                # good 저장
                if ratio >= min_ratio:
                    key = tuple(sorted((c[0], c[1], round(float(c[2]), 6)) for c in (conds + [lit])))
                    prev = good.get(key)
                    if (prev is None) or (cnt > prev[0]) or (cnt == prev[0] and ratio > prev[1]):
                        good[key] = (cnt, ratio, up, conds + [lit])

                # 다음 depth 확장 후보
                if ratio >= expand_ratio:
                    if len(heap) == beam:
                        worst_ratio, worst_cnt = heap[0][0], heap[0][1]
                        if (ratio < worst_ratio) or (ratio == worst_ratio and cnt <= worst_cnt):
                            continue

                    item = (ratio, cnt, next(uid), m, conds + [lit])

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heapreplace(heap, item)

        new = sorted(heap, key=lambda x: (-x[0], -x[1]))
        print("new", len(new))

        if not new:
            print("no expandable candidates; stopping.")
            break

        tail = new[-1]
        print("tail ratio,cnt:", tail[0], tail[1], "conds:", tail[4])

        beams = [(m, conds) for _, _, _, m, conds in new]

    # ---- 최종 out 정렬(핵심 변경) ----
    def out_key(x):
        cnt, ratio, up, conds = x
        if ratio >= cnt_priority_ratio:
            # ratio가 충분히 높으면 cnt가 더 중요
            return (0, -cnt, -ratio, len(conds))
        else:
            # ratio가 아직 낮으면 ratio를 더 중요
            return (1, -ratio, -cnt, len(conds))

    out = sorted(good.values(), key=out_key)

    if top_n is not None:
        out = out[:top_n]

    return out


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


# ✅ 여기서 "최대한 많이" 얻고 싶으면 top_n 크게
rules = mine_rules(min_ratio=MIN_RATE, min_count=30, max_depth=7, beam=5000, expand_ratio=0.45)



# 미장
# rules = mine_rules(min_ratio=MIN_RATE, min_count=50, max_depth=7, beam=500, expand_ratio=0.45)   # 0.85 > 82% // 0.9 > 87%

top_n = min(5000, len(rules))   # 필요하면 1000도 가능(조건 엄청 많아짐)
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

print("top_n:", top_n)
print("selected:", len(selected))
print("first 3 selected:", [n for n, _ in selected[:3]])

import re

m = re.search(r'RULE_NAMES\s*=\s*\[(.*?)\]\s*\n', txt, flags=re.S)
block = m.group(1) if m else ""
print("RULE_NAMES entries (exact):", len(re.findall(r'"rule_\d+__n', block)))

m = re.search(r'conditions\s*=\s*\{(.*?)\n\s*\}\s*\n\s*return conditions', txt, flags=re.S)
block = m.group(1) if m else ""
print("conditions entries (exact):", len(re.findall(r'^\s*"rule_\d+__n.*":', block, flags=re.M)))
