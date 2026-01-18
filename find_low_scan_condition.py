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


def mine_rules(
        min_ratio=0.7,
        min_count=20,
        max_depth=4,
        beam=400,            # 탐색을 어디까지/어떤 후보를 계속 확장할지, 단계별 상위 수량만 다음 뎁스로 가져간다, // 안정성 목표(결과 흔들림 줄이기) 최소 5000 가능하면 10000~30000 쪽이 훨씬 안정적
        expand_ratio=0.45,   # 중간 단계 확장용(너무 높이면 길이 막힘), 각 후보의 성능이 이 값보다 커야 다음 뎁스로 진행, “이 정도 성능(ratio)이 안 나오면 더 깊게 안 파겠다”는 확장 게이트
):
    """
    (count, ratio, up_cnt, conds) 리스트 반환
    - out 정렬: ratio 우선, 그 다음 count, 그 다음 길이
    - beam 확장 후보는 heap(Top-K)으로 유지하여 메모리 폭발 방지
    - B) early-skip: heap이 가득 찼을 때 최악보다 나쁜 후보는 heap 연산 자체를 생략

    정책(beam 선별):
    - r2 = round(ratio, 2) 를 1순위로 (예: 0.842 -> 0.84, 0.858 -> 0.86)
    - r2 동률이면 cnt 큰 후보 우대
    - (추가 동률은 uid로 안정적으로 처리)

    주의:
    - good 저장(최종 룰 채택)은 '실제 ratio' 기준(min_ratio) 그대로 유지
    - beam 확장 경로는 r2 기준으로 바뀌므로 결과(탐색 경로)는 달라질 수 있음

    expand_ratio = 0.45 → beam 5k~10k도 그럭저럭
    expand_ratio = 0.40 → beam 10k~30k 권장
    expand_ratio = 0.35 → beam 30k~ 아니면 빔 컷이 너무 심해질 가능성 큼
    """

    beams = [(np.ones(N, dtype=bool), [])]
    good = {}

    print('beam', beam)
    print("expand_ratio", expand_ratio, "min_ratio", min_ratio, "min_count", min_count)

    for depth in range(max_depth):
        print('----------------------------------')
        print("depth", depth)

        # heap item: (r2, cnt, uid, ratio, mask, conds)
        # min-heap이라 "가장 나쁜 후보"가 맨 앞:
        #  - r2 낮을수록 나쁨
        #  - r2 같으면 cnt 작을수록 나쁨
        heap = []
        uid = count()

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

                # good 저장 (최종 룰), 순위는 실제 ratio로
                if ratio >= min_ratio:
                    key = tuple(sorted((c[0], c[1], round(float(c[2]), 6)) for c in (conds + [lit])))
                    prev = good.get(key)
                    if (prev is None) or (cnt > prev[0]) or (cnt == prev[0] and ratio > prev[1]):
                        good[key] = (cnt, ratio, up, conds + [lit])

                # 다음 depth로 확장 후보
                if ratio >= expand_ratio:
                    r2 = round(float(ratio), 2)

                    # early-skip: heap이 꽉 찼으면 최악보다 나쁘면 패스
                    if len(heap) == beam:
                        worst_r2, worst_cnt = heap[0][0], heap[0][1]
                        if (r2 < worst_r2) or (r2 == worst_r2 and cnt <= worst_cnt):
                            continue

                    item = (r2, cnt, next(uid), ratio, m, conds + [lit])

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        heapq.heapreplace(heap, item)


        # 다음 depth로 넘길 후보들: r2 내림차순, cnt 내림차순 (동률이면 실제 ratio도 내림차순으로 살짝 정리)
        new = sorted(heap, key=lambda x: (-x[0], -x[1], -x[3]))
        print("new", len(new))

        if not new:
            print("no expandable candidates; stopping.")
            beams = []
            break

        tail = new[-1]
        print("tail r2,cnt,ratio:", tail[0], tail[1], tail[3], "conds:", tail[5])

        # 다음 depth beams 갱신
        beams = [(m, conds) for _, _, _, _, m, conds in new]

    # 최종 out 정렬도 원하면 r2 기반으로 바꿀 수 있지만,
    # 보통은 실제 ratio를 쓰는 게 더 합리적이라 그대로 둠:
    out = sorted(good.values(), key=lambda x: (-x[1], -x[0], len(x[3])))
    # 최종 out도 “r2 동률이면 cnt 우대”로 정렬하고 싶다면
    # out = sorted(good.values(), key=lambda x: (-round(float(x[1]), 1), -x[0], -x[1], len(x[3])))
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
