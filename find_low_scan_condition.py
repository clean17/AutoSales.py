"""
생성한 csv 파일을 가지고 저점 조건을 찾는 스크립트
"""

import pandas as pd
import numpy as np
from pathlib import Path
import heapq
from itertools import count

df = pd.read_csv("csv/low_result_7_desc.csv")
# df = pd.read_csv("csv/low_result_us_6_desc.csv")   # 미장

MIN_RATE     = 0.82
MIN_CNT      = 30
MAX_DEPTH    = 4
EXPAND_RATIO = [0.32, 0.38, 0.53, 0.7, 0.8]
# TARGET_COL = "validation_chg_rate"         # 검증등락률
# target = (df[TARGET_COL].to_numpy() >= 7)  # 7퍼 이상 검증 통과
TARGET_COL = "is_success"                  # 동적 검증등락률 통과
target = df[TARGET_COL].to_numpy() == 1

out_path = Path("lowscan_rules.py")
# out_path = Path("lowscan_rules_us.py")   # 미장

# 제외할 피쳐
exclude = {
    "ticker", "stock_name", "predict_str", "today",

    "validation_chg_rate",
    "validation_chg_rate1",
    "validation_chg_rate2",
    "validation_chg_rate3",
    "validation_chg_rate4",
    "validation_chg_rate5",
    "validation_chg_rate6",
    "validation_chg_rate7",

    "hit_day", "is_success", "target",

    "_gap_pct",
    "_vol_ratio_15_60",
    "_RSI_rebound",
    "_rebound_power",
    "_MACD_hist_1d",
    "MACD_hist_3d",
    "tr_volume_rank_20d"
    # "trend_signal_tanh",
    # "dist_from_low_tanh",
    # "tr_value_ratio_tanh",
}

features = [c for c in df.columns if c not in exclude]

# --- literal(원자 조건) 만들기: feature <= q or feature > q ---
literals = []
literal_masks = []

"""
0.05, 0.10, 0.15, 0.20, ... 0.95 → 총 19개 threshold
0.1, 0.2, 0.3, ... 0.9           → 총 9개 threshold

depth4에서 데드캣을 이미 거른 뒤 depth5를 만들 거면 np.linspace(0.2, 0.8, 7) 괜찮음
0.05~0.95, 19개	많음	세밀하지만 과적합 위험 큼
0.1~0.9, 9개	중간	균형
0.2~0.8, 7개	적음	안정적, 룰 적음, 과적합 감소
"""
for f in features:
    # print(f)
    col = df[f].astype(float).to_numpy()  # flaat 형변환 >  Series > Numpy 배열
    col_nonan = col[~np.isnan(col)]       # ~np.isnan() 으로 True 값만 남는다 > NaN 제거
    print(f, len(col_nonan), len(np.unique(col_nonan)))

    # 분위수(percentile) 배열 생성 (덜 촘촘하게 = 과적합 방지하면서 빠르게)
    unique_vals = np.unique(col_nonan)
    if len(unique_vals) < 50:
        qs = unique_vals  # quantile 안씀
    else:
        # n_bins = min(19, len(unique_vals) - 1)
        # qs = np.unique(np.quantile(col_nonan, np.linspace(0.05, 0.95, n_bins)))

        qs = np.unique(np.quantile(col_nonan, np.linspace(0.05, 0.95, 19)))
        # qs = np.unique(np.quantile(col_nonan, np.linspace(0.1, 0.9, 9)))
        # qs = np.unique(np.quantile(col_nonan, np.linspace(0.2, 0.8, 7)))

    for thr in qs:
        thr = round(float(thr), 4)
        literals.append((f, "<=", thr))
        literal_masks.append(col <= thr)

        literals.append((f, ">", thr))
        literal_masks.append(col > thr)

# print(literals)

feature_groups = {
    # 가격 / 당일 반등 강도
    "today_pct": "PRICE",

    # 추세 전환
    "trend_signal": "TREND",

    # MACD 모멘텀
    "MACD_acc": "MACD",
    "MACD_hist_3d_rank": "MACD",

    # 저점 위치
    "dist_from_low": "POSITION",

    # 수급
    "tr_value_ratio": "VOLUME",

    # 눌림 강도
    "max_drop_7d": "DROP",
}

group_limits = {
    "PRICE": 1,
    "TREND": 1,
    "MACD": 2,
    "POSITION": 1,
    "VOLUME": 1,
    "DROP": 1,
}


"""
beam : 단계별 상위 수량만 다음 뎁스로 가져간다, 한 depth에서 유지할 후보 개수
expand_ratio : 각 후보의 성능이 이 값보다 커야 다음 뎁스로 진행, 다음 depth로 확장할 최소 기준
min_ratio : “좋은 룰”로 저장할 기준
cnt_priority_ratio : 넘기면 ratio 보다 cnt 를 더 중요시
"""
def mine_rules(
        min_ratio=0.7,
        min_count=20,
        max_depth=4,
        beam=400,
        expand_ratio=0.50,
        cnt_priority_ratio=0.85,   # <-- 여기부터 cnt를 더 중요하게 볼 임계치 (고정)
        top_n=500,                 # 원하면 최종 결과 상위 N개만 리턴
        #  그룹 제약 추가
        feature_groups=None,       # dict: feature_name -> group_name
        group_limits=None,         # dict: group_name -> max_allowed_in_rule
):
    """
    (count, ratio, up_cnt, conds) 리스트 반환

    beam(확장 후보) 선별 정책(유지):
    - ratio 우선
    - ratio 동률이면 cnt 큰 후보 우대

    최종 out 정렬 정책(변경):
    - ratio >= cnt_priority_ratio 이면 cnt 우선(그 다음 ratio)
    - ratio <  cnt_priority_ratio 이면 ratio 우선(그 다음 cnt)

    그룹 제약:
    - feature_groups에 매핑된 피쳐들은 같은 그룹에서 group_limits 개수 이상 못 씀
    """

    beams = [(np.ones(len(df), dtype=bool), [])]  # df 길이 만큼의 True 배열(불리언 마스크), 빈 배열의 튜플 리스트 > "데이터프레임의 모든 행을 선택한다"는 상태
    good = {}

    print("\nbeam", beam, "min_ratio", min_ratio, "min_count", min_count, "max_depth", max_depth, "expand_ratio", EXPAND_RATIO, "top_n", top_n)
    print("cnt_priority_ratio", cnt_priority_ratio, '\n')

    if feature_groups is None:
        feature_groups = {}
    if group_limits is None:
        group_limits = {}

    def get_group(feat_name: str):
        return feature_groups.get(feat_name)

    # beam(확장 후보)용 비교키
    # "좋은 후보"가 더 큰 key를 갖도록 설계 (나중에 key 비교로 worst 교체)
    def beam_key(ratio, cnt):
        if ratio >= cnt_priority_ratio:
            # 0그룹: cnt 우선, ratio 보조
            return (1, cnt, ratio)
        else:
            # 1그룹: ratio 우선, cnt 보조
            return (0, ratio, cnt)

    for depth in range(max_depth):
        print('----------------------------------')
        print("depth", depth)

        # heap item: (key, uid, mask, conds, ratio, cnt)
        # heap[0] = 가장 "나쁜" 후보 (key가 가장 작음)
        heap = []
        uid = count()  # 카운트 제네레이터 (itertools 모듈)

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}  # c[0]: 피쳐 명, set comprehension

            # 현재 rule에서 그룹 사용량 계산
            group_used = {}
            for f in used_feats:
                g = get_group(f)
                if g is None:
                    continue
                group_used[g] = group_used.get(g, 0) + 1

            for (lit, lmask) in zip(literals, literal_masks):  # zip() 이용한 동시 순회
                feat = lit[0]

                # 동일 feature 중복 금지 (기존)
                if feat in used_feats:
                    continue

                # 그룹 제약 체크
                g = get_group(feat)
                if g is not None:
                    limit = group_limits.get(g, None)
                    if limit is not None:
                        if group_used.get(g, 0) >= limit:
                            continue

                # 새로운 subset
                m = base_mask & lmask
                cnt = int(m.sum())
                if cnt < MIN_CNT:
                    continue

                up = int((m & target).sum())
                ratio = up / cnt
                # score = ratio * np.log1p(cnt)
                # score = (ratio ** 2) * np.log1p(cnt)  # ratio에 더 강하게 보상
                score = (ratio ** 3) * np.log1p(cnt)  # ratio에 더 강하게 보상

                # good 저장
                if ratio >= min_ratio:
                    key2 = tuple(sorted((c[0], c[1], round(float(c[2]), 6)) for c in (conds + [lit])))
                    prev = good.get(key2)
                    if (prev is None) or (score > prev[4]):
                        good[key2] = (cnt, ratio, up, conds + [lit], score)

                """
                new ≈ beam의 30~70%가 되도록 expand_ratio 조절 필요

                depth 0 >> new가 beam의 1~20%여도 괜찮음
                depth 1 >> beam의 20~70%
                depth 2+ > beam의 30~70%
                """
                # 확장 후보
                if ratio >= EXPAND_RATIO[depth]:
                    k = beam_key(ratio, cnt)
                    item = (k, next(uid), m, conds + [lit], ratio, cnt)

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        # early-skip: worst(=heap[0])보다 좋아야 교체
                        if k <= heap[0][0]:
                            continue
                        heapq.heapreplace(heap, item)

        # 다음 depth로 넘길 후보들(좋은 순): key 내림차순
        # key는 (group, primary, secondary)인데 primary/secondary는 큰 게 좋게 만들어 둠
        new = sorted(heap, key=lambda x: x[0], reverse=True)  # 좋은 순으로 정렬
        print("new", len(new))

        if not new:
            print("no expandable candidates; stopping.")
            break

        tail = new[-1]
        print("tail ratio:", round(tail[4],3), "cnt:", tail[5], "conds:", tail[3])  # 최악 후보 출력 (디버깅용)

        beams = [(m, conds) for _, _, m, conds, _, _ in new]

    # ---- 최종 out 정렬 ----
    # def out_key(x):
    #     cnt, ratio, up, conds = x
    #     if ratio >= cnt_priority_ratio:
    #         return (0, -cnt, -ratio, len(conds))
    #     else:
    #         return (1, -ratio, -cnt, len(conds))

    def out_key(x):
        cnt, ratio, up, conds, score = x
        return (-score, -ratio, -cnt, len(conds))

    out = sorted(good.values(), key=out_key)
    if top_n is not None:
        out = out[:top_n]

    # for cnt, ratio, up, conds in out[:20]:
    #     print(cnt, ratio, len(conds))

    for cnt, ratio, up, conds, score in out[:20]:
        print(cnt, round(ratio, 2), round(score, 2), len(conds))

    return out



def test_condition(name, cond, df, verbose=False):
    """
    param cond: 조건식 결과, 각 행마다 True / False가 들어 있는 bool mask
    return: 저장할 가치가 있으면 True, 아니면 False
    """
    # 조건을 만족한 종목/날짜만
    sub = df[cond]

    if len(sub) == 0:
        if verbose:
            print(f"\n=== {name} ===")
            print("선택된 행이 없습니다.")
        return False

    # up_cnt = (sub["validation_chg_rate"] >= 7).sum()   # 검증등락률
    up_cnt = (sub["is_success"] == 1).sum()            # 동적 검증등락률
    ratio = up_cnt / len(sub)

    """
    confidence: 성공률에 “표본 수 신뢰도 할인”을 적용한 값
    ratio = 0.70 일 때
    # 표본 20개
    confidence = 0.70 - (1 / sqrt(20))
               = 0.70 - 0.224
               = 0.476
    
    # 표본 200개
    confidence = 0.70 - (1 / sqrt(200))
               = 0.70 - 0.071
               = 0.629
    """
    confidence = ratio - (1 / np.sqrt(len(sub)))

    # if confidence < (MIN_RATE - 0.2):
    if confidence < 0.55:
        return False

    # if ratio < (MIN_RATE - 0.2):
    #     return False

    # 표본이 너무 적으면 과적합 가능성이 큼
    if len(sub) < 20:
        return False

    if verbose:
        print(f"\n=== {name} ===")
        print(f"선택된 행 수: {len(sub)}")
        print(f"성공 개수   : {up_cnt}")
        print(f"성공률      : {ratio:.3f}")

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

# 뎁스가 진행되어도 cnt가 급격히 줄면 안된다
# 뎁스 증가에 따른 ratio 상승이 완만해야 한다

# 국장
rules = mine_rules(
    min_ratio=MIN_RATE,
    min_count=MIN_CNT,
    max_depth=MAX_DEPTH,
    beam=30000,
    top_n=1000,
    feature_groups=feature_groups,
    group_limits=group_limits,
)
# 80 25 4 42 > 18/54 75
# 77 25 5 42 > 12/37 75



# 미장
# rules = mine_rules(min_ratio=MIN_RATE, min_count=50, max_depth=6, beam=30000, expand_ratio=0.45, top_n=10000)

top_n = min(1000, len(rules))
selected = []  # (name, conds)만 저장해두고 파일로 씀

# for i, (cnt, ratio, up, conds) in enumerate(rules[:top_n], start=1):
    # name = f"rule_{i:03d}__n{cnt}__r{ratio:.3f}"
for i, (cnt, ratio, up, conds, score) in enumerate(rules[:top_n], start=1):
    name = f"rule_{i:03d}__n{cnt}__r{ratio:.3f}__s{score:.2f}"

    # df로 mask 생성
    mask = np.ones(len(df), dtype=bool)
    for f, op, thr in conds:
        if op == "<=":
            mask &= (df[f] <= thr)
        else:
            # rule_to_code가 >=를 찍고 있으니 검증도 >=로 맞추기
            mask &= (df[f] >= thr)

    # 통과한 룰만 담기
    if test_condition(name, mask, df, verbose=False):
        selected.append((name, conds))

print(f"\n통과 룰 개수: {len(selected)} / {min(len(rules), top_n)}")


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

# import re
# txt = out_path.read_text(encoding="utf-8")
#
# m = re.search(r'RULE_NAMES\s*=\s*\[(.*?)\]\s*\n', txt, flags=re.S)
# block = m.group(1) if m else ""
# print("RULE_NAMES entries (exact):", len(re.findall(r'"rule_\d+__n', block)))
#
# m = re.search(r'conditions\s*=\s*\{(.*?)\n\s*\}\s*\n\s*return conditions', txt, flags=re.S)
# block = m.group(1) if m else ""
# print("conditions entries (exact):", len(re.findall(r'^\s*"rule_\d+__n.*":', block, flags=re.M)))
