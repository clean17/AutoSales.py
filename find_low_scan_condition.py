import pandas as pd
import numpy as np
from pathlib import Path
import heapq
from itertools import count

df = pd.read_csv("csv/low_result_7_desc.csv")
# df = pd.read_csv("csv/low_result_us_6_desc.csv")   # 미장

TARGET_COL = "validation_chg_rate"
MIN_RATE = 0.8
target = (df[TARGET_COL].to_numpy() >= 7)
out_path = Path("lowscan_rules.py")
# out_path = Path("lowscan_rules_us.py")   # 미장

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


feature_groups = {
    # 전환 캔들 축 (둘 중 1개만)
    "lower_wick_ratio": "TURN_CANDLE",
    "close_pos": "TURN_CANDLE",

    # 볼린저/위치 축 (둘 중 1개만)
    "bb_recover": "BOLL",
    "z20": "BOLL",

    # 모멘텀 축 (최대 2개)
    "today_pct": "MOMENTUM",
    "ma5_chg_rate": "MOMENTUM",
    "macd_hist_chg": "MOMENTUM",

    # 거래대금 축 (둘 다 같이 허용)
    "today_tr_val": "VOLUME",
    "chg_tr_val": "VOLUME",

    # 주간 퍼센트 축 (최대 1개)
    "pct_vs_lastweek": "WEEK",
    "pct_vs_last4week": "WEEK",

    # 3개월 레짐 축 (둘 다 같이 허용해도 됨)
    "three_m_chg_rate": "REGIME",
    "today_chg_rate": "REGIME",

    # 환경 축 (둘 다 같이 허용해도 됨)
    "vol20": "ENV",
    "pos20_ratio": "ENV",
}

group_limits = {
    "TURN_CANDLE": 1,  # lower_wick_ratio + close_pos 동시 금지
    "BOLL": 1,         # bb_recover + z20 동시 금지
    "MOMENTUM": 2,     # today_pct/ma5/macd 중 2개까지만
    "WEEK": 1,         # lastweek/last4week 둘 중 1개만
    "VOLUME": 2,       # 둘 다 허용
    "REGIME": 2,       # 둘 다 허용
    "ENV": 2,          # 둘 다 허용
}


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
        #  그룹 제약 추가
        feature_groups=None,     # dict: feature_name -> group_name
        group_limits=None,       # dict: group_name -> max_allowed_in_rule
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

    beams = [(np.ones(N, dtype=bool), [])]
    good = {}

    print('beam', beam)
    print("expand_ratio", expand_ratio, "min_ratio", min_ratio, "min_count", min_count)
    print("cnt_priority_ratio", cnt_priority_ratio)

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
        uid = count()

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}

            # 현재 rule에서 그룹 사용량 계산
            group_used = {}
            for f in used_feats:
                g = get_group(f)
                if g is None:
                    continue
                group_used[g] = group_used.get(g, 0) + 1

            for (lit, lmask) in zip(literals, literal_masks):
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

                m = base_mask & lmask
                cnt = int(m.sum())
                if cnt < min_count:
                    continue

                up = int((m & target).sum())
                ratio = up / cnt

                # good 저장
                if ratio >= min_ratio:
                    key2 = tuple(sorted((c[0], c[1], round(float(c[2]), 6)) for c in (conds + [lit])))
                    prev = good.get(key2)
                    if (prev is None) or (cnt > prev[0]) or (cnt == prev[0] and ratio > prev[1]):
                        good[key2] = (cnt, ratio, up, conds + [lit])

                # 확장 후보
                if ratio >= expand_ratio:
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
        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print("new", len(new))

        if not new:
            print("no expandable candidates; stopping.")
            break

        tail = new[-1]
        print("tail ratio,cnt:", tail[4], tail[5], "conds:", tail[3])

        beams = [(m, conds) for _, _, m, conds, _, _ in new]

    # ---- 최종 out 정렬 ----
    def out_key(x):
        cnt, ratio, up, conds = x
        if ratio >= cnt_priority_ratio:
            return (0, -cnt, -ratio, len(conds))
        else:
            return (1, -ratio, -cnt, len(conds))

    out = sorted(good.values(), key=out_key)
    if top_n is not None:
        out = out[:top_n]

    for cnt, ratio, up, conds in out[:20]:
        print(cnt, ratio, len(conds))

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

# 뎁스가 진행되어도 cnt가 급격히 줄면 안된다
# 뎁스 증가에 따른 ratio 상승이 완만해야 한다

# 국장
rules = mine_rules(
    min_ratio=MIN_RATE, min_count=30, max_depth=5,
    beam=30000, expand_ratio=0.42,
    cnt_priority_ratio=MIN_RATE, top_n=300,
    feature_groups=feature_groups,
    group_limits=group_limits,
)
# 0.75, 30, 4, 0.38, 10000 > 50일 조건 > .... 67%
# 0.75, 30, 4, 0.4, 400 > 50일 조건 > 35/83 > 70%
# 0.75, 30, 4, 0.4, 10000 > 50일 조건 > 42/97 > 70%
# 0.75, 30, 4, 0.42, 10000 > 50일 조건 > 35/85 > 70%
# 0.75, 30, 4, 0.43, 500 > 50일 조건 > 16/39 > 70%
# 0.75, 30, 4, 0.42, 500 > 50일 조건 > 34/85 > 71% ------------------
# 0.75, 30, 4, 0.42, 300 > 50일 조건 > 29/76 > 72%

# 0.8, 30, 5, 0.44, 10000 > 50일 조건 > 20/55 73%
# 0.8, 30, 5, 0.43, 10000 > 50일 조건 > 20/55 73%
# 0.8, 30, 5, 0.42, 1000 > 50일 조건 > 24/62 72%
# 0.8, 30, 5, 0.42, 500 > 50일 조건 > 15/57 79% ---------------------

# 미장
# rules = mine_rules(min_ratio=MIN_RATE, min_count=50, max_depth=6, beam=30000, expand_ratio=0.45, top_n=10000)

top_n = min(10000, len(rules))
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
