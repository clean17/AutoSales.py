import pandas as pd
import numpy as np

df = pd.read_csv("csv/low_result_250409_desc.csv")

TARGET_COL = "validation_chg_rate"
target = (df[TARGET_COL].to_numpy() >= 7)

# 숫자 피처들(원하면 더/덜 제외 가능)
exclude = {"ticker", "stock_name", "predict_str", "today", TARGET_COL}
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
        min_ratio=0.8,
        min_count=20,
        max_depth=4,
        beam=400,
        expand_ratio=0.45,   # 중간 단계 확장용(너무 높이면 길이 막힘)
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
rules = mine_rules(min_ratio=0.9, min_count=20, max_depth=4, beam=500, expand_ratio=0.45)

top_n = min(300, len(rules))   # 필요하면 1000도 가능(조건 엄청 많아짐)
conditions = {}

for i, (cnt, ratio, up, conds) in enumerate(rules[:top_n], start=1):
    name = f"rule_{i:03d}__n{cnt}__r{ratio:.3f}__" + make_name(conds)

    mask = np.ones(len(df), dtype=bool)
    for f, op, thr in conds:
        if op == "<=":
            mask &= (df[f] <= thr)
        else:
            mask &= (df[f] > thr)

    conditions[name] = mask

print(f"총 생성된 조건 수(top_n 기준): {len(conditions)}")
print("가장 큰 count 조건(상위 5개):")
for k in list(conditions.keys())[:5]:
    print(" -", k)

# --- 네 test_condition 돌리기 ---
def test_condition(name, cond):
    sub = df[cond]
    if len(sub) == 0:
        return
    up_cnt = (sub[TARGET_COL] >= 7).sum()
    ratio = up_cnt / len(sub)

    # ✅ 통과: ratio >= 0.8
    if ratio < 0.9:
        return

    print(f"\n=== {name} ===")
    print(f"선택된 행 수: {len(sub)}")
    print(f"  validation_chg_rate >= 7 개수 : {up_cnt}")
    print(f"  Ratio (>=7)      : {ratio:.3f}")

for name, cond in conditions.items():
    test_condition(name, cond)
