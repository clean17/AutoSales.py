"""
생성한 csv 파일을 가지고 저점 조건을 찾는 스크립트

사용할 때는

buy_conditions = lowscan_rules.build_conditions(df)
avoid_conditions = lowscan_avoid_rules.build_conditions(df)

buy_mask = np.zeros(len(df), dtype=bool)
for cond in buy_conditions.values():
    buy_mask |= cond

avoid_mask = np.zeros(len(df), dtype=bool)
for cond in avoid_conditions.values():
    avoid_mask |= cond

final_buy_mask = buy_mask & ~avoid_mask
"""

import pandas as pd
import numpy as np
from pathlib import Path
import heapq
from itertools import count


CSV_PATH = "csv/low_result_7_desc.csv"

MIN_RATE        = 0.90
MIN_CNT         = 80
MIN_CNT_AVOID   = 30
MAX_DEPTH       = 4
MAX_DEPTH_AVOID = 5

GOOD_EXPAND_RATIO = [0.45, 0.65, 0.85, 0.90]

"""
MIN_BAD_RATE = 0.50      # 룰에 걸린 종목 중 class_0이 최소 55% 이상이어야 저장
MAX_PROTECT_RATE = 0.40  # 룰에 걸린 종목 중 class_2+3이 35% 이하여야 저장
MAX_STRONG_RATE = 0.20   # 룰에 걸린 종목 중 class_3이 25% 이하여야 저장
AVOID_EXPAND_BAD_RATIO   # 해당 룰에 걸린 종목 중 class0 비율이 최소 몇 % 이상이어야 다음 단계로 확장할지
AVOID_EXPAND_MAX_STRONG_RATE # 해당 룰에 걸린 종목 중 class3 비율이 최대 몇 % 이하여야 다음 단계로 확장할지
"""
CLASS_3_LOSS_LIMIT = 0.013
AVOID_TOP_N = 5000

MIN_BAD_RATE = 0.60
MAX_PROTECT_RATE = 0.25
MAX_STRONG_RATE = 0.20

AVOID_EXPAND_BAD_RATIO = [0.33, 0.45, 0.58, 0.68, 0.78]
AVOID_EXPAND_MAX_STRONG_RATE = [0.50, 0.40, 0.30, 0.20, 0.12]

CLASS_2_SCORE = 2.0
CLASS_3_SCORE = 2.5


GOOD_OUT_PATH = Path("lowscan_rules.py")
GOOD_EVAL_PATH = Path("csv/good_rule_eval.csv")
AVOID_OUT_PATH = Path("lowscan_avoid_rules.py")
AVOID_EVAL_PATH = Path("csv/class0_remove_rule_eval.csv")





def add_target_class(df):
    """
    7일간 최고가로 클래스를 분류, df에 "target_class"가 없을 경우 생성
    """
    high_cols = [f"validation_high_rate{i}" for i in range(1, 8)]

    df["max_high_7d"] = df[high_cols].max(axis=1)

    df["target_class"] = np.select(
        [
            df["max_high_7d"] < 5,
            df["max_high_7d"] < 7,
            df["max_high_7d"] < 10,
            df["max_high_7d"] >= 10,
            ],
        [0, 1, 2, 3]
    )

    return df


def get_exclude_columns():
    """
    제외할 피쳐 Set
    """
    return {
        "ticker", "stock_name", "predict_str", "today", "idx",

        "validation_high_rate_max",
        "validation_high_rate_max_adj",

        # 저가 기준 최저 수익률
        "validation_low_rate_min",

        # 종가 기준 일별 수익률
        "validation_close_rate1",
        "validation_close_rate2",
        "validation_close_rate3",
        "validation_close_rate4",
        "validation_close_rate5",
        "validation_close_rate6",
        "validation_close_rate7",

        # 고가 기준 일별 수익률
        "validation_high_rate1",
        "validation_high_rate2",
        "validation_high_rate3",
        "validation_high_rate4",
        "validation_high_rate5",
        "validation_high_rate6",
        "validation_high_rate7",

        # 저가 기준 일별 수익률
        "validation_low_rate1",
        "validation_low_rate2",
        "validation_low_rate3",
        "validation_low_rate4",
        "validation_low_rate5",
        "validation_low_rate6",
        "validation_low_rate7",

        # 시가 기준 일별 수익률
        "validation_open_rate1",
        "validation_open_rate2",
        "validation_open_rate3",
        "validation_open_rate4",
        "validation_open_rate5",
        "validation_open_rate6",
        "validation_open_rate7",

        "is_success",
        "is_success5",
        "is_success7",
        "is_success10",
        "target_pct",
        "target_class",
        "stop_loss",

        "_close_pos_20d",
        "_tr_value_ratio",
        "_tr_value_ratio_5d",
        "_dist_to_high_20d",
        "_BB_perc",
        "_UltimateOsc",
        "_CCI14",
        "_ADX14",
        "_gap_pct",
        "_vol_ratio_15_60",
        "_RSI_rebound",
        "_rebound_power",
        "_MACD_hist_1d",
        "_MACD_acc",
        "_MACD_hist_3d_close_norm",
    }


def get_features(df):
    exclude = get_exclude_columns()
    return [c for c in df.columns if c not in exclude]


def get_feature_groups():
    feature_groups = {
        # 가격 / 당일 반등 강도
        "today_pct": "PRICE",
        # 최근 눌림 강도
        "max_drop_7d": "DROP",

        # 저점 대비 회복 위치
        "dist_from_low_20d": "POSITION",
        # 단기 이평선 대비 위치
        "dist_to_ma5": "MA_POSITION",
        # 추세 개선
        "ma5_ma20_gap_chg_1d": "TREND",

        # 거래대금 / 수급
        "today_tr_val_eok": "VOLUME",
        "tr_val_rank_20d": "VOLUME",

        # MACD 모멘텀
        "MACD_hist_3d": "MACD",
        # 변동성
        "ATR_pct": "VOLATILITY",
    }

    group_limits = {
        "PRICE": 1,
        "DROP": 1,
        "POSITION": 1,
        "MA_POSITION": 1,
        "TREND": 1,
        "VOLUME": 1,
        "MACD": 1,
        "VOLATILITY": 1,
    }

    return feature_groups, group_limits


def build_literals(df, features):
    """
    literals(원자 조건)을 만드는 함수: feature <= q or feature > q
    """
    literals = []
    literal_masks = []

    for f in features:
        col = df[f].astype(float).to_numpy()  # flaat 형변환 >  Series > Numpy 배열
        col_nonan = col[~np.isnan(col)]       # ~np.isnan() 으로 True 값만 남는다 > NaN 제거

        if len(col_nonan) == 0:
            continue

        print(f, len(col_nonan), len(np.unique(col_nonan)))

        # 분위수(percentile) 배열 생성 (덜 촘촘하게 = 과적합 방지하면서 빠르게)
        unique_vals = np.unique(col_nonan)

        if len(unique_vals) < 50:  # quantile 안씀
            qs = unique_vals
        else:
            """
            0.05, 0.10, 0.15, 0.20, ... 0.95 → 총 19개 threshold
            0.1, 0.2, 0.3, ... 0.9           → 총 9개 threshold
            
            depth4에서 데드캣을 이미 거른 뒤 depth5를 만들 거면 np.linspace(0.2, 0.8, 7) 괜찮음
            0.05~0.95, 19개	많음	세밀하지만 과적합 위험 큼
            0.1~0.9, 9개	중간	균형
            0.2~0.8, 7개	적음	안정적, 룰 적음, 과적합 감소
            """
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

    return literals, literal_masks


def mine_good_rules(
        df,
        literals,
        literal_masks,
        target,
        min_ratio=0.83,
        min_count=25,
        max_depth=4,
        beam=30000,
        top_n=1000,
        feature_groups=None,
        group_limits=None,
        cnt_priority_ratio=0.85,
):
    """
    beam : 단계별 상위 수량만 다음 뎁스로 가져간다, 한 depth에서 유지할 후보 개수
    expand_ratio : 각 후보의 성능이 이 값보다 커야 다음 뎁스로 진행, 다음 depth로 확장할 최소 기준
    min_ratio : “좋은 룰”로 저장할 기준
    cnt_priority_ratio : 넘기면 ratio 보다 cnt 를 더 중요시
    - ratio >= cnt_priority_ratio 이면 cnt 우선(그 다음 ratio)
    - ratio <  cnt_priority_ratio 이면 ratio 우선(그 다음 cnt)

    권장 depth
    | depth | 적당한 new 개수 | 해석              |
    | ----: | -------------: | ---------------- |
    |     0 |       50 ~ 500 | 단일 조건 후보     |
    |     1 |    500 ~ 5,000 | 2개 조건 조합      |
    |     2 | 1,000 ~ 10,000 | 3개 조건 조합      |
    |     3 | 1,000 ~ 10,000 | 4개 조건 조합      |
    |     4 |    100 ~ 3,000 | 5개 조건 최종 후보 |

    :returns (count, ratio, up_cnt, conds) 리스트 반환
    """

    # df 길이 만큼의 True 배열(불리언 마스크), 빈 배열의 튜플 리스트 > "데이터프레임의 모든 행을 선택한다"는 상태
    beams = [(np.ones(len(df), dtype=bool), [])]
    good = {}

    print(
        "\nbeam", beam,
        "\nmin_ratio", min_ratio,
        "\nmin_count", min_count,
        "\nmax_depth", max_depth,
        "\nexpand_ratio", GOOD_EXPAND_RATIO,
        "\ntop_n", top_n,
        "\n"
    )

    if feature_groups is None:
        feature_groups = {}
    if group_limits is None:
        group_limits = {}

    def get_group(feat_name):
        return feature_groups.get(feat_name)

    def beam_key(ratio, cnt):
        """
        beam(확장 후보)용 비교키
        "좋은 후보"가 더 큰 key를 갖도록 설계 (나중에 key 비교로 worst 교체)
        """
        if ratio >= cnt_priority_ratio:
            return (1, cnt, ratio)  # cnt 우선
        return (0, ratio, cnt)      # ratio 우선

    for depth in range(max_depth):
        print("----------------------------------")
        print("[GOOD] depth", depth)

        # heap item: (key, uid, mask, conds, ratio, cnt)
        # heap[0] = 가장 "나쁜" 후보 (key가 가장 작음)
        heap = []
        uid = count()  # 카운트 제네레이터 (itertools 모듈)

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}

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

                if cnt < min_count:
                    continue

                up = int((m & target).sum())
                ratio = up / cnt
                # score = ratio * np.log1p(cnt)
                # score = (ratio ** 2) * np.log1p(cnt)
                score = (ratio ** 3) * np.log1p(cnt)  # ratio에 더 강하게 보상

                # good 저장
                if ratio >= min_ratio:
                    key2 = tuple(sorted(
                        (c[0], c[1], round(float(c[2]), 6))
                        for c in (conds + [lit])
                    ))

                    prev = good.get(key2)
                    if prev is None or score > prev[4]:
                        good[key2] = (cnt, ratio, up, conds + [lit], score)

                """
                new ≈ beam의 30~70%가 되도록 expand_ratio 조절 필요

                depth 0 >> new가 beam의 1~20%여도 괜찮음
                depth 1 >> beam의 20~70%
                depth 2+ > beam의 30~70%
                """
                # 확장 후보
                if ratio >= GOOD_EXPAND_RATIO[depth]:
                    k = beam_key(ratio, cnt)
                    item = (k, next(uid), m, conds + [lit], ratio, cnt)

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        if k <= heap[0][0]:
                            continue
                        heapq.heapreplace(heap, item)

        # 다음 depth로 넘길 후보들(좋은 순): key 내림차순
        # key는 (group, primary, secondary)인데 primary/secondary는 큰 게 좋게 만들어 둠
        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print("[GOOD] new", len(new))

        if not new:
            print("[GOOD] no expandable candidates; stopping.")
            break

        tail = new[-1]
        print("[GOOD] tail ratio:", round(tail[4], 3), "cnt:", tail[5], "conds:", tail[3])  # 최악 후보 출력 (디버깅용)

        beams = [(m, conds) for _, _, m, conds, _, _ in new]

    # 점수 내림차순 정렬
    def out_key(x):
        cnt, ratio, up, conds, score = x
        return (-score, -ratio, -cnt, len(conds))

    out = sorted(good.values(), key=out_key)

    if top_n is not None:
        out = out[:top_n]

    return out


def mine_avoid_rules(
        df,
        literals,
        literal_masks,
        bad,
        protect,
        strong,
        min_count=25,
        max_depth=4,
        beam=30000,
        top_n=1000,
        feature_groups=None,
        group_limits=None,
):
    beams = [(np.ones(len(df), dtype=bool), [])]
    good = {}

    print(
        "\nbeam", beam,
        "\nmin_count", min_count,
        "\nmax_depth", max_depth,
        "\nexpand_bad_ratio", AVOID_EXPAND_BAD_RATIO,
        "\nexpand_max_strong_rate", AVOID_EXPAND_MAX_STRONG_RATE,
        "\ntop_n", top_n,
        "\nmin_bad_rate", MIN_BAD_RATE,
        "\nmax_protect_rate", MAX_PROTECT_RATE,
        "\nmax_strong_rate", MAX_STRONG_RATE,
        "\nclass_3_loss_limit", CLASS_3_LOSS_LIMIT,
        "\nclass_2_score", CLASS_2_SCORE,
        "\nclass_3_score", CLASS_3_SCORE,
        "\n"
    )

    if feature_groups is None:
        feature_groups = {}
    if group_limits is None:
        group_limits = {}

    def get_group(feat_name):
        return feature_groups.get(feat_name)

    for depth in range(max_depth):
        print("----------------------------------")
        print("[AVOID] depth", depth)

        heap = []
        uid = count()

        for base_mask, conds in beams:
            used_feats = {c[0] for c in conds}

            group_used = {}
            for f in used_feats:
                g = get_group(f)
                if g is not None:
                    group_used[g] = group_used.get(g, 0) + 1

            for lit, lmask in zip(literals, literal_masks):
                feat = lit[0]

                if feat in used_feats:
                    continue

                g = get_group(feat)
                if g is not None:
                    limit = group_limits.get(g)
                    if limit is not None and group_used.get(g, 0) >= limit:
                        continue

                m = base_mask & lmask
                cnt = int(m.sum())

                if cnt < min_count:
                    continue

                bad_cnt = int((m & bad).sum())
                protect_cnt = int((m & protect).sum())
                strong_cnt = int((m & strong).sum())

                bad_rate = bad_cnt / cnt
                protect_rate = protect_cnt / cnt
                strong_rate = strong_cnt / cnt

                score = (
                        (bad_rate ** 2.5)
                        * np.log1p(cnt)
                        * ((1 - protect_rate) ** 1.5)
                        * ((1 - strong_rate) ** 2.0)
                )

                if (
                        bad_rate >= MIN_BAD_RATE
                        and protect_rate <= MAX_PROTECT_RATE
                        and strong_rate <= MAX_STRONG_RATE
                ):
                    key2 = tuple(sorted(
                        (c[0], c[1], round(float(c[2]), 6))
                        for c in (conds + [lit])
                    ))

                    prev = good.get(key2)
                    if prev is None or score > prev[8]:
                        good[key2] = (
                            cnt,
                            bad_cnt,
                            bad_rate,
                            protect_cnt,
                            protect_rate,
                            strong_cnt,
                            strong_rate,
                            conds + [lit],
                            score,
                        )

                if (
                        # 이 룰 subset 안에 class_0 비율이 충분히 높아야 함
                        bad_rate >= AVOID_EXPAND_BAD_RATIO[depth]
                        # 이 룰 subset 안에 class_3 비율이 너무 높으면 안 됨
                        and strong_rate <= AVOID_EXPAND_MAX_STRONG_RATE[depth]
                ):
                    k = (score, bad_rate, -strong_rate, -protect_rate, cnt)
                    item = (k, next(uid), m, conds + [lit], bad_rate, cnt)

                    if len(heap) < beam:
                        heapq.heappush(heap, item)
                    else:
                        if k <= heap[0][0]:
                            continue
                        heapq.heapreplace(heap, item)

        new = sorted(heap, key=lambda x: x[0], reverse=True)
        print("[AVOID] new", len(new))

        if not new:
            print("[AVOID] no expandable candidates; stopping.")
            break

        tail = new[-1]
        print("[AVOID] tail bad_rate:", round(tail[4], 3), "cnt:", tail[5], "conds:", tail[3])

        beams = [(m, conds) for _, _, m, conds, _, _ in new]

    def out_key(x):
        cnt, bad_cnt, bad_rate, protect_cnt, protect_rate, strong_cnt, strong_rate, conds, score = x
        return (-score, -bad_rate, strong_rate, protect_rate, -cnt)

    out = sorted(good.values(), key=out_key)

    if top_n is not None:
        out = out[:top_n]

    return out


def make_mask_from_conds(df, conds):
    mask = np.ones(len(df), dtype=bool)

    for f, op, thr in conds:
        if op == "<=":
            mask &= (df[f] <= thr)
        else:
            mask &= (df[f] > thr)

    return mask


def test_good_condition(name, cond, df, min_count=20, verbose=False):
    """
    룰 검증
    """
    # 조건을 만족한 종목/날짜만
    sub = df[cond]

    if len(sub) == 0:
        return False

    up_cnt = int((sub["is_success7"] == 1).sum())
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

    if confidence < 0.55:
        return False

    # 표본이 너무 적으면 과적합 가능성이 큼
    if len(sub) < min_count:
        return False

    if verbose:
        print(f"\n=== {name} ===")
        print(f"선택된 행 수: {len(sub)}")
        print(f"성공 개수   : {up_cnt}")
        print(f"성공률      : {ratio:.3f}")
        print(f"confidence : {confidence:.3f}")

    return True


def test_avoid_condition(name, cond, df, min_count=25, verbose=False):
    sub = df[cond]

    if len(sub) == 0:
        return False

    cnt = len(sub)

    class0_cnt = int((sub["target_class"] == 0).sum())
    class2_cnt = int((sub["target_class"] == 2).sum())
    class3_cnt = int((sub["target_class"] == 3).sum())

    class0_rate = class0_cnt / cnt
    protect_rate = (class2_cnt + class3_cnt) / cnt
    class3_rate = class3_cnt / cnt

    # 전체 class별 총량 대비 얼마나 제거했는지도 중요
    total_class0 = int((df["target_class"] == 0).sum())
    total_class2 = int((df["target_class"] == 2).sum())
    total_class3 = int((df["target_class"] == 3).sum())

    class0_remove_rate = class0_cnt / total_class0 if total_class0 else 0
    class2_loss_rate = class2_cnt / total_class2 if total_class2 else 0
    class3_loss_rate = class3_cnt / total_class3 if total_class3 else 0

    # 핵심: class_0은 많이 제거하고 class_2/3은 적게 잃는 룰
    score = (
            class0_remove_rate
            - class2_loss_rate
            - class3_loss_rate * 1.5
    )

    if cnt < min_count:
        return False

    # 룰 subset 내부가 class_0 위주여야 함
    if class0_rate < MIN_BAD_RATE:
        return False

    # 좋은 종목이 너무 많이 섞이면 제외 룰로 부적합
    if protect_rate > MAX_PROTECT_RATE:
        return False

    # class_3 손실은 특히 제한
    if class3_rate > MAX_STRONG_RATE:
        return False

    # 전체 기준으로도 의미 있는 제거 성능이 있어야 함
    if score <= 0:
        return False

    if verbose:
        print(f"\n=== {name} ===")
        print(f"선택된 행 수: {cnt}")
        print(f"class_0 비율: {class0_rate:.3f}")
        print(f"class_2+3 비율: {protect_rate:.3f}")
        print(f"class_0 제거율: {class0_remove_rate:.3f}")
        print(f"class_2 손실률: {class2_loss_rate:.3f}")
        print(f"class_3 손실률: {class3_loss_rate:.3f}")
        print(f"score: {score:.3f}")

    return True


def rule_to_code(name, conds, thr_round=3):
    lines = [f'    "{name}":']
    parts = []

    for f, op, thr in conds:
        thr = float(np.round(thr, thr_round))

        if op == "<=":
            parts.append(f'(df["{f}"] <= {thr})')
        else:
            parts.append(f'(df["{f}"] > {thr})')  # ">": ge로 출력

    joined = " &\n        ".join(parts)
    lines.append(f"        {joined},")

    return "\n".join(lines)


def write_rule_file(out_path, selected, header_comment):
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header_comment + "\n")
        f.write("import numpy as np\n\n")

        f.write("RULE_NAMES = [\n")
        for name, _ in selected:
            f.write(f'    "{name}",\n')
        f.write("]\n\n")

        f.write("def build_conditions(df):\n")
        f.write("    conditions = {\n")

        for name, conds in selected:
            code = rule_to_code(name, conds)
            lines = code.splitlines()

            f.write("        " + lines[0] + "\n")
            for line in lines[1:]:
                f.write("        " + line + "\n")

        f.write("    }\n")
        f.write("    return conditions\n")

    print(f"saved to: {out_path.resolve()}")


def save_good_rule_eval(df, selected, out_csv=GOOD_EVAL_PATH):
    rows = []

    for name, conds in selected:
        mask = make_mask_from_conds(df, conds)
        sub = df[mask]

        if len(sub) == 0:
            continue

        up_cnt = int((sub["is_success7"] == 1).sum())
        ratio = up_cnt / len(sub)

        rows.append({
            "rule": name,
            "count": len(sub),
            "success_cnt": up_cnt,
            "success_rate": round(ratio * 100, 1),
        })

    result = pd.DataFrame(rows)

    if len(result):
        result = result.sort_values(
            ["success_rate", "count"],
            ascending=[False, False]
        )

    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"saved eval to: {out_csv.resolve()}")


def save_avoid_rule_eval(df, selected, out_csv=AVOID_EVAL_PATH):
    rows = []

    total0 = int((df["target_class"] == 0).sum())
    total1 = int((df["target_class"] == 1).sum())
    total2 = int((df["target_class"] == 2).sum())
    total3 = int((df["target_class"] == 3).sum())

    for name, conds in selected:
        mask = make_mask_from_conds(df, conds)
        sub = df[mask]

        if len(sub) == 0:
            continue

        cnt = len(sub)

        c0 = int((sub["target_class"] == 0).sum())
        c1 = int((sub["target_class"] == 1).sum())
        c2 = int((sub["target_class"] == 2).sum())
        c3 = int((sub["target_class"] == 3).sum())

        rows.append({
            "rule": name,
            "remove_count": cnt,

            "class0_cnt": c0,
            "class1_cnt": c1,
            "class2_cnt": c2,
            "class3_cnt": c3,

            "class0_in_rule_rate": round(c0 / cnt * 100, 1),
            "class23_in_rule_rate": round((c2 + c3) / cnt * 100, 1),

            "class0_remove_rate": round(c0 / total0 * 100, 1) if total0 else 0,
            "class1_remove_rate": round(c1 / total1 * 100, 1) if total1 else 0,
            "class2_loss_rate": round(c2 / total2 * 100, 1) if total2 else 0,
            "class3_loss_rate": round(c3 / total3 * 100, 1) if total3 else 0,

            "score": round(
                (c0 / total0 if total0 else 0)
                - (c2 / total2 if total2 else 0)
                - 1.5 * (c3 / total3 if total3 else 0),
                4
            ),
        })

    result = pd.DataFrame(rows)

    if len(result):
        result = result.sort_values(
            ["score", "class0_in_rule_rate", "class3_loss_rate"],
            ascending=[False, False, True]
        )

    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"saved avoid eval to: {out_csv.resolve()}")


def save_combined_avoid_eval(df, selected, out_csv="combined_avoid_eval.csv"):
    avoid_mask = np.zeros(len(df), dtype=bool)

    for name, conds in selected:
        avoid_mask |= make_mask_from_conds(df, conds)

    removed = df[avoid_mask]
    remain = df[~avoid_mask]

    total0 = (df["target_class"] == 0).sum()
    total1 = (df["target_class"] == 1).sum()
    total2 = (df["target_class"] == 2).sum()
    total3 = (df["target_class"] == 3).sum()

    c0 = (removed["target_class"] == 0).sum()
    c1 = (removed["target_class"] == 1).sum()
    c2 = (removed["target_class"] == 2).sum()
    c3 = (removed["target_class"] == 3).sum()

    summary = pd.DataFrame([{
        "total_count": len(df),
        "removed_count": len(removed),
        "remain_count": len(remain),
        "removed_rate": round(len(removed) / len(df) * 100, 1),

        "removed_class0": int(c0),
        "removed_class1": int(c1),
        "removed_class2": int(c2),
        "removed_class3": int(c3),

        "class0_remove_rate": round(c0 / total0 * 100, 1) if total0 else 0,
        "class1_remove_rate": round(c1 / total1 * 100, 1) if total1 else 0,
        "class2_loss_rate": round(c2 / total2 * 100, 1) if total2 else 0,
        "class3_loss_rate": round(c3 / total3 * 100, 1) if total3 else 0,

        "removed_class0_rate": round((removed["target_class"] == 0).mean() * 100, 1),
        "removed_class1_rate": round((removed["target_class"] == 1).mean() * 100, 1),
        "removed_class2_rate": round((removed["target_class"] == 2).mean() * 100, 1),
        "removed_class3_rate": round((removed["target_class"] == 3).mean() * 100, 1),

        "remain_class0_rate": round((remain["target_class"] == 0).mean() * 100, 1),
        "remain_class1_rate": round((remain["target_class"] == 1).mean() * 100, 1),
        "remain_class2_rate": round((remain["target_class"] == 2).mean() * 100, 1),
        "remain_class3_rate": round((remain["target_class"] == 3).mean() * 100, 1),
    }])

    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(summary.T)


def find_good_rule(m_ratio, m_count):
    # df = pd.read_csv(CSV_PATH)
    df = pd.read_csv(CSV_PATH, low_memory=False)

    df["today"] = pd.to_datetime(df["today"])

    train = df[df["today"] < "2026-04-01"].copy()
    valid = df[df["today"] >= "2026-04-01"].copy()

    features = get_features(df)
    literals, literal_masks = build_literals(df, features)
    feature_groups, group_limits = get_feature_groups()
    target = df["is_success7"].to_numpy() == 1

    # features = get_features(train)
    # literals, literal_masks = build_literals(train, features)
    # feature_groups, group_limits = get_feature_groups()
    # target = train["is_success7"].to_numpy() == 1

    rules = mine_good_rules(
        df=df,
        literals=literals,
        literal_masks=literal_masks,
        target=target,
        min_ratio=m_ratio,
        min_count=m_count,
        max_depth=MAX_DEPTH,
        beam=10000,
        top_n=300,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    selected = []

    for i, (cnt, ratio, up, conds, score) in enumerate(rules, start=1):
        name = f"rule_{i:03d}_s{score:.2f}_n{cnt}_r{ratio:.3f}"

        mask = make_mask_from_conds(df, conds)

        # 테스트 통과만
        if test_good_condition(name, mask, df, verbose=False):
            selected.append((name, conds))

        # train_mask = make_mask_from_conds(train, conds)
        # valid_mask = make_mask_from_conds(valid, conds)
        #
        # if not test_good_condition(name, train_mask, train, min_count=MIN_CNT):
        #     continue
        #
        # if not test_good_condition(name, valid_mask, valid, min_count=10):
        #     continue
        #
        # selected.append((name, conds))

    print(f"[GOOD] 통과 룰 개수: {len(selected)} / {len(rules)}")

    write_rule_file(
        GOOD_OUT_PATH,
        selected,
        header_comment=(
            "# auto-generated: lowscan good buy rules\n"
            "# usage:\n"
            "#    import lowscan_rules\n"
            "#    buy_conditions = lowscan_rules.build_conditions(df)\n"
            "#    \n"
            "#    buy_mask = np.zeros(len(df), dtype=bool)\n"
            "#    for cond in buy_conditions.values():\n"
            "#        buy_mask |= cond\n"
            "#    \n"
            "#    df = df[buy_mask].copy()\n"
        )
    )

    save_good_rule_eval(df, selected)


def find_avoid_rule():
    # df = pd.read_csv(CSV_PATH)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    # df = add_target_class(df)

    if "target_class" not in df.columns:
        raise ValueError("target_class 컬럼이 없습니다. 데이터 생성 시 target_class를 먼저 저장하세요.")

    features = get_features(df)
    literals, literal_masks = build_literals(df, features)

    feature_groups, group_limits = get_feature_groups()

    bad = df["target_class"].to_numpy() == 0
    protect = df["target_class"].to_numpy() >= 2
    strong = df["target_class"].to_numpy() == 3

    rules = mine_avoid_rules(
        df=df,
        literals=literals,
        literal_masks=literal_masks,
        bad=bad,
        protect=protect,
        strong=strong,
        min_count=MIN_CNT_AVOID,
        max_depth=MAX_DEPTH_AVOID,
        beam=30000,
        top_n=AVOID_TOP_N,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    selected, avoid_mask = select_avoid_rules_max_class0_under_class3_limit(
        df=df,
        rules=rules,
        class3_loss_limit=CLASS_3_LOSS_LIMIT,  # 2.5%
        min_count=MIN_CNT_AVOID,
        max_rules=None,
        verbose=True,
    )

    print(f"[AVOID] 통과 룰 개수: {len(selected)} / {len(rules)}")

    write_rule_file(
        AVOID_OUT_PATH,
        selected,
        header_comment=(
            "# auto-generated: lowscan avoid rules for class_0\n"
            "# usage:\n"
            "#   import lowscan_avoid_rules\n"
            "#    \n"
            "#   avoid_conditions = lowscan_avoid_rules.build_conditions(df)\n"
            "#   avoid_mask = np.zeros(len(df), dtype=bool)\n"
            "#   for cond in avoid_conditions.values():\n"
            "#       avoid_mask |= cond\n"
            "#    \n"
            "#   df = df[~avoid_mask].copy()\n"
        )
    )

    # save_avoid_rule_eval(df, selected)
    save_combined_avoid_eval(df, selected)


def select_avoid_rules_max_class0_under_class3_limit(
        df,
        rules,
        class3_loss_limit=0.025,
        min_count=40,
        max_rules=None,
        verbose=True,
):
    total_class3 = int((df["target_class"] == 3).sum())
    class3_limit_count = int(total_class3 * class3_loss_limit)

    avoid_mask = np.zeros(len(df), dtype=bool)
    selected = []

    used_class3 = 0

    # 후보 룰을 먼저 만들어둠
    candidates = []

    for i, (
            cnt,
            bad_cnt,
            bad_rate,
            protect_cnt,
            protect_rate,
            strong_cnt,
            strong_rate,
            conds,
            score,
    ) in enumerate(rules, start=1):

        name = (
            f"avoid_{i:03d}"
            f"_s{score:.2f}"
            f"_n{cnt}"
            f"_bad{bad_rate:.3f}"
            f"_p{protect_rate:.3f}"
            f"_strong{strong_rate:.3f}"
        )

        rule_mask = make_mask_from_conds(df, conds)

        if not test_avoid_condition(name, rule_mask, df, min_count=min_count, verbose=False):
            continue

        candidates.append({
            "name": name,
            "conds": conds,
            "rule_mask": rule_mask,
            "score": score,
            "bad_rate": bad_rate,
            "protect_rate": protect_rate,
            "strong_rate": strong_rate,
        })

    # 매번 "현재 avoid_mask 기준으로" 추가 효율이 가장 좋은 룰을 고름
    while True:
        best = None

        for cand in candidates:
            rule_mask = cand["rule_mask"]
            new_remove = rule_mask & ~avoid_mask

            if not new_remove.any():
                continue

            added_class0 = int(((df["target_class"] == 0) & new_remove).sum())
            added_class2 = int(((df["target_class"] == 2) & new_remove).sum())
            added_class3 = int(((df["target_class"] == 3) & new_remove).sum())
            added_total = int(new_remove.sum())

            if added_class0 <= 0:
                continue

            if used_class3 + added_class3 > class3_limit_count:
                continue

            # 핵심 점수:
            # class0 추가 제거를 크게 보상
            # class3 추가 제거를 강하게 패널티
            # class2도 보호해야 하므로 약하게 패널티
            efficiency = (
                    added_class0
                    - CLASS_3_SCORE * added_class3
                    - CLASS_2_SCORE * added_class2
            )

            # 너무 작은 추가 효과는 제외
            if added_total < 5:
                continue

            # 동률이면 class0 순도 높은 것 우대
            added_class0_rate = added_class0 / added_total

            key = (
                efficiency,
                added_class0,
                added_class0_rate,
                -added_class3,
                cand["score"],
            )

            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "cand": cand,
                    "added_class0": added_class0,
                    "added_class2": added_class2,
                    "added_class3": added_class3,
                    "added_total": added_total,
                }

        if best is None:
            break

        cand = best["cand"]
        avoid_mask |= cand["rule_mask"]
        selected.append((cand["name"], cand["conds"]))
        used_class3 += best["added_class3"]

        if max_rules is not None and len(selected) >= max_rules:
            break

    if verbose:
        removed = df[avoid_mask]
        total0 = int((df["target_class"] == 0).sum())
        total2 = int((df["target_class"] == 2).sum())

        c0 = int((removed["target_class"] == 0).sum())
        c2 = int((removed["target_class"] == 2).sum())
        c3 = int((removed["target_class"] == 3).sum())

        print(
            f"[CLASS3_LIMIT] selected={len(selected)}, "
            f"class0_remove_rate={c0 / total0 * 100:.2f}%, "
            f"class2_loss_rate={c2 / total2 * 100:.2f}%, "
            f"class3_loss_rate={c3 / total_class3 * 100:.2f}% "
            f"({c3}/{class3_limit_count})"
        )

    return selected, avoid_mask



if __name__ == "__main__":
    # MODE = "both"
    MODE = "good"
    # MODE = "avoid"

    for i in range(1):
        for_cnt = MIN_CNT + (i*1)

        for j in range(1):
            for_rate = MIN_RATE + (0.01*j)

            if MODE in ("good", "both"):
                find_good_rule(for_rate, for_cnt)

            if MODE in ("avoid", "both"):
                # class_0 제거용 데드캣 제외 룰 생성
                find_avoid_rule()

    import winsound

    winsound.Beep(1500, 500)  # 1000Hz, 0.5초
    winsound.Beep(1000, 500)  # 1000Hz, 0.5초

