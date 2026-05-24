"""
생성한 csv 파일을 가지고 저점 조건을 찾는 스크립트
binning 기반 rule mining
beam search 기반 조건 조합 모델

사용할 때는

import lowscan_avoid_class0_rules
import lowscan_avoid_stop_first_rules

avoid_class0_conditions = lowscan_avoid_class0_rules.build_conditions(df)
avoid_stop_conditions = lowscan_avoid_stop_first_rules.build_conditions(df)

avoid_class0_mask = np.zeros(len(df), dtype=bool)
for cond in avoid_class0_conditions.values():
    avoid_class0_mask |= cond

avoid_stop_mask = np.zeros(len(df), dtype=bool)
for cond in avoid_stop_conditions.values():
    avoid_stop_mask |= cond

final_avoid_mask = avoid_class0_mask | avoid_stop_mask
final_buy_mask = buy_mask & ~final_avoid_mask
"""

import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
import heapq
from itertools import count

# 자동 탐색 (utils.py를 찾을 때까지 위로 올라가 탐색)
here = Path(__file__).resolve()
for parent in [here.parent, *here.parents]:
    if (parent / "utils.py").exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise FileNotFoundError("utils.py를 상위 디렉터리에서 찾지 못했습니다.")

from utils import build_literals, split_train_valid_by_date_ratio, write_rule_file, make_mask_from_conds, get_features

script_dir = os.path.dirname(os.path.abspath(__file__))  # 실행하는 파이썬 파일 위치(root/low)
project_root = os.path.dirname(script_dir)               # root

csv_dir = os.path.join(project_root, "csv")
os.makedirs(csv_dir, exist_ok=True)

CSV_PATH = os.path.join(csv_dir, "low_result_7_desc.csv")
GOOD_OUT_PATH = Path("lowscan_good_rules.py")

GOOD_EXPAND_RATIO = [0.32, 0.52, 0.65, 0.70, 0.75, 0,80, 0.80]
MIN_CNT = 150
MAX_DEPTH = 7
# MIN_RATE = GOOD_EXPAND_RATIO[(MAX_DEPTH-1)]
MIN_RATE = 0.75
BEAM = 30000
TOP_N = 10000

VALID_RATIO = 0.20

# ============================================================
# TRAIN 내부 WALK-FORWARD 검증 설정
# 목적: valid를 보기 전에 train 안에서 과거 -> 미래 검증을 여러 번 수행
# train 내부에서 5개 fold를 만들고,
# 각 fold의 미래 구간에서 룰이 60% 근처로 반복되는지 검사
# ============================================================
USE_TRAIN_WF_FILTER = True

TRAIN_WF_FOLDS = 5
TRAIN_WF_START_RATIO = 0.35
TRAIN_WF_VALID_RATIO = 0.13

TRAIN_WF_MIN_COUNT = 10
TRAIN_WF_MIN_RATE = 0.60
TRAIN_WF_MIN_MEAN_RATE = 0.60
TRAIN_WF_MIN_RECENT_RATE = 0.55
TRAIN_WF_MIN_PASS_FOLDS = 3
TRAIN_WF_MIN_RECENT_COUNT = 8

# ============================================================
# Wilson lower bound 설정
# 목적: 성공률이 같아도 표본 수가 작은 룰을 보수적으로 평가
# 기본값 False: 기존 선택 결과는 바꾸지 않고 출력만 추가
# True: train_wilson_low 기준 필터까지 적용
# ============================================================
USE_WILSON_FILTER = False
TRAIN_WILSON_LOW_MIN = 0.45
WILSON_Z = 1.96

redule_rule = 1  # 중복 룰 제거 조건









def get_feature_groups():
    feature_groups = {
        "vol5": "VOLATILITY",
        "vol_ratio_5_15": "VOLATILITY",

        "today_pct": "PRICE",
        "max_drop_7d": "DROP",
        "gap_pct": "GAP",

        "pct_vs_lastweek": "WEEK_POSITION",
        "dist_to_ma5": "POSITION",
        "ma5_chg_rate": "TREND",

        "today_tr_val_eok": "VOLUME",

        "BB_perc": "BAND",

        "lower_wick_ratio": "CANDLE",
        "upper_wick_ratio": "CANDLE",
        "body_ratio": "CANDLE",
        "intraday_return": "INTRADAY",

        "rebound_from_7d_low": "REBOUND",
        "rebound_vs_prior_drop": "REBOUND",

        "price_power_value": "POWER",
        "body_value_power": "POWER",

        "room_to_20d_high": "HIGH_ROOM",
        "room_to_60d_high": "HIGH_ROOM",
    }

    group_limits = {
        "VOLATILITY": 2,
        "PRICE": 1,
        "DROP": 1,
        "GAP": 1,
        "WEEK_POSITION": 1,
        "POSITION": 1,
        "TREND": 1,
        "VOLUME": 1,
        "BAND": 1,
        "CANDLE": 2,
        "INTRADAY": 1,
        "REBOUND": 2,
        "POWER": 1,
        "HIGH_ROOM": 1,
    }

    return feature_groups, group_limits


def wilson_lower_bound(success, total, z=WILSON_Z):
    """
    Wilson score interval lower bound.
    단순 성공률보다 표본 수가 작은 룰을 보수적으로 평가한다.

    예:
        8/10 = 80%
        80/100 = 80%

    단순 성공률은 같지만 Wilson lower bound는 80/100 쪽이 더 높다.
    """
    if total <= 0:
        return 0.0

    p = success / total
    denom = 1.0 + (z * z / total)
    center = p + (z * z / (2 * total))
    margin = z * np.sqrt((p * (1 - p) + (z * z / (4 * total))) / total)

    return max(0.0, (center - margin) / denom)


def mine_good_rules(
        df,
        literals,
        literal_masks,
        target,
        min_ratio=0.75,
        min_count=50,
        max_depth=4,
        beam=30000,
        top_n=1000,
        feature_groups=None,
        group_limits=None,
        cnt_priority_ratio=0.80,
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
        "\ntop_n", top_n,
        "\nmin_ratio", min_ratio,
        "\nmin_count", min_count,
        "\nmax_depth", max_depth,
        "\nexpand_ratio", GOOD_EXPAND_RATIO,
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

                # 동일 feature 중복 금지
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



def test_good_condition(name, cond, df, min_count=20, verbose=False):
    """
    룰 검증
    """
    # 조건을 만족한 종목/날짜만
    sub = df[cond]

    if len(sub) == 0:
        return False

    up_cnt = int((sub["target_before_stop_7"] == 1).sum())
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


def make_train_walk_forward_folds(df, n_folds=TRAIN_WF_FOLDS):
    """
    train 내부 walk-forward fold 생성.
    각 fold는 train 내부에서 과거 구간을 학습으로 보고,
    바로 다음 구간을 내부 검증 구간으로 사용한다.

    반환 예:
        [{"fold": 1, "train_idx": ..., "valid_idx": ...}, ...]
    """
    n = len(df)
    folds = []

    if n <= 0:
        return folds

    start_train_end = int(n * TRAIN_WF_START_RATIO)
    valid_size = max(1, int(n * TRAIN_WF_VALID_RATIO))
    remaining = n - start_train_end - valid_size

    if remaining < 0:
        return folds

    step = max(1, remaining // max(1, n_folds - 1)) if n_folds > 1 else valid_size

    for i in range(n_folds):
        train_end = start_train_end + i * step
        valid_start = train_end
        valid_end = min(n, valid_start + valid_size)

        if train_end <= 0 or valid_end <= valid_start:
            continue

        folds.append({
            "fold": i + 1,
            "train_idx": np.arange(0, train_end),
            "valid_idx": np.arange(valid_start, valid_end),
        })

    return folds


def eval_rule_train_walk_forward(train, conds, folds):
    """
    하나의 룰을 train 내부 walk-forward fold별로 평가한다.
    target은 target_before_stop_7 == 1 기준으로 고정한다.
    """
    if not folds:
        return {
            "wf_pass": True,
            "wf_active_folds": 0,
            "wf_pass_folds": 0,
            "wf_mean_rate": 0.0,
            "wf_min_rate": 0.0,
            "wf_recent_rate": 0.0,
            "wf_total_count": 0,
            "wf_total_success": 0,
            "wf_rows": [],
        }

    full_mask = make_mask_from_conds(train, conds)
    target = (train["target_before_stop_7"].to_numpy() == 1)

    rows = []
    active_rates = []
    pass_folds = 0
    total_count = 0
    total_success = 0

    for f in folds:
        vi = f["valid_idx"]
        m = full_mask[vi]
        y = target[vi]

        cnt = int(m.sum())
        suc = int((m & y).sum())
        rate = suc / cnt if cnt else 0.0

        if cnt > 0:
            active_rates.append(rate)
            total_count += cnt
            total_success += suc

        if cnt >= TRAIN_WF_MIN_COUNT and rate >= TRAIN_WF_MIN_RATE:
            pass_folds += 1

        rows.append({
            "fold": f["fold"],
            "count": cnt,
            "success": suc,
            "rate": rate,
        })

    recent_count = rows[-1]["count"] if rows else 0
    recent_rate = rows[-1]["rate"] if rows else 0.0
    mean_rate = float(np.mean(active_rates)) if active_rates else 0.0
    min_rate = float(np.min(active_rates)) if active_rates else 0.0
    active_folds = len(active_rates)

    wf_pass = (
            pass_folds >= TRAIN_WF_MIN_PASS_FOLDS
            and mean_rate >= TRAIN_WF_MIN_MEAN_RATE
            and recent_count >= TRAIN_WF_MIN_RECENT_COUNT
            and recent_rate >= TRAIN_WF_MIN_RECENT_RATE
    )

    return {
        "wf_pass": wf_pass,
        "wf_active_folds": active_folds,
        "wf_pass_folds": pass_folds,
        "wf_mean_rate": mean_rate,
        "wf_min_rate": min_rate,
        "wf_recent_rate": recent_rate,
        "wf_total_count": total_count,
        "wf_total_success": total_success,
        "wf_rows": rows,
    }


def eval_selected_train_walk_forward(train, selected, folds, title="TRAIN_WF"):
    """
    최종 OR 룰셋을 train 내부 walk-forward 검증 구간들에서 평가한다.
    """
    if not folds:
        print(f"\n[{title}] no walk-forward folds")
        return

    full_mask = np.zeros(len(train), dtype=bool)
    for name, conds in selected:
        full_mask |= make_mask_from_conds(train, conds)

    target = (train["target_before_stop_7"].to_numpy() == 1)
    oof_mask = np.zeros(len(train), dtype=bool)

    print(f"\n[{title}] combined walk-forward")
    print("rule_count:", len(selected))

    total_cnt = 0
    total_suc = 0
    rates = []

    for f in folds:
        vi = f["valid_idx"]
        m = full_mask[vi]
        y = target[vi]

        cnt = int(m.sum())
        suc = int((m & y).sum())
        rate = suc / cnt if cnt else 0.0

        oof_mask[vi] = m

        if cnt > 0:
            rates.append(rate)
            total_cnt += cnt
            total_suc += suc

        print(
            f"fold {f['fold']}: count={cnt} success={suc} "
            f"rate={rate * 100:.2f}%"
        )

    oof_cnt = int(oof_mask.sum())
    oof_suc = int((oof_mask & target).sum())
    oof_rate = oof_suc / oof_cnt if oof_cnt else 0.0

    print("oof_count:", oof_cnt)
    print("oof_success:", oof_suc)
    print("oof_rate:", round(oof_rate * 100, 2), "%")
    print("mean_active_fold_rate:", round(float(np.mean(rates)) * 100, 2) if rates else 0.0, "%")
    print("min_active_fold_rate:", round(float(np.min(rates)) * 100, 2) if rates else 0.0, "%")
    print("recent_fold_rate:", round(rates[-1] * 100, 2) if rates else 0.0, "%")


def find_good_rule(m_ratio, m_count):
    df = pd.read_csv(CSV_PATH, low_memory=False)

    df["today"] = pd.to_datetime(df["today"], errors="coerce")  # 문자열 > 시간, 에러나면 NaT 변환
    df = df.dropna(subset=["today"])  # NaT 제거

    train, final_valid, split_date = split_train_valid_by_date_ratio(df, valid_ratio=VALID_RATIO)

    # train 내부 walk-forward 검증 fold 생성
    train_wf_folds = make_train_walk_forward_folds(train)

    print("[TRAIN WF] fold_count:", len(train_wf_folds))
    for f in train_wf_folds:
        print(
            "[TRAIN WF]",
            "fold", f["fold"],
            "train_n", len(f["train_idx"]),
            "valid_n", len(f["valid_idx"])
        )
    print("\n")

    # 룰 생성은 train으로만 한다
    features = get_features(train)
    literals, literal_masks = build_literals(train, features)

    feature_groups, group_limits = get_feature_groups()
    target = train["target_before_stop_7"].to_numpy() == 1  # metric 통일

    rules = mine_good_rules(
        df=train,
        literals=literals,
        literal_masks=literal_masks,
        target=target,
        min_ratio=m_ratio,
        min_count=m_count,
        max_depth=MAX_DEPTH,
        beam=BEAM,
        top_n=TOP_N,
        feature_groups=feature_groups,
        group_limits=group_limits,
    )

    fail_reason = {
        "train_count": 0,
        "train_ratio": 0,
        "train_wf": 0,
        "train_wilson": 0,
        "selected": 0,
    }

    debug_candidates = []
    passed = []

    for i, (cnt, ratio, up, conds, score) in enumerate(rules, start=1):
        train_mask = make_mask_from_conds(train, conds)

        train_sub = train[train_mask]

        train_cnt = len(train_sub)

        if train_cnt > 0:
            train_up = int((train_sub["target_before_stop_7"] == 1).sum())
            train_ratio = train_up / train_cnt
        else:
            train_up = 0
            train_ratio = 0.0

        train_wilson_low = wilson_lower_bound(train_up, train_cnt)
        if USE_WILSON_FILTER and train_wilson_low < TRAIN_WILSON_LOW_MIN:
            fail_reason["train_wilson"] += 1
            continue

        wf = eval_rule_train_walk_forward(train, conds, train_wf_folds)

        debug_candidates.append({
            "train_count": train_cnt,
            "train_success_cnt": train_up,
            "train_success_rate": round(train_ratio * 100, 1),
            "score": round(float(score), 6),
            "train_wilson_low": round(train_wilson_low * 100, 1),
            "conds": conds,

            "wf_pass": wf["wf_pass"],
            "wf_active_folds": wf["wf_active_folds"],
            "wf_pass_folds": wf["wf_pass_folds"],
            "wf_total_count": wf["wf_total_count"],
            "wf_total_success": wf["wf_total_success"],
            "wf_mean_rate": round(wf["wf_mean_rate"] * 100, 1),
            "wf_min_rate": round(wf["wf_min_rate"] * 100, 1),
            "wf_recent_rate": round(wf["wf_recent_rate"] * 100, 1),
        })

        if train_cnt < m_count:
            fail_reason["train_count"] += 1
            continue

        if train_ratio < m_ratio:
            fail_reason["train_ratio"] += 1
            continue

        if USE_TRAIN_WF_FILTER and not wf["wf_pass"]:
            fail_reason["train_wf"] += 1
            continue

        fail_reason["selected"] += 1

        passed.append({
            "conds": conds,
            "train_count": train_cnt,
            "train_success_cnt": train_up,
            "train_ratio": train_ratio,
            "train_wilson_low": train_wilson_low,

            "wf_active_folds": wf["wf_active_folds"],
            "wf_pass_folds": wf["wf_pass_folds"],
            "wf_mean_rate": wf["wf_mean_rate"],
            "wf_min_rate": wf["wf_min_rate"],
            "wf_recent_rate": wf["wf_recent_rate"],

            "score": score,
        })

    # ============================================================
    # DEBUG 출력
    # ============================================================
    debug_df = pd.DataFrame(debug_candidates)

    if len(debug_df):
        print("\n[DEBUG] top train/wf candidates")

        tmp = debug_df.sort_values(
            ["wf_pass", "wf_mean_rate", "wf_recent_rate", "train_success_rate", "train_count"],
            ascending=[False, False, False, False, False],
        )

        print(
            tmp.head(30)[[
                "train_count",
                "train_success_rate",
                "train_wilson_low",
                "wf_pass",
                "wf_pass_folds",
                "wf_mean_rate",
                "wf_min_rate",
                "wf_recent_rate",
                "wf_total_count",
                "wf_total_success",
                "conds",
            ]].to_string(index=False)
        )

        print("\n[DEBUG] train/wf candidate summary")
        print("candidate_count:", len(debug_df))
        print("wf_pass count:", int(debug_df["wf_pass"].sum()))
        print("wf_mean_rate max:", debug_df["wf_mean_rate"].max())
        print("wf_mean_rate p90:", np.percentile(debug_df["wf_mean_rate"], 90))
        print("wf_mean_rate p50:", np.percentile(debug_df["wf_mean_rate"], 50))
        print("wf_recent_rate max:", debug_df["wf_recent_rate"].max())
        print("wf_recent_rate p90:", np.percentile(debug_df["wf_recent_rate"], 90))
        print("wf_recent_rate p50:", np.percentile(debug_df["wf_recent_rate"], 50))

    print("[GOOD] fail reason:", fail_reason)
    print(f"\n[GOOD] train/wf 통과 룰 개수: {len(passed)} / {len(rules)}")

    # ============================================================
    # WF/train 성능순으로 통과 룰 정렬
    # ============================================================
    passed = sorted(
        passed,
        key=lambda x: (
            x["wf_mean_rate"],
            x["wf_recent_rate"],
            x["train_wilson_low"],
            x["train_ratio"],
            x["train_count"],
            x["score"],
        ),
        reverse=True,
    )

    selected = []

    for row in passed:
        name = f"rule_{len(selected) + 1:03d}"
        selected.append((name, row["conds"]))

    # ============================================================
    # 중복 룰 제거
    # ============================================================
    if redule_rule == 1:
        print(f"[GOOD] before reduce: {len(selected)}")
        selected = reduce_rules_by_new_rows(train, selected, min_new_rows=3)
        print(f"[GOOD] after reduce: {len(selected)}")

    # reduce 이후 이름 재정렬
    selected = [
        (f"rule_{i + 1:03d}", conds)
        for i, (_, conds) in enumerate(selected)
    ]


    # ============================================================
    # 최종 선택 룰 평가 / 저장
    # ============================================================
    eval_combined_good_rules(train, selected, title="TRAIN")
    eval_selected_train_walk_forward(train, selected, train_wf_folds, title="TRAIN_WF")
    eval_combined_good_rules(final_valid, selected, title="FINAL_VALID")

    write_rule_file(
        GOOD_OUT_PATH,
        selected,
        header_comment=(
            "# auto-generated: lowscan good buy rules\n"
            "# train/valid split applied\n"
            f"# split_date: {pd.to_datetime(split_date).date()}\n"
            f"# train_min_rate: {m_ratio}\n"
            f"# train_min_count: {m_count}\n"
            f"# use_wilson_filter: {USE_WILSON_FILTER}\n"
            f"# train_wilson_low_min: {TRAIN_WILSON_LOW_MIN}\n"
            f"# train_wf_filter: {USE_TRAIN_WF_FILTER}\n"
            f"# train_wf_min_rate: {TRAIN_WF_MIN_RATE}\n"
            f"# train_wf_min_mean_rate: {TRAIN_WF_MIN_MEAN_RATE}\n"
            f"# train_wf_min_recent_rate: {TRAIN_WF_MIN_RECENT_RATE}\n"
            f"# train_wf_min_pass_folds: {TRAIN_WF_MIN_PASS_FOLDS}\n"
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



def eval_combined_good_rules(df, selected, title=""):
    buy_mask = np.zeros(len(df), dtype=bool)

    for name, conds in selected:
        buy_mask |= make_mask_from_conds(df, conds)

    sub = df[buy_mask]

    total_count = len(df)
    selected_count = len(sub)  # 룰 합집합에 의해 선택된 행 수

    target_mask_all = (df["target_before_stop_7"] == 1)
    positive_total = int(target_mask_all.sum())

    if selected_count > 0:
        target_mask_selected = (sub["target_before_stop_7"] == 1)
        positive_count = int(target_mask_selected.sum())
    else:
        positive_count = 0

    wilson_low = wilson_lower_bound(positive_count, selected_count)

    # 선택은 했는데 정답이 아님
    false_positive_count = selected_count - positive_count
    # 룰이 선택 못한 양성
    missed_positive_count = positive_total - positive_count

    # 정밀도
    precision = positive_count / selected_count if selected_count else 0.0
    # 전체 데이터의 양성 비율(베이스라인)
    base_positive_rate = positive_total / total_count if total_count else 0.0
    # 전체 중에서 룰이 선택한 비율, 룰이 얼마나 넓게 커버하느냐(선택 범위 크기)
    selected_coverage = selected_count / total_count if total_count else 0.0
    # 양성 커버리지, 재현율(Recall), 전체 양성 중에서 룰이 잡아낸 양성 비율
    positive_coverage = positive_count / positive_total if positive_total else 0.0
    # 룰 적용 후 양성비율이 베이스라인 대비 몇 배 좋아졌는지
    lift = precision / base_positive_rate if base_positive_rate else 0.0

    print(f"\n[{title}] combined good rules")
    print("rule_count:", len(selected))
    print("total_count:", total_count)
    print("selected_count:", selected_count)
    print("selected_coverage:", round(selected_coverage * 100, 2), "%")
    print("positive_count:", f"{positive_count} / {positive_total}")
    print("precision:", round(precision * 100, 2), "%")
    print("wilson_low:", round(wilson_low * 100, 2), "%")
    print("base_positive_rate:", round(base_positive_rate * 100, 2), "%")
    print("lift:", round(lift, 3), "x")
    print("positive_coverage:", round(positive_coverage * 100, 2), "%")
    print("false_positive_count:", false_positive_count)
    print("missed_positive_count:", missed_positive_count)

    if len(sub) == 0:
        return

    print("target_before_stop_7:", round((sub["target_before_stop_7"] == 1).mean() * 100, 2), "%")
    print("target_before_stop_12:", round((sub["target_before_stop_12"] == 1).mean() * 100, 2), "%")

    if "target_class" in sub.columns:
        print("class0:", round((sub["target_class"] == 0).mean() * 100, 2), "%")
        print("class1:", round((sub["target_class"] == 1).mean() * 100, 2), "%")
        print("class2:", round((sub["target_class"] == 2).mean() * 100, 2), "%")
        print("class3:", round((sub["target_class"] == 3).mean() * 100, 2), "%")

    if "validation_high_rate_max" in sub.columns:
        print("avg_high:", round(sub["validation_high_rate_max"].mean(), 2))
        print("median_high:", round(sub["validation_high_rate_max"].median(), 2))

    if "validation_low_rate_min" in sub.columns:
        print("avg_low:", round(sub["validation_low_rate_min"].mean(), 2))
        print("median_low:", round(sub["validation_low_rate_min"].median(), 2))


def reduce_rules_by_new_rows(df, selected, min_new_rows=2):
    final = []
    used_mask = np.zeros(len(df), dtype=bool)

    for name, conds in selected:
        rule_mask = make_mask_from_conds(df, conds)
        new_rows = rule_mask & ~used_mask
        new_cnt = int(new_rows.sum())
        total_cnt = int(rule_mask.sum())

        if new_cnt >= min_new_rows:
            final.append((name, conds))
            used_mask |= rule_mask
            print(
                f"[REDUCE KEEP] {name} total_cnt={total_cnt} new_cnt={new_cnt} conds={conds}"
            )
        else:
            print(
                f"[REDUCE DROP] {name} total_cnt={total_cnt} new_cnt={new_cnt} conds={conds}"
            )

    print("[REDUCE] final_rule_count:", len(final))
    print("[REDUCE] final_row_count:", int(used_mask.sum()))

    return final



if __name__ == "__main__":
    for i in range(1):
        for_cnt = MIN_CNT + (i*1)

        for j in range(1):
            for_rate = MIN_RATE + (0.01*j)
            find_good_rule(for_rate, for_cnt)

    try:
        import winsound
        winsound.Beep(1500, 500)  # 1000Hz, 0.5초
        winsound.Beep(1000, 500)  # 1000Hz, 0.5초
    except ImportError:
        pass

