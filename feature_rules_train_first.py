import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


TARGET_COL = "target_before_stop_7"


DEFAULT_FEATURES = [
    "ATR_pct",
    "vol5",
    "vol_ratio_5_15",
    "three_m_cur_max_chg_rate",
    "today_pct",
    "max_drop_7d",
    "gap_pct",
    "dist_to_ma20",
    "pct_vs_lastweek",
    "dist_from_low_20d",
    "dist_to_ma5",
    "ma5_ma20_gap_chg_1d",

    # 이전 결과상 소표본 과최적화 가능성이 커서 기본 제외
    # 필요하면 주석 해제
    # "MACD_hist_3d",

    "today_tr_val_eok",
    "tr_val_rank_20d",
    "tr_value_ratio_5d",
]


@dataclass(frozen=True)
class Atom:
    feature: str
    op: str
    threshold: float

    def name(self):
        return f"{self.feature} {self.op} {self.threshold:.6g}"


@dataclass
class Rule:
    atoms: Tuple[Atom, ...]
    train_metrics: Dict
    valid_metrics: Dict
    train_score: float

    def name(self):
        return " AND ".join([a.name() for a in self.atoms])

    def features(self):
        return sorted(set(a.feature for a in self.atoms))


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "date",
        "Date",
        "datetime",
        "trade_date",
        "trd_date",
        "일자",
        "날짜",
        "기준일",
        "ymd",
        "YMD",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def split_train_valid(
        df: pd.DataFrame,
        date_col: Optional[str],
        valid_ratio: float,
):
    df = df.copy()

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
        print(f"[INFO] sorted by date_col: {date_col}")
    else:
        df = df.reset_index(drop=True)
        print("[WARN] date_col not found. Splitting by current row order.")

    n = len(df)
    split_idx = int(n * (1 - valid_ratio))

    train = df.iloc[:split_idx].copy()
    valid = df.iloc[split_idx:].copy()

    return train, valid


def precision_recall_lift(y, mask):
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask).astype(bool)

    total_n = len(y)
    total_pos = int((y == 1).sum())
    base_rate = total_pos / total_n if total_n else np.nan

    count = int(mask.sum())

    if count == 0:
        return {
            "count": 0,
            "target_count": 0,
            "precision": np.nan,
            "recall": 0.0,
            "lift": np.nan,
            "coverage": 0.0,
            "base_rate": base_rate,
        }

    target_count = int(y[mask].sum())
    precision = target_count / count
    recall = target_count / total_pos if total_pos > 0 else 0.0
    lift = precision / base_rate if base_rate and base_rate > 0 else np.nan
    coverage = count / total_n

    return {
        "count": count,
        "target_count": target_count,
        "precision": precision,
        "recall": recall,
        "lift": lift,
        "coverage": coverage,
        "base_rate": base_rate,
    }


def make_atoms(
        train: pd.DataFrame,
        features: List[str],
        quantiles: List[float],
):
    atoms = []

    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        x = x[np.isfinite(x)]

        if len(x) < 100:
            continue

        if x.nunique() <= 3:
            continue

        qs = np.nanquantile(x, quantiles)
        qs = sorted(set([float(q) for q in qs if np.isfinite(q)]))

        for q in qs:
            atoms.append(Atom(f, ">=", q))
            atoms.append(Atom(f, "<=", q))

    return atoms


def apply_atom(df: pd.DataFrame, atom: Atom):
    x = pd.to_numeric(df[atom.feature], errors="coerce")

    if atom.op == ">=":
        return x >= atom.threshold

    if atom.op == "<=":
        return x <= atom.threshold

    raise ValueError(atom.op)


def apply_rule(df: pd.DataFrame, atoms: Tuple[Atom, ...]):
    mask = np.ones(len(df), dtype=bool)

    for atom in atoms:
        m = apply_atom(df, atom).fillna(False).values
        mask &= m

    return mask


def train_rule_score(
        train_metrics: Dict,
        min_train_count: int,
        min_train_lift: float,
        max_train_coverage: float = 0.50,
):
    """
    TRAIN 기준 점수.

    중요한 점:
    - VALID 정보는 절대 사용하지 않는다.
    - 너무 넓은 룰은 coverage penalty로 점수를 깎는다.
    - lift와 precision을 중심으로 평가한다.
    """

    count = train_metrics["count"]
    precision = train_metrics["precision"]
    lift = train_metrics["lift"]
    target_count = train_metrics["target_count"]
    recall = train_metrics["recall"]
    coverage = train_metrics["coverage"]

    if count < min_train_count:
        return -1e18

    if target_count <= 0:
        return -1e18

    if not np.isfinite(precision) or not np.isfinite(lift):
        return -1e18

    if lift < min_train_lift:
        return -1e18

    # 너무 넓은 룰 패널티
    wide_penalty = 0.0
    if coverage > max_train_coverage:
        wide_penalty = (coverage - max_train_coverage) * 150.0

    # 너무 작은 룰도 약간 패널티
    small_penalty = 0.0
    if count < min_train_count * 1.5:
        small_penalty = 2.0

    score = (
            precision * 100.0
            + lift * 35.0
            + math.log1p(target_count) * 1.2
            + recall * 3.0
            - wide_penalty
            - small_penalty
    )

    return score


def search_rules_train_only(
        train: pd.DataFrame,
        valid: pd.DataFrame,
        features: List[str],
        target_col: str,
        max_depth: int,
        beam_width: int,
        top_k: int,
        min_train_count: int,
        min_valid_count: int,
        min_train_lift: float,
        beam_min_train_lift: float = 1.00,
        max_train_coverage: float = 0.50,
):
    """
    TRAIN에서만 룰을 선택한다.

    개선 포인트:
    - beam 확장용 기준과 최종 채택 기준을 분리한다.
    - beam_min_train_lift는 느슨하게 둔다.
    - min_train_lift는 최종 룰 채택에만 사용한다.
    - 이렇게 해야 단일 조건은 약하지만 조합하면 강한 룰을 찾을 수 있다.
    """

    y_train = train[target_col].astype(int).values
    y_valid = valid[target_col].astype(int).values

    quantiles = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ]

    atoms = make_atoms(train, features, quantiles)

    print(f"[INFO] atoms generated: {len(atoms)}")

    beam = [tuple()]
    all_rules = []
    final_rules = []
    seen = set()

    for depth in range(1, max_depth + 1):
        print(f"[INFO] searching depth={depth}")

        candidates_for_beam = []
        candidates_for_final = []

        for base_atoms in beam:
            used_features = set(a.feature for a in base_atoms)

            for atom in atoms:
                if atom.feature in used_features:
                    continue

                new_atoms = tuple(list(base_atoms) + [atom])

                # feature 순서 고정해서 순서만 다른 중복 룰 방지
                feature_order = [a.feature for a in new_atoms]
                if feature_order != sorted(feature_order):
                    continue

                rule_key = tuple(
                    (a.feature, a.op, round(a.threshold, 10))
                    for a in new_atoms
                )

                if rule_key in seen:
                    continue

                seen.add(rule_key)

                train_mask = apply_rule(train, new_atoms)
                train_m = precision_recall_lift(y_train, train_mask)

                # 1) beam 확장용 점수: 느슨한 lift 기준
                beam_score = train_rule_score(
                    train_m,
                    min_train_count=min_train_count,
                    min_train_lift=beam_min_train_lift,
                    max_train_coverage=max_train_coverage,
                )

                if beam_score <= -1e17:
                    continue

                valid_mask = apply_rule(valid, new_atoms)
                valid_m = precision_recall_lift(y_valid, valid_mask)

                rule_for_beam = Rule(
                    atoms=new_atoms,
                    train_metrics=train_m,
                    valid_metrics=valid_m,
                    train_score=beam_score,
                )

                candidates_for_beam.append(rule_for_beam)

                # 2) 최종 채택용 점수: 엄격한 lift 기준
                final_score = train_rule_score(
                    train_m,
                    min_train_count=min_train_count,
                    min_train_lift=min_train_lift,
                    max_train_coverage=max_train_coverage,
                )

                if final_score > -1e17:
                    rule_for_final = Rule(
                        atoms=new_atoms,
                        train_metrics=train_m,
                        valid_metrics=valid_m,
                        train_score=final_score,
                    )
                    candidates_for_final.append(rule_for_final)

        if not candidates_for_beam:
            print(f"[WARN] no beam candidates at depth={depth}")
            break

        candidates_for_beam = sorted(
            candidates_for_beam,
            key=lambda r: r.train_score,
            reverse=True,
        )

        candidates_for_final = sorted(
            candidates_for_final,
            key=lambda r: r.train_score,
            reverse=True,
        )

        all_rules.extend(candidates_for_beam[:top_k])
        final_rules.extend(candidates_for_final[:top_k])

        beam = [r.atoms for r in candidates_for_beam[:beam_width]]

        print(
            f"[INFO] depth={depth}, "
            f"beam_candidates={len(candidates_for_beam)}, "
            f"final_candidates={len(candidates_for_final)}, "
            f"kept_beam={len(beam)}"
        )

    # 중복 제거
    unique_raw = {}
    for r in all_rules:
        unique_raw[r.name()] = r

    unique_final = {}
    for r in final_rules:
        unique_final[r.name()] = r

    raw_rules = list(unique_raw.values())
    final_rules = list(unique_final.values())

    raw_rules = sorted(
        raw_rules,
        key=lambda r: r.train_score,
        reverse=True,
    )

    final_rules = sorted(
        final_rules,
        key=lambda r: r.train_score,
        reverse=True,
    )

    # valid_count는 최종 평가 안정성용 필터
    filtered_final = []
    for r in final_rules:
        if r.valid_metrics["count"] >= min_valid_count:
            filtered_final.append(r)

    return filtered_final[:top_k], raw_rules[:top_k]


def rules_to_df(rules: List[Rule]):
    rows = []

    for rank, r in enumerate(rules, start=1):
        row = {
            "rank_by_train": rank,
            "rule": r.name(),
            "features": ",".join(r.features()),
            "n_features": len(r.features()),
            "train_score": r.train_score,
        }

        for prefix, m in [
            ("train", r.train_metrics),
            ("valid", r.valid_metrics),
        ]:
            row[f"{prefix}_count"] = m["count"]
            row[f"{prefix}_target_count"] = m["target_count"]
            row[f"{prefix}_precision"] = m["precision"]
            row[f"{prefix}_recall"] = m["recall"]
            row[f"{prefix}_lift"] = m["lift"]
            row[f"{prefix}_coverage"] = m["coverage"]
            row[f"{prefix}_base_rate"] = m["base_rate"]

        row["precision_gap_valid_minus_train"] = (
            row["valid_precision"] - row["train_precision"]
            if np.isfinite(row["valid_precision"]) and np.isfinite(row["train_precision"])
            else np.nan
        )

        row["lift_gap_valid_minus_train"] = (
            row["valid_lift"] - row["train_lift"]
            if np.isfinite(row["valid_lift"]) and np.isfinite(row["train_lift"])
            else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)


def feature_usage_report(
        rules_df: pd.DataFrame,
        features: List[str],
        top_n: int,
):
    top = rules_df.head(top_n).copy()
    rows = []

    for f in features:
        used = top["features"].fillna("").apply(
            lambda s: f in [x.strip() for x in s.split(",") if x.strip()]
        )

        used_df = top[used]

        row = {
            "feature": f,
            f"top{top_n}_usage_count": int(used.sum()),
            f"top{top_n}_usage_rate": float(used.mean()) if len(top) else 0.0,
        }

        if len(used_df):
            row["avg_train_precision_when_used"] = used_df["train_precision"].mean()
            row["avg_valid_precision_when_used"] = used_df["valid_precision"].mean()
            row["avg_train_lift_when_used"] = used_df["train_lift"].mean()
            row["avg_valid_lift_when_used"] = used_df["valid_lift"].mean()
            row["best_train_rank_when_used"] = used_df["rank_by_train"].min()
            row["avg_precision_gap_when_used"] = used_df[
                "precision_gap_valid_minus_train"
            ].mean()
        else:
            row["avg_train_precision_when_used"] = np.nan
            row["avg_valid_precision_when_used"] = np.nan
            row["avg_train_lift_when_used"] = np.nan
            row["avg_valid_lift_when_used"] = np.nan
            row["best_train_rank_when_used"] = np.nan
            row["avg_precision_gap_when_used"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def leave_one_feature_out_train_first(
        train,
        valid,
        features,
        target_col,
        baseline_best,
        args,
):
    rows = []

    base_train_score = baseline_best.train_score
    base_train_precision = baseline_best.train_metrics["precision"]
    base_valid_precision = baseline_best.valid_metrics["precision"]
    base_train_lift = baseline_best.train_metrics["lift"]
    base_valid_lift = baseline_best.valid_metrics["lift"]

    for f in features:
        print(f"[INFO] leave-one-out: remove {f}")

        sub_features = [x for x in features if x != f]

        filtered_rules, raw_rules = search_rules_train_only(
            train=train,
            valid=valid,
            features=sub_features,
            target_col=target_col,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            top_k=args.loo_top_k,
            min_train_count=args.min_train_count,
            min_valid_count=args.min_valid_count,
            min_train_lift=args.min_train_lift,
            beam_min_train_lift=args.beam_min_train_lift,
            max_train_coverage=args.max_train_coverage,
        )

        rules = filtered_rules if len(filtered_rules) else raw_rules

        if len(rules) == 0:
            rows.append({
                "feature": f,
                "best_rule_without": "",
                "train_score_without": np.nan,
                "train_score_drop": np.nan,
                "train_precision_without": np.nan,
                "valid_precision_without": np.nan,
                "train_precision_drop": np.nan,
                "valid_precision_drop": np.nan,
                "train_lift_without": np.nan,
                "valid_lift_without": np.nan,
                "train_lift_drop": np.nan,
                "valid_lift_drop": np.nan,
            })
            continue

        best = rules[0]

        rows.append({
            "feature": f,
            "best_rule_without": best.name(),
            "train_score_without": best.train_score,
            "train_score_drop": base_train_score - best.train_score,
            "train_precision_without": best.train_metrics["precision"],
            "valid_precision_without": best.valid_metrics["precision"],
            "train_precision_drop": base_train_precision - best.train_metrics["precision"],
            "valid_precision_drop": base_valid_precision - best.valid_metrics["precision"],
            "train_lift_without": best.train_metrics["lift"],
            "valid_lift_without": best.valid_metrics["lift"],
            "train_lift_drop": base_train_lift - best.train_metrics["lift"],
            "valid_lift_drop": base_valid_lift - best.valid_metrics["lift"],
        })

    return pd.DataFrame(rows).sort_values(
        "train_score_drop",
        ascending=False,
    )


def final_judgement_train_first(
        usage_df,
        loo_df,
        top_n,
):
    df = usage_df.merge(loo_df, on="feature", how="left")

    usage_col = f"top{top_n}_usage_count"

    labels = []

    for _, r in df.iterrows():
        f = r["feature"]

        usage_count = r.get(usage_col, 0)
        train_score_drop = r.get("train_score_drop", 0)
        train_precision_drop = r.get("train_precision_drop", 0)
        avg_train_lift = r.get("avg_train_lift_when_used", np.nan)
        avg_valid_lift = r.get("avg_valid_lift_when_used", np.nan)

        used_often = usage_count >= max(3, int(top_n * 0.15))

        train_drop_meaningful = (
                                        np.isfinite(train_score_drop) and train_score_drop > 2.0
                                ) or (
                                        np.isfinite(train_precision_drop) and train_precision_drop > 0.01
                                )

        train_ok = (
                np.isfinite(avg_train_lift) and avg_train_lift >= 1.10
        )

        valid_ok = (
                np.isfinite(avg_valid_lift) and avg_valid_lift >= 1.05
        )

        if used_often and train_drop_meaningful and train_ok and valid_ok:
            label = "CORE"
            reason = "TRAIN 기준 상위 룰에서 자주 사용되고, 제거 시 TRAIN 성능이 떨어지며 VALID도 유지됨"
        elif used_often and train_ok and valid_ok:
            label = "USEFUL"
            reason = "TRAIN 기준 상위 룰에서 자주 사용되고 VALID에서도 유지됨"
        elif train_drop_meaningful and valid_ok:
            label = "USEFUL_COMBO"
            reason = "사용 빈도는 낮지만 제거 시 TRAIN 성능 저하가 있고 VALID도 유지됨"
        elif used_often and not valid_ok:
            label = "TRAIN_ONLY_RISK"
            reason = "TRAIN에서는 자주 쓰이나 VALID 성능이 약해 과최적화 가능성 있음"
        elif usage_count == 0 and not train_drop_meaningful:
            label = "WEAK_REMOVE_CANDIDATE"
            reason = "TRAIN 기준 상위 룰에서 거의 사용되지 않고 제거 영향도 작음"
        else:
            label = "CHECK_MANUALLY"
            reason = "TRAIN/VALID 기여도가 애매함"

        labels.append({
            "feature": f,
            "judgement": label,
            "reason": reason,
        })

    label_df = pd.DataFrame(labels)
    out = df.merge(label_df, on="feature", how="left")

    order_map = {
        "CORE": 1,
        "USEFUL": 2,
        "USEFUL_COMBO": 3,
        "TRAIN_ONLY_RISK": 4,
        "CHECK_MANUALLY": 5,
        "WEAK_REMOVE_CANDIDATE": 6,
    }

    out["_order"] = out["judgement"].map(order_map).fillna(99)
    out = out.sort_values(
        ["_order", "train_score_drop"],
        ascending=[True, False],
    )
    out = out.drop(columns=["_order"])

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="train_first_rule_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--loo-top-k", type=int, default=80)

    parser.add_argument("--min-train-count", type=int, default=200)
    parser.add_argument("--min-valid-count", type=int, default=100)

    # 최종 룰 채택 기준
    parser.add_argument("--min-train-lift", type=float, default=1.20)

    # beam 확장 기준
    # 단일 조건은 약하지만 조합하면 강한 룰을 찾기 위해 느슨하게 둔다.
    parser.add_argument("--beam-min-train-lift", type=float, default=1.00)

    # 너무 넓은 룰 패널티 기준
    parser.add_argument("--max-train-coverage", type=float, default=0.50)

    parser.add_argument("--top-n-usage", type=int, default=50)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    if args.target not in df.columns:
        raise ValueError(f"target column not found: {args.target}")

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df[df[args.target].isin([0, 1])].copy()
    df[args.target] = df[args.target].astype(int)

    features = [f for f in DEFAULT_FEATURES if f in df.columns]

    if not features:
        raise ValueError("No usable features found.")

    date_col = args.date_col
    if date_col is None:
        date_col = find_date_col(df)

    print("=" * 80)
    print("[INFO] rows:", len(df))
    print("[INFO] target:", args.target)
    print("[INFO] target_rate:", df[args.target].mean())
    print("[INFO] date_col:", date_col)
    print("[INFO] features:", features)
    print("=" * 80)

    train, valid = split_train_valid(
        df=df,
        date_col=date_col,
        valid_ratio=args.valid_ratio,
    )

    print("[INFO] train rows:", len(train), "target_rate:", train[args.target].mean())
    print("[INFO] valid rows:", len(valid), "target_rate:", valid[args.target].mean())

    print("[INFO] searching rules using TRAIN only...")

    filtered_rules, raw_rules = search_rules_train_only(
        train=train,
        valid=valid,
        features=features,
        target_col=args.target,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        top_k=args.top_k,
        min_train_count=args.min_train_count,
        min_valid_count=args.min_valid_count,
        min_train_lift=args.min_train_lift,
        beam_min_train_lift=args.beam_min_train_lift,
        max_train_coverage=args.max_train_coverage,
    )

    raw_rules_df = rules_to_df(raw_rules)
    raw_rules_df.to_csv(
        os.path.join(args.out, "01_rules_ranked_by_train_raw_beam.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    filtered_rules_df = rules_to_df(filtered_rules)
    filtered_rules_df.to_csv(
        os.path.join(args.out, "02_rules_ranked_by_train_final.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if len(filtered_rules) == 0:
        print("[WARN] No final rules passed filters.")
        print("[WARN] Check 01_rules_ranked_by_train_raw_beam.csv")
        print("[HINT] Try lowering --min-train-lift or --min-valid-count")
        return

    best = filtered_rules[0]

    print("=" * 80)
    print("[BEST RULE SELECTED BY TRAIN]")
    print(best.name())
    print("[TRAIN]", best.train_metrics)
    print("[VALID]", best.valid_metrics)
    print("=" * 80)

    usage_df = feature_usage_report(
        filtered_rules_df,
        features,
        top_n=args.top_n_usage,
    )

    usage_df.to_csv(
        os.path.join(args.out, "03_feature_usage_train_selected_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("[INFO] running leave-one-feature-out using TRAIN selection...")

    loo_df = leave_one_feature_out_train_first(
        train=train,
        valid=valid,
        features=features,
        target_col=args.target,
        baseline_best=best,
        args=args,
    )

    loo_df.to_csv(
        os.path.join(args.out, "04_leave_one_feature_out_train_first.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    final_df = final_judgement_train_first(
        usage_df=usage_df,
        loo_df=loo_df,
        top_n=args.top_n_usage,
    )

    final_df.to_csv(
        os.path.join(args.out, "05_final_feature_judgement_train_first.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[FINAL FEATURE JUDGEMENT]")
    show_cols = [
        "feature",
        "judgement",
        "reason",
        f"top{args.top_n_usage}_usage_count",
        "avg_train_precision_when_used",
        "avg_valid_precision_when_used",
        "avg_train_lift_when_used",
        "avg_valid_lift_when_used",
        "train_score_drop",
        "train_precision_drop",
        "valid_precision_drop",
    ]
    show_cols = [c for c in show_cols if c in final_df.columns]
    print(final_df[show_cols].to_string(index=False))

    print("\n[TOP RULES SELECTED BY TRAIN]")
    show_rule_cols = [
        "rank_by_train",
        "rule",
        "train_count",
        "train_target_count",
        "train_precision",
        "train_lift",
        "valid_count",
        "valid_target_count",
        "valid_precision",
        "valid_lift",
        "precision_gap_valid_minus_train",
    ]
    show_rule_cols = [c for c in show_rule_cols if c in filtered_rules_df.columns]
    print(filtered_rules_df[show_rule_cols].head(30).to_string(index=False))

    print("=" * 80)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()