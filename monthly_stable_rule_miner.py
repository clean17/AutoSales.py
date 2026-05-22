import argparse
import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


TARGET_COL = "target_before_stop_7"


DEFAULT_FEATURES = [
    "gap_pct",
    "lower_wick_ratio",
    "today_pct",

    "vol5",
    "dist_to_ma5",
    "max_drop_7d",

    "body_ratio",
    "recent_runup",
    "intraday_return",

    "rebound_from_7d_low",
    "BB_perc",

    "tr_val_rank_20d",
    "today_tr_val_eok",
    # "vol_ratio_5_15",

    "pct_vs_lastweek",
    "upper_wick_ratio",
]


@dataclass(frozen=True)
class Atom:
    feature: str
    op: str
    threshold: float

    def name(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.6g}"


@dataclass
class Rule:
    atoms: Tuple[Atom, ...]
    train_metrics: Dict
    train_monthly_summary: Dict
    valid_metrics: Dict
    valid_monthly_summary: Dict
    score: float

    def name(self) -> str:
        return " AND ".join([a.name() for a in self.atoms])

    def features(self) -> List[str]:
        return sorted(set(a.feature for a in self.atoms))


# ============================================================
# 기본 유틸
# ============================================================

def find_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "today",
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


def prepare_df(df: pd.DataFrame, target_col: str, date_col: str) -> pd.DataFrame:
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    if date_col not in df.columns:
        raise ValueError(f"date column not found: {date_col}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    df[target_col] = df[target_col].astype(int)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    df["month"] = df[date_col].dt.to_period("M").astype(str)
    df["quarter"] = df[date_col].dt.to_period("Q").astype(str)
    df["year"] = df[date_col].dt.year.astype(str)

    return df


def split_train_valid_by_date(
        df: pd.DataFrame,
        valid_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - valid_ratio))

    train = df.iloc[:split_idx].copy().reset_index(drop=True)
    valid = df.iloc[split_idx:].copy().reset_index(drop=True)

    return train, valid

def calc_metrics(y: np.ndarray, mask: np.ndarray) -> Dict:
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask).astype(bool)

    total_count = len(y)
    total_target = int(y.sum())
    base_rate = total_target / total_count if total_count > 0 else np.nan

    selected_count = int(mask.sum())

    if selected_count == 0:
        return {
            "total_count": total_count,
            "total_target": total_target,
            "base_rate": base_rate,
            "selected_count": 0,
            "selected_target": 0,
            "precision": np.nan,
            "lift": np.nan,
            "recall": 0.0,
            "coverage": 0.0,
        }

    selected_target = int(y[mask].sum())
    precision = selected_target / selected_count
    lift = precision / base_rate if base_rate and base_rate > 0 else np.nan
    recall = selected_target / total_target if total_target > 0 else 0.0
    coverage = selected_count / total_count if total_count > 0 else 0.0

    return {
        "total_count": total_count,
        "total_target": total_target,
        "base_rate": base_rate,
        "selected_count": selected_count,
        "selected_target": selected_target,
        "precision": precision,
        "lift": lift,
        "recall": recall,
        "coverage": coverage,
    }


def apply_atom(df: pd.DataFrame, atom: Atom) -> np.ndarray:
    x = pd.to_numeric(df[atom.feature], errors="coerce")

    if atom.op == "<=":
        return (x <= atom.threshold).fillna(False).values

    if atom.op == ">=":
        return (x >= atom.threshold).fillna(False).values

    raise ValueError(atom.op)


def apply_rule(df: pd.DataFrame, atoms: Tuple[Atom, ...]) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)

    for atom in atoms:
        mask &= apply_atom(df, atom)

    return mask


# ============================================================
# 상관 피쳐 처리
# ============================================================

def build_corr_pairs(
        df: pd.DataFrame,
        features: List[str],
        corr_threshold: float,
) -> Tuple[pd.DataFrame, Set[frozenset]]:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman")

    rows = []
    pairs = set()

    for i, a in enumerate(features):
        for b in features[i + 1:]:
            c = corr.loc[a, b]

            if pd.isna(c):
                continue

            if abs(c) >= corr_threshold:
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "spearman_corr": c,
                    "abs_corr": abs(c),
                })
                pairs.add(frozenset([a, b]))

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values("abs_corr", ascending=False)

    return out, pairs


def has_correlated_pair(
        used_features: Set[str],
        new_feature: str,
        corr_pairs: Set[frozenset],
) -> bool:
    for f in used_features:
        if frozenset([f, new_feature]) in corr_pairs:
            return True

    return False


# ============================================================
# Atom 후보 생성
# ============================================================

def make_atoms(
        train: pd.DataFrame,
        features: List[str],
        quantiles: List[float],
        extra_thresholds: Dict[str, List[Tuple[str, float]]],
) -> List[Atom]:
    atoms = []

    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        x = x[np.isfinite(x)]

        if len(x) < 100:
            continue

        if x.nunique() <= 3:
            continue

        qs = np.nanquantile(x, quantiles)
        qs = sorted(set(float(q) for q in qs if np.isfinite(q)))

        for q in qs:
            atoms.append(Atom(f, "<=", q))
            atoms.append(Atom(f, ">=", q))

        # 수동 threshold 추가
        for op, th in extra_thresholds.get(f, []):
            atoms.append(Atom(f, op, float(th)))

    # 중복 제거
    unique = {}
    for a in atoms:
        unique[(a.feature, a.op, round(a.threshold, 10))] = a

    atoms = list(unique.values())

    return atoms


def default_extra_thresholds() -> Dict[str, List[Tuple[str, float]]]:
    """
    기존 분석에서 자주 등장한 기준값을 후보 threshold에 강제로 포함.
    단, 선택은 TRAIN 월별 안정성 기준으로만 함.
    """

    return {
        "gap_pct": [
            ("<=", -1.0),
            ("<=", -0.5),
            ("<=", 0.0),
            ("<=", 0.5),
            ("<=", 1.0),
        ],
        "lower_wick_ratio": [
            ("<=", 0.004),
            ("<=", 0.01),
            ("<=", 0.02),
            ("<=", 0.05),
            ("<=", 0.10),
        ],
        "today_pct": [
            (">=", 3.5),
            (">=", 4.0),
            (">=", 4.5),
            (">=", 4.88),
            (">=", 5.0),
            (">=", 5.5),
            (">=", 6.0),
        ],
        "vol5": [
            (">=", 3.5),
            (">=", 4.0),
            (">=", 4.378),
            (">=", 5.0),
        ],
        "dist_to_ma5": [
            ("<=", -0.0182),
            ("<=", -0.3),
            ("<=", -0.5),
            ("<=", -1.0),
            ("<=", -2.0),
        ],
        "max_drop_7d": [
            ("<=", -4.0),
            ("<=", -5.0),
            ("<=", -5.95),
            ("<=", -7.0),
            ("<=", -10.0),
        ],
        "body_ratio": [
            (">=", 0.6),
            (">=", 0.7),
            (">=", 0.8),
            (">=", 0.839),
            (">=", 0.9),
        ],
        "BB_perc": [
            ("<=", 0.10),
            ("<=", 0.148),
            ("<=", 0.1765),
            ("<=", 0.20),
            ("<=", 0.30),
        ],
        "intraday_return": [
            (">=", 4.0),
            (">=", 4.5),
            (">=", 5.0),
            (">=", 5.185),
            (">=", 6.0),
        ],
        "rebound_from_7d_low": [
            (">=", 3.0),
            (">=", 4.0),
            (">=", 5.0),
            (">=", 5.6268),
            (">=", 7.0),
        ],
        "upper_wick_ratio": [
            ("<=", 0.20),
            ("<=", 0.2995),
            ("<=", 0.31),
            ("<=", 0.40),
        ],
        "today_tr_val_eok": [
            (">=", 3.0),
            (">=", 5.0),
            ("<=", 40.0),
            ("<=", 52.8),
            ("<=", 63.0),
        ],
        "pct_vs_lastweek": [
            ("<=", -3.0),
            ("<=", -1.0),
            ("<=", 0.0),
            ("<=", 0.21),
            ("<=", 1.0),
        ],
        "up_down_tr_value_ratio_5d_log": [
            ("<=", 0.3),
            ("<=", 0.43),
            ("<=", 0.5),
        ],
    }


# ============================================================
# 월별 안정성 평가
# ============================================================

def monthly_eval(
        df: pd.DataFrame,
        target_col: str,
        atoms: Tuple[Atom, ...],
        min_month_count: int,
) -> Tuple[pd.DataFrame, Dict]:
    """
    월별 룰 성능 평가.

    중요:
    df가 train/valid로 잘린 뒤 원본 index를 유지하면,
    groupby().groups가 원본 index를 반환해서 numpy array indexing 에러가 난다.
    따라서 반드시 reset_index(drop=True)를 먼저 한다.
    """

    df = df.reset_index(drop=True).copy()

    y = df[target_col].values
    mask_all = apply_rule(df, atoms)

    rows = []

    for month, g in df.groupby("month", sort=True):
        idx = g.index.to_numpy()

        y_m = y[idx]
        mask_m = mask_all[idx]

        m = calc_metrics(y_m, mask_m)
        m["month"] = month
        rows.append(m)

    monthly_df = pd.DataFrame(rows)

    usable = monthly_df[monthly_df["selected_count"] >= min_month_count].copy()

    if len(usable) == 0:
        summary = {
            "n_months": monthly_df["month"].nunique() if len(monthly_df) else 0,
            "n_usable_months": 0,
            "mean_month_precision": np.nan,
            "median_month_precision": np.nan,
            "min_month_precision": np.nan,
            "std_month_precision": np.nan,
            "mean_month_lift": np.nan,
            "median_month_lift": np.nan,
            "min_month_lift": np.nan,
            "mean_month_selected_count": np.nan,
            "total_month_selected_count": int(monthly_df["selected_count"].sum()) if len(monthly_df) else 0,
            "pass_month_rate": 0.0,
            "bad_month_count": len(monthly_df),
        }
        return monthly_df, summary

    pass_mask = (
            (usable["precision"] >= 0.60) &
            (usable["lift"] >= 1.25)
    )

    summary = {
        "n_months": monthly_df["month"].nunique(),
        "n_usable_months": len(usable),
        "mean_month_precision": usable["precision"].mean(),
        "median_month_precision": usable["precision"].median(),
        "min_month_precision": usable["precision"].min(),
        "std_month_precision": usable["precision"].std(),
        "mean_month_lift": usable["lift"].mean(),
        "median_month_lift": usable["lift"].median(),
        "min_month_lift": usable["lift"].min(),
        "mean_month_selected_count": usable["selected_count"].mean(),
        "total_month_selected_count": int(monthly_df["selected_count"].sum()),
        "pass_month_rate": pass_mask.mean(),
        "bad_month_count": int((~pass_mask).sum()),
    }

    return monthly_df, summary

def train_stability_score(
        overall_metrics: Dict,
        monthly_summary: Dict,
        min_train_count: int,
        min_train_lift: float,
        min_usable_months: int,
) -> float:
    count = overall_metrics["selected_count"]
    precision = overall_metrics["precision"]
    lift = overall_metrics["lift"]

    if count < min_train_count:
        return -1e18

    if not np.isfinite(precision) or not np.isfinite(lift):
        return -1e18

    if lift < min_train_lift:
        return -1e18

    n_usable = monthly_summary["n_usable_months"]

    if n_usable < min_usable_months:
        return -1e18

    mean_p = monthly_summary["mean_month_precision"]
    min_p = monthly_summary["min_month_precision"]
    std_p = monthly_summary["std_month_precision"]
    mean_lift = monthly_summary["mean_month_lift"]
    min_lift = monthly_summary["min_month_lift"]
    pass_rate = monthly_summary["pass_month_rate"]
    bad_count = monthly_summary["bad_month_count"]
    total_selected = monthly_summary["total_month_selected_count"]

    if not np.isfinite(mean_p) or not np.isfinite(min_p):
        return -1e18

    if not np.isfinite(std_p):
        std_p = 0.0

    score = (
            precision * 50
            + lift * 10
            + mean_p * 80
            + min_p * 80
            + mean_lift * 10
            + min_lift * 10
            + pass_rate * 50
            + math.log1p(total_selected) * 2
            - std_p * 70
            - bad_count * 2.5
    )

    return score


def eval_rule(
        df: pd.DataFrame,
        target_col: str,
        atoms: Tuple[Atom, ...],
        min_month_count: int,
) -> Tuple[Dict, Dict, pd.DataFrame]:
    mask = apply_rule(df, atoms)
    overall = calc_metrics(df[target_col].values, mask)

    monthly_df, monthly_summary = monthly_eval(
        df=df,
        target_col=target_col,
        atoms=atoms,
        min_month_count=min_month_count,
    )

    return overall, monthly_summary, monthly_df


# ============================================================
# 룰 탐색
# ============================================================

def search_rules(
        train: pd.DataFrame,
        valid: pd.DataFrame,
        features: List[str],
        target_col: str,
        atoms: List[Atom],
        corr_pairs: Set[frozenset],
        max_depth: int,
        beam_width: int,
        top_k: int,
        min_train_count: int,
        min_train_lift: float,
        min_month_count: int,
        min_usable_months: int,
        block_correlated: bool,
) -> List[Rule]:
    beam = [tuple()]
    all_rules: Dict[str, Rule] = {}

    for depth in range(1, max_depth + 1):
        print(f"[INFO] searching depth={depth}")

        candidates = []

        for base_atoms in beam:
            used_features = set(a.feature for a in base_atoms)

            for atom in atoms:
                if atom.feature in used_features:
                    continue

                if block_correlated and has_correlated_pair(
                        used_features,
                        atom.feature,
                        corr_pairs,
                ):
                    continue

                new_atoms = tuple(list(base_atoms) + [atom])

                # feature 순서 고정으로 중복 방지
                feature_order = [a.feature for a in new_atoms]
                if feature_order != sorted(feature_order):
                    continue

                rule_key = " AND ".join(
                    f"{a.feature}|{a.op}|{round(a.threshold, 10)}"
                    for a in new_atoms
                )

                if rule_key in all_rules:
                    continue

                train_overall, train_monthly_summary, _ = eval_rule(
                    df=train,
                    target_col=target_col,
                    atoms=new_atoms,
                    min_month_count=min_month_count,
                )

                score = train_stability_score(
                    overall_metrics=train_overall,
                    monthly_summary=train_monthly_summary,
                    min_train_count=min_train_count,
                    min_train_lift=min_train_lift,
                    min_usable_months=min_usable_months,
                )

                if score <= -1e17:
                    continue

                valid_overall, valid_monthly_summary, _ = eval_rule(
                    df=valid,
                    target_col=target_col,
                    atoms=new_atoms,
                    min_month_count=min_month_count,
                )

                rule = Rule(
                    atoms=new_atoms,
                    train_metrics=train_overall,
                    train_monthly_summary=train_monthly_summary,
                    valid_metrics=valid_overall,
                    valid_monthly_summary=valid_monthly_summary,
                    score=score,
                )

                candidates.append(rule)
                all_rules[rule_key] = rule

        if not candidates:
            print(f"[WARN] no candidates at depth={depth}")
            break

        candidates = sorted(
            candidates,
            key=lambda r: r.score,
            reverse=True,
        )

        beam = [r.atoms for r in candidates[:beam_width]]

        print(
            f"[INFO] depth={depth}, candidates={len(candidates)}, "
            f"kept_beam={len(beam)}"
        )

    rules = list(all_rules.values())
    rules = sorted(rules, key=lambda r: r.score, reverse=True)

    return rules[:top_k]


def rules_to_df(rules: List[Rule]) -> pd.DataFrame:
    rows = []

    for rank, r in enumerate(rules, start=1):
        row = {
            "rank": rank,
            "rule": r.name(),
            "features": ",".join(r.features()),
            "n_features": len(r.features()),
            "score": r.score,

            "train_count": r.train_metrics["selected_count"],
            "train_target": r.train_metrics["selected_target"],
            "train_precision": r.train_metrics["precision"],
            "train_lift": r.train_metrics["lift"],
            "train_recall": r.train_metrics["recall"],
            "train_coverage": r.train_metrics["coverage"],
            "train_base_rate": r.train_metrics["base_rate"],

            "valid_count": r.valid_metrics["selected_count"],
            "valid_target": r.valid_metrics["selected_target"],
            "valid_precision": r.valid_metrics["precision"],
            "valid_lift": r.valid_metrics["lift"],
            "valid_recall": r.valid_metrics["recall"],
            "valid_coverage": r.valid_metrics["coverage"],
            "valid_base_rate": r.valid_metrics["base_rate"],

            "precision_gap_valid_minus_train": (
                r.valid_metrics["precision"] - r.train_metrics["precision"]
                if np.isfinite(r.valid_metrics["precision"]) and np.isfinite(r.train_metrics["precision"])
                else np.nan
            ),
        }

        for prefix, s in [
            ("train_month", r.train_monthly_summary),
            ("valid_month", r.valid_monthly_summary),
        ]:
            for k, v in s.items():
                row[f"{prefix}_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)


def collect_monthly_details(
        rules: List[Rule],
        train: pd.DataFrame,
        valid: pd.DataFrame,
        target_col: str,
        min_month_count: int,
) -> pd.DataFrame:
    rows = []

    for rank, r in enumerate(rules, start=1):
        for dataset_name, df in [
            ("TRAIN", train),
            ("VALID", valid),
        ]:
            _, _, monthly_df = eval_rule(
                df=df,
                target_col=target_col,
                atoms=r.atoms,
                min_month_count=min_month_count,
            )

            monthly_df = monthly_df.copy()
            monthly_df["rank"] = rank
            monthly_df["rule"] = r.name()
            monthly_df["dataset"] = dataset_name

            rows.append(monthly_df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


# ============================================================
# Walk-forward: 고정 룰 평가
# ============================================================

def walk_forward_eval_fixed_rules(
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        rules: List[Rule],
        n_splits: int,
        min_train_months: int,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    months = sorted(df["month"].unique())

    if len(months) <= min_train_months:
        return pd.DataFrame()

    start_idx = min_train_months

    if n_splits > 0:
        start_idx = max(min_train_months, len(months) - n_splits)

    rows = []

    for i in range(start_idx, len(months)):
        train_months = months[:i]
        valid_month = months[i]

        train = df[df["month"].isin(train_months)].copy().reset_index(drop=True)
        valid = df[df["month"] == valid_month].copy().reset_index(drop=True)

        for rank, r in enumerate(rules, start=1):
            train_mask = apply_rule(train, r.atoms)
            valid_mask = apply_rule(valid, r.atoms)

            train_m = calc_metrics(train[target_col].values, train_mask)
            valid_m = calc_metrics(valid[target_col].values, valid_mask)

            rows.append({
                "rank": rank,
                "rule": r.name(),
                "valid_month": valid_month,

                "train_count": train_m["selected_count"],
                "train_precision": train_m["precision"],
                "train_lift": train_m["lift"],
                "train_base_rate": train_m["base_rate"],

                "valid_count": valid_m["selected_count"],
                "valid_target": valid_m["selected_target"],
                "valid_precision": valid_m["precision"],
                "valid_lift": valid_m["lift"],
                "valid_base_rate": valid_m["base_rate"],
            })

    return pd.DataFrame(rows)


def summarize_wf(wf_df: pd.DataFrame, min_count: int, min_precision: float, min_lift: float) -> pd.DataFrame:
    rows = []

    if len(wf_df) == 0:
        return pd.DataFrame()

    for (rank, rule), g in wf_df.groupby(["rank", "rule"]):
        usable = g[g["valid_count"] >= min_count].copy()

        if len(usable) == 0:
            rows.append({
                "rank": rank,
                "rule": rule,
                "n_splits": len(g),
                "n_usable_splits": 0,
                "mean_valid_precision": np.nan,
                "median_valid_precision": np.nan,
                "min_valid_precision": np.nan,
                "std_valid_precision": np.nan,
                "mean_valid_lift": np.nan,
                "min_valid_lift": np.nan,
                "total_valid_count": int(g["valid_count"].sum()),
                "pass_split_rate": 0.0,
                "bad_split_count": len(g),
                "wf_score": -999,
            })
            continue

        pass_mask = (
                (usable["valid_precision"] >= min_precision) &
                (usable["valid_lift"] >= min_lift)
        )

        std_p = usable["valid_precision"].std()
        if not np.isfinite(std_p):
            std_p = 0.0

        wf_score = (
                usable["valid_precision"].mean() * 100
                + usable["valid_precision"].min() * 80
                + usable["valid_lift"].mean() * 10
                + pass_mask.mean() * 40
                + math.log1p(usable["valid_count"].sum()) * 2
                - std_p * 50
                - int((~pass_mask).sum()) * 3
        )

        rows.append({
            "rank": rank,
            "rule": rule,
            "n_splits": len(g),
            "n_usable_splits": len(usable),
            "mean_valid_precision": usable["valid_precision"].mean(),
            "median_valid_precision": usable["valid_precision"].median(),
            "min_valid_precision": usable["valid_precision"].min(),
            "std_valid_precision": std_p,
            "mean_valid_lift": usable["valid_lift"].mean(),
            "min_valid_lift": usable["valid_lift"].min(),
            "total_valid_count": int(g["valid_count"].sum()),
            "pass_split_rate": pass_mask.mean(),
            "bad_split_count": int((~pass_mask).sum()),
            "wf_score": wf_score,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("wf_score", ascending=False)

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="monthly_stable_rule_miner_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=100)

    parser.add_argument("--min-train-count", type=int, default=200)
    parser.add_argument("--min-train-lift", type=float, default=1.10)
    parser.add_argument("--min-month-count", type=int, default=20)
    parser.add_argument("--min-usable-months", type=int, default=8)

    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--allow-correlated-in-rule", action="store_true")

    parser.add_argument("--wf-splits", type=int, default=12)
    parser.add_argument("--wf-min-train-months", type=int, default=6)
    parser.add_argument("--wf-min-count", type=int, default=20)
    parser.add_argument("--wf-min-precision", type=float, default=0.60)
    parser.add_argument("--wf-min-lift", type=float, default=1.25)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    date_col = args.date_col or find_date_col(df)

    if date_col is None:
        raise ValueError("date column not found. Please pass --date-col today")

    df = prepare_df(
        df=df,
        target_col=args.target,
        date_col=date_col,
    )

    features = [f for f in DEFAULT_FEATURES if f in df.columns]

    missing = [f for f in DEFAULT_FEATURES if f not in df.columns]
    if missing:
        print("[WARN] missing features:", missing)

    train, valid = split_train_valid_by_date(
        df=df,
        valid_ratio=args.valid_ratio,
    )

    print("=" * 80)
    print("[INFO] rows:", len(df))
    print("[INFO] train rows:", len(train), "base_rate:", train[args.target].mean())
    print("[INFO] valid rows:", len(valid), "base_rate:", valid[args.target].mean())
    print("[INFO] date_col:", date_col)
    print("[INFO] features:", features)
    print("=" * 80)

    high_corr_df, corr_pairs = build_corr_pairs(
        df=train,
        features=features,
        corr_threshold=args.corr_threshold,
    )

    high_corr_df.to_csv(
        os.path.join(args.out, "00_high_corr_pairs.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if len(high_corr_df):
        print("\n[HIGH CORR PAIRS]")
        print(high_corr_df.to_string(index=False))
    else:
        print("\n[HIGH CORR PAIRS] none")

    quantiles = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ]

    atoms = make_atoms(
        train=train,
        features=features,
        quantiles=quantiles,
        extra_thresholds=default_extra_thresholds(),
    )

    atoms_df = pd.DataFrame([
        {
            "feature": a.feature,
            "op": a.op,
            "threshold": a.threshold,
            "atom": a.name(),
        }
        for a in atoms
    ])

    atoms_df.to_csv(
        os.path.join(args.out, "01_atoms.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("[INFO] atoms:", len(atoms))

    rules = search_rules(
        train=train,
        valid=valid,
        features=features,
        target_col=args.target,
        atoms=atoms,
        corr_pairs=corr_pairs,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        top_k=args.top_k,
        min_train_count=args.min_train_count,
        min_train_lift=args.min_train_lift,
        min_month_count=args.min_month_count,
        min_usable_months=args.min_usable_months,
        block_correlated=not args.allow_correlated_in_rule,
    )

    rules_df = rules_to_df(rules)

    rules_df.to_csv(
        os.path.join(args.out, "02_selected_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    monthly_detail_df = collect_monthly_details(
        rules=rules,
        train=train,
        valid=valid,
        target_col=args.target,
        min_month_count=args.min_month_count,
    )

    monthly_detail_df.to_csv(
        os.path.join(args.out, "03_monthly_details.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    wf_df = walk_forward_eval_fixed_rules(
        df=df,
        date_col=date_col,
        target_col=args.target,
        rules=rules[:30],
        n_splits=args.wf_splits,
        min_train_months=args.wf_min_train_months,
    )

    wf_df.to_csv(
        os.path.join(args.out, "04_walk_forward_eval.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    wf_summary = summarize_wf(
        wf_df=wf_df,
        min_count=args.wf_min_count,
        min_precision=args.wf_min_precision,
        min_lift=args.wf_min_lift,
    )

    wf_summary.to_csv(
        os.path.join(args.out, "05_walk_forward_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[TOP SELECTED RULES]")
    show_cols = [
        "rank",
        "rule",
        "score",

        "train_count",
        "train_precision",
        "train_lift",
        "train_month_n_usable_months",
        "train_month_mean_month_precision",
        "train_month_min_month_precision",
        "train_month_pass_month_rate",

        "valid_count",
        "valid_precision",
        "valid_lift",
        "valid_month_n_usable_months",
        "valid_month_mean_month_precision",
        "valid_month_min_month_precision",
        "valid_month_pass_month_rate",
    ]
    show_cols = [c for c in show_cols if c in rules_df.columns]

    if len(rules_df):
        print(rules_df[show_cols].head(30).to_string(index=False))
    else:
        print("No rules found.")

    print("\n[WALK FORWARD SUMMARY]")
    if len(wf_summary):
        show_cols = [
            "rank",
            "rule",
            "n_splits",
            "n_usable_splits",
            "mean_valid_precision",
            "median_valid_precision",
            "min_valid_precision",
            "mean_valid_lift",
            "min_valid_lift",
            "total_valid_count",
            "pass_split_rate",
            "bad_split_count",
            "wf_score",
        ]
        show_cols = [c for c in show_cols if c in wf_summary.columns]
        print(wf_summary[show_cols].head(30).to_string(index=False))
    else:
        print("No walk-forward summary.")

    print("=" * 80)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()

# 실행 방법
# python monthly_stable_rule_miner.py   --csv csv/low_result_7_desc.csv   --out monthly_stable_rule_miner_out   --date-col today   --max-depth 4   --beam-width 300   --top-k 100   --min-train-count 200   --min-train-lift 1.10   --min-month-count 20   --min-usable-months 8   --wf-splits 12H