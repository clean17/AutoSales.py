import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

"""
"사용자가 지정한 피쳐들 중 valid 평균 60% 룰 포트폴리오에 기여하는 피쳐를 선별"

사용자가 지정한 피쳐 풀로 valid precision 평균 60% 이상 룰 포트폴리오를 만들고 그 룰 묶음에서 유용한 피쳐를 판별

>>>
좋은 점:
실제 valid 60% 목표에 직접 맞음
고정밀 룰에 필요한 피쳐를 좁히기 좋음
중복 룰을 줄이는 옵션이 있음

나쁜 점:
선택된 avg60 룰에 안 들어간 피쳐는 낮게 평가될 수 있음
전체적으로 유용한 피쳐라도 avg60 룰 밖이면 밀림
데이터 split이나 룰 선택 조건에 민감함
"""

TARGET_COL = "target_before_stop_10"


# ============================================================
# Feature pool
# 목적:
# - valid precision 평균 60% 이상 룰 묶음을 만들고
# - 그 룰 묶음에서 유용한 피쳐를 판별
# ============================================================
DEFAULT_FEATURES = [
    "vol5",
    "rebound_from_7d_low",
    "today_pct",
    "price_power_value",
    "dist_to_ma5",
    "intraday_return",
    "tr_value_ratio_5d",
    "max_drop_7d",
    "body_value_power",
    "rebound_vs_prior_drop",
    "upper_wick_ratio",
    "vol15",
    "ATR_pct",
    "dist_to_ma20",
    "BB_perc",
    "gap_pct",
    "room_to_60d_high",
    "ma5_chg_rate",
    "pct_vs_lastweek",
]


# 제외 피쳐. 현재 19개 피쳐셋에서는 강제 제외 없음.
EXCLUDE_FEATURES = []


# 단조 방향으로 보기 어려운 피쳐.
# AUC 방향성과 룰 방향이 충돌해도 바로 감점하지 않고,
# conditional contribution을 더 중요하게 본다.
NON_MONOTONIC_FEATURES = [
    # 전체 방향성이 약해도 특정 구간에서 필터로 의미가 있는 피쳐
    "dist_to_ma5",
    "dist_to_ma20",
    "rebound_vs_prior_drop",

    # legacy 재투입 피쳐: 구간/필터 성격이 강함
    "BB_perc",
    "gap_pct",
    "room_to_60d_high",
    "ma5_chg_rate",
    "pct_vs_lastweek",
]


REGIME_FEATURES = [
    # 현재 19개 피쳐에는 market_* 장세 피쳐가 없음
]


WEAK_HINT_FEATURES = []


ALLOWED_OPS = {
    # 거래량/수급 강도
    "vol5": [">="],
    "vol15": [">="],
    "tr_value_ratio_5d": [">="],

    # 변동성/낙폭
    "ATR_pct": [">="],
    "max_drop_7d": ["<="],

    # 반등/위치/돌파
    "rebound_from_7d_low": [">="],
    "dist_to_ma5": ["<=", ">="],
    "dist_to_ma20": ["<=", ">="],
    "rebound_vs_prior_drop": ["<=", ">="],
    "BB_perc": ["<=", ">="],
    "room_to_60d_high": ["<=", ">="],
    "pct_vs_lastweek": ["<=", ">="],
    "ma5_chg_rate": ["<=", ">="],
    "gap_pct": ["<=", ">="],

    # 당일 힘/캔들
    "today_pct": [">="],
    "price_power_value": [">="],
    "intraday_return": [">="],
    "body_value_power": [">="],
    "upper_wick_ratio": ["<="],
}


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
    valid_metrics: Dict
    train_score: float
    profile_name: str

    def name(self) -> str:
        return " AND ".join([a.name() for a in self.atoms])

    def features(self) -> List[str]:
        return sorted(set(a.feature for a in self.atoms))


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


def prepare_df(df: pd.DataFrame, target_col: str, date_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    df[target_col] = df[target_col].astype(int)

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col].notna()].copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        df["month"] = df[date_col].dt.to_period("M").astype(str)
    else:
        df = df.reset_index(drop=True)
        df["month"] = "unknown"

    return df


def split_train_valid(df: pd.DataFrame, valid_ratio: float):
    split_idx = int(len(df) * (1 - valid_ratio))

    train = df.iloc[:split_idx].copy().reset_index(drop=True)
    valid = df.iloc[split_idx:].copy().reset_index(drop=True)

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


def auc_with_direction(y_true, x):
    y = np.asarray(y_true)
    s = np.asarray(x)

    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0 or len(np.unique(s)) <= 1:
        return {
            "auc_raw": np.nan,
            "auc_oriented": np.nan,
            "auc_direction": "",
        }

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)

    tmp = pd.DataFrame({"s": s, "rank": ranks})
    avg_rank = tmp.groupby("s")["rank"].transform("mean").values

    pos_rank_sum = avg_rank[y == 1].sum()
    auc_raw = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    auc_raw = float(auc_raw)

    if auc_raw >= 0.5:
        return {
            "auc_raw": auc_raw,
            "auc_oriented": auc_raw,
            "auc_direction": "higher_success",
        }

    return {
        "auc_raw": auc_raw,
        "auc_oriented": 1.0 - auc_raw,
        "auc_direction": "lower_success",
    }


def single_feature_report(train, features, target_col):
    rows = []

    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        mask = x.notna() & np.isfinite(x)

        xv = x[mask].values
        yv = train.loc[mask, target_col].astype(int).values

        base_rate = float(np.mean(yv)) if len(yv) else np.nan

        if len(xv) < 100 or len(np.unique(xv)) <= 2:
            rows.append({
                "feature": f,
                "auc_raw": np.nan,
                "auc_direction": "",
                "auc_oriented": np.nan,
                "best_bin": "",
                "best_bin_precision": np.nan,
                "best_bin_lift": np.nan,
                "best_bin_count": 0,
                "missing_rate": 1 - mask.mean(),
            })
            continue

        auc_info = auc_with_direction(yv, xv)

        try:
            bins = pd.qcut(x[mask], q=10, duplicates="drop")
        except Exception:
            bins = pd.cut(x[mask], bins=10, duplicates="drop")

        tmp = pd.DataFrame({
            "bin": bins.astype(str),
            "x": xv,
            "y": yv,
        })

        bin_stats = tmp.groupby("bin", observed=False)["y"].agg(
            ["count", "sum", "mean"]
        )

        bin_stats["lift"] = bin_stats["mean"] / base_rate if base_rate > 0 else np.nan

        best = bin_stats.sort_values(
            ["lift", "mean", "count"],
            ascending=False,
        ).iloc[0]

        rows.append({
            "feature": f,
            "auc_raw": auc_info["auc_raw"],
            "auc_direction": auc_info["auc_direction"],
            "auc_oriented": auc_info["auc_oriented"],
            "best_bin": best.name,
            "best_bin_precision": best["mean"],
            "best_bin_lift": best["lift"],
            "best_bin_count": int(best["count"]),
            "missing_rate": 1 - mask.mean(),
        })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["auc_oriented", "best_bin_lift"],
            ascending=[False, False],
        )

    return out


def build_corr_report(train, features, corr_threshold):
    x = train[features].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman")

    rows = []
    corr_pairs = set()

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

                corr_pairs.add(frozenset([a, b]))

    high_corr_df = pd.DataFrame(
        rows,
        columns=[
            "feature_a",
            "feature_b",
            "spearman_corr",
            "abs_corr",
        ],
    )

    if len(high_corr_df):
        high_corr_df = high_corr_df.sort_values(
            "abs_corr",
            ascending=False,
        )

    return corr, high_corr_df, corr_pairs


def has_correlated_pair(features_in_rule: Set[str], new_feature: str, corr_pairs):
    for f in features_in_rule:
        if frozenset([f, new_feature]) in corr_pairs:
            return True

    return False


def apply_atom(df, atom):
    x = pd.to_numeric(df[atom.feature], errors="coerce")

    if atom.op == ">=":
        return (x >= atom.threshold).fillna(False).values

    if atom.op == "<=":
        return (x <= atom.threshold).fillna(False).values

    raise ValueError(atom.op)


def apply_rule(df, atoms):
    mask = np.ones(len(df), dtype=bool)

    for atom in atoms:
        mask &= apply_atom(df, atom)

    return mask


def default_extra_thresholds() -> Dict[str, List[Tuple[str, float]]]:
    return {
        # 거래량/수급 강도
        "vol5": [
            (">=", 4.0), (">=", 5.0), (">=", 6.0), (">=", 6.507),
            (">=", 8.0), (">=", 8.467), (">=", 10.0), (">=", 10.5491),
            (">=", 10.6824), (">=", 13.4364),
        ],
        "vol15": [
            (">=", 4.0), (">=", 5.0), (">=", 6.0),
            (">=", 6.771), (">=", 8.0), (">=", 10.0),
            (">=", 12.0), (">=", 18.423),
        ],
        "tr_value_ratio_5d": [
            (">=", 1.0), (">=", 1.2), (">=", 1.5),
            (">=", 2.0), (">=", 3.0), (">=", 5.0),
        ],

        # 변동성/낙폭
        "ATR_pct": [
            (">=", 5.0), (">=", 6.0), (">=", 7.0),
            (">=", 8.0), (">=", 9.59), (">=", 12.0),
        ],
        "max_drop_7d": [
            ("<=", -3.54), ("<=", -3.7931), ("<=", -5.0),
            ("<=", -7.0), ("<=", -10.0), ("<=", -12.0),
        ],

        # 반등/위치
        "rebound_from_7d_low": [
            (">=", 20.0), (">=", 25.0), (">=", 30.0),
            (">=", 32.9122), (">=", 33.1722), (">=", 35.0),
            (">=", 40.0), (">=", 40.7563), (">=", 45.0),
            (">=", 50.0),
        ],
        "dist_to_ma5": [
            ("<=", -2.0), ("<=", -1.0), ("<=", -0.5), ("<=", -0.3),
            ("<=", -0.0182), (">=", 8.0), (">=", 10.0),
            (">=", 14.6684), (">=", 16.0), (">=", 20.2474),
            (">=", 20.8499),
        ],
        "dist_to_ma20": [
            (">=", 8.0), (">=", 10.0), (">=", 13.4333),
            (">=", 16.0), (">=", 19.2487),
            ("<=", -5.0), ("<=", -10.0),
        ],
        "rebound_vs_prior_drop": [
            (">=", 1.0), (">=", 2.0), (">=", 3.0),
            (">=", 4.859), (">=", 5.0),
            ("<=", 5.0), ("<=", 10.0),
        ],
        "room_to_60d_high": [
            (">=", -10.0), (">=", -5.0), (">=", 0.0),
            (">=", 3.0), (">=", 5.0), (">=", 10.0),
            (">=", 60.0), (">=", 65.0),
            ("<=", 5.0), ("<=", 10.0), ("<=", 18.9896),
            ("<=", 20.0), ("<=", 26.0304), ("<=", 30.0),
            ("<=", 50.0), ("<=", 58.511),
        ],

        # 레거시/돌파/갭 필터
        "BB_perc": [
            ("<=", 0.05), ("<=", 0.10), ("<=", 0.1765),
            ("<=", 0.25), ("<=", 0.30),
            (">=", 1.0), (">=", 1.054), (">=", 1.07), (">=", 1.20),
        ],
        "gap_pct": [
            ("<=", -2.0), ("<=", -1.0), ("<=", -0.5),
            ("<=", 0.0), ("<=", 0.5),
            (">=", -0.962), (">=", 2.0), (">=", 3.5), (">=", 4.7),
        ],
        "ma5_chg_rate": [
            ("<=", -5.0), ("<=", -3.0), ("<=", -1.5),
            ("<=", 0.0), ("<=", 10.0),
            (">=", 3.0), (">=", 5.0), (">=", 8.0), (">=", 10.0),
        ],
        "pct_vs_lastweek": [
            ("<=", -5.0), ("<=", -3.0), ("<=", -1.0),
            ("<=", 0.0), ("<=", 0.21),
            (">=", 5.0), (">=", 10.0), (">=", 11.1886),
        ],

        # 당일 힘/캔들
        "today_pct": [
            (">=", 5.0), (">=", 8.0), (">=", 10.0), (">=", 11.253),
            (">=", 13.0), (">=", 15.0), (">=", 15.86),
            (">=", 18.0), (">=", 20.0), (">=", 22.2708),
            (">=", 24.4445),
        ],
        "price_power_value": [
            (">=", 20.0), (">=", 40.0), (">=", 57.1494),
            (">=", 60.0), (">=", 80.0), (">=", 94.8492),
            (">=", 100.0),
        ],
        "intraday_return": [
            (">=", 4.0), (">=", 5.0), (">=", 6.0),
            (">=", 8.0), (">=", 10.0), (">=", 19.8174),
        ],
        "body_value_power": [
            (">=", 5.0), (">=", 10.0), (">=", 15.0),
            (">=", 20.0), (">=", 30.0), (">=", 99.5581),
        ],
        "upper_wick_ratio": [
            ("<=", 0.0), ("<=", 0.008), ("<=", 0.02),
            ("<=", 0.038), ("<=", 0.05), ("<=", 0.1),
            ("<=", 0.2), ("<=", 0.3),
        ],
    }

def make_atoms(
        train,
        features,
        quantiles,
        target_col,
        min_atom_count,
        min_atom_lift,
        min_atom_precision,
        allowed_ops,
        extra_thresholds,
):
    atoms = []
    y = train[target_col].astype(int).values
    base_rate = train[target_col].mean()

    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        x_valid = x[np.isfinite(x)]

        if len(x_valid) < 100:
            continue

        if x_valid.nunique() <= 3:
            continue

        qs = np.nanquantile(x_valid, quantiles)
        qs = sorted(set(float(q) for q in qs if np.isfinite(q)))

        raw_candidates = []

        for q in qs:
            for op in ["<=", ">="]:
                if f in allowed_ops and op not in allowed_ops[f]:
                    continue

                raw_candidates.append((op, float(q)))

        for op, th in extra_thresholds.get(f, []):
            if f in allowed_ops and op not in allowed_ops[f]:
                continue

            raw_candidates.append((op, float(th)))

        unique = {}
        for op, th in raw_candidates:
            unique[(op, round(th, 10))] = (op, th)

        for op, th in unique.values():
            atom = Atom(f, op, th)
            mask = apply_atom(train, atom)

            count = int(mask.sum())
            if count < min_atom_count:
                continue

            precision = float(y[mask].mean()) if count else np.nan
            lift = precision / base_rate if base_rate > 0 else np.nan

            if not np.isfinite(precision) or not np.isfinite(lift):
                continue

            if lift < min_atom_lift:
                continue

            if precision < min_atom_precision:
                continue

            atoms.append(atom)

    return atoms


def train_rule_score(
        train_metrics,
        min_train_count,
        min_train_lift,
        min_train_precision,
        max_train_coverage,
):
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

    if precision < min_train_precision:
        return -1e18

    if lift < min_train_lift:
        return -1e18

    wide_penalty = 0.0
    if coverage > max_train_coverage:
        wide_penalty = (coverage - max_train_coverage) * 120.0

    score = (
            precision * 100.0
            + lift * 30.0
            + math.log1p(target_count) * 1.5
            + recall * 4.0
            - wide_penalty
    )

    return score


def make_profiles(args):
    base_quantiles = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
        0.975,
    ]

    return [
        {
            "name": "avg60_broad",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "quantiles": base_quantiles,

            "min_atom_count": 60,
            "min_atom_lift": 1.02,
            "min_atom_precision": 0.41,

            "min_train_count": 100,
            "min_valid_count": 40,

            "beam_min_train_precision": 0.41,
            "beam_min_train_lift": 1.00,

            "min_train_precision": 0.47,
            "min_train_lift": 1.08,

            "min_valid_precision": 0.50,
            "min_valid_lift": 1.10,

            "max_train_coverage": 0.70,
        },
        {
            "name": "avg60_precision",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "quantiles": base_quantiles,

            "min_atom_count": 50,
            "min_atom_lift": 1.04,
            "min_atom_precision": 0.43,

            "min_train_count": 70,
            "min_valid_count": 30,

            "beam_min_train_precision": 0.42,
            "beam_min_train_lift": 1.00,

            "min_train_precision": 0.50,
            "min_train_lift": 1.15,

            "min_valid_precision": 0.55,
            "min_valid_lift": 1.25,

            "max_train_coverage": 0.55,
        },
        {
            "name": "avg60_high_precision",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "quantiles": base_quantiles,

            "min_atom_count": 40,
            "min_atom_lift": 1.06,
            "min_atom_precision": 0.45,

            "min_train_count": 50,
            "min_valid_count": 25,

            "beam_min_train_precision": 0.43,
            "beam_min_train_lift": 1.00,

            "min_train_precision": 0.52,
            "min_train_lift": 1.20,

            "min_valid_precision": 0.58,
            "min_valid_lift": 1.35,

            "max_train_coverage": 0.45,
        },
    ]


def search_rules_train_only(
        train,
        valid,
        features,
        target_col,
        profile,
        corr_pairs,
        block_correlated_in_rule=True,
):
    y_train = train[target_col].astype(int).values
    y_valid = valid[target_col].astype(int).values

    atoms = make_atoms(
        train=train,
        features=features,
        quantiles=profile["quantiles"],
        target_col=target_col,
        min_atom_count=profile["min_atom_count"],
        min_atom_lift=profile["min_atom_lift"],
        min_atom_precision=profile["min_atom_precision"],
        allowed_ops=ALLOWED_OPS,
        extra_thresholds=default_extra_thresholds(),
    )

    print(f"[INFO][{profile['name']}] atoms generated: {len(atoms)}")

    beam = [tuple()]
    raw_rules = []
    final_rules = []
    seen = set()

    for depth in range(1, profile["max_depth"] + 1):
        candidates_for_beam = []
        candidates_for_final = []

        for base_atoms in beam:
            used_features = set(a.feature for a in base_atoms)

            for atom in atoms:
                if atom.feature in used_features:
                    continue

                if block_correlated_in_rule:
                    if has_correlated_pair(used_features, atom.feature, corr_pairs):
                        continue

                new_atoms = tuple(list(base_atoms) + [atom])

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

                beam_score = train_rule_score(
                    train_m,
                    min_train_count=profile["min_train_count"],
                    min_train_lift=profile["beam_min_train_lift"],
                    min_train_precision=profile["beam_min_train_precision"],
                    max_train_coverage=profile["max_train_coverage"],
                )

                if beam_score <= -1e17:
                    continue

                valid_mask = apply_rule(valid, new_atoms)
                valid_m = precision_recall_lift(y_valid, valid_mask)

                beam_rule = Rule(
                    atoms=new_atoms,
                    train_metrics=train_m,
                    valid_metrics=valid_m,
                    train_score=beam_score,
                    profile_name=profile["name"],
                )

                candidates_for_beam.append(beam_rule)

                final_score = train_rule_score(
                    train_m,
                    min_train_count=profile["min_train_count"],
                    min_train_lift=profile["min_train_lift"],
                    min_train_precision=profile["min_train_precision"],
                    max_train_coverage=profile["max_train_coverage"],
                )

                valid_ok = (
                        valid_m["count"] >= profile["min_valid_count"]
                        and np.isfinite(valid_m["precision"])
                        and np.isfinite(valid_m["lift"])
                        and valid_m["precision"] >= profile["min_valid_precision"]
                        and valid_m["lift"] >= profile["min_valid_lift"]
                )

                if final_score > -1e17 and valid_ok:
                    final_rule = Rule(
                        atoms=new_atoms,
                        train_metrics=train_m,
                        valid_metrics=valid_m,
                        train_score=final_score,
                        profile_name=profile["name"],
                    )
                    candidates_for_final.append(final_rule)

        if not candidates_for_beam:
            print(f"[WARN][{profile['name']}] no beam candidates at depth={depth}")
            break

        candidates_for_beam = sorted(
            candidates_for_beam,
            key=lambda r: r.train_score,
            reverse=True,
        )

        candidates_for_final = sorted(
            candidates_for_final,
            key=lambda r: (
                r.valid_metrics["precision"]
                if np.isfinite(r.valid_metrics["precision"])
                else -1,
                r.valid_metrics["count"],
                r.train_score,
            ),
            reverse=True,
        )

        raw_rules.extend(candidates_for_beam[:profile["top_k"]])
        final_rules.extend(candidates_for_final[:profile["top_k"]])

        beam = [r.atoms for r in candidates_for_beam[:profile["beam_width"]]]

        print(
            f"[INFO][{profile['name']}] depth={depth}, "
            f"beam={len(candidates_for_beam)}, "
            f"final={len(candidates_for_final)}, "
            f"kept={len(beam)}"
        )

    unique_final = {}
    for r in final_rules:
        unique_final[(r.profile_name, r.name())] = r

    unique_raw = {}
    for r in raw_rules:
        unique_raw[(r.profile_name, r.name())] = r

    final_rules = list(unique_final.values())
    raw_rules = list(unique_raw.values())

    final_rules = sorted(
        final_rules,
        key=lambda r: (
            r.valid_metrics["precision"]
            if np.isfinite(r.valid_metrics["precision"])
            else -1,
            r.valid_metrics["count"],
            r.train_score,
        ),
        reverse=True,
    )

    raw_rules = sorted(raw_rules, key=lambda r: r.train_score, reverse=True)

    return final_rules[:profile["top_k"]], raw_rules[:profile["top_k"]]


def rules_to_df(rules):
    rows = []

    for i, r in enumerate(rules, start=1):
        row = {
            "profile": r.profile_name,
            "rank": i,
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
            if np.isfinite(row["valid_precision"])
               and np.isfinite(row["train_precision"])
            else np.nan
        )

        row["abs_precision_gap"] = abs(row["precision_gap_valid_minus_train"])

        row["pass_valid_55"] = (
                row["valid_precision"] >= 0.55
                and row["valid_lift"] >= 1.25
                and row["valid_count"] >= 50
        )

        row["pass_valid_60"] = (
                row["valid_precision"] >= 0.60
                and row["valid_lift"] >= 1.40
                and row["valid_count"] >= 30
        )

        rows.append(row)

    return pd.DataFrame(rows)


def parse_rule_text_to_atoms(rule_text: str) -> List[Atom]:
    atoms = []

    if not rule_text or rule_text == "nan":
        return atoms

    for part in rule_text.split(" AND "):
        part = part.strip()

        if " >= " in part:
            f, th = part.split(" >= ")
            try:
                atoms.append(Atom(f.strip(), ">=", float(th)))
            except Exception:
                pass

        elif " <= " in part:
            f, th = part.split(" <= ")
            try:
                atoms.append(Atom(f.strip(), "<=", float(th)))
            except Exception:
                pass

    return atoms


def rule_feature_tuple(rule_text: str) -> Tuple[str, ...]:
    atoms = parse_rule_text_to_atoms(str(rule_text))
    return tuple(sorted(set(a.feature for a in atoms)))


def rule_feature_pair_keys(rule_text: str) -> List[Tuple[str, str]]:
    feats = list(rule_feature_tuple(rule_text))
    pairs = []

    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            pairs.append((a, b))

    return pairs


def atom_signature(atom: Atom) -> Tuple[str, str, float]:
    """
    near-duplicate rule 제거용.
    threshold를 너무 정밀하게 보지 않고 피쳐별로 적당히 둥글린다.
    """
    f = atom.feature
    th = float(atom.threshold)

    if f in ["BB_perc"]:
        rounded = round(th, 2)
    elif f in ["gap_pct", "upper_wick_ratio"]:
        rounded = round(th, 2)
    elif f in [
        "vol5", "vol15", "today_pct", "max_drop_7d",
        "rebound_from_7d_low", "dist_to_ma5", "dist_to_ma20",
        "pct_vs_lastweek", "room_to_60d_high", "rebound_vs_prior_drop",
        "intraday_return", "body_value_power", "price_power_value",
        "tr_value_ratio_5d", "ATR_pct", "ma5_chg_rate",
    ]:
        rounded = round(th, 1)
    else:
        rounded = round(th, 2)

    return f, atom.op, rounded


def rule_signature(rule_text: str) -> Tuple[Tuple[str, str, float], ...]:
    atoms = parse_rule_text_to_atoms(str(rule_text))
    sig = tuple(sorted(atom_signature(a) for a in atoms))
    return sig


def rule_family_signature(rule_text: str) -> Tuple[Tuple[str, str], ...]:
    """
    threshold는 무시하고 feature + op 조합만 본다.
    """
    atoms = parse_rule_text_to_atoms(str(rule_text))
    sig = tuple(sorted((a.feature, a.op) for a in atoms))
    return sig


def add_rule_signature_columns(rules_df: pd.DataFrame) -> pd.DataFrame:
    if len(rules_df) == 0:
        return rules_df.copy()

    out = rules_df.copy()

    out["feature_tuple"] = out["rule"].apply(rule_feature_tuple)
    out["feature_set_key"] = out["feature_tuple"].apply(lambda x: "|".join(x))
    out["rule_signature"] = out["rule"].apply(rule_signature)
    out["rule_signature_key"] = out["rule_signature"].apply(lambda x: str(x))
    out["rule_family"] = out["rule"].apply(rule_family_signature)
    out["rule_family_key"] = out["rule_family"].apply(lambda x: str(x))

    return out


def summarize_selected_portfolio(selected_df: pd.DataFrame) -> Dict:
    if len(selected_df) == 0:
        return {
            "rule_count": 0,
            "avg_valid_precision": np.nan,
            "avg_valid_lift": np.nan,
            "avg_valid_count": np.nan,
            "pass_valid_60_count": 0,
            "unique_feature_sets": 0,
            "unique_rule_families": 0,
        }

    return {
        "rule_count": len(selected_df),
        "avg_valid_precision": selected_df["valid_precision"].mean(),
        "avg_valid_lift": selected_df["valid_lift"].mean(),
        "avg_valid_count": selected_df["valid_count"].mean(),
        "pass_valid_60_count": int(selected_df["pass_valid_60"].sum())
        if "pass_valid_60" in selected_df.columns else 0,
        "unique_feature_sets": selected_df["feature_set_key"].nunique()
        if "feature_set_key" in selected_df.columns else np.nan,
        "unique_rule_families": selected_df["rule_family_key"].nunique()
        if "rule_family_key" in selected_df.columns else np.nan,
    }


def save_portfolio_summary(selected_df: pd.DataFrame, out_dir: str):
    summary = summarize_selected_portfolio(selected_df)
    summary_df = pd.DataFrame([summary])

    summary_df.to_csv(
        os.path.join(out_dir, "04_selected_avg60_rules_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    return summary_df


def select_avg60_rule_portfolio(
        rules_df,
        target_avg_precision,
        min_rule_count,
        max_rule_count,
        min_valid_count,
        max_same_feature_set=3,
        max_same_rule_family=2,
        max_same_feature_pair=8,
        min_precision_floor=0.55,
        prefer_monthly_stable=False,
        monthly_summary_df=None,
):
    """
    valid precision 평균 60% 이상을 유지하면서,
    중복 룰을 줄이고 다양한 feature 조합을 고르는 포트폴리오 선택 함수.
    """

    if len(rules_df) == 0:
        return pd.DataFrame()

    use = rules_df[
        (rules_df["valid_count"] >= min_valid_count)
        & (rules_df["valid_precision"].notna())
        & (rules_df["valid_lift"].notna())
        & (rules_df["valid_precision"] >= min_precision_floor)
        ].copy()

    if len(use) == 0:
        return pd.DataFrame()

    use = add_rule_signature_columns(use)

    # 완전히 비슷한 룰 제거.
    # 같은 rounded signature이면 valid_precision 높은 것만 남긴다.
    use = use.sort_values(
        ["rule_signature_key", "valid_precision", "valid_lift", "valid_count"],
        ascending=[True, False, False, False],
    )

    use = use.drop_duplicates(
        subset=["rule_signature_key"],
        keep="first",
    )

    if prefer_monthly_stable and monthly_summary_df is not None and len(monthly_summary_df):
        ms = monthly_summary_df.copy()
        ms_cols = [
            "rule",
            "months_used",
            "median_month_precision",
            "min_month_precision",
            "bad_months_precision_lt_50",
            "bad_months_lift_lt_1",
        ]
        ms_cols = [c for c in ms_cols if c in ms.columns]

        use = use.merge(
            ms[ms_cols],
            on="rule",
            how="left",
        )

        use["months_used"] = use["months_used"].fillna(0)
        use["bad_months_precision_lt_50"] = use["bad_months_precision_lt_50"].fillna(99)
        use["bad_months_lift_lt_1"] = use["bad_months_lift_lt_1"].fillna(99)
        use["median_month_precision"] = use["median_month_precision"].fillna(0)

        use["portfolio_sort_score"] = (
                use["valid_precision"] * 100
                + use["valid_lift"] * 8
                + np.log1p(use["valid_count"]) * 2
                + use["months_used"] * 2
                + use["median_month_precision"] * 5
                - use["bad_months_precision_lt_50"] * 3
                - use["bad_months_lift_lt_1"] * 3
        )
    else:
        use["portfolio_sort_score"] = (
                use["valid_precision"] * 100
                + use["valid_lift"] * 8
                + np.log1p(use["valid_count"]) * 2
        )

    use = use.sort_values(
        ["portfolio_sort_score", "valid_precision", "valid_lift", "valid_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    selected = []

    feature_set_counts = {}
    rule_family_counts = {}
    feature_pair_counts = {}

    for _, r in use.iterrows():
        fs_key = r["feature_set_key"]
        fam_key = r["rule_family_key"]
        pair_keys = rule_feature_pair_keys(r["rule"])

        if feature_set_counts.get(fs_key, 0) >= max_same_feature_set:
            continue

        if rule_family_counts.get(fam_key, 0) >= max_same_rule_family:
            continue

        pair_too_many = False
        for p in pair_keys:
            if feature_pair_counts.get(p, 0) >= max_same_feature_pair:
                pair_too_many = True
                break

        if pair_too_many:
            continue

        candidate = selected + [r]
        avg_precision = np.mean([x["valid_precision"] for x in candidate])

        if avg_precision < target_avg_precision:
            continue

        selected.append(r)

        feature_set_counts[fs_key] = feature_set_counts.get(fs_key, 0) + 1
        rule_family_counts[fam_key] = rule_family_counts.get(fam_key, 0) + 1

        for p in pair_keys:
            feature_pair_counts[p] = feature_pair_counts.get(p, 0) + 1

        if len(selected) >= max_rule_count:
            break

    out = pd.DataFrame(selected)

    # 너무 적게 잡히면 제한을 완화해서 fallback
    if len(out) < min_rule_count:
        selected = []

        feature_set_counts = {}
        rule_family_counts = {}

        fallback = use[
            use["valid_precision"] >= target_avg_precision
            ].copy()

        for _, r in fallback.iterrows():
            fs_key = r["feature_set_key"]
            fam_key = r["rule_family_key"]

            if feature_set_counts.get(fs_key, 0) >= max_same_feature_set + 2:
                continue

            if rule_family_counts.get(fam_key, 0) >= max_same_rule_family + 2:
                continue

            selected.append(r)

            feature_set_counts[fs_key] = feature_set_counts.get(fs_key, 0) + 1
            rule_family_counts[fam_key] = rule_family_counts.get(fam_key, 0) + 1

            if len(selected) >= max_rule_count:
                break

        out = pd.DataFrame(selected)

    if len(out):
        out = out.copy()
        out["portfolio_avg_valid_precision"] = out["valid_precision"].mean()
        out["portfolio_avg_valid_lift"] = out["valid_lift"].mean()
        out["portfolio_rule_count"] = len(out)
        out["portfolio_unique_feature_sets"] = out["feature_set_key"].nunique()
        out["portfolio_unique_rule_families"] = out["rule_family_key"].nunique()

    return out


def feature_usage_from_rules(rules_df, features, top_n):
    rows = []

    if len(rules_df) == 0:
        return pd.DataFrame()

    top = rules_df.sort_values(
        ["valid_precision", "valid_lift"],
        ascending=[False, False],
    ).head(top_n)

    for f in features:
        used = top["features"].fillna("").apply(
            lambda s: f in [x.strip() for x in str(s).split(",") if x.strip()]
        )

        used_df = top[used]

        row = {
            "feature": f,
            "top_usage_count": int(used.sum()),
            "top_usage_rate": float(used.mean()) if len(top) else 0.0,
        }

        if len(used_df):
            row["avg_train_precision_when_used"] = used_df["train_precision"].mean()
            row["avg_valid_precision_when_used"] = used_df["valid_precision"].mean()
            row["avg_train_lift_when_used"] = used_df["train_lift"].mean()
            row["avg_valid_lift_when_used"] = used_df["valid_lift"].mean()
            row["avg_abs_precision_gap_when_used"] = used_df["abs_precision_gap"].mean()
            row["best_valid_precision_when_used"] = used_df["valid_precision"].max()
            row["best_valid_lift_when_used"] = used_df["valid_lift"].max()
            row["pass_valid_60_count"] = int(used_df["pass_valid_60"].sum())
        else:
            row["avg_train_precision_when_used"] = np.nan
            row["avg_valid_precision_when_used"] = np.nan
            row["avg_train_lift_when_used"] = np.nan
            row["avg_valid_lift_when_used"] = np.nan
            row["avg_abs_precision_gap_when_used"] = np.nan
            row["best_valid_precision_when_used"] = np.nan
            row["best_valid_lift_when_used"] = np.nan
            row["pass_valid_60_count"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def direction_stability_report(rules_df, features, top_n):
    rows = []

    if len(rules_df) == 0:
        return pd.DataFrame()

    top = rules_df.sort_values(
        ["valid_precision", "valid_lift"],
        ascending=[False, False],
    ).head(top_n)

    for f in features:
        op_list = []
        threshold_list = []
        used_rules = 0

        for _, r in top.iterrows():
            atoms = str(r["rule"]).split(" AND ")

            used_here = False

            for atom_text in atoms:
                atom_text = atom_text.strip()

                prefix_ge = f"{f} >= "
                prefix_le = f"{f} <= "

                if atom_text.startswith(prefix_ge):
                    op_list.append(">=")
                    try:
                        threshold_list.append(float(atom_text.replace(prefix_ge, "")))
                    except Exception:
                        pass
                    used_here = True

                elif atom_text.startswith(prefix_le):
                    op_list.append("<=")
                    try:
                        threshold_list.append(float(atom_text.replace(prefix_le, "")))
                    except Exception:
                        pass
                    used_here = True

            if used_here:
                used_rules += 1

        ge_count = sum(1 for x in op_list if x == ">=")
        le_count = sum(1 for x in op_list if x == "<=")
        total = ge_count + le_count

        if total > 0:
            dominant_op = ">=" if ge_count >= le_count else "<="
            dominant_count = max(ge_count, le_count)
            direction_consistency = dominant_count / total
            median_threshold = float(np.median(threshold_list)) if threshold_list else np.nan
        else:
            dominant_op = ""
            direction_consistency = np.nan
            median_threshold = np.nan

        rows.append({
            "feature": f,
            "used_rules": used_rules,
            "ge_count": ge_count,
            "le_count": le_count,
            "dominant_op": dominant_op,
            "direction_consistency": direction_consistency,
            "median_threshold": median_threshold,
        })

    return pd.DataFrame(rows)


def monthly_rule_stability(valid, rules_df, target_col):
    if len(rules_df) == 0 or "month" not in valid.columns:
        return pd.DataFrame()

    tmp_valid = valid.copy().reset_index(drop=True)
    y_all = tmp_valid[target_col].astype(int).values

    rows = []

    for rule_idx, r in rules_df.reset_index(drop=True).iterrows():
        atoms = parse_rule_text_to_atoms(str(r["rule"]))

        if not atoms:
            continue

        mask_all = apply_rule(tmp_valid, tuple(atoms))

        for month, idx_values in tmp_valid.groupby("month").groups.items():
            pos = np.array(list(idx_values), dtype=int)
            m = precision_recall_lift(y_all[pos], mask_all[pos])

            rows.append({
                "rule_rank": rule_idx + 1,
                "rule": r["rule"],
                "month": month,
                "month_count": m["count"],
                "month_target_count": m["target_count"],
                "month_precision": m["precision"],
                "month_lift": m["lift"],
                "month_coverage": m["coverage"],
                "month_base_rate": m["base_rate"],
            })

    return pd.DataFrame(rows)


def conditional_feature_contribution(
        train,
        valid,
        rules_df,
        target_col,
):
    rows = []

    if len(rules_df) == 0:
        return pd.DataFrame()

    y_train = train[target_col].astype(int).values
    y_valid = valid[target_col].astype(int).values

    for rule_idx, r in rules_df.reset_index(drop=True).iterrows():
        atoms = tuple(parse_rule_text_to_atoms(str(r["rule"])))

        if len(atoms) == 0:
            continue

        full_train_mask = apply_rule(train, atoms)
        full_valid_mask = apply_rule(valid, atoms)

        full_train_m = precision_recall_lift(y_train, full_train_mask)
        full_valid_m = precision_recall_lift(y_valid, full_valid_mask)

        rule_features = sorted(set(a.feature for a in atoms))

        for f in rule_features:
            reduced_atoms = tuple(a for a in atoms if a.feature != f)

            if len(reduced_atoms) == 0:
                reduced_train_mask = np.ones(len(train), dtype=bool)
                reduced_valid_mask = np.ones(len(valid), dtype=bool)
            else:
                reduced_train_mask = apply_rule(train, reduced_atoms)
                reduced_valid_mask = apply_rule(valid, reduced_atoms)

            reduced_train_m = precision_recall_lift(y_train, reduced_train_mask)
            reduced_valid_m = precision_recall_lift(y_valid, reduced_valid_mask)

            rows.append({
                "rule_rank": rule_idx + 1,
                "rule": r["rule"],
                "feature": f,

                "full_valid_count": full_valid_m["count"],
                "full_valid_precision": full_valid_m["precision"],
                "full_valid_lift": full_valid_m["lift"],

                "reduced_valid_count": reduced_valid_m["count"],
                "reduced_valid_precision": reduced_valid_m["precision"],
                "reduced_valid_lift": reduced_valid_m["lift"],

                "valid_precision_gain": (
                    full_valid_m["precision"] - reduced_valid_m["precision"]
                    if np.isfinite(full_valid_m["precision"])
                       and np.isfinite(reduced_valid_m["precision"])
                    else np.nan
                ),

                "valid_lift_gain": (
                    full_valid_m["lift"] - reduced_valid_m["lift"]
                    if np.isfinite(full_valid_m["lift"])
                       and np.isfinite(reduced_valid_m["lift"])
                    else np.nan
                ),

                "valid_count_change": full_valid_m["count"] - reduced_valid_m["count"],

                "full_train_count": full_train_m["count"],
                "full_train_precision": full_train_m["precision"],
                "full_train_lift": full_train_m["lift"],

                "reduced_train_count": reduced_train_m["count"],
                "reduced_train_precision": reduced_train_m["precision"],
                "reduced_train_lift": reduced_train_m["lift"],

                "train_precision_gain": (
                    full_train_m["precision"] - reduced_train_m["precision"]
                    if np.isfinite(full_train_m["precision"])
                       and np.isfinite(reduced_train_m["precision"])
                    else np.nan
                ),
            })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["valid_precision_gain", "valid_lift_gain"],
            ascending=False,
        )

    return out


def summarize_conditional_contribution(contrib_df):
    if len(contrib_df) == 0:
        return pd.DataFrame()

    rows = []

    for f, g in contrib_df.groupby("feature"):
        useful = g[
            (g["valid_precision_gain"] > 0)
            & (g["valid_lift_gain"] > 0)
            ].copy()

        strong = g[
            (g["valid_precision_gain"] >= 0.02)
            & (g["valid_lift_gain"] > 0)
            ].copy()

        rows.append({
            "feature": f,
            "rules_used": g["rule"].nunique(),

            "mean_valid_precision_gain": g["valid_precision_gain"].mean(),
            "median_valid_precision_gain": g["valid_precision_gain"].median(),
            "max_valid_precision_gain": g["valid_precision_gain"].max(),

            "mean_valid_lift_gain": g["valid_lift_gain"].mean(),
            "median_valid_lift_gain": g["valid_lift_gain"].median(),
            "max_valid_lift_gain": g["valid_lift_gain"].max(),

            "positive_gain_rate": len(useful) / len(g) if len(g) else 0.0,
            "positive_gain_rules": useful["rule"].nunique() if len(useful) else 0,

            "strong_gain_rate": len(strong) / len(g) if len(g) else 0.0,
            "strong_gain_rules": strong["rule"].nunique() if len(strong) else 0,
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["positive_gain_rate", "mean_valid_precision_gain"],
        ascending=False,
    )

    return out


def grade_features(
        features,
        single_df,
        usage_df,
        direction_df,
        conditional_summary_df,
):
    rows = []

    for f in features:
        s = single_df[single_df["feature"] == f]
        u = usage_df[usage_df["feature"] == f] if len(usage_df) else pd.DataFrame()
        d = direction_df[direction_df["feature"] == f] if len(direction_df) else pd.DataFrame()
        c = conditional_summary_df[
            conditional_summary_df["feature"] == f
            ] if len(conditional_summary_df) else pd.DataFrame()

        if len(s):
            auc_oriented = s.iloc[0].get("auc_oriented", np.nan)
            auc_direction = s.iloc[0].get("auc_direction", "")
            best_bin = s.iloc[0].get("best_bin", "")
            best_bin_precision = s.iloc[0].get("best_bin_precision", np.nan)
            best_bin_lift = s.iloc[0].get("best_bin_lift", np.nan)
        else:
            auc_oriented = np.nan
            auc_direction = ""
            best_bin = ""
            best_bin_precision = np.nan
            best_bin_lift = np.nan

        if len(u):
            total_usage_count = int(u.iloc[0].get("top_usage_count", 0))
            avg_valid_precision = u.iloc[0].get("avg_valid_precision_when_used", np.nan)
            avg_valid_lift = u.iloc[0].get("avg_valid_lift_when_used", np.nan)
            best_valid_precision = u.iloc[0].get("best_valid_precision_when_used", np.nan)
            best_valid_lift = u.iloc[0].get("best_valid_lift_when_used", np.nan)
            avg_gap = u.iloc[0].get("avg_abs_precision_gap_when_used", np.nan)
            pass_valid_60_count = int(u.iloc[0].get("pass_valid_60_count", 0))
        else:
            total_usage_count = 0
            avg_valid_precision = np.nan
            avg_valid_lift = np.nan
            best_valid_precision = np.nan
            best_valid_lift = np.nan
            avg_gap = np.nan
            pass_valid_60_count = 0

        if len(d):
            dominant_op = d.iloc[0].get("dominant_op", "")
            avg_direction_consistency = d.iloc[0].get("direction_consistency", np.nan)
        else:
            dominant_op = ""
            avg_direction_consistency = np.nan

        if len(c):
            rules_used_conditional = c.iloc[0].get("rules_used", 0)
            mean_valid_precision_gain = c.iloc[0].get("mean_valid_precision_gain", np.nan)
            median_valid_precision_gain = c.iloc[0].get("median_valid_precision_gain", np.nan)
            max_valid_precision_gain = c.iloc[0].get("max_valid_precision_gain", np.nan)
            positive_gain_rate = c.iloc[0].get("positive_gain_rate", 0.0)
            strong_gain_rate = c.iloc[0].get("strong_gain_rate", 0.0)
        else:
            rules_used_conditional = 0
            mean_valid_precision_gain = np.nan
            median_valid_precision_gain = np.nan
            max_valid_precision_gain = np.nan
            positive_gain_rate = 0.0
            strong_gain_rate = 0.0

        is_non_mono = f in NON_MONOTONIC_FEATURES
        is_regime = f in REGIME_FEATURES

        if is_non_mono:
            direction_ok = True
        else:
            direction_ok = (
                    (auc_direction == "higher_success" and dominant_op == ">=")
                    or (auc_direction == "lower_success" and dominant_op == "<=")
                    or dominant_op == ""
            )

        score = 0.0

        if np.isfinite(auc_oriented):
            score += max(0.0, auc_oriented - 0.5) * 35

        if np.isfinite(best_bin_lift):
            score += max(0.0, best_bin_lift - 1.0) * 12

        if np.isfinite(avg_valid_lift):
            score += max(0.0, avg_valid_lift - 1.0) * 30

        if np.isfinite(avg_valid_precision):
            score += max(0.0, avg_valid_precision - 0.5) * 80

        if np.isfinite(mean_valid_precision_gain):
            score += max(0.0, mean_valid_precision_gain) * 150

        if np.isfinite(max_valid_precision_gain):
            score += max(0.0, max_valid_precision_gain) * 30

        score += positive_gain_rate * 18
        score += strong_gain_rate * 25
        score += total_usage_count * 0.8
        score += pass_valid_60_count * 10

        if np.isfinite(avg_gap):
            score -= avg_gap * 20

        if not direction_ok:
            score -= 8

        if f in WEAK_HINT_FEATURES:
            score -= 2

        # ============================================================
        # 강화된 등급 기준
        # ============================================================
        enough_usage_for_s = total_usage_count >= 4
        enough_usage_for_a = total_usage_count >= 2

        conditional_not_bad = (
                not np.isfinite(mean_valid_precision_gain)
                or mean_valid_precision_gain >= -0.005
        )

        is_core_signal = (
                direction_ok
                and not is_non_mono
                and enough_usage_for_s
                and conditional_not_bad
                and (
                        pass_valid_60_count >= 3
                        or (
                                np.isfinite(avg_valid_precision)
                                and avg_valid_precision >= 0.62
                                and np.isfinite(avg_valid_lift)
                                and avg_valid_lift >= 1.45
                        )
                        or (
                                np.isfinite(auc_oriented)
                                and auc_oriented >= 0.575
                                and pass_valid_60_count >= 1
                        )
                )
        )

        is_regime_filter = (
                is_regime
                and enough_usage_for_a
                and conditional_not_bad
                and (
                        pass_valid_60_count >= 2
                        or (
                                np.isfinite(avg_valid_lift)
                                and avg_valid_lift >= 1.35
                                and np.isfinite(avg_valid_precision)
                                and avg_valid_precision >= 0.60
                        )
                        or (
                                np.isfinite(best_bin_lift)
                                and best_bin_lift >= 1.35
                                and pass_valid_60_count >= 1
                        )
                )
        )

        is_conditional_filter = (
                is_non_mono
                and enough_usage_for_a
                and (
                        pass_valid_60_count >= 2
                        or (
                                rules_used_conditional >= 2
                                and positive_gain_rate >= 0.30
                                and np.isfinite(mean_valid_precision_gain)
                                and mean_valid_precision_gain > 0.005
                        )
                        or (
                                rules_used_conditional >= 1
                                and np.isfinite(max_valid_precision_gain)
                                and max_valid_precision_gain >= 0.04
                                and pass_valid_60_count >= 1
                        )
                )
        )

        is_candidate = (
                total_usage_count > 0
                or (np.isfinite(best_bin_lift) and best_bin_lift >= 1.12)
                or (np.isfinite(auc_oriented) and auc_oriented >= 0.53)
                or rules_used_conditional > 0
        )

        if is_regime_filter:
            grade = "S"
            role = "REGIME_FILTER"
            action = "KEEP"
            reason = "avg60 룰에서 장세 필터로 반복 기여"

        elif is_core_signal:
            grade = "S"
            role = "CORE_SIGNAL"
            action = "KEEP"
            reason = "avg60 룰에서 평균 valid precision을 올리는 핵심 신호"

        elif is_conditional_filter:
            grade = "A"
            role = "CONDITIONAL_FILTER"
            action = "KEEP_AS_FILTER"
            reason = "단독보다 룰 조합에서 valid precision을 올리는 조건부 필터"

        elif is_candidate:
            grade = "B"
            role = "CANDIDATE"
            action = "TEST"
            reason = "조건부 가능성은 있으나 avg60 룰 기여는 제한적"

        else:
            grade = "C"
            role = "DROP_CANDIDATE"
            action = "DROP_OR_LAST_CHECK"
            reason = "avg60 룰 기준 근거 약함"

        rows.append({
            "feature": f,
            "grade": grade,
            "role": role,
            "action": action,
            "score": score,
            "reason": reason,

            "is_non_monotonic": is_non_mono,
            "is_regime": is_regime,

            "total_usage_count": total_usage_count,
            "avg_valid_precision_when_used": avg_valid_precision,
            "avg_valid_lift_when_used": avg_valid_lift,
            "best_valid_precision_when_used": best_valid_precision,
            "best_valid_lift_when_used": best_valid_lift,
            "pass_valid_60_count": pass_valid_60_count,
            "avg_abs_precision_gap_when_used": avg_gap,

            "auc_oriented": auc_oriented,
            "auc_direction": auc_direction,
            "best_bin": best_bin,
            "best_bin_precision": best_bin_precision,
            "best_bin_lift": best_bin_lift,

            "dominant_op": dominant_op,
            "avg_direction_consistency": avg_direction_consistency,
            "direction_ok": direction_ok,

            "conditional_rules_used": rules_used_conditional,
            "mean_valid_precision_gain": mean_valid_precision_gain,
            "median_valid_precision_gain": median_valid_precision_gain,
            "max_valid_precision_gain": max_valid_precision_gain,
            "positive_gain_rate": positive_gain_rate,
            "strong_gain_rate": strong_gain_rate,
        })

    out = pd.DataFrame(rows)

    grade_order = {
        "S": 1,
        "A": 2,
        "B": 3,
        "C": 4,
    }

    out["_grade_order"] = out["grade"].map(grade_order).fillna(99)

    out = out.sort_values(
        ["_grade_order", "score"],
        ascending=[True, False],
    ).drop(columns=["_grade_order"])

    return out


def write_feature_grade_text(feature_grade_df, out_dir):
    s_core = feature_grade_df[
        (feature_grade_df["grade"] == "S")
        & (feature_grade_df["role"] == "CORE_SIGNAL")
        ]

    s_regime = feature_grade_df[
        (feature_grade_df["grade"] == "S")
        & (feature_grade_df["role"] == "REGIME_FILTER")
        ]

    a_filter = feature_grade_df[
        (feature_grade_df["grade"] == "A")
        & (feature_grade_df["role"] == "CONDITIONAL_FILTER")
        ]

    b_test = feature_grade_df[feature_grade_df["grade"] == "B"]
    c_drop = feature_grade_df[feature_grade_df["grade"] == "C"]

    groups = [
        ("S_CORE_SIGNAL", s_core),
        ("S_REGIME_FILTER", s_regime),
        ("A_CONDITIONAL_FILTERS", a_filter),
        ("B_TEST_CANDIDATES", b_test),
        ("C_DROP_CANDIDATES", c_drop),
    ]

    lines = []

    for name, g in groups:
        lines.append(f"{name} = [")
        for f in g["feature"].tolist():
            lines.append(f'    "{f}",')
        lines.append("]\n")

    path = os.path.join(out_dir, "13_recommended_feature_roles.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n[RECOMMENDED FEATURE ROLES]")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="feature_selector_v36_custom_avg60_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=600)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-n-usage", type=int, default=120)

    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--allow-correlated-in-rule", action="store_true")

    parser.add_argument("--target-avg-valid-precision", type=float, default=0.60)
    parser.add_argument("--portfolio-min-rule-count", type=int, default=10)
    parser.add_argument("--portfolio-max-rule-count", type=int, default=80)
    parser.add_argument("--portfolio-min-valid-count", type=int, default=25)

    # 중복 룰 방지 옵션
    parser.add_argument("--max-same-feature-set", type=int, default=3)
    parser.add_argument("--max-same-rule-family", type=int, default=2)
    parser.add_argument("--max-same-feature-pair", type=int, default=8)
    parser.add_argument("--portfolio-min-precision-floor", type=float, default=0.55)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    date_col = args.date_col or find_date_col(df)

    df = prepare_df(
        df=df,
        target_col=args.target,
        date_col=date_col,
    )

    features = [
        f for f in DEFAULT_FEATURES
        if f in df.columns and f not in EXCLUDE_FEATURES
    ]

    missing = [f for f in DEFAULT_FEATURES if f not in df.columns]

    if missing:
        print("[WARN] missing features skipped:", missing)

    if not features:
        raise ValueError("No usable features found.")

    train, valid = split_train_valid(df, args.valid_ratio)

    print("=" * 80)
    print("[INFO] rows:", len(df))
    print("[INFO] train rows:", len(train), "target_rate:", train[args.target].mean())
    print("[INFO] valid rows:", len(valid), "target_rate:", valid[args.target].mean())
    print("[INFO] date_col:", date_col)
    print("[INFO] features:", features)
    print("[SCRIPT_VERSION] v36_custom_avg60_user_feature_pool")
    print("[INFO] goal: selected rule portfolio avg valid precision >= ", args.target_avg_valid_precision)
    print("=" * 80)

    corr_mat, high_corr_df, corr_pairs = build_corr_report(
        train=train,
        features=features,
        corr_threshold=args.corr_threshold,
    )

    corr_mat.to_csv(
        os.path.join(args.out, "00_corr_matrix_spearman.csv"),
        encoding="utf-8-sig",
    )

    high_corr_df.to_csv(
        os.path.join(args.out, "00_high_corr_pairs.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    single_df = single_feature_report(train, features, args.target)

    single_df.to_csv(
        os.path.join(args.out, "01_single_feature_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    profiles = make_profiles(args)

    all_rules = []
    all_raw_rules = []

    block_correlated_in_rule = not args.allow_correlated_in_rule

    for profile in profiles:
        print("=" * 80)
        print("[PROFILE]", profile["name"])
        print("=" * 80)

        rules, raw_rules = search_rules_train_only(
            train=train,
            valid=valid,
            features=features,
            target_col=args.target,
            profile=profile,
            corr_pairs=corr_pairs,
            block_correlated_in_rule=block_correlated_in_rule,
        )

        all_rules.extend(rules)
        all_raw_rules.extend(raw_rules)

    rules_df = rules_to_df(all_rules)
    raw_rules_df = rules_to_df(all_raw_rules)

    rules_df.to_csv(
        os.path.join(args.out, "02_all_final_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    raw_rules_df.to_csv(
        os.path.join(args.out, "03_all_raw_beam_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if len(rules_df) == 0:
        print("[WARN] No final rules found.")
        return

    selected_rules_df = select_avg60_rule_portfolio(
        rules_df=rules_df,
        target_avg_precision=args.target_avg_valid_precision,
        min_rule_count=args.portfolio_min_rule_count,
        max_rule_count=args.portfolio_max_rule_count,
        min_valid_count=args.portfolio_min_valid_count,
        max_same_feature_set=args.max_same_feature_set,
        max_same_rule_family=args.max_same_rule_family,
        max_same_feature_pair=args.max_same_feature_pair,
        min_precision_floor=args.portfolio_min_precision_floor,
    )

    selected_rules_df.to_csv(
        os.path.join(args.out, "04_selected_avg60_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_portfolio_summary(
        selected_df=selected_rules_df,
        out_dir=args.out,
    )

    if len(selected_rules_df) == 0:
        print("[WARN] No avg60 rule portfolio found. Falling back to all final rules.")
        rules_for_analysis = rules_df.copy()
    else:
        rules_for_analysis = selected_rules_df.copy()

    usage_df = feature_usage_from_rules(
        rules_df=rules_for_analysis,
        features=features,
        top_n=args.top_n_usage,
    )

    usage_df.to_csv(
        os.path.join(args.out, "05_feature_usage_avg60_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    direction_df = direction_stability_report(
        rules_df=rules_for_analysis,
        features=features,
        top_n=args.top_n_usage,
    )

    direction_df.to_csv(
        os.path.join(args.out, "06_feature_direction_avg60_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    monthly_df = monthly_rule_stability(
        valid=valid,
        rules_df=rules_for_analysis,
        target_col=args.target,
    )

    monthly_df.to_csv(
        os.path.join(args.out, "07_monthly_rule_stability_avg60.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    contrib_df = conditional_feature_contribution(
        train=train,
        valid=valid,
        rules_df=rules_for_analysis,
        target_col=args.target,
    )

    contrib_df.to_csv(
        os.path.join(args.out, "08_conditional_feature_contribution_avg60.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    conditional_summary_df = summarize_conditional_contribution(contrib_df)

    conditional_summary_df.to_csv(
        os.path.join(args.out, "09_conditional_feature_contribution_summary_avg60.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    feature_grade_df = grade_features(
        features=features,
        single_df=single_df,
        usage_df=usage_df,
        direction_df=direction_df,
        conditional_summary_df=conditional_summary_df,
    )

    feature_grade_df.to_csv(
        os.path.join(args.out, "10_feature_usefulness_grade_avg60.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    write_feature_grade_text(feature_grade_df, args.out)

    print("\n[AVG60 RULE PORTFOLIO]")
    if len(selected_rules_df):
        print("rule_count:", len(selected_rules_df))
        print("avg_valid_precision:", selected_rules_df["valid_precision"].mean())
        print("avg_valid_lift:", selected_rules_df["valid_lift"].mean())

        if "portfolio_unique_feature_sets" in selected_rules_df.columns:
            print("unique_feature_sets:", selected_rules_df["portfolio_unique_feature_sets"].iloc[0])
            print("unique_rule_families:", selected_rules_df["portfolio_unique_rule_families"].iloc[0])

        show_cols = [
            "profile",
            "rank",
            "rule",
            "features",
            "train_count",
            "train_precision",
            "valid_count",
            "valid_precision",
            "valid_lift",
            "pass_valid_60",
            "feature_set_key",
            "rule_family_key",
        ]
        show_cols = [c for c in show_cols if c in selected_rules_df.columns]
        print(selected_rules_df[show_cols].head(50).to_string(index=False))
    else:
        print("[NONE]")

    print("\n[FEATURE USEFULNESS GRADES - AVG60]")
    show_cols = [
        "feature",
        "grade",
        "role",
        "action",
        "score",
        "reason",
        "total_usage_count",
        "avg_valid_precision_when_used",
        "avg_valid_lift_when_used",
        "best_valid_precision_when_used",
        "pass_valid_60_count",
        "auc_oriented",
        "auc_direction",
        "best_bin",
        "best_bin_lift",
        "dominant_op",
        "direction_ok",
        "conditional_rules_used",
        "mean_valid_precision_gain",
        "positive_gain_rate",
        "max_valid_precision_gain",
    ]
    show_cols = [c for c in show_cols if c in feature_grade_df.columns]
    print(feature_grade_df[show_cols].to_string(index=False))

    print("=" * 80)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()


# ============================================================
# 실행 예시
# ============================================================
"""
기본 실행:
python low\feature_selector_v36_custom_avg60.py ^
  --csv csv\low_result_7_v2_desc.csv ^
  --out feature_selector_v36_custom_avg60_out ^
  --date-col today ^
  --max-depth 5 ^
  --beam-width 600 ^
  --top-k 200 ^
  --top-n-usage 120 ^
  --corr-threshold 0.90 ^
  --target-avg-valid-precision 0.60 ^
  --portfolio-min-rule-count 10 ^
  --portfolio-max-rule-count 80 ^
  --portfolio-min-valid-count 25 ^
  --max-same-feature-set 3 ^
  --max-same-rule-family 2 ^
  --max-same-feature-pair 8 ^
  --portfolio-min-precision-floor 0.55

더 넓게:
python low\feature_selector_v36_custom_avg60.py ^
  --csv csv\low_result_7_v2_desc.csv ^
  --out feature_selector_v36_custom_avg60_out_wide ^
  --date-col today ^
  --max-depth 6 ^
  --beam-width 1000 ^
  --top-k 250 ^
  --top-n-usage 150 ^
  --corr-threshold 0.90 ^
  --target-avg-valid-precision 0.60 ^
  --portfolio-min-rule-count 10 ^
  --portfolio-max-rule-count 100 ^
  --portfolio-min-valid-count 25 ^
  --max-same-feature-set 4 ^
  --max-same-rule-family 2 ^
  --max-same-feature-pair 10 ^
  --portfolio-min-precision-floor 0.55
"""