import argparse
import hashlib
import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

"""
- v2 기능 포함
- room_to_20d_high / room_to_60d_high / rebound_vs_prior_drop 구간형 조건 허용
- 단순화 강화
- OR 룰셋 평가 추가
- 12_or_rule_sets.csv 출력

단일 룰 여러 개를 OR로 묶어 평가
"""

TARGET_COL = "target_before_stop_7"


DEFAULT_FEATURES = [
    # 핵심
    "room_to_20d_high",
    "room_to_60d_high",
    "upper_wick_ratio",
    "today_pct",
    "vol5",

    # 보조
    "today_tr_val_eok",
    "body_ratio",
    "rebound_from_7d_low",

    # 상관쌍 중 우선 선택
    "intraday_return",
    "ma5_chg_rate",

    # 조건부
    "price_power_value",
    "body_value_power",
    "rebound_vs_prior_drop",
    "max_drop_7d",

    # 시장 피쳐는 실험용에 가깝지만 일단 유지
    "market_today_pct",
]


ALLOWED_OPS = {
    "room_to_20d_high": [">=", "<="],
    "room_to_60d_high": [">=", "<="],
    "upper_wick_ratio": ["<="],
    "today_pct": [">="],
    "vol5": [">="],

    "today_tr_val_eok": [">=", "<="],
    "body_ratio": [">="],
    "rebound_from_7d_low": [">="],

    "intraday_return": [">="],
    "ma5_chg_rate": [">=", "<="],

    "price_power_value": [">="],
    "body_value_power": [">="],
    "rebound_vs_prior_drop": [">=", "<="],
    "max_drop_7d": ["<="],

    "market_today_pct": [">="],
}


# 같은 룰 안에서 상한/하한을 동시에 허용할 피쳐
INTERVAL_FEATURES = {
    "room_to_20d_high",
    "room_to_60d_high",
    "rebound_vs_prior_drop",
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
    train_monthly: Dict
    valid_monthly: Dict
    score: float
    train_mask_key: str
    valid_mask_key: str

    def name(self) -> str:
        return " AND ".join([a.name() for a in self.atoms])

    def features(self) -> List[str]:
        return sorted(set(a.feature for a in self.atoms))


# ============================================================
# 기본 유틸
# ============================================================

def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "today", "date", "Date", "datetime", "trade_date",
        "trd_date", "일자", "날짜", "기준일", "ymd", "YMD",
    ]:
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


def split_train_valid_by_date(df: pd.DataFrame, valid_ratio: float):
    split_idx = int(len(df) * (1 - valid_ratio))
    train = df.iloc[:split_idx].copy().reset_index(drop=True)
    valid = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train, valid


def wilson_lcb(success: int, n: int, z: float = 1.64) -> float:
    if n <= 0:
        return np.nan

    p = success / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    return (centre - margin) / denom


def calc_metrics(y, mask) -> Dict:
    y = np.asarray(y).astype(int)
    mask = np.asarray(mask).astype(bool)

    total_count = len(y)
    total_target = int(y.sum())
    base_rate = total_target / total_count if total_count else np.nan

    selected_count = int(mask.sum())

    if selected_count == 0:
        return {
            "total_count": total_count,
            "total_target": total_target,
            "base_rate": base_rate,
            "selected_count": 0,
            "selected_target": 0,
            "precision": np.nan,
            "precision_lcb": np.nan,
            "lift": np.nan,
            "recall": 0.0,
            "coverage": 0.0,
        }

    selected_target = int(y[mask].sum())
    precision = selected_target / selected_count
    precision_lcb = wilson_lcb(selected_target, selected_count)
    lift = precision / base_rate if base_rate and base_rate > 0 else np.nan
    recall = selected_target / total_target if total_target else 0.0
    coverage = selected_count / total_count if total_count else 0.0

    return {
        "total_count": total_count,
        "total_target": total_target,
        "base_rate": base_rate,
        "selected_count": selected_count,
        "selected_target": selected_target,
        "precision": precision,
        "precision_lcb": precision_lcb,
        "lift": lift,
        "recall": recall,
        "coverage": coverage,
    }


def mask_hash(mask: np.ndarray) -> str:
    packed = np.packbits(mask.astype(np.uint8))
    return hashlib.md5(packed.tobytes()).hexdigest()


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


def apply_or_rules(df: pd.DataFrame, rules: List[Rule]) -> np.ndarray:
    if not rules:
        return np.zeros(len(df), dtype=bool)

    mask = np.zeros(len(df), dtype=bool)

    for r in rules:
        mask |= apply_rule(df, r.atoms)

    return mask


# ============================================================
# 상관 피쳐
# ============================================================

def build_corr_pairs(df: pd.DataFrame, features: List[str], corr_threshold: float):
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
# threshold 후보
# ============================================================

def default_extra_thresholds() -> Dict[str, List[Tuple[str, float]]]:
    return {
        "room_to_20d_high": [
            (">=", -10.0), (">=", -5.0), (">=", 0.0), (">=", 3.0), (">=", 5.0),
            ("<=", -5.0), ("<=", 0.0), ("<=", 3.0), ("<=", 4.4438), ("<=", 5.0),
            ("<=", 10.0), ("<=", 20.0), ("<=", 30.0),
        ],
        "room_to_60d_high": [
            (">=", -10.0), (">=", -5.0), (">=", 0.0), (">=", 3.0), (">=", 5.0),
            ("<=", -5.0), ("<=", 0.0), ("<=", 5.0), ("<=", 10.0),
            ("<=", 12.2626), ("<=", 18.9896), ("<=", 20.0), ("<=", 30.0),
            ("<=", 50.0),
        ],
        "upper_wick_ratio": [
            ("<=", 0.0), ("<=", 0.02), ("<=", 0.04), ("<=", 0.064),
            ("<=", 0.087), ("<=", 0.1), ("<=", 0.111), ("<=", 0.158),
            ("<=", 0.21), ("<=", 0.31),
        ],
        "today_pct": [
            (">=", 5.0), (">=", 8.0), (">=", 10.0), (">=", 11.253),
            (">=", 13.0), (">=", 15.0), (">=", 15.86), (">=", 18.0),
            (">=", 20.0), (">=", 22.2708), (">=", 25.0),
        ],
        "vol5": [
            (">=", 4.0), (">=", 5.0), (">=", 6.0), (">=", 6.507),
            (">=", 8.0), (">=", 8.467), (">=", 10.0), (">=", 10.5478),
            (">=", 10.5491), (">=", 13.4364),
        ],
        "today_tr_val_eok": [
            (">=", 3.0), (">=", 5.0), (">=", 8.0), (">=", 10.0),
            (">=", 12.4428), (">=", 15.3325), (">=", 20.0),
            (">=", 23.4034), (">=", 50.0), (">=", 100.0),
            ("<=", 300.0), ("<=", 500.0), ("<=", 841.482), ("<=", 1391.67),
        ],
        "body_ratio": [
            (">=", 0.6), (">=", 0.7), (">=", 0.706), (">=", 0.8),
            (">=", 0.839), (">=", 0.9),
        ],
        "rebound_from_7d_low": [
            (">=", 25.0), (">=", 30.0), (">=", 32.9122), (">=", 35.0),
            (">=", 40.0), (">=", 40.7563), (">=", 45.0), (">=", 50.0),
            (">=", 55.0), (">=", 60.0),
        ],
        "intraday_return": [
            (">=", 4.0), (">=", 5.0), (">=", 6.0), (">=", 8.0),
            (">=", 10.0), (">=", 15.0), (">=", 18.6578),
        ],
        "ma5_chg_rate": [
            (">=", 3.0), (">=", 5.0), (">=", 5.8628), (">=", 8.0),
            (">=", 10.0), ("<=", -3.0), ("<=", -5.0),
        ],
        "price_power_value": [
            (">=", 20.0), (">=", 40.0), (">=", 60.0), (">=", 80.0),
            (">=", 100.0),
        ],
        "body_value_power": [
            (">=", 5.0), (">=", 10.0), (">=", 15.0), (">=", 20.0),
            (">=", 30.0), (">=", 40.0),
        ],
        "rebound_vs_prior_drop": [
            (">=", 1.0), (">=", 2.0), (">=", 3.0), (">=", 5.0),
            ("<=", 5.0), ("<=", 10.0), ("<=", 20.0),
        ],
        "max_drop_7d": [
            ("<=", -3.54), ("<=", -3.7931), ("<=", -5.0),
            ("<=", -7.0), ("<=", -10.0),
        ],
        "market_today_pct": [
            (">=", -2.0), (">=", -1.0), (">=", 0.0), (">=", 0.5), (">=", 1.0),
        ],
    }


def make_atoms(train, features, quantiles, extra_thresholds, allowed_ops):
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
            for op in ["<=", ">="]:
                if f in allowed_ops and op not in allowed_ops[f]:
                    continue
                atoms.append(Atom(f, op, q))

        for op, th in extra_thresholds.get(f, []):
            if f in allowed_ops and op not in allowed_ops[f]:
                continue
            atoms.append(Atom(f, op, float(th)))

    unique = {}

    for a in atoms:
        unique[(a.feature, a.op, round(a.threshold, 10))] = a

    return list(unique.values())


# ============================================================
# 같은 피쳐 구간형 조건 허용
# ============================================================

def can_add_atom(base_atoms: Tuple[Atom, ...], atom: Atom, corr_pairs, args) -> bool:
    used_features = [a.feature for a in base_atoms]
    same_feature_atoms = [a for a in base_atoms if a.feature == atom.feature]

    # 일반 피쳐는 한 룰에 한 번만
    if atom.feature not in INTERVAL_FEATURES:
        if atom.feature in used_features:
            return False

        if not args.allow_correlated_in_rule:
            if has_correlated_pair(set(used_features), atom.feature, corr_pairs):
                return False

        return True

    # 구간형 피쳐는 최대 2개까지 허용
    if len(same_feature_atoms) >= 2:
        return False

    # 같은 op 두 번은 불허
    if any(a.op == atom.op for a in same_feature_atoms):
        return False

    # 같은 피쳐의 하한/상한 조합이 모순이면 불허
    candidate = list(same_feature_atoms) + [atom]
    lowers = [a.threshold for a in candidate if a.op == ">="]
    uppers = [a.threshold for a in candidate if a.op == "<="]

    if lowers and uppers:
        if max(lowers) > min(uppers):
            return False

    # 구간형 피쳐는 자기 자신 상관 체크 제외, 다른 피쳐와는 체크
    if not args.allow_correlated_in_rule:
        other_features = set(f for f in used_features if f != atom.feature)
        if has_correlated_pair(other_features, atom.feature, corr_pairs):
            return False

    return True


def canonical_atoms_key(atoms: Tuple[Atom, ...]):
    return tuple(
        sorted(
            [(a.feature, a.op, round(a.threshold, 10)) for a in atoms],
            key=lambda x: (x[0], x[1], x[2]),
        )
    )


def canonicalize_atoms(atoms: Tuple[Atom, ...]) -> Tuple[Atom, ...]:
    return tuple(
        sorted(
            atoms,
            key=lambda a: (a.feature, a.op, a.threshold),
        )
    )


# ============================================================
# 월별 평가
# ============================================================

def monthly_summary(
        df,
        target_col,
        atoms,
        min_month_count,
        pass_precision,
        pass_lift,
        crash_precision,
):
    df = df.reset_index(drop=True).copy()

    y_all = df[target_col].values
    mask_all = apply_rule(df, atoms)

    rows = []

    for month, g in df.groupby("month", sort=True):
        idx = g.index.to_numpy()
        m = calc_metrics(y_all[idx], mask_all[idx])
        m["month"] = month
        rows.append(m)

    monthly_df = pd.DataFrame(rows)
    usable = monthly_df[monthly_df["selected_count"] >= min_month_count].copy()

    if len(usable) == 0:
        return monthly_df, {
            "n_months": monthly_df["month"].nunique() if len(monthly_df) else 0,
            "n_usable_months": 0,
            "mean_month_precision": np.nan,
            "median_month_precision": np.nan,
            "min_month_precision": np.nan,
            "std_month_precision": np.nan,
            "mean_month_lift": np.nan,
            "min_month_lift": np.nan,
            "mean_month_count": np.nan,
            "total_month_count": int(monthly_df["selected_count"].sum()) if len(monthly_df) else 0,
            "pass_month_rate": 0.0,
            "bad_month_count": len(monthly_df),
            "crash_month_count": len(monthly_df),
        }

    pass_mask = (
            (usable["precision"] >= pass_precision) &
            (usable["lift"] >= pass_lift)
    )

    crash_mask = usable["precision"] < crash_precision

    std_p = usable["precision"].std()

    if not np.isfinite(std_p):
        std_p = 0.0

    return monthly_df, {
        "n_months": monthly_df["month"].nunique(),
        "n_usable_months": len(usable),
        "mean_month_precision": usable["precision"].mean(),
        "median_month_precision": usable["precision"].median(),
        "min_month_precision": usable["precision"].min(),
        "std_month_precision": std_p,
        "mean_month_lift": usable["lift"].mean(),
        "min_month_lift": usable["lift"].min(),
        "mean_month_count": usable["selected_count"].mean(),
        "total_month_count": int(monthly_df["selected_count"].sum()),
        "pass_month_rate": pass_mask.mean(),
        "bad_month_count": int((~pass_mask).sum()),
        "crash_month_count": int(crash_mask.sum()),
    }


def monthly_summary_from_mask(
        df,
        target_col,
        mask,
        min_month_count,
        pass_precision,
        pass_lift,
        crash_precision,
):
    df = df.reset_index(drop=True).copy()
    y_all = df[target_col].values
    mask_all = np.asarray(mask).astype(bool)

    rows = []

    for month, g in df.groupby("month", sort=True):
        idx = g.index.to_numpy()
        m = calc_metrics(y_all[idx], mask_all[idx])
        m["month"] = month
        rows.append(m)

    monthly_df = pd.DataFrame(rows)
    usable = monthly_df[monthly_df["selected_count"] >= min_month_count].copy()

    if len(usable) == 0:
        return monthly_df, {
            "n_usable_months": 0,
            "mean_month_precision": np.nan,
            "min_month_precision": np.nan,
            "pass_month_rate": 0.0,
            "crash_month_count": len(monthly_df),
        }

    pass_mask = (
            (usable["precision"] >= pass_precision) &
            (usable["lift"] >= pass_lift)
    )
    crash_mask = usable["precision"] < crash_precision

    return monthly_df, {
        "n_usable_months": len(usable),
        "mean_month_precision": usable["precision"].mean(),
        "min_month_precision": usable["precision"].min(),
        "pass_month_rate": pass_mask.mean(),
        "crash_month_count": int(crash_mask.sum()),
    }


def eval_rule(df, target_col, atoms, args):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics(df[target_col].values, mask)

    monthly_df, mon = monthly_summary(
        df=df,
        target_col=target_col,
        atoms=atoms,
        min_month_count=args.min_month_count,
        pass_precision=args.month_pass_precision,
        pass_lift=args.month_pass_lift,
        crash_precision=args.month_crash_precision,
    )

    return metrics, mon, monthly_df, mask


# ============================================================
# 점수 함수
# ============================================================

def beam_score(metrics, mon, args):
    count = metrics["selected_count"]
    precision = metrics["precision"]
    lift = metrics["lift"]

    if count < args.beam_min_count:
        return -1e18

    if not np.isfinite(precision) or not np.isfinite(lift):
        return -1e18

    if lift < args.beam_min_lift:
        return -1e18

    mean_p = mon["mean_month_precision"]
    if not np.isfinite(mean_p):
        mean_p = 0.0

    pass_rate = mon["pass_month_rate"]
    if not np.isfinite(pass_rate):
        pass_rate = 0.0

    return (
            precision * 45
            + lift * 12
            + mean_p * 35
            + pass_rate * 15
            + math.log1p(count) * 1.5
            - mon.get("crash_month_count", 0) * 10
    )


def final_score(train_m, valid_m, train_mon, valid_mon, args):
    train_count = train_m["selected_count"]
    valid_count = valid_m["selected_count"]

    train_p = train_m["precision"]
    valid_p = valid_m["precision"]
    train_lift = train_m["lift"]
    valid_lift = valid_m["lift"]

    if train_count < args.min_train_count:
        return -1e18

    if valid_count < args.min_valid_count:
        return -1e18

    if not all(np.isfinite(x) for x in [train_p, valid_p, train_lift, valid_lift]):
        return -1e18

    if train_lift < args.min_train_lift:
        return -1e18

    if valid_lift < args.min_valid_lift:
        return -1e18

    if valid_p < args.min_valid_precision:
        return -1e18

    if train_p < args.min_train_precision:
        return -1e18

    gap = abs(train_p - valid_p)

    if gap > args.max_precision_gap:
        return -1e18

    if valid_mon["n_usable_months"] < args.min_valid_usable_months:
        return -1e18

    if valid_mon["pass_month_rate"] < args.min_valid_pass_month_rate:
        return -1e18

    if np.isfinite(valid_mon["min_month_precision"]):
        if valid_mon["min_month_precision"] < args.min_valid_month_min_precision:
            return -1e18

    if valid_mon.get("crash_month_count", 0) > args.max_valid_crash_months:
        return -1e18

    valid_lcb = valid_m["precision_lcb"]

    if np.isfinite(valid_lcb) and valid_lcb < args.min_valid_lcb:
        return -1e18

    valid_std = valid_mon["std_month_precision"]
    if not np.isfinite(valid_std):
        valid_std = 0.0

    valid_min_p = valid_mon["min_month_precision"]
    if not np.isfinite(valid_min_p):
        valid_min_p = 0.0

    precision_bonus_70 = max(0.0, valid_p - 0.70) * 250
    precision_bonus_65 = max(0.0, valid_p - 0.65) * 120
    count_bonus = math.log1p(valid_count) * args.valid_count_weight

    score = (
            valid_p * 140
            + valid_lcb * 80
            + valid_lift * 25
            + valid_mon["mean_month_precision"] * 70
            + valid_min_p * 55
            + valid_mon["pass_month_rate"] * 70
            + train_p * 20
            + count_bonus
            + precision_bonus_70
            + precision_bonus_65
            - gap * 90
            - valid_std * 60
            - valid_mon["bad_month_count"] * 4
            - valid_mon.get("crash_month_count", 0) * 25
    )

    return score


# ============================================================
# 룰 탐색 / 중복 제거
# ============================================================

def is_near_duplicate(new_rule: Rule, selected: List[Rule], args) -> bool:
    for r in selected:
        if new_rule.train_mask_key == r.train_mask_key:
            return True

        if new_rule.valid_mask_key == r.valid_mask_key:
            return True

        c1 = new_rule.valid_metrics["selected_count"]
        c0 = r.valid_metrics["selected_count"]

        p1 = new_rule.valid_metrics["precision"]
        p0 = r.valid_metrics["precision"]

        if c0 == 0 or c1 == 0:
            continue

        count_change = abs(c1 - c0) / max(c0, 1)
        precision_gain = p1 - p0 if np.isfinite(p1) and np.isfinite(p0) else 0.0

        f_new = set(new_rule.features())
        f_old = set(r.features())

        if count_change <= args.dup_count_tol and precision_gain <= args.dup_precision_gain:
            if f_old.issubset(f_new) or f_new.issubset(f_old):
                return True

    return False


def search_rules(train, valid, atoms, corr_pairs, args):
    beam = [tuple()]
    selected_candidates = []
    seen_rules = set()
    seen_train_masks = set()

    for depth in range(1, args.max_depth + 1):
        print(f"[INFO] searching depth={depth}")

        beam_candidates = []

        for base_atoms in beam:
            for atom in atoms:
                if not can_add_atom(base_atoms, atom, corr_pairs, args):
                    continue

                new_atoms = canonicalize_atoms(tuple(list(base_atoms) + [atom]))
                key = canonical_atoms_key(new_atoms)

                if key in seen_rules:
                    continue

                seen_rules.add(key)

                train_m, train_mon, _, train_mask = eval_rule(
                    train,
                    args.target,
                    new_atoms,
                    args,
                )

                bscore = beam_score(train_m, train_mon, args)

                if bscore <= -1e17:
                    continue

                train_key = mask_hash(train_mask)

                if train_key in seen_train_masks and len(new_atoms) > 1:
                    continue

                seen_train_masks.add(train_key)

                valid_m, valid_mon, _, valid_mask = eval_rule(
                    valid,
                    args.target,
                    new_atoms,
                    args,
                )

                fscore = final_score(train_m, valid_m, train_mon, valid_mon, args)

                rule = Rule(
                    atoms=new_atoms,
                    train_metrics=train_m,
                    valid_metrics=valid_m,
                    train_monthly=train_mon,
                    valid_monthly=valid_mon,
                    score=fscore if fscore > -1e17 else bscore,
                    train_mask_key=train_key,
                    valid_mask_key=mask_hash(valid_mask),
                )

                beam_candidates.append(rule)

                if fscore > -1e17:
                    selected_candidates.append(rule)

        if not beam_candidates:
            print(f"[WARN] no beam candidates at depth={depth}")
            break

        beam_candidates = sorted(
            beam_candidates,
            key=lambda r: r.score,
            reverse=True,
        )

        beam = [r.atoms for r in beam_candidates[:args.beam_width]]

        print(
            f"[INFO] depth={depth}, beam_candidates={len(beam_candidates)}, "
            f"selected_candidates={len(selected_candidates)}, kept_beam={len(beam)}"
        )

    selected_candidates = sorted(
        selected_candidates,
        key=lambda r: r.score,
        reverse=True,
    )

    deduped = []

    for r in selected_candidates:
        if is_near_duplicate(r, deduped, args):
            continue

        deduped.append(r)

        if len(deduped) >= args.top_k:
            break

    return deduped


# ============================================================
# 룰 단순화
# ============================================================

def simplify_rule_by_dropping_atoms(rule: Rule, train, valid, args) -> Rule:
    current_rule = rule
    improved = True

    while improved and len(current_rule.atoms) > 1:
        improved = False
        best_candidate = None
        best_tuple = None

        for i in range(len(current_rule.atoms)):
            candidate_atoms = tuple(
                a for j, a in enumerate(current_rule.atoms) if j != i
            )

            train_m, train_mon, _, train_mask = eval_rule(
                train,
                args.target,
                candidate_atoms,
                args,
            )

            valid_m, valid_mon, _, valid_mask = eval_rule(
                valid,
                args.target,
                candidate_atoms,
                args,
            )

            score = final_score(train_m, valid_m, train_mon, valid_mon, args)

            if score <= -1e17:
                continue

            old_vp = current_rule.valid_metrics["precision"]
            new_vp = valid_m["precision"]
            old_vc = current_rule.valid_metrics["selected_count"]
            new_vc = valid_m["selected_count"]

            if not np.isfinite(old_vp) or not np.isfinite(new_vp):
                continue

            precision_drop = old_vp - new_vp
            count_gain = new_vc - old_vc

            # 기존보다 강화:
            # precision 1%p 이하 하락은 허용.
            # count가 늘면 더 적극적으로 단순화.
            if precision_drop <= args.simplify_max_precision_drop:
                candidate = Rule(
                    atoms=candidate_atoms,
                    train_metrics=train_m,
                    valid_metrics=valid_m,
                    train_monthly=train_mon,
                    valid_monthly=valid_mon,
                    score=score,
                    train_mask_key=mask_hash(train_mask),
                    valid_mask_key=mask_hash(valid_mask),
                )

                compare_tuple = (
                    count_gain,
                    -precision_drop,
                    -len(candidate_atoms),
                    score,
                )

                if best_tuple is None or compare_tuple > best_tuple:
                    best_tuple = compare_tuple
                    best_candidate = candidate

        if best_candidate is not None:
            current_rule = best_candidate
            improved = True

    return current_rule


# ============================================================
# Walk-forward
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
                "valid_precision_lcb": valid_m["precision_lcb"],
                "valid_lift": valid_m["lift"],
                "valid_base_rate": valid_m["base_rate"],
            })

    return pd.DataFrame(rows)


def summarize_wf(
        wf_df: pd.DataFrame,
        min_count: int,
        min_precision: float,
        min_lift: float,
        crash_precision: float,
) -> pd.DataFrame:
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
                "crash_split_count": len(g),
                "wf_score": -999,
            })
            continue

        pass_mask = (
                (usable["valid_precision"] >= min_precision) &
                (usable["valid_lift"] >= min_lift)
        )

        crash_mask = usable["valid_precision"] < crash_precision

        std_p = usable["valid_precision"].std()
        if not np.isfinite(std_p):
            std_p = 0.0

        wf_score = (
                usable["valid_precision"].mean() * 120
                + usable["valid_precision"].min() * 90
                + usable["valid_lift"].mean() * 20
                + pass_mask.mean() * 70
                + math.log1p(usable["valid_count"].sum()) * 5
                - std_p * 70
                - int((~pass_mask).sum()) * 5
                - int(crash_mask.sum()) * 30
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
            "total_valid_count": int(usable["valid_count"].sum()),
            "pass_split_rate": pass_mask.mean(),
            "bad_split_count": int((~pass_mask).sum()),
            "crash_split_count": int(crash_mask.sum()),
            "wf_score": wf_score,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("wf_score", ascending=False)

    return out


# ============================================================
# OR 룰셋 평가
# ============================================================

def eval_or_rule_set(train, valid, df_all, date_col, rules, target_col, args):
    train_mask = apply_or_rules(train, rules)
    valid_mask = apply_or_rules(valid, rules)

    train_m = calc_metrics(train[target_col].values, train_mask)
    valid_m = calc_metrics(valid[target_col].values, valid_mask)

    _, valid_mon = monthly_summary_from_mask(
        df=valid,
        target_col=target_col,
        mask=valid_mask,
        min_month_count=args.min_month_count,
        pass_precision=args.month_pass_precision,
        pass_lift=args.month_pass_lift,
        crash_precision=args.month_crash_precision,
    )

    # OR walk-forward
    df_all = df_all.sort_values(date_col).reset_index(drop=True)
    months = sorted(df_all["month"].unique())

    if len(months) <= args.wf_min_train_months:
        return train_m, valid_m, valid_mon, {}

    start_idx = max(args.wf_min_train_months, len(months) - args.wf_splits)

    wf_rows = []

    for i in range(start_idx, len(months)):
        train_months = months[:i]
        valid_month = months[i]

        wf_valid = df_all[df_all["month"] == valid_month].copy().reset_index(drop=True)
        wf_mask = apply_or_rules(wf_valid, rules)
        wf_m = calc_metrics(wf_valid[target_col].values, wf_mask)
        wf_m["valid_month"] = valid_month
        wf_rows.append(wf_m)

    wf_df = pd.DataFrame(wf_rows)
    usable = wf_df[wf_df["selected_count"] >= args.wf_min_count].copy()

    if len(usable) == 0:
        wf_summary = {
            "wf_n_splits": len(wf_df),
            "wf_n_usable_splits": 0,
            "wf_mean_precision": np.nan,
            "wf_min_precision": np.nan,
            "wf_total_count": int(wf_df["selected_count"].sum()) if len(wf_df) else 0,
            "wf_pass_split_rate": 0.0,
            "wf_crash_split_count": len(wf_df),
        }
    else:
        pass_mask = (
                (usable["precision"] >= args.wf_min_precision) &
                (usable["lift"] >= args.wf_min_lift)
        )
        crash_mask = usable["precision"] < args.wf_crash_precision

        wf_summary = {
            "wf_n_splits": len(wf_df),
            "wf_n_usable_splits": len(usable),
            "wf_mean_precision": usable["precision"].mean(),
            "wf_min_precision": usable["precision"].min(),
            "wf_total_count": int(usable["selected_count"].sum()),
            "wf_pass_split_rate": pass_mask.mean(),
            "wf_crash_split_count": int(crash_mask.sum()),
        }

    return train_m, valid_m, valid_mon, wf_summary


def evaluate_or_rule_sets(rules, train, valid, df_all, date_col, args):
    if len(rules) == 0:
        return pd.DataFrame()

    # OR 후보는 final filter 통과 룰이 있으면 그걸 우선 사용하고,
    # 없으면 상위 룰 사용.
    candidate_rules = rules[:args.or_top_rules]

    rows = []

    for k in range(2, args.or_max_size + 1):
        for combo in itertools.combinations(candidate_rules, k):
            train_m, valid_m, valid_mon, wf_s = eval_or_rule_set(
                train=train,
                valid=valid,
                df_all=df_all,
                date_col=date_col,
                rules=list(combo),
                target_col=args.target,
                args=args,
            )

            rule_ranks = []
            rule_names = []

            for r in combo:
                # rank는 나중에 rules_to_df와 맞추기 위해 name 기준만 저장
                rule_names.append(r.name())

            valid_p = valid_m["precision"]
            valid_count = valid_m["selected_count"]
            wf_mean = wf_s.get("wf_mean_precision", np.nan)
            wf_min = wf_s.get("wf_min_precision", np.nan)
            wf_total = wf_s.get("wf_total_count", 0)
            wf_pass = wf_s.get("wf_pass_split_rate", 0.0)
            wf_crash = wf_s.get("wf_crash_split_count", 999)

            if not np.isfinite(valid_p):
                continue

            # 너무 낮은 OR 조합은 버림
            if valid_p < args.or_min_valid_precision:
                continue

            if valid_count < args.or_min_valid_count:
                continue

            if np.isfinite(wf_mean) and wf_mean < args.or_min_wf_mean_precision:
                continue

            if np.isfinite(wf_min) and wf_min < args.or_min_wf_min_precision:
                continue

            if wf_pass < args.or_min_wf_pass_split_rate:
                continue

            if wf_crash > args.or_max_wf_crash_splits:
                continue

            score = (
                    valid_p * 130
                    + math.log1p(valid_count) * args.or_count_weight
                    + (wf_mean if np.isfinite(wf_mean) else 0) * 100
                    + (wf_min if np.isfinite(wf_min) else 0) * 60
                    + math.log1p(wf_total) * 8
                    + wf_pass * 60
                    - wf_crash * 30
            )

            rows.append({
                "or_size": k,
                "rules": " || ".join(rule_names),
                "train_count": train_m["selected_count"],
                "train_precision": train_m["precision"],
                "train_lift": train_m["lift"],
                "valid_count": valid_count,
                "valid_precision": valid_p,
                "valid_lift": valid_m["lift"],
                "valid_month_min_precision": valid_mon.get("min_month_precision", np.nan),
                "valid_month_pass_rate": valid_mon.get("pass_month_rate", np.nan),
                "valid_month_crash_count": valid_mon.get("crash_month_count", np.nan),
                "wf_mean_precision": wf_mean,
                "wf_min_precision": wf_min,
                "wf_total_count": wf_total,
                "wf_pass_split_rate": wf_pass,
                "wf_crash_split_count": wf_crash,
                "or_score": score,
            })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["or_score", "valid_precision", "valid_count", "wf_mean_precision", "wf_total_count"],
            ascending=[False, False, False, False, False],
        )

    return out


# ============================================================
# 출력 변환
# ============================================================

def rules_to_df(rules: List[Rule]) -> pd.DataFrame:
    rows = []

    for rank, r in enumerate(rules, start=1):
        train_p = r.train_metrics["precision"]
        valid_p = r.valid_metrics["precision"]

        row = {
            "rank": rank,
            "rule": r.name(),
            "features": ",".join(r.features()),
            "n_features": len(r.features()),
            "score": r.score,

            "train_count": r.train_metrics["selected_count"],
            "train_target": r.train_metrics["selected_target"],
            "train_precision": train_p,
            "train_precision_lcb": r.train_metrics["precision_lcb"],
            "train_lift": r.train_metrics["lift"],
            "train_base_rate": r.train_metrics["base_rate"],

            "valid_count": r.valid_metrics["selected_count"],
            "valid_target": r.valid_metrics["selected_target"],
            "valid_precision": valid_p,
            "valid_precision_lcb": r.valid_metrics["precision_lcb"],
            "valid_lift": r.valid_metrics["lift"],
            "valid_base_rate": r.valid_metrics["base_rate"],

            "precision_gap_abs": (
                abs(train_p - valid_p)
                if np.isfinite(train_p) and np.isfinite(valid_p)
                else np.nan
            ),
        }

        for prefix, mon in [
            ("train_month", r.train_monthly),
            ("valid_month", r.valid_monthly),
        ]:
            for k, v in mon.items():
                row[f"{prefix}_{k}"] = v

        row["pass_valid_60"] = (
                row["valid_precision"] >= 0.60
                and row["valid_lift"] >= 1.45
                and row["valid_count"] >= 60
        )

        row["pass_valid_60_large"] = (
                row["valid_precision"] >= 0.60
                and row["valid_lift"] >= 1.45
                and row["valid_count"] >= 80
        )

        row["pass_valid_65"] = (
                row["valid_precision"] >= 0.65
                and row["valid_lift"] >= 1.55
                and row["valid_count"] >= 50
        )

        row["pass_valid_70"] = (
                row["valid_precision"] >= 0.70
                and row["valid_lift"] >= 1.65
                and row["valid_count"] >= 30
        )

        row["pass_valid_70_large"] = (
                row["valid_precision"] >= 0.70
                and row["valid_lift"] >= 1.65
                and row["valid_count"] >= 60
        )

        row["pass_train_valid_70"] = (
                row["train_precision"] >= 0.70
                and row["valid_precision"] >= 0.70
                and row["valid_count"] >= 30
        )

        rows.append(row)

    return pd.DataFrame(rows)


def collect_monthly_details(rules, train, valid, args):
    dfs = []

    for rank, r in enumerate(rules, start=1):
        for dataset_name, df in [("TRAIN", train), ("VALID", valid)]:
            _, _, monthly_df, _ = eval_rule(
                df,
                args.target,
                r.atoms,
                args,
            )

            monthly_df = monthly_df.copy()
            monthly_df["rank"] = rank
            monthly_df["rule"] = r.name()
            monthly_df["dataset"] = dataset_name
            dfs.append(monthly_df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def build_final_wf_filtered_rules(
        rules_df: pd.DataFrame,
        wf_summary: pd.DataFrame,
        args,
) -> pd.DataFrame:
    if len(rules_df) == 0 or len(wf_summary) == 0:
        return pd.DataFrame()

    merged = rules_df.merge(
        wf_summary,
        on=["rank", "rule"],
        how="left",
        suffixes=("", "_wf"),
    )

    out = merged[
        (merged["train_precision"] >= args.final_min_train_precision)
        & (merged["valid_precision"] >= args.final_min_valid_precision)
        & (merged["valid_count"] >= args.final_min_valid_count)
        & (merged["valid_lift"] >= args.final_min_valid_lift)
        & (merged["valid_month_min_month_precision"] >= args.final_min_valid_month_min_precision)
        & (merged["valid_month_crash_month_count"] <= args.final_max_valid_crash_months)
        & (merged["mean_valid_precision"] >= args.final_min_wf_mean_precision)
        & (merged["min_valid_precision"] >= args.final_min_wf_min_precision)
        & (merged["pass_split_rate"] >= args.final_min_wf_pass_split_rate)
        & (merged["total_valid_count"] >= args.final_min_wf_total_count)
        & (merged["crash_split_count"] <= args.final_max_wf_crash_splits)
        ].copy()

    if len(out):
        out = out.sort_values(
            ["valid_precision", "valid_count", "mean_valid_precision", "total_valid_count"],
            ascending=[False, False, False, False],
        )

    return out


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="stable_rule_miner_final_v3_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=1500)
    parser.add_argument("--top-k", type=int, default=150)

    parser.add_argument("--min-train-count", type=int, default=60)
    parser.add_argument("--min-valid-count", type=int, default=40)

    parser.add_argument("--min-train-precision", type=float, default=0.55)
    parser.add_argument("--min-valid-precision", type=float, default=0.62)

    parser.add_argument("--min-train-lift", type=float, default=1.25)
    parser.add_argument("--min-valid-lift", type=float, default=1.50)

    parser.add_argument("--min-valid-lcb", type=float, default=0.50)
    parser.add_argument("--max-precision-gap", type=float, default=0.18)

    parser.add_argument("--beam-min-count", type=int, default=60)
    parser.add_argument("--beam-min-lift", type=float, default=1.00)

    parser.add_argument("--min-month-count", type=int, default=5)
    parser.add_argument("--month-pass-precision", type=float, default=0.55)
    parser.add_argument("--month-pass-lift", type=float, default=1.20)
    parser.add_argument("--month-crash-precision", type=float, default=0.45)

    parser.add_argument("--min-valid-usable-months", type=int, default=2)
    parser.add_argument("--min-valid-pass-month-rate", type=float, default=0.20)
    parser.add_argument("--min-valid-month-min-precision", type=float, default=0.55)
    parser.add_argument("--max-valid-crash-months", type=int, default=0)

    parser.add_argument("--valid-count-weight", type=float, default=7.0)

    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--allow-correlated-in-rule", action="store_true")

    parser.add_argument("--dup-count-tol", type=float, default=0.02)
    parser.add_argument("--dup-precision-gain", type=float, default=0.01)

    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--simplify-max-precision-drop", type=float, default=0.01)

    # walk-forward
    parser.add_argument("--wf-splits", type=int, default=12)
    parser.add_argument("--wf-min-train-months", type=int, default=6)
    parser.add_argument("--wf-min-count", type=int, default=5)
    parser.add_argument("--wf-min-precision", type=float, default=0.60)
    parser.add_argument("--wf-min-lift", type=float, default=1.25)
    parser.add_argument("--wf-crash-precision", type=float, default=0.45)
    parser.add_argument("--wf-top-rules", type=int, default=150)

    # 최종 실전 후보 필터
    parser.add_argument("--final-min-train-precision", type=float, default=0.60)
    parser.add_argument("--final-min-valid-precision", type=float, default=0.70)
    parser.add_argument("--final-min-valid-count", type=int, default=50)
    parser.add_argument("--final-min-valid-lift", type=float, default=1.65)
    parser.add_argument("--final-min-valid-month-min-precision", type=float, default=0.55)
    parser.add_argument("--final-max-valid-crash-months", type=int, default=0)

    parser.add_argument("--final-min-wf-mean-precision", type=float, default=0.70)
    parser.add_argument("--final-min-wf-min-precision", type=float, default=0.55)
    parser.add_argument("--final-min-wf-pass-split-rate", type=float, default=0.70)
    parser.add_argument("--final-min-wf-total-count", type=int, default=50)
    parser.add_argument("--final-max-wf-crash-splits", type=int, default=0)

    # OR 룰셋 평가
    parser.add_argument("--or-top-rules", type=int, default=15)
    parser.add_argument("--or-max-size", type=int, default=3)
    parser.add_argument("--or-min-valid-precision", type=float, default=0.68)
    parser.add_argument("--or-min-valid-count", type=int, default=70)
    parser.add_argument("--or-min-wf-mean-precision", type=float, default=0.68)
    parser.add_argument("--or-min-wf-min-precision", type=float, default=0.50)
    parser.add_argument("--or-min-wf-pass-split-rate", type=float, default=0.60)
    parser.add_argument("--or-max-wf-crash-splits", type=int, default=1)
    parser.add_argument("--or-count-weight", type=float, default=12.0)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    date_col = args.date_col or find_date_col(df)

    if date_col is None:
        raise ValueError("date column not found. Use --date-col today")

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
    print("[INFO] goal: valid 70% + count + no monthly/wf crash + OR sets")
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
        0.975,
    ]

    atoms = make_atoms(
        train=train,
        features=features,
        quantiles=quantiles,
        extra_thresholds=default_extra_thresholds(),
        allowed_ops=ALLOWED_OPS,
    )

    pd.DataFrame([
        {
            "feature": a.feature,
            "op": a.op,
            "threshold": a.threshold,
            "atom": a.name(),
        }
        for a in atoms
    ]).to_csv(
        os.path.join(args.out, "01_atoms.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("[INFO] atoms:", len(atoms))

    rules = search_rules(
        train=train,
        valid=valid,
        atoms=atoms,
        corr_pairs=corr_pairs,
        args=args,
    )

    if args.simplify:
        simplified = []

        for r in rules:
            sr = simplify_rule_by_dropping_atoms(
                rule=r,
                train=train,
                valid=valid,
                args=args,
            )
            simplified.append(sr)

        simplified = sorted(
            simplified,
            key=lambda r: r.score,
            reverse=True,
        )

        deduped = []

        for r in simplified:
            if is_near_duplicate(r, deduped, args):
                continue

            deduped.append(r)

            if len(deduped) >= args.top_k:
                break

        rules = deduped

    rules_df = rules_to_df(rules)

    rules_df.to_csv(
        os.path.join(args.out, "02_selected_rules.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    rules_df[rules_df["pass_valid_60"]].to_csv(
        os.path.join(args.out, "03_pass_valid_60.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    rules_df[rules_df["pass_valid_65"]].to_csv(
        os.path.join(args.out, "04_pass_valid_65.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    rules_df[rules_df["pass_valid_70"]].to_csv(
        os.path.join(args.out, "05_pass_valid_70.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    rules_df[rules_df["pass_valid_70_large"]].to_csv(
        os.path.join(args.out, "06_pass_valid_70_large.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    rules_df[rules_df["pass_train_valid_70"]].to_csv(
        os.path.join(args.out, "07_pass_train_valid_70.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    monthly_df = collect_monthly_details(
        rules=rules,
        train=train,
        valid=valid,
        args=args,
    )

    monthly_df.to_csv(
        os.path.join(args.out, "08_monthly_details.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    wf_df = walk_forward_eval_fixed_rules(
        df=df,
        date_col=date_col,
        target_col=args.target,
        rules=rules[:args.wf_top_rules],
        n_splits=args.wf_splits,
        min_train_months=args.wf_min_train_months,
    )

    wf_df.to_csv(
        os.path.join(args.out, "09_walk_forward_eval.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    wf_summary = summarize_wf(
        wf_df=wf_df,
        min_count=args.wf_min_count,
        min_precision=args.wf_min_precision,
        min_lift=args.wf_min_lift,
        crash_precision=args.wf_crash_precision,
    )

    wf_summary.to_csv(
        os.path.join(args.out, "10_walk_forward_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    final_wf_filtered = build_final_wf_filtered_rules(
        rules_df=rules_df,
        wf_summary=wf_summary,
        args=args,
    )

    final_wf_filtered.to_csv(
        os.path.join(args.out, "11_final_rules_wf_filtered.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # OR 룰셋 평가
    or_df = evaluate_or_rule_sets(
        rules=rules,
        train=train,
        valid=valid,
        df_all=df,
        date_col=date_col,
        args=args,
    )

    or_df.to_csv(
        os.path.join(args.out, "12_or_rule_sets.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[SELECTED RULES]")
    show_cols = [
        "rank",
        "rule",
        "train_count",
        "train_precision",
        "train_lift",
        "valid_count",
        "valid_precision",
        "valid_precision_lcb",
        "valid_lift",
        "precision_gap_abs",
        "valid_month_n_usable_months",
        "valid_month_mean_month_precision",
        "valid_month_min_month_precision",
        "valid_month_crash_month_count",
        "valid_month_pass_month_rate",
        "pass_valid_60",
        "pass_valid_65",
        "pass_valid_70",
        "pass_valid_70_large",
        "pass_train_valid_70",
    ]

    show_cols = [c for c in show_cols if c in rules_df.columns]

    if len(rules_df):
        print(rules_df[show_cols].head(50).to_string(index=False))
    else:
        print("No selected rules found.")

    print("\n[FINAL WF FILTERED RULES]")
    if len(final_wf_filtered):
        final_cols = [
            "rank",
            "rule",
            "train_precision",
            "valid_count",
            "valid_precision",
            "valid_lift",
            "valid_month_min_month_precision",
            "mean_valid_precision",
            "min_valid_precision",
            "total_valid_count",
            "pass_split_rate",
            "crash_split_count",
        ]
        final_cols = [c for c in final_cols if c in final_wf_filtered.columns]
        print(final_wf_filtered[final_cols].head(30).to_string(index=False))
    else:
        print("No final rules passed monthly + walk-forward crash filters.")

    print("\n[OR RULE SETS]")
    if len(or_df):
        or_cols = [
            "or_size",
            "valid_count",
            "valid_precision",
            "valid_lift",
            "wf_mean_precision",
            "wf_min_precision",
            "wf_total_count",
            "wf_pass_split_rate",
            "wf_crash_split_count",
            "or_score",
            "rules",
        ]
        or_cols = [c for c in or_cols if c in or_df.columns]
        print(or_df[or_cols].head(30).to_string(index=False))
    else:
        print("No OR rule sets passed filters.")

    print("\n[SUMMARY]")
    if len(rules_df):
        print("selected rules:", len(rules_df))
        print("pass_valid_60:", int(rules_df["pass_valid_60"].sum()))
        print("pass_valid_65:", int(rules_df["pass_valid_65"].sum()))
        print("pass_valid_70:", int(rules_df["pass_valid_70"].sum()))
        print("pass_valid_70_large:", int(rules_df["pass_valid_70_large"].sum()))
        print("pass_train_valid_70:", int(rules_df["pass_train_valid_70"].sum()))
        print("final_wf_filtered:", len(final_wf_filtered))
        print("or_rule_sets:", len(or_df))
        print("best_valid_precision:", rules_df["valid_precision"].max())
        print("max_valid_count:", rules_df["valid_count"].max())
    else:
        print("No rules.")

    print("=" * 80)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
python stable_rule_miner_final_v3.py \
  --csv csv/low_result_7_desc.csv \
  --out stable_rule_miner_final_v3_next \
  --date-col today \
  --n-jobs -4 \
  --max-depth 6 \
  --beam-width 1500 \
  --top-k 150 \
  --min-train-count 60 \
  --min-valid-count 40 \
  --min-train-precision 0.55 \
  --min-valid-precision 0.62 \
  --min-train-lift 1.25 \
  --min-valid-lift 1.50 \
  --max-precision-gap 0.18 \
  --min-month-count 5 \
  --month-pass-precision 0.55 \
  --month-crash-precision 0.45 \
  --min-valid-month-min-precision 0.55 \
  --max-valid-crash-months 0 \
  --valid-count-weight 7 \
  --wf-top-rules 150 \
  --wf-min-precision 0.60 \
  --wf-crash-precision 0.45 \
  --final-min-train-precision 0.60 \
  --final-min-valid-precision 0.70 \
  --final-min-valid-count 50 \
  --final-min-wf-mean-precision 0.70 \
  --final-min-wf-min-precision 0.55 \
  --final-min-wf-pass-split-rate 0.70 \
  --final-min-wf-total-count 50 \
  --final-max-wf-crash-splits 0 \
  --or-top-rules 15 \
  --or-max-size 3 \
  --or-min-valid-precision 0.70 \
  --or-min-valid-count 70 \
  --or-min-wf-mean-precision 0.70 \
  --or-min-wf-min-precision 0.55 \
  --or-min-wf-pass-split-rate 0.70 \
  --or-max-wf-crash-splits 0 \
  --simplify
"""