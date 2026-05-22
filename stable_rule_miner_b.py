import argparse
import hashlib
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

    "pct_vs_lastweek",
    "upper_wick_ratio",
]

# 기본에서는 제외. 필요하면 --include-vol-ratio 로만 추가.
EXPERIMENTAL_FEATURES = [
    "vol_ratio_5_15",
]


# 방향 제한.
# 직전 결과에서 upper_wick_ratio >= 같은 방향은 train 과최적화 냄새가 강했으므로 기본 차단.
ALLOWED_OPS = {
    "gap_pct": ["<="],
    "lower_wick_ratio": ["<="],
    "today_pct": [">="],

    "vol5": [">="],
    "dist_to_ma5": ["<="],
    "max_drop_7d": ["<="],

    "body_ratio": [">="],
    "recent_runup": ["<="],
    "intraday_return": [">="],

    "rebound_from_7d_low": [">="],
    "BB_perc": ["<="],

    "tr_val_rank_20d": ["<="],
    "today_tr_val_eok": [">=", "<="],

    "pct_vs_lastweek": ["<="],
    "upper_wick_ratio": ["<="],
    "up_down_tr_value_ratio_5d_log": ["<="],

    "vol_ratio_5_15": [">="],
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


def has_correlated_pair(used_features: Set[str], new_feature: str, corr_pairs: Set[frozenset]) -> bool:
    for f in used_features:
        if frozenset([f, new_feature]) in corr_pairs:
            return True
    return False


def default_extra_thresholds() -> Dict[str, List[Tuple[str, float]]]:
    return {
        "gap_pct": [
            ("<=", -2.0), ("<=", -1.0), ("<=", -0.5),
            ("<=", 0.0), ("<=", 0.5),
        ],
        "lower_wick_ratio": [
            ("<=", 0.0), ("<=", 0.004), ("<=", 0.01),
            ("<=", 0.02), ("<=", 0.05), ("<=", 0.10),
        ],
        "today_pct": [
            (">=", 3.5), (">=", 4.0), (">=", 5.0),
            (">=", 6.0), (">=", 8.0), (">=", 8.17),
            (">=", 10.0), (">=", 11.253), (">=", 15.86),
        ],
        "vol5": [
            (">=", 3.5), (">=", 4.0), (">=", 5.0),
            (">=", 6.0), (">=", 8.0), (">=", 8.467),
            (">=", 10.0), (">=", 10.5491),
        ],
        "upper_wick_ratio": [
            ("<=", 0.0), ("<=", 0.02), ("<=", 0.04),
            ("<=", 0.064), ("<=", 0.10), ("<=", 0.20),
            ("<=", 0.31),
        ],
        "body_ratio": [
            (">=", 0.6), (">=", 0.643), (">=", 0.7),
            (">=", 0.8), (">=", 0.839), (">=", 0.9),
        ],
        "dist_to_ma5": [
            ("<=", -0.0182), ("<=", -0.3), ("<=", -0.5),
            ("<=", -1.0), ("<=", -2.0),
        ],
        "max_drop_7d": [
            ("<=", -4.0), ("<=", -5.0), ("<=", -5.95),
            ("<=", -7.0), ("<=", -10.0),
        ],
        "recent_runup": [
            ("<=", -16.0), ("<=", -15.0), ("<=", -12.0),
            ("<=", -10.7), ("<=", -8.0), ("<=", -5.0),
            ("<=", 0.0),
        ],
        "intraday_return": [
            (">=", 4.0), (">=", 4.44), (">=", 5.0),
            (">=", 5.185), (">=", 6.0), (">=", 8.0),
        ],
        "rebound_from_7d_low": [
            (">=", 3.0), (">=", 5.0), (">=", 5.6268),
            (">=", 7.0), (">=", 10.0),
        ],
        "BB_perc": [
            ("<=", 0.05), ("<=", 0.10), ("<=", 0.148),
            ("<=", 0.1765), ("<=", 0.25), ("<=", 0.30),
        ],
        "today_tr_val_eok": [
            (">=", 3.0), (">=", 5.0), (">=", 10.0),
            ("<=", 40.0), ("<=", 52.8), ("<=", 63.0),
        ],
        "tr_val_rank_20d": [
            ("<=", 0.45), ("<=", 0.5), ("<=", 0.7),
            ("<=", 0.8), ("<=", 0.95),
        ],
        "pct_vs_lastweek": [
            ("<=", -5.0), ("<=", -3.0), ("<=", -1.0),
            ("<=", 0.0), ("<=", 0.21),
        ],
        "up_down_tr_value_ratio_5d_log": [
            ("<=", 0.3), ("<=", 0.43), ("<=", 0.5),
            ("<=", 1.0), ("<=", 1.631),
        ],
        "vol_ratio_5_15": [
            (">=", 1.0), (">=", 1.2), (">=", 1.3),
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


def monthly_summary(df, target_col, atoms, min_month_count, pass_precision, pass_lift):
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
        }

    pass_mask = (
            (usable["precision"] >= pass_precision) &
            (usable["lift"] >= pass_lift)
    )

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
    }


def eval_rule(df, target_col, atoms, min_month_count, pass_precision, pass_lift):
    mask = apply_rule(df, atoms)
    metrics = calc_metrics(df[target_col].values, mask)

    monthly_df, mon = monthly_summary(
        df=df,
        target_col=target_col,
        atoms=atoms,
        min_month_count=min_month_count,
        pass_precision=pass_precision,
        pass_lift=pass_lift,
    )

    return metrics, mon, monthly_df, mask


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

    gap = train_p - valid_p

    if gap > args.max_precision_gap:
        return -1e18

    if valid_mon["n_usable_months"] < args.min_valid_usable_months:
        return -1e18

    if valid_mon["pass_month_rate"] < args.min_valid_pass_month_rate:
        return -1e18

    valid_lcb = valid_m["precision_lcb"]
    if np.isfinite(valid_lcb) and valid_lcb < args.min_valid_lcb:
        return -1e18

    valid_std = valid_mon["std_month_precision"]
    if not np.isfinite(valid_std):
        valid_std = 0.0

    score = (
            valid_p * 110
            + valid_lcb * 70
            + valid_lift * 20
            + train_p * 25
            + valid_mon["mean_month_precision"] * 60
            + valid_mon["min_month_precision"] * 40
            + valid_mon["pass_month_rate"] * 50
            + math.log1p(valid_count) * 3
            - gap * 80
            - valid_std * 50
            - valid_mon["bad_month_count"] * 2
    )

    return score


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
            used_features = set(a.feature for a in base_atoms)

            for atom in atoms:
                if atom.feature in used_features:
                    continue

                if not args.allow_correlated_in_rule:
                    if has_correlated_pair(used_features, atom.feature, corr_pairs):
                        continue

                new_atoms = tuple(list(base_atoms) + [atom])

                feature_order = [a.feature for a in new_atoms]
                if feature_order != sorted(feature_order):
                    continue

                key = tuple((a.feature, a.op, round(a.threshold, 10)) for a in new_atoms)

                if key in seen_rules:
                    continue
                seen_rules.add(key)

                train_m, train_mon, _, train_mask = eval_rule(
                    train,
                    args.target,
                    new_atoms,
                    args.min_month_count,
                    args.month_pass_precision,
                    args.month_pass_lift,
                )

                bscore = beam_score(train_m, train_mon, args)

                if bscore <= -1e17:
                    continue

                train_key = mask_hash(train_mask)

                # 같은 train selection이면 더 복잡한 룰 제거
                if train_key in seen_train_masks and len(new_atoms) > 1:
                    continue

                seen_train_masks.add(train_key)

                valid_m, valid_mon, _, valid_mask = eval_rule(
                    valid,
                    args.target,
                    new_atoms,
                    args.min_month_count,
                    args.month_pass_precision,
                    args.month_pass_lift,
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

            "precision_gap_train_minus_valid": (
                train_p - valid_p
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
                args.min_month_count,
                args.month_pass_precision,
                args.month_pass_lift,
            )
            monthly_df = monthly_df.copy()
            monthly_df["rank"] = rank
            monthly_df["rule"] = r.name()
            monthly_df["dataset"] = dataset_name
            dfs.append(monthly_df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="stable_rule_miner_b_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--include-vol-ratio", action="store_true")

    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=700)
    parser.add_argument("--top-k", type=int, default=100)

    # B안 현실적 기준
    parser.add_argument("--min-train-count", type=int, default=120)
    parser.add_argument("--min-valid-count", type=int, default=80)

    parser.add_argument("--min-train-precision", type=float, default=0.52)
    parser.add_argument("--min-valid-precision", type=float, default=0.52)

    parser.add_argument("--min-train-lift", type=float, default=1.20)
    parser.add_argument("--min-valid-lift", type=float, default=1.20)

    parser.add_argument("--min-valid-lcb", type=float, default=0.45)

    parser.add_argument("--max-precision-gap", type=float, default=0.12)

    # Beam은 더 느슨하게
    parser.add_argument("--beam-min-count", type=int, default=80)
    parser.add_argument("--beam-min-lift", type=float, default=1.00)

    # 월별 안정성
    parser.add_argument("--min-month-count", type=int, default=8)
    parser.add_argument("--month-pass-precision", type=float, default=0.52)
    parser.add_argument("--month-pass-lift", type=float, default=1.15)
    parser.add_argument("--min-valid-usable-months", type=int, default=2)
    parser.add_argument("--min-valid-pass-month-rate", type=float, default=0.20)

    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--allow-correlated-in-rule", action="store_true")

    parser.add_argument("--dup-count-tol", type=float, default=0.02)
    parser.add_argument("--dup-precision-gain", type=float, default=0.01)

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

    features = list(DEFAULT_FEATURES)

    if args.include_vol_ratio:
        features += EXPERIMENTAL_FEATURES

    features = [f for f in features if f in df.columns]

    train, valid = split_train_valid_by_date(
        df=df,
        valid_ratio=args.valid_ratio,
    )

    print("=" * 80)
    print("[INFO] rows:", len(df))
    print("[INFO] train rows:", len(train), "base_rate:", train[args.target].mean())
    print("[INFO] valid rows:", len(valid), "base_rate:", valid[args.target].mean())
    print("[INFO] features:", features)
    print("[INFO] B-plan target: stable valid precision >=", args.min_valid_precision)
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
        atoms=atoms,
        corr_pairs=corr_pairs,
        args=args,
    )

    rules_df = rules_to_df(rules)

    rules_df.to_csv(
        os.path.join(args.out, "02_selected_rules.csv"),
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
        os.path.join(args.out, "03_monthly_details.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[SELECTED STABLE RULES]")
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

        "precision_gap_train_minus_valid",

        "valid_month_n_usable_months",
        "valid_month_mean_month_precision",
        "valid_month_min_month_precision",
        "valid_month_pass_month_rate",
    ]
    show_cols = [c for c in show_cols if c in rules_df.columns]

    if len(rules_df):
        print(rules_df[show_cols].head(50).to_string(index=False))
    else:
        print("No stable rules found. Try looser thresholds.")

    print("=" * 80)
    print("[DONE]")
    print("Output directory:", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()

# 실행방법
# python stable_rule_miner_b.py   --csv csv/low_result_7_desc.csv   --out stable_rule_miner_b_out   --date-col today   --max-depth 5   --beam-width 700   --top-k 100   --min-train-count 120   --min-valid-count 80   --min-train-precision 0.52   --min-valid-precision 0.52   --min-train-lift 1.20   --min-valid-lift 1.20   --max-precision-gap 0.12FF