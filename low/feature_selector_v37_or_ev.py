import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

"""
"피쳐가 유용한지 알고 싶다 - custom feature pool"

“이 피쳐가 유용한가?”
“단독으로 약해도 룰 조합에서 필터 역할을 하는가?”

좋은 룰을 많이 뽑고, 그 룰들에서 피쳐 유용성을 평가하는 방식

>>>
전체 룰 후보 기준으로 피쳐 유용성 평가
조건부 기여도 추가
비단조 피쳐 예외 처리
수동 threshold + ALLOWED_OPS 적용

좋은 점:
현재 네 목적 “유용한 피쳐를 알고 싶다”에 가장 잘 맞음
BB_perc, dist_to_ma5, gap_pct 같은 필터 피쳐를 더 공정하게 봄

나쁜 점:
S 등급이 넓게 나올 수 있음
valid 평균 60%를 보장하지 않음
"""
TARGET_COL = "target_before_stop_10"


# ============================================================
# Feature pool
# 현재 rule_features + stable_rule_miner에서 쓰는 후보를 같이 커버
# 실제 CSV에 없는 컬럼은 자동 skip
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
    "lower_wick_ratio",
    "vol15",
    "ATR_pct",
    "dist_to_ma20",

    "BB_perc",
    "gap_pct",
    "room_to_60d_high",
    "ma5_chg_rate",
    "pct_vs_lastweek",
]

# 단조 방향으로 보기 어려운 피쳐.
# AUC 방향성과 룰 방향이 충돌해도 바로 감점하지 않고,
# conditional contribution을 더 중요하게 본다.
NON_MONOTONIC_FEATURES = [
    "gap_pct",
    "pct_vs_lastweek",
    "dist_to_ma5",
    "dist_to_ma20",
    "ma5_chg_rate",
    "BB_perc",
    "room_to_60d_high",
    "rebound_vs_prior_drop",
    "lower_wick_ratio",
]


# 시장/장세 필터
REGIME_FEATURES = [
    # 현재 19개 피쳐에는 market_* 장세 피쳐가 없음
]


# 반복적으로 약했던 피쳐.
# 강제 제거는 아니고, 최종 등급에서 참고만 한다.
WEAK_HINT_FEATURES = [
]


# stable_rule_miner의 방향 제약을 반영.
# 여기에 없으면 기본적으로 <=, >= 모두 허용.
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
    "lower_wick_ratio": ["<=", ">="],
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
    for c in [
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
    ]:
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



def _safe_div(a, b):
    return a / b if b not in (0, 0.0) and np.isfinite(b) else np.nan


def _woe_iv_from_bin_stats(bin_stats, total_good, total_bad):
    # 0.5 smoothing to avoid zero division.
    k = max(1, len(bin_stats))
    good = bin_stats["sum"].astype(float)
    bad = bin_stats["count"].astype(float) - good
    good_dist = (good + 0.5) / (total_good + 0.5 * k)
    bad_dist = (bad + 0.5) / (total_bad + 0.5 * k)
    woe = np.log(good_dist / bad_dist)
    iv_part = (good_dist - bad_dist) * woe
    return woe, iv_part


def single_feature_bin_report(train, features, target_col):
    rows = []

    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        mask = x.notna() & np.isfinite(x)
        xv = x[mask]
        yv = train.loc[mask, target_col].astype(int)
        base_rate = float(yv.mean()) if len(yv) else np.nan

        if len(xv) < 100 or xv.nunique() <= 2:
            continue

        try:
            bins = pd.qcut(xv, q=10, duplicates="drop")
        except Exception:
            bins = pd.cut(xv, bins=10, duplicates="drop")

        tmp = pd.DataFrame({"bin": bins.astype(str), "x": xv.values, "y": yv.values})
        bin_stats = tmp.groupby("bin", observed=False)["y"].agg(["count", "sum", "mean"])
        bin_stats["lift"] = bin_stats["mean"] / base_rate if base_rate > 0 else np.nan

        total_good = float((yv == 1).sum())
        total_bad = float((yv == 0).sum())
        woe, iv_part = _woe_iv_from_bin_stats(bin_stats, total_good, total_bad)
        bin_stats["woe"] = woe
        bin_stats["iv_part"] = iv_part

        # Add numeric interval boundaries when possible from original categorical bins.
        for bname, r in bin_stats.iterrows():
            rows.append({
                "feature": f,
                "bin": bname,
                "count": int(r["count"]),
                "target_count": int(r["sum"]),
                "precision": float(r["mean"]),
                "lift": float(r["lift"]),
                "woe": float(r["woe"]),
                "iv_part": float(r["iv_part"]),
                "base_rate": base_rate,
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["feature", "lift", "precision"], ascending=[True, False, False])
    return out


def infer_stop_loss(df, default=-6.0):
    if "stop_loss" in df.columns:
        s = pd.to_numeric(df["stop_loss"], errors="coerce").dropna()
        if len(s):
            return float(s.mode().iloc[0])
    return float(default)


def ev_metrics_for_mask(df, mask, target_level, stop_loss=None):
    mask = np.asarray(mask).astype(bool)
    sub = df.loc[mask].copy()
    n = int(mask.sum())
    target_col = f"target_before_stop_{target_level}"
    stop_col = f"stop_before_target_{target_level}"
    same_col = f"target_stop_same_day_{target_level}"
    none_col = f"no_target_no_stop_{target_level}"
    day_col = f"day_to_{target_level}"

    if stop_loss is None:
        stop_loss = infer_stop_loss(df)
    stop_abs = abs(float(stop_loss))
    target_pct = float(target_level)

    out = {
        "count": n,
        "target_level_pct": target_pct,
        "stop_loss_pct": float(stop_loss),
        "risk_reward_R": _safe_div(target_pct, stop_abs),
        "breakeven_win_rate": _safe_div(stop_abs, target_pct + stop_abs),
    }

    if n == 0 or target_col not in sub.columns or stop_col not in sub.columns:
        out.update({
            "win_count": 0,
            "win_rate": np.nan,
            "loss_count": 0,
            "loss_rate": np.nan,
            "same_day_count": 0,
            "same_day_rate": np.nan,
            "no_touch_count": 0,
            "no_touch_rate": np.nan,
            "profit_factor": np.nan,
            "ev_conservative_pct": np.nan,
            "ev_same_day_as_win_pct": np.nan,
        })
        return out

    win = pd.to_numeric(sub[target_col], errors="coerce").fillna(0).astype(float)
    loss = pd.to_numeric(sub[stop_col], errors="coerce").fillna(0).astype(float)
    same = pd.to_numeric(sub[same_col], errors="coerce").fillna(0).astype(float) if same_col in sub.columns else pd.Series(0.0, index=sub.index)
    none = pd.to_numeric(sub[none_col], errors="coerce").fillna(0).astype(float) if none_col in sub.columns else pd.Series(0.0, index=sub.index)

    win_rate = float(win.mean())
    loss_rate = float(loss.mean())
    same_rate = float(same.mean())
    none_rate = float(none.mean())
    gross_win = win_rate * target_pct
    gross_loss = loss_rate * stop_abs

    out.update({
        "win_count": int(win.sum()),
        "win_rate": win_rate,
        "loss_count": int(loss.sum()),
        "loss_rate": loss_rate,
        "same_day_count": int(same.sum()),
        "same_day_rate": same_rate,
        "no_touch_count": int(none.sum()),
        "no_touch_rate": none_rate,
        "profit_factor": _safe_div(gross_win, gross_loss),
        "ev_conservative_pct": gross_win + loss_rate * float(stop_loss),
        "ev_same_day_as_win_pct": (win_rate + same_rate) * target_pct + loss_rate * float(stop_loss),
        "edge_vs_breakeven_pctp": win_rate - _safe_div(stop_abs, target_pct + stop_abs),
    })

    if day_col in sub.columns:
        day = pd.to_numeric(sub.loc[win.eq(1).values, day_col], errors="coerce")
        out.update({
            "avg_day_to_target_hit_only": float(day.mean()) if len(day) else np.nan,
            "median_day_to_target_hit_only": float(day.median()) if len(day) else np.nan,
            "p75_day_to_target_hit_only": float(day.quantile(0.75)) if len(day) else np.nan,
            "p90_day_to_target_hit_only": float(day.quantile(0.90)) if len(day) else np.nan,
            "hit_within_3d_rate": float(day.le(3).mean()) if len(day) else np.nan,
            "hit_within_4d_rate": float(day.le(4).mean()) if len(day) else np.nan,
            "hit_within_5d_rate": float(day.le(5).mean()) if len(day) else np.nan,
        })

    if "stop_day" in sub.columns:
        stop_day = pd.to_numeric(sub.loc[loss.eq(1).values, "stop_day"], errors="coerce")
        out.update({
            "avg_stop_day_loss_only": float(stop_day.mean()) if len(stop_day) else np.nan,
            "median_stop_day_loss_only": float(stop_day.median()) if len(stop_day) else np.nan,
            "stop_within_2d_rate": float(stop_day.le(2).mean()) if len(stop_day) else np.nan,
            "stop_within_3d_rate": float(stop_day.le(3).mean()) if len(stop_day) else np.nan,
        })

    return out


def target_ev_overall_report(df, split_name):
    rows = []
    mask = np.ones(len(df), dtype=bool)
    for target_level in [10, 15]:
        if f"target_before_stop_{target_level}" not in df.columns:
            continue
        row = {"split": split_name, **ev_metrics_for_mask(df, mask, target_level)}
        rows.append(row)
    return pd.DataFrame(rows)


def timing_distribution_report(df, split_name):
    rows = []
    n_all = len(df)
    for target_level in [10, 15]:
        tcol = f"target_before_stop_{target_level}"
        dcol = f"day_to_{target_level}"
        scol = f"stop_before_target_{target_level}"
        if tcol in df.columns and dcol in df.columns:
            hit = pd.to_numeric(df[tcol], errors="coerce").fillna(0).eq(1)
            day = pd.to_numeric(df.loc[hit, dcol], errors="coerce")
            total = int(hit.sum())
            cum = 0
            for d, cnt in day.value_counts().sort_index().items():
                cum += int(cnt)
                rows.append({
                    "split": split_name,
                    "event": f"target_{target_level}_hit_day",
                    "day": d,
                    "count": int(cnt),
                    "pct_of_event": _safe_div(cnt, total),
                    "cum_pct_of_event": _safe_div(cum, total),
                    "pct_of_all": _safe_div(cnt, n_all),
                })
        if scol in df.columns and "stop_day" in df.columns:
            stop = pd.to_numeric(df[scol], errors="coerce").fillna(0).eq(1)
            stop_day = pd.to_numeric(df.loc[stop, "stop_day"], errors="coerce")
            total = int(stop.sum())
            cum = 0
            for d, cnt in stop_day.value_counts().sort_index().items():
                cum += int(cnt)
                rows.append({
                    "split": split_name,
                    "event": f"target_{target_level}_stop_day",
                    "day": d,
                    "count": int(cnt),
                    "pct_of_event": _safe_div(cnt, total),
                    "cum_pct_of_event": _safe_div(cum, total),
                    "pct_of_all": _safe_div(cnt, n_all),
                })
    return pd.DataFrame(rows)


def rule_ev_report(rules, train, valid):
    rows = []
    for i, r in enumerate(rules, start=1):
        train_mask = apply_rule(train, r.atoms)
        valid_mask = apply_rule(valid, r.atoms)
        base = {
            "profile": r.profile_name,
            "rank": i,
            "rule": r.name(),
            "features": ",".join(r.features()),
            "n_features": len(r.features()),
        }
        for split_name, df_part, mask in [("train", train, train_mask), ("valid", valid, valid_mask)]:
            for target_level in [10, 15]:
                if f"target_before_stop_{target_level}" not in df_part.columns:
                    continue
                m = ev_metrics_for_mask(df_part, mask, target_level)
                row = dict(base)
                row["split"] = split_name
                row["target_level_pct"] = target_level
                row.update(m)
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["split", "target_level_pct", "ev_conservative_pct", "profit_factor", "count"], ascending=[True, True, False, False, False])
    return out

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
                "iv": np.nan,
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
        total_good = float((yv == 1).sum())
        total_bad = float((yv == 0).sum())
        _, iv_part = _woe_iv_from_bin_stats(bin_stats, total_good, total_bad)
        iv = float(iv_part.sum()) if len(iv_part) else np.nan

        best = bin_stats.sort_values(
            ["lift", "mean", "count"],
            ascending=False,
        ).iloc[0]

        rows.append({
            "feature": f,
            "auc_raw": auc_info["auc_raw"],
            "auc_direction": auc_info["auc_direction"],
            "auc_oriented": auc_info["auc_oriented"],
            "iv": iv,
            "best_bin": best.name,
            "best_bin_precision": best["mean"],
            "best_bin_lift": best["lift"],
            "best_bin_count": int(best["count"]),
            "missing_rate": 1 - mask.mean(),
        })

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["auc_oriented", "iv", "best_bin_lift"],
            ascending=[False, False, False],
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
        "lower_wick_ratio": [
            ("<=", 0.05), ("<=", 0.10), ("<=", 0.20),
            ("<=", 0.30), ("<=", 0.409),
            (">=", 0.20), (">=", 0.30), (">=", 0.4113),
            (">=", 0.50), (">=", 0.60),
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
            "name": "usefulness_stable",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "quantiles": base_quantiles,

            "min_atom_count": 80,
            "min_atom_lift": 1.03,
            "min_atom_precision": 0.42,

            "min_train_count": 160,
            "min_valid_count": 80,

            "beam_min_train_precision": 0.42,
            "beam_min_train_lift": 1.00,

            "min_train_precision": 0.48,
            "min_train_lift": 1.08,

            "min_valid_precision": 0.52,
            "min_valid_lift": 1.15,

            "max_train_coverage": 0.65,
        },
        {
            "name": "usefulness_precision",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "quantiles": base_quantiles,

            "min_atom_count": 60,
            "min_atom_lift": 1.05,
            "min_atom_precision": 0.45,

            "min_train_count": 100,
            "min_valid_count": 50,

            "beam_min_train_precision": 0.43,
            "beam_min_train_lift": 1.00,

            "min_train_precision": 0.50,
            "min_train_lift": 1.12,

            "min_valid_precision": 0.55,
            "min_valid_lift": 1.25,

            "max_train_coverage": 0.50,
        },
        {
            "name": "usefulness_high_precision_hint",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "quantiles": base_quantiles,

            "min_atom_count": 50,
            "min_atom_lift": 1.08,
            "min_atom_precision": 0.46,

            "min_train_count": 60,
            "min_valid_count": 30,

            "beam_min_train_precision": 0.43,
            "beam_min_train_lift": 1.00,

            "min_train_precision": 0.52,
            "min_train_lift": 1.15,

            "min_valid_precision": 0.58,
            "min_valid_lift": 1.35,

            "max_train_coverage": 0.40,
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


def feature_usage_from_rules(rules_df, features, top_n):
    rows = []

    if len(rules_df) == 0:
        return pd.DataFrame()

    for profile_name, g in rules_df.groupby("profile"):
        top = g.sort_values("rank").head(top_n)

        for f in features:
            used = top["features"].fillna("").apply(
                lambda s: f in [x.strip() for x in str(s).split(",") if x.strip()]
            )

            used_df = top[used]

            row = {
                "profile": profile_name,
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

    for profile_name, g in rules_df.groupby("profile"):
        top = g.sort_values("rank").head(top_n)

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
                "profile": profile_name,
                "feature": f,
                "used_rules": used_rules,
                "ge_count": ge_count,
                "le_count": le_count,
                "dominant_op": dominant_op,
                "direction_consistency": direction_consistency,
                "median_threshold": median_threshold,
            })

    return pd.DataFrame(rows)


def monthly_rule_stability(valid, rules, target_col):
    if "month" not in valid.columns:
        return pd.DataFrame()

    tmp_valid = valid.copy().reset_index(drop=True)
    y_all = tmp_valid[target_col].astype(int).values

    rows = []

    for rule_idx, r in enumerate(rules, start=1):
        mask_all = apply_rule(tmp_valid, r.atoms)

        for month, idx_values in tmp_valid.groupby("month").groups.items():
            pos = np.array(list(idx_values), dtype=int)
            m = precision_recall_lift(y_all[pos], mask_all[pos])

            rows.append({
                "profile": r.profile_name,
                "rule_rank": rule_idx,
                "rule": r.name(),
                "month": month,
                "month_count": m["count"],
                "month_target_count": m["target_count"],
                "month_precision": m["precision"],
                "month_lift": m["lift"],
                "month_coverage": m["coverage"],
                "month_base_rate": m["base_rate"],
            })

    return pd.DataFrame(rows)


def summarize_monthly_stability(monthly_df, min_month_count=20):
    if len(monthly_df) == 0:
        return pd.DataFrame()

    use = monthly_df[monthly_df["month_count"] >= min_month_count].copy()

    if len(use) == 0:
        return pd.DataFrame()

    rows = []

    for rule, g in use.groupby("rule"):
        rows.append({
            "rule": rule,
            "months_used": g["month"].nunique(),
            "mean_month_precision": g["month_precision"].mean(),
            "median_month_precision": g["month_precision"].median(),
            "min_month_precision": g["month_precision"].min(),
            "mean_month_lift": g["month_lift"].mean(),
            "median_month_lift": g["month_lift"].median(),
            "min_month_lift": g["month_lift"].min(),
            "bad_months_precision_lt_50": int((g["month_precision"] < 0.50).sum()),
            "bad_months_lift_lt_1": int((g["month_lift"] < 1.0).sum()),
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        [
            "bad_months_lift_lt_1",
            "bad_months_precision_lt_50",
            "median_month_precision",
        ],
        ascending=[True, True, False],
    )

    return out


def conditional_feature_contribution(
        train,
        valid,
        rules,
        target_col,
):
    rows = []

    y_train = train[target_col].astype(int).values
    y_valid = valid[target_col].astype(int).values

    for rule_idx, r in enumerate(rules, start=1):
        full_train_mask = apply_rule(train, r.atoms)
        full_valid_mask = apply_rule(valid, r.atoms)

        full_train_m = precision_recall_lift(y_train, full_train_mask)
        full_valid_m = precision_recall_lift(y_valid, full_valid_mask)

        rule_features = sorted(set(a.feature for a in r.atoms))

        for f in rule_features:
            reduced_atoms = tuple(a for a in r.atoms if a.feature != f)

            if len(reduced_atoms) == 0:
                reduced_train_mask = np.ones(len(train), dtype=bool)
                reduced_valid_mask = np.ones(len(valid), dtype=bool)
            else:
                reduced_train_mask = apply_rule(train, reduced_atoms)
                reduced_valid_mask = apply_rule(valid, reduced_atoms)

            reduced_train_m = precision_recall_lift(y_train, reduced_train_mask)
            reduced_valid_m = precision_recall_lift(y_valid, reduced_valid_mask)

            rows.append({
                "rule_rank": rule_idx,
                "profile": r.profile_name,
                "rule": r.name(),
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

                "valid_count_change": (
                        full_valid_m["count"] - reduced_valid_m["count"]
                ),

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
        monthly_df,
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
            profiles_used = int((u["top_usage_count"] > 0).sum())
            total_usage_count = int(u["top_usage_count"].sum())
            avg_valid_precision = u["avg_valid_precision_when_used"].mean()
            avg_valid_lift = u["avg_valid_lift_when_used"].mean()
            best_valid_precision = u["best_valid_precision_when_used"].max()
            best_valid_lift = u["best_valid_lift_when_used"].max()
            avg_gap = u["avg_abs_precision_gap_when_used"].mean()
            pass_valid_60_count = int(u["pass_valid_60_count"].sum())
        else:
            profiles_used = 0
            total_usage_count = 0
            avg_valid_precision = np.nan
            avg_valid_lift = np.nan
            best_valid_precision = np.nan
            best_valid_lift = np.nan
            avg_gap = np.nan
            pass_valid_60_count = 0

        if len(d):
            used_d = d[d["used_rules"] > 0]
            if len(used_d):
                avg_direction_consistency = used_d["direction_consistency"].mean()
                dominant_ops = used_d["dominant_op"].dropna().tolist()
                dominant_op = max(set(dominant_ops), key=dominant_ops.count) if dominant_ops else ""
            else:
                avg_direction_consistency = np.nan
                dominant_op = ""
        else:
            avg_direction_consistency = np.nan
            dominant_op = ""

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

        if f in NON_MONOTONIC_FEATURES:
            direction_ok = True
        else:
            direction_ok = (
                    (auc_direction == "higher_success" and dominant_op == ">=")
                    or (auc_direction == "lower_success" and dominant_op == "<=")
                    or dominant_op == ""
            )

        is_non_mono = f in NON_MONOTONIC_FEATURES
        is_regime = f in REGIME_FEATURES

        score = 0.0

        if np.isfinite(auc_oriented):
            score += max(0.0, auc_oriented - 0.5) * 35

        if np.isfinite(best_bin_lift):
            score += max(0.0, best_bin_lift - 1.0) * 12

        if np.isfinite(avg_valid_lift):
            score += max(0.0, avg_valid_lift - 1.0) * 25

        if np.isfinite(avg_valid_precision):
            score += max(0.0, avg_valid_precision - 0.5) * 70

        if np.isfinite(mean_valid_precision_gain):
            score += max(0.0, mean_valid_precision_gain) * 120

        if np.isfinite(max_valid_precision_gain):
            score += max(0.0, max_valid_precision_gain) * 30

        score += positive_gain_rate * 15
        score += strong_gain_rate * 20
        score += profiles_used * 4
        score += total_usage_count * 0.4
        score += pass_valid_60_count * 8

        if np.isfinite(avg_gap):
            score -= avg_gap * 20

        if not direction_ok:
            score -= 8

        if f in WEAK_HINT_FEATURES:
            score -= 2

        is_core_signal = (
                direction_ok
                and not is_non_mono
                and (
                        (np.isfinite(auc_oriented) and auc_oriented >= 0.555)
                        or (np.isfinite(avg_valid_lift) and avg_valid_lift >= 1.35)
                        or pass_valid_60_count > 0
                )
        )

        is_regime_filter = (
                is_regime
                and (
                        (np.isfinite(best_bin_lift) and best_bin_lift >= 1.30)
                        or (np.isfinite(avg_valid_lift) and avg_valid_lift >= 1.30)
                        or pass_valid_60_count > 0
                )
        )

        is_conditional_filter = (
                is_non_mono
                and rules_used_conditional > 0
                and (
                        positive_gain_rate >= 0.25
                        or (
                                np.isfinite(mean_valid_precision_gain)
                                and mean_valid_precision_gain > 0.005
                        )
                        or pass_valid_60_count > 0
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
            reason = "시장/장세 필터로 valid lift 또는 best bin이 좋음"

        elif is_core_signal:
            grade = "S"
            role = "CORE_SIGNAL"
            action = "KEEP"
            reason = "단독 방향성, AUC, 룰 성능이 좋은 핵심 신호"

        elif is_conditional_filter:
            grade = "A"
            role = "CONDITIONAL_FILTER"
            action = "KEEP_AS_FILTER"
            reason = "단독 AUC는 약해도 룰에서 제거 시 valid 성능이 떨어지는 조건부 필터"

        elif is_candidate:
            grade = "B"
            role = "CANDIDATE"
            action = "TEST"
            reason = "특정 구간/조합에서 유용 가능성"

        else:
            grade = "C"
            role = "DROP_CANDIDATE"
            action = "DROP_OR_LAST_CHECK"
            reason = "현재 기준 근거 약함"

        rows.append({
            "feature": f,
            "grade": grade,
            "role": role,
            "action": action,
            "score": score,
            "reason": reason,

            "is_non_monotonic": is_non_mono,
            "is_regime": is_regime,

            "profiles_used": profiles_used,
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
    groups = [
        ("S_KEEP", feature_grade_df[feature_grade_df["grade"] == "S"]),
        ("A_CONDITIONAL_FILTERS", feature_grade_df[feature_grade_df["grade"] == "A"]),
        ("B_TEST_CANDIDATES", feature_grade_df[feature_grade_df["grade"] == "B"]),
        ("C_DROP_CANDIDATES", feature_grade_df[feature_grade_df["grade"] == "C"]),
    ]

    lines = []

    for name, g in groups:
        lines.append(f"{name} = [")
        for f in g["feature"].tolist():
            lines.append(f'    "{f}",')
        lines.append("]\n")

    path = os.path.join(out_dir, "11_recommended_feature_roles.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n[RECOMMENDED FEATURE ROLES]")
    print("\n".join(lines))



# ============================================================
# EV based OR ruleset builder
# 1) filter high-EV individual rules
# 2) extract target10/target15 candidate rules
# 3) greedily OR rules until target coverage is reached
# ============================================================

def _rule_key(profile, rule):
    return (str(profile), str(rule))


def _make_rule_lookup(rules):
    lookup = {}
    for r in rules:
        lookup[_rule_key(r.profile_name, r.name())] = r
    return lookup


def filter_ev_candidate_rows(
        rule_ev_df,
        target_level,
        split='valid',
        min_count=30,
        min_ev=2.0,
        min_profit_factor=2.0,
        min_win_rate=0.0,
        max_stop_within_2d=None,
):
    if len(rule_ev_df) == 0:
        return pd.DataFrame()

    use = rule_ev_df.copy()

    if 'split' in use.columns:
        use = use[use['split'].astype(str) == str(split)].copy()

    if 'target_level_pct' in use.columns:
        use = use[pd.to_numeric(use['target_level_pct'], errors='coerce') == float(target_level)].copy()

    if len(use) == 0:
        return use

    for c in [
        'count', 'ev_conservative_pct', 'profit_factor', 'win_rate',
        'loss_rate', 'coverage', 'stop_within_2d_rate', 'hit_within_3d_rate',
        'hit_within_4d_rate',
    ]:
        if c in use.columns:
            use[c] = pd.to_numeric(use[c], errors='coerce')

    mask = (
            (use['count'] >= min_count)
            & (use['ev_conservative_pct'] >= min_ev)
            & (use['profit_factor'] >= min_profit_factor)
    )

    if 'win_rate' in use.columns:
        mask &= use['win_rate'] >= min_win_rate

    if max_stop_within_2d is not None and 'stop_within_2d_rate' in use.columns:
        # stop_within_2d_rate is conditional on losing trades.
        # Keep NaN because it can mean there were no losing trades.
        mask &= (use['stop_within_2d_rate'].isna() | (use['stop_within_2d_rate'] <= max_stop_within_2d))

    use = use[mask].copy()

    if len(use):
        use = use.sort_values(
            ['ev_conservative_pct', 'profit_factor', 'win_rate', 'count'],
            ascending=[False, False, False, False],
        )

    return use


def _or_metrics_row(
        selected_rules,
        df_part,
        target_level,
        split,
        ruleset_name,
        step=None,
):
    if len(df_part) == 0:
        mask = np.zeros(0, dtype=bool)
    else:
        mask = np.zeros(len(df_part), dtype=bool)
        for r in selected_rules:
            mask |= apply_rule(df_part, r.atoms)

    m = ev_metrics_for_mask(df_part, mask, target_level)
    row = {
        'ruleset_name': ruleset_name,
        'split': split,
        'target_level_pct': float(target_level),
        'step': step if step is not None else len(selected_rules),
        'n_rules': len(selected_rules),
        'rules': ' || '.join([r.name() for r in selected_rules]),
        'features': ','.join(sorted(set(f for r in selected_rules for f in r.features()))),
    }
    row.update(m)
    return row


def greedy_or_ruleset(
        candidate_rows,
        rule_lookup,
        train,
        valid,
        target_level,
        ruleset_name,
        max_rules=20,
        target_coverage=0.05,
        min_incremental_count=10,
        min_set_ev=1.0,
        min_set_profit_factor=1.2,
        max_candidate_pool=200,
):
    """Greedy OR selection using valid EV as the main objective.

    The search starts from the strongest single rule and keeps adding rules that:
    - add enough new valid rows,
    - keep the whole OR set EV above min_set_ev,
    - keep the whole OR set profit factor above min_set_profit_factor.
    """
    if len(candidate_rows) == 0:
        return pd.DataFrame(), pd.DataFrame()

    candidates = []
    for _, row in candidate_rows.head(max_candidate_pool).iterrows():
        key = _rule_key(row.get('profile', ''), row.get('rule', ''))
        r = rule_lookup.get(key)
        if r is not None:
            candidates.append((r, row.to_dict()))

    if not candidates:
        return pd.DataFrame(), pd.DataFrame()

    selected = []
    selected_names = set()
    selected_valid_mask = np.zeros(len(valid), dtype=bool)
    step_rows = []

    for step in range(1, max_rules + 1):
        best = None
        best_score = -1e18

        for r, meta in candidates:
            rname = r.name()
            if rname in selected_names:
                continue

            cand_valid_mask = apply_rule(valid, r.atoms)
            new_rows_mask = cand_valid_mask & (~selected_valid_mask)
            incremental_count = int(new_rows_mask.sum())
            if incremental_count < min_incremental_count:
                continue

            test_rules = selected + [r]
            test_row = _or_metrics_row(
                selected_rules=test_rules,
                df_part=valid,
                target_level=target_level,
                split='valid',
                ruleset_name=ruleset_name,
                step=step,
            )

            ev = test_row.get('ev_conservative_pct', np.nan)
            pf = test_row.get('profit_factor', np.nan)
            coverage = test_row.get('count', 0) / len(valid) if len(valid) else 0.0
            win_rate = test_row.get('win_rate', np.nan)
            loss_rate = test_row.get('loss_rate', np.nan)

            if not np.isfinite(ev) or not np.isfinite(pf):
                continue
            if ev < min_set_ev:
                continue
            if pf < min_set_profit_factor:
                continue

            # Prefer EV, then coverage expansion, then high win/loss separation.
            # A mild penalty prevents adding very low incremental rules too early.
            score = (
                    ev * 100.0
                    + pf * 5.0
                    + coverage * 80.0
                    + incremental_count * 0.08
                    + (win_rate - loss_rate) * 20.0
            )

            if score > best_score:
                best_score = score
                best = (r, test_row, cand_valid_mask, incremental_count, meta)

        if best is None:
            break

        r, test_row, cand_valid_mask, incremental_count, meta = best
        selected.append(r)
        selected_names.add(r.name())
        selected_valid_mask |= cand_valid_mask

        test_row['added_rule'] = r.name()
        test_row['added_profile'] = r.profile_name
        test_row['incremental_valid_count'] = incremental_count
        test_row['added_rule_valid_ev'] = meta.get('ev_conservative_pct', np.nan)
        test_row['added_rule_valid_pf'] = meta.get('profit_factor', np.nan)
        test_row['added_rule_valid_win_rate'] = meta.get('win_rate', np.nan)
        step_rows.append(test_row)

        if test_row.get('coverage', 0.0) >= target_coverage:
            break

    if not selected:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    for split_name, df_part in [('train', train), ('valid', valid)]:
        summary_rows.append(_or_metrics_row(
            selected_rules=selected,
            df_part=df_part,
            target_level=target_level,
            split=split_name,
            ruleset_name=ruleset_name,
        ))

    return pd.DataFrame(summary_rows), pd.DataFrame(step_rows)


def build_or_rulesets_from_ev(
        all_rules,
        rule_ev_df,
        train,
        valid,
        args,
):
    """Build target10/target15 OR rulesets from EV-filtered rules."""
    if len(rule_ev_df) == 0 or not all_rules:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    rule_lookup = _make_rule_lookup(all_rules)
    all_candidate_rows = []
    all_summary_rows = []
    all_step_rows = []

    for target_level in [10, 15]:
        candidates = filter_ev_candidate_rows(
            rule_ev_df=rule_ev_df,
            target_level=target_level,
            split='valid',
            min_count=args.or_min_valid_count,
            min_ev=args.or_min_rule_ev,
            min_profit_factor=args.or_min_rule_pf,
            min_win_rate=args.or_min_rule_win_rate,
            max_stop_within_2d=args.or_max_stop_within_2d,
        )

        if len(candidates):
            tmp = candidates.copy()
            tmp['ruleset_target_level_pct'] = target_level
            all_candidate_rows.append(tmp)

        summary_df, steps_df = greedy_or_ruleset(
            candidate_rows=candidates,
            rule_lookup=rule_lookup,
            train=train,
            valid=valid,
            target_level=target_level,
            ruleset_name=f'OR_T{target_level}_EV',
            max_rules=args.or_max_rules,
            target_coverage=args.or_target_coverage,
            min_incremental_count=args.or_min_incremental_count,
            min_set_ev=args.or_min_set_ev,
            min_set_profit_factor=args.or_min_set_pf,
            max_candidate_pool=args.or_candidate_pool,
        )

        if len(summary_df):
            all_summary_rows.append(summary_df)
        if len(steps_df):
            all_step_rows.append(steps_df)

    candidate_df = pd.concat(all_candidate_rows, ignore_index=True) if all_candidate_rows else pd.DataFrame()
    summary_df = pd.concat(all_summary_rows, ignore_index=True) if all_summary_rows else pd.DataFrame()
    steps_df = pd.concat(all_step_rows, ignore_index=True) if all_step_rows else pd.DataFrame()
    return candidate_df, summary_df, steps_df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="feature_selector_v35_usefulness_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--top-n-usage", type=int, default=80)

    parser.add_argument("--corr-threshold", type=float, default=0.90)
    parser.add_argument("--allow-correlated-in-rule", action="store_true")

    # EV/OR ruleset options. These do not change beam search itself;
    # they post-process final rules into practical OR rulesets.
    parser.add_argument("--or-min-valid-count", type=int, default=30)
    parser.add_argument("--or-min-rule-ev", type=float, default=2.0)
    parser.add_argument("--or-min-rule-pf", type=float, default=2.0)
    parser.add_argument("--or-min-rule-win-rate", type=float, default=0.0)
    parser.add_argument("--or-max-stop-within-2d", type=float, default=None)
    parser.add_argument("--or-max-rules", type=int, default=20)
    parser.add_argument("--or-target-coverage", type=float, default=0.05)
    parser.add_argument("--or-min-incremental-count", type=int, default=10)
    parser.add_argument("--or-min-set-ev", type=float, default=1.0)
    parser.add_argument("--or-min-set-pf", type=float, default=1.20)
    parser.add_argument("--or-candidate-pool", type=int, default=200)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    date_col = args.date_col or find_date_col(df)

    df = prepare_df(
        df=df,
        target_col=args.target,
        date_col=date_col,
    )

    features = [f for f in DEFAULT_FEATURES if f in df.columns]
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
    print("[SCRIPT_VERSION] feature_selector_v37_or_ev_features20_lower_wick")
    print("[INFO] goal: feature usefulness + valid precision improvement")
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

    single_bin_df = single_feature_bin_report(train, features, args.target)
    single_bin_df.to_csv(
        os.path.join(args.out, "01b_single_feature_bin_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    overall_ev_df = pd.concat([
        target_ev_overall_report(train, "train"),
        target_ev_overall_report(valid, "valid"),
        target_ev_overall_report(df, "all"),
    ], ignore_index=True)
    overall_ev_df.to_csv(
        os.path.join(args.out, "12_target10_15_overall_ev.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    timing_df = pd.concat([
        timing_distribution_report(train, "train"),
        timing_distribution_report(valid, "valid"),
        timing_distribution_report(df, "all"),
    ], ignore_index=True)
    timing_df.to_csv(
        os.path.join(args.out, "13_target10_15_timing_distribution.csv"),
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

    rule_ev_df = rule_ev_report(all_rules, train, valid)
    rule_ev_df.to_csv(
        os.path.join(args.out, "14_rule_target10_15_ev_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    raw_rule_ev_df = rule_ev_report(all_raw_rules, train, valid)
    raw_rule_ev_df.to_csv(
        os.path.join(args.out, "15_raw_rule_target10_15_ev_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    or_candidate_df, or_summary_df, or_steps_df = build_or_rulesets_from_ev(
        all_rules=all_rules,
        rule_ev_df=rule_ev_df,
        train=train,
        valid=valid,
        args=args,
    )

    or_candidate_df.to_csv(
        os.path.join(args.out, "16_or_candidate_rules_ev_filtered.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    or_summary_df.to_csv(
        os.path.join(args.out, "17_or_ruleset_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    or_steps_df.to_csv(
        os.path.join(args.out, "18_or_ruleset_steps.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if len(rules_df) == 0:
        print("[WARN] No final rules found. Try higher --beam-width or lower thresholds in make_profiles().")
        return

    usage_df = feature_usage_from_rules(
        rules_df=rules_df,
        features=features,
        top_n=args.top_n_usage,
    )

    usage_df.to_csv(
        os.path.join(args.out, "04_feature_usage_by_profile.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    direction_df = direction_stability_report(
        rules_df=rules_df,
        features=features,
        top_n=args.top_n_usage,
    )

    direction_df.to_csv(
        os.path.join(args.out, "05_feature_direction_stability.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    monthly_df = monthly_rule_stability(
        valid=valid,
        rules=all_rules,
        target_col=args.target,
    )

    monthly_df.to_csv(
        os.path.join(args.out, "06_monthly_rule_stability.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    monthly_summary_df = summarize_monthly_stability(monthly_df)

    monthly_summary_df.to_csv(
        os.path.join(args.out, "07_monthly_rule_stability_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    contrib_df = conditional_feature_contribution(
        train=train,
        valid=valid,
        rules=all_rules,
        target_col=args.target,
    )

    contrib_df.to_csv(
        os.path.join(args.out, "08_conditional_feature_contribution.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    conditional_summary_df = summarize_conditional_contribution(contrib_df)

    conditional_summary_df.to_csv(
        os.path.join(args.out, "09_conditional_feature_contribution_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    feature_grade_df = grade_features(
        features=features,
        single_df=single_df,
        usage_df=usage_df,
        direction_df=direction_df,
        conditional_summary_df=conditional_summary_df,
        monthly_df=monthly_df,
    )

    feature_grade_df.to_csv(
        os.path.join(args.out, "10_feature_usefulness_grade.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    write_feature_grade_text(feature_grade_df, args.out)

    print("\n[TOP FINAL RULES]")
    show_cols = [
        "profile",
        "rank",
        "rule",
        "features",
        "train_count",
        "train_precision",
        "train_lift",
        "valid_count",
        "valid_precision",
        "valid_lift",
        "pass_valid_55",
        "pass_valid_60",
    ]
    show_cols = [c for c in show_cols if c in rules_df.columns]
    print(rules_df[show_cols].head(40).to_string(index=False))

    print("\n[FEATURE USEFULNESS GRADES]")
    show_cols = [
        "feature",
        "grade",
        "role",
        "action",
        "score",
        "reason",
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

    print("\n[TARGET 10/15 OVERALL EV]")
    if 'overall_ev_df' in locals() and len(overall_ev_df):
        print(overall_ev_df.to_string(index=False))

    print("\n[TOP RULE EV - VALID]")
    if 'rule_ev_df' in locals() and len(rule_ev_df):
        ev_show = rule_ev_df[(rule_ev_df["split"] == "valid")].sort_values(
            ["target_level_pct", "ev_conservative_pct", "profit_factor"],
            ascending=[True, False, False],
        )
        show_cols = [
            "profile", "rank", "target_level_pct", "count", "win_rate", "loss_rate",
            "profit_factor", "ev_conservative_pct", "avg_day_to_target_hit_only",
            "stop_within_2d_rate", "rule",
        ]
        show_cols = [c for c in show_cols if c in ev_show.columns]
        print(ev_show[show_cols].head(30).to_string(index=False))

    print("\n[OR RULESET SUMMARY]")
    if 'or_summary_df' in locals() and len(or_summary_df):
        show_cols = [
            "ruleset_name", "split", "target_level_pct", "n_rules", "count",
            "coverage", "win_rate", "loss_rate", "profit_factor",
            "ev_conservative_pct", "ev_same_day_as_win_pct", "rules",
        ]
        show_cols = [c for c in show_cols if c in or_summary_df.columns]
        print(or_summary_df[show_cols].to_string(index=False))
    else:
        print("No OR ruleset built. Try lowering --or-min-rule-ev/--or-min-rule-pf or --or-min-valid-count.")

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
python low\feature_selector_v37_or_ev.py ^
  --csv csv\low_result_7_v2_desc.csv ^
  --out eature_selector_v37_or_ev ^
  --date-col today ^
  --max-depth 5 ^
  --beam-width 500 ^
  --top-k 150 ^
  --top-n-usage 80 ^
  --corr-threshold 0.90

더 넓게:
python low\feature_selector_v37_or_ev.py ^
  --csv csv\low_result_7_v2_desc.csv ^
  --out eature_selector_v37_or_ev_wide ^
  --date-col today ^
  --max-depth 6 ^
  --beam-width 3000 ^
  --top-k 200 ^
  --top-n-usage 100 ^
  --corr-threshold 0.90
"""