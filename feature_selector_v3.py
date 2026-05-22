import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


TARGET_COL = "target_before_stop_7"


# ============================================================
# Feature candidates
# 현재 7_find_low_point_processPool.py 의 rule_features 키 기준
# ============================================================
DEFAULT_FEATURES = [
    # ============================================================
    # 1) 변동성 / 반등 강도
    # ============================================================
    "vol5",
    "vol_ratio_5_15",
    "today_pct",
    "max_drop_7d",

    # ============================================================
    # 2) 갭 / 위치 / 단기 회복
    # ============================================================
    "gap_pct",
    "pct_vs_lastweek",
    "dist_to_ma5",

    # ============================================================
    # 3) 거래대금
    # ============================================================
    "today_tr_val_eok",

    # ============================================================
    # 4) 밴드 / 캔들
    # ============================================================
    "BB_perc",
    "lower_wick_ratio",
    "upper_wick_ratio",
    "body_ratio",
    "recent_runup",
    "intraday_return",

    # ============================================================
    # 5) 저점 반등 / 상승 여력
    # ============================================================
    "rebound_from_7d_low",
    "price_power_value",
    "room_to_20d_high",
    "room_to_60d_high",

    # ============================================================
    # 6) 시장 상태
    # ============================================================
    "market_today_pct",
    "market_5d_pct",
    "market_breadth_up_ratio",
]


# 강제 제거가 아님.
# 여기에 넣으면 selection_score에서 -5점만 감점된다.
# 지금은 피쳐를 공정하게 보기 위해 비워둔다.
REMOVE_CANDIDATES_HINT = []


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
    profile_name: str

    def name(self):
        return " AND ".join([a.name() for a in self.atoms])

    def features(self):
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


def split_train_valid(df, date_col, valid_ratio):
    df = df.copy()

    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
        print(f"[INFO] sorted by date_col: {date_col}")
    else:
        df = df.reset_index(drop=True)
        print("[WARN] date_col not found. Splitting by current row order.")

    split_idx = int(len(df) * (1 - valid_ratio))

    # 중요:
    # valid가 원본 index를 유지하면 monthly_rule_stability에서
    # numpy mask position과 DataFrame index label이 충돌할 수 있다.
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
                "best_bin_lift": np.nan,
                "best_bin_precision": np.nan,
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
            "best_bin_lift": best["lift"],
            "best_bin_precision": best["mean"],
            "best_bin_count": best["count"],
            "missing_rate": 1 - mask.mean(),
        })

    return pd.DataFrame(rows)


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
        return x >= atom.threshold

    if atom.op == "<=":
        return x <= atom.threshold

    raise ValueError(atom.op)


def apply_rule(df, atoms):
    mask = np.ones(len(df), dtype=bool)

    for atom in atoms:
        mask &= apply_atom(df, atom).fillna(False).values

    return mask


def make_atoms(
        train,
        features,
        quantiles,
        target_col=None,
        min_atom_count=80,
        min_atom_lift=1.03,
):
    atoms = []

    y = None
    base_rate = None

    if target_col is not None and target_col in train.columns:
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
        qs = sorted(set([float(q) for q in qs if np.isfinite(q)]))

        for q in qs:
            for op in [">=", "<="]:
                atom = Atom(f, op, q)

                if y is not None and base_rate and base_rate > 0:
                    mask = apply_atom(train, atom).fillna(False).values
                    count = int(mask.sum())

                    if count < min_atom_count:
                        continue

                    precision = float(y[mask].mean()) if count else np.nan
                    lift = precision / base_rate if base_rate > 0 else np.nan

                    if not np.isfinite(lift) or lift < min_atom_lift:
                        continue

                atoms.append(atom)

    return atoms


def train_rule_score(
        train_metrics,
        min_train_count,
        min_train_lift,
        max_train_coverage,
):
    count = train_metrics["count"]
    target_count = train_metrics["target_count"]
    precision = train_metrics["precision"]
    lift = train_metrics["lift"]
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

    wide_penalty = 0.0
    if coverage > max_train_coverage:
        wide_penalty = (coverage - max_train_coverage) * 150.0

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
        min_atom_count=profile.get("min_atom_count", 80),
        min_atom_lift=profile.get("min_atom_lift", 1.03),
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
                    max_train_coverage=profile["max_train_coverage"],
                )

                valid_precision_ok = (
                        np.isfinite(valid_m["precision"])
                        and valid_m["precision"] >= profile.get("min_valid_precision", 0.0)
                )

                valid_lift_ok = (
                        np.isfinite(valid_m["lift"])
                        and valid_m["lift"] >= profile.get("min_valid_lift", 1.0)
                )

                if (
                        final_score > -1e17
                        and valid_m["count"] >= profile["min_valid_count"]
                        and valid_precision_ok
                        and valid_lift_ok
                ):
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
            key=lambda r: r.train_score,
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

    unique = {}
    for r in final_rules:
        unique[(r.profile_name, r.name())] = r

    final_rules = list(unique.values())
    final_rules = sorted(final_rules, key=lambda r: r.train_score, reverse=True)

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

        rows.append(row)

    return pd.DataFrame(rows)


def feature_usage_from_rules(rules_df, features, top_n):
    rows = []

    for profile_name, g in rules_df.groupby("profile"):
        top = g.sort_values("rank").head(top_n)

        for f in features:
            used = top["features"].fillna("").apply(
                lambda s: f in [x.strip() for x in s.split(",") if x.strip()]
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
                row["best_rank_when_used"] = used_df["rank"].min()
            else:
                row["avg_train_precision_when_used"] = np.nan
                row["avg_valid_precision_when_used"] = np.nan
                row["avg_train_lift_when_used"] = np.nan
                row["avg_valid_lift_when_used"] = np.nan
                row["avg_abs_precision_gap_when_used"] = np.nan
                row["best_rank_when_used"] = np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_top_rules(rules_df, top_n):
    rows = []

    for profile_name, g in rules_df.groupby("profile"):
        top = g.sort_values("rank").head(top_n)

        rows.append({
            "profile": profile_name,
            "top_n": len(top),
            "mean_train_score": top["train_score"].mean(),
            "mean_train_count": top["train_count"].mean(),
            "mean_valid_count": top["valid_count"].mean(),
            "mean_train_precision": top["train_precision"].mean(),
            "mean_valid_precision": top["valid_precision"].mean(),
            "mean_train_lift": top["train_lift"].mean(),
            "mean_valid_lift": top["valid_lift"].mean(),
            "mean_abs_precision_gap": top["abs_precision_gap"].mean(),
            "median_valid_precision": top["valid_precision"].median(),
            "median_valid_lift": top["valid_lift"].median(),
        })

    return pd.DataFrame(rows)


def direction_stability_report(rules_df, features, top_n):
    rows = []

    for profile_name, g in rules_df.groupby("profile"):
        top = g.sort_values("rank").head(top_n)

        for f in features:
            op_list = []
            threshold_list = []

            used_rules = 0

            for _, r in top.iterrows():
                rule_text = str(r["rule"])
                atoms = rule_text.split(" AND ")

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


def leave_one_feature_out_topn(
        train,
        valid,
        features,
        target_col,
        baseline_rules_df,
        profiles,
        corr_pairs,
        block_correlated_in_rule,
        top_n,
):
    rows = []

    baseline_summary = summarize_top_rules(
        baseline_rules_df,
        top_n=top_n,
    )

    baseline_map = {
        r["profile"]: r
        for _, r in baseline_summary.iterrows()
    }

    for profile in profiles:
        profile_name = profile["name"]

        if profile_name not in baseline_map:
            continue

        base = baseline_map[profile_name]

        for f in features:
            print(f"[INFO][{profile_name}] topN leave-one-out remove: {f}")

            sub_features = [x for x in features if x != f]

            rules, _ = search_rules_train_only(
                train=train,
                valid=valid,
                features=sub_features,
                target_col=target_col,
                profile=profile,
                corr_pairs=corr_pairs,
                block_correlated_in_rule=block_correlated_in_rule,
            )

            rules_df = rules_to_df(rules)

            if len(rules_df) == 0:
                rows.append({
                    "profile": profile_name,
                    "feature": f,
                    "removed_top_n": 0,
                    "topn_train_precision_drop": np.nan,
                    "topn_valid_precision_drop": np.nan,
                    "topn_train_lift_drop": np.nan,
                    "topn_valid_lift_drop": np.nan,
                    "topn_abs_gap_change": np.nan,
                    "baseline_valid_precision": base["mean_valid_precision"],
                    "removed_valid_precision": np.nan,
                    "baseline_valid_lift": base["mean_valid_lift"],
                    "removed_valid_lift": np.nan,
                })
                continue

            removed_summary = summarize_top_rules(
                rules_df,
                top_n=top_n,
            )

            if len(removed_summary) == 0:
                continue

            rem = removed_summary.iloc[0]

            rows.append({
                "profile": profile_name,
                "feature": f,
                "removed_top_n": rem["top_n"],
                "topn_train_precision_drop": base["mean_train_precision"] - rem["mean_train_precision"],
                "topn_valid_precision_drop": base["mean_valid_precision"] - rem["mean_valid_precision"],
                "topn_train_lift_drop": base["mean_train_lift"] - rem["mean_train_lift"],
                "topn_valid_lift_drop": base["mean_valid_lift"] - rem["mean_valid_lift"],
                "topn_abs_gap_change": rem["mean_abs_precision_gap"] - base["mean_abs_precision_gap"],
                "baseline_valid_precision": base["mean_valid_precision"],
                "removed_valid_precision": rem["mean_valid_precision"],
                "baseline_valid_lift": base["mean_valid_lift"],
                "removed_valid_lift": rem["mean_valid_lift"],
            })

    return pd.DataFrame(rows)


def monthly_rule_stability(valid, rules, target_col, date_col):
    if not date_col or date_col not in valid.columns:
        return pd.DataFrame()

    tmp_valid = valid.copy()
    tmp_valid[date_col] = pd.to_datetime(tmp_valid[date_col], errors="coerce")
    tmp_valid = tmp_valid[tmp_valid[date_col].notna()].copy()

    if len(tmp_valid) == 0:
        return pd.DataFrame()

    # 중요:
    # groupby index label과 numpy position 불일치 방지
    tmp_valid = tmp_valid.reset_index(drop=True)

    tmp_valid["month"] = tmp_valid[date_col].dt.strftime("%Y-%m")

    rows = []

    for rule_idx, r in enumerate(rules, start=1):
        mask_all = apply_rule(tmp_valid, r.atoms)
        mask_all = np.asarray(mask_all).astype(bool)

        for month, idx_values in tmp_valid.groupby("month").groups.items():
            pos = np.array(list(idx_values), dtype=int)

            y_m = tmp_valid.iloc[pos][target_col].astype(int).values
            mask_m = mask_all[pos]

            m = precision_recall_lift(y_m, mask_m)

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


def aggregate_feature_selection(
        features,
        single_df,
        usage_df,
        loo_topn_df,
        direction_df,
):
    rows = []

    for f in features:
        u = usage_df[usage_df["feature"] == f].copy()
        l = loo_topn_df[loo_topn_df["feature"] == f].copy()
        d = direction_df[direction_df["feature"] == f].copy()

        if len(u):
            profiles_used = int((u["top_usage_count"] > 0).sum())
            total_usage_count = int(u["top_usage_count"].sum())
            avg_usage_rate = u["top_usage_rate"].mean()
            avg_train_lift = u["avg_train_lift_when_used"].mean()
            avg_valid_lift = u["avg_valid_lift_when_used"].mean()
            avg_valid_precision = u["avg_valid_precision_when_used"].mean()
            avg_gap = u["avg_abs_precision_gap_when_used"].mean()
        else:
            profiles_used = 0
            total_usage_count = 0
            avg_usage_rate = 0.0
            avg_train_lift = np.nan
            avg_valid_lift = np.nan
            avg_valid_precision = np.nan
            avg_gap = np.nan

        if len(l):
            mean_topn_train_precision_drop = l["topn_train_precision_drop"].mean()
            mean_topn_valid_precision_drop = l["topn_valid_precision_drop"].mean()
            mean_topn_train_lift_drop = l["topn_train_lift_drop"].mean()
            mean_topn_valid_lift_drop = l["topn_valid_lift_drop"].mean()
            mean_topn_abs_gap_change = l["topn_abs_gap_change"].mean()
        else:
            mean_topn_train_precision_drop = 0.0
            mean_topn_valid_precision_drop = 0.0
            mean_topn_train_lift_drop = 0.0
            mean_topn_valid_lift_drop = 0.0
            mean_topn_abs_gap_change = 0.0

        if len(d):
            used_d = d[d["used_rules"] > 0]
            if len(used_d):
                avg_direction_consistency = used_d["direction_consistency"].mean()
                dominant_ops = used_d["dominant_op"].dropna().tolist()
                dominant_op = max(set(dominant_ops), key=dominant_ops.count) if dominant_ops else ""
                median_threshold = used_d["median_threshold"].median()
            else:
                avg_direction_consistency = np.nan
                dominant_op = ""
                median_threshold = np.nan
        else:
            avg_direction_consistency = np.nan
            dominant_op = ""
            median_threshold = np.nan

        s = single_df[single_df["feature"] == f]

        if len(s):
            auc_raw = s.iloc[0].get("auc_raw", np.nan)
            auc_direction = s.iloc[0].get("auc_direction", "")
            auc = s.iloc[0].get("auc_oriented", np.nan)
            best_bin = s.iloc[0].get("best_bin", "")
            best_bin_lift = s.iloc[0].get("best_bin_lift", np.nan)
        else:
            auc_raw = np.nan
            auc_direction = ""
            auc = np.nan
            best_bin = ""
            best_bin_lift = np.nan

        score = 0.0

        score += profiles_used * 8.0
        score += total_usage_count * 0.25
        score += max(0.0, avg_usage_rate) * 10.0

        if np.isfinite(avg_valid_lift):
            score += max(0.0, avg_valid_lift - 1.0) * 20.0

        if np.isfinite(avg_train_lift):
            score += max(0.0, avg_train_lift - 1.0) * 10.0

        if np.isfinite(avg_gap):
            score -= avg_gap * 20.0

        if np.isfinite(mean_topn_valid_precision_drop):
            score += max(0.0, mean_topn_valid_precision_drop) * 80.0

        if np.isfinite(mean_topn_train_precision_drop):
            score += max(0.0, mean_topn_train_precision_drop) * 40.0

        if np.isfinite(mean_topn_valid_lift_drop):
            score += max(0.0, mean_topn_valid_lift_drop) * 15.0

        if np.isfinite(avg_direction_consistency):
            score += max(0.0, avg_direction_consistency - 0.5) * 8.0

        if np.isfinite(auc):
            score += max(0.0, auc - 0.5) * 15.0

        if np.isfinite(best_bin_lift):
            score += max(0.0, best_bin_lift - 1.0) * 5.0

        if f in REMOVE_CANDIDATES_HINT:
            score -= 5.0

        strong_usage = profiles_used >= 2 and total_usage_count >= 20

        stable_direction = (
                np.isfinite(avg_direction_consistency)
                and avg_direction_consistency >= 0.75
        )

        valid_stable = (
                               np.isfinite(avg_valid_lift)
                               and avg_valid_lift >= 1.25
                       ) and (
                               not np.isfinite(avg_gap)
                               or avg_gap <= 0.10
                       )

        necessary = (
                            np.isfinite(mean_topn_valid_precision_drop)
                            and mean_topn_valid_precision_drop >= 0.015
                    ) or (
                            np.isfinite(mean_topn_valid_lift_drop)
                            and mean_topn_valid_lift_drop >= 0.03
                    )

        additive = (
                np.isfinite(avg_valid_lift)
                and avg_valid_lift >= 1.45
                and total_usage_count >= 5
        )

        weak = (
                total_usage_count == 0
                and (
                        not np.isfinite(mean_topn_valid_precision_drop)
                        or mean_topn_valid_precision_drop <= 0
                )
        )

        if strong_usage and necessary and valid_stable and stable_direction:
            judgement = "CORE_NECESSARY"
            reason = "상위 룰 묶음 기준 제거 시 VALID 성능이 하락하고 방향도 안정적"

        elif strong_usage and additive and valid_stable:
            judgement = "USEFUL_ADDITIVE"
            reason = "여러 설정에서 반복 사용되고 사용된 룰의 VALID 성능이 좋음"

        elif total_usage_count > 0 and additive:
            judgement = "CONDITIONAL"
            reason = "특정 룰에서 유용하지만 필수성은 약함"

        elif weak:
            judgement = "REMOVE"
            reason = "상위 룰 사용이 없고 제거 영향도 없음"

        else:
            judgement = "CHECK"
            reason = "사용 빈도, 제거 영향, 방향 안정성이 애매함"

        rows.append({
            "feature": f,
            "judgement": judgement,
            "selection_score": score,
            "reason": reason,

            "profiles_used": profiles_used,
            "total_usage_count": total_usage_count,
            "avg_valid_precision_when_used": avg_valid_precision,
            "avg_valid_lift_when_used": avg_valid_lift,
            "avg_abs_precision_gap_when_used": avg_gap,

            "mean_topn_train_precision_drop": mean_topn_train_precision_drop,
            "mean_topn_valid_precision_drop": mean_topn_valid_precision_drop,
            "mean_topn_train_lift_drop": mean_topn_train_lift_drop,
            "mean_topn_valid_lift_drop": mean_topn_valid_lift_drop,
            "mean_topn_abs_gap_change": mean_topn_abs_gap_change,

            "avg_direction_consistency": avg_direction_consistency,
            "dominant_op": dominant_op,
            "median_threshold": median_threshold,

            "auc_raw": auc_raw,
            "auc_direction": auc_direction,
            "auc_oriented": auc,
            "best_bin": best_bin,
            "best_bin_lift": best_bin_lift,
        })

    out = pd.DataFrame(rows)

    order_map = {
        "CORE_NECESSARY": 1,
        "USEFUL_ADDITIVE": 2,
        "CONDITIONAL": 3,
        "CHECK": 4,
        "REMOVE": 5,
    }

    out["_order"] = out["judgement"].map(order_map).fillna(99)

    out = out.sort_values(
        ["_order", "selection_score"],
        ascending=[True, False],
    )

    out = out.drop(columns=["_order"])

    return out


def correlated_feature_pruning(final_df, high_corr_df):
    if len(high_corr_df) == 0:
        final_df["corr_prune_status"] = "KEEP"
        final_df["corr_prune_reason"] = ""
        return final_df, pd.DataFrame()

    score_map = final_df.set_index("feature")["selection_score"].to_dict()
    judgement_map = final_df.set_index("feature")["judgement"].to_dict()

    keep_status = {f: "KEEP" for f in final_df["feature"]}
    reasons = {f: "" for f in final_df["feature"]}

    rows = []

    judgement_rank = {
        "CORE_NECESSARY": 5,
        "USEFUL_ADDITIVE": 4,
        "CONDITIONAL": 3,
        "CHECK": 2,
        "REMOVE": 1,
    }

    for _, r in high_corr_df.iterrows():
        a = r["feature_a"]
        b = r["feature_b"]
        corr = r["spearman_corr"]
        abs_corr = r["abs_corr"]

        if a not in score_map or b not in score_map:
            continue

        rank_a = judgement_rank.get(judgement_map.get(a, ""), 0)
        rank_b = judgement_rank.get(judgement_map.get(b, ""), 0)

        if rank_a > rank_b:
            winner, loser = a, b
        elif rank_b > rank_a:
            winner, loser = b, a
        else:
            winner, loser = (a, b) if score_map[a] >= score_map[b] else (b, a)

        if keep_status.get(loser) != "CORRELATED_DROP":
            keep_status[loser] = "CORRELATED_DROP"
            reasons[loser] = (
                f"{winner}와 상관 {corr:.3f}. "
                f"{winner}의 judgement/score가 더 우수하여 중복 후보"
            )

        rows.append({
            "feature_a": a,
            "feature_b": b,
            "spearman_corr": corr,
            "abs_corr": abs_corr,
            "winner": winner,
            "drop_candidate": loser,
            "winner_score": score_map[winner],
            "drop_score": score_map[loser],
            "winner_judgement": judgement_map[winner],
            "drop_judgement": judgement_map[loser],
        })

    final_df = final_df.copy()
    final_df["corr_prune_status"] = final_df["feature"].map(keep_status)
    final_df["corr_prune_reason"] = final_df["feature"].map(reasons)

    corr_prune_df = pd.DataFrame(rows)

    return final_df, corr_prune_df


def make_profiles(args):
    base_quantiles = [
        0.05, 0.10, 0.15, 0.20, 0.25,
        0.30, 0.40, 0.50, 0.60, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.95,
    ]

    return [
        {
            "name": "stable",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "min_train_count": 200,
            "min_valid_count": 100,
            "min_train_lift": 1.20,
            "beam_min_train_lift": 1.00,
            "max_train_coverage": 0.50,
            "quantiles": base_quantiles,
            "min_atom_count": 120,
            "min_atom_lift": 1.03,
            "min_valid_precision": 0.55,
            "min_valid_lift": 1.10,
        },
        {
            "name": "precision",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "min_train_count": 100,
            "min_valid_count": 50,
            "min_train_lift": 1.30,
            "beam_min_train_lift": 1.00,
            "max_train_coverage": 0.35,
            "quantiles": base_quantiles,
            "min_atom_count": 80,
            "min_atom_lift": 1.05,
            "min_valid_precision": 0.60,
            "min_valid_lift": 1.20,
        },
        {
            "name": "loose",
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "top_k": args.top_k,
            "min_train_count": 150,
            "min_valid_count": 80,
            "min_train_lift": 1.15,
            "beam_min_train_lift": 1.00,
            "max_train_coverage": 0.60,
            "quantiles": base_quantiles,
            "min_atom_count": 100,
            "min_atom_lift": 1.02,
            "min_valid_precision": 0.53,
            "min_valid_lift": 1.05,
        },
    ]


def print_recommended_features(final_df):
    final_df = final_df.copy()

    keep = final_df[
        final_df["corr_prune_status"] != "CORRELATED_DROP"
        ].copy()

    direction_ok = (
            (
                    (keep["auc_direction"] == "higher_success")
                    & (keep["dominant_op"] == ">=")
            )
            |
            (
                    (keep["auc_direction"] == "lower_success")
                    & (keep["dominant_op"] == "<=")
            )
    )

    default = keep[
        (
                keep["judgement"].isin([
                    "CORE_NECESSARY",
                    "USEFUL_ADDITIVE",
                ])
                & direction_ok
        )
        |
        (
                (keep["judgement"] == "CONDITIONAL")
                & (keep["avg_valid_lift_when_used"] >= 1.40)
                & (keep["avg_valid_precision_when_used"] >= 0.56)
                & direction_ok
        )
        ].copy()

    candidate = keep[
        ~keep["feature"].isin(default["feature"])
        & keep["judgement"].isin([
            "CONDITIONAL",
            "CHECK",
        ])
        ].copy()

    remove = final_df[
        final_df["judgement"].isin(["REMOVE"])
    ]["feature"].tolist()

    correlated_drop = final_df[
        final_df["corr_prune_status"] == "CORRELATED_DROP"
        ]["feature"].tolist()

    print("\n[RECOMMENDED DEFAULT_FEATURES]")
    print("DEFAULT_FEATURES = [")
    for f in default["feature"].tolist():
        print(f'    "{f}",')
    print("]")

    print("\n[CANDIDATE_FEATURES]")
    print("CANDIDATE_FEATURES = [")
    for f in candidate["feature"].tolist():
        print(f'    "{f}",')
    print("]")

    print("\n[REMOVE_CANDIDATES]")
    print("REMOVE_CANDIDATES = [")
    for f in remove:
        print(f'    "{f}",')
    print("]")

    print("\n[CORRELATED_DROP_CANDIDATES]")
    print("CORRELATED_DROP_CANDIDATES = [")
    for f in correlated_drop:
        print(f'    "{f}",')
    print("]")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="feature_selector_v3_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--top-n-usage", type=int, default=50)
    parser.add_argument("--top-n-loo", type=int, default=20)

    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--corr-threshold", type=float, default=0.90)

    parser.add_argument(
        "--allow-correlated-in-rule",
        action="store_true",
        help="If set, correlated features can appear together in one rule.",
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)

    if args.target not in df.columns:
        raise ValueError(f"target column not found: {args.target}")

    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df[df[args.target].isin([0, 1])].copy()
    df[args.target] = df[args.target].astype(int)

    features = [f for f in DEFAULT_FEATURES if f in df.columns]

    missing_features = [f for f in DEFAULT_FEATURES if f not in df.columns]

    if missing_features:
        print("[WARN] missing features skipped:", missing_features)

    if not features:
        raise ValueError("No usable features found.")

    date_col = args.date_col or find_date_col(df)

    print("=" * 80)
    print("[INFO] rows:", len(df))
    print("[INFO] target:", args.target)
    print("[INFO] target_rate:", df[args.target].mean())
    print("[INFO] date_col:", date_col)
    print("[INFO] features:", features)
    print("[INFO] corr_threshold:", args.corr_threshold)
    print("[INFO] allow_correlated_in_rule:", args.allow_correlated_in_rule)
    print("=" * 80)

    train, valid = split_train_valid(df, date_col, args.valid_ratio)

    print("[INFO] train rows:", len(train), "target_rate:", train[args.target].mean())
    print("[INFO] valid rows:", len(valid), "target_rate:", valid[args.target].mean())

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

    if len(high_corr_df):
        print("\n[HIGH CORR PAIRS]")
        print(high_corr_df.to_string(index=False))
    else:
        print("\n[HIGH CORR PAIRS] none")

    profiles = make_profiles(args)

    single_df = single_feature_report(train, features, args.target)
    single_df.to_csv(
        os.path.join(args.out, "01_single_feature_report.csv"),
        index=False,
        encoding="utf-8-sig",
    )

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

    monthly_df = monthly_rule_stability(
        valid=valid,
        rules=all_rules,
        target_col=args.target,
        date_col=date_col,
    )

    monthly_df.to_csv(
        os.path.join(args.out, "09_monthly_rule_stability.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    monthly_summary_df = summarize_monthly_stability(monthly_df)

    monthly_summary_df.to_csv(
        os.path.join(args.out, "10_monthly_rule_stability_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

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

    if args.skip_loo:
        loo_topn_df = pd.DataFrame(columns=["profile", "feature"])
    else:
        loo_topn_df = leave_one_feature_out_topn(
            train=train,
            valid=valid,
            features=features,
            target_col=args.target,
            baseline_rules_df=rules_df,
            profiles=profiles,
            corr_pairs=corr_pairs,
            block_correlated_in_rule=block_correlated_in_rule,
            top_n=args.top_n_loo,
        )

    loo_topn_df.to_csv(
        os.path.join(args.out, "06_leave_one_feature_out_topn.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    final_df = aggregate_feature_selection(
        features=features,
        single_df=single_df,
        usage_df=usage_df,
        loo_topn_df=loo_topn_df,
        direction_df=direction_df,
    )

    final_df, corr_prune_df = correlated_feature_pruning(
        final_df=final_df,
        high_corr_df=high_corr_df,
    )

    final_df.to_csv(
        os.path.join(args.out, "07_final_feature_selection.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    corr_prune_df.to_csv(
        os.path.join(args.out, "08_correlated_feature_pruning.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[FINAL FEATURE SELECTION]")

    show_cols = [
        "feature",
        "judgement",
        "corr_prune_status",
        "selection_score",
        "reason",
        "corr_prune_reason",

        "profiles_used",
        "total_usage_count",
        "avg_valid_precision_when_used",
        "avg_valid_lift_when_used",
        "avg_abs_precision_gap_when_used",

        "mean_topn_valid_precision_drop",
        "mean_topn_valid_lift_drop",

        "avg_direction_consistency",
        "dominant_op",
        "median_threshold",

        "auc_raw",
        "auc_direction",
        "auc_oriented",
        "best_bin",
        "best_bin_lift",
    ]

    show_cols = [c for c in show_cols if c in final_df.columns]
    print(final_df[show_cols].to_string(index=False))

    print_recommended_features(final_df)

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
빠른 테스트:
python feature_selector_v3.py \
  --csv csv/low_result_7_desc.csv \
  --out feature_selector_v3_out_fast \
  --date-col today \
  --max-depth 4 \
  --beam-width 250 \
  --top-k 120 \
  --top-n-usage 50 \
  --top-n-loo 20 \
  --corr-threshold 0.90 \
  --skip-loo

최종 분석:
python feature_selector_v3.py \
  --csv csv/low_result_7_desc.csv \
  --out feature_selector_v3_out_final \
  --date-col today \
  --max-depth 4 \
  --beam-width 300 \
  --top-k 150 \
  --top-n-usage 50 \
  --top-n-loo 20 \
  --corr-threshold 0.90
"""