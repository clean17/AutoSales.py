import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd


TARGET_COL = "target_before_stop_7"


DEFAULT_FEATURES = [
    "gap_pct",
    "today_pct",
    "dist_to_ma5",

    "tr_val_rank_20d",
    "today_tr_val_eok",
    "vol_ratio_5_15",
    "vol5",
    "tr_value_ratio_5d",

    "BB_perc",

    "pct_vs_lastweek",
    "ma5_chg_rate",
    "max_drop_7d",

    "lower_wick_ratio",
    "upper_wick_ratio",
    "body_ratio",
    "recent_runup",
    "intraday_return",
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
    profile_name: str

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


def safe_auc(y_true, x):
    y = np.asarray(y_true)
    s = np.asarray(x)

    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0 or len(np.unique(s)) <= 1:
        return np.nan

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)

    tmp = pd.DataFrame({"s": s, "rank": ranks})
    avg_rank = tmp.groupby("s")["rank"].transform("mean").values

    pos_rank_sum = avg_rank[y == 1].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return max(float(auc), float(1 - auc))


def single_feature_report(train, features, target_col):
    rows = []

    base_rate = train[target_col].mean()

    for f in features:
        x = pd.to_numeric(train[f], errors="coerce")
        mask = x.notna()

        xv = x[mask].values
        yv = train.loc[mask, target_col].astype(int).values

        if len(xv) < 100 or len(np.unique(xv)) <= 2:
            rows.append({
                "feature": f,
                "auc_oriented": np.nan,
                "best_bin_lift": np.nan,
                "best_bin_precision": np.nan,
                "best_bin_count": 0,
                "missing_rate": 1 - mask.mean(),
            })
            continue

        auc = safe_auc(yv, xv)

        try:
            bins = pd.qcut(x[mask], q=10, duplicates="drop")
        except Exception:
            bins = pd.cut(x[mask], bins=10, duplicates="drop")

        tmp = pd.DataFrame({
            "bin": bins,
            "y": yv,
        })

        bin_stats = tmp.groupby("bin", observed=False)["y"].agg(
            ["count", "sum", "mean"]
        )

        bin_stats["lift"] = bin_stats["mean"] / base_rate

        best = bin_stats.sort_values(
            ["lift", "mean", "count"],
            ascending=False,
        ).iloc[0]

        rows.append({
            "feature": f,
            "auc_oriented": auc,
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


def make_atoms(train, features, quantiles):
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

    quantiles = profile["quantiles"]
    atoms = make_atoms(train, features, quantiles)

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

                if final_score > -1e17 and valid_m["count"] >= profile["min_valid_count"]:
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


def leave_one_feature_out(
        train,
        valid,
        features,
        target_col,
        baseline_rules_df,
        profiles,
        corr_pairs,
        block_correlated_in_rule,
):
    rows = []

    for profile in profiles:
        profile_rules = baseline_rules_df[
            baseline_rules_df["profile"] == profile["name"]
            ].copy()

        if len(profile_rules) == 0:
            continue

        base_best = profile_rules.sort_values("rank").iloc[0]

        for f in features:
            print(f"[INFO][{profile['name']}] leave-one-out remove: {f}")

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
                    "profile": profile["name"],
                    "feature": f,
                    "best_rule_without": "",
                    "train_score_drop": np.nan,
                    "train_precision_drop": np.nan,
                    "valid_precision_drop": np.nan,
                    "train_lift_drop": np.nan,
                    "valid_lift_drop": np.nan,
                })
                continue

            best = rules_df.sort_values("rank").iloc[0]

            rows.append({
                "profile": profile["name"],
                "feature": f,
                "best_rule_without": best["rule"],
                "train_score_drop": base_best["train_score"] - best["train_score"],
                "train_precision_drop": base_best["train_precision"] - best["train_precision"],
                "valid_precision_drop": base_best["valid_precision"] - best["valid_precision"],
                "train_lift_drop": base_best["train_lift"] - best["train_lift"],
                "valid_lift_drop": base_best["valid_lift"] - best["valid_lift"],
                "best_train_precision_without": best["train_precision"],
                "best_valid_precision_without": best["valid_precision"],
                "best_train_lift_without": best["train_lift"],
                "best_valid_lift_without": best["valid_lift"],
            })

    return pd.DataFrame(rows)


def aggregate_feature_selection(
        features,
        single_df,
        usage_df,
        loo_df,
):
    rows = []

    for f in features:
        u = usage_df[usage_df["feature"] == f].copy()
        l = loo_df[loo_df["feature"] == f].copy()

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
            mean_train_score_drop = l["train_score_drop"].mean()
            max_train_score_drop = l["train_score_drop"].max()
            mean_train_precision_drop = l["train_precision_drop"].mean()
            mean_valid_precision_drop = l["valid_precision_drop"].mean()
            max_valid_precision_drop = l["valid_precision_drop"].max()
        else:
            mean_train_score_drop = 0.0
            max_train_score_drop = 0.0
            mean_train_precision_drop = 0.0
            mean_valid_precision_drop = 0.0
            max_valid_precision_drop = 0.0

        s = single_df[single_df["feature"] == f]
        if len(s):
            auc = s.iloc[0]["auc_oriented"]
            best_bin_lift = s.iloc[0]["best_bin_lift"]
        else:
            auc = np.nan
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

        if np.isfinite(mean_train_score_drop):
            score += max(0.0, mean_train_score_drop) * 0.8

        if np.isfinite(mean_valid_precision_drop):
            score += max(0.0, mean_valid_precision_drop) * 40.0

        if np.isfinite(auc):
            score += max(0.0, auc - 0.5) * 15.0

        if np.isfinite(best_bin_lift):
            score += max(0.0, best_bin_lift - 1.0) * 5.0

        strong_usage = profiles_used >= 2 and total_usage_count >= 20

        strong_drop = (
                              np.isfinite(mean_train_score_drop)
                              and mean_train_score_drop >= 2.0
                      ) or (
                              np.isfinite(mean_valid_precision_drop)
                              and mean_valid_precision_drop >= 0.03
                      )

        valid_stable = (
                               np.isfinite(avg_valid_lift)
                               and avg_valid_lift >= 1.25
                       ) and (
                               not np.isfinite(avg_gap)
                               or avg_gap <= 0.10
                       )

        weak = (
                total_usage_count == 0
                and (
                        not np.isfinite(mean_train_score_drop)
                        or mean_train_score_drop <= 0
                )
                and (
                        not np.isfinite(mean_valid_precision_drop)
                        or mean_valid_precision_drop <= 0
                )
        )

        # CORE 조건 강화:
        # valid precision drop이 양수여야 진짜 핵심으로 본다.
        if (
                strong_usage
                and strong_drop
                and valid_stable
                and np.isfinite(mean_valid_precision_drop)
                and mean_valid_precision_drop >= 0.01
        ):
            judgement = "KEEP_CORE"
            reason = "여러 설정에서 반복 사용되고 제거 시 TRAIN/VALID 성능 하락이 확인됨"

        elif strong_usage and valid_stable:
            judgement = "KEEP_USEFUL"
            reason = "여러 설정에서 반복 사용되고 VALID 성능도 유지됨"

        elif total_usage_count > 0 and valid_stable:
            judgement = "CONDITIONAL"
            reason = "특정 설정/룰에서만 유용하므로 보조 피쳐로 유지"

        elif weak:
            judgement = "REMOVE_CANDIDATE"
            reason = "상위 룰 사용이 없고 제거 영향도 거의 없음"

        else:
            judgement = "CHECK_MANUALLY"
            reason = "사용 빈도, 제거 영향, VALID 안정성이 애매함"

        rows.append({
            "feature": f,
            "selection_score": score,
            "judgement": judgement,
            "reason": reason,
            "profiles_used": profiles_used,
            "total_usage_count": total_usage_count,
            "avg_usage_rate": avg_usage_rate,
            "avg_train_lift_when_used": avg_train_lift,
            "avg_valid_lift_when_used": avg_valid_lift,
            "avg_valid_precision_when_used": avg_valid_precision,
            "avg_abs_precision_gap_when_used": avg_gap,
            "mean_train_score_drop": mean_train_score_drop,
            "max_train_score_drop": max_train_score_drop,
            "mean_train_precision_drop": mean_train_precision_drop,
            "mean_valid_precision_drop": mean_valid_precision_drop,
            "max_valid_precision_drop": max_valid_precision_drop,
            "auc_oriented": auc,
            "best_bin_lift": best_bin_lift,
        })

    out = pd.DataFrame(rows)

    order_map = {
        "KEEP_CORE": 1,
        "KEEP_USEFUL": 2,
        "CONDITIONAL": 3,
        "CHECK_MANUALLY": 4,
        "REMOVE_CANDIDATE": 5,
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

    for _, r in high_corr_df.iterrows():
        a = r["feature_a"]
        b = r["feature_b"]
        corr = r["spearman_corr"]
        abs_corr = r["abs_corr"]

        if a not in score_map or b not in score_map:
            continue

        score_a = score_map.get(a, -1e18)
        score_b = score_map.get(b, -1e18)

        judge_a = judgement_map.get(a, "")
        judge_b = judgement_map.get(b, "")

        # 핵심 피쳐는 약간 보호.
        # 둘 다 CORE가 아니면 점수 높은 쪽 유지.
        core_order = {
            "KEEP_CORE": 4,
            "KEEP_USEFUL": 3,
            "CONDITIONAL": 2,
            "CHECK_MANUALLY": 1,
            "REMOVE_CANDIDATE": 0,
        }

        rank_a = core_order.get(judge_a, 0)
        rank_b = core_order.get(judge_b, 0)

        if rank_a > rank_b:
            winner, loser = a, b
        elif rank_b > rank_a:
            winner, loser = b, a
        else:
            if score_a >= score_b:
                winner, loser = a, b
            else:
                winner, loser = b, a

        # 이미 더 강한 이유로 DROP된 피쳐는 유지
        if keep_status.get(loser) != "CORRELATED_DROP_CANDIDATE":
            keep_status[loser] = "CORRELATED_DROP_CANDIDATE"
            reasons[loser] = (
                f"{winner}와 상관 {corr:.3f}. "
                f"{winner}의 judgement/score가 더 우수하여 중복 후보로 분류"
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
        },
    ]


def print_recommended_features(final_df):
    usable = final_df[
        final_df["judgement"].isin([
            "KEEP_CORE",
            "KEEP_USEFUL",
            "CONDITIONAL",
        ])
    ].copy()

    usable = usable[
        usable["corr_prune_status"] != "CORRELATED_DROP_CANDIDATE"
        ]

    remove = final_df[
        final_df["judgement"].isin(["REMOVE_CANDIDATE"])
    ]["feature"].tolist()

    correlated_drop = final_df[
        final_df["corr_prune_status"] == "CORRELATED_DROP_CANDIDATE"
        ]["feature"].tolist()

    print("\n[RECOMMENDED DEFAULT_FEATURES]")
    print("DEFAULT_FEATURES = [")
    for f in usable["feature"].tolist():
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
    parser.add_argument("--out", default="feature_selector_out")
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.30)

    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--top-n-usage", type=int, default=50)

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

    if args.skip_loo:
        loo_df = pd.DataFrame(columns=["profile", "feature"])
    else:
        loo_df = leave_one_feature_out(
            train=train,
            valid=valid,
            features=features,
            target_col=args.target,
            baseline_rules_df=rules_df,
            profiles=profiles,
            corr_pairs=corr_pairs,
            block_correlated_in_rule=block_correlated_in_rule,
        )

    loo_df.to_csv(
        os.path.join(args.out, "05_leave_one_feature_out_by_profile.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    final_df = aggregate_feature_selection(
        features=features,
        single_df=single_df,
        usage_df=usage_df,
        loo_df=loo_df,
    )

    final_df, corr_prune_df = correlated_feature_pruning(
        final_df=final_df,
        high_corr_df=high_corr_df,
    )

    final_df.to_csv(
        os.path.join(args.out, "06_final_feature_selection.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    corr_prune_df.to_csv(
        os.path.join(args.out, "07_correlated_feature_pruning.csv"),
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
        "mean_train_score_drop",
        "mean_valid_precision_drop",
        "auc_oriented",
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