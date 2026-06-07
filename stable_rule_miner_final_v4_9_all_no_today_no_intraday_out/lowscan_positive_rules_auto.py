# auto-generated: lowscan good buy rules
# source: stable_rule_miner_final_v4_9_mine_and_export_rules.py
# feature_set: all_no_today_no_intraday
# split: train / selection_valid / final_test applied
# target: target_before_stop_10 == 1
# final filter: precision >= 0.50, avg_return >= 2.00
# excluded features: today_pct, price_power_value, body_value_power, intraday_return
# usage:
#    import lowscan_auto_buy_rules as lowscan_rules
#    buy_conditions = lowscan_rules.build_conditions(df)
#    buy_mask = lowscan_rules.build_mask(df)
#    df_buy = df[buy_mask].copy()

from __future__ import annotations
import numpy as np

RULE_NAMES = [
    "rule_001_precision_high",
    "rule_002_stable_forward",
    "rule_003_coverage_expand",
]

EXCLUDED_FEATURES = {
    "today_pct",
    "price_power_value",
    "body_value_power",
    "intraday_return",
}

REQUIRED_COLUMNS = [
    "dist_to_ma20",
    "gap_pct",
    "lower_wick_ratio",
    "max_drop_7d",
    "upper_wick_ratio",
    "vol15",
    "vol5",
]

RULE_META = {
    "rule_001_precision_high": {
        "style": "HIGH_PRECISION",
        "description": "고확률형: final precision / 기대수익률 우선 룰",
        "target": "target_before_stop_10",
        "selection_selected_count": 27,
        "selection_precision": 0.8148148148,
        "selection_avg_return": 7.037037037,
        "final_selected_count": 72,
        "final_precision": 0.5694444444,
        "final_avg_return": 3.2966527778,
        "fw_mean_precision": 0.6,
        "fw_min_precision": 0.5,
        "fw_mean_return": 3.6,
        "fw_min_return": 2,
        "fw_pass_split_rate": 1,
        "fw_crash_split_count": 0,
    },
    "rule_002_stable_forward": {
        "style": "STABLE_FORWARD",
        "description": "안정형: forward 안정성 / crash 최소화 우선 룰",
        "target": "target_before_stop_10",
        "selection_selected_count": 33,
        "selection_precision": 0.8181818182,
        "selection_avg_return": 7.0909090909,
        "final_selected_count": 133,
        "final_precision": 0.5112781955,
        "final_avg_return": 2.2255639098,
        "fw_mean_precision": 0.6265546772,
        "fw_min_precision": 0.5454545455,
        "fw_mean_return": 4.0248748353,
        "fw_min_return": 2.7272727273,
        "fw_pass_split_rate": 1,
        "fw_crash_split_count": 0,
    },
    "rule_003_coverage_expand": {
        "style": "COVERAGE_EXPAND",
        "description": "커버리지형: final count / coverage 확장 우선 룰",
        "target": "target_before_stop_10",
        "selection_selected_count": 35,
        "selection_precision": 0.8285714286,
        "selection_avg_return": 7.2571428571,
        "final_selected_count": 134,
        "final_precision": 0.5223880597,
        "final_avg_return": 2.4029850746,
        "fw_mean_precision": 0.6445745076,
        "fw_min_precision": 0.5,
        "fw_mean_return": 4.3131921219,
        "fw_min_return": 2,
        "fw_pass_split_rate": 1,
        "fw_crash_split_count": 0,
    },
}

def _require_columns(df, columns=REQUIRED_COLUMNS):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f'Missing required columns for lowscan rules: {missing}')

def build_conditions(df, validate: bool = True):
    if validate:
        _require_columns(df)
    conditions = {
        # HIGH_PRECISION
        # final_count=72, final_precision=56.94%, final_avg_return=+3.30%
        # fw_pass_split_rate=100.00%, fw_crash_split_count=0
        "rule_001_precision_high":
            (df["dist_to_ma20"] >= 19.1457) &
            (df["lower_wick_ratio"] <= 0.2332) &
            (df["max_drop_7d"] <= -3.854) &
            (df["upper_wick_ratio"] <= 0.257) &
            (df["vol15"] >= 6.8021),

        # STABLE_FORWARD
        # final_count=133, final_precision=51.13%, final_avg_return=+2.23%
        # fw_pass_split_rate=100.00%, fw_crash_split_count=0
        "rule_002_stable_forward":
            (df["dist_to_ma20"] >= -16.5991) &
            (df["gap_pct"] >= 0) &
            (df["lower_wick_ratio"] <= 0.2332) &
            (df["upper_wick_ratio"] <= 0.035) &
            (df["vol5"] >= 9.0373),

        # COVERAGE_EXPAND
        # final_count=134, final_precision=52.24%, final_avg_return=+2.40%
        # fw_pass_split_rate=100.00%, fw_crash_split_count=0
        "rule_003_coverage_expand":
            (df["gap_pct"] >= 0) &
            (df["lower_wick_ratio"] <= 0.2332) &
            (df["max_drop_7d"] <= -3.209) &
            (df["upper_wick_ratio"] <= 0.035) &
            (df["vol5"] >= 9.0373),

    }
    return conditions

def build_mask(df, validate: bool = True):
    mask = np.zeros(len(df), dtype=bool)
    for cond in build_conditions(df, validate=validate).values():
        mask |= np.asarray(cond, dtype=bool)
    return mask

def build_rule_name_series(df, sep: str = ',', validate: bool = True):
    conditions = build_conditions(df, validate=validate)
    names = []
    for i in range(len(df)):
        matched = []
        for name, cond in conditions.items():
            if bool(cond.iloc[i] if hasattr(cond, 'iloc') else cond[i]):
                matched.append(name)
        names.append(sep.join(matched))
    return names

def build_rule_count_series(df, validate: bool = True):
    conditions = build_conditions(df, validate=validate)
    count = np.zeros(len(df), dtype=int)
    for cond in conditions.values():
        count += np.asarray(cond, dtype=bool).astype(int)
    return count

def build_priority_score(df, validate: bool = True):
    conditions = build_conditions(df, validate=validate)
    weights = {
        'rule_001_precision_high': 3.0,
        'rule_002_stable_forward': 3.0,
        'rule_003_coverage_expand': 2.0,
    }
    score = np.zeros(len(df), dtype=float)
    for name, cond in conditions.items():
        score += np.asarray(cond, dtype=bool).astype(float) * weights.get(name, 1.0)
    return score

def apply_rules(df, copy: bool = True):
    out = df.copy() if copy else df
    out['lowscan_buy_signal'] = build_mask(out)
    out['lowscan_rule_names'] = build_rule_name_series(out)
    out['lowscan_rule_count'] = build_rule_count_series(out)
    out['lowscan_priority_score'] = build_priority_score(out)
    return out[out['lowscan_buy_signal']].copy() if copy else out

if __name__ == '__main__':
    print('This module is intended to be imported and applied to a pandas DataFrame.')
    for name in RULE_NAMES:
        print(f'- {name}: {RULE_META[name]}')
