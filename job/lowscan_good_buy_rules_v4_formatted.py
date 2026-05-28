# auto-generated: lowscan good buy rules
# source: lowscan_positive_rules_v4_trial.py
# split: train / selection_valid / final_test applied
# purpose: positive buy mask for target_before_stop_7 == 1
# usage:
#    import lowscan_positive_rules_v4_good_buy_rules as lowscan_rules
#    buy_conditions = lowscan_rules.build_conditions(df)
#
#    buy_mask = np.zeros(len(df), dtype=bool)
#    for cond in buy_conditions.values():
#        buy_mask |= cond
#
#    df = df[buy_mask].copy()

import numpy as np

RULE_NAMES = [
    "rule_001_precision_high",
    "rule_002_stable_forward",
    "rule_003_coverage_expand",
]

def build_conditions(df):
    conditions = {
        # final_test precision 최고 룰
        # final_count=49, final_precision=65.31%, final_lift=1.576x
        "rule_001_precision_high":
            (df["BB_perc"] >= 1.054) &
            (df["dist_to_ma5"] >= 20.2474) &
            (df["ma5_chg_rate"] <= 10.0) &
            (df["price_power_value"] >= 57.1494) &
            (df["room_to_60d_high"] <= 18.9896),

        # fixed forward 안정성 최고 룰
        # final_count=62, final_precision=61.29%, final_lift=1.479x
        # fw_mean=73.10%, fw_min=66.67%, fw_pass_split_rate=100%
        "rule_002_stable_forward":
            (df["dist_to_ma5"] >= 20.2474) &
            (df["ma5_chg_rate"] <= 10.0) &
            (df["room_to_60d_high"] <= 26.0304) &
            (df["vol5"] >= 10.6824),

        # final_test count 확장형 룰
        # final_count=89, final_precision=58.43%, final_lift=1.410x
        "rule_003_coverage_expand":
            (df["gap_pct"] >= -0.962) &
            (df["pct_vs_lastweek"] >= 11.1886) &
            (df["rebound_from_7d_low"] >= 33.1722) &
            (df["room_to_60d_high"] <= 58.511) &
            (df["upper_wick_ratio"] <= 0.02),
    }
    return conditions

def build_mask(df):
    mask = np.zeros(len(df), dtype=bool)
    for cond in build_conditions(df).values():
        mask |= cond
    return mask

def build_rule_name_series(df, sep=","):
    conditions = build_conditions(df)
    names = []
    for i in range(len(df)):
        matched = [name for name, cond in conditions.items() if bool(cond.iloc[i] if hasattr(cond, "iloc") else cond[i])]
        names.append(sep.join(matched))
    return names
