# auto-generated: lowscan bad avoid rules
# source: 8-1_find_stop_avoid_rules.py
# target: stop_before_target_10 == 1
# split_date: 2025-11-28
# applied_scenario: depth4_precision69_guard
# valid_rate: 0.699187
# valid_coverage: 0.126732
# valid_matched: 615
# valid_target: 430
# purpose: exclude candidates likely to hit stop before target within 10 days
# target preference: valid_rate >= 68%, valid_coverage >= 13.5% when possible
# usage:
#    import numpy as np
#    import lowscan_stop_before_target_10_rules as lowscan_rules
#    avoid_conditions = lowscan_rules.build_conditions(df)
#
#    avoid_mask = np.zeros(len(df), dtype=bool)
#    for cond in avoid_conditions.values():
#        avoid_mask |= cond
#
#    df = df[~avoid_mask].copy()

import numpy as np

RULE_NAMES = [
    "rule_001",
    "rule_002",
    "rule_003",
    "rule_004",
    "rule_005",
    "rule_006",
    "rule_007",
    "rule_008",
    "rule_009",
    "rule_010",
    "rule_011",
    "rule_012",
    "rule_013",
    "rule_014",
]

def build_conditions(df):
    conditions = {
        "rule_001":
            (df["ma5_chg_rate"] >= 2.491) &
            (df["ATR_pct"] >= 7.752) &
            (df["vol15"] <= 5.78) &
            (df["upper_wick_ratio"] >= 0.047),
        "rule_002":
            (df["ATR_pct"] >= 9.5312) &
            (df["dist_to_ma5"] <= 7.0088) &
            (df["room_to_60d_high"] <= 53.3488) &
            (df["tr_value_ratio_5d"] >= 1.209),
        "rule_003":
            (df["ATR_pct"] >= 7.752) &
            (df["room_to_60d_high"] <= 61.0866) &
            (df["dist_to_ma5"] >= 2.1414) &
            (df["vol15"] <= 3.904),
        "rule_004":
            (df["ma5_chg_rate"] >= 3.349) &
            (df["vol5"] <= 7.378) &
            (df["ATR_pct"] >= 8.438) &
            (df["gap_pct"] <= 1.9498),
        "rule_005":
            (df["dist_to_ma20"] >= 1.7436) &
            (df["ATR_pct"] >= 7.247) &
            (df["tr_value_ratio_5d"] <= 0.823) &
            (df["BB_perc"] <= 0.897),
        "rule_006":
            (df["ATR_pct"] >= 7.752) &
            (df["vol5"] <= 3.559) &
            (df["lower_wick_ratio"] >= 0.105) &
            (df["dist_to_ma20"] >= -4.4644),
        "rule_007":
            (df["ATR_pct"] >= 9.5312) &
            (df["room_to_60d_high"] <= 53.3488) &
            (df["vol15"] <= 5.78) &
            (df["pct_vs_lastweek"] >= 0.595),
        "rule_008":
            (df["ATR_pct"] >= 6.816) &
            (df["dist_to_ma5"] >= 4.5144) &
            (df["max_drop_7d"] >= -4.251) &
            (df["ma5_chg_rate"] <= 1.5096),
        "rule_009":
            (df["dist_to_ma20"] >= 1.7436) &
            (df["ATR_pct"] >= 7.247) &
            (df["rebound_from_7d_low"] <= 21.097) &
            (df["vol5"] >= 6.4598),
        "rule_010":
            (df["ATR_pct"] >= 7.247) &
            (df["dist_to_ma20"] >= 1.7436) &
            (df["dist_to_ma5"] <= 7.0088) &
            (df["lower_wick_ratio"] >= 0.131),
        "rule_011":
            (df["ma5_chg_rate"] >= 3.349) &
            (df["upper_wick_ratio"] >= 0.333) &
            (df["ATR_pct"] >= 9.5312) &
            (df["tr_value_ratio_5d"] >= 1.369),
        "rule_012":
            (df["ATR_pct"] >= 7.752) &
            (df["max_drop_7d"] >= -5.363) &
            (df["lower_wick_ratio"] >= 0.167) &
            (df["ma5_chg_rate"] <= 0.3686),
        "rule_013":
            (df["ATR_pct"] >= 6.816) &
            (df["dist_to_ma5"] >= 11.1128) &
            (df["vol5"] <= 8.8874) &
            (df["dist_to_ma20"] <= 5.099),
        "rule_014":
            (df["ATR_pct"] >= 7.752) &
            (df["tr_value_ratio_5d"] >= 3.3068) &
            (df["rebound_from_7d_low"] <= 26.4828) &
            (df["upper_wick_ratio"] <= 0.333),
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
        matched = [
            name
            for name, cond in conditions.items()
            if bool(cond.iloc[i] if hasattr(cond, "iloc") else cond[i])
        ]
        names.append(sep.join(matched))
    return names
