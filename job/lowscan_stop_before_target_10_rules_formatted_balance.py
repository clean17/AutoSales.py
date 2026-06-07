# auto-generated: lowscan bad avoid rules
# source: lowscan_stop_before_target_10_rule_report.csv
# target: stop_before_target_10 == 1
# applied_scenario: precision69_strict_balance
# scenario_valid_rate: 0.695335
# scenario_valid_coverage: 0.140584
# scenario_valid_matched: 686
# scenario_valid_target: 477
# export_note: heuristic top valid_pass rules because report selected column does not match requested scenario; to export exact scenario, script must save per-scenario selected rule names
# purpose: exclude candidates likely to hit stop before target within 10 days
# usage:
#    import numpy as np
#    import lowscan_stop_before_target_10_rules_precision69_strict_balance as lowscan_rules
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
    "rule_015",
    "rule_016",
]

def build_conditions(df):
    conditions = {
        "rule_001":
            (df["ATR_pct"] >= 7.752) &
            (df["dist_to_ma5"] >= 2.1414) &
            (df["room_to_60d_high"] <= 53.3488) &
            (df["vol15"] <= 3.904),
        "rule_002":
            (df["ATR_pct"] >= 9.5312) &
            (df["room_to_60d_high"] <= 53.3488) &
            (df["vol15"] <= 5.78) &
            (df["pct_vs_lastweek"] >= 0.595),
        "rule_003":
            (df["ATR_pct"] >= 7.247) &
            (df["dist_to_ma20"] >= 3.2336) &
            (df["dist_to_ma5"] <= 7.0088) &
            (df["lower_wick_ratio"] >= 0.105),
        "rule_004":
            (df["ATR_pct"] >= 7.752) &
            (df["room_to_60d_high"] <= 53.3488) &
            (df["dist_to_ma5"] >= 1.461) &
            (df["vol15"] <= 3.904),
        "rule_005":
            (df["ATR_pct"] >= 7.752) &
            (df["room_to_60d_high"] <= 61.0866) &
            (df["dist_to_ma5"] >= 2.1414) &
            (df["vol15"] <= 3.904),
        "rule_006":
            (df["ma5_chg_rate"] >= 2.491) &
            (df["ATR_pct"] >= 7.752) &
            (df["vol15"] <= 5.78) &
            (df["vol5"] >= 4.454),
        "rule_007":
            (df["pct_vs_lastweek"] >= 15.0156) &
            (df["gap_pct"] <= 0) &
            (df["rebound_from_7d_low"] <= 26.4828) &
            (df["vol15"] >= 4.436),
        "rule_008":
            (df["pct_vs_lastweek"] >= 15.0156) &
            (df["gap_pct"] <= 0) &
            (df["rebound_from_7d_low"] <= 26.4828) &
            (df["ATR_pct"] >= 5.504),
        "rule_009":
            (df["pct_vs_lastweek"] >= 6.2964) &
            (df["ATR_pct"] >= 8.438) &
            (df["vol15"] <= 5.78) &
            (df["room_to_60d_high"] <= 47.2118),
        "rule_010":
            (df["ATR_pct"] >= 7.752) &
            (df["dist_to_ma5"] >= 1.461) &
            (df["room_to_60d_high"] <= 61.0866) &
            (df["vol15"] <= 3.904),
        "rule_011":
            (df["ATR_pct"] >= 9.5312) &
            (df["room_to_60d_high"] <= 61.0866) &
            (df["vol15"] <= 5.78) &
            (df["ma5_chg_rate"] >= -0.209),
        "rule_012":
            (df["pct_vs_lastweek"] >= 15.0156) &
            (df["gap_pct"] <= 0) &
            (df["rebound_from_7d_low"] <= 26.4828) &
            (df["ATR_pct"] >= 5.1984),
        "rule_013":
            (df["ma5_chg_rate"] >= 2.491) &
            (df["ATR_pct"] >= 7.752) &
            (df["vol15"] <= 5.78) &
            (df["tr_value_ratio_5d"] >= 2.112),
        "rule_014":
            (df["ATR_pct"] >= 9.5312) &
            (df["room_to_60d_high"] <= 53.3488) &
            (df["vol15"] <= 5.78) &
            (df["tr_value_ratio_5d"] >= 0.823),
        "rule_015":
            (df["ma5_chg_rate"] >= 2.491) &
            (df["ATR_pct"] >= 7.752) &
            (df["vol15"] <= 5.78) &
            (df["vol5"] >= 3.299),
        "rule_016":
            (df["ma5_chg_rate"] >= 2.491) &
            (df["ATR_pct"] >= 7.752) &
            (df["vol15"] <= 5.78) &
            (df["vol5"] >= 3.018),
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
