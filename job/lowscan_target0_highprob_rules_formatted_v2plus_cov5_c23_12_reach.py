# auto-generated: lowscan bad avoid rules
# selected_scenario_export_name: v2plus_cov5_c23_12_reach
# alias: v2plus_cov5
# source: 8-2_find_no_bounce_avoid_rules.py
# target: target_class == 0
# split_date: 2025-11-28
# scenario: v2plus_cov5_c23_12_reach
# alias: v2plus_cov5
# target_coverage: 0.05
# max_class23: 0.12
# max_class3: 0.045
# purpose: exclude no-bounce / target_class 0 candidates while controlling class2/class3 contamination
# usage:
#    import numpy as np
#    import lowscan_target0_highprob_rules_formatted as lowscan_rules
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
    "rule_017",
    "rule_018",
    "rule_019",
    "rule_020",
    "rule_021",
    "rule_022",
    "rule_023",
    "rule_024",
    "rule_025",
    "rule_026",
    "rule_027",
    "rule_028",
    "rule_029",
    "rule_030",
    "rule_031",
    "rule_032",
    "rule_033",
    "rule_034",
    "rule_035",
    "rule_036",
    "rule_037",
    "rule_038",
    "rule_039",
]

def build_conditions(df):
    conditions = {
        "rule_001":
            (df["room_to_60d_high"] <= 22.52421412) &
            (df["ATR_pct"] <= 3.21796) &
            (df["upper_wick_ratio"] <= 0.114) &
            (df["dist_to_ma5"] <= 6.819),
        "rule_002":
            (df["rebound_from_7d_low"] <= 5) &
            (df["upper_wick_ratio"] <= 0.031) &
            (df["vol_ratio_5_15"] >= 1.345) &
            (df["room_to_20d_high"] >= 7.914),
        "rule_003":
            (df["rebound_vs_prior_drop"] <= 2.778) &
            (df["max_drop_7d"] >= -3.292) &
            (df["body_ratio"] >= 0.886) &
            (df["vol_ratio_5_15"] <= 1.235),
        "rule_004":
            (df["room_to_60d_high"] <= 25.78055529) &
            (df["vol5"] <= 4.054404706) &
            (df["dist_to_ma20"] <= -7.104571765) &
            (df["vol15"] <= 3.800468235),
        "rule_005":
            (df["rebound_from_7d_low"] <= 9.038682353) &
            (df["room_to_60d_high"] <= 19.15580706) &
            (df["body_ratio"] >= 0.936) &
            (df["today_tr_val_eok"] <= 33.35342824),
        "rule_006":
            (df["max_drop_7d"] >= -3.073) &
            (df["tr_value_ratio_5d"] <= 1.2) &
            (df["vol_ratio_5_15"] <= 1.345) &
            (df["dist_to_ma20"] <= 1.702722353),
        "rule_007":
            (df["upper_wick_ratio"] <= 0.057) &
            (df["vol15"] <= 2.398) &
            (df["max_drop_7d"] <= -3.769) &
            (df["dist_to_ma20"] <= -4.631658823),
        "rule_008":
            (df["ATR_pct"] <= 3) &
            (df["today_tr_val_eok"] <= 26.10440471) &
            (df["upper_wick_ratio"] <= 0.329) &
            (df["lower_wick_ratio"] <= 0.242),
        "rule_009":
            (df["max_drop_7d"] >= -3.292) &
            (df["rebound_vs_prior_drop"] <= 3.132) &
            (df["body_ratio"] >= 0.936) &
            (df["vol_ratio_5_15"] <= 1.56),
        "rule_010":
            (df["rebound_from_7d_low"] <= 5.263) &
            (df["room_to_60d_high"] <= 20) &
            (df["upper_wick_ratio"] <= 0.114) &
            (df["today_tr_val_eok"] <= 33.35342824),
        "rule_011":
            (df["rebound_vs_prior_drop"] <= 1.703) &
            (df["vol15"] <= 2.634) &
            (df["pct_vs_lastweek"] >= 5.624) &
            (df["ma5_chg_rate"] <= 0.948),
        "rule_012":
            (df["vol15"] <= 2.634) &
            (df["upper_wick_ratio"] <= 0.02) &
            (df["dist_to_ma5"] <= 0.9471505882) &
            (df["vol5"] <= 4),
        "rule_013":
            (df["vol15"] <= 2.398) &
            (df["rebound_vs_prior_drop"] <= 1.542) &
            (df["body_ratio"] >= 0.737) &
            (df["room_to_20d_high"] >= 14.82421412),
        "rule_014":
            (df["ATR_pct"] <= 4) &
            (df["tr_value_ratio_5d"] <= 1.5) &
            (df["pct_vs_lastweek"] >= 6.427976471) &
            (df["intraday_return"] <= 4.242809412),
        "rule_015":
            (df["vol15"] <= 2.398) &
            (df["today_tr_val_eok"] <= 26.10440471) &
            (df["body_ratio"] >= 0.807) &
            (df["room_to_20d_high"] >= 16.62074588),
        "rule_016":
            (df["vol15"] <= 2.398) &
            (df["upper_wick_ratio"] <= 0.2) &
            (df["rebound_vs_prior_drop"] <= 1.542) &
            (df["room_to_20d_high"] >= 16.62074588),
        "rule_017":
            (df["vol15"] <= 2.84) &
            (df["upper_wick_ratio"] <= 0) &
            (df["max_drop_7d"] >= -3.292) &
            (df["today_tr_val_eok"] <= 20.51177647),
        "rule_018":
            (df["max_drop_7d"] >= -3.292) &
            (df["rebound_from_7d_low"] <= 5) &
            (df["upper_wick_ratio"] <= 0.254) &
            (df["room_to_20d_high"] <= 38.34321647),
        "rule_019":
            (df["ma5_chg_rate"] >= -0.1987223529) &
            (df["ATR_pct"] <= 3.21796) &
            (df["today_tr_val_eok"] <= 12.54642824) &
            (df["intraday_return"] <= 5.938658823),
        "rule_020":
            (df["today_tr_val_eok"] <= 12.54642824) &
            (df["max_drop_7d"] >= -3.073) &
            (df["room_to_20d_high"] <= 13.043) &
            (df["vol15"] <= 3.047),
        "rule_021":
            (df["rebound_from_7d_low"] <= 5) &
            (df["ATR_pct"] <= 4.671150588) &
            (df["lower_wick_ratio"] >= 0.242) &
            (df["room_to_60d_high"] <= 32.19761882),
        "rule_022":
            (df["room_to_60d_high"] <= 19.15580706) &
            (df["rebound_from_7d_low"] <= 5) &
            (df["lower_wick_ratio"] >= 0.167) &
            (df["vol5"] >= 2.544023529),
        "rule_023":
            (df["vol15"] <= 2.634) &
            (df["upper_wick_ratio"] <= 0.083) &
            (df["rebound_from_7d_low"] <= 5.263) &
            (df["body_ratio"] <= 0.771),
        "rule_024":
            (df["rebound_from_7d_low"] <= 7.547) &
            (df["room_to_60d_high"] <= 19.15580706) &
            (df["body_ratio"] >= 0.886) &
            (df["BB_perc"] >= 0.332),
        "rule_025":
            (df["room_to_60d_high"] <= 19.15580706) &
            (df["rebound_from_7d_low"] <= 9.910809412) &
            (df["body_ratio"] >= 0.936) &
            (df["intraday_return"] >= 4.608),
        "rule_026":
            (df["vol15"] <= 2.398) &
            (df["rebound_vs_prior_drop"] <= 1.108150588) &
            (df["upper_wick_ratio"] <= 0.136) &
            (df["ma5_chg_rate"] >= -1.243),
        "rule_027":
            (df["rebound_from_7d_low"] <= 5) &
            (df["ATR_pct"] <= 4.671150588) &
            (df["lower_wick_ratio"] >= 0.242) &
            (df["BB_perc"] <= 0.239),
        "rule_028":
            (df["rebound_from_7d_low"] <= 5) &
            (df["room_to_60d_high"] <= 22.52421412) &
            (df["upper_wick_ratio"] <= 0.114) &
            (df["lower_wick_ratio"] >= 0.167),
        "rule_029":
            (df["vol_ratio_5_15"] >= 1.5) &
            (df["vol5"] <= 3.574) &
            (df["body_ratio"] >= 0.771) &
            (df["room_to_60d_high"] >= 29.032),
        "rule_030":
            (df["vol15"] <= 2.00896) &
            (df["room_to_60d_high"] <= 19.15580706) &
            (df["lower_wick_ratio"] <= 0.02) &
            (df["tr_value_ratio_5d"] >= 0.867),
        "rule_031":
            (df["today_tr_val_eok"] <= 12.54642824) &
            (df["max_drop_7d"] >= -3.073) &
            (df["vol15"] <= 2.84) &
            (df["room_to_20d_high"] <= 23.077),
        "rule_032":
            (df["max_drop_7d"] >= -3.292) &
            (df["rebound_from_7d_low"] <= 5.263) &
            (df["upper_wick_ratio"] <= 0.2) &
            (df["room_to_60d_high"] <= 67.67684941),
        "rule_033":
            (df["ATR_pct"] <= 3) &
            (df["max_drop_7d"] <= -4.26) &
            (df["lower_wick_ratio"] <= 0.111) &
            (df["today_tr_val_eok"] <= 104.3396235),
        "rule_034":
            (df["upper_wick_ratio"] <= 0) &
            (df["vol15"] <= 2.84) &
            (df["room_to_60d_high"] <= 19.15580706) &
            (df["today_tr_val_eok"] <= 10),
        "rule_035":
            (df["rebound_from_7d_low"] <= 6.826) &
            (df["room_to_60d_high"] <= 22.52421412) &
            (df["body_ratio"] >= 0.886) &
            (df["intraday_return"] >= 4.242809412),
        "rule_036":
            (df["vol15"] <= 2.634) &
            (df["upper_wick_ratio"] <= 0) &
            (df["max_drop_7d"] >= -3.292) &
            (df["dist_to_ma5"] >= 2.162),
        "rule_037":
            (df["room_to_60d_high"] <= 19.15580706) &
            (df["rebound_from_7d_low"] <= 5.263) &
            (df["lower_wick_ratio"] >= 0.167) &
            (df["upper_wick_ratio"] <= 0.254),
        "rule_038":
            (df["rebound_vs_prior_drop"] <= 1.542) &
            (df["max_drop_7d"] >= -3.292) &
            (df["upper_wick_ratio"] <= 0.3) &
            (df["room_to_20d_high"] <= 38.34321647),
        "rule_039":
            (df["vol5"] <= 3.574) &
            (df["upper_wick_ratio"] <= 0.163) &
            (df["vol_ratio_5_15"] >= 1.64) &
            (df["dist_to_ma5"] <= 2.663),
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
