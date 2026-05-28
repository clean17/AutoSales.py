# auto-generated: lowscan bad avoid rules
# source: lowscan_target0_highprob_rules_cov4_c3_06.py
# scenario: cov4_c3_06_balanced
# target: target_class == 0 no-bounce / low-quality rebound risk
# split_date: 2025-11-18
# purpose: exclude candidates likely to fail rebound
# usage:
#    import lowscan_target0_highprob_rules_cov4_c3_06_formatted as lowscan_rules
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
            (df["today_pct"] <= 4.01) &
            (df["vol5"] <= 3.784) &
            (df["vol_ratio_5_15"] >= 1.573) &
            (df["body_value_power"] >= 6.1621657143) &
            (df["upper_wick_ratio"] >= 0.021),
        "rule_002":
            (df["body_value_power"] >= 6.1621657143) &
            (df["vol5"] <= 2.242) &
            (df["BB_perc"] <= 0.335) &
            (df["lower_wick_ratio"] <= 0.109) &
            (df["dist_to_ma5"] >= 1.651),
        "rule_003":
            (df["max_drop_7d"] >= -3.1273857143) &
            (df["rebound_from_7d_low"] <= 3.795) &
            (df["gap_pct"] <= 0.5) &
            (df["BB_perc"] <= 0.361) &
            (df["ma5_chg_rate"] <= 0.061),
        "rule_004":
            (df["rebound_from_7d_low"] <= 8.661) &
            (df["room_to_60d_high"] <= 51.802) &
            (df["vol_ratio_5_15"] <= 0.545) &
            (df["upper_wick_ratio"] <= 0.152) &
            (df["intraday_return"] <= 2.748),
        "rule_005":
            (df["vol5"] <= 2.242) &
            (df["body_ratio"] >= 0.494) &
            (df["intraday_return"] <= 1.852) &
            (df["room_to_60d_high"] >= 21.4940828571) &
            (df["today_pct"] <= 4.2),
        "rule_006":
            (df["rebound_vs_prior_drop"] <= 1.434) &
            (df["max_drop_7d"] >= -3.328) &
            (df["today_tr_val_eok"] <= 8.4199285714) &
            (df["room_to_60d_high"] <= 26.6129) &
            (df["body_value_power"] <= 6.8404314286),
        "rule_007":
            (df["upper_wick_ratio"] <= 0.067) &
            (df["vol5"] <= 2.88) &
            (df["intraday_return"] <= 1.5) &
            (df["lower_wick_ratio"] <= 0.579) &
            (df["rebound_vs_prior_drop"] <= 3.0115685714),
        "rule_008":
            (df["room_to_60d_high"] <= 47.471) &
            (df["rebound_from_7d_low"] <= 8.661) &
            (df["vol_ratio_5_15"] <= 0.545) &
            (df["gap_pct"] >= 1.0) &
            (df["price_power_value"] <= 17.5708257143),
        "rule_009":
            (df["vol_ratio_5_15"] <= 0.545) &
            (df["upper_wick_ratio"] <= 0.114) &
            (df["rebound_from_7d_low"] <= 8.661) &
            (df["room_to_60d_high"] <= 70.7467571429) &
            (df["max_drop_7d"] >= -4.438),
        "rule_010":
            (df["rebound_from_7d_low"] <= 7.166) &
            (df["vol5"] <= 2.88) &
            (df["lower_wick_ratio"] >= 0.4113114286) &
            (df["room_to_60d_high"] <= 34.3906514286) &
            (df["body_ratio"] <= 0.409),
        "rule_011":
            (df["lower_wick_ratio"] >= 0.4113114286) &
            (df["vol5"] <= 3.0) &
            (df["room_to_60d_high"] <= 18.8669371429) &
            (df["today_pct"] <= 4.2) &
            (df["rebound_vs_prior_drop"] <= 3.0115685714),
        "rule_012":
            (df["vol5"] <= 3.59) &
            (df["today_pct"] <= 4.01) &
            (df["vol_ratio_5_15"] >= 1.573) &
            (df["body_value_power"] >= 6.1621657143) &
            (df["max_drop_7d"] <= -3.52),
        "rule_013":
            (df["vol5"] <= 3.59) &
            (df["rebound_vs_prior_drop"] <= 2.0) &
            (df["vol_ratio_5_15"] >= 1.686) &
            (df["upper_wick_ratio"] <= 0.2) &
            (df["ma5_chg_rate"] >= -0.884),
        "rule_014":
            (df["vol5"] <= 1.793) &
            (df["lower_wick_ratio"] >= 0.278) &
            (df["gap_pct"] >= 1.376) &
            (df["max_drop_7d"] <= -3.1273857143) &
            (df["BB_perc"] <= 0.886),
        "rule_015":
            (df["lower_wick_ratio"] >= 0.4113114286) &
            (df["vol5"] <= 3.0) &
            (df["room_to_60d_high"] <= 18.8669371429) &
            (df["today_pct"] <= 4.2) &
            (df["dist_to_ma5"] <= 4.3275228571),
        "rule_016":
            (df["rebound_from_7d_low"] <= 8.661) &
            (df["room_to_60d_high"] <= 47.471) &
            (df["vol_ratio_5_15"] <= 0.545) &
            (df["upper_wick_ratio"] <= 0.152) &
            (df["gap_pct"] >= 0.105),
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
