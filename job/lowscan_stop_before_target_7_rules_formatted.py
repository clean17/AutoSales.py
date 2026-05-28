# auto-generated: lowscan bad avoid rules
# source: lowscan_stop_before_target_7_rules.py
# target: stop_before_target_7 == 1
# split_date: 2025-11-18
# purpose: exclude candidates likely to hit stop before target
# usage:
#    import lowscan_stop_before_target_7_rules_formatted as lowscan_rules
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
]

def build_conditions(df):
    conditions = {
        "rule_001":
            (df["vol_ratio_5_15"] <= 1.13) &
            (df["pct_vs_lastweek"] <= -3.2016) &
            (df["ma5_chg_rate"] >= -1.4086) &
            (df["market_5d_pct"] >= -1.709),
        "rule_002":
            (df["market_today_pct"] <= -0.509) &
            (df["BB_perc"] <= 0.314) &
            (df["rebound_from_7d_low"] >= 9.1948) &
            (df["today_pct"] >= 4.51),
        "rule_003":
            (df["market_today_pct"] <= -0.509) &
            (df["dist_to_ma5"] <= 4.6292) &
            (df["room_to_20d_high"] >= 30.286) &
            (df["rebound_vs_prior_drop"] >= 1.4481),
        "rule_004":
            (df["price_power_value"] >= 56.895) &
            (df["upper_wick_ratio"] >= 0.065) &
            (df["today_tr_val_eok"] <= 99.773) &
            (df["ma5_chg_rate"] >= 2.145),
        "rule_005":
            (df["market_today_pct"] <= 0.726) &
            (df["today_pct"] >= 6.77) &
            (df["price_power_value"] <= 27.4165) &
            (df["room_to_20d_high"] >= 13.1998),
        "rule_006":
            (df["pct_vs_lastweek"] >= 14.3522) &
            (df["upper_wick_ratio"] >= 0.159) &
            (df["today_tr_val_eok"] <= 36.1055) &
            (df["lower_wick_ratio"] <= 0.24),
        "rule_007":
            (df["body_value_power"] >= 36.7865) &
            (df["upper_wick_ratio"] >= 0.013) &
            (df["today_tr_val_eok"] <= 99.773) &
            (df["room_to_60d_high"] <= 24.3964),
        "rule_008":
            (df["market_today_pct"] <= -0.509) &
            (df["BB_perc"] <= 0.369) &
            (df["lower_wick_ratio"] >= 0.172) &
            (df["intraday_return"] >= 3.309),
        "rule_009":
            (df["market_today_pct"] <= -0.509) &
            (df["rebound_vs_prior_drop"] <= 3.744) &
            (df["intraday_return"] >= 7.5032) &
            (df["max_drop_7d"] >= -11.421),
        "rule_010":
            (df["market_today_pct"] <= 0.223) &
            (df["room_to_20d_high"] >= 30.286) &
            (df["vol5"] <= 5.395) &
            (df["today_pct"] >= 5.8),
        "rule_011":
            (df["vol_ratio_5_15"] <= 1.334) &
            (df["market_today_pct"] <= 0.726) &
            (df["lower_wick_ratio"] >= 0.347) &
            (df["intraday_return"] >= 3.841),
        "rule_012":
            (df["market_today_pct"] <= 0.958) &
            (df["today_pct"] >= 6.77) &
            (df["price_power_value"] <= 27.4165) &
            (df["room_to_20d_high"] >= 19.921),
        "rule_013":
            (df["vol_ratio_5_15"] <= 1.13) &
            (df["lower_wick_ratio"] >= 0.347) &
            (df["market_5d_pct"] <= -0.971) &
            (df["price_power_value"] <= 22.1118),
        "rule_014":
            (df["market_today_pct"] <= 0.223) &
            (df["body_value_power"] <= 22.8912) &
            (df["room_to_20d_high"] >= 19.921) &
            (df["intraday_return"] >= 7.5032),
        "rule_015":
            (df["market_today_pct"] <= 0.416) &
            (df["room_to_20d_high"] >= 40.7908) &
            (df["body_value_power"] <= 22.8912) &
            (df["intraday_return"] >= 5.176),
        "rule_016":
            (df["market_today_pct"] <= 0.223) &
            (df["lower_wick_ratio"] >= 0.172) &
            (df["intraday_return"] >= 3.841) &
            (df["ma5_chg_rate"] <= -0.0989),
        "rule_017":
            (df["market_today_pct"] <= -0.509) &
            (df["dist_to_ma5"] <= 7.323) &
            (df["pct_vs_lastweek"] >= 9.475) &
            (df["room_to_20d_high"] >= 7.5624),
        "rule_018":
            (df["market_today_pct"] <= 0.223) &
            (df["room_to_20d_high"] >= 30.286) &
            (df["gap_pct"] <= 0) &
            (df["today_tr_val_eok"] <= 36.1055),
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
