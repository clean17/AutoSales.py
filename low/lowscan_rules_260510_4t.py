# auto-generated: lowscan good buy rules
# train/valid split applied
# split_date: 2026-03-17
# train_min_rate: 0.91
# train_min_count: 80
# valid_min_rate: 0.8
# valid_min_count: 15
# usage:
#    import lowscan_rules
#    buy_conditions = lowscan_rules.build_conditions(df)
#
#    buy_mask = np.zeros(len(df), dtype=bool)
#    for cond in buy_conditions.values():
#        buy_mask |= cond
#
#    df = df[buy_mask].copy()

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
]

def build_conditions(df):
    conditions = {
        "rule_001":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763),
        "rule_002":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972),
        "rule_003":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 2.227),
        "rule_004":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 1.558),
        "rule_005":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 0.906),
        "rule_006":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 1.052),
        "rule_007":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 1.214),
        "rule_008":
            (df["ATR_pct"] > 10.008) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -4.286) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763),
        "rule_009":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.092) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763),
        "rule_010":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.092) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972),
        "rule_011":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["ma5_ma20_gap_chg_1d"] > 1.376),
        "rule_012":
            (df["today_pct"] > 17.65) &
            (df["ATR_pct"] > 9.084) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["max_drop_7d"] <= -7.298),
        "rule_013":
            (df["today_pct"] > 17.65) &
            (df["ATR_pct"] > 9.084) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["max_drop_7d"] <= -7.298),
        "rule_014":
            (df["ATR_pct"] > 10.008) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -4.286) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972),
        "rule_015":
            (df["ma5_ma20_gap_chg_1d"] > 2.227) &
            (df["today_pct"] > 17.65) &
            (df["ATR_pct"] > 9.084) &
            (df["max_drop_7d"] <= -7.298),
        "rule_016":
            (df["ATR_pct"] > 10.008) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -4.286) &
            (df["ma5_ma20_gap_chg_1d"] > 2.227),
        "rule_017":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -4.583) &
            (df["ATR_pct"] > 10.008) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763),
        "rule_018":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -4.583) &
            (df["ATR_pct"] > 10.008) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972),
        "rule_019":
            (df["ATR_pct"] > 10.008) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -4.91) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763),
        "rule_020":
            (df["ATR_pct"] > 10.008) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -4.91) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972),
        "rule_021":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["ATR_pct"] > 7.85) &
            (df["ma5_ma20_gap_chg_1d"] > 1.558),
    }
    return conditions
