# auto-generated: lowscan good buy rules
# train/valid split applied
# split_date: 2026-03-17
# train_min_rate: 0.88
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
]

def build_conditions(df):
    conditions = {
        "rule_001":
            (df["dist_from_low_20d"] > 36.381) &
            (df["MACD_hist_3d"] <= 184.922) &
            (df["tr_val_rank_20d"] > 0.95) &
            (df["ma5_ma20_gap_chg_1d"] > 4.522),
        "rule_002":
            (df["dist_from_low_20d"] > 36.381) &
            (df["MACD_hist_3d"] <= 184.922) &
            (df["tr_val_rank_20d"] > 0.95) &
            (df["max_drop_7d"] > -7.298),
        "rule_003":
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > 16.003) &
            (df["today_tr_val_eok"] <= 878.596) &
            (df["ATR_pct"] > 7.43),
        "rule_004":
            (df["ma5_ma20_gap_chg_1d"] > 1.376) &
            (df["dist_to_ma5"] > 16.003) &
            (df["ATR_pct"] > 10.008) &
            (df["MACD_hist_3d"] <= 501.5),
        "rule_005":
            (df["ma5_ma20_gap_chg_1d"] > 1.052) &
            (df["dist_to_ma5"] > 16.003) &
            (df["ATR_pct"] > 10.008) &
            (df["MACD_hist_3d"] <= 501.5),
        "rule_006":
            (df["ma5_ma20_gap_chg_1d"] > 1.214) &
            (df["dist_to_ma5"] > 16.003) &
            (df["ATR_pct"] > 10.008) &
            (df["MACD_hist_3d"] <= 501.5),
        "rule_007":
            (df["dist_to_ma5"] > 16.003) &
            (df["ma5_ma20_gap_chg_1d"] > 0.906) &
            (df["ATR_pct"] > 10.008) &
            (df["MACD_hist_3d"] <= 501.5),
        "rule_008":
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["dist_to_ma5"] > 16.003) &
            (df["ATR_pct"] > 10.008) &
            (df["MACD_hist_3d"] <= 501.5),
        "rule_009":
            (df["ma5_ma20_gap_chg_1d"] > 1.558) &
            (df["dist_to_ma5"] > 16.003) &
            (df["ATR_pct"] > 10.008) &
            (df["MACD_hist_3d"] <= 501.5),
        "rule_010":
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["dist_to_ma5"] > 16.003) &
            (df["ATR_pct"] > 10.008) &
            (df["MACD_hist_3d"] <= 501.5),
        "rule_011":
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > 16.003) &
            (df["ma5_ma20_gap_chg_1d"] > 4.522) &
            (df["ATR_pct"] > 6.48),
    }
    return conditions
