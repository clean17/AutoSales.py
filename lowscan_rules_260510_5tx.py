# auto-generated: lowscan good buy rules
# train/valid split applied
# split_date: 2026-03-17
# train_min_rate: 0.9
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
]

def build_conditions(df):
    conditions = {
        "rule_001":
            (df["tr_val_rank_20d"] > 0.95) &
            (df["dist_from_low_20d"] > 36.381) &
            (df["max_drop_7d"] > -7.298) &
            (df["MACD_hist_3d"] <= 184.922) &
            (df["ATR_pct"] > 4.231),
        "rule_002":
            (df["tr_val_rank_20d"] > 0.95) &
            (df["dist_from_low_20d"] > 36.381) &
            (df["max_drop_7d"] > -7.298) &
            (df["MACD_hist_3d"] <= 184.922) &
            (df["ATR_pct"] > 4.542),
        "rule_003":
            (df["tr_val_rank_20d"] > 0.95) &
            (df["dist_from_low_20d"] > 36.381) &
            (df["max_drop_7d"] > -7.298) &
            (df["MACD_hist_3d"] <= 184.922) &
            (df["ma5_ma20_gap_chg_1d"] > 3.522),
        "rule_004":
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > 16.003) &
            (df["MACD_hist_3d"] <= 932.973) &
            (df["today_tr_val_eok"] <= 878.596) &
            (df["ATR_pct"] > 7.43),
        "rule_005":
            (df["tr_val_rank_20d"] > 0.95) &
            (df["dist_from_low_20d"] > 36.381) &
            (df["max_drop_7d"] > -7.298) &
            (df["MACD_hist_3d"] <= 184.922) &
            (df["ATR_pct"] > 4.782),
    }
    return conditions
