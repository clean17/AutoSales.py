# auto-generated: lowscan good buy rules
# train/valid split applied
# split_date: 2026-03-17
# train_min_rate: 0.9
# train_min_count: 80
# valid_min_rate: 0.8
# valid_min_count: 18
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
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
]

def build_conditions(df):
    conditions = {
            "001":
                (df["max_drop_7d"] <= -4.286) &
                (df["today_pct"] > 17.65) &
                (df["ATR_pct"] > 10.008) &
                (df["ma5_ma20_gap_chg_1d"] > 1.763),
            "002":
                (df["today_pct"] > 17.65) &
                (df["max_drop_7d"] <= -4.91) &
                (df["ATR_pct"] > 10.008) &
                (df["ma5_ma20_gap_chg_1d"] > 1.763),
            "003":
                (df["ATR_pct"] > 10.008) &
                (df["today_pct"] > 17.65) &
                (df["max_drop_7d"] <= -4.583) &
                (df["ma5_ma20_gap_chg_1d"] > 1.763),
            "004":
                (df["today_pct"] > 17.65) &
                (df["max_drop_7d"] <= -3.262) &
                (df["ATR_pct"] > 10.008) &
                (df["ma5_ma20_gap_chg_1d"] > 1.763),
            "005":
                (df["today_pct"] > 17.65) &
                (df["ATR_pct"] > 10.008) &
                (df["max_drop_7d"] <= -3.777) &
                (df["ma5_ma20_gap_chg_1d"] > 1.763),
            "006":
                (df["ATR_pct"] > 10.008) &
                (df["today_pct"] > 17.65) &
                (df["max_drop_7d"] <= -3.509) &
                (df["ma5_ma20_gap_chg_1d"] > 1.763),
            "007":
                (df["max_drop_7d"] <= -4.286) &
                (df["today_pct"] > 17.65) &
                (df["ATR_pct"] > 10.008) &
                (df["ma5_ma20_gap_chg_1d"] > 1.558),
    }
    return conditions
