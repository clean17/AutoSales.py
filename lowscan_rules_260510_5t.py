# auto-generated: lowscan good buy rules
# train/valid split applied
# split_date: 2026-03-17
# train_min_rate: 0.928
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
    "rule_040",
    "rule_041",
    "rule_042",
    "rule_043",
    "rule_044",
    "rule_045",
    "rule_046",
    "rule_047",
    "rule_048",
    "rule_049",
    "rule_050",
    "rule_051",
    "rule_052",
    "rule_053",
    "rule_054",
    "rule_055",
    "rule_056",
    "rule_057",
    "rule_058",
    "rule_059",
    "rule_060",
    "rule_061",
    "rule_062",
    "rule_063",
    "rule_064",
    "rule_065",
    "rule_066",
    "rule_067",
    "rule_068",
    "rule_069",
    "rule_070",
    "rule_071",
    "rule_072",
    "rule_073",
    "rule_074",
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
            (df["MACD_hist_3d"] > 2.956) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763),
        "rule_004":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ATR_pct"] > 7.43) &
            (df["MACD_hist_3d"] > 2.956) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972),
        "rule_005":
            (df["tr_val_rank_20d"] > 0.25) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_006":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 1.583) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_007":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 2.123) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_008":
            (df["today_pct"] > 17.65) &
            (df["tr_val_rank_20d"] > 0.2) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_009":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 11.727) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_010":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 2.558) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_011":
            (df["tr_val_rank_20d"] > 0.05) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_012":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 2.933) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_013":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 10.889) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_014":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 3.301) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_015":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 3.74) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_016":
            (df["tr_val_rank_20d"] > 0.15) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_017":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 12.646) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_018":
            (df["tr_val_rank_20d"] > 0.1) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_019":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 13.58) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_020":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["today_tr_val_eok"] > 3.441) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_021":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 14.539) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_022":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 15.789) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_023":
            (df["dist_from_low_20d"] > 10.093) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_024":
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["tr_val_rank_20d"] <= 1.0) &
            (df["ATR_pct"] > 7.43),
        "rule_025":
            (df["today_pct"] > 17.65) &
            (df["tr_val_rank_20d"] > 0.3) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_026":
            (df["dist_from_low_20d"] > 9.4) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_027":
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_from_low_20d"] > 7.21) &
            (df["ATR_pct"] > 7.43),
        "rule_028":
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_from_low_20d"] > 8.675) &
            (df["ATR_pct"] > 7.43),
        "rule_029":
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > 0.983) &
            (df["ATR_pct"] > 7.43),
        "rule_030":
            (df["MACD_hist_3d"] > -5.485) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_031":
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["max_drop_7d"] <= -9.009) &
            (df["MACD_hist_3d"] > -44.029) &
            (df["ATR_pct"] > 7.43),
        "rule_032":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["dist_from_low_20d"] > 6.454) &
            (df["ATR_pct"] > 7.43),
        "rule_033":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["dist_to_ma5"] > 0.099) &
            (df["ATR_pct"] > 7.43),
        "rule_034":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 5.527) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_035":
            (df["today_pct"] > 17.65) &
            (df["tr_val_rank_20d"] > 0.35) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_036":
            (df["today_tr_val_eok"] > 5.185) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_037":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 17.238) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["ATR_pct"] > 7.43),
        "rule_038":
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_from_low_20d"] > 7.922) &
            (df["ATR_pct"] > 7.43),
        "rule_039":
            (df["ma5_ma20_gap_chg_1d"] > 1.763) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > -1.658) &
            (df["ATR_pct"] > 7.43),
        "rule_040":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 10.889) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_041":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 1.583) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_042":
            (df["today_pct"] > 17.65) &
            (df["tr_val_rank_20d"] > 0.2) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_043":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 2.123) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_044":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 11.727) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_045":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 2.558) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_046":
            (df["tr_val_rank_20d"] > 0.05) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_047":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 2.933) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_048":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 3.301) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_049":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["dist_to_ma5"] > 3.74) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_050":
            (df["tr_val_rank_20d"] > 0.1) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_051":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 13.58) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_052":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["today_tr_val_eok"] > 3.441) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_053":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 14.539) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_054":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 15.789) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_055":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["tr_val_rank_20d"] > 0.15) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_056":
            (df["tr_val_rank_20d"] > 0.25) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_057":
            (df["tr_val_rank_20d"] > 0.3) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_058":
            (df["dist_from_low_20d"] > 7.922) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_059":
            (df["today_pct"] > 17.65) &
            (df["tr_val_rank_20d"] > 0.35) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_060":
            (df["dist_from_low_20d"] > 5.527) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_061":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 12.646) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_062":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["today_tr_val_eok"] > 5.185) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_063":
            (df["today_pct"] > 17.65) &
            (df["dist_from_low_20d"] > 17.238) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_064":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_from_low_20d"] > 6.454) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_065":
            (df["dist_from_low_20d"] > 10.093) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_066":
            (df["dist_from_low_20d"] > 9.4) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_067":
            (df["dist_from_low_20d"] > 8.675) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_068":
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["MACD_hist_3d"] > -5.485) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_069":
            (df["today_pct"] > 17.65) &
            (df["tr_val_rank_20d"] <= 1.0) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_070":
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > 0.099) &
            (df["ATR_pct"] > 7.43),
        "rule_071":
            (df["max_drop_7d"] <= -9.009) &
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["MACD_hist_3d"] > -44.029) &
            (df["ATR_pct"] > 7.43),
        "rule_072":
            (df["dist_from_low_20d"] > 7.21) &
            (df["today_pct"] > 17.65) &
            (df["max_drop_7d"] <= -9.009) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["ATR_pct"] > 7.43),
        "rule_073":
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > -1.658) &
            (df["ATR_pct"] > 7.43),
        "rule_074":
            (df["today_pct"] > 17.65) &
            (df["ma5_ma20_gap_chg_1d"] > 1.972) &
            (df["max_drop_7d"] <= -9.009) &
            (df["dist_to_ma5"] > 0.983) &
            (df["ATR_pct"] > 7.43),
    }
    return conditions
