# auto-generated: lowscan rules (filtered)
# usage:
#   from lowscan_rules import build_conditions, RULE_NAMES
#   find_conditions = build_conditions(df)

import numpy as np

RULE_NAMES = [
    "rule_019__n22__r0.773__s1.87",
    "rule_029__n21__r0.762__s1.79",
    "rule_032__n21__r0.762__s1.79",
    "rule_044__n20__r0.750__s1.71",
    "rule_045__n20__r0.750__s1.71",
    "rule_046__n20__r0.750__s1.71",
    "rule_048__n20__r0.750__s1.71",
]

def build_conditions(df):
    conditions = {
            "rule_019__n22__r0.773__s1.87":
                (df["three_m_max_cur"] >= -7.582) &
                (df["vol_ratio"] <= 0.81) &
                (df["volume_rank_20d"] >= 0.85) &
                (df["ma5_chg_rate"] <= 4.242),
            "rule_029__n21__r0.762__s1.79":
                (df["three_m_max_cur"] >= -7.582) &
                (df["vol_ratio"] <= 0.81) &
                (df["volume_rank_20d"] >= 0.85) &
                (df["ma5_chg_rate"] >= 1.06),
            "rule_032__n21__r0.762__s1.79":
                (df["ma5_chg_rate"] <= -0.132) &
                (df["close_pos"] <= 0.83) &
                (df["three_m_max_cur"] >= -18.421) &
                (df["volume_rank_20d"] <= 0.75),
            "rule_044__n20__r0.750__s1.71":
                (df["ma5_chg_rate"] <= -0.132) &
                (df["close_pos"] >= 0.941) &
                (df["vol_ratio"] <= 1.15) &
                (df["three_m_max_cur"] <= -25.562),
            "rule_045__n20__r0.750__s1.71":
                (df["ma5_chg_rate"] <= -0.132) &
                (df["three_m_max_cur"] <= -25.562) &
                (df["close_pos"] >= 0.941) &
                (df["vol_ratio"] <= 1.079),
            "rule_046__n20__r0.750__s1.71":
                (df["vol_ratio"] <= 0.725) &
                (df["close_pos"] <= 0.83) &
                (df["ma5_chg_rate"] >= 1.794) &
                (df["volume_rank_20d"] <= 0.85),
            "rule_048__n20__r0.750__s1.71":
                (df["ma5_chg_rate"] >= 1.394) &
                (df["vol_ratio"] <= 0.725) &
                (df["volume_rank_20d"] <= 0.8) &
                (df["three_m_max_cur"] <= -18.421),
    }
    return conditions
