# auto-generated: lowscan rules (filtered)
# usage:
#   from lowscan_rules import build_conditions, RULE_NAMES
#   find_conditions = build_conditions(df)

import numpy as np

RULE_NAMES = [
    "rule_001__n35__r0.771",
    "rule_002__n35__r0.771",
    "rule_003__n35__r0.771",
    "rule_004__n35__r0.771",
    "rule_005__n31__r0.774",
    "rule_006__n31__r0.774",
    "rule_007__n31__r0.774",
    "rule_008__n29__r0.793",
    "rule_009__n29__r0.793",
    "rule_010__n29__r0.793",
    "rule_011__n29__r0.793",
    "rule_012__n29__r0.793",
    "rule_013__n29__r0.793",
    "rule_014__n28__r0.786",
    "rule_015__n28__r0.786",
    "rule_016__n28__r0.786",
    "rule_017__n28__r0.786",
    "rule_018__n28__r0.786",
    "rule_019__n28__r0.786",
    "rule_020__n28__r0.786",
    "rule_021__n28__r0.786",
    "rule_022__n28__r0.786",
    "rule_023__n28__r0.786",
    "rule_024__n28__r0.786",
    "rule_025__n28__r0.786",
    "rule_026__n27__r0.815",
    "rule_027__n27__r0.778",
    "rule_028__n27__r0.778",
    "rule_029__n27__r0.778",
    "rule_030__n27__r0.778",
    "rule_031__n27__r0.778",
    "rule_032__n27__r0.778",
    "rule_033__n27__r0.778",
    "rule_034__n27__r0.778",
    "rule_035__n27__r0.778",
    "rule_036__n27__r0.778",
    "rule_037__n27__r0.778",
    "rule_038__n27__r0.778",
    "rule_039__n27__r0.778",
    "rule_040__n27__r0.778",
    "rule_041__n27__r0.778",
    "rule_043__n27__r0.778",
    "rule_044__n27__r0.778",
    "rule_045__n27__r0.778",
    "rule_046__n27__r0.778",
    "rule_047__n27__r0.778",
    "rule_048__n27__r0.778",
    "rule_049__n27__r0.778",
    "rule_050__n27__r0.778",
    "rule_051__n27__r0.778",
    "rule_052__n26__r0.808",
    "rule_053__n26__r0.808",
    "rule_054__n26__r0.808",
    "rule_055__n26__r0.808",
    "rule_056__n26__r0.808",
    "rule_057__n26__r0.808",
    "rule_058__n26__r0.808",
    "rule_059__n25__r0.840",
    "rule_060__n25__r0.840",
    "rule_061__n25__r0.800",
    "rule_062__n25__r0.800",
    "rule_063__n25__r0.800",
    "rule_064__n25__r0.800",
    "rule_065__n25__r0.800",
    "rule_066__n25__r0.800",
    "rule_067__n25__r0.800",
    "rule_068__n25__r0.800",
    "rule_069__n25__r0.800",
    "rule_070__n25__r0.800",
]

def build_conditions(df):
    conditions = {
            "rule_001__n35__r0.771":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] <= 84.093) &
                (df["today_chg_rate"] >= -9.949) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 1157498578.667),
            "rule_002__n35__r0.771":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["today_chg_rate"] >= -5.727) &
                (df["three_m_chg_rate"] <= 84.093) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 1157498578.667),
            "rule_003__n35__r0.771":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -8.035) &
                (df["three_m_chg_rate"] <= 84.093) &
                (df["mean_prev3"] >= 1157498578.667),
            "rule_004__n35__r0.771":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -11.506) &
                (df["three_m_chg_rate"] <= 84.093) &
                (df["mean_prev3"] >= 1157498578.667),
            "rule_005__n31__r0.774":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["today_chg_rate"] >= -5.727) &
                (df["mean_prev3"] >= 1642430983.333) &
                (df["vol30"] >= 5.389) &
                (df["three_m_chg_rate"] <= 67.379),
            "rule_006__n31__r0.774":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["mean_prev3"] >= 1642430983.333) &
                (df["vol30"] >= 5.389) &
                (df["today_chg_rate"] >= -8.035) &
                (df["three_m_chg_rate"] <= 67.379),
            "rule_007__n31__r0.774":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 1642430983.333) &
                (df["three_m_chg_rate"] <= 84.093) &
                (df["today_chg_rate"] >= -12.93),
            "rule_008__n29__r0.793":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -10.89),
            "rule_009__n29__r0.793":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -9.376),
            "rule_010__n29__r0.793":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -8.082),
            "rule_011__n29__r0.793":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -9.376),
            "rule_012__n29__r0.793":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -8.082),
            "rule_013__n29__r0.793":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -6.828),
            "rule_014__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -6.828),
            "rule_015__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -5.64),
            "rule_016__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -4.596),
            "rule_017__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["today_chg_rate"] >= -38.443),
            "rule_018__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -10.89),
            "rule_019__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -9.376),
            "rule_020__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -8.082),
            "rule_021__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -6.828),
            "rule_022__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -5.64),
            "rule_023__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -4.596),
            "rule_024__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -5.64),
            "rule_025__n28__r0.786":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -4.596),
            "rule_026__n27__r0.815":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -38.443),
            "rule_027__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 4.857) &
                (df["today_pct"] >= 5.405) &
                (df["three_m_chg_rate"] <= 47.28) &
                (df["mean_prev3"] >= 359096543.0),
            "rule_028__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["today_pct"] >= 4.753) &
                (df["vol30"] >= 4.857) &
                (df["three_m_chg_rate"] <= 47.28) &
                (df["mean_prev3"] >= 359096543.0),
            "rule_029__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 4.857) &
                (df["today_pct"] >= 5.062) &
                (df["three_m_chg_rate"] <= 47.28) &
                (df["mean_prev3"] >= 359096543.0),
            "rule_030__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 4.857) &
                (df["today_pct"] >= 5.834) &
                (df["three_m_chg_rate"] <= 47.28) &
                (df["mean_prev3"] >= 359096543.0),
            "rule_031__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 4.857) &
                (df["today_pct"] >= 6.286) &
                (df["three_m_chg_rate"] <= 47.28) &
                (df["mean_prev3"] >= 359096543.0),
            "rule_032__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -3.597),
            "rule_033__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["today_chg_rate"] >= -30.368) &
                (df["vol30"] >= 4.857),
            "rule_034__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["today_chg_rate"] >= -30.368) &
                (df["vol30"] >= 5.389),
            "rule_035__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -2.479),
            "rule_036__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -1.339),
            "rule_037__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -0.307),
            "rule_038__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -3.597),
            "rule_039__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -2.479),
            "rule_040__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -1.339),
            "rule_041__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -0.307),
            "rule_043__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["mean_prev3"] >= 1642430983.333) &
                (df["vol30"] >= 6.079) &
                (df["three_m_chg_rate"] <= 74.789) &
                (df["today_chg_rate"] >= -12.93),
            "rule_044__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["mean_prev3"] >= 1371635202.4) &
                (df["vol30"] >= 6.079) &
                (df["three_m_chg_rate"] <= 67.379) &
                (df["today_chg_rate"] >= -12.93),
            "rule_045__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 1157498578.667) &
                (df["today_chg_rate"] >= -9.949) &
                (df["three_m_chg_rate"] <= 67.379),
            "rule_046__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 1157498578.667) &
                (df["today_chg_rate"] >= -5.727) &
                (df["three_m_chg_rate"] <= 67.379),
            "rule_047__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["mean_prev3"] >= 1157498578.667) &
                (df["today_chg_rate"] >= -8.035) &
                (df["vol30"] >= 6.079) &
                (df["three_m_chg_rate"] <= 67.379),
            "rule_048__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["today_chg_rate"] >= -11.506) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 1157498578.667) &
                (df["three_m_chg_rate"] <= 67.379),
            "rule_049__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -3.597),
            "rule_050__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -2.479),
            "rule_051__n27__r0.778":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["vol15"] >= 8.145) &
                (df["today_tr_val"] <= 27312427005.0) &
                (df["pct_vs_last4week"] >= -1.339),
            "rule_052__n26__r0.808":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -2.479),
            "rule_053__n26__r0.808":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -1.339),
            "rule_054__n26__r0.808":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -0.307),
            "rule_055__n26__r0.808":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= 0.881),
            "rule_056__n26__r0.808":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= 2.332),
            "rule_057__n26__r0.808":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= 0.881),
            "rule_058__n26__r0.808":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= 2.332),
            "rule_059__n25__r0.840":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= 0.881),
            "rule_060__n25__r0.840":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= 2.332),
            "rule_061__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= 4.118),
            "rule_062__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -10.89),
            "rule_063__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -9.376),
            "rule_064__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -8.082),
            "rule_065__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -6.828),
            "rule_066__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -5.64),
            "rule_067__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 67.379) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= -4.596),
            "rule_068__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["pct_vs_last4week"] >= 4.118),
            "rule_069__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["three_m_chg_rate"] >= 74.789) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["vol15"] >= 6.091) &
                (df["today_chg_rate"] >= -38.443),
            "rule_070__n25__r0.800":
                (df["ma5_chg_rate"] >= 4.42) &
                (df["mean_prev3"] >= 1642430983.333) &
                (df["vol30"] >= 6.079) &
                (df["three_m_chg_rate"] <= 67.379) &
                (df["today_chg_rate"] >= -12.93),
    }
    return conditions
