# auto-generated: lowscan rules (filtered)
# usage:
#   from lowscan_rules import build_conditions, RULE_NAMES
#   find_conditions = build_conditions(df)

import numpy as np

RULE_NAMES = [
    "rule_001__n54__r0.907",
    "rule_002__n53__r0.906",
    "rule_003__n53__r0.906",
    "rule_004__n52__r0.904",
    "rule_005__n52__r0.904",
    "rule_006__n52__r0.904",
    "rule_007__n52__r0.904",
    "rule_010__n51__r0.902",
    "rule_011__n51__r0.902",
    "rule_012__n51__r0.902",
    "rule_013__n51__r0.902",
    "rule_014__n51__r0.902",
    "rule_016__n51__r0.902",
    "rule_017__n51__r0.902",
    "rule_018__n51__r0.902",
    "rule_019__n50__r0.920",
    "rule_020__n50__r0.900",
    "rule_021__n50__r0.900",
    "rule_022__n50__r0.900",
    "rule_023__n50__r0.900",
    "rule_024__n50__r0.900",
    "rule_025__n50__r0.900",
    "rule_026__n50__r0.900",
    "rule_027__n50__r0.900",
    "rule_028__n50__r0.900",
    "rule_029__n50__r0.900",
    "rule_030__n50__r0.900",
    "rule_031__n50__r0.900",
    "rule_032__n50__r0.900",
    "rule_033__n50__r0.900",
    "rule_034__n50__r0.900",
    "rule_035__n50__r0.900",
    "rule_036__n50__r0.900",
    "rule_037__n50__r0.900",
    "rule_038__n50__r0.900",
    "rule_039__n50__r0.900",
    "rule_040__n50__r0.900",
    "rule_041__n50__r0.900",
    "rule_042__n50__r0.900",
    "rule_043__n50__r0.900",
    "rule_044__n50__r0.900",
    "rule_045__n50__r0.900",
    "rule_046__n50__r0.900",
    "rule_047__n50__r0.900",
    "rule_048__n50__r0.900",
    "rule_049__n50__r0.900",
    "rule_050__n50__r0.900",
    "rule_051__n50__r0.900",
    "rule_052__n50__r0.900",
    "rule_053__n50__r0.900",
    "rule_054__n50__r0.900",
    "rule_055__n50__r0.900",
    "rule_056__n50__r0.900",
    "rule_057__n50__r0.900",
    "rule_058__n50__r0.900",
    "rule_059__n50__r0.900",
    "rule_060__n50__r0.900",
    "rule_061__n50__r0.900",
    "rule_062__n50__r0.900",
    "rule_063__n50__r0.900",
    "rule_064__n50__r0.900",
    "rule_065__n50__r0.900",
    "rule_066__n50__r0.900",
    "rule_067__n50__r0.900",
    "rule_068__n50__r0.900",
    "rule_069__n50__r0.900",
    "rule_070__n50__r0.900",
    "rule_071__n50__r0.900",
    "rule_072__n50__r0.900",
    "rule_073__n50__r0.900",
    "rule_074__n50__r0.900",
    "rule_075__n50__r0.900",
    "rule_076__n50__r0.900",
    "rule_077__n50__r0.900",
    "rule_078__n50__r0.900",
    "rule_079__n50__r0.900",
    "rule_080__n50__r0.900",
    "rule_081__n50__r0.900",
    "rule_082__n50__r0.900",
    "rule_083__n50__r0.900",
    "rule_084__n50__r0.900",
    "rule_085__n50__r0.900",
    "rule_086__n50__r0.900",
    "rule_087__n50__r0.900",
    "rule_091__n50__r0.900",
    "rule_092__n50__r0.900",
    "rule_093__n50__r0.900",
    "rule_094__n50__r0.900",
    "rule_095__n50__r0.900",
    "rule_096__n50__r0.900",
    "rule_097__n50__r0.900",
    "rule_098__n50__r0.900",
    "rule_099__n50__r0.900",
    "rule_100__n50__r0.900",
    "rule_101__n50__r0.900",
    "rule_102__n50__r0.900",
    "rule_103__n50__r0.900",
    "rule_104__n50__r0.900",
    "rule_105__n50__r0.900",
    "rule_106__n50__r0.900",
    "rule_107__n50__r0.900",
    "rule_108__n50__r0.900",
    "rule_109__n50__r0.900",
    "rule_110__n50__r0.900",
    "rule_111__n50__r0.900",
    "rule_112__n50__r0.900",
    "rule_116__n50__r0.900",
    "rule_117__n50__r0.900",
    "rule_118__n50__r0.900",
    "rule_119__n50__r0.900",
    "rule_120__n50__r0.900",
    "rule_121__n50__r0.900",
    "rule_122__n50__r0.900",
    "rule_123__n50__r0.900",
    "rule_124__n50__r0.900",
    "rule_125__n50__r0.900",
    "rule_126__n50__r0.900",
    "rule_127__n50__r0.900",
    "rule_128__n50__r0.900",
    "rule_129__n50__r0.900",
    "rule_130__n50__r0.900",
    "rule_131__n50__r0.900",
    "rule_132__n50__r0.900",
    "rule_133__n50__r0.900",
    "rule_134__n50__r0.900",
    "rule_136__n50__r0.900",
]

def build_conditions(df):
    conditions = {
            "rule_001__n54__r0.907":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64),
            "rule_002__n53__r0.906":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93),
            "rule_003__n53__r0.906":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.1) &
                (df["vol30"] >= 4.64),
            "rule_004__n52__r0.904":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 5.26),
            "rule_005__n52__r0.904":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.1) &
                (df["vol30"] >= 4.93),
            "rule_006__n52__r0.904":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.3) &
                (df["vol30"] >= 4.64),
            "rule_007__n52__r0.904":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.3) &
                (df["vol30"] >= 4.93),
            "rule_010__n51__r0.902":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["ma20_chg_rate"] <= -1.59) &
                (df["ma5_chg_rate"] <= 2.56),
            "rule_011__n51__r0.902":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["ma20_chg_rate"] <= -1.37) &
                (df["ma5_chg_rate"] <= 2.56),
            "rule_012__n51__r0.902":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.25) &
                (df["ma5_chg_rate"] <= 2.56),
            "rule_013__n51__r0.902":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.1) &
                (df["vol30"] >= 5.26),
            "rule_014__n51__r0.902":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.3) &
                (df["vol30"] >= 5.26),
            "rule_016__n51__r0.902":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.6) &
                (df["vol30"] >= 4.64),
            "rule_017__n51__r0.902":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.6) &
                (df["vol30"] >= 4.93),
            "rule_018__n51__r0.902":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 7.49) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64),
            "rule_019__n50__r0.920":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.6) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.64) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_020__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34),
            "rule_021__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma5_chg_rate"] >= -3.99),
            "rule_022__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma5_chg_rate"] <= 3.72),
            "rule_023__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma5_chg_rate"] <= 4.85),
            "rule_024__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma5_chg_rate"] <= 7.11),
            "rule_025__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] >= -4.13),
            "rule_026__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -1.59),
            "rule_027__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -1.37),
            "rule_028__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -1.18),
            "rule_029__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -1.02),
            "rule_030__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.88),
            "rule_031__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.75),
            "rule_032__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.62),
            "rule_033__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.51),
            "rule_034__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.4),
            "rule_035__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.29),
            "rule_036__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.18),
            "rule_037__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= -0.07),
            "rule_038__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= 0.06),
            "rule_039__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= 0.24),
            "rule_040__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["ma20_chg_rate"] <= 0.62),
            "rule_041__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] >= -3.187),
            "rule_042__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -1.25),
            "rule_043__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -1.08),
            "rule_044__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.91),
            "rule_045__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.79),
            "rule_046__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.66),
            "rule_047__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.54),
            "rule_048__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.44),
            "rule_049__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.35),
            "rule_050__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.24),
            "rule_051__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.15),
            "rule_052__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= -0.04),
            "rule_053__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= 0.08),
            "rule_054__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= 0.22),
            "rule_055__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= 0.42),
            "rule_056__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["mean_ret20"] <= 0.9),
            "rule_057__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pos20_ratio"] <= 55.0),
            "rule_058__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pos30_ratio"] <= 56.67),
            "rule_059__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_tr_val"] >= 448593509.92),
            "rule_060__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["chg_tr_val"] >= -70.8),
            "rule_061__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["three_m_chg_rate"] >= 25.943),
            "rule_062__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["three_m_chg_rate"] >= 33.33),
            "rule_063__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["three_m_chg_rate"] >= 39.999),
            "rule_064__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["three_m_chg_rate"] >= 46.22),
            "rule_065__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["three_m_chg_rate"] >= 51.655),
            "rule_066__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["three_m_chg_rate"] >= 57.67),
            "rule_067__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["three_m_chg_rate"] >= 64.48),
            "rule_068__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -30.037),
            "rule_069__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -27.384),
            "rule_070__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -24.631),
            "rule_071__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -21.674),
            "rule_072__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -18.61),
            "rule_073__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -15.57),
            "rule_074__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -12.25),
            "rule_075__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -8.39),
            "rule_076__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["today_chg_rate"] <= -3.703),
            "rule_077__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_lastweek"] <= 22.274),
            "rule_078__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_lastweek"] <= 33.724),
            "rule_079__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] >= -39.724),
            "rule_080__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 0.0),
            "rule_081__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 1.739),
            "rule_082__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 3.542),
            "rule_083__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 5.455),
            "rule_084__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 7.69),
            "rule_085__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 10.803),
            "rule_086__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 15.512),
            "rule_087__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.34) &
                (df["pct_vs_last3week"] <= 25.348),
            "rule_091__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma5_chg_rate"] >= -2.19) &
                (df["ma20_chg_rate"] <= -1.18),
            "rule_092__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma5_chg_rate"] >= -2.19) &
                (df["mean_ret20"] <= -1.08),
            "rule_093__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma5_chg_rate"] >= -2.19) &
                (df["mean_ret30"] <= 0.19),
            "rule_094__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma5_chg_rate"] >= -2.19) &
                (df["mean_ret30"] <= 0.34),
            "rule_095__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma5_chg_rate"] >= -2.19) &
                (df["mean_ret30"] <= 0.604),
            "rule_096__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma5_chg_rate"] >= -2.19) &
                (df["mean_ret30"] <= 1.307),
            "rule_097__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma20_chg_rate"] <= -1.18) &
                (df["mean_ret30"] <= 0.19),
            "rule_098__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma20_chg_rate"] <= -1.18) &
                (df["mean_ret30"] <= 0.34),
            "rule_099__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma20_chg_rate"] <= -1.18) &
                (df["mean_ret30"] <= 0.604),
            "rule_100__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma20_chg_rate"] <= -1.18) &
                (df["mean_ret30"] <= 1.307),
            "rule_101__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["ma20_chg_rate"] <= -1.18) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_102__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret20"] <= -1.08) &
                (df["mean_ret30"] <= 0.19),
            "rule_103__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret20"] <= -1.08) &
                (df["mean_ret30"] <= 0.34),
            "rule_104__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret20"] <= -1.08) &
                (df["mean_ret30"] <= 0.604),
            "rule_105__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret20"] <= -1.08) &
                (df["mean_ret30"] <= 1.307),
            "rule_106__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret20"] <= -1.08) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_107__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret30"] <= 0.19) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_108__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret30"] <= 0.34) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_109__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret30"] <= 0.604) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_110__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93) &
                (df["mean_ret30"] <= 1.307) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_111__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.05) &
                (df["ma5_chg_rate"] >= -2.19),
            "rule_112__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["mean_ret30"] <= -0.84) &
                (df["vol30"] >= 4.05) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_116__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["ma5_chg_rate"] >= -2.19),
            "rule_117__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["ma5_chg_rate"] <= 3.05),
            "rule_118__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["mean_ret30"] <= -0.03),
            "rule_119__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["mean_ret30"] <= 0.075),
            "rule_120__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["mean_ret30"] <= 0.19),
            "rule_121__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["mean_ret30"] <= 0.34),
            "rule_122__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["mean_ret30"] <= 0.604),
            "rule_123__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["mean_ret30"] <= 1.307),
            "rule_124__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["today_chg_rate"] >= -82.327),
            "rule_125__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["pct_vs_firstweek"] >= -72.411),
            "rule_126__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.49) &
                (df["pct_vs_last3week"] >= -28.418),
            "rule_127__n50__r0.900":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pct_vs_last2week"] <= 5.986) &
                (df["pos30_ratio"] <= 36.67) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64),
            "rule_128__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["ma20_chg_rate"] <= -1.59) &
                (df["mean_ret30"] <= -0.71),
            "rule_129__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["ma20_chg_rate"] <= -1.37) &
                (df["mean_ret30"] <= -0.71),
            "rule_130__n50__r0.900":
                (df["pct_vs_last4week"] <= -26.896) &
                (df["vol20"] <= 6.186) &
                (df["pct_vs_last2week"] >= -5.765) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.64) &
                (df["mean_ret20"] <= -1.25) &
                (df["mean_ret30"] <= -0.71),
            "rule_131__n50__r0.900":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.6) &
                (df["vol30"] >= 5.26),
            "rule_132__n50__r0.900":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 7.49) &
                (df["today_pct"] >= 4.8) &
                (df["vol30"] >= 4.93),
            "rule_133__n50__r0.900":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.9) &
                (df["vol30"] >= 4.64),
            "rule_134__n50__r0.900":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 9.24) &
                (df["today_pct"] >= 5.9) &
                (df["vol30"] >= 4.93),
            "rule_136__n50__r0.900":
                (df["mean_ret30"] <= -1.84) &
                (df["vol20"] <= 7.86) &
                (df["pct_vs_lastweek"] >= 7.233) &
                (df["pos30_ratio"] <= 36.67) &
                (df["pct_vs_last2week"] <= 7.49) &
                (df["today_pct"] >= 5.1) &
                (df["vol30"] >= 4.64),
    }
    return conditions
