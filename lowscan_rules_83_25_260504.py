# auto-generated: lowscan rules (filtered)
# usage:
#   from lowscan_rules import build_conditions, RULE_NAMES
#   find_conditions = build_conditions(df)

import numpy as np

RULE_NAMES = [
    "rule_001__n27__r0.963__s2.98",
    "rule_002__n25__r0.960__s2.88",
    "rule_003__n25__r0.960__s2.88",
    "rule_004__n32__r0.906__s2.60",
    "rule_005__n26__r0.923__s2.59",
    "rule_006__n31__r0.903__s2.55",
    "rule_007__n25__r0.920__s2.54",
    "rule_008__n29__r0.897__s2.45",
    "rule_009__n29__r0.897__s2.45",
    "rule_010__n29__r0.897__s2.45",
    "rule_011__n28__r0.893__s2.40",
    "rule_012__n28__r0.893__s2.40",
    "rule_013__n27__r0.889__s2.34",
    "rule_014__n27__r0.889__s2.34",
    "rule_015__n27__r0.889__s2.34",
    "rule_016__n27__r0.889__s2.34",
    "rule_017__n27__r0.889__s2.34",
    "rule_018__n27__r0.889__s2.34",
    "rule_019__n27__r0.889__s2.34",
    "rule_020__n27__r0.889__s2.34",
    "rule_021__n27__r0.889__s2.34",
    "rule_022__n36__r0.861__s2.31",
    "rule_023__n26__r0.885__s2.28",
    "rule_024__n26__r0.885__s2.28",
    "rule_025__n26__r0.885__s2.28",
    "rule_026__n26__r0.885__s2.28",
    "rule_027__n26__r0.885__s2.28",
    "rule_028__n26__r0.885__s2.28",
    "rule_029__n26__r0.885__s2.28",
    "rule_030__n26__r0.885__s2.28",
    "rule_031__n26__r0.885__s2.28",
    "rule_032__n26__r0.885__s2.28",
    "rule_033__n25__r0.880__s2.22",
    "rule_034__n25__r0.880__s2.22",
    "rule_035__n25__r0.880__s2.22",
    "rule_036__n25__r0.880__s2.22",
    "rule_037__n25__r0.880__s2.22",
    "rule_038__n25__r0.880__s2.22",
    "rule_039__n25__r0.880__s2.22",
    "rule_040__n25__r0.880__s2.22",
    "rule_041__n25__r0.880__s2.22",
    "rule_042__n25__r0.880__s2.22",
    "rule_043__n25__r0.880__s2.22",
    "rule_044__n25__r0.880__s2.22",
    "rule_045__n25__r0.880__s2.22",
    "rule_046__n34__r0.853__s2.21",
    "rule_047__n29__r0.862__s2.18",
    "rule_048__n29__r0.862__s2.18",
    "rule_049__n33__r0.848__s2.15",
    "rule_050__n33__r0.848__s2.15",
    "rule_051__n33__r0.848__s2.15",
    "rule_052__n37__r0.838__s2.14",
    "rule_053__n28__r0.857__s2.12",
    "rule_054__n28__r0.857__s2.12",
    "rule_055__n28__r0.857__s2.12",
    "rule_056__n28__r0.857__s2.12",
    "rule_057__n28__r0.857__s2.12",
    "rule_058__n28__r0.857__s2.12",
    "rule_059__n28__r0.857__s2.12",
    "rule_060__n28__r0.857__s2.12",
    "rule_061__n28__r0.857__s2.12",
    "rule_062__n28__r0.857__s2.12",
    "rule_063__n28__r0.857__s2.12",
    "rule_064__n28__r0.857__s2.12",
    "rule_065__n28__r0.857__s2.12",
    "rule_066__n28__r0.857__s2.12",
    "rule_067__n28__r0.857__s2.12",
    "rule_068__n28__r0.857__s2.12",
    "rule_069__n28__r0.857__s2.12",
    "rule_070__n28__r0.857__s2.12",
    "rule_071__n28__r0.857__s2.12",
    "rule_072__n28__r0.857__s2.12",
    "rule_073__n28__r0.857__s2.12",
    "rule_074__n28__r0.857__s2.12",
    "rule_075__n32__r0.844__s2.10",
    "rule_076__n32__r0.844__s2.10",
    "rule_077__n32__r0.844__s2.10",
    "rule_078__n32__r0.844__s2.10",
    "rule_079__n32__r0.844__s2.10",
    "rule_080__n32__r0.844__s2.10",
    "rule_081__n32__r0.844__s2.10",
    "rule_082__n36__r0.833__s2.09",
    "rule_083__n27__r0.852__s2.06",
    "rule_084__n27__r0.852__s2.06",
    "rule_085__n27__r0.852__s2.06",
    "rule_086__n27__r0.852__s2.06",
    "rule_087__n27__r0.852__s2.06",
    "rule_088__n27__r0.852__s2.06",
    "rule_089__n27__r0.852__s2.06",
    "rule_090__n27__r0.852__s2.06",
    "rule_091__n27__r0.852__s2.06",
    "rule_092__n27__r0.852__s2.06",
    "rule_093__n27__r0.852__s2.06",
    "rule_094__n27__r0.852__s2.06",
    "rule_095__n27__r0.852__s2.06",
    "rule_096__n27__r0.852__s2.06",
    "rule_097__n27__r0.852__s2.06",
    "rule_098__n27__r0.852__s2.06",
    "rule_099__n27__r0.852__s2.06",
    "rule_100__n27__r0.852__s2.06",
    "rule_101__n27__r0.852__s2.06",
    "rule_102__n27__r0.852__s2.06",
    "rule_103__n27__r0.852__s2.06",
    "rule_104__n27__r0.852__s2.06",
    "rule_105__n27__r0.852__s2.06",
    "rule_106__n27__r0.852__s2.06",
    "rule_107__n27__r0.852__s2.06",
    "rule_108__n27__r0.852__s2.06",
    "rule_109__n31__r0.839__s2.04",
    "rule_110__n31__r0.839__s2.04",
    "rule_111__n31__r0.839__s2.04",
    "rule_112__n26__r0.846__s2.00",
    "rule_113__n26__r0.846__s2.00",
    "rule_114__n26__r0.846__s2.00",
    "rule_115__n26__r0.846__s2.00",
    "rule_116__n26__r0.846__s2.00",
    "rule_117__n26__r0.846__s2.00",
    "rule_118__n26__r0.846__s2.00",
    "rule_119__n26__r0.846__s2.00",
    "rule_120__n26__r0.846__s2.00",
    "rule_121__n26__r0.846__s2.00",
    "rule_122__n26__r0.846__s2.00",
    "rule_123__n26__r0.846__s2.00",
    "rule_124__n26__r0.846__s2.00",
    "rule_125__n26__r0.846__s2.00",
    "rule_126__n26__r0.846__s2.00",
    "rule_127__n26__r0.846__s2.00",
    "rule_128__n26__r0.846__s2.00",
    "rule_129__n26__r0.846__s2.00",
    "rule_130__n26__r0.846__s2.00",
    "rule_131__n26__r0.846__s2.00",
    "rule_132__n26__r0.846__s2.00",
    "rule_133__n26__r0.846__s2.00",
    "rule_134__n26__r0.846__s2.00",
    "rule_135__n26__r0.846__s2.00",
    "rule_136__n26__r0.846__s2.00",
    "rule_137__n26__r0.846__s2.00",
    "rule_138__n26__r0.846__s2.00",
    "rule_139__n26__r0.846__s2.00",
    "rule_140__n26__r0.846__s2.00",
    "rule_141__n26__r0.846__s2.00",
    "rule_142__n26__r0.846__s2.00",
    "rule_143__n26__r0.846__s2.00",
    "rule_144__n26__r0.846__s2.00",
    "rule_145__n26__r0.846__s2.00",
    "rule_146__n30__r0.833__s1.99",
    "rule_147__n30__r0.833__s1.99",
    "rule_148__n30__r0.833__s1.99",
    "rule_149__n30__r0.833__s1.99",
    "rule_150__n30__r0.833__s1.99",
    "rule_151__n30__r0.833__s1.99",
    "rule_152__n30__r0.833__s1.99",
    "rule_153__n25__r0.840__s1.93",
    "rule_154__n25__r0.840__s1.93",
    "rule_155__n25__r0.840__s1.93",
    "rule_156__n25__r0.840__s1.93",
    "rule_157__n25__r0.840__s1.93",
    "rule_158__n25__r0.840__s1.93",
    "rule_159__n25__r0.840__s1.93",
    "rule_160__n25__r0.840__s1.93",
    "rule_161__n25__r0.840__s1.93",
    "rule_162__n25__r0.840__s1.93",
    "rule_163__n25__r0.840__s1.93",
    "rule_164__n25__r0.840__s1.93",
    "rule_165__n25__r0.840__s1.93",
    "rule_166__n25__r0.840__s1.93",
    "rule_167__n25__r0.840__s1.93",
    "rule_168__n25__r0.840__s1.93",
    "rule_169__n25__r0.840__s1.93",
    "rule_170__n25__r0.840__s1.93",
    "rule_171__n25__r0.840__s1.93",
    "rule_172__n25__r0.840__s1.93",
    "rule_173__n25__r0.840__s1.93",
    "rule_174__n25__r0.840__s1.93",
    "rule_175__n25__r0.840__s1.93",
    "rule_176__n25__r0.840__s1.93",
    "rule_177__n25__r0.840__s1.93",
    "rule_178__n25__r0.840__s1.93",
    "rule_179__n25__r0.840__s1.93",
    "rule_180__n25__r0.840__s1.93",
    "rule_181__n25__r0.840__s1.93",
    "rule_182__n25__r0.840__s1.93",
    "rule_183__n25__r0.840__s1.93",
    "rule_184__n25__r0.840__s1.93",
    "rule_185__n25__r0.840__s1.93",
    "rule_186__n25__r0.840__s1.93",
    "rule_187__n25__r0.840__s1.93",
    "rule_188__n25__r0.840__s1.93",
    "rule_189__n25__r0.840__s1.93",
    "rule_190__n25__r0.840__s1.93",
    "rule_191__n25__r0.840__s1.93",
    "rule_192__n25__r0.840__s1.93",
    "rule_193__n25__r0.840__s1.93",
    "rule_194__n25__r0.840__s1.93",
    "rule_195__n25__r0.840__s1.93",
    "rule_196__n25__r0.840__s1.93",
]

def build_conditions(df):
    conditions = {
            "rule_001__n27__r0.963__s2.98":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.71),
            "rule_002__n25__r0.960__s2.88":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.71),
            "rule_003__n25__r0.960__s2.88":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.71),
            "rule_004__n32__r0.906__s2.60":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_005__n26__r0.923__s2.59":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_006__n31__r0.903__s2.55":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.71),
            "rule_007__n25__r0.920__s2.54":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_008__n29__r0.897__s2.45":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_009__n29__r0.897__s2.45":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_010__n29__r0.897__s2.45":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.661),
            "rule_011__n28__r0.893__s2.40":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_012__n28__r0.893__s2.40":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.76),
            "rule_013__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.261),
            "rule_014__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.31),
            "rule_015__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.36),
            "rule_016__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.41),
            "rule_017__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_018__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_019__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_020__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.661),
            "rule_021__n27__r0.889__s2.34":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.661),
            "rule_022__n36__r0.861__s2.31":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_023__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.261),
            "rule_024__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.31),
            "rule_025__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.36),
            "rule_026__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.41),
            "rule_027__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_028__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_029__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_030__n26__r0.885__s2.28":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.661),
            "rule_031__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.71),
            "rule_032__n26__r0.885__s2.28":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_033__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -4.29),
            "rule_034__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -4.024),
            "rule_035__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -3.755),
            "rule_036__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -3.491),
            "rule_037__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.261),
            "rule_038__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.31),
            "rule_039__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.36),
            "rule_040__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.41),
            "rule_041__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_042__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_043__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_044__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.71),
            "rule_045__n25__r0.880__s2.22":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.71),
            "rule_046__n34__r0.853__s2.21":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_047__n29__r0.862__s2.18":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_048__n29__r0.862__s2.18":
                (df["trend_signal"] >= -21.917) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_049__n33__r0.848__s2.15":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_050__n33__r0.848__s2.15":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_051__n33__r0.848__s2.15":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.661),
            "rule_052__n37__r0.838__s2.14":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_053__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345),
            "rule_054__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 1.398),
            "rule_055__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 2.83),
            "rule_056__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 4.294),
            "rule_057__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 5.92),
            "rule_058__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 5.181),
            "rule_059__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 6.015),
            "rule_060__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 6.779),
            "rule_061__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 7.473),
            "rule_062__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 8.148),
            "rule_063__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 8.84),
            "rule_064__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 9.524),
            "rule_065__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 10.222),
            "rule_066__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 11.06),
            "rule_067__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -2.98),
            "rule_068__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -2.749),
            "rule_069__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.06),
            "rule_070__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.111),
            "rule_071__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.16),
            "rule_072__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.21),
            "rule_073__n28__r0.857__s2.12":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_074__n28__r0.857__s2.12":
                (df["trend_signal"] >= -11.333) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_075__n32__r0.844__s2.10":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.261),
            "rule_076__n32__r0.844__s2.10":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.31),
            "rule_077__n32__r0.844__s2.10":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.36),
            "rule_078__n32__r0.844__s2.10":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.41),
            "rule_079__n32__r0.844__s2.10":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_080__n32__r0.844__s2.10":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_081__n32__r0.844__s2.10":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_082__n36__r0.833__s2.09":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_083__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345),
            "rule_084__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 1.398),
            "rule_085__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 2.83),
            "rule_086__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 4.294),
            "rule_087__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 5.92),
            "rule_088__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 5.181),
            "rule_089__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 6.015),
            "rule_090__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 6.779),
            "rule_091__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 7.473),
            "rule_092__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 8.148),
            "rule_093__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 8.84),
            "rule_094__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 9.524),
            "rule_095__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 10.222),
            "rule_096__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 11.06),
            "rule_097__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -2.98),
            "rule_098__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -2.749),
            "rule_099__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.06),
            "rule_100__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.111),
            "rule_101__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.16),
            "rule_102__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.21),
            "rule_103__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.661),
            "rule_104__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_105__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_106__n27__r0.852__s2.06":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= -4.083) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_107__n27__r0.852__s2.06":
                (df["MACD_hist_3d_rank"] >= 0.71) &
                (df["dist_from_low"] >= 22.317) &
                (df["trend_signal"] <= -4.083) &
                (df["MACD_acc"] <= 87.776),
            "rule_108__n27__r0.852__s2.06":
                (df["trend_signal"] >= -21.917) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 22.316),
            "rule_109__n31__r0.839__s2.04":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_110__n31__r0.839__s2.04":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_111__n31__r0.839__s2.04":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_112__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477),
            "rule_113__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 7.687),
            "rule_114__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 9.717),
            "rule_115__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 11.92),
            "rule_116__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 12.852),
            "rule_117__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -3.241),
            "rule_118__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 11.92),
            "rule_119__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 12.852),
            "rule_120__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_acc"] >= 1.398),
            "rule_121__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_acc"] >= 2.83),
            "rule_122__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_acc"] >= 4.294),
            "rule_123__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_acc"] >= 5.92),
            "rule_124__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 5.181),
            "rule_125__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 6.015),
            "rule_126__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 6.779),
            "rule_127__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 7.473),
            "rule_128__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 8.148),
            "rule_129__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 8.84),
            "rule_130__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 9.524),
            "rule_131__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 10.222),
            "rule_132__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 11.06),
            "rule_133__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["max_drop_7d"] <= -2.98),
            "rule_134__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["max_drop_7d"] <= -2.749),
            "rule_135__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.06),
            "rule_136__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.111),
            "rule_137__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.16),
            "rule_138__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.21),
            "rule_139__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.661),
            "rule_140__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_141__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_142__n26__r0.846__s2.00":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.816) &
                (df["dist_from_low"] >= 22.317),
            "rule_143__n26__r0.846__s2.00":
                (df["dist_from_low"] >= 26.35) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_144__n26__r0.846__s2.00":
                (df["trend_signal"] >= -11.333) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 22.316),
            "rule_145__n26__r0.846__s2.00":
                (df["trend_signal"] >= -21.917) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 27.328),
            "rule_146__n30__r0.833__s1.99":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -4.024),
            "rule_147__n30__r0.833__s1.99":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -3.755),
            "rule_148__n30__r0.833__s1.99":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -3.491),
            "rule_149__n30__r0.833__s1.99":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_150__n30__r0.833__s1.99":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_hist_3d_rank"] >= 0.56),
            "rule_151__n30__r0.833__s1.99":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_152__n30__r0.833__s1.99":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.611),
            "rule_153__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227),
            "rule_154__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 12.158),
            "rule_155__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 58.342) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] >= -15.044),
            "rule_156__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 7.687),
            "rule_157__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 9.717),
            "rule_158__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 86.812) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] <= -3.241),
            "rule_159__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_acc"] >= 7.687),
            "rule_160__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["MACD_acc"] >= 9.717),
            "rule_161__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 11.92),
            "rule_162__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 132.083) &
                (df["tr_value_ratio"] <= 1.477) &
                (df["dist_from_low"] >= 12.852),
            "rule_163__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_acc"] >= 1.398),
            "rule_164__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 5.181),
            "rule_165__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 6.015),
            "rule_166__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 6.779),
            "rule_167__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 7.473),
            "rule_168__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 8.148),
            "rule_169__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 8.84),
            "rule_170__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 9.524),
            "rule_171__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 10.222),
            "rule_172__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["dist_from_low"] >= 11.06),
            "rule_173__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["max_drop_7d"] <= -2.98),
            "rule_174__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["max_drop_7d"] <= -2.749),
            "rule_175__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.06),
            "rule_176__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.111),
            "rule_177__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.16),
            "rule_178__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.21),
            "rule_179__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.261),
            "rule_180__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.31),
            "rule_181__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.36),
            "rule_182__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.41),
            "rule_183__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.46),
            "rule_184__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_185__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["dist_from_low"] >= 14.954),
            "rule_186__n25__r0.840__s1.93":
                (df["trend_signal"] >= 38.819) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["max_drop_7d"] >= -11.519),
            "rule_187__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_188__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 25.833) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_189__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_190__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 15.429) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_hist_3d_rank"] >= 0.509),
            "rule_191__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 7.333) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 18.238),
            "rule_192__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= 1.502) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 22.316),
            "rule_193__n25__r0.840__s1.93":
                (df["today_pct"] >= 16.138) &
                (df["MACD_hist_3d_rank"] <= 0.41) &
                (df["trend_signal"] >= 25.833) &
                (df["MACD_acc"] >= 14.902),
            "rule_194__n25__r0.840__s1.93":
                (df["today_pct"] >= 9.87) &
                (df["trend_signal"] >= -4.083) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 22.316),
            "rule_195__n25__r0.840__s1.93":
                (df["trend_signal"] >= -11.333) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.345) &
                (df["MACD_acc"] >= 27.328),
            "rule_196__n25__r0.840__s1.93":
                (df["trend_signal"] >= -21.917) &
                (df["today_pct"] >= 9.87) &
                (df["tr_value_ratio"] <= 1.227) &
                (df["MACD_acc"] >= 14.902),
    }
    return conditions
