# auto-generated: lowscan rules (filtered)
# usage:
#   from lowscan_rules import build_conditions, RULE_NAMES
#   find_conditions = build_conditions(df)

import numpy as np

RULE_NAMES = [
    "rule_001__n35__r0.800",
    "rule_002__n35__r0.800",
    "rule_003__n34__r0.824",
    "rule_004__n33__r0.818",
    "rule_005__n33__r0.818",
    "rule_006__n33__r0.818",
    "rule_007__n33__r0.818",
    "rule_008__n32__r0.844",
    "rule_009__n31__r0.806",
    "rule_010__n31__r0.806",
    "rule_011__n31__r0.806",
    "rule_012__n30__r0.833",
    "rule_013__n30__r0.800",
    "rule_014__n30__r0.800",
    "rule_015__n30__r0.800",
    "rule_016__n30__r0.800",
    "rule_017__n30__r0.800",
    "rule_018__n30__r0.800",
    "rule_019__n30__r0.800",
    "rule_020__n30__r0.800",
    "rule_021__n30__r0.800",
    "rule_022__n30__r0.800",
    "rule_023__n30__r0.800",
    "rule_024__n29__r0.828",
    "rule_025__n29__r0.828",
    "rule_026__n28__r0.821",
    "rule_027__n28__r0.821",
    "rule_028__n28__r0.821",
    "rule_029__n28__r0.821",
    "rule_030__n28__r0.821",
    "rule_031__n28__r0.821",
    "rule_032__n27__r0.815",
    "rule_033__n27__r0.815",
    "rule_034__n27__r0.815",
    "rule_035__n27__r0.815",
    "rule_036__n27__r0.815",
    "rule_037__n27__r0.815",
    "rule_038__n27__r0.815",
    "rule_039__n27__r0.815",
    "rule_040__n27__r0.815",
    "rule_041__n27__r0.815",
    "rule_042__n26__r0.846",
    "rule_043__n26__r0.846",
    "rule_044__n26__r0.808",
    "rule_045__n26__r0.808",
    "rule_046__n26__r0.808",
    "rule_047__n26__r0.808",
    "rule_048__n26__r0.808",
    "rule_049__n26__r0.808",
    "rule_050__n26__r0.808",
    "rule_051__n26__r0.808",
    "rule_052__n26__r0.808",
    "rule_053__n26__r0.808",
    "rule_054__n26__r0.808",
    "rule_055__n26__r0.808",
    "rule_056__n26__r0.808",
    "rule_057__n26__r0.808",
    "rule_058__n26__r0.808",
    "rule_062__n25__r0.840",
    "rule_063__n25__r0.840",
    "rule_064__n25__r0.840",
    "rule_065__n25__r0.840",
    "rule_066__n25__r0.840",
    "rule_067__n25__r0.840",
    "rule_068__n25__r0.840",
    "rule_069__n25__r0.840",
    "rule_070__n25__r0.840",
    "rule_071__n25__r0.840",
    "rule_072__n25__r0.840",
    "rule_073__n25__r0.840",
    "rule_074__n25__r0.840",
    "rule_075__n25__r0.840",
    "rule_076__n25__r0.840",
    "rule_077__n25__r0.840",
    "rule_078__n25__r0.840",
    "rule_079__n25__r0.840",
    "rule_080__n25__r0.840",
    "rule_081__n25__r0.840",
    "rule_082__n25__r0.840",
    "rule_083__n25__r0.840",
    "rule_084__n25__r0.840",
    "rule_085__n25__r0.840",
    "rule_086__n25__r0.840",
    "rule_087__n25__r0.840",
    "rule_088__n25__r0.840",
    "rule_089__n25__r0.840",
    "rule_090__n25__r0.840",
    "rule_091__n25__r0.840",
    "rule_092__n25__r0.840",
    "rule_093__n25__r0.840",
    "rule_094__n25__r0.840",
    "rule_095__n25__r0.840",
    "rule_096__n25__r0.840",
    "rule_097__n25__r0.840",
    "rule_098__n25__r0.840",
    "rule_099__n25__r0.840",
    "rule_100__n25__r0.840",
    "rule_101__n25__r0.840",
    "rule_102__n25__r0.840",
    "rule_103__n25__r0.840",
    "rule_104__n25__r0.840",
    "rule_105__n25__r0.840",
    "rule_106__n25__r0.840",
    "rule_107__n25__r0.840",
    "rule_108__n25__r0.840",
    "rule_109__n25__r0.840",
    "rule_110__n25__r0.840",
    "rule_111__n25__r0.840",
    "rule_112__n25__r0.840",
    "rule_113__n25__r0.840",
    "rule_114__n25__r0.840",
    "rule_115__n25__r0.840",
    "rule_116__n25__r0.840",
    "rule_117__n25__r0.840",
    "rule_118__n25__r0.840",
    "rule_119__n25__r0.840",
    "rule_120__n25__r0.840",
    "rule_121__n25__r0.800",
    "rule_122__n25__r0.800",
    "rule_123__n25__r0.800",
    "rule_124__n25__r0.800",
    "rule_125__n25__r0.800",
    "rule_126__n25__r0.800",
    "rule_127__n25__r0.800",
    "rule_128__n25__r0.800",
    "rule_129__n25__r0.800",
    "rule_130__n25__r0.800",
    "rule_131__n25__r0.800",
    "rule_132__n25__r0.800",
    "rule_133__n25__r0.800",
    "rule_134__n25__r0.800",
    "rule_135__n25__r0.800",
    "rule_136__n25__r0.800",
    "rule_137__n25__r0.800",
    "rule_138__n25__r0.800",
    "rule_139__n25__r0.800",
    "rule_140__n25__r0.800",
    "rule_141__n25__r0.800",
    "rule_142__n25__r0.800",
    "rule_143__n25__r0.800",
    "rule_144__n25__r0.800",
]

def build_conditions(df):
    conditions = {
            "rule_001__n35__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.866) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_002__n35__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -46.931) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.841),
            "rule_003__n34__r0.824":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.817) &
                (df["today_chg_rate"] >= -46.931),
            "rule_004__n33__r0.818":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 45.0),
            "rule_005__n33__r0.818":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["close_pos"] <= 0.817) &
                (df["today_chg_rate"] >= -25.422) &
                (df["today_tr_val"] >= 9896998668.0),
            "rule_006__n33__r0.818":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.841) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_007__n33__r0.818":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.817) &
                (df["today_chg_rate"] >= -38.443),
            "rule_008__n32__r0.844":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["close_pos"] <= 0.817) &
                (df["today_chg_rate"] >= -25.422) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_009__n31__r0.806":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -23.42) &
                (df["pos20_ratio"] <= 45.0),
            "rule_010__n31__r0.806":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.841) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_011__n31__r0.806":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.866) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_012__n30__r0.833":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["close_pos"] <= 0.817) &
                (df["today_chg_rate"] >= -25.422) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_013__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -38.443) &
                (df["close_pos"] <= 0.656) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_014__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.841) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_015__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["close_pos"] <= 0.817) &
                (df["today_tr_val"] >= 43714162796.0) &
                (df["today_chg_rate"] >= -46.931),
            "rule_016__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["close_pos"] <= 0.817) &
                (df["today_tr_val"] >= 43714162796.0) &
                (df["today_chg_rate"] >= -38.443),
            "rule_017__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_tr_val"] >= 43714162796.0) &
                (df["close_pos"] <= 0.841) &
                (df["today_chg_rate"] >= -46.931),
            "rule_018__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_tr_val"] >= 43714162796.0) &
                (df["close_pos"] <= 0.841) &
                (df["today_chg_rate"] >= -38.443),
            "rule_019__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.789) &
                (df["today_tr_val"] >= 9896998668.0),
            "rule_020__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.817) &
                (df["today_chg_rate"] >= -33.63),
            "rule_021__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -46.931) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.73),
            "rule_022__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -46.931) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.76),
            "rule_023__n30__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -46.931) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.789),
            "rule_024__n29__r0.828":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["close_pos"] <= 0.817) &
                (df["today_chg_rate"] >= -25.422) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_025__n29__r0.828":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.789) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_026__n28__r0.821":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["mean_prev3"] >= 3053509940.667),
            "rule_027__n28__r0.821":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["ma5_chg_rate"] >= 2.089),
            "rule_028__n28__r0.821":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -38.443) &
                (df["close_pos"] <= 0.656) &
                (df["pos20_ratio"] <= 40.0),
            "rule_029__n28__r0.821":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["close_pos"] <= 0.817) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_030__n28__r0.821":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.76) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_031__n28__r0.821":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["vol15"] <= 8.145) &
                (df["close_pos"] <= 0.866),
            "rule_032__n27__r0.815":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["mean_prev3"] >= 4019396870.667),
            "rule_033__n27__r0.815":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["pct_vs_lastweek"] >= 12.005),
            "rule_034__n27__r0.815":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["today_pct"] >= 6.286),
            "rule_035__n27__r0.815":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 118.428) &
                (df["vol30"] >= 6.079) &
                (df["pos20_ratio"] <= 50.0),
            "rule_036__n27__r0.815":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -38.443) &
                (df["close_pos"] <= 0.656) &
                (df["pct_vs_lastweek"] >= 4.883),
            "rule_037__n27__r0.815":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.73) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_038__n27__r0.815":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -27.822) &
                (df["pos20_ratio"] <= 40.0),
            "rule_039__n27__r0.815":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.789) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_040__n27__r0.815":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -46.931) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.697),
            "rule_041__n27__r0.815":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 4.857) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0),
            "rule_042__n26__r0.846":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.866),
            "rule_043__n26__r0.846":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 5.389) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0),
            "rule_044__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["vol15"] >= 3.463),
            "rule_045__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["mean_prev3"] >= 5364848951.667),
            "rule_046__n26__r0.808":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["ma5_chg_rate"] >= 2.456),
            "rule_047__n26__r0.808":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["today_pct"] >= 6.942),
            "rule_048__n26__r0.808":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["today_pct"] >= 7.749),
            "rule_049__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.73) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_050__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["close_pos"] <= 0.817) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_051__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["close_pos"] <= 0.817) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_052__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.76) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_053__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.76) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_054__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -38.443) &
                (df["close_pos"] <= 0.697) &
                (df["mean_prev3"] >= 19373791910.0),
            "rule_055__n26__r0.808":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.789) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_056__n26__r0.808":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 4.857) &
                (df["pos20_ratio"] <= 40.0) &
                (df["today_chg_rate"] >= -23.42),
            "rule_057__n26__r0.808":
                (df["vol15"] >= 8.145) &
                (df["pct_vs_last4week"] >= 0.881) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["three_m_chg_rate"] >= 67.379),
            "rule_058__n26__r0.808":
                (df["vol15"] >= 8.145) &
                (df["pct_vs_last4week"] >= 2.332) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["three_m_chg_rate"] >= 67.379),
            "rule_062__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 6.079) &
                (df["three_m_chg_rate"] <= 98.006),
            "rule_063__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= -1.527),
            "rule_064__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= -0.903),
            "rule_065__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= -0.564),
            "rule_066__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= -0.322),
            "rule_067__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= -0.084),
            "rule_068__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 0.103),
            "rule_069__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 0.271),
            "rule_070__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 0.435),
            "rule_071__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 0.596),
            "rule_072__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 0.754),
            "rule_073__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 0.942),
            "rule_074__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 1.095),
            "rule_075__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 1.288),
            "rule_076__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 1.509),
            "rule_077__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 1.783),
            "rule_078__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["ma5_chg_rate"] >= 2.089),
            "rule_079__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pos20_ratio"] <= 50.0),
            "rule_080__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pos20_ratio"] <= 55.0),
            "rule_081__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 359096543.0),
            "rule_082__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 429850133.6),
            "rule_083__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 504448587.333),
            "rule_084__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 590656526.267),
            "rule_085__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 685127739.5),
            "rule_086__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 818561798.333),
            "rule_087__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["mean_prev3"] >= 961260284.5),
            "rule_088__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -46.931),
            "rule_089__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -38.443),
            "rule_090__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -33.63),
            "rule_091__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -30.368),
            "rule_092__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_chg_rate"] >= -27.822),
            "rule_093__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_lastweek"] >= -3.284),
            "rule_094__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_lastweek"] >= -1.092),
            "rule_095__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_lastweek"] >= 0.402),
            "rule_096__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_lastweek"] >= 1.689),
            "rule_097__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -23.863),
            "rule_098__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -18.498),
            "rule_099__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -15.204),
            "rule_100__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -12.823),
            "rule_101__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -10.89),
            "rule_102__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["pct_vs_last4week"] >= -9.376),
            "rule_103__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 3.107),
            "rule_104__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 3.233),
            "rule_105__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 3.366),
            "rule_106__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 3.525),
            "rule_107__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 3.682),
            "rule_108__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 3.852),
            "rule_109__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 4.053),
            "rule_110__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["today_pct"] >= 4.255),
            "rule_111__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["close_pos"] >= 0.338),
            "rule_112__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 6.079) &
                (df["close_pos"] <= 1.0),
            "rule_113__n25__r0.840":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["vol30"] >= 4.499),
            "rule_114__n25__r0.840":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["vol30"] >= 4.857),
            "rule_115__n25__r0.840":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["vol30"] >= 5.389),
            "rule_116__n25__r0.840":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["vol30"] >= 6.079),
            "rule_117__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["pct_vs_lastweek"] >= 14.756),
            "rule_118__n25__r0.840":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.841),
            "rule_119__n25__r0.840":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["vol15"] <= 8.145) &
                (df["close_pos"] <= 0.841),
            "rule_120__n25__r0.840":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["today_chg_rate"] >= -23.42) &
                (df["vol30"] >= 5.389) &
                (df["pos20_ratio"] <= 40.0),
            "rule_121__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["pos20_ratio"] <= 40.0) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_122__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["pos20_ratio"] <= 40.0) &
                (df["vol15"] >= 2.999),
            "rule_123__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["pos20_ratio"] <= 40.0) &
                (df["vol15"] >= 3.133),
            "rule_124__n25__r0.800":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["three_m_chg_rate"] <= 98.006) &
                (df["vol30"] >= 5.389) &
                (df["today_pct"] >= 8.958),
            "rule_125__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.656) &
                (df["pos20_ratio"] <= 50.0),
            "rule_126__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.656) &
                (df["mean_prev3"] >= 3053509940.667),
            "rule_127__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.697) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_128__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -38.443) &
                (df["close_pos"] <= 0.656) &
                (df["today_tr_val"] >= 27312427005.0),
            "rule_129__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.73) &
                (df["pos20_ratio"] <= 45.0),
            "rule_130__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -25.422) &
                (df["close_pos"] <= 0.73) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_131__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["pos20_ratio"] <= 45.0) &
                (df["mean_prev3"] >= 11555255167.0),
            "rule_132__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -38.443) &
                (df["close_pos"] <= 0.697) &
                (df["pct_vs_lastweek"] >= 6.786),
            "rule_133__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.889),
            "rule_134__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["vol30"] <= 8.359) &
                (df["close_pos"] <= 0.889),
            "rule_135__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["pct_vs_last4week"] >= -10.89) &
                (df["mean_prev3"] >= 19373791910.0) &
                (df["close_pos"] <= 0.866),
            "rule_136__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["close_pos"] <= 0.656) &
                (df["today_chg_rate"] >= -33.63) &
                (df["today_tr_val"] >= 19121245980.0),
            "rule_137__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -23.42) &
                (df["close_pos"] <= 0.789) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_138__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -21.803) &
                (df["close_pos"] <= 0.817) &
                (df["today_tr_val"] >= 13478294480.0),
            "rule_139__n25__r0.800":
                (df["three_m_chg_rate"] >= 160.255) &
                (df["today_chg_rate"] >= -30.368) &
                (df["close_pos"] <= 0.817) &
                (df["pct_vs_lastweek"] >= 8.684),
            "rule_140__n25__r0.800":
                (df["today_tr_val"] >= 88739853940.0) &
                (df["vol30"] >= 4.857) &
                (df["pos20_ratio"] <= 40.0) &
                (df["today_chg_rate"] >= -21.803),
            "rule_141__n25__r0.800":
                (df["vol15"] >= 8.145) &
                (df["pct_vs_last4week"] >= 0.881) &
                (df["mean_prev3"] <= 1982160968.0) &
                (df["three_m_chg_rate"] >= 61.478),
            "rule_142__n25__r0.800":
                (df["vol15"] >= 8.145) &
                (df["three_m_chg_rate"] >= 61.478) &
                (df["mean_prev3"] <= 1371635202.4) &
                (df["pct_vs_last4week"] >= -4.596),
            "rule_143__n25__r0.800":
                (df["vol15"] >= 8.145) &
                (df["mean_prev3"] <= 2442790786.667) &
                (df["today_chg_rate"] >= -30.368) &
                (df["three_m_chg_rate"] >= 67.379),
            "rule_144__n25__r0.800":
                (df["vol15"] >= 8.145) &
                (df["pct_vs_last4week"] >= 2.332) &
                (df["mean_prev3"] <= 1982160968.0) &
                (df["three_m_chg_rate"] >= 61.478),
    }
    return conditions
