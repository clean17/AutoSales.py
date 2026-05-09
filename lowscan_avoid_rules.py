# auto-generated: lowscan avoid rules for class_0
# usage:
#   import lowscan_avoid_rules
#
#   avoid_conditions = lowscan_avoid_rules.build_conditions(df)
#   avoid_mask = np.zeros(len(df), dtype=bool)
#   for cond in avoid_conditions.values():
#       avoid_mask |= cond
#
#   df = df[~avoid_mask].copy()
import numpy as np

RULE_NAMES = [
    "avoid_896_s1.80_n85_bad0.788_p0.118_strong0.059",
    "avoid_1286_s1.75_n83_bad0.783_p0.120_strong0.060",
    "avoid_1896_s1.70_n55_bad0.764_p0.073_strong0.036",
    "avoid_083_s2.12_n45_bad0.800_p0.022_strong0.000",
    "avoid_374_s1.93_n57_bad0.807_p0.088_strong0.035",
    "avoid_156_s2.03_n50_bad0.820_p0.080_strong0.020",
    "avoid_1991_s1.70_n44_bad0.795_p0.091_strong0.045",
    "avoid_047_s2.24_n32_bad0.875_p0.031_strong0.031",
    "avoid_2523_s1.66_n54_bad0.759_p0.074_strong0.037",
    "avoid_2196_s1.69_n41_bad0.805_p0.098_strong0.049",
    "avoid_766_s1.83_n30_bad0.833_p0.067_strong0.033",
    "avoid_861_s1.82_n35_bad0.829_p0.057_strong0.057",
    "avoid_3192_s1.63_n60_bad0.783_p0.150_strong0.033",
    "avoid_1583_s1.72_n46_bad0.804_p0.109_strong0.043",
    "avoid_724_s1.84_n34_bad0.853_p0.088_strong0.059",
    "avoid_1675_s1.72_n33_bad0.818_p0.061_strong0.061",
    "avoid_394_s1.92_n30_bad0.867_p0.100_strong0.033",
    "avoid_292_s1.95_n32_bad0.844_p0.062_strong0.031",
    "avoid_3298_s1.63_n31_bad0.774_p0.032_strong0.032",
    "avoid_3943_s1.61_n46_bad0.761_p0.065_strong0.043",
    "avoid_3491_s1.63_n31_bad0.806_p0.097_strong0.032",
    "avoid_2601_s1.66_n34_bad0.765_p0.059_strong0.000",
    "avoid_4666_s1.58_n33_bad0.788_p0.091_strong0.030",
    "avoid_257_s1.97_n31_bad0.871_p0.097_strong0.032",
    "avoid_2348_s1.68_n31_bad0.839_p0.097_strong0.065",
    "avoid_3708_s1.62_n30_bad0.833_p0.100_strong0.067",
    "avoid_2218_s1.68_n32_bad0.812_p0.094_strong0.031",
    "avoid_1561_s1.73_n40_bad0.800_p0.100_strong0.025",
    "avoid_1488_s1.74_n30_bad0.833_p0.100_strong0.033",
    "avoid_1854_s1.71_n30_bad0.833_p0.067_strong0.067",
    "avoid_2350_s1.68_n31_bad0.839_p0.097_strong0.065",
    "avoid_1327_s1.74_n36_bad0.806_p0.083_strong0.028",
    "avoid_3715_s1.62_n30_bad0.833_p0.100_strong0.067",
    "avoid_826_s1.82_n32_bad0.844_p0.062_strong0.062",
    "avoid_2646_s1.66_n32_bad0.812_p0.062_strong0.062",
    "avoid_097_s2.08_n37_bad0.865_p0.054_strong0.054",
    "avoid_918_s1.80_n40_bad0.800_p0.075_strong0.025",
    "avoid_1915_s1.70_n33_bad0.848_p0.121_strong0.061",
    "avoid_1920_s1.70_n33_bad0.848_p0.121_strong0.061",
    "avoid_3199_s1.63_n41_bad0.829_p0.098_strong0.098",
    "avoid_1885_s1.71_n40_bad0.800_p0.075_strong0.050",
    "avoid_1437_s1.74_n37_bad0.757_p0.027_strong0.000",
    "avoid_1580_s1.72_n39_bad0.821_p0.103_strong0.051",
    "avoid_3710_s1.62_n30_bad0.833_p0.100_strong0.067",
    "avoid_436_s1.89_n42_bad0.810_p0.071_strong0.024",
    "avoid_4962_s1.58_n32_bad0.812_p0.094_strong0.062",
    "avoid_2231_s1.68_n33_bad0.788_p0.091_strong0.000",
    "avoid_776_s1.83_n31_bad0.806_p0.065_strong0.000",
    "avoid_4846_s1.58_n37_bad0.757_p0.054_strong0.027",
    "avoid_4388_s1.59_n30_bad0.767_p0.067_strong0.000",
    "avoid_4466_s1.59_n30_bad0.767_p0.067_strong0.000",
    "avoid_090_s2.09_n34_bad0.824_p0.029_strong0.000",
    "avoid_645_s1.85_n32_bad0.844_p0.094_strong0.031",
    "avoid_3949_s1.61_n32_bad0.781_p0.062_strong0.031",
    "avoid_4671_s1.58_n33_bad0.788_p0.091_strong0.030",
    "avoid_4976_s1.58_n37_bad0.784_p0.108_strong0.027",
    "avoid_4999_s1.58_n42_bad0.738_p0.071_strong0.000",
    "avoid_2224_s1.68_n32_bad0.812_p0.094_strong0.031",
    "avoid_939_s1.79_n31_bad0.839_p0.097_strong0.032",
    "avoid_2266_s1.68_n30_bad0.800_p0.100_strong0.000",
    "avoid_1237_s1.76_n33_bad0.818_p0.121_strong0.000",
]

def build_conditions(df):
    conditions = {
            "avoid_896_s1.80_n85_bad0.788_p0.118_strong0.059":
                (df["tr_val_rank_20d"] <= 0.45) &
                (df["max_drop_7d"] > -3.265) &
                (df["today_pct"] > 3.77) &
                (df["dist_to_ma5"] <= 2.639) &
                (df["dist_from_low_20d"] > 5.243),
            "avoid_1286_s1.75_n83_bad0.783_p0.120_strong0.060":
                (df["ma5_ma20_gap_chg_1d"] <= -0.505) &
                (df["ATR_pct"] <= 7.106) &
                (df["tr_val_rank_20d"] <= 0.15) &
                (df["today_pct"] > 3.6) &
                (df["dist_from_low_20d"] <= 9.649),
            "avoid_1896_s1.70_n55_bad0.764_p0.073_strong0.036":
                (df["ATR_pct"] <= 5.476) &
                (df["today_tr_val_eok"] <= 3.192) &
                (df["today_pct"] <= 4.13) &
                (df["ma5_ma20_gap_chg_1d"] <= 0.744) &
                (df["MACD_hist_3d"] > -10.947),
            "avoid_083_s2.12_n45_bad0.800_p0.022_strong0.000":
                (df["max_drop_7d"] <= -4.342) &
                (df["ATR_pct"] <= 3.68) &
                (df["ma5_ma20_gap_chg_1d"] <= 0.903) &
                (df["today_pct"] <= 4.81) &
                (df["today_tr_val_eok"] > 4.727),
            "avoid_374_s1.93_n57_bad0.807_p0.088_strong0.035":
                (df["ma5_ma20_gap_chg_1d"] <= -0.505) &
                (df["dist_to_ma5"] > 1.388) &
                (df["max_drop_7d"] > -7.098) &
                (df["today_tr_val_eok"] <= 26.055) &
                (df["ATR_pct"] <= 6.512),
            "avoid_156_s2.03_n50_bad0.820_p0.080_strong0.020":
                (df["ATR_pct"] <= 3.68) &
                (df["ma5_ma20_gap_chg_1d"] <= 0.252) &
                (df["dist_from_low_20d"] > 5.243) &
                (df["today_tr_val_eok"] > 8.164) &
                (df["dist_to_ma5"] > 0.848),
            "avoid_1991_s1.70_n44_bad0.795_p0.091_strong0.045":
                (df["dist_to_ma5"] > 3.369) &
                (df["ATR_pct"] <= 4.126) &
                (df["today_tr_val_eok"] <= 8.164) &
                (df["ma5_ma20_gap_chg_1d"] <= 1.793) &
                (df["today_pct"] <= 6.67),
            "avoid_047_s2.24_n32_bad0.875_p0.031_strong0.031":
                (df["ATR_pct"] <= 5.744) &
                (df["max_drop_7d"] <= -9.595) &
                (df["MACD_hist_3d"] <= 31.298) &
                (df["today_pct"] <= 5.1) &
                (df["tr_val_rank_20d"] <= 0.8),
            "avoid_2523_s1.66_n54_bad0.759_p0.074_strong0.037":
                (df["tr_val_rank_20d"] <= 0.15) &
                (df["dist_to_ma5"] <= 1.388) &
                (df["ATR_pct"] <= 5.241) &
                (df["ma5_ma20_gap_chg_1d"] > 0.061) &
                (df["max_drop_7d"] <= -2.762),
            "avoid_2196_s1.69_n41_bad0.805_p0.098_strong0.049":
                (df["max_drop_7d"] > -5.317) &
                (df["tr_val_rank_20d"] <= 0.4) &
                (df["MACD_hist_3d"] > 537.196) &
                (df["today_pct"] > 3.94) &
                (df["dist_to_ma5"] > 1.388),
            "avoid_766_s1.83_n30_bad0.833_p0.067_strong0.033":
                (df["today_pct"] <= 4.33) &
                (df["tr_val_rank_20d"] <= 0.2) &
                (df["dist_from_low_20d"] > 16.505) &
                (df["dist_to_ma5"] > 1.86) &
                (df["MACD_hist_3d"] <= 184.71),
            "avoid_861_s1.82_n35_bad0.829_p0.057_strong0.057":
                (df["MACD_hist_3d"] <= -10.947) &
                (df["dist_to_ma5"] > 1.388) &
                (df["tr_val_rank_20d"] <= 0.65) &
                (df["ma5_ma20_gap_chg_1d"] > -0.505) &
                (df["today_pct"] <= 6.67),
            "avoid_3192_s1.63_n60_bad0.783_p0.150_strong0.033":
                (df["ma5_ma20_gap_chg_1d"] <= -0.505) &
                (df["tr_val_rank_20d"] <= 0.65) &
                (df["ATR_pct"] <= 6.249) &
                (df["MACD_hist_3d"] <= -57.266) &
                (df["dist_to_ma5"] <= 0.023),
            "avoid_1583_s1.72_n46_bad0.804_p0.109_strong0.043":
                (df["ma5_ma20_gap_chg_1d"] <= -0.186) &
                (df["ATR_pct"] <= 5.476) &
                (df["tr_val_rank_20d"] <= 0.4) &
                (df["MACD_hist_3d"] > 0.666) &
                (df["max_drop_7d"] <= -3.265),
            "avoid_724_s1.84_n34_bad0.853_p0.088_strong0.059":
                (df["ATR_pct"] <= 3.68) &
                (df["MACD_hist_3d"] <= 52.056) &
                (df["dist_to_ma5"] > 3.369) &
                (df["max_drop_7d"] > -3.0) &
                (df["dist_from_low_20d"] <= 13.02),
            "avoid_1675_s1.72_n33_bad0.818_p0.061_strong0.061":
                (df["today_tr_val_eok"] <= 6.386) &
                (df["max_drop_7d"] > -3.265) &
                (df["MACD_hist_3d"] > 52.056) &
                (df["today_pct"] <= 4.81) &
                (df["dist_to_ma5"] > 0.848),
            "avoid_394_s1.92_n30_bad0.867_p0.100_strong0.033":
                (df["dist_from_low_20d"] <= 13.973) &
                (df["max_drop_7d"] > -3.265) &
                (df["today_tr_val_eok"] <= 10.518) &
                (df["dist_to_ma5"] > 4.213) &
                (df["MACD_hist_3d"] <= 40.68),
            "avoid_292_s1.95_n32_bad0.844_p0.062_strong0.031":
                (df["ATR_pct"] > 5.476) &
                (df["tr_val_rank_20d"] <= 0.05) &
                (df["max_drop_7d"] > -4.989) &
                (df["dist_to_ma5"] <= 1.86) &
                (df["MACD_hist_3d"] <= 31.298),
            "avoid_3298_s1.63_n31_bad0.774_p0.032_strong0.032":
                (df["MACD_hist_3d"] > 52.056) &
                (df["ATR_pct"] <= 3.68) &
                (df["tr_val_rank_20d"] <= 0.8) &
                (df["dist_from_low_20d"] <= 6.087),
            "avoid_3943_s1.61_n46_bad0.761_p0.065_strong0.043":
                (df["ma5_ma20_gap_chg_1d"] <= 0.252) &
                (df["ATR_pct"] <= 4.126) &
                (df["tr_val_rank_20d"] <= 0.5) &
                (df["dist_from_low_20d"] > 5.243) &
                (df["MACD_hist_3d"] > -57.266),
            "avoid_3491_s1.63_n31_bad0.806_p0.097_strong0.032":
                (df["ATR_pct"] <= 3.68) &
                (df["ma5_ma20_gap_chg_1d"] <= 1.793) &
                (df["dist_from_low_20d"] > 11.223) &
                (df["dist_to_ma5"] <= 5.79) &
                (df["today_pct"] <= 6.67),
            "avoid_2601_s1.66_n34_bad0.765_p0.059_strong0.000":
                (df["tr_val_rank_20d"] <= 0.6) &
                (df["ATR_pct"] <= 4.755) &
                (df["max_drop_7d"] > -3.265) &
                (df["ma5_ma20_gap_chg_1d"] > 1.048) &
                (df["today_pct"] <= 4.57),
            "avoid_4666_s1.58_n33_bad0.788_p0.091_strong0.030":
                (df["ATR_pct"] <= 4.126) &
                (df["max_drop_7d"] <= -5.317) &
                (df["today_pct"] <= 4.33) &
                (df["tr_val_rank_20d"] > 0.55) &
                (df["dist_to_ma5"] > -1.554),
            "avoid_257_s1.97_n31_bad0.871_p0.097_strong0.032":
                (df["today_tr_val_eok"] <= 10.518) &
                (df["ma5_ma20_gap_chg_1d"] <= -0.505) &
                (df["max_drop_7d"] > -6.571) &
                (df["today_pct"] > 5.79) &
                (df["dist_to_ma5"] <= 3.003),
            "avoid_2348_s1.68_n31_bad0.839_p0.097_strong0.065":
                (df["today_pct"] <= 3.44) &
                (df["ma5_ma20_gap_chg_1d"] <= 0.575) &
                (df["ATR_pct"] <= 4.471) &
                (df["MACD_hist_3d"] > 10.89) &
                (df["max_drop_7d"] > -4.989),
            "avoid_3708_s1.62_n30_bad0.833_p0.100_strong0.067":
                (df["today_pct"] <= 3.77) &
                (df["max_drop_7d"] <= -11.619) &
                (df["ATR_pct"] <= 7.834) &
                (df["dist_to_ma5"] <= 7.468) &
                (df["tr_val_rank_20d"] > 0.05),
            "avoid_2218_s1.68_n32_bad0.812_p0.094_strong0.031":
                (df["max_drop_7d"] > -4.065) &
                (df["MACD_hist_3d"] <= 16.282) &
                (df["tr_val_rank_20d"] <= 0.15) &
                (df["today_pct"] > 3.94) &
                (df["ATR_pct"] > 4.126),
            "avoid_1561_s1.73_n40_bad0.800_p0.100_strong0.025":
                (df["today_tr_val_eok"] <= 8.164) &
                (df["ATR_pct"] <= 4.991) &
                (df["ma5_ma20_gap_chg_1d"] <= -0.186) &
                (df["max_drop_7d"] > -4.661) &
                (df["today_pct"] <= 4.33),
            "avoid_1488_s1.74_n30_bad0.833_p0.100_strong0.033":
                (df["ATR_pct"] <= 4.126) &
                (df["MACD_hist_3d"] <= 31.298) &
                (df["today_tr_val_eok"] > 53.876) &
                (df["dist_from_low_20d"] > 6.842) &
                (df["dist_to_ma5"] <= 7.468),
            "avoid_1854_s1.71_n30_bad0.833_p0.067_strong0.067":
                (df["dist_to_ma5"] <= 3.789) &
                (df["today_tr_val_eok"] <= 4.727) &
                (df["ATR_pct"] <= 4.755) &
                (df["dist_from_low_20d"] > 8.94) &
                (df["max_drop_7d"] > -5.317),
            "avoid_2350_s1.68_n31_bad0.839_p0.097_strong0.065":
                (df["max_drop_7d"] > -3.265) &
                (df["ma5_ma20_gap_chg_1d"] <= 1.048) &
                (df["today_tr_val_eok"] <= 13.144) &
                (df["dist_to_ma5"] > 3.789) &
                (df["ATR_pct"] <= 5.744),
            "avoid_1327_s1.74_n36_bad0.806_p0.083_strong0.028":
                (df["ma5_ma20_gap_chg_1d"] <= -1.087) &
                (df["ATR_pct"] <= 6.249) &
                (df["dist_from_low_20d"] <= 8.235) &
                (df["tr_val_rank_20d"] <= 0.75) &
                (df["today_pct"] > 4.33),
            "avoid_3715_s1.62_n30_bad0.833_p0.100_strong0.067":
                (df["today_pct"] <= 4.33) &
                (df["tr_val_rank_20d"] <= 0.15) &
                (df["dist_from_low_20d"] > 16.505) &
                (df["dist_to_ma5"] > 0.848) &
                (df["ma5_ma20_gap_chg_1d"] <= 1.573),
            "avoid_826_s1.82_n32_bad0.844_p0.062_strong0.062":
                (df["max_drop_7d"] > -3.265) &
                (df["dist_from_low_20d"] <= 8.235) &
                (df["tr_val_rank_20d"] <= 0.3) &
                (df["MACD_hist_3d"] > 31.298) &
                (df["ATR_pct"] <= 6.512),
            "avoid_2646_s1.66_n32_bad0.812_p0.062_strong0.062":
                (df["dist_to_ma5"] <= 1.86) &
                (df["ATR_pct"] <= 4.755) &
                (df["dist_from_low_20d"] > 8.94) &
                (df["ma5_ma20_gap_chg_1d"] <= 0.903) &
                (df["MACD_hist_3d"] > -57.266),
            "avoid_097_s2.08_n37_bad0.865_p0.054_strong0.054":
                (df["tr_val_rank_20d"] <= 0.8) &
                (df["ATR_pct"] <= 3.68) &
                (df["dist_from_low_20d"] > 7.547) &
                (df["ma5_ma20_gap_chg_1d"] <= 1.793) &
                (df["today_pct"] > 3.44),
            "avoid_918_s1.80_n40_bad0.800_p0.075_strong0.025":
                (df["ma5_ma20_gap_chg_1d"] <= -0.505) &
                (df["max_drop_7d"] > -5.317) &
                (df["dist_to_ma5"] > 0.848) &
                (df["ATR_pct"] <= 6.794) &
                (df["tr_val_rank_20d"] <= 0.75),
            "avoid_1915_s1.70_n33_bad0.848_p0.121_strong0.061":
                (df["ma5_ma20_gap_chg_1d"] <= -0.186) &
                (df["dist_from_low_20d"] <= 5.243) &
                (df["MACD_hist_3d"] <= -57.266) &
                (df["max_drop_7d"] > -8.503) &
                (df["today_pct"] > 3.44),
            "avoid_1920_s1.70_n33_bad0.848_p0.121_strong0.061":
                (df["ATR_pct"] <= 7.441) &
                (df["ma5_ma20_gap_chg_1d"] <= -1.087) &
                (df["tr_val_rank_20d"] <= 0.4) &
                (df["today_pct"] <= 5.1) &
                (df["max_drop_7d"] <= -6.095),
            "avoid_3199_s1.63_n41_bad0.829_p0.098_strong0.098":
                (df["ma5_ma20_gap_chg_1d"] <= 0.414) &
                (df["max_drop_7d"] > -3.523) &
                (df["tr_val_rank_20d"] <= 0.4) &
                (df["dist_from_low_20d"] > 6.087) &
                (df["ATR_pct"] <= 6.249),
            "avoid_1885_s1.71_n40_bad0.800_p0.075_strong0.050":
                (df["ATR_pct"] <= 3.68) &
                (df["dist_to_ma5"] > 3.789) &
                (df["MACD_hist_3d"] <= 52.056) &
                (df["today_pct"] <= 7.22) &
                (df["dist_from_low_20d"] <= 13.973),
            "avoid_1437_s1.74_n37_bad0.757_p0.027_strong0.000":
                (df["MACD_hist_3d"] <= 16.282) &
                (df["tr_val_rank_20d"] <= 0.15) &
                (df["ATR_pct"] <= 5.241) &
                (df["ma5_ma20_gap_chg_1d"] > 0.061) &
                (df["max_drop_7d"] <= -3.523),
            "avoid_1580_s1.72_n39_bad0.821_p0.103_strong0.051":
                (df["ma5_ma20_gap_chg_1d"] <= 0.414) &
                (df["ATR_pct"] <= 3.68) &
                (df["today_pct"] <= 4.33) &
                (df["today_tr_val_eok"] > 21.009) &
                (df["dist_to_ma5"] <= 3.003),
            "avoid_3710_s1.62_n30_bad0.833_p0.100_strong0.067":
                (df["dist_to_ma5"] > 1.388) &
                (df["MACD_hist_3d"] <= -10.947) &
                (df["today_tr_val_eok"] <= 10.518) &
                (df["ATR_pct"] > 4.126) &
                (df["today_pct"] > 4.57),
            "avoid_436_s1.89_n42_bad0.810_p0.071_strong0.024":
                (df["ma5_ma20_gap_chg_1d"] <= -0.505) &
                (df["max_drop_7d"] > -7.098) &
                (df["dist_to_ma5"] > 1.388) &
                (df["today_tr_val_eok"] <= 16.555) &
                (df["MACD_hist_3d"] <= 0.666),
            "avoid_4962_s1.58_n32_bad0.812_p0.094_strong0.062":
                (df["ATR_pct"] <= 5.476) &
                (df["max_drop_7d"] <= -9.595) &
                (df["dist_to_ma5"] <= 3.789) &
                (df["MACD_hist_3d"] <= 40.68) &
                (df["today_pct"] <= 6.67),
            "avoid_2231_s1.68_n33_bad0.788_p0.091_strong0.000":
                (df["today_pct"] > 3.94) &
                (df["ATR_pct"] <= 4.126) &
                (df["tr_val_rank_20d"] <= 0.55) &
                (df["dist_from_low_20d"] > 8.235),
            "avoid_776_s1.83_n31_bad0.806_p0.065_strong0.000":
                (df["ATR_pct"] <= 4.126) &
                (df["MACD_hist_3d"] > 52.056) &
                (df["today_tr_val_eok"] <= 10.518) &
                (df["ma5_ma20_gap_chg_1d"] <= 1.048) &
                (df["today_pct"] > 3.6),
            "avoid_4846_s1.58_n37_bad0.757_p0.054_strong0.027":
                (df["dist_to_ma5"] > 1.388) &
                (df["MACD_hist_3d"] <= -10.947) &
                (df["today_pct"] <= 6.2) &
                (df["max_drop_7d"] > -7.692) &
                (df["tr_val_rank_20d"] <= 0.75),
            "avoid_4388_s1.59_n30_bad0.767_p0.067_strong0.000":
                (df["dist_from_low_20d"] <= 5.243) &
                (df["dist_to_ma5"] <= 0.848) &
                (df["ATR_pct"] <= 3.68) &
                (df["today_tr_val_eok"] > 4.727),
            "avoid_4466_s1.59_n30_bad0.767_p0.067_strong0.000":
                (df["max_drop_7d"] > -5.317) &
                (df["tr_val_rank_20d"] <= 0.4) &
                (df["MACD_hist_3d"] > 1019.356) &
                (df["dist_to_ma5"] > 1.388) &
                (df["ATR_pct"] > 4.126),
            "avoid_090_s2.09_n34_bad0.824_p0.029_strong0.000":
                (df["ATR_pct"] <= 5.999) &
                (df["ma5_ma20_gap_chg_1d"] <= -0.505) &
                (df["tr_val_rank_20d"] <= 0.3) &
                (df["MACD_hist_3d"] > -10.947) &
                (df["dist_from_low_20d"] <= 7.547),
            "avoid_645_s1.85_n32_bad0.844_p0.094_strong0.031":
                (df["ma5_ma20_gap_chg_1d"] <= 0.414) &
                (df["max_drop_7d"] > -3.523) &
                (df["tr_val_rank_20d"] <= 0.4) &
                (df["today_pct"] > 4.33) &
                (df["MACD_hist_3d"] > -57.266),
            "avoid_3949_s1.61_n32_bad0.781_p0.062_strong0.031":
                (df["max_drop_7d"] > -4.342) &
                (df["today_tr_val_eok"] <= 3.192) &
                (df["ma5_ma20_gap_chg_1d"] <= 0.744) &
                (df["ATR_pct"] <= 4.991) &
                (df["dist_to_ma5"] <= 1.86),
            "avoid_4671_s1.58_n33_bad0.788_p0.091_strong0.030":
                (df["max_drop_7d"] > -7.098) &
                (df["ma5_ma20_gap_chg_1d"] <= -0.186) &
                (df["tr_val_rank_20d"] <= 0.05) &
                (df["dist_to_ma5"] > -1.554) &
                (df["today_pct"] > 3.94),
            "avoid_4976_s1.58_n37_bad0.784_p0.108_strong0.027":
                (df["ATR_pct"] <= 4.126) &
                (df["max_drop_7d"] <= -5.317) &
                (df["today_pct"] <= 4.81) &
                (df["tr_val_rank_20d"] > 0.55) &
                (df["dist_to_ma5"] <= 4.213),
            "avoid_4999_s1.58_n42_bad0.738_p0.071_strong0.000":
                (df["MACD_hist_3d"] <= 16.282) &
                (df["ATR_pct"] <= 4.126) &
                (df["tr_val_rank_20d"] <= 0.55) &
                (df["dist_from_low_20d"] > 5.243) &
                (df["dist_to_ma5"] > 1.388),
            "avoid_2224_s1.68_n32_bad0.812_p0.094_strong0.031":
                (df["max_drop_7d"] <= -5.317) &
                (df["ATR_pct"] <= 4.126) &
                (df["today_pct"] <= 4.57) &
                (df["today_tr_val_eok"] > 6.386) &
                (df["dist_from_low_20d"] <= 8.235),
            "avoid_939_s1.79_n31_bad0.839_p0.097_strong0.032":
                (df["ATR_pct"] <= 3.68) &
                (df["today_tr_val_eok"] <= 16.555) &
                (df["dist_to_ma5"] > 3.369) &
                (df["ma5_ma20_gap_chg_1d"] <= 1.048) &
                (df["dist_from_low_20d"] <= 11.223),
            "avoid_2266_s1.68_n30_bad0.800_p0.100_strong0.000":
                (df["today_pct"] <= 6.2) &
                (df["MACD_hist_3d"] <= -10.947) &
                (df["dist_to_ma5"] > 1.86) &
                (df["max_drop_7d"] > -11.619),
            "avoid_1237_s1.76_n33_bad0.818_p0.121_strong0.000":
                (df["ATR_pct"] <= 3.68) &
                (df["today_tr_val_eok"] <= 16.555) &
                (df["dist_to_ma5"] > 3.003) &
                (df["ma5_ma20_gap_chg_1d"] <= 0.903) &
                (df["today_pct"] <= 6.67),
    }
    return conditions
