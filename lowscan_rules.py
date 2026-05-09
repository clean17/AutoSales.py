# auto-generated: lowscan good buy rules
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
    "rule_001_s3.56_n124_r0.903",
    "rule_002_s3.55_n104_r0.913",
    "rule_003_s3.54_n83_r0.928",
    "rule_004_s3.53_n103_r0.913",
    "rule_005_s3.52_n82_r0.927",
    "rule_006_s3.52_n82_r0.927",
    "rule_007_s3.52_n82_r0.927",
    "rule_008_s3.52_n82_r0.927",
    "rule_009_s3.52_n82_r0.927",
    "rule_010_s3.52_n82_r0.927",
    "rule_011_s3.51_n102_r0.912",
    "rule_012_s3.51_n102_r0.912",
    "rule_013_s3.51_n102_r0.912",
    "rule_014_s3.51_n102_r0.912",
    "rule_015_s3.51_n95_r0.916",
    "rule_016_s3.50_n88_r0.920",
    "rule_017_s3.50_n81_r0.926",
    "rule_018_s3.50_n81_r0.926",
    "rule_019_s3.50_n81_r0.926",
    "rule_020_s3.50_n81_r0.926",
    "rule_021_s3.50_n101_r0.911",
    "rule_022_s3.50_n101_r0.911",
    "rule_023_s3.50_n101_r0.911",
    "rule_024_s3.50_n101_r0.911",
    "rule_025_s3.49_n107_r0.907",
    "rule_026_s3.49_n107_r0.907",
    "rule_027_s3.49_n107_r0.907",
    "rule_028_s3.49_n94_r0.915",
    "rule_029_s3.49_n94_r0.915",
    "rule_030_s3.48_n113_r0.903",
    "rule_031_s3.48_n80_r0.925",
    "rule_032_s3.48_n80_r0.925",
    "rule_033_s3.48_n80_r0.925",
    "rule_034_s3.48_n100_r0.910",
    "rule_035_s3.48_n100_r0.910",
    "rule_036_s3.48_n100_r0.910",
    "rule_037_s3.48_n100_r0.910",
    "rule_038_s3.48_n100_r0.910",
    "rule_039_s3.48_n100_r0.910",
    "rule_040_s3.48_n100_r0.910",
    "rule_041_s3.48_n100_r0.910",
    "rule_042_s3.48_n100_r0.910",
    "rule_043_s3.47_n106_r0.906",
    "rule_044_s3.47_n106_r0.906",
    "rule_045_s3.47_n106_r0.906",
    "rule_046_s3.47_n106_r0.906",
    "rule_047_s3.47_n106_r0.906",
    "rule_048_s3.47_n93_r0.914",
    "rule_049_s3.47_n93_r0.914",
    "rule_050_s3.46_n86_r0.919",
    "rule_051_s3.46_n99_r0.909",
    "rule_052_s3.46_n99_r0.909",
    "rule_053_s3.46_n99_r0.909",
    "rule_054_s3.45_n105_r0.905",
    "rule_055_s3.45_n105_r0.905",
    "rule_056_s3.45_n111_r0.901",
    "rule_057_s3.45_n111_r0.901",
    "rule_058_s3.45_n111_r0.901",
    "rule_059_s3.45_n92_r0.913",
    "rule_060_s3.45_n92_r0.913",
    "rule_061_s3.45_n92_r0.913",
    "rule_062_s3.44_n85_r0.918",
    "rule_063_s3.44_n85_r0.918",
    "rule_064_s3.44_n85_r0.918",
    "rule_065_s3.44_n85_r0.918",
    "rule_066_s3.44_n98_r0.908",
    "rule_067_s3.44_n98_r0.908",
    "rule_068_s3.44_n98_r0.908",
    "rule_069_s3.44_n98_r0.908",
    "rule_070_s3.44_n104_r0.904",
    "rule_071_s3.44_n104_r0.904",
    "rule_072_s3.44_n104_r0.904",
    "rule_073_s3.44_n104_r0.904",
    "rule_074_s3.44_n104_r0.904",
    "rule_075_s3.44_n104_r0.904",
    "rule_076_s3.44_n104_r0.904",
    "rule_077_s3.44_n104_r0.904",
    "rule_078_s3.44_n104_r0.904",
    "rule_079_s3.44_n104_r0.904",
    "rule_080_s3.44_n104_r0.904",
    "rule_081_s3.44_n104_r0.904",
    "rule_082_s3.44_n104_r0.904",
    "rule_083_s3.44_n104_r0.904",
    "rule_084_s3.44_n104_r0.904",
    "rule_085_s3.44_n104_r0.904",
    "rule_086_s3.44_n104_r0.904",
    "rule_087_s3.44_n104_r0.904",
    "rule_088_s3.44_n104_r0.904",
    "rule_089_s3.44_n104_r0.904",
    "rule_090_s3.44_n104_r0.904",
    "rule_091_s3.44_n104_r0.904",
    "rule_092_s3.44_n104_r0.904",
    "rule_093_s3.44_n104_r0.904",
    "rule_094_s3.44_n104_r0.904",
    "rule_095_s3.44_n104_r0.904",
    "rule_096_s3.44_n104_r0.904",
    "rule_097_s3.44_n104_r0.904",
    "rule_098_s3.44_n104_r0.904",
    "rule_099_s3.44_n104_r0.904",
    "rule_100_s3.43_n110_r0.900",
    "rule_101_s3.43_n110_r0.900",
    "rule_102_s3.43_n110_r0.900",
    "rule_103_s3.43_n110_r0.900",
    "rule_104_s3.43_n91_r0.912",
    "rule_105_s3.43_n91_r0.912",
    "rule_106_s3.43_n91_r0.912",
    "rule_107_s3.43_n91_r0.912",
    "rule_108_s3.42_n97_r0.907",
    "rule_109_s3.42_n97_r0.907",
    "rule_110_s3.42_n97_r0.907",
    "rule_111_s3.42_n97_r0.907",
    "rule_112_s3.42_n97_r0.907",
    "rule_113_s3.42_n97_r0.907",
    "rule_114_s3.42_n97_r0.907",
    "rule_115_s3.42_n97_r0.907",
    "rule_116_s3.42_n84_r0.917",
    "rule_117_s3.42_n84_r0.917",
    "rule_118_s3.42_n84_r0.917",
    "rule_119_s3.42_n84_r0.917",
    "rule_120_s3.42_n84_r0.917",
    "rule_121_s3.42_n103_r0.903",
    "rule_122_s3.42_n103_r0.903",
    "rule_123_s3.42_n103_r0.903",
    "rule_124_s3.42_n103_r0.903",
    "rule_125_s3.42_n103_r0.903",
    "rule_126_s3.42_n103_r0.903",
    "rule_127_s3.42_n103_r0.903",
    "rule_128_s3.41_n90_r0.911",
    "rule_129_s3.41_n90_r0.911",
    "rule_130_s3.41_n90_r0.911",
    "rule_131_s3.41_n90_r0.911",
    "rule_132_s3.41_n90_r0.911",
    "rule_133_s3.40_n96_r0.906",
    "rule_134_s3.40_n96_r0.906",
    "rule_135_s3.40_n96_r0.906",
    "rule_136_s3.40_n96_r0.906",
    "rule_137_s3.40_n83_r0.916",
    "rule_138_s3.40_n83_r0.916",
    "rule_139_s3.40_n83_r0.916",
    "rule_140_s3.40_n83_r0.916",
    "rule_141_s3.40_n83_r0.916",
    "rule_142_s3.40_n83_r0.916",
    "rule_143_s3.40_n83_r0.916",
    "rule_144_s3.40_n83_r0.916",
    "rule_145_s3.40_n83_r0.916",
    "rule_146_s3.40_n83_r0.916",
    "rule_147_s3.40_n83_r0.916",
    "rule_148_s3.40_n83_r0.916",
    "rule_149_s3.40_n83_r0.916",
    "rule_150_s3.40_n83_r0.916",
    "rule_151_s3.40_n83_r0.916",
    "rule_152_s3.40_n102_r0.902",
    "rule_153_s3.40_n102_r0.902",
    "rule_154_s3.40_n102_r0.902",
    "rule_155_s3.40_n102_r0.902",
    "rule_156_s3.40_n102_r0.902",
    "rule_157_s3.40_n102_r0.902",
    "rule_158_s3.40_n102_r0.902",
    "rule_159_s3.40_n102_r0.902",
    "rule_160_s3.40_n102_r0.902",
    "rule_161_s3.40_n102_r0.902",
    "rule_162_s3.40_n102_r0.902",
    "rule_163_s3.39_n89_r0.910",
    "rule_164_s3.39_n89_r0.910",
    "rule_165_s3.39_n89_r0.910",
    "rule_166_s3.39_n89_r0.910",
    "rule_167_s3.39_n89_r0.910",
    "rule_168_s3.39_n89_r0.910",
    "rule_169_s3.39_n89_r0.910",
    "rule_170_s3.39_n89_r0.910",
    "rule_171_s3.39_n89_r0.910",
    "rule_172_s3.39_n89_r0.910",
    "rule_173_s3.39_n89_r0.910",
    "rule_174_s3.39_n95_r0.905",
    "rule_175_s3.39_n95_r0.905",
    "rule_176_s3.39_n95_r0.905",
    "rule_177_s3.39_n95_r0.905",
    "rule_178_s3.39_n95_r0.905",
    "rule_179_s3.39_n95_r0.905",
    "rule_180_s3.39_n95_r0.905",
    "rule_181_s3.38_n101_r0.901",
    "rule_182_s3.38_n101_r0.901",
    "rule_183_s3.38_n101_r0.901",
    "rule_184_s3.38_n101_r0.901",
    "rule_185_s3.38_n101_r0.901",
    "rule_186_s3.38_n101_r0.901",
    "rule_187_s3.38_n101_r0.901",
    "rule_188_s3.38_n101_r0.901",
    "rule_189_s3.38_n82_r0.915",
    "rule_190_s3.38_n82_r0.915",
    "rule_191_s3.38_n82_r0.915",
    "rule_192_s3.38_n82_r0.915",
    "rule_193_s3.38_n82_r0.915",
    "rule_194_s3.38_n82_r0.915",
    "rule_195_s3.38_n82_r0.915",
    "rule_196_s3.38_n82_r0.915",
    "rule_197_s3.38_n82_r0.915",
    "rule_198_s3.38_n82_r0.915",
    "rule_199_s3.38_n82_r0.915",
    "rule_200_s3.37_n88_r0.909",
    "rule_201_s3.37_n88_r0.909",
    "rule_202_s3.37_n88_r0.909",
    "rule_203_s3.37_n88_r0.909",
    "rule_204_s3.37_n88_r0.909",
    "rule_205_s3.37_n94_r0.904",
    "rule_206_s3.37_n94_r0.904",
    "rule_207_s3.37_n94_r0.904",
    "rule_208_s3.37_n94_r0.904",
    "rule_209_s3.37_n94_r0.904",
    "rule_210_s3.36_n100_r0.900",
    "rule_211_s3.36_n100_r0.900",
    "rule_212_s3.36_n100_r0.900",
    "rule_213_s3.36_n100_r0.900",
    "rule_214_s3.36_n100_r0.900",
    "rule_215_s3.36_n100_r0.900",
    "rule_216_s3.36_n100_r0.900",
    "rule_217_s3.36_n100_r0.900",
    "rule_218_s3.36_n100_r0.900",
    "rule_219_s3.36_n100_r0.900",
    "rule_220_s3.36_n100_r0.900",
    "rule_221_s3.36_n100_r0.900",
    "rule_222_s3.36_n100_r0.900",
    "rule_223_s3.36_n100_r0.900",
    "rule_224_s3.36_n100_r0.900",
    "rule_225_s3.36_n81_r0.914",
    "rule_226_s3.36_n81_r0.914",
    "rule_227_s3.36_n81_r0.914",
    "rule_228_s3.36_n81_r0.914",
    "rule_229_s3.36_n81_r0.914",
    "rule_230_s3.36_n81_r0.914",
    "rule_231_s3.36_n81_r0.914",
    "rule_232_s3.36_n81_r0.914",
    "rule_233_s3.35_n87_r0.908",
    "rule_234_s3.35_n87_r0.908",
    "rule_235_s3.35_n87_r0.908",
    "rule_236_s3.35_n87_r0.908",
    "rule_237_s3.35_n87_r0.908",
    "rule_238_s3.35_n87_r0.908",
    "rule_239_s3.35_n93_r0.903",
    "rule_240_s3.35_n93_r0.903",
    "rule_241_s3.35_n93_r0.903",
    "rule_242_s3.35_n93_r0.903",
    "rule_243_s3.35_n93_r0.903",
    "rule_244_s3.35_n93_r0.903",
    "rule_245_s3.35_n93_r0.903",
    "rule_246_s3.35_n93_r0.903",
    "rule_247_s3.35_n93_r0.903",
    "rule_248_s3.34_n80_r0.912",
    "rule_249_s3.34_n80_r0.912",
    "rule_250_s3.34_n80_r0.912",
    "rule_251_s3.34_n80_r0.912",
    "rule_252_s3.34_n80_r0.912",
    "rule_253_s3.34_n80_r0.912",
    "rule_254_s3.34_n80_r0.912",
    "rule_255_s3.34_n80_r0.912",
    "rule_256_s3.34_n80_r0.912",
    "rule_257_s3.34_n80_r0.912",
    "rule_258_s3.34_n80_r0.912",
    "rule_259_s3.34_n80_r0.912",
    "rule_260_s3.34_n80_r0.912",
    "rule_261_s3.34_n80_r0.912",
    "rule_262_s3.34_n80_r0.912",
    "rule_263_s3.34_n80_r0.912",
    "rule_264_s3.34_n80_r0.912",
    "rule_265_s3.34_n80_r0.912",
    "rule_266_s3.34_n80_r0.912",
    "rule_267_s3.34_n80_r0.912",
    "rule_268_s3.34_n80_r0.912",
    "rule_269_s3.34_n80_r0.912",
    "rule_270_s3.34_n80_r0.912",
    "rule_271_s3.34_n80_r0.912",
    "rule_272_s3.34_n80_r0.912",
    "rule_273_s3.34_n80_r0.912",
    "rule_274_s3.34_n80_r0.912",
    "rule_275_s3.34_n80_r0.912",
    "rule_276_s3.34_n80_r0.912",
    "rule_277_s3.34_n80_r0.912",
    "rule_278_s3.34_n80_r0.912",
    "rule_279_s3.34_n80_r0.912",
    "rule_280_s3.34_n80_r0.912",
    "rule_281_s3.34_n80_r0.912",
    "rule_282_s3.34_n80_r0.912",
    "rule_283_s3.34_n80_r0.912",
    "rule_284_s3.34_n80_r0.912",
    "rule_285_s3.34_n80_r0.912",
    "rule_286_s3.34_n80_r0.912",
    "rule_287_s3.34_n80_r0.912",
    "rule_288_s3.34_n80_r0.912",
    "rule_289_s3.33_n86_r0.907",
    "rule_290_s3.33_n86_r0.907",
    "rule_291_s3.33_n86_r0.907",
    "rule_292_s3.33_n86_r0.907",
    "rule_293_s3.33_n86_r0.907",
    "rule_294_s3.33_n92_r0.902",
    "rule_295_s3.33_n92_r0.902",
    "rule_296_s3.33_n92_r0.902",
    "rule_297_s3.33_n92_r0.902",
    "rule_298_s3.33_n92_r0.902",
    "rule_299_s3.33_n92_r0.902",
    "rule_300_s3.33_n92_r0.902",
]

def build_conditions(df):
    conditions = {
            "rule_001_s3.56_n124_r0.903":
                (df["MACD_hist_3d"] <= 154.042) &
                (df["dist_from_low_20d"] > 35.416) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] <= 9.172),
            "rule_002_s3.55_n104_r0.913":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_003_s3.54_n83_r0.928":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697) &
                (df["ATR_pct"] > 9.172),
            "rule_004_s3.53_n103_r0.913":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_005_s3.52_n82_r0.927":
                (df["max_drop_7d"] <= -4.775) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_006_s3.52_n82_r0.927":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.9) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_007_s3.52_n82_r0.927":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_008_s3.52_n82_r0.927":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.447) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_009_s3.52_n82_r0.927":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.346) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_010_s3.52_n82_r0.927":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.622) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_011_s3.51_n102_r0.912":
                (df["ma5_ma20_gap_chg_1d"] > 0.163) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_012_s3.51_n102_r0.912":
                (df["ma5_ma20_gap_chg_1d"] > -0.976) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_013_s3.51_n102_r0.912":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > -0.065) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_014_s3.51_n102_r0.912":
                (df["dist_to_ma5"] > 15.22) &
                (df["ma5_ma20_gap_chg_1d"] > -0.364) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_015_s3.51_n95_r0.916":
                (df["max_drop_7d"] <= -5.106) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_016_s3.50_n88_r0.920":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_017_s3.50_n81_r0.926":
                (df["MACD_hist_3d"] <= 201.561) &
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_018_s3.50_n81_r0.926":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898) &
                (df["ATR_pct"] > 9.172),
            "rule_019_s3.50_n81_r0.926":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.736) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_020_s3.50_n81_r0.926":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 1064.1) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_021_s3.50_n101_r0.911":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_022_s3.50_n101_r0.911":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.353) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_023_s3.50_n101_r0.911":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.687) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_024_s3.50_n101_r0.911":
                (df["ma5_ma20_gap_chg_1d"] > 0.512) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_025_s3.49_n107_r0.907":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.9) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_026_s3.49_n107_r0.907":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.622) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_027_s3.49_n107_r0.907":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.346) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_028_s3.49_n94_r0.915":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_029_s3.49_n94_r0.915":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 369.675) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_030_s3.48_n113_r0.903":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_031_s3.48_n80_r0.925":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["MACD_hist_3d"] <= 120.328),
            "rule_032_s3.48_n80_r0.925":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146) &
                (df["ATR_pct"] > 9.172),
            "rule_033_s3.48_n80_r0.925":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 20.771) &
                (df["ATR_pct"] > 11.749) &
                (df["max_drop_7d"] <= -5.106),
            "rule_034_s3.48_n100_r0.910":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_035_s3.48_n100_r0.910":
                (df["max_drop_7d"] <= -4.775) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_036_s3.48_n100_r0.910":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.447) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_037_s3.48_n100_r0.910":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 23.472) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_038_s3.48_n100_r0.910":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_039_s3.48_n100_r0.910":
                (df["ma5_ma20_gap_chg_1d"] > 1.148) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_040_s3.48_n100_r0.910":
                (df["dist_to_ma5"] > 15.22) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_041_s3.48_n100_r0.910":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_042_s3.48_n100_r0.910":
                (df["ma5_ma20_gap_chg_1d"] > 1.317) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_043_s3.47_n106_r0.906":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 1064.1) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_044_s3.47_n106_r0.906":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 18.87) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_045_s3.47_n106_r0.906":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_046_s3.47_n106_r0.906":
                (df["ma5_ma20_gap_chg_1d"] > 1.148) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_047_s3.47_n106_r0.906":
                (df["dist_to_ma5"] > 15.22) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998) &
                (df["ATR_pct"] > 10.057) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_048_s3.47_n93_r0.914":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_049_s3.47_n93_r0.914":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -5.451) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_050_s3.46_n86_r0.919":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_051_s3.46_n99_r0.909":
                (df["MACD_hist_3d"] > 8.633) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_052_s3.46_n99_r0.909":
                (df["ma5_ma20_gap_chg_1d"] > 1.697) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_053_s3.46_n99_r0.909":
                (df["dist_to_ma5"] > 11.056) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_054_s3.45_n105_r0.905":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_055_s3.45_n105_r0.905":
                (df["ma5_ma20_gap_chg_1d"] > 1.317) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_056_s3.45_n111_r0.901":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_057_s3.45_n111_r0.901":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_058_s3.45_n111_r0.901":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_059_s3.45_n92_r0.913":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_060_s3.45_n92_r0.913":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_061_s3.45_n92_r0.913":
                (df["max_drop_7d"] <= -5.106) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_062_s3.44_n85_r0.918":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 5.548) &
                (df["MACD_hist_3d"] <= 120.328),
            "rule_063_s3.44_n85_r0.918":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_064_s3.44_n85_r0.918":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_065_s3.44_n85_r0.918":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.254) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_066_s3.44_n98_r0.908":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 5.548) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_067_s3.44_n98_r0.908":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 565.471) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_068_s3.44_n98_r0.908":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 23.472) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_069_s3.44_n98_r0.908":
                (df["dist_to_ma5"] > 15.22) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_070_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_071_s3.44_n104_r0.904":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["MACD_hist_3d"] <= 269.338),
            "rule_072_s3.44_n104_r0.904":
                (df["dist_from_low_20d"] > 20.771) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_073_s3.44_n104_r0.904":
                (df["MACD_hist_3d"] > -43.139) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_074_s3.44_n104_r0.904":
                (df["dist_from_low_20d"] > 8.765) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_075_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 13.626) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_076_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 6.495) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_077_s3.44_n104_r0.904":
                (df["dist_from_low_20d"] > 9.471) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_078_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 8.042) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_079_s3.44_n104_r0.904":
                (df["dist_from_low_20d"] > 7.298) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_080_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 10.168) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_081_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 11.848) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_082_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 14.57) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_083_s3.44_n104_r0.904":
                (df["dist_from_low_20d"] > 5.563) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_084_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 15.797) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_085_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 18.87) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_086_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 20.771) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_087_s3.44_n104_r0.904":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 3.47) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_088_s3.44_n104_r0.904":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["max_drop_7d"] <= -3.053) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_089_s3.44_n104_r0.904":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 12.73) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_090_s3.44_n104_r0.904":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["dist_from_low_20d"] > 17.229) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_091_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["dist_from_low_20d"] > 10.962) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_092_s3.44_n104_r0.904":
                (df["dist_from_low_20d"] > 23.472) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_093_s3.44_n104_r0.904":
                (df["MACD_hist_3d"] > -4.55) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_094_s3.44_n104_r0.904":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["max_drop_7d"] <= -2.787) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_095_s3.44_n104_r0.904":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["max_drop_7d"] <= -3.622) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_096_s3.44_n104_r0.904":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["max_drop_7d"] <= -3.346) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_097_s3.44_n104_r0.904":
                (df["dist_from_low_20d"] > 27.586) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_098_s3.44_n104_r0.904":
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 3.65) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_099_s3.44_n104_r0.904":
                (df["ma5_ma20_gap_chg_1d"] > 1.697) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_100_s3.43_n110_r0.900":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_101_s3.43_n110_r0.900":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["MACD_hist_3d"] > 8.633),
            "rule_102_s3.43_n110_r0.900":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_103_s3.43_n110_r0.900":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_104_s3.43_n91_r0.912":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["ATR_pct"] > 9.172),
            "rule_105_s3.43_n91_r0.912":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_106_s3.43_n91_r0.912":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_107_s3.43_n91_r0.912":
                (df["MACD_hist_3d"] > 20.21) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_108_s3.42_n97_r0.907":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_109_s3.42_n97_r0.907":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.147) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_110_s3.42_n97_r0.907":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.913) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_111_s3.42_n97_r0.907":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] > 3.32) &
                (df["ATR_pct"] > 9.172),
            "rule_112_s3.42_n97_r0.907":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497),
            "rule_113_s3.42_n97_r0.907":
                (df["max_drop_7d"] <= -4.775) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_114_s3.42_n97_r0.907":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.447) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_115_s3.42_n97_r0.907":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_116_s3.42_n84_r0.917":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.147) &
                (df["MACD_hist_3d"] <= 120.328),
            "rule_117_s3.42_n84_r0.917":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.913) &
                (df["MACD_hist_3d"] <= 120.328),
            "rule_118_s3.42_n84_r0.917":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 201.561) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_119_s3.42_n84_r0.917":
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -7.23),
            "rule_120_s3.42_n84_r0.917":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381) &
                (df["max_drop_7d"] <= -6.736),
            "rule_121_s3.42_n103_r0.903":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.9) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_122_s3.42_n103_r0.903":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.622) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_123_s3.42_n103_r0.903":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.346) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_124_s3.42_n103_r0.903":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["MACD_hist_3d"] > 3.32) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_125_s3.42_n103_r0.903":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["MACD_hist_3d"] <= 1064.1) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_126_s3.42_n103_r0.903":
                (df["dist_to_ma5"] > 15.22) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898) &
                (df["ATR_pct"] > 10.057) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_127_s3.42_n103_r0.903":
                (df["dist_to_ma5"] > 9.077) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_128_s3.41_n90_r0.911":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 5.548) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_129_s3.41_n90_r0.911":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 369.675) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_130_s3.41_n90_r0.911":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -5.451) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_131_s3.41_n90_r0.911":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_132_s3.41_n90_r0.911":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 201.561) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_133_s3.40_n96_r0.906":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_134_s3.40_n96_r0.906":
                (df["max_drop_7d"] <= -8.644) &
                (df["dist_to_ma5"] > 15.22) &
                (df["MACD_hist_3d"] <= 565.471) &
                (df["ma5_ma20_gap_chg_1d"] > 4.41),
            "rule_135_s3.40_n96_r0.906":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_136_s3.40_n96_r0.906":
                (df["dist_to_ma5"] > 11.056) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_137_s3.40_n83_r0.916":
                (df["ATR_pct"] > 11.749) &
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_138_s3.40_n83_r0.916":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_139_s3.40_n83_r0.916":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_140_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.254) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_141_s3.40_n83_r0.916":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998) &
                (df["max_drop_7d"] <= -7.23),
            "rule_142_s3.40_n83_r0.916":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148) &
                (df["max_drop_7d"] <= -7.23),
            "rule_143_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] <= 1.0) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_144_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["max_drop_7d"] <= -3.053) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_145_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -2.787) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_146_s3.40_n83_r0.916":
                (df["max_drop_7d"] <= -4.775) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 18.87),
            "rule_147_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.9) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 18.87),
            "rule_148_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 18.87),
            "rule_149_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.447) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 18.87),
            "rule_150_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.346) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 18.87),
            "rule_151_s3.40_n83_r0.916":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.622) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 18.87),
            "rule_152_s3.40_n102_r0.902":
                (df["MACD_hist_3d"] <= 120.328) &
                (df["dist_from_low_20d"] > 35.416) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] <= 9.172),
            "rule_153_s3.40_n102_r0.902":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 1064.1) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_154_s3.40_n102_r0.902":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 18.87) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_155_s3.40_n102_r0.902":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_156_s3.40_n102_r0.902":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_157_s3.40_n102_r0.902":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_158_s3.40_n102_r0.902":
                (df["max_drop_7d"] <= -5.106) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_159_s3.40_n102_r0.902":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["max_drop_7d"] <= -3.9) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_160_s3.40_n102_r0.902":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 4.02) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_161_s3.40_n102_r0.902":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 3.83) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_162_s3.40_n102_r0.902":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_163_s3.39_n89_r0.910":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.147) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_164_s3.39_n89_r0.910":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.913) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_165_s3.39_n89_r0.910":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_166_s3.39_n89_r0.910":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998) &
                (df["ATR_pct"] > 9.172),
            "rule_167_s3.39_n89_r0.910":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148) &
                (df["ATR_pct"] > 9.172),
            "rule_168_s3.39_n89_r0.910":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_169_s3.39_n89_r0.910":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317) &
                (df["ATR_pct"] > 9.172),
            "rule_170_s3.39_n89_r0.910":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_171_s3.39_n89_r0.910":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] > 8.633) &
                (df["ATR_pct"] > 9.172),
            "rule_172_s3.39_n89_r0.910":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 201.561) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_173_s3.39_n89_r0.910":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 201.561) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_174_s3.39_n95_r0.905":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 7.844) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_175_s3.39_n95_r0.905":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 23.472) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_176_s3.39_n95_r0.905":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_177_s3.39_n95_r0.905":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_178_s3.39_n95_r0.905":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_179_s3.39_n95_r0.905":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 27.586) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697),
            "rule_180_s3.39_n95_r0.905":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.443) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_181_s3.38_n101_r0.901":
                (df["dist_from_low_20d"] > 20.771) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_182_s3.38_n101_r0.901":
                (df["max_drop_7d"] <= -5.106) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_183_s3.38_n101_r0.901":
                (df["max_drop_7d"] <= -5.106) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_184_s3.38_n101_r0.901":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 369.675) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_185_s3.38_n101_r0.901":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 4.24) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_186_s3.38_n101_r0.901":
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 4.71) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_187_s3.38_n101_r0.901":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 4.47) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_188_s3.38_n101_r0.901":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 5.0) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_189_s3.38_n82_r0.915":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 7.844) &
                (df["MACD_hist_3d"] <= 120.328),
            "rule_190_s3.38_n82_r0.915":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317) &
                (df["max_drop_7d"] <= -7.23),
            "rule_191_s3.38_n82_r0.915":
                (df["ATR_pct"] > 11.749) &
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] > 0.15) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_192_s3.38_n82_r0.915":
                (df["today_tr_val_eok"] > 5.297) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_193_s3.38_n82_r0.915":
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] > 0.25) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_194_s3.38_n82_r0.915":
                (df["ATR_pct"] > 11.749) &
                (df["today_pct"] > 17.104) &
                (df["today_tr_val_eok"] > 3.472) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_195_s3.38_n82_r0.915":
                (df["ATR_pct"] > 11.749) &
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] > 0.2) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_196_s3.38_n82_r0.915":
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] > 0.05) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_197_s3.38_n82_r0.915":
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] > 0.3) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_198_s3.38_n82_r0.915":
                (df["ATR_pct"] > 11.749) &
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] > 0.1) &
                (df["dist_from_low_20d"] > 20.771),
            "rule_199_s3.38_n82_r0.915":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] <= 1064.1),
            "rule_200_s3.37_n88_r0.909":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_201_s3.37_n88_r0.909":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497) &
                (df["ATR_pct"] > 9.172),
            "rule_202_s3.37_n88_r0.909":
                (df["max_drop_7d"] <= -5.106) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_203_s3.37_n88_r0.909":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 201.561) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_204_s3.37_n88_r0.909":
                (df["ma5_ma20_gap_chg_1d"] > 0.845) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.736),
            "rule_205_s3.37_n94_r0.904":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 565.471) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_206_s3.37_n94_r0.904":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_207_s3.37_n94_r0.904":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_208_s3.37_n94_r0.904":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_209_s3.37_n94_r0.904":
                (df["MACD_hist_3d"] > 13.75) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_210_s3.36_n100_r0.900":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.163) &
                (df["MACD_hist_3d"] <= 201.561),
            "rule_211_s3.36_n100_r0.900":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["max_drop_7d"] <= -8.644) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_212_s3.36_n100_r0.900":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_213_s3.36_n100_r0.900":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_214_s3.36_n100_r0.900":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_215_s3.36_n100_r0.900":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.877) &
                (df["MACD_hist_3d"] > 8.633),
            "rule_216_s3.36_n100_r0.900":
                (df["max_drop_7d"] <= -5.106) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_217_s3.36_n100_r0.900":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 369.675) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_218_s3.36_n100_r0.900":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 369.675) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_219_s3.36_n100_r0.900":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -5.451) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
            "rule_220_s3.36_n100_r0.900":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697) &
                (df["MACD_hist_3d"] <= 369.675),
            "rule_221_s3.36_n100_r0.900":
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 5.32) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_222_s3.36_n100_r0.900":
                (df["dist_to_ma5"] > 15.22) &
                (df["today_pct"] > 5.66) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_223_s3.36_n100_r0.900":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.443) &
                (df["MACD_hist_3d"] <= 565.471),
            "rule_224_s3.36_n100_r0.900":
                (df["dist_to_ma5"] > 9.077) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_225_s3.36_n81_r0.914":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 5.548) &
                (df["MACD_hist_3d"] <= 94.182),
            "rule_226_s3.36_n81_r0.914":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_227_s3.36_n81_r0.914":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_228_s3.36_n81_r0.914":
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -7.23),
            "rule_229_s3.36_n81_r0.914":
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381) &
                (df["max_drop_7d"] <= -7.23),
            "rule_230_s3.36_n81_r0.914":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["max_drop_7d"] <= -5.106),
            "rule_231_s3.36_n81_r0.914":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.35),
            "rule_232_s3.36_n81_r0.914":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 20.771) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.35),
            "rule_233_s3.35_n87_r0.908":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.163) &
                (df["MACD_hist_3d"] <= 120.328),
            "rule_234_s3.35_n87_r0.908":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 7.844) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_235_s3.35_n87_r0.908":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 369.675) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_236_s3.35_n87_r0.908":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 201.561) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497),
            "rule_237_s3.35_n87_r0.908":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.736) &
                (df["ma5_ma20_gap_chg_1d"] > 0.998),
            "rule_238_s3.35_n87_r0.908":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.736) &
                (df["ma5_ma20_gap_chg_1d"] > 1.148),
            "rule_239_s3.35_n93_r0.903":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 9.172) &
                (df["max_drop_7d"] <= -7.23) &
                (df["ma5_ma20_gap_chg_1d"] > 2.443),
            "rule_240_s3.35_n93_r0.903":
                (df["max_drop_7d"] <= -4.775) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_241_s3.35_n93_r0.903":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.447) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_242_s3.35_n93_r0.903":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 1.697) &
                (df["ATR_pct"] > 8.51),
            "rule_243_s3.35_n93_r0.903":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497),
            "rule_244_s3.35_n93_r0.903":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_245_s3.35_n93_r0.903":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 27.586) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_246_s3.35_n93_r0.903":
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ma5_ma20_gap_chg_1d"] > 2.826) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_247_s3.35_n93_r0.903":
                (df["dist_to_ma5"] > 11.056) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_248_s3.34_n80_r0.912":
                (df["ATR_pct"] > 11.749) &
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_249_s3.34_n80_r0.912":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.147) &
                (df["MACD_hist_3d"] <= 94.182),
            "rule_250_s3.34_n80_r0.912":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["dist_to_ma5"] > 6.913) &
                (df["MACD_hist_3d"] <= 94.182),
            "rule_251_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.9) &
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22),
            "rule_252_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.622) &
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22),
            "rule_253_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.346) &
                (df["ATR_pct"] > 10.057) &
                (df["dist_to_ma5"] > 15.22),
            "rule_254_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 201.561) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_255_s3.34_n80_r0.912":
                (df["dist_to_ma5"] > 15.22) &
                (df["MACD_hist_3d"] <= 565.471) &
                (df["max_drop_7d"] <= -9.791) &
                (df["ma5_ma20_gap_chg_1d"] > 4.41),
            "rule_256_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 10.962) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_257_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["tr_val_rank_20d"] <= 1.0) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_258_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 10.168) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_259_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 8.765) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_260_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 13.626) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_261_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["max_drop_7d"] <= -3.053) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_262_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -2.787) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_263_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 17.229) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_264_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 14.57) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_265_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 9.471) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_266_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 7.298) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_267_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 6.495) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_268_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["dist_from_low_20d"] > 12.73) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_269_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 15.797) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_270_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 8.042) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_271_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 11.848) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_272_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 5.563) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_273_s3.34_n80_r0.912":
                (df["max_drop_7d"] <= -4.775) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_274_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.9) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_275_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_276_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.447) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_277_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.346) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_278_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -3.622) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_279_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["max_drop_7d"] <= -5.84),
            "rule_280_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["max_drop_7d"] <= -5.451),
            "rule_281_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.4),
            "rule_282_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.45),
            "rule_283_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_284_s3.34_n80_r0.912":
                (df["dist_from_low_20d"] > 18.87) &
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_285_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 20.771) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.4),
            "rule_286_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 20.771) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.45),
            "rule_287_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 20.771) &
                (df["ATR_pct"] > 11.749) &
                (df["tr_val_rank_20d"] > 0.5),
            "rule_288_s3.34_n80_r0.912":
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 20.771) &
                (df["ATR_pct"] > 11.749) &
                (df["MACD_hist_3d"] > -43.139),
            "rule_289_s3.33_n86_r0.907":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.353) &
                (df["MACD_hist_3d"] <= 120.328),
            "rule_290_s3.33_n86_r0.907":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -5.451) &
                (df["ma5_ma20_gap_chg_1d"] > 2.146),
            "rule_291_s3.33_n86_r0.907":
                (df["today_pct"] > 17.104) &
                (df["MACD_hist_3d"] <= 269.338) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 1.898),
            "rule_292_s3.33_n86_r0.907":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.736) &
                (df["ma5_ma20_gap_chg_1d"] > 1.317),
            "rule_293_s3.33_n86_r0.907":
                (df["MACD_hist_3d"] > 27.937) &
                (df["dist_to_ma5"] > 15.22) &
                (df["ATR_pct"] > 10.057) &
                (df["today_tr_val_eok"] <= 905.381),
            "rule_294_s3.33_n92_r0.902":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["today_tr_val_eok"] <= 905.381) &
                (df["dist_to_ma5"] > 15.22),
            "rule_295_s3.33_n92_r0.902":
                (df["max_drop_7d"] <= -8.644) &
                (df["today_pct"] > 17.104) &
                (df["ma5_ma20_gap_chg_1d"] > 0.163) &
                (df["MACD_hist_3d"] <= 154.042),
            "rule_296_s3.33_n92_r0.902":
                (df["today_pct"] > 17.104) &
                (df["max_drop_7d"] <= -4.153) &
                (df["ATR_pct"] > 10.057) &
                (df["ma5_ma20_gap_chg_1d"] > 2.443),
            "rule_297_s3.33_n92_r0.902":
                (df["ATR_pct"] > 10.057) &
                (df["today_pct"] > 17.104) &
                (df["dist_from_low_20d"] > 23.472) &
                (df["ma5_ma20_gap_chg_1d"] > 2.443),
            "rule_298_s3.33_n92_r0.902":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["MACD_hist_3d"] > 3.32) &
                (df["max_drop_7d"] <= -7.23),
            "rule_299_s3.33_n92_r0.902":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -5.84) &
                (df["ma5_ma20_gap_chg_1d"] > 1.497),
            "rule_300_s3.33_n92_r0.902":
                (df["today_pct"] > 17.104) &
                (df["ATR_pct"] > 10.057) &
                (df["max_drop_7d"] <= -6.254) &
                (df["ma5_ma20_gap_chg_1d"] > 0.845),
    }
    return conditions
