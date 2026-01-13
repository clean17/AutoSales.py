from utils import sort_csv_by_today_desc
import pandas as pd

# df = pd.read_csv("csv/low_result_us_desc.csv")
df = pd.read_csv("csv/low_result_250409_desc.csv")

# 각 조건 정의
conditions = {
    ###################### 7일 #####################

    # count=42, Ratio(>=7)=100.00%
    "rule_01":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 6.12) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["pct_vs_last2week"] <= -9.15) &
        (df["pos20_ratio"] > 27.5),

    # count=41, Ratio(>=7)=100.00%
    "rule_02":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 6.12) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["pct_vs_last2week"] <= -7.845) &
        (df["ma20_chg_rate"] <= -0.79) &
        (df["mean_ret20"] > -1.025),

    # count=41, Ratio(>=7)=100.00%
    "rule_03":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 6.12) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["pct_vs_last2week"] <= -9.15) &
        (df["pos20_ratio"] > 30.5),

    # count=40, Ratio(>=7)=100.00%
    "rule_04":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 4.88) &
        (df["vol20"] > 3.33) &
        (df["mean_ret30"] <= -0.645) &
        (df["pct_vs_last4week"] <= -18.425),

    # count=39, Ratio(>=7)=100.00%
    "rule_05":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["vol30"] <= 6.12) &
        (df["pct_vs_last2week"] <= -9.15) &
        (df["pos20_ratio"] > 27.5) &
        (df["mean_ret30"] <= -0.645),

    # count=39, Ratio(>=7)=100.00%
    "rule_06":
        (df["pct_vs_last4week"] <= -15.345) &
        (df["vol20"] <= 6.945) &
        (df["ma5_chg_rate"] <= -0.595) &
        (df["three_m_chg_rate"] > 41.95) &
        (df["three_m_chg_rate"] <= 128.28) &
        (df["pct_vs_lastweek"] <= -3.96) &
        (df["today_chg_rate"] > -43.15) &
        (df["mean_ret30"] <= -0.655) &
        (df["pct_vs_last4week"] <= -18.425) &
        (df["chg_tr_val"] > -40.45),

    # count=38, Ratio(>=7)=100.00%
    "rule_07":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["vol30"] <= 4.88) &
        (df["pct_vs_last2week"] <= -9.15) &
        (df["pos20_ratio"] > 27.5),

    # count=38, Ratio(>=7)=100.00%
    "rule_08":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["mean_ret30"] <= -0.645) &
        (df["vol30"] <= 4.88) &
        (df["vol20"] > 3.33),

    # count=37, Ratio(>=7)=100.00%
    "rule_09":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 6.12) &
        (df["three_m_chg_rate"] > 55.26) &
        (df["pct_vs_last2week"] <= -9.15),

    # count=37, Ratio(>=7)=100.00%
    "rule_10":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 6.12) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["pct_vs_last2week"] <= -10.18),

    # count=36, Ratio(>=7)=100.00%
    "rule_11":
        (df["pct_vs_last4week"] <= -15.345) &
        (df["vol20"] <= 6.945) &
        (df["ma5_chg_rate"] <= -0.595) &
        (df["three_m_chg_rate"] > 41.95) &
        (df["pct_vs_lastweek"] <= -3.96) &
        (df["today_chg_rate"] > -43.15) &
        (df["mean_ret30"] <= -0.655) &
        (df["pct_vs_last4week"] <= -18.425) &
        (df["chg_tr_val"] > -40.45),

    # count=36, Ratio(>=7)=100.00%
    "rule_12":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_last4week"] <= -18.425) &
        (df["mean_ret30"] <= -0.645) &
        (df["vol30"] <= 4.88) &
        (df["vol20"] > 3.33),

    # count=35, Ratio(>=7)=100.00%
    "rule_14":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 4.88) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["pct_vs_last2week"] <= -9.15),

    # count=35, Ratio(>=7)=100.00%
    "rule_15":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["mean_ret30"] <= -0.645) &
        (df["vol30"] <= 6.12) &
        (df["pct_vs_last2week"] <= -9.15),

    # count=34, Ratio(>=7)=100.00%
    "rule_16":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["three_m_chg_rate"] > 49.56) &
        (df["vol30"] <= 4.88) &
        (df["pct_vs_last2week"] <= -9.15) &
        (df["pos20_ratio"] > 27.5),

    # count=34, Ratio(>=7)=100.00%
    "rule_17":
        (df["pct_vs_last4week"] <= -15.345) &
        (df["vol20"] <= 6.945) &
        (df["ma5_chg_rate"] <= -0.595) &
        (df["three_m_chg_rate"] > 41.95) &
        (df["pct_vs_lastweek"] <= -3.96) &
        (df["today_chg_rate"] > -43.15) &
        (df["mean_ret30"] <= -0.655) &
        (df["pct_vs_last4week"] <= -18.425),

    # count=33, Ratio(>=7)=100.00%
    "rule_18":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -14.38) &
        (df["vol30"] <= 6.12) &
        (df["three_m_chg_rate"] > 60.06) &
        (df["pct_vs_last2week"] <= -9.15),

    # count=33, Ratio(>=7)=100.00%
    "rule_19":
        (df["pct_vs_lastweek"] <= -3.965) &
        (df["today_chg_rate"] <= -18.495) &
        (df["pct_vs_firstweek"] > -34.9) &
        (df["pct_vs_last4week"] <= -18.425) &
        (df["mean_ret30"] <= -0.645) &
        (df["vol30"] <= 6.12),

    # count=33, Ratio(>=7)=100.00%
    "rule_20":
        (df["pct_vs_last4week"] <= -15.345) &
        (df["vol20"] <= 6.945) &
        (df["ma5_chg_rate"] <= -0.595) &
        (df["three_m_chg_rate"] > 41.95) &
        (df["pct_vs_lastweek"] <= -3.96) &
        (df["today_chg_rate"] > -43.15) &
        (df["mean_ret30"] <= -0.655) &
        (df["chg_tr_val"] > -40.45),

    "rule_001__n31__r0.903":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 5.6),
    "rule_002__n30__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.76) &
        (df["today_chg_rate"] >= -40.344),
    "rule_003__n30__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 5.9),
    "rule_004__n25__r0.920":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.1) &
        (df["mean_ret30"] <= -0.67),
    "rule_005__n25__r0.920":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["vol30"] <= 3.96) &
        (df["today_pct"] >= 5.1),
    "rule_006__n25__r0.920":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 6.2),
    "rule_007__n25__r0.920":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.6) &
        (df["today_chg_rate"] >= -40.344),

    "rule_010__n24__r0.917":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["mean_ret30"] <= -0.77) &
        (df["three_m_chg_rate"] <= 64.993) &
        (df["pct_vs_last4week"] <= -17.377),
    "rule_011__n24__r0.917":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.6) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["pct_vs_firstweek"] <= -14.414),
    "rule_012__n24__r0.917":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 70.766) &
        (df["mean_prev3"] <= 6149198580.0),
    "rule_013__n24__r0.917":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 70.766) &
        (df["today_tr_val"] <= 33576497460.0),
    "rule_014__n24__r0.917":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 70.766) &
        (df["today_tr_val"] <= 53173286208.0),
    "rule_015__n24__r0.917":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.76) &
        (df["pct_vs_firstweek"] <= -17.32) &
        (df["today_chg_rate"] >= -40.344),
    "rule_016__n24__r0.917":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 959522214.35) &
        (df["pct_vs_lastweek"] <= -1.257),
    "rule_017__n23__r0.913":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["today_chg_rate"] >= -40.344),
    "rule_018__n23__r0.913":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.3) &
        (df["mean_ret30"] <= -0.57),

    "rule_020__n23__r0.913":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["pct_vs_lastweek"] <= -1.257),
    "rule_021__n23__r0.913":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["vol30"] <= 3.96) &
        (df["today_pct"] >= 5.3),
    "rule_022__n23__r0.913":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -15.276) &
        (df["pct_vs_firstweek"] >= -17.32) &
        (df["three_m_chg_rate"] >= 56.451),
    "rule_023__n23__r0.913":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= 0.454) &
        (df["pct_vs_firstweek"] <= 30.465),
    "rule_024__n23__r0.913":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= 0.454) &
        (df["pct_vs_firstweek"] <= 42.542),
    "rule_025__n23__r0.913":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["vol20"] <= 6.566),
    "rule_026__n23__r0.913":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["pct_vs_last4week"] <= -14.88) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.1),
    "rule_027__n23__r0.913":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["ma20_chg_rate"] >= -1.08) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["ma5_chg_rate"] <= -0.03),
    "rule_028__n22__r0.955":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.44) &
        (df["today_chg_rate"] >= -40.344),
    "rule_029__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["three_m_chg_rate"] >= 35.97),
    "rule_030__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["three_m_chg_rate"] <= 78.15),
    "rule_031__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["pct_vs_firstweek"] >= -31.118),
    "rule_032__n22__r0.909":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_ret20"] <= -0.81) &
        (df["pos20_ratio"] <= 40.0),
    "rule_033__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["ma5_chg_rate"] <= 0.19),
    "rule_034__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["ma5_chg_rate"] <= 0.42),
    "rule_035__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["ma5_chg_rate"] <= 0.62),
    "rule_036__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["ma5_chg_rate"] <= 0.83),
    "rule_037__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["ma5_chg_rate"] <= 1.03),
    "rule_038__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["mean_ret30"] <= -0.57),
    "rule_039__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["mean_ret30"] <= -0.48),
    "rule_040__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["chg_tr_val"] <= 19.5),
    "rule_041__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_lastweek"] <= 3.23),
    "rule_042__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_lastweek"] <= 4.137),
    "rule_043__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_lastweek"] <= 4.55),
    "rule_044__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_lastweek"] <= 5.11),
    "rule_045__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_lastweek"] <= 5.66),

    "rule_047__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["today_pct"] >= 4.7),
    "rule_048__n22__r0.909":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["vol30"] <= 3.96) &
        (df["today_pct"] >= 5.6),
    "rule_049__n22__r0.909":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["pct_vs_firstweek"] >= -31.118) &
        (df["vol30"] <= 6.05),
    "rule_050__n22__r0.909":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["pct_vs_firstweek"] >= -31.118) &
        (df["mean_ret30"] <= -0.77),
    "rule_051__n22__r0.909":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.44) &
        (df["today_pct"] >= 5.1),
    "rule_052__n22__r0.909":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.44) &
        (df["today_pct"] >= 5.1) &
        (df["vol20"] >= 2.84),
    "rule_053__n22__r0.909":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 70.766) &
        (df["mean_prev3"] <= 4846759743.98),
    "rule_054__n22__r0.909":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 70.766) &
        (df["today_tr_val"] <= 23522210796.0),
    "rule_055__n22__r0.909":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= 0.454) &
        (df["pct_vs_firstweek"] <= 22.562),
    "rule_056__n22__r0.909":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["vol30"] <= 6.05),
    "rule_057__n22__r0.909":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["pct_vs_lastweek"] <= 1.915),
    "rule_058__n22__r0.909":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["pct_vs_lastweek"] <= 3.23),
    "rule_059__n22__r0.909":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["pct_vs_lastweek"] <= 4.137),
    "rule_060__n22__r0.909":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.76) &
        (df["pct_vs_firstweek"] <= -17.32) &
        (df["three_m_chg_rate"] <= 78.15),
    "rule_061__n22__r0.909":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 5.3),
    "rule_062__n22__r0.909":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["vol20"] <= 5.166) &
        (df["mean_prev3"] <= 6149198580.0),
    "rule_063__n22__r0.909":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["vol20"] <= 5.166) &
        (df["today_tr_val"] <= 6399938903.0),
    "rule_064__n22__r0.909":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["vol20"] <= 5.166) &
        (df["today_tr_val"] <= 8047378016.0),
    "rule_065__n21__r0.952":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.3) &
        (df["mean_ret30"] <= -0.67),
    "rule_066__n21__r0.952":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["ma5_chg_rate"] <= -0.03),
    "rule_067__n21__r0.952":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["vol20"] <= 5.607),
    "rule_068__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["three_m_chg_rate"] >= 39.05),
    "rule_069__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["today_chg_rate"] <= -22.38),
    "rule_070__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["today_chg_rate"] <= -20.75),
    "rule_071__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["ma5_chg_rate"] >= -1.499),
    "rule_072__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["mean_prev3"] <= 1520290369.3),
    "rule_073__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["mean_prev3"] <= 1783551861.23),
    "rule_074__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.9) &
        (df["three_m_chg_rate"] >= 35.97),
    "rule_075__n21__r0.905":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.44) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["chg_tr_val"] >= -45.39),
    "rule_076__n21__r0.905":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.44) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["pct_vs_firstweek"] <= -14.414),
    "rule_077__n21__r0.905":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.44) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["pct_vs_firstweek"] <= -11.369),
    "rule_078__n21__r0.905":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["pct_vs_firstweek"] <= 30.465),
    "rule_079__n21__r0.905":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["pct_vs_firstweek"] <= 42.542),
    "rule_080__n21__r0.905":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_ret20"] <= -0.81) &
        (df["today_pct"] <= 8.6),
    "rule_081__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.3) &
        (df["vol20"] >= 2.84),
    "rule_082__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.3) &
        (df["pct_vs_firstweek"] <= -11.369),
    "rule_083__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.3) &
        (df["pct_vs_firstweek"] <= -8.504),
    "rule_084__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_tr_val"] <= 1393261612.0) &
        (df["ma5_chg_rate"] <= -0.03),
    "rule_085__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["mean_ret30"] <= -0.67),
    "rule_086__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["chg_tr_val"] <= 9.5),
    "rule_087__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["three_m_chg_rate"] >= 42.201),
    "rule_088__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["three_m_chg_rate"] >= 45.726),
    "rule_089__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["today_chg_rate"] <= -24.172),
    "rule_090__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_firstweek"] <= -11.369),
    "rule_091__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_lastweek"] <= 0.454),
    "rule_092__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 853190505.3) &
        (df["pct_vs_lastweek"] <= 1.915),

    "rule_094__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["pct_vs_firstweek"] <= -14.414),
    "rule_095__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["today_pct"] >= 4.85),
    "rule_096__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["vol30"] <= 3.96) &
        (df["today_pct"] >= 5.9),
    "rule_097__n21__r0.905":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["mean_ret30"] <= -0.77) &
        (df["three_m_chg_rate"] <= 64.993) &
        (df["mean_ret20"] <= -0.95),
    "rule_098__n21__r0.905":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.44) &
        (df["today_pct"] >= 5.3),
    "rule_099__n21__r0.905":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_prev3"] <= 3049111908.5) &
        (df["today_pct"] <= 10.67),
    "rule_100__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["ma5_chg_rate"] <= -0.908) &
        (df["vol20"] >= 2.5),
    "rule_101__n21__r0.905":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["mean_ret20"] >= -1.42) &
        (df["vol30"] <= 6.05),

    "rule_104__n21__r0.905":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -15.276) &
        (df["pct_vs_firstweek"] >= -17.32) &
        (df["three_m_chg_rate"] >= 61.086),
    "rule_105__n21__r0.905":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.6) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["pct_vs_firstweek"] <= -17.32),
    "rule_106__n21__r0.905":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 78.15) &
        (df["mean_prev3"] <= 8442524204.02),
    "rule_107__n21__r0.905":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 78.15) &
        (df["mean_prev3"] <= 12308939514.31),
    "rule_108__n21__r0.905":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 78.15) &
        (df["today_tr_val"] <= 33576497460.0),
    "rule_109__n21__r0.905":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 78.15) &
        (df["today_tr_val"] <= 53173286208.0),
    "rule_110__n21__r0.905":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 1520290369.3),
    "rule_111__n21__r0.905":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 70.766) &
        (df["mean_prev3"] <= 3843839864.0),
    "rule_112__n21__r0.905":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.6) &
        (df["three_m_chg_rate"] <= 87.674),
    "rule_113__n21__r0.905":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.6) &
        (df["today_pct"] >= 5.1),
    "rule_114__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.6) &
        (df["today_tr_val"] <= 1768327900.0) &
        (df["pct_vs_firstweek"] <= -20.896),
    "rule_115__n21__r0.905":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["today_tr_val"] >= 552978057.0),
    "rule_116__n21__r0.905":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["pct_vs_lastweek"] <= 0.454),
    "rule_117__n21__r0.905":
        (df["ma5_chg_rate"] <= -0.28) &
        (df["today_pct"] >= 9.4) &
        (df["mean_prev3"] <= 3049111908.5) &
        (df["today_tr_val"] >= 1113627808.1),
    "rule_118__n21__r0.905":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.76) &
        (df["three_m_chg_rate"] <= 78.15),
    "rule_119__n21__r0.905":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["pct_vs_last4week"] <= -14.88) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.3),
    "rule_120__n21__r0.905":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["vol20"] <= 5.166) &
        (df["mean_prev3"] <= 4846759743.98),
    "rule_121__n21__r0.905":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["vol20"] <= 5.166) &
        (df["today_tr_val"] <= 5070318318.0),
    "rule_122__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 959522214.35) &
        (df["chg_tr_val"] <= -2.76),
    "rule_123__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 959522214.35) &
        (df["pct_vs_lastweek"] <= -2.88),
    "rule_124__n21__r0.905":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 959522214.35) &
        (df["today_pct"] >= 5.3),
    "rule_125__n20__r0.950":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["mean_ret30"] <= -0.57),
    "rule_126__n20__r0.950":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["today_pct"] >= 5.1),
    "rule_127__n20__r0.950":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 78.15) &
        (df["mean_prev3"] <= 6149198580.0),
    "rule_128__n20__r0.950":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.6) &
        (df["pct_vs_firstweek"] <= -17.32) &
        (df["today_chg_rate"] >= -40.344),
    "rule_129__n20__r0.950":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["vol30"] <= 6.05) &
        (df["pos20_ratio"] >= 30.0),
    "rule_130__n20__r0.950":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 5.6),
    "rule_131__n20__r0.950":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 5.9),
    "rule_132__n20__r0.950":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 959522214.35) &
        (df["today_pct"] >= 5.6),
    "rule_133__n20__r0.950":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 959522214.35) &
        (df["today_pct"] >= 5.9),
    "rule_134__n20__r0.900":
        (df["pct_vs_last4week"] <= -20.298) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 1768327900.0) &
        (df["three_m_chg_rate"] <= 165.087),
    "rule_135__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["ma20_chg_rate"] <= -0.82) &
        (df["vol30"] <= 2.96) &
        (df["ma5_chg_rate"] <= -0.58),
    "rule_136__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["ma20_chg_rate"] <= -0.82) &
        (df["vol30"] <= 2.96) &
        (df["three_m_chg_rate"] <= 87.674),
    "rule_137__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["ma20_chg_rate"] <= -0.82) &
        (df["vol30"] <= 2.96) &
        (df["three_m_chg_rate"] <= 100.82),
    "rule_138__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["ma20_chg_rate"] <= -0.82) &
        (df["vol30"] <= 2.96) &
        (df["three_m_chg_rate"] <= 118.43),
    "rule_139__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["ma20_chg_rate"] <= -0.82) &
        (df["vol30"] <= 2.96) &
        (df["today_chg_rate"] >= -47.818),
    "rule_140__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["ma20_chg_rate"] <= -0.82) &
        (df["vol30"] <= 2.96) &
        (df["today_chg_rate"] >= -40.344),
    "rule_141__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["ma20_chg_rate"] <= -0.82) &
        (df["vol30"] <= 2.96) &
        (df["pct_vs_firstweek"] >= -31.118),
    "rule_142__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["chg_tr_val"] <= -27.4) &
        (df["vol20"] >= 2.5),
    "rule_143__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["chg_tr_val"] <= -27.4) &
        (df["pct_vs_lastweek"] <= 4.55),
    "rule_144__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["chg_tr_val"] <= -2.76) &
        (df["ma5_chg_rate"] >= -1.499),
    "rule_145__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["vol20"] >= 2.84),
    "rule_146__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["three_m_chg_rate"] >= 42.201),
    "rule_147__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["three_m_chg_rate"] >= 45.726),
    "rule_148__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["today_chg_rate"] <= -24.172),
    "rule_149__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.6) &
        (df["pct_vs_lastweek"] <= 3.23),
    "rule_150__n20__r0.900":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["pct_vs_firstweek"] >= -17.32) &
        (df["vol30"] >= 4.17),
    "rule_151__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_tr_val"] <= 1113627808.1) &
        (df["mean_ret30"] <= -0.57),
    "rule_152__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_tr_val"] <= 1113627808.1) &
        (df["mean_ret30"] <= -0.48),
    "rule_153__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_tr_val"] <= 1113627808.1) &
        (df["three_m_chg_rate"] >= 35.97),
    "rule_154__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_tr_val"] <= 1113627808.1) &
        (df["today_chg_rate"] <= -20.75),
    "rule_155__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.9) &
        (df["three_m_chg_rate"] >= 39.05),
    "rule_156__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.9) &
        (df["today_chg_rate"] <= -22.38),
    "rule_157__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.9) &
        (df["today_chg_rate"] <= -20.75),
    "rule_158__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["ma5_chg_rate"] <= -0.03),
    "rule_159__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["mean_ret30"] <= -0.57),
    "rule_160__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["mean_ret30"] <= -0.48),
    "rule_161__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["three_m_chg_rate"] >= 35.97),
    "rule_162__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["today_chg_rate"] <= -20.75),
    "rule_163__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= -1.257) &
        (df["pct_vs_firstweek"] <= 22.562),
    "rule_164__n20__r0.900":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["mean_ret30"] <= -0.77) &
        (df["three_m_chg_rate"] <= 61.086) &
        (df["vol20"] >= 2.67),
    "rule_165__n20__r0.900":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["mean_ret30"] <= -0.77) &
        (df["three_m_chg_rate"] <= 61.086) &
        (df["pct_vs_last4week"] <= -17.377),
    "rule_166__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 2.96) &
        (df["mean_ret30"] <= -0.67) &
        (df["three_m_chg_rate"] <= 87.674),
    "rule_167__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["today_pct"] >= 7.9) &
        (df["vol20"] <= 6.566) &
        (df["ma5_chg_rate"] <= -0.28),
    "rule_168__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["today_pct"] >= 7.9) &
        (df["vol20"] <= 6.566) &
        (df["ma5_chg_rate"] <= -0.03),
    "rule_169__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["today_pct"] >= 7.9) &
        (df["vol20"] <= 6.566) &
        (df["ma5_chg_rate"] <= 0.19),
    "rule_170__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["today_pct"] >= 7.9) &
        (df["vol20"] <= 6.566) &
        (df["chg_tr_val"] <= 177.88),
    "rule_171__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["today_pct"] >= 7.9) &
        (df["vol20"] <= 6.566) &
        (df["chg_tr_val"] <= 220.6),
    "rule_172__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 5.166) &
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["pct_vs_firstweek"] <= 30.465),
    "rule_173__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 5.166) &
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["pct_vs_firstweek"] <= 42.542),
    "rule_174__n20__r0.900":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["pos30_ratio"] >= 36.67) &
        (df["vol20"] <= 7.878),
    "rule_175__n20__r0.900":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["pos30_ratio"] >= 36.67) &
        (df["mean_prev3"] >= 652229313.7),
    "rule_176__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_tr_val"] <= 1393261612.0) &
        (df["pct_vs_firstweek"] <= -11.369),
    "rule_177__n20__r0.900":
        (df["pct_vs_last4week"] <= -20.298) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 2210902064.0) &
        (df["pct_vs_lastweek"] <= 0.454),
    "rule_178__n20__r0.900":
        (df["pct_vs_last4week"] <= -20.298) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 2210902064.0) &
        (df["pct_vs_lastweek"] <= 1.915),

    "rule_180__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.6) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["today_chg_rate"] >= -40.344),
    "rule_181__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.6) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["pct_vs_firstweek"] <= -11.369),
    "rule_182__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_pct"] >= 5.1) &
        (df["ma5_chg_rate"] <= -0.58),

    "rule_184__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["pct_vs_lastweek"] <= -2.88),
    "rule_185__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["vol30"] <= 3.96) &
        (df["pct_vs_firstweek"] <= -20.896),
    "rule_186__n20__r0.900":
        (df["pct_vs_last2week"] >= 25.891) &
        (df["pos30_ratio"] <= 36.67) &
        (df["three_m_chg_rate"] >= 70.766) &
        (df["mean_prev3"] <= 6149198580.0),
    "rule_187__n20__r0.900":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_prev3"] <= 3049111908.5) &
        (df["ma5_chg_rate"] <= -0.03),
    "rule_188__n20__r0.900":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_prev3"] <= 3049111908.5) &
        (df["ma5_chg_rate"] <= 0.19),
    "rule_189__n20__r0.900":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_prev3"] <= 3049111908.5) &
        (df["ma5_chg_rate"] <= 0.42),
    "rule_190__n20__r0.900":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_prev3"] <= 3049111908.5) &
        (df["ma5_chg_rate"] <= 0.62),
    "rule_191__n20__r0.900":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["ma20_chg_rate"] >= -1.26) &
        (df["mean_prev3"] <= 3049111908.5) &
        (df["pct_vs_firstweek"] >= -31.118),
    "rule_192__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["ma5_chg_rate"] <= -0.908) &
        (df["today_chg_rate"] >= -40.344),
    "rule_193__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -19.699) &
        (df["mean_ret20"] >= -1.42) &
        (df["vol20"] <= 6.566),

    "rule_196__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 3049111908.5),
    "rule_197__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 5.1),
    "rule_198__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.76) &
        (df["today_pct"] >= 5.3),
    "rule_199__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.44) &
        (df["today_tr_val"] <= 850188004.0) &
        (df["today_chg_rate"] >= -40.344),
    "rule_200__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["today_pct"] >= 7.4) &
        (df["vol20"] <= 6.566) &
        (df["pct_vs_lastweek"] <= 1.915),
    "rule_201__n20__r0.900":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.4) &
        (df["vol20"] <= 4.32) &
        (df["chg_tr_val"] <= 19.5),
    "rule_202__n20__r0.900":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.4) &
        (df["vol20"] <= 4.32) &
        (df["chg_tr_val"] <= 31.94),
    "rule_203__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.6) &
        (df["today_tr_val"] <= 1393261612.0) &
        (df["pct_vs_firstweek"] <= -20.896),
    "rule_204__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["ma5_chg_rate"] <= -0.908) &
        (df["mean_prev3"] <= 2499554280.14),
    "rule_205__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["ma5_chg_rate"] <= -0.908) &
        (df["today_pct"] >= 5.1),
    "rule_206__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -15.276) &
        (df["pct_vs_firstweek"] >= -17.32) &
        (df["mean_ret30"] <= -0.67),
    "rule_207__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -15.276) &
        (df["pct_vs_firstweek"] >= -17.32) &
        (df["mean_ret30"] <= -0.57),
    "rule_208__n20__r0.900":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.6) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["pct_vs_firstweek"] <= -20.896),
    "rule_209__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 2.96) &
        (df["mean_ret30"] <= -0.57) &
        (df["today_chg_rate"] >= -40.344),
    "rule_210__n20__r0.900":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.25) &
        (df["vol20"] >= 2.84) &
        (df["today_pct"] >= 4.3),
    "rule_211__n20__r0.900":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.25) &
        (df["vol20"] >= 2.84) &
        (df["today_pct"] >= 4.5),
    "rule_212__n20__r0.900":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.44) &
        (df["three_m_chg_rate"] <= 78.15) &
        (df["ma5_chg_rate"] <= 0.19),
    "rule_213__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["three_m_chg_rate"] <= 64.993) &
        (df["mean_ret30"] <= -0.67),
    "rule_214__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.76) &
        (df["today_chg_rate"] >= -35.7),
    "rule_215__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.44) &
        (df["today_tr_val"] <= 1393261612.0) &
        (df["today_pct"] >= 5.3),
    "rule_216__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.44) &
        (df["today_tr_val"] <= 1393261612.0) &
        (df["today_pct"] >= 5.6),
    "rule_217__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.6) &
        (df["mean_prev3"] <= 959522214.35) &
        (df["pct_vs_lastweek"] <= -1.257),
    "rule_218__n20__r0.900":
        (df["ma5_chg_rate"] >= 5.499) &
        (df["mean_prev3"] <= 3843839864.0) &
        (df["vol20"] >= 7.878) &
        (df["three_m_chg_rate"] >= 70.766),
    "rule_219__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["three_m_chg_rate"] >= 56.451) &
        (df["vol30"] <= 3.6) &
        (df["today_pct"] >= 6.2),
    "rule_220__n20__r0.900":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.44) &
        (df["today_pct"] >= 5.3) &
        (df["vol20"] >= 2.84),
    "rule_221__n20__r0.900":
        (df["pct_vs_last3week"] <= -19.699) &
        (df["pct_vs_firstweek"] >= -17.32) &
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["vol20"] >= 3.71),
    "rule_222__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.55) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["pct_vs_firstweek"] <= 30.465),
    "rule_223__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.55) &
        (df["ma5_chg_rate"] <= -0.58) &
        (df["pct_vs_firstweek"] <= 42.542),
    "rule_224__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.6) &
        (df["today_pct"] >= 5.3),
    "rule_225__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.09) &
        (df["today_tr_val"] <= 1768327900.0) &
        (df["pct_vs_firstweek"] <= -11.369),
    "rule_226__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= 0.454) &
        (df["pct_vs_firstweek"] <= 3.73),
    "rule_227__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= 0.454) &
        (df["pct_vs_firstweek"] <= 7.313),
    "rule_228__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= 0.454) &
        (df["pct_vs_firstweek"] <= 12.062),
    "rule_229__n20__r0.900":
        (df["pct_vs_last4week"] <= -24.697) &
        (df["vol20"] <= 4.835) &
        (df["pct_vs_lastweek"] <= 0.454) &
        (df["pct_vs_firstweek"] <= 16.715),
    "rule_230__n20__r0.900":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["vol30"] <= 5.42),
    "rule_231__n20__r0.900":
        (df["pct_vs_last2week"] <= -8.998) &
        (df["today_pct"] >= 7.9) &
        (df["chg_tr_val"] <= -2.76) &
        (df["pct_vs_lastweek"] <= -1.257),
    "rule_232__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["pct_vs_last3week"] <= -15.276) &
        (df["pos30_ratio"] >= 36.67) &
        (df["today_chg_rate"] >= -40.344),
    "rule_233__n20__r0.900":
        (df["pct_vs_lastweek"] <= -4.858) &
        (df["ma20_chg_rate"] <= -1.08) &
        (df["vol30"] <= 4.39) &
        (df["three_m_chg_rate"] >= 61.086),
    "rule_234__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.6) &
        (df["today_tr_val"] <= 1113627808.1) &
        (df["vol20"] <= 3.39),
    "rule_235__n20__r0.900":
        (df["pct_vs_lastweek"] <= -2.88) &
        (df["today_chg_rate"] <= -32.46) &
        (df["vol30"] <= 3.76) &
        (df["mean_prev3"] <= 1783551861.23),
    "rule_236__n20__r0.900":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["pct_vs_last4week"] <= -14.88) &
        (df["vol30"] <= 3.25) &
        (df["today_pct"] >= 5.6),
    "rule_237__n20__r0.900":
        (df["mean_ret30"] <= -0.938) &
        (df["vol30"] <= 3.44) &
        (df["today_chg_rate"] >= -40.344) &
        (df["ma5_chg_rate"] <= 0.19),
    "rule_238__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["ma20_chg_rate"] >= -1.08) &
        (df["mean_prev3"] <= 738962733.49) &
        (df["ma5_chg_rate"] <= 0.19),
    "rule_239__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["ma20_chg_rate"] >= -1.08) &
        (df["mean_prev3"] <= 738962733.49) &
        (df["ma5_chg_rate"] <= 0.42),
    "rule_240__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["ma20_chg_rate"] >= -1.08) &
        (df["mean_prev3"] <= 738962733.49) &
        (df["ma5_chg_rate"] <= 0.62),
    "rule_241__n20__r0.900":
        (df["pct_vs_last3week"] <= -15.276) &
        (df["vol30"] <= 3.25) &
        (df["today_chg_rate"] <= -30.015) &
        (df["today_pct"] >= 4.5),
    "rule_242__n20__r0.900":
        (df["pct_vs_last4week"] <= -20.298) &
        (df["vol30"] <= 3.96) &
        (df["today_tr_val"] <= 1393261612.0) &
        (df["pct_vs_lastweek"] <= 0.454),
    "rule_243__n20__r0.900":
        (df["pct_vs_last4week"] <= -20.298) &
        (df["vol30"] <= 3.96) &
        (df["today_tr_val"] <= 1393261612.0) &
        (df["pct_vs_lastweek"] <= 1.915),

    "rule_245__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.44) &
        (df["today_tr_val"] <= 1768327900.0) &
        (df["today_pct"] >= 5.3),
    "rule_246__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.44) &
        (df["today_tr_val"] <= 1768327900.0) &
        (df["today_pct"] >= 5.6),
    "rule_247__n20__r0.900":
        (df["ma5_chg_rate"] <= -0.908) &
        (df["today_pct"] >= 6.5) &
        (df["mean_ret30"] <= -0.77) &
        (df["vol20"] <= 3.88),
    "rule_248__n20__r0.900":
        (df["pct_vs_last4week"] <= -17.377) &
        (df["vol30"] <= 3.25) &
        (df["mean_ret30"] <= -0.77) &
        (df["today_pct"] >= 5.6),

    ####################### 미장 #######################
    # # n=32, 27/32 = 84.4%
    # "hi80_rule_us_A_ratio_0_844_n32":
    #     (df["pct_vs_last2week"] < -39.0846) &
    #     (df["today_tr_val"] < 9.20874e+08),
    #
    # # n=36, 29/36 = 80.6%
    # "hi80_rule_us_B_ratio_0_806_n36":
    #     (df["pct_vs_last2week"] < -39.0846) &
    #     (df["today_tr_val"] < 1.02635e+09),
    #
    # # (표본은 줄지만 더 강함) n=23, 20/23 = 87.0%
    # "hi80_rule_us_C_ratio_0_870_n23":
    #     (df["pct_vs_last2week"] < -39.0846) &
    #     (df["today_tr_val"] < 6.41818e+08),
    #
    # # n=17, 16/17 = 94.1%
    # "hi80_us_941_n17_last2w_lt_-39_0846_and_tr_lt_447_1m":
    #     (df["pct_vs_last2week"] < -39.0846) &
    #     (df["today_tr_val"] < 447130958.5),
    #
    # # n=23, 20/23 = 87.0%
    # "hi80_us_870_n23_last2w_lt_-39_0846_and_tr_lt_641_8m":
    #     (df["pct_vs_last2week"] < -39.0846) &
    #     (df["today_tr_val"] < 641818383.82),
    #
    # # n=20, 17/20 = 85.0%
    # "hi80_us_850_n20_last2w_lt_-39_0846_and_tr_lt_523_0m":
    #     (df["pct_vs_last2week"] < -39.0846) &
    #     (df["today_tr_val"] < 522960472.58),
    #
    # # n=29, 24/29 = 82.8%
    # "hi80_us_828_n29_last2w_lt_-39_0846_and_tr_lt_869_0m":
    #     (df["pct_vs_last2week"] < -39.0846) &
    #     (df["today_tr_val"] < 869029586.805),
    #
    # # n=23, 19/23 = 82.6%
    # "hi80_us_826_n23_last2w_lt_-33_2714_and_tr_lt_447_1m":
    #     (df["pct_vs_last2week"] < -33.2714) &
    #     (df["today_tr_val"] < 447130958.5),
    #
    # # n=31, 25/31 = 80.6%
    # "hi80_us_806_n31_last2w_lt_-29_2584_and_tr_lt_447_1m":
    #     (df["pct_vs_last2week"] < -29.2584) &
    #     (df["today_tr_val"] < 447130958.5),
    #
    # # n=21, 17/21 = 81.0%
    # "hi80_us_810_n21_todaychg_lt_-96_2615_and_ma5_gt_2_18":
    #     (df["today_chg_rate"] < -96.2615) &
    #     (df["ma5_chg_rate"] > 2.18),
    #
    # # n=18, 15/18 = 83.3%
    # "hi80_us_833_n18_tr_lt_148_2m_and_ma20_lt_-1_38":
    #     (df["today_tr_val"] < 148169623.835) &
    #     (df["ma20_chg_rate"] < -1.38),
    #
    # # n=17, 14/17 = 82.4%
    # "hi80_us_824_n17_tr_lt_148_2m_and_todaychg_lt_-46_107":
    #     (df["today_tr_val"] < 148169623.835) &
    #     (df["today_chg_rate"] < -46.107),
    #
    # # n=20, 16/20 = 80.0%
    # "hi80_us_800_n20_tr_lt_148_2m_and_todaychg_lt_-42_391":
    #     (df["today_tr_val"] < 148169623.835) &
    #     (df["today_chg_rate"] < -42.391),
    #
    # # n=31, ratio=1.000
    # "hi80_us_rule_1_r1.000_n31":
    #     (df["chg_tr_val"] <= 31.35) &
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] > -64.24) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 3.795),
    #
    # # n=26, ratio=1.000
    # "hi80_us_rule_2_r1.000_n26":
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] > -64.24) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 8.415),
    #
    # # n=26, ratio=1.000
    # "hi80_us_rule_3_r1.000_n26":
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] > -64.24) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] > 9.315) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 3.795),
    #
    # # n=21, ratio=1.000
    # "hi80_us_rule_4_r1.000_n21":
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_prev3"] > 2665270784) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] > -64.24) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 3.795),
    #
    # # n=21, ratio=1.000
    # "hi80_us_rule_5_r1.000_n21":
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] > -64.24) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 8.93),
    #
    # # n=41, ratio=0.927
    # "hi80_us_rule_6_r0.927_n41":
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] > -64.24) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 3.795),
    #
    # # n=22, ratio=0.864
    # "hi80_us_rule_7_r0.864_n22":
    #     (df["mean_ret30"] > -0.945) &
    #     (df["pct_vs_firstweek"] <= -36.175) &
    #     (df["pct_vs_last2week"] <= -10.995) &
    #     (df["pct_vs_last3week"] <= -16.79) &
    #     (df["pct_vs_lastweek"] > 4.07) &
    #     (df["three_m_chg_rate"] > 84.15) &
    #     (df["today_pct"] <= 41.45) &
    #     (df["vol30"] > 3.795),
    #
    # # n=21, ratio=0.857
    # "hi80_us_rule_10_r0.857_n21":
    #     (df["chg_tr_val"] > 48.05) &
    #     (df["mean_ret20"] <= -0.655) &
    #     (df["mean_ret30"] > 0.135) &
    #     (df["three_m_chg_rate"] <= 84.15) &
    #     (df["today_pct"] <= 41.45) &
    #     (df["vol30"] > 3.795),
    #
    # # n=20, ratio=0.850
    # "hi80_us_rule_11_r0.850_n20":
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] <= -75.465) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 3.795),
    #
    # # n=20, ratio=0.850
    # "hi80_us_rule_12_r0.850_n20":
    #     (df["ma5_chg_rate"] > -11.895) &
    #     (df["mean_prev3"] <= 2665270784) &
    #     (df["mean_ret30"] <= -0.945) &
    #     (df["pct_vs_firstweek"] > -64.24) &
    #     (df["pct_vs_last2week"] <= -38.495) &
    #     (df["vol20"] <= 19.4) &
    #     (df["vol30"] > 3.795),
}



# 조건을 만족하는 행등만 골라서(sub) 확인
def test_condition(name, cond):
    """
    param cond: 조건식 결과
    """

    # 조건을 만족한 행들만 모인 DataFrame
    sub = df[cond]

    if len(sub) == 0:
        print(f"\n=== {name} ===")
        print("선택된 행이 없습니다.")
        return

    up_cnt = (sub["validation_chg_rate"] >= 7).sum()
    ratio = up_cnt / len(sub)

    # if (ratio > 0.88):
    #     return

    # if len(sub) > 19:
    #     return

    print(f"\n=== {name} ===")
    print(f"선택된 행 수: {len(sub)}")
    print(f"  validation_chg_rate >= 7 개수 : {up_cnt}")
    print(f"  Ratio (>=7)      : {ratio:.3f}")

    # validation_chg_rate >= 7 인 행들만 보기
    # sub_up = sub[sub["validation_chg_rate"] >= 7]
    # print("\n  ▶ validation_chg_rate >= 7 종목 목록")
    # print(sub_up[["ticker", "stock_name", "validation_chg_rate"]])   # 국장
    # print(sub_up[["ticker", "validation_chg_rate"]])   # 미장


# 모든 조건 테스트
for name, cond in conditions.items():
    test_condition(name, cond)


# saved = sort_csv_by_today_desc(
#     in_path=r"csv/low_result_us.csv",
#     out_path=r"csv/low_result_us_desc.csv",
# )
# print("saved:", saved)


