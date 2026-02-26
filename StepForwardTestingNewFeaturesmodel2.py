from __future__ import annotations

from dotenv import load_dotenv
from numpy.f2py.crackfortran import verbose
from pandas.api.types import is_datetime64_any_dtype
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from dateutil.relativedelta import relativedelta
import requests
import os
import hmac
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from betacal import BetaCalibration
from sklearn.base import clone
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
from pybit.unified_trading import HTTP

import os, json, uuid, time
from collections import defaultdict
from typing import Optional, Dict, List, Literal, Tuple, Callable, Union, Any, cast
from dataclasses import dataclass, asdict

from pathlib import Path
from math import sqrt
import warnings
import smtplib
from email.mime.text import MIMEText
import pytz
from numba import njit
# from rules_engine import validate_start_order_fast
import pandas as pd

import math
import seaborn as sns


# helpers de librerías opcionales
_HAS_LGBM, _HAS_XGB = False, False
try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except Exception:
    pass

# ===== NUEVO: import opcional betacal =====
_HAS_BETACAL = False
try:
    from betacal import BetaCalibration

    _HAS_BETACAL = True
except Exception:
    pass

load_dotenv()

api_key = os.getenv("API_KEY")
api_secret = os.getenv("API_SECRET")
api_key_roge = os.getenv("API_KEY_ROGE")
api_secret_roge = os.getenv("API_SECRET_ROGE")

api_key_demo = os.getenv("API_KEY_DEMO")
api_secret_demo = os.getenv("API_SECRET_DEMO")

email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")

EMAIL_SMTP = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_RECEIVER = "candrelosolaz@gmail.com"

RESTART_HOUR_1 = 3
RESTART_HOUR_2 = 12
RESTART_HOUR_3 = 17
RESTART_HOUR_4 = 22

# Winrate 45.79 %
# FEATURES = [
#     'open', 'volume', 'k100', 'e100', 'k', 'e', 'ATR', 'macd', 'macd_hist', 'VWAP', 'return_1_raw',
#     'volatility_10', 'volatility_50', 'atr_norm', 'rsi_14', 'dist_PP', 'dist_S1', 'dist_R1', 'near_PP_pct',
#     'near_PP_atr', 'near_S1_pct', 'near_S1_atr', 'near_R1_pct', 'near_R1_atr', 'near_S2_pct', 'near_S2_atr',
#     'near_R2_pct', 'near_R2_atr', 'dist_resist_rl', 'dist_support_rl', 'near_resist_rl_pct', 'near_support_rl_pct',
#     'near_resist_rl_atr', 'near_support_rl_atr', 'price_change', 'avg_volume_last_n', 'rsi_50', 'roc_10', 'roc_30',
#     'momentum_diff', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos', 'session_ny_open', 'session_london_open',
#     'session_asia_open', 'dist_P_8h_atr', 'dist_R1_8h_atr', 'dist_S1_8h_atr', 'dist_R2_8h_atr', 'dist_S2_8h_atr',
#     'near_res_8h', 'near_sup_8h', 'stoch_overbought_80', 'stoch_overbought_90', 'stoch_oversold_20',
#     'stoch_oversold_10', 'stoch_k_above_d', 'trend_slope_norm_100', 'trend_r2_100', 'trend_dist_pct_100',
#     'trend_channel_slope_high', 'candle_doji', 'candle_pinbar', 'candle_engulfing', 'rsi14_oversold_near_support',
#     'rsi14_overbought_near_resist', 'trend_ema200_down', 'range_prev_day_abs', 'range_prev_day_pct',
#     'gap_open_abs', 'overnight_range_abs', 'overnight_range_pct', 'range_prev_day_atr', 'overnight_range_atr',
#     'vwap_up', 'dist_to_vwap', 'vol_ratio', 'z_atr_norm', 'z_volume', 'min_dist_level_atr', 'delta_trend_slope_100',
#     'rsi_40_60', 'ret_abs', 'ret_diff2', 'ret_mean_s', 'ret_mean_m', 'ret_skew_m', 'ret_kurt_m', 'var_95_m', 'acf1_s',
#     'acf2_s', 'p_entropy_s', 'hurst_m', 'slope_m', 'vol_rel_m', 'vol_std_m', 'up_vol_ratio', 'corr_close_vol_m',
#     'var_ratio', 'cross_ma_m', 'market_regime', 'hmm_state', 'buy_pressure', 'sell_pressure', 'signed_volume_ema',
#     'upper_wick', 'lower_wick', 'wick_imbalance', 'micro_slope', 'ret_ema_20', 'shock_ema', 'trend_slope_ema',
#     'trend_r2_50', 'trend_acc'
# ]

# Winrate: 46.51% correlation < 0.85
# FEATURES = [
#     'open', 'volume', 'k100', 'e100', 'k', 'e', 'ATR', 'macd', 'macd_hist', 'VWAP', 'return_1_raw', 'volatility_10',
#     'volatility_50', 'rsi_14', 'dist_PP', 'dist_S1', 'dist_R1', 'near_PP_pct', 'near_PP_atr', 'near_S1_pct',
#     'near_S1_atr', 'near_R1_pct', 'near_R1_atr', 'near_S2_pct', 'near_S2_atr', 'near_R2_pct', 'near_R2_atr',
#     'dist_resist_rl', 'dist_support_rl', 'near_resist_rl_pct', 'near_support_rl_pct', 'near_resist_rl_atr',
#     'near_support_rl_atr', 'price_change', 'avg_volume_last_n', 'rsi_50', 'roc_10', 'momentum_diff', 'minute_sin',
#     'minute_cos', 'dow_sin', 'dow_cos', 'session_ny_open', 'session_london_open', 'session_asia_open', 'dist_P_8h_atr',
#     'dist_R1_8h_atr', 'dist_S1_8h_atr', 'near_res_8h', 'near_sup_8h', 'stoch_overbought_80', 'stoch_overbought_90',
#     'stoch_oversold_20', 'stoch_oversold_10', 'stoch_k_above_d', 'trend_r2_100', 'trend_dist_pct_100',
#     'trend_channel_slope_high', 'candle_doji', 'candle_pinbar', 'candle_engulfing', 'rsi14_oversold_near_support',
#     'rsi14_overbought_near_resist', 'trend_ema200_down', 'range_prev_day_abs', 'range_prev_day_pct', 'gap_open_abs',
#     'overnight_range_pct', 'range_prev_day_atr', 'overnight_range_atr', 'vwap_up', 'vol_ratio', 'z_atr_norm',
#     'z_volume', 'min_dist_level_atr', 'delta_trend_slope_100', 'rsi_40_60', 'ret_abs', 'ret_diff2', 'ret_skew_m',
#     'ret_kurt_m', 'var_95_m', 'acf1_s', 'acf2_s', 'p_entropy_s', 'hurst_m', 'vol_std_m', 'up_vol_ratio',
#     'corr_close_vol_m', 'var_ratio', 'cross_ma_m', 'market_regime', 'hmm_state', 'signed_volume_ema', 'upper_wick',
#     'lower_wick', 'wick_imbalance', 'micro_slope', 'shock_ema', 'trend_slope_ema', 'trend_r2_50', 'trend_acc'
# ]

# # # Winrate: 48.46% correlation < 0.75 sin baja relevancia
# FEATURES = [
#     'open', 'volume', 'k100', 'e100', 'k', 'e', 'macd', 'macd_hist', 'VWAP', 'return_1_raw', 'volatility_10',
#     'volatility_50', 'rsi_14', 'dist_PP', 'dist_S1', 'dist_R1', 'near_PP_pct', 'near_PP_atr', 'near_S1_pct',
#     'near_R1_pct', 'near_R1_atr', 'near_R2_atr',
#     'dist_resist_rl', 'dist_support_rl', 'near_resist_rl_atr', 'near_support_rl_atr', 'price_change',
#     'avg_volume_last_n', 'momentum_diff', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos', 'session_ny_open',
#     'session_london_open', 'session_asia_open', 'dist_P_8h_atr', 'dist_R1_8h_atr', 'dist_S1_8h_atr', 'near_res_8h',
#     'stoch_overbought_80', 'stoch_overbought_90', 'stoch_oversold_20',
#     'stoch_k_above_d', 'trend_r2_100', 'trend_dist_pct_100', 'candle_pinbar', 'candle_engulfing',
#     'rsi14_oversold_near_support', 'rsi14_overbought_near_resist', 'trend_ema200_down',
#     'overnight_range_pct', 'range_prev_day_atr', 'overnight_range_atr', 'vwap_up', 'vol_ratio', 'z_atr_norm',
#     'z_volume', 'min_dist_level_atr', 'delta_trend_slope_100', 'rsi_40_60', 'ret_abs', 'ret_diff2', 'ret_skew_m',
#     'ret_kurt_m', 'acf1_s', 'acf2_s', 'p_entropy_s', 'hurst_m', 'vol_std_m', 'up_vol_ratio', 'corr_close_vol_m',
#     'var_ratio', 'cross_ma_m', 'market_regime', 'hmm_state', 'signed_volume_ema', 'upper_wick', 'lower_wick',
#     'wick_imbalance', 'micro_slope', 'shock_ema', 'trend_slope_ema', 'trend_r2_50', 'trend_acc'
# ]


# # Winrate: 48.46% correlation < 0.75 sin baja relevancia
# FEATURES = [
#     'open', 'volume', 'k100', 'e100', 'k', 'e', 'macd', 'macd_hist', 'VWAP', 'return_1_raw', 'volatility_10',
#     'volatility_50', 'rsi_14', 'dist_PP', 'dist_S1', 'dist_R1', 'near_PP_pct', 'near_PP_atr', 'near_S1_pct',
#     'near_R1_pct', 'near_R1_atr', 'near_R2_atr',
#     'dist_resist_rl', 'dist_support_rl', 'near_resist_rl_atr', 'near_support_rl_atr', 'price_change',
#     'avg_volume_last_n', 'momentum_diff', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos', 'session_ny_open',
#     'session_london_open', 'session_asia_open', 'dist_P_8h_atr', 'dist_R1_8h_atr', 'dist_S1_8h_atr', 'near_res_8h',
#     'stoch_overbought_80', 'stoch_overbought_90', 'stoch_oversold_20',
#     'stoch_k_above_d', 'trend_r2_100', 'trend_dist_pct_100', 'candle_pinbar', 'candle_engulfing',
#     'rsi14_oversold_near_support', 'rsi14_overbought_near_resist', 'trend_ema200_down',
#     'overnight_range_pct', 'range_prev_day_atr', 'overnight_range_atr', 'vwap_up', 'vol_ratio', 'z_atr_norm',
#     'z_volume', 'min_dist_level_atr', 'delta_trend_slope_100', 'rsi_40_60', 'ret_abs', 'ret_diff2', 'ret_skew_m',
#     'ret_kurt_m', 'acf1_s', 'acf2_s', 'p_entropy_s', 'hurst_m', 'vol_std_m', 'up_vol_ratio', 'corr_close_vol_m',
#     'var_ratio', 'cross_ma_m', 'market_regime', 'hmm_state', 'signed_volume_ema', 'upper_wick', 'lower_wick',
#     'wick_imbalance', 'micro_slope', 'shock_ema', 'trend_slope_ema', 'trend_r2_50', 'trend_acc'
# ]

FEATURES = [
    'VWAP',
    'overnight_range_pct',
    'open',
    'dist_R1',
    'dist_S1',
    'minute_sin',
    'volatility_50',
    'minute_cos',
    'dist_PP',
    'e100',
    'trend_slope_ema',
    'trend_r2_100',
    'min_dist_level_atr',
    'dist_R1_8h_atr',
    'avg_volume_last_n',
    'dist_P_8h_atr',
    'vol_std_m',
    'overnight_range_atr',
    'shock_ema',
    'range_prev_day_atr',
    'dist_S1_8h_atr',
    'ret_kurt_m',
    'ret_skew_m',
    'e',
    'z_atr_norm',
    'corr_close_vol_m',
    'trend_r2_50',
    'k100',
    'hurst_m',
    'trend_dist_pct_100',
    'var_ratio',
    'up_vol_ratio',
    'signed_volume_ema',
    'delta_trend_slope_100',
    # 'dom_sin',
    # 'dom_cos',
    'dow_sin',
    'dow_cos',
    'macd',
    'vol_ratio',
    'acf1_s',
    'acf2_s',
    'volatility_10',
    'dist_support_rl',
    'macd_hist',
    'dist_resist_rl',
    'momentum_diff',
    'price_change',
    'cross_ma_m',
    'rsi_14',
    'k',
    'z_volume',
    'volume',
    'micro_slope',
    'p_entropy_s',
    'wick_imbalance',
    'ret_abs',
    'trend_acc',
    'ret_diff2',
    'lower_wick',
    'upper_wick',
    'return_1_raw',
    'hmm_state',
    'vwap_up',
    'market_regime',
    'trend_ema200',
    # 'session_london_open',
    'near_PP_pct',
    'near_S1_pct',
    'near_R1_pct',
    # 'rsi_40_60',
    # 'session_ny_open',
    # 'session_asia_open',
    # 'candle_engulfing',
    # 'stoch_overbought_80',
    # 'stoch_k_above_d',
    # 'rsi14_oversold_near_support',
    # 'stoch_oversold_20',
    # 'near_PP_atr',
    # 'candle_pinbar',
    # 'near_R1_atr',
    # 'near_support_rl_atr',
    # 'rsi14_overbought_near_resist',
    # 'near_resist_rl_atr',
    # 'stoch_overbought_90',
    # 'near_R2_atr',
    # 'near_res_8h'
]
#56.8%
# FEATURES = [
#     'VWAP',
#     'overnight_range_pct',
#     'open',
#     'dist_R1',
#     'dist_S1',
#     'minute_sin',
#     'volatility_50',
#     'minute_cos',
#     'dist_PP',
#     'e100',
#     'trend_slope_ema',
#     'trend_r2_100',
#     'min_dist_level_atr',
#     'dist_R1_8h_atr',
#     'avg_volume_last_n',
#     'dist_P_8h_atr',
#     'vol_std_m',
#     'overnight_range_atr',
#     'shock_ema',
#     'range_prev_day_atr',
#     'dist_S1_8h_atr',
#     'ret_kurt_m',
#     'ret_skew_m',
#     'e',
#     'z_atr_norm',
#     'corr_close_vol_m',
#     'dow_sin',
#     'trend_r2_50',
#     'k100',
#     'hurst_m',
#     'trend_dist_pct_100',
#     'var_ratio',
#     'up_vol_ratio',
#     'signed_volume_ema',
#     'delta_trend_slope_100',
#     'dow_cos',
#     'macd',
#     'vol_ratio',
#     'acf1_s',
#     'acf2_s',
#     'volatility_10',
#     'dist_support_rl',
#     'macd_hist',
#     'dist_resist_rl',
#     'momentum_diff',
#     'price_change',
#     'cross_ma_m',
#     'rsi_14',
#     'k',
#     'z_volume',
#     'volume',
#     'micro_slope',
#     'p_entropy_s',
#     'wick_imbalance',
#     'ret_abs',
#     'trend_acc',
#     'ret_diff2',
#     'lower_wick',
#     'upper_wick',
#     'return_1_raw',
#     'hmm_state',
#     # 'vwap_up',
#     'market_regime',
#     'trend_ema200_down',
#     # 'session_london_open',
#     'near_PP_pct',
#     'near_S1_pct',
#     'near_R1_pct',
# ]

# # # %62
# FEATURES = [
#     'VWAP',
#     'overnight_range_pct',
#     'open',
#     'dist_R1',
#     'dist_S1',
#     'minute_sin',
#     'volatility_50',
#     'minute_cos',
#     'dist_PP',
#     'e100',
#     'trend_slope_ema',
#     'trend_r2_100',
#     'min_dist_level_atr',
#     'dist_R1_8h_atr',
#     'avg_volume_last_n',
#     'dist_P_8h_atr',
#     'vol_std_m',
#     'overnight_range_atr',
#     'shock_ema',
#     'range_prev_day_atr',
#     'dist_S1_8h_atr',
#     'ret_kurt_m',
#     'ret_skew_m',
#     'e',
#     'z_atr_norm',
#     'corr_close_vol_m',
#     'dow_sin',
#     'trend_r2_50',
#     'k100',
#     'hurst_m',
#     'trend_dist_pct_100',
#     'var_ratio',
#     'up_vol_ratio',
#     'signed_volume_ema',
#     'delta_trend_slope_100',
#     'dow_cos',
#     'macd',
#     'vol_ratio',
#     'acf1_s',
#     'acf2_s',
#     'volatility_10',
#     'dist_support_rl',
#     'macd_hist',
#     'dist_resist_rl',
#     'momentum_diff',
#     'price_change',
#     'cross_ma_m',
#     'rsi_14',
#     'k',
#     'z_volume',
#     'volume',
#     'micro_slope',
#     'p_entropy_s',
#     'wick_imbalance',
#     'ret_abs',
#     'trend_acc',
#     'ret_diff2',
#     'lower_wick',
#     'upper_wick',
#     # 'return_1_raw',
#     'hmm_state',
#     # 'vwap_up',
#     'market_regime',
#     'trend_ema200_down',
#     # 'session_london_open',
#     'near_PP_pct',
#     'near_S1_pct',
#     'near_R1_pct',
# ]


# %62
# FEATURES = [
#     'VWAP',
#     'overnight_range_pct',
#     'open',
#     'dist_R1',
#     'dist_S1',
#     'minute_sin',
#     'volatility_50',
#     'minute_cos',
#     'dist_PP',
#     'e100',
#     'trend_slope_ema',
#     'trend_r2_100',
#     'min_dist_level_atr',
#     'dist_R1_8h_atr',
#     'avg_volume_last_n',
#     'dist_P_8h_atr',
#     'vol_std_m',
#     'overnight_range_atr',
#     'shock_ema',
#     'range_prev_day_atr',
#     'dist_S1_8h_atr',
#     'ret_kurt_m',
#     'ret_skew_m',
#     'e',
#     'z_atr_norm',
#     'corr_close_vol_m',
#     'dow_sin',
#     'trend_r2_50',
#     'k100',
#     'hurst_m',
#     'trend_dist_pct_100',
#     'var_ratio',
#     'up_vol_ratio',
#     'signed_volume_ema',
#     'delta_trend_slope_100',
#     'dow_cos',
#     'macd',
#     'vol_ratio',
#     'acf1_s',
#     'acf2_s',
#     'volatility_10',
#     'dist_support_rl',
#     'macd_hist',
#     'dist_resist_rl',
#     'momentum_diff',
#     'price_change',
#     'cross_ma_m',
#     'rsi_14',
#     'k',
#     'z_volume',
#     'volume',
#     'micro_slope',
#     'p_entropy_s',
#     'wick_imbalance',
#     'ret_abs',
#     'trend_acc',
#     'ret_diff2',
#     'lower_wick',
#     'upper_wick',
#     # 'return_1_raw',
#     'hmm_state',
#     # 'vwap_up',
#     'market_regime',
#     'trend_ema200_down',
#     # 'session_london_open',
#     'near_PP_pct',
#     'near_S1_pct',
#     'near_R1_pct',
# ]

# FEATURES = [
#     # Winrate: 56.56%
#     'VWAP', 'overnight_range_pct', 'open', 'dist_R1', 'dist_S1', 'minute_sin', 'volatility_50', 'minute_cos', 'dist_PP',
#     'e100', 'trend_slope_ema',
#     'trend_r2_100', 'overnight_range_atr',
#     'ret_kurt_m', 'corr_close_vol_m',
#     'trend_r2_50',
#     'momentum_diff', 'price_change',
#     'market_regime',
# ]

# Winrate: 48.25% correlation < 0.85
# FEATURES = [
#     'dist_S2', 'dist_R2', 'close', 'dist_PP', 'volatility_50', 'minute_sin', 'minute_cos',
#     'overnight_range_pct', 'dist_S2_8h_atr', 'avg_volume_last_n', 'dist_R2_8h_atr',
#     'range_prev_day_pct', 'volume_lag1',
#     'overnight_range_atr', 'dist_P_8h_atr', 'range_prev_day_atr', 'trend_channel_slope_low',
#     'dist_resist_rl', 'dist_support_rl', 'd', 'dow_sin',
#     'volume', 'dow_cos', 'macd_minus_signal', 'vol_ratio', 'z_atr_norm', 'min_dist_level_atr',
#     'delta_trend_slope_100', 'ret', 'ret_diff2', 'ret_skew_m', 'ret_kurt_m',
#     'acf1_s', 'acf2_s', 'p_entropy_s', 'hurst_m', 'vol_rel_m',
#     'vol_std_m', 'up_vol_ratio', 'corr_close_vol_m', 'cross_ma_m', 'market_regime', 'hmm_state',
#     'signed_volume_ema', 'upper_wick', 'lower_wick', 'wick_imbalance',
#     'micro_slope', 'shock_ema', 'trend_slope_ema', 'trend_r2_50', 'trend_acc'
# ]

# FEATS_M2 = ['p_hat', 'margin', 'direction_sign', 'equity_dd', 'win_streak_len', 'loss_streak_len', 'hit_rate_50']
FEATS_M2 = ['p_hat', 'margin', 'direction_sign']

warnings.filterwarnings('ignore')



# Test regime feature
# params_to_features = {
#     "version": "v3-testing-features-1-model2model   ",  # cambia esto si modificas reglas de features
# }

# Test regime feature
# params_to_features = {
#     "version": "tp_5_sl3",  # cambia esto si modificas reglas de features
# }

params_to_features = {
    "version": "tp_4_sl3",  # cambia esto si modificas reglas de features
    "lookahead": "60",  # cambia esto si modificas reglas de features
}

# params_to_features = {
#     "version": "tp_4_sl3",  # cambia esto si modificas reglas de features
#     "lookahead": "50",  # cambia esto si modificas reglas de features
# }

def feature_signature(params: dict) -> str:
    """
    Crea un hash estable con los parámetros/versión de tu pipeline de features.
    Ej: ventanas trend, lags, sesiones, umbrales, versión de código.
    """
    key = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def feature_cache_path(base_dir: str, sig: str) -> Path:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return Path(base_dir) / f"features_{sig}.parquet"


def prepare_features_incremental(
        raw_df: pd.DataFrame,
        preparar_features_fn,  # tu función actual
        params: dict,  # lags, trend_windows, sesiones, versión, etc.
        cache_dir: str = "feature_cache",
        datetime_col: str = "datetime",
        warmup_bars: int = 200  # ≈ max(lags, ventanas rolling, canal)
):
    """
    Devuelve (df_features, feature_list, cache_path).
    - Si hay cache y está alineado, actualiza sólo lo nuevo.
    - Si no hay cache, crea desde cero.
    """
    sig = feature_signature(params)
    path = feature_cache_path(cache_dir, sig)

    raw_df = raw_df.sort_values(datetime_col).reset_index(drop=True)

    if path.exists():
        # Carga cache existente
        cached = pd.read_parquet(path)
        # Verifica columnas mínimas
        _, expected_cols = preparar_features_fn(raw_df.iloc[:warmup_bars + 1].copy())
        cols_ok = all(c in cached.columns for c in expected_cols)
        # Último timestamp procesado
        last_ts = cached[datetime_col].iloc[-1]
        new_part = raw_df[raw_df[datetime_col] > last_ts]

        if not cols_ok or new_part.empty:
            # 1) Si cambió el esquema (nuevas columnas), re-procesa (parcial) desde last_ts - warmup
            # 2) Si no hay datos nuevos, retorna cache
            if new_part.empty and cols_ok:
                return cached, expected_cols, path
            # recalcular parcial
            start_idx = max(0, raw_df.index[raw_df[datetime_col] <= last_ts].max() - (warmup_bars - 1))
            slice_df = raw_df.iloc[start_idx:].copy()
            df_new, feats = preparar_features_fn(slice_df)
            # Empalmar: conserva filas <= last_ts del cache, luego concatena lo nuevo
            merged = pd.concat([
                cached[cached[datetime_col] < df_new[datetime_col].min()],
                df_new
            ], axis=0, ignore_index=True).drop_duplicates(subset=[datetime_col], keep="last")
            merged.to_parquet(path, index=False)
            return merged, feats, path
        else:
            # Hay datos nuevos y esquema ok: recalcular sólo la cola con warmup
            start_idx = max(0, raw_df.index[raw_df[datetime_col] <= last_ts].max() - (warmup_bars - 1))
            slice_df = raw_df.iloc[start_idx:].copy()
            df_new, feats = preparar_features_fn(slice_df)
            merged = pd.concat([
                cached[cached[datetime_col] < df_new[datetime_col].min()],
                df_new
            ], axis=0, ignore_index=True).drop_duplicates(subset=[datetime_col], keep="last")
            merged.to_parquet(path, index=False)
            return merged, feats, path
    else:
        # No hay cache: construir completo
        df_feat, feats = preparar_features_fn(raw_df.copy())
        _save_parquet_fallback(df_feat, path)
        return df_feat, feats, path


def _save_parquet_fallback(df, path):
    """
    Guarda df en parquet si hay engine, si no cae a CSV.
    """
    try:
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    except Exception as e1:
        try:
            df.to_parquet(path, index=False, engine="fastparquet", compression="snappy")
        except Exception as e2:
            warnings.warn(
                f"[feature_cache] No hay soporte Parquet (pyarrow/fastparquet). "
                f"Fallo con {e1} y {e2}. Guardando en CSV como fallback."
            )
            csv_path = str(path).replace(".parquet", ".csv")
            df.to_csv(csv_path, index=False)
            return csv_path
    return path


def _load_parquet_fallback(path):
    """
    Intenta leer parquet, si no existe o no hay engine, prueba CSV.
    """
    try:
        return pd.read_parquet(path)
    except Exception as e:
        csv_path = str(path).replace(".parquet", ".csv")
        if Path(csv_path).exists():
            warnings.warn(
                f"[feature_cache] No se pudo leer {path} ({e}). Usando fallback CSV {csv_path}"
            )
            return pd.read_csv(csv_path, parse_dates=["datetime"])
        else:
            raise


def add_sr_features(
        df: pd.DataFrame,
        datetime_col: str = "datetime",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        atr_window: int = 14,
        fractal_window: int = 20,
        near_pct: float = 0.005,  # 0.5% del precio
        near_k_atr: float = 0.5,  # 0.5 * ATR
        use_daily_pivots: bool = True,
        session_day_start: str | None = None,  # ej. "08:00"
        tz_for_session: str | None = None,  # ej. "America/Bogota"
) -> pd.DataFrame:
    """
    Agrega features de Soporte/Resistencia sin fuga de info.
    """
    df = df.copy()

    # -------- Validaciones mínimas --------
    for c in (datetime_col, high_col, low_col, close_col):
        if c not in df.columns:
            raise ValueError(f"Falta la columna requerida: '{c}'")

    # Asegurar datetime
    if not is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
    else:
        # Si es naive, localízala a UTC para tener base consistente
        if df[datetime_col].dt.tz is None:
            if tz_for_session:
                df[datetime_col] = df[datetime_col].dt.tz_localize(tz_for_session)
            else:
                df[datetime_col] = df[datetime_col].dt.tz_localize("UTC")

    # -------- Ordenar por tiempo --------
    df.sort_values(datetime_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # -------- ATR (si no existe) --------
    if "ATR" not in df.columns:
        h = df[high_col].astype(float)
        l = df[low_col].astype(float)
        c = df[close_col].astype(float)
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(),
                        (h - prev_c).abs(),
                        (l - prev_c).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(atr_window, min_periods=1).mean()

    # -------- Construcción de “día” para pivotes --------
    # Convertir a tz de sesión (si se pide)
    if tz_for_session:
        df[datetime_col] = df[datetime_col].dt.tz_convert(tz_for_session)

    # Día de sesión o día calendario
    if session_day_start:
        hh, mm = map(int, session_day_start.split(":"))
        shifted = df[datetime_col] - pd.to_timedelta(hh, unit="h") - pd.to_timedelta(mm, unit="m")
        day_key = shifted.dt.floor("D")
    else:
        day_key = df[datetime_col].dt.floor("D")

    df["_day"] = day_key

    # Clave numérica robusta para merges (evita problemas de tz)
    # Nota: view('int64') requiere dtype datetime64[ns, tz]; convertimos a UTC para robustez.
    _day_utc = df["_day"].dt.tz_convert("UTC") if df["_day"].dt.tz is not None else df["_day"].dt.tz_localize("UTC")
    df["_day_key"] = _day_utc.view("int64")

    # -------- Pivotes diarios clásicos del DÍA PREVIO --------
    if use_daily_pivots:
        daily = df.groupby("_day").agg(
            day_high=(high_col, "max"),
            day_low=(low_col, "min"),
            day_close=(close_col, "last"),
        )

        piv = pd.DataFrame(index=daily.index)
        piv["PP"] = (daily["day_high"] + daily["day_low"] + daily["day_close"]) / 3.0
        piv["R1"] = 2 * piv["PP"] - daily["day_low"]
        piv["S1"] = 2 * piv["PP"] - daily["day_high"]
        rng = (daily["day_high"] - daily["day_low"]).astype(float)
        piv["R2"] = piv["PP"] + rng
        piv["S2"] = piv["PP"] - rng

        # Pivotes del día previo
        piv_prev = piv.shift(1).reset_index(names="_day")

        # Clave numérica en el lado derecho también
        _day_right = (piv_prev["_day"].dt.tz_convert("UTC")
                      if piv_prev["_day"].dt.tz is not None
                      else piv_prev["_day"].dt.tz_localize("UTC"))
        piv_prev["_day_key"] = _day_right.view("int64")

        # Mantener solo lo necesario
        piv_prev = piv_prev[["_day_key", "PP", "R1", "S1", "R2", "S2"]]

        # ---- Merge robusto
        df = df.merge(piv_prev, on="_day_key", how="left")

        # Garantizar que existan las columnas aunque no matchee nada
        for lvl in ["PP", "S1", "R1", "S2", "R2"]:
            if lvl not in df.columns:
                df[lvl] = np.nan

        # Distancias relativas al close
        close = df[close_col].astype(float).replace(0, np.nan)
        eps = 1e-12
        for lvl in ["PP", "S1", "R1", "S2", "R2"]:
            df[f"dist_{lvl}"] = (df[lvl].astype(float) - close) / (close + eps)

        # Flags de cercanía por % y por ATR
        for lvl in ["PP", "S1", "R1", "S2", "R2"]:
            df[f"near_{lvl}_pct"] = (df[f"dist_{lvl}"].abs() <= float(near_pct)).astype("Int64").fillna(0).astype(int)
            df[f"near_{lvl}_atr"] = ((df[lvl].astype(float) - close).abs() <= float(near_k_atr) * df["ATR"]) \
                .astype("Int64").fillna(0).astype(int)

    # -------- Soporte/Resistencia rolling (fractales) --------
    df["resist_rl"] = df[high_col].astype(float).rolling(fractal_window, min_periods=1).max().shift(1)
    df["support_rl"] = df[low_col].astype(float).rolling(fractal_window, min_periods=1).min().shift(1)

    close = df[close_col].astype(float).replace(0, np.nan)
    df["dist_resist_rl"] = (df["resist_rl"] - close) / (close + 1e-12)
    df["dist_support_rl"] = (close - df["support_rl"]) / (close + 1e-12)

    df["near_resist_rl_pct"] = (df["dist_resist_rl"].abs() <= float(near_pct)).astype(int)
    df["near_support_rl_pct"] = (df["dist_support_rl"].abs() <= float(near_pct)).astype(int)

    df["near_resist_rl_atr"] = ((df["resist_rl"] - close).abs() <= float(near_k_atr) * df["ATR"]).astype(int)
    df["near_support_rl_atr"] = ((close - df["support_rl"]).abs() <= float(near_k_atr) * df["ATR"]).astype(int)

    # -------- Limpieza: rellenar NaNs iniciales / flags --------
    def _ffill_if_exists(cols):
        for c in cols:
            if c in df.columns:
                df[c] = df[c].ffill()

    cont_cols = ["PP", "S1", "R1", "S2", "R2", "resist_rl", "support_rl",
                 "dist_PP", "dist_S1", "dist_R1", "dist_S2", "dist_R2",
                 "dist_resist_rl", "dist_support_rl"]
    _ffill_if_exists(cont_cols)

    for c in df.columns:
        if c.startswith("near_"):
            df[c] = df[c].fillna(0).astype(int)

    # Quitar helper
    df.drop(columns=["_day_key"], errors="ignore", inplace=True)

    return df


def add_sr_features_8h(df: pd.DataFrame, freq: str = "8H",
                       near_k_atr: float = 0.25) -> pd.DataFrame:
    """
    Calcula pivots de 'sesiones' de 8h usando SOLO la sesión anterior (evita lookahead).
    Añade columnas: P_8h, R1_8h..R3_8h, S1_8h..S3_8h, dist_*_atr y flags near_res_8h / near_sup_8h.
    Robusto a tz / merges sin match.
    """
    out = df.copy()

    # --- Asegura datetime tz-aware y orden ---
    if "datetime" not in out.columns:
        raise ValueError("Se requiere columna 'datetime'.")
    if not pd.api.types.is_datetime64_any_dtype(out["datetime"]):
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    if getattr(out["datetime"].dt, "tz", None) is None:
        out["datetime"] = out["datetime"].dt.tz_localize("UTC")

    out = out.sort_values("datetime").reset_index(drop=True).set_index("datetime")

    # --- Resample 8h (OHLC) ---
    ses = out[["high", "low", "close"]].resample(freq).agg({"high": "max", "low": "min", "close": "last"})

    # --- Pivots de la SESIÓN ANTERIOR (shift(1)) ---
    prev = ses.shift(1).dropna()
    P = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    R1 = 2 * P - prev["low"]
    S1 = 2 * P - prev["high"]
    R2 = P + (prev["high"] - prev["low"])
    S2 = P - (prev["high"] - prev["low"])
    R3 = prev["high"] + 2 * (P - prev["low"])
    S3 = prev["low"] - 2 * (prev["high"] - P)

    piv = pd.DataFrame({
        "P_8h": P, "R1_8h": R1, "S1_8h": S1,
        "R2_8h": R2, "S2_8h": S2,
        "R3_8h": R3, "S3_8h": S3,
    }).rename_axis("session").reset_index()

    # --- Marca de sesión por vela y merge ---
    out = out.copy()
    out["session"] = out.index.floor(freq)

    # Alinea tz de 'session' en piv
    if getattr(piv["session"].dt, "tz", None) is None:
        piv["session"] = piv["session"].dt.tz_localize(out.index.tz)
    else:
        piv["session"] = piv["session"].dt.tz_convert(out.index.tz)

    out = out.reset_index().merge(piv, on="session", how="left", validate="many_to_one")

    # --- GARANTIZA EXISTENCIA de columnas de pivots (clave para evitar KeyError) ---
    pivot_cols = ["P_8h", "R1_8h", "S1_8h", "R2_8h", "S2_8h", "R3_8h", "S3_8h"]
    for c in pivot_cols:
        if c not in out.columns:
            out[c] = np.nan

    # --- ffill/bfill dentro de cada sesión (si hay alguna no-nula) ---
    if out[pivot_cols].isna().all(axis=None):
        # No hubo ninguna sesión previa con datos (al inicio del dataset); seguimos con NaN
        pass
    else:
        out[pivot_cols] = out.groupby("session", group_keys=False)[pivot_cols].apply(
            lambda g: g.ffill().bfill()
        )

    # --- ATR (si falta) y distancias normalizadas ---
    if "ATR" not in out.columns:
        tr1 = (out["high"] - out["low"])
        tr2 = (out["high"] - out["close"].shift(1)).abs()
        tr3 = (out["low"] - out["close"].shift(1)).abs()
        out["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    eps = 1e-12
    atr = out["ATR"].replace(0, eps).ffill()

    for lvl in pivot_cols:
        out[f"dist_{lvl}_atr"] = (out[lvl] - out["close"]).abs() / (atr + eps)

    # Flags de cercanía a R/S
    res_dists = out[[f"dist_{l}_atr" for l in ["R1_8h", "R2_8h", "R3_8h"]]].min(axis=1)
    sup_dists = out[[f"dist_{l}_atr" for l in ["S1_8h", "S2_8h", "S3_8h"]]].min(axis=1)
    out["near_res_8h"] = (res_dists <= near_k_atr).astype(int)
    out["near_sup_8h"] = (sup_dists <= near_k_atr).astype(int)

    # --- Limpieza y retorno ---
    out = out.drop(columns=["session"], errors="ignore").set_index("datetime").reset_index()
    return out


def add_stoch_extreme_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    k = out.get("STO_K") or out.get("%K") or out.get("k100")
    d = out.get("STO_D") or out.get("%D") or out.get("d100")
    if k is None or d is None:
        return out  # ya lo calculas en otra parte

    out["stoch_overbought_80"] = (k >= 80).astype(int)
    out["stoch_overbought_90"] = (k >= 90).astype(int)
    out["stoch_oversold_20"] = (k <= 20).astype(int)
    out["stoch_oversold_10"] = (k <= 10).astype(int)
    out["stoch_k_slope"] = k.diff()
    out["stoch_k_above_d"] = (k > d).astype(int)
    return out


def _trades_to_df(self) -> pd.DataFrame:
    """Convierte self.trades (lista de Trade) en DataFrame listo para estadísticas."""
    rows = []
    for t in self.trades:
        rows.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "side": getattr(t, "side", None),  # "long"/"short"
            "entry_price": getattr(t, "entry_price", np.nan),
            "exit_price": getattr(t, "exit_price", np.nan),
            "qty": getattr(t, "qty", np.nan),
            "leverage": getattr(t, "leverage", np.nan),
            "entry_pred": getattr(t, "entry_pred", None),
            "entry_proba": getattr(t, "entry_proba", np.nan),
            "exit_reason": getattr(t, "exit_reason", None),
            "fees": float(t.fees_usdt()) if hasattr(t, "fees_usdt") else np.nan,
            "pnl": float(t.pnl_usdt()) if hasattr(t, "pnl_usdt") else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pnl_net"] = df["pnl"] - df.get("fees", 0.0)
    return df


# === Multi-class Beta Calibration (uno-vs-rest) ===
class MultiClassBetaCalibrator:
    """
    Aplica BetaCalibration por clase (uno-vs-rest) y renormaliza.
    Úsalo con probabilidades (n, K). Requiere betacal instalado.
    """

    def __init__(self, parameters: str = "abm", classes_=None, label_encoder=None):
        if not _HAS_BETACAL:
            raise ImportError("Requiere betacal. Instala con: pip install betacal")
        self.parameters = parameters
        self.classes_ = None if classes_ is None else np.array(classes_)
        self.label_encoder = label_encoder
        self._per_class = []  # lista de BetaCalibration por clase

    def fit(self, proba: np.ndarray, y_idx: np.ndarray):
        """
        proba: (n, K) del modelo base en X_cal
        y_idx: (n,) enteros 0..K-1 (no one-hot)
        """
        n, K = proba.shape
        self._per_class = []
        for k in range(K):
            yk = (y_idx == k).astype(int)  # binario 0/1
            pk = proba[:, k].reshape(-1)  # vector 1D
            beta = BetaCalibration(parameters=self.parameters)
            beta.fit(pk, yk)  # <- binario correcto
            self._per_class.append(beta)
        self.n_classes_ = K
        return self

    def predict_proba(self, proba_raw: np.ndarray) -> np.ndarray:
        """
        proba_raw: (n, K) del modelo base en cualquier X
        Devuelve (n, K) calibrado y renormalizado.
        """
        n, K = proba_raw.shape
        out = np.zeros_like(proba_raw, dtype=float)
        for k in range(K):
            # BetaCalibration.predict_proba devuelve (n, 2); [:,1] es P(clase=1)
            pk = proba_raw[:, k].reshape(-1)
            pk_cal = self._per_class[k].predict(pk)
            out[:, k] = pk_cal

        # renormalización para sumar 1 por fila
        row_sum = out.sum(axis=1, keepdims=True)
        eps = 1e-12
        row_sum = np.where(row_sum < eps, eps, row_sum)
        out = out / row_sum
        return out


class BetaMultiWrapper:
    """
    Wrapper para exponer API similar a CalibratedClassifierCV pero con MultiClassBetaCalibrator.
    - Para RF/LGBM: classes_ son las de base_model.
    - Para XGB + LabelEncoder: devuelve etiquetas en texto.
    """

    def __init__(self, base_model, mc_beta: MultiClassBetaCalibrator, classes_, label_encoder=None):
        self.base_model = base_model
        self.mc_beta = mc_beta
        self.le = label_encoder
        self.classes_ = np.array(classes_) if label_encoder is None else label_encoder.classes_

    def predict_proba(self, X):
        proba_raw = self.base_model.predict_proba(X)
        return self.mc_beta.predict_proba(proba_raw)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        if self.le is not None:
            return self.le.inverse_transform(idx)
        return self.classes_[idx]


class BetaWrapper:
    """Calibrador beta para RF/LGBM con etiquetas en texto."""

    def __init__(self, base_model, beta_cal, classes_):
        self.base_model = base_model
        self.beta_cal = beta_cal
        self.classes_ = np.array(classes_)

    def predict_proba(self, X):
        proba_raw = self.base_model.predict_proba(X)
        return self.beta_cal.predict_proba(proba_raw)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


class LabeledBetaWrapper:
    """Calibrador beta para XGB entrenado con y codificado (0..K-1), devolviendo etiquetas en texto."""

    def __init__(self, base_model, beta_cal, label_encoder):
        self.base_model = base_model
        self.beta_cal = beta_cal
        self.le = label_encoder
        self.classes_ = self.le.classes_

    def predict_proba(self, X):
        proba_raw = self.base_model.predict_proba(X)
        return self.beta_cal.predict_proba(proba_raw)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.le.inverse_transform(idx)


@dataclass
class Fees:
    """Estructura de comisiones. Todas en proporción (p.ej., 0.00055 = 0.055%)."""
    taker_open: float = 0.00055
    taker_close: float = 0.00055
    maker_open: float = 0.0002
    maker_close: float = 0.0002


@dataclass
class Trade:
    side: Literal["long", "short"]
    entry_time: pd.Timestamp
    entry_price: float
    tp_price: float
    sl_price: float
    qty: float  # contratos (para lineales USDT: qty = notional / price)
    leverage: float
    opened_with: Literal["taker", "maker"] = "taker"
    closed_with: Literal["taker", "maker"] = "taker"
    entry_index: Optional[int] = None

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[Literal["tp", "sl", "close", "early_opposite"]] = None
    entry_proba: Optional[float] = None  # 0..1
    entry_pred: Optional[str] = None  # 'bullish' | 'bearish' | 'lateral'

    def is_open(self) -> bool:
        return self.exit_time is None

    def pnl_usdt(self) -> float:
        if self.exit_price is None:
            return 0.0
        # Lineal USDT: PnL = (Px_exit - Px_entry) * qty (para long), invertido para short
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.exit_price) * self.qty

    def fees_usdt(self) -> float:
        # Comisiones sobre el nocional (precio * qty)
        notional_entry = self.entry_price * self.qty
        notional_exit = (self.exit_price if self.exit_price is not None else self.entry_price) * self.qty
        open_fee = notional_entry * (self.opened_with == "taker") * fees.taker_open + \
                   notional_entry * (self.opened_with == "maker") * fees.maker_open
        close_fee = notional_exit * (self.closed_with == "taker") * fees.taker_close + \
                    notional_exit * (self.closed_with == "maker") * fees.maker_close
        return float(open_fee + close_fee)


# Variable global simple para que Trade.fees_usdt tenga acceso a la instancia Fees
fees = Fees()

@dataclass
class CacheConfig:
    cache_path: str = "m1_outputs.parquet"
    engine: str = "pyarrow"  # recomendado
    partition_cols: Optional[list] = None  # normalmente None; puedes añadir 'date' si quieres

class M1OutputCacheBuilder:
    """
    Genera un dataset incremental con:
      - Salida del Modelo 1 por vela (probabilidades, predicción, p_hat, margin, tp/sl)
      - Resultado hipotético (tp/sl/close) evaluado con barras futuras
      - Performance features LAGGED (sin fuga), en base a resultados pasados
    Entrena el modelo una sola vez con histórico previo a `test_start`.
    """

    def __init__(
        self,
        analyzer,
        candles: pd.DataFrame,
        proba_threshold: float = 0.55,       # ÚNICO umbral para "operadas"
        atr_period: int = 14,
        atr_mult_tp: float = 4.0,
        atr_mult_sl: float = 3.0,
        use_adapter_tp_sl_when_missing: bool = True,
        order_conflict_policy: Literal["worst", "best", "tp_first", "sl_first"] = "sl_first",
        offline_lookback_bars: int = 200,
        n_keep_for_fit: int = 40000,
        cache: CacheConfig = CacheConfig(),
    ):
        self.analyzer = analyzer
        self.df = candles.reset_index(drop=True).copy()
        # Normaliza datetime a tz existente o UTC
        if pd.api.types.is_datetime64_any_dtype(self.df["datetime"]):
            if getattr(self.df["datetime"].dt, "tz", None) is None:
                self.df["datetime"] = self.df["datetime"].dt.tz_localize("UTC")
        else:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], utc=True)
        self.tz = self.df["datetime"].dt.tz

        self.proba_threshold = float(proba_threshold)
        self.atr_period = int(atr_period)
        self.atr_mult_tp = float(atr_mult_tp)
        self.atr_mult_sl = float(atr_mult_sl)
        self.use_adapter_tp_sl_when_missing = bool(use_adapter_tp_sl_when_missing)
        self.order_conflict_policy = order_conflict_policy
        self.offline_lookback_bars = int(offline_lookback_bars)
        self.n_keep_for_fit = int(n_keep_for_fit)
        self.cache = cache

        # Precalcula ATR simple para fallback de TP/SL
        self.df["tr"] = self._true_range(self.df)
        self.df["atr"] = self.df["tr"].rolling(self.atr_period).mean()

    # --------------------------
    # Entrenamiento (una vez)
    # --------------------------
    def _fit_once(self, train_df: pd.DataFrame, suppress_fit_stdout: bool = True):
        self.analyzer.data = train_df.copy()
        N_KEEP = min(len(train_df), self.n_keep_for_fit)
        if suppress_fit_stdout:
            import io, contextlib
            buf = io.StringIO()
            # with contextlib.redirect_stdout(buf):
            #     self.analyzer.create_predictive_model(
            #         model_type="lgbm",
            #         calibrate_method="beta",
            #         N_KEEP=N_KEEP,
            #         display_graphic=False,
            #         model_kwargs={
            #             "objective": "multiclass",
            #             "num_class": 3,  # ajusta según tus clases reales
            #             "n_estimators": 4000,  # más árboles + LR más bajo
            #             "learning_rate": 0.03,  # más bajo para mejorar logloss
            #             "num_leaves": 96,  # complejidad del árbol
            #             "max_depth": -1,  # deja que LGBM lo maneje
            #             "min_child_samples": 200,  # regularización clave con 220k filas
            #
            #             "subsample": 0.7,  # antes 0.9, bajamos para más robustez
            #             "subsample_freq": 1,
            #             "colsample_bytree": 0.7,  # menos columnas por árbol, menos overfit
            #
            #             "reg_lambda": 15.0,  # L2 fuerte → baja variancia, mejora logloss
            #             "reg_alpha": 0.1,  # L1 ligera
            #
            #             "min_split_gain": 0.0,
            #             "random_state": 42,
            #             "n_jobs": -1
            #         },
            #         display_feature_plot=False
            #     )
        else:
            self.analyzer.create_predictive_model(
                    model_type="lgbm",
                    calibrate_method="sigmoid",
                    N_KEEP=N_KEEP,
                    display_graphic=False,
                    model_kwargs={
                        "objective": "multiclass",
                        "num_class": 3,  # ajusta según tus clases reales
                        "n_estimators": 1000,  # más árboles + LR más bajo
                        "learning_rate": 0.03,  # más bajo para mejorar logloss
                        "num_leaves": 96,  # complejidad del árbol
                        "max_depth": -1,  # deja que LGBM lo maneje
                        "min_child_samples": 200,  # regularización clave con 220k filas

                        "subsample": 0.7,  # antes 0.9, bajamos para más robustez
                        "subsample_freq": 1,
                        "colsample_bytree": 0.7,  # menos columnas por árbol, menos overfit

                        "reg_lambda": 15.0,  # L2 fuerte → baja variancia, mejora logloss
                        "reg_alpha": 0.1,  # L1 ligera

                        "min_split_gain": 0.0,
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    display_feature_plot=False
                )



    # --------------------------
    # Señal offline (M1)
    # --------------------------
    def _offline_predict_at_index(self, i: int) -> Optional[Dict]:
        start_i = max(0, i - self.offline_lookback_bars + 1)
        # df_win = self.df.loc[start_i:i, ["datetime", "open", "high", "low", "close", "volume"]].copy()
        df_ready = self.df.loc[start_i:i]

        features = FEATURES
        if df_ready.empty:
            return None
        last_row_feat = df_ready[features].iloc[-1:]
        last_meta = df_ready.iloc[-1]

        last_scaled = self.analyzer.scaler.transform(last_row_feat)
        proba_vec = self.analyzer.model.predict_proba(last_scaled)[0]
        classes = list(self.analyzer.model.classes_)
        probs = {cls: float(p) for cls, p in zip(classes, proba_vec)}

        # Clase predicha (usa helper si existe)
        if hasattr(self.analyzer, "_apply_thresholds"):
            pred_class = self.analyzer._apply_thresholds(
                proba_vec, classes, {"bearish": self.proba_threshold, "bullish": self.proba_threshold, "lateral": 0.0},
                default_class="lateral", global_floor=0.0, min_margin=0.0
            )
        else:
            pred_class = classes[int(np.argmax(proba_vec))]

        return {
            "prediction": pred_class,
            "probabilities": probs,
            "timestamp": last_meta["datetime"],
            "close": float(last_meta["close"]),
            "atr_here": float(last_meta.get("ATR", np.nan)),
        }

    # --------------------------
    # TP/SL & resolución de barras
    # --------------------------
    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        hl = df["high"] - df["low"]
        hc = (df["high"] - prev_close).abs()
        lc = (df["low"] - prev_close).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    def _tp_sl_from_adapter_or_atr(
        self, i: int, side: Literal["long", "short"], maybe_tp: Optional[float], maybe_sl: Optional[float]
    ) -> Tuple[float, float]:
        px = float(self.df.loc[i, "close"])
        if self.use_adapter_tp_sl_when_missing and (maybe_tp is not None) and (maybe_sl is not None):
            return float(maybe_tp), float(maybe_sl)

        atr = float(self.df.loc[i, "ATR"])
        if np.isnan(atr) or atr <= 0:
            atr = max(px * 0.005, 1e-8)

        if side == "long":
            tp = px + self.atr_mult_tp * atr
            sl = px - self.atr_mult_sl * atr
        else:
            tp = px - self.atr_mult_tp * atr
            sl = px + self.atr_mult_sl * atr
        return tp, sl

    def _apply_conflict_policy(
        self, bar_high: float, bar_low: float, tp: float, sl: float, side: Literal["long","short"]
    ) -> Optional[str]:
        hit_tp = bar_high >= tp if side == "long" else bar_low <= tp
        hit_sl = bar_low <= sl if side == "long" else bar_high >= sl
        if hit_tp and hit_sl:
            if self.order_conflict_policy == "tp_first":
                return "tp"
            elif self.order_conflict_policy == "sl_first":
                return "sl"
            elif self.order_conflict_policy == "best":
                return "tp"
            else:
                return "sl"
        elif hit_tp:
            return "tp"
        elif hit_sl:
            return "sl"
        return None

    def _resolve_outcome_forward(self, entry_idx: int, side: Literal["long","short"], tp: float, sl: float) -> Tuple[str, float]:
        """
        Barre barras posteriores hasta encontrar TP o SL.
        Devuelve (exit_reason, r_multiple), donde r_multiple se calcula con riesgo por barra de entrada.
        """
        entry_px = float(self.df.loc[entry_idx, "close"])
        risk = abs(entry_px - sl)
        risk = max(risk, 1e-12)

        for j in range(entry_idx + 1, len(self.df)):
            bar = self.df.loc[j]
            decision = self._apply_conflict_policy(bar.high, bar.low, tp, sl, side)
            if decision is not None:
                if side == "long":
                    pnl = (tp - entry_px) if decision == "tp" else (sl - entry_px)
                else:
                    pnl = (entry_px - tp) if decision == "tp" else (entry_px - sl)
                r_mult = pnl / risk
                return decision, float(r_mult)

        # Si no hubo toque hasta el final del dataset, cerramos a mercado (marca 'close')
        last_px = float(self.df.iloc[-1]["close"])
        pnl = (last_px - entry_px) if side == "long" else (entry_px - last_px)
        r_mult = pnl / risk
        return "close", float(r_mult)

    # --------------------------
    # Performance features (lagged)
    # --------------------------
    @staticmethod
    def _streak(series: pd.Series, target: str) -> int:
        best = cur = 0
        for x in series:
            if x == target:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def _build_performance_features(self, dfm: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe el dataframe por-señal con columnas:
          - 'is_operated' (bool en función de proba_threshold y dirección válida)
          - 'result' ∈ {'tp','sl','close'}
          - 'direction' ∈ {'bullish','bearish', None}
          - 'pnl_R' (float)
        Devuelve el mismo df con columnas LAGGED:
          - hit_rate_50, hit_rate_200
          - hit_rate_50_dir
          - win_streak_len, loss_streak_len
          - equity_dd
        """
        out = dfm.copy()

        # 1) Señales operadas (para performance)
        op = out["is_operated"].fillna(False)

        # 2) Acierto como 1/0 en operadas (tp cuenta como 1; 'close' se cuenta 1 si pnl_R>0)
        win_flag = np.where(out["result"].eq("tp"), 1,
                            np.where(out["result"].eq("sl"), 0,
                                     np.where(out["pnl_R"].fillna(0) > 0, 1, 0)))
        win_flag = pd.Series(win_flag, index=out.index).where(op, np.nan)

        # 3) Hit rates rolling (lagged)
        for W in (50, 200):
            out[f"hit_rate_{W}"] = (
                win_flag.rolling(W, min_periods=max(5, W//5)).mean().shift(1)
            )

        # 4) Por dirección
        dir_map = out["direction_sign"].fillna("")
        for dname in (1.0, -1.0):
            m = op & dir_map.eq(dname)
            wf_dir = pd.Series(np.where(m, win_flag, np.nan), index=out.index)
            out[f"hit_rate_50_{dname}"] = wf_dir.rolling(50, min_periods=10).mean().shift(1)

        # 5) Equity y drawdown (sobre operadas)
        pnl_series = pd.Series(np.where(op, out["pnl_R"].fillna(0.0), np.nan), index=out.index)
        eq = pnl_series.fillna(0.0).cumsum()
        # drawdown con shift(1) para no mirar el resultado actual
        peak = eq.shift(1).cummax()
        dd = (eq.shift(1) - peak) / (peak.replace(0, np.nan))
        out["equity_dd"] = dd.fillna(0.0)

        # 6) Rachas (lagged) — computamos retrospectivo y luego desplazamos
        res_seq = out["result"].astype(str).where(op, np.nan)
        wins_so_far = []
        losses_so_far = []
        cur_w = cur_l = 0
        for val in res_seq.fillna("NA"):
            if val == "tp":
                cur_w += 1
                cur_l = 0
            elif val == "sl":
                cur_l += 1
                cur_w = 0
            else:
                # señales no operadas o 'close' no cuentan en racha
                cur_w = cur_w
                cur_l = cur_l
            wins_so_far.append(cur_w)
            losses_so_far.append(cur_l)

        out["win_streak_len"] = pd.Series(wins_so_far, index=out.index).shift(1).fillna(0).astype(int)
        out["loss_streak_len"] = pd.Series(losses_so_far, index=out.index).shift(1).fillna(0).astype(int)

        return out

    # --------------------------
    # Proceso principal + cache
    # --------------------------
    def build_and_cache(self, test_start: str, test_end: Optional[str] = None, verbose: int = 1, export_parquet: bool = True, skip_fit: bool = False) -> pd.DataFrame:
        # Alinear fechas a tz del DF
        tz = self.tz
        start = pd.to_datetime(test_start)
        start = start.tz_localize(tz) if start.tzinfo is None else start.tz_convert(tz)

        end = None
        if test_end is not None:
            end = pd.to_datetime(test_end)
            end = end.tz_localize(tz) if end.tzinfo is None else end.tz_convert(tz)

        print("preparando datos")
        self.df, features, _ = prepare_features_incremental(
            raw_df=self.df,
            preparar_features_fn=self.analyzer.preparar_features,  # tu propia función
            params=params_to_features,
            cache_dir="feature_cache",
            datetime_col="datetime",
            warmup_bars=150
        )
        # self.analyzer.drop_highly_correlated_features(self.df[FEATURES], threshold=0.65)
        # self.analyzer.drop_highly_correlated_features(self.df, threshold=0.9)
        # >>> NUEVO: guarda la lista de features a copiar
        self._prepared_feature_cols = list(features)  # o usa tu lista 'features_final' si prefieres
        print("datos listos")

        mask = self.df["datetime"].between(start, end) if end is not None else (self.df["datetime"] >= start)
        idxs = self.df.index[mask].tolist()
        if not idxs:
            raise ValueError("Rango de test vacío. Revisa las fechas.")




        # Entrenar UNA VEZ con histórico previo a test_start
        first_i = idxs[0]
        train_slice = self.df.loc[: first_i - 1]
        if len(train_slice) < 200:
            raise ValueError("Histórico previo insuficiente para entrenar antes del test (>=200 velas).")
        
        if not skip_fit:
            self._fit_once(train_slice, suppress_fit_stdout=False)

        # Lee cache y define desde dónde continuar
        existing = None
        last_dt = None
        try:
            existing = pd.read_parquet(self.cache.cache_path, engine=self.cache.engine)
            if not existing.empty:
                last_dt = existing["datetime"].max()
                # filtra idxs para procesar solo posteriores
                if pd.notna(last_dt):
                    idxs = [i for i in idxs if self.df.loc[i, "datetime"] > last_dt]
        except Exception:
            existing = None

        rows: List[dict] = []

        for pos, i in enumerate(idxs):
            sig = self._offline_predict_at_index(i)
            if not sig:
                continue

            probs = sig["probabilities"]
            pred = sig["prediction"]
            ts = self.df.loc[i, "datetime"]

            p_bull = float(probs.get("bullish", np.nan))
            p_bear = float(probs.get("bearish", np.nan))
            p_lat = float(probs.get("lateral", np.nan))

            triples = [("bullish", p_bull), ("bearish", p_bear), ("lateral", p_lat)]
            triples = [(c, p) for c, p in triples if not np.isnan(p)]
            if not triples:
                continue
            triples.sort(key=lambda x: x[1], reverse=True)
            top1, p1 = triples[0]
            p2 = triples[1][1] if len(triples) > 1 else 0.0
            margin = float(p1 - p2)
            p_hat = float(p1)

            direction = pred if pred in ("bullish", "bearish") else None
            if direction is None:
                # opcional: si quieres guardar mínimos campos de M1 para auditoría, puedes
                # construir un row con is_operated=False, result=np.nan, y NO hacer forward-scan.
                # Pero para el parquet de M2, lo más limpio es continuar:
                continue

            tp = sl = np.nan
            result = None
            pnl_R = np.nan
            if direction is not None:
                # side_str para funciones que esperan Literal["long", "short"]
                side_str = cast(Literal["long", "short"], "long" if direction == "bullish" else "short")
                # direction_sign para el valor numérico
                direction_sign = 1.0 if direction == "bullish" else -1.0
                tp, sl = self._tp_sl_from_adapter_or_atr(i, side_str, maybe_tp=None, maybe_sl=None)
                result, pnl_R = self._resolve_outcome_forward(i, side_str, tp, sl)

            # ====== NUEVO: copiar features preparados de self.df en este índice ======
            # Elige qué columnas quieres conservar:
            feature_cols_to_keep = self._prepared_feature_cols  # o tu lista 'features_final' si ya la tienes
            # Toma los valores de ese índice (solo esas columnas y las que existan)
            cols_presentes = [c for c in feature_cols_to_keep if c in self.df.columns]
            feat_values = self.df.loc[i, cols_presentes].to_dict()

            row = {
                "datetime": ts,
                "close": float(self.df.loc[i, "close"]),
                "prediction": pred,
                "proba_bullish": p_bull,
                "proba_bearish": p_bear,
                "proba_lateral": p_lat,
                "p_hat": p_hat,
                "second_p": float(p2),
                "margin": margin,
                "direction": direction,
                "direction_sign": direction_sign if direction is not None else np.nan,
                "tp": float(tp) if not np.isnan(tp) else np.nan,
                "sl": float(sl) if not np.isnan(sl) else np.nan,
                "result": result,
                "pnl_R": float(pnl_R) if pd.notna(pnl_R) else np.nan,
            }

            # Mezcla: los valores de features NO pisan tus claves anteriores excepto 'close' si ya está en features.
            # Si quieres que prevalezca el 'close' del row, elimina 'close' de feat_values:
            feat_values.pop("close", None)

            # Fusiona
            row.update(feat_values)

            # Flag de operación (para performance features)
            is_operated = (direction is not None) and (p_hat >= self.proba_threshold)
            row["is_operated"] = bool(is_operated)

            rows.append(row)

            if verbose and ((pos + 1) % 500 == 0 or (pos + 1) == len(idxs)):
                print(f"[M1Output] {pos + 1}/{len(idxs)} | {ts} | pred={pred} p_hat={p_hat:.3f} margin={margin:.3f}")

        if not rows and existing is not None:
            return existing.sort_values("datetime").reset_index(drop=True)

        new_df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

        # ====== (Opcional) limpieza robusta en features copiados ======
        # Reemplaza inf por NaN solo en columnas numéricas
        num_cols = [c for c in new_df.columns if pd.api.types.is_numeric_dtype(new_df[c])]
        new_df[num_cols] = new_df[num_cols].replace([np.inf, -np.inf], np.nan)

        # Construye performance features LAGGED (esto añade columnas, no borra las existentes)
        new_df = self._build_performance_features(new_df)

        # ====== Concatena alineando columnas ======
        if existing is not None and not existing.empty:
            # Unión de columnas; pandas alinea por nombre y rellena faltantes con NaN
            combo = pd.concat([existing, new_df], ignore_index=True, sort=False)
            combo = combo.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        else:
            combo = new_df

        # Escritura parquet
        if export_parquet:
            if self.cache is None:
                if verbose > 0:
                    print("[WARN] export_parquet=True pero no hay CacheConfig (self.cache is None). Se omite guardado.")
            else:
                write_kwargs = {}
                if self.cache.partition_cols:
                    write_kwargs["partition_cols"] = self.cache.partition_cols

                combo.to_parquet(self.cache.cache_path, engine=self.cache.engine, index=False, **write_kwargs)

        return combo.sort_values("datetime").reset_index(drop=True)

class TradeSimulator:
    """
    Simulador walk-forward para una clase externa `StochasticAnalyzer` del usuario.

    Flujo (v1.2):
      1) Entrena **una sola vez** con los datos **previos al inicio del período de test** (por defecto).
         *Opcional:* si `retrain_each_bar=True`, reentrena en cada vela.
      2) Para cada vela del rango de test, obtiene la señal. Por defecto usa un **predictor offline**
         (sin tocar el API) que toma una ventana de `offline_lookback_bars` y replica tu `predict_realtime`
         usando únicamente el dataset local.
      3) Si aprueba validaciones estilo `monitor_predictions`, abre trade.
      4) Cierra por TP/SL usando high/low y aplica fees. Puedes forzar que el cierre no pase de `test_end`.
    """

    def __init__(
            self,
            analyzer,  # instancia de StochasticAnalyzer (ya creada por el usuario)
            candles: pd.DataFrame,
            symbol: str = "AVAXUSDT",
            interval: str = "5",
            initial_equity: float = 1000.0,
            leverage: float = 10.0,
            risk_fraction: float = 0.01,
            bybit_fees: Fees = Fees(),
            proba_threshold_day: float = 0.56,
            proba_threshold_night: float = 0.60,
            day_hours: Tuple[dtime, dtime] = (dtime(8, 0), dtime(20, 0)),
            order_conflict_policy: Literal["worst", "best", "tp_first", "sl_first"] = "sl_first",
            position_mode: Literal["fixed_risk", "fixed_notional"] = "fixed_risk",
            fixed_notional_usdt: float = 100.0,
            use_adapter_tp_sl_when_missing: bool = True,
            atr_period: int = 14,
            atr_mult_tp: float = 4,
            atr_mult_sl: float = 2.7,
            retrain_each_bar: bool = False,
            retrain_every_bars: int | None = None,
            suppress_fit_stdout: bool = False,
            suppress_predict_stdout: bool = True,
            use_offline_predictor: bool = True,
            offline_lookback_bars: int = 200,
            thresholds: Optional[Dict[str, float]] = None,
            early_exit_on_opposite_signal: bool = True,
            early_exit_threshold: float = 0.60,
            n_keep_for_fit: int = 40000,
            close_trades_at_test_end: bool = True,
            on_event: Optional[Callable[[str, dict], None]] = None,
            verbose: int = 1,
            log_every_bars: int = 250,
            use_tqdm: bool = False,
            m2_model=None,
            n_estimators=700,
            start_date="2024-01-01",
            end_date="",
            enable_withdrawals: bool = True,  # Nuevo parámetro
            m2_df: pd.DataFrame = None
    ):
        assert {"datetime", "open", "high", "low", "close"}.issubset(candles.columns), \
            "El DataFrame de velas debe contener: datetime, open, high, low, close"
        self.analyzer = analyzer
        self.df = candles.reset_index(drop=True).copy()
        self.m2_model = m2_model
        self.m2_df = m2_df
        # self.m2_model_bullish = m2_model_bullish

        if not is_datetime64_any_dtype(self.df["datetime"]):
            # Si viniera como string/epoch, parsea y asume UTC (luego podrás convertir)
            self.df["datetime"] = pd.to_datetime(self.df["datetime"], utc=True)

        # Si quedó naive (sin tz), localiza como UTC (tu pipeline ya la convierte a Bogota antes)
        if getattr(self.df["datetime"].dt, "tz", None) is None:
            self.df["datetime"] = self.df["datetime"].dt.tz_localize("UTC")

        # Guarda la tz para usarla en run()
        self.tz = self.df["datetime"].dt.tz

        self.symbol = symbol
        self.interval = interval
        self.equity = float(initial_equity)
        self.leverage = float(leverage)
        self.risk_fraction = float(risk_fraction)
        self.fees = bybit_fees
        self.proba_threshold_day = proba_threshold_day
        self.proba_threshold_night = proba_threshold_night
        self.day_hours = day_hours
        self.order_conflict_policy = order_conflict_policy
        self.position_mode = position_mode
        self.fixed_notional_usdt = fixed_notional_usdt
        self.use_adapter_tp_sl_when_missing = use_adapter_tp_sl_when_missing
        self.atr_period = int(atr_period)
        self.atr_mult_tp = float(atr_mult_tp)
        self.atr_mult_sl = float(atr_mult_sl)
        self.retrain_each_bar = bool(retrain_each_bar)
        self.retrain_every_bars = int(retrain_every_bars) if retrain_every_bars else None
        self.suppress_fit_stdout = bool(suppress_fit_stdout)
        self.suppress_predict_stdout = bool(suppress_predict_stdout)
        self.use_offline_predictor = bool(use_offline_predictor)
        self.offline_lookback_bars = int(offline_lookback_bars)
        self.thresholds = thresholds or {"bearish": 0.55, "bullish": 0.55, "lateral": 0.40}
        self.early_exit_on_opposite_signal = bool(early_exit_on_opposite_signal)
        self.early_exit_threshold = float(early_exit_threshold)
        self.n_keep_for_fit = int(n_keep_for_fit)
        self.close_trades_at_test_end = bool(close_trades_at_test_end)
        # Verbosidad y callback de eventos
        self.on_event = on_event
        self.verbose = int(verbose)
        self.log_every_bars = int(log_every_bars)
        self.use_tqdm = bool(use_tqdm)
        self.n_estimators = n_estimators
        self.start_date = start_date
        self.end_date = end_date
        self.enable_withdrawals = enable_withdrawals  # Guardar parámetro

        global fees
        fees = bybit_fees

        # # ATR para fallback TP/SL
        # self.df["tr"] = self._true_range(self.df)
        # self.df["atr"] = self.df["tr"].rolling(self.atr_period).mean()

        self.trades: List[Trade] = []
        self.logs: List[str] = []
        
        # Sistema de retiros automáticos
        self.initial_equity_for_withdrawals = float(initial_equity)
        self.withdrawn_capital = 0.0  # Capital total retirado
        self.last_withdrawal_target = float(initial_equity) * 2.0  # Próximo objetivo para retiro
        self.withdrawals: List[Dict] = []  # Historial de retiros
        # Marcadores de resultado en la vela de entrada
        if "entry_flag" not in self.df.columns:
            self.df["entry_flag"] = False  # True si en esa vela se abrió un trade
        if "entry_win" not in self.df.columns:
            self.df["entry_win"] = False  # True si ese trade terminó con profit
        if "entry_result" not in self.df.columns:
            self.df["entry_result"] = pd.Series(pd.NA, index=self.df.index, dtype="object")  # 'win'/'loss'/'breakeven'
        if "entry_exit_time" not in self.df.columns:
            self.df["entry_exit_time"] = pd.NaT
        if "entry_exit_reason" not in self.df.columns:
            self.df["entry_exit_reason"] = pd.Series(pd.NA, index=self.df.index, dtype="object")
        if "entry_pnl" not in self.df.columns:
            self.df["entry_pnl"] = np.nan
        if "entry_side" not in self.df.columns:
            # 'long' o 'short' como texto (útil para debug)
            self.df["entry_side"] = pd.Series(pd.NA, index=self.df.index, dtype="object")

        if "entry_pred" not in self.df.columns:
            # 'bullish' o 'bearish' según la dirección, para segmentar reglas
            self.df["entry_pred"] = pd.Series(pd.NA, index=self.df.index, dtype="object")

        self.rule_stats = defaultdict(lambda: {
            "side": None, "used": 0, "wins": 0, "losses": 0, "breakeven": 0, "pnl_sum": 0.0
        })

        # columnas auxiliares para rastrear la regla de entrada
        for col, dtype in [
            ("entry_rule", "object"),
            ("entry_rule_side", "object"),
        ]:
            if col not in self.df.columns:
                self.df[col] = pd.Series(pd.NA, index=self.df.index, dtype=dtype)

    # --------------------------
    # ADAPTERS / INTEGRACIONES
    # --------------------------

    def fit_model(self, train_df: pd.DataFrame):
        """Entrena el modelo del usuario **solo** con `train_df`.
        Seteamos `analyzer.data = train_df` y llamamos `create_predictive_model` con N_KEEP capado.
        """
        self.analyzer.data = train_df.copy()
        N_KEEP = min(len(train_df), self.n_keep_for_fit)
        if self.suppress_fit_stdout:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                # # RF con beta
                # self.analyzer.create_predictive_model(model_type="rf", calibrate_method="beta", N_KEEP=N_KEEP,
                #                                       model_kwargs={
                #                                           "n_estimators": self.n_estimators,
                #                                           "max_features": "sqrt",
                #                                           "n_jobs": -1,
                #                                           "random_state": 42
                #                                       })
                #
                # # LGBM con beta
                self.analyzer.create_predictive_model(
                    model_type="lgbm",
                    calibrate_method="beta",
                    N_KEEP=N_KEEP,
                    display_graphic=False,
                    model_kwargs={
                        "objective": "multiclass",
                        "num_class": 3,  # ajusta según tus clases reales
                        "n_estimators": self.n_estimators,  # más árboles + LR más bajo
                        "learning_rate": 0.03,  # más bajo para mejorar logloss
                        "num_leaves": 96,  # complejidad del árbol
                        "max_depth": -1,  # deja que LGBM lo maneje
                        "min_child_samples": 200,  # regularización clave con 220k filas

                        "subsample": 0.7,  # antes 0.9, bajamos para más robustez
                        "subsample_freq": 1,
                        "colsample_bytree": 0.7,  # menos columnas por árbol, menos overfit

                        "reg_lambda": 15.0,  # L2 fuerte → baja variancia, mejora logloss
                        "reg_alpha": 0.1,  # L1 ligera

                        "min_split_gain": 0.0,
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    display_feature_plot=False,
                    use_df_parquet=True
                )

                # XGB con beta (usa LabelEncoder interno)
                # self.analyzer.create_predictive_model(model_type="xgb", calibrate_method="sigmoid",
                #                                       model_kwargs={
                #                                           "n_estimators": 700,
                #                                           "learning_rate": 0.05,
                #                                           "subsample": 0.9,
                #                                           "colsample_bytree": 0.9,
                #                                           "max_depth": 8,
                #                                           "random_state": 42,
                #                                           "n_jobs": -1,
                #                                           "tree_method": "hist",
                #                                           "objective": "multi:softprob"
                #                                       })
                # self.analyzer.create_predictive_model(model_type="xgb", calibrate_method="beta", N_KEEP=N_KEEP,
                #                                       model_kwargs={
                #                                           "n_estimators": 700,
                #                                           "learning_rate": 0.05,
                #                                           "subsample": 0.9,
                #                                           "colsample_bytree": 0.9,
                #                                           "max_depth": 8,
                #                                           "random_state": 42,
                #                                           "n_jobs": -1,
                #                                           "tree_method": "hist",
                #                                           "objective": "multi:softprob"
                #                                       })
        else:

            # RF con beta # isotonic, sigmoid, beta
            # self.analyzer.create_predictive_model(model_type="rf", calibrate_method="sigmoid", N_KEEP=N_KEEP,
            #                                       model_kwargs={
            #                                           "n_estimators": self.n_estimators,
            #                                           "max_features": "sqrt",
            #                                           "n_jobs": -1,
            #                                           "random_state": 42
            #                                       })

            #
            self.analyzer.create_predictive_model(
                    model_type="lgbm",
                    calibrate_method="sigmoid",
                    N_KEEP=N_KEEP,
                    display_graphic=False,
                    model_kwargs={
                        "objective": "multiclass",
                        "num_class": 3,  # ajusta según tus clases reales
                        "n_estimators": self.n_estimators,  # más árboles + LR más bajo
                        "learning_rate": 0.03,  # más bajo para mejorar logloss
                        "num_leaves": 96,  # complejidad del árbol
                        "max_depth": -1,  # deja que LGBM lo maneje
                        "min_child_samples": 200,  # regularización clave con 220k filas

                        "subsample": 0.7,  # antes 0.9, bajamos para más robustez
                        "subsample_freq": 1,
                        "colsample_bytree": 0.7,  # menos columnas por árbol, menos overfit

                        "reg_lambda": 15.0,  # L2 fuerte → baja variancia, mejora logloss
                        "reg_alpha": 0.1,  # L1 ligera

                        "min_split_gain": 0.0,
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    display_feature_plot=False,
                    use_df_parquet=True
            )

            # # XGB con beta (usa LabelEncoder interno)
            # self.analyzer.create_predictive_model(model_type="xgb", calibrate_method="isotonic", N_KEEP=N_KEEP,
            #                              model_kwargs={
            #                                  "n_estimators": 700,
            #                                  "learning_rate": 0.05,
            #                                  "subsample": 0.9,
            #                                  "colsample_bytree": 0.9,
            #                                  "max_depth": 8,
            #                                  "random_state": 42,
            #                                  "n_jobs": -1,
            #                                  "tree_method": "hist",
            #                                  "objective": "multi:softprob"
            #                              })

    def _extract_proba_for_monitor(self, sig: Dict, return_margin: bool = False):
        """
        Devuelve la probabilidad relevante para evaluar umbrales de monitor.
        Si return_margin=True, retorna (top_prob, margin_top1_top2).
        - Usa 'proba' si viene directo.
        - Si hay 'probabilities' (dict) y 'prediction', usa la prob. de esa clase.
        - Si no hay 'prediction', usa la mayor del dict.
        """
        if sig is None:
            return None
        if "proba" in sig and sig["proba"] is not None:
            try:
                return float(sig["proba"])
            except Exception:
                pass
        probs = sig.get("probabilities") or sig.get("probs")
        pred = sig.get("prediction")
        if probs and isinstance(probs, dict):
            if pred in probs:
                return float(probs[pred])
            # si no hay pred, usa la mayor
            try:
                return float(max(probs.values()))
            except Exception:
                return None
        return None

    def offline_predict_at_index(self, i: int) -> Optional[Dict]:
        """Replica `predict_realtime` pero **offline** usando el dataset local.
        Usa una ventana de `offline_lookback_bars` terminando en el índice `i` (incluido).
        """
        start_i = max(0, i - self.offline_lookback_bars + 1)
        # df_win = self.df.loc[start_i:i, ["datetime", "open", "high", "low", "close", "volume"]].copy()
        df_ready = self.df.loc[start_i:i]


        features = FEATURES
        if df_ready.empty:
            return None
        last_row_feat = df_ready[features].iloc[-1:]
        last_meta = df_ready.iloc[-1]

        # escalar + predecir
        last_scaled = self.analyzer.scaler.transform(last_row_feat)
        proba_vec = self.analyzer.model.predict_proba(last_scaled)[0]
        classes = list(self.analyzer.model.classes_)

        # # aplicar thresholds del usuario (si tiene helper interno)
        if hasattr(self.analyzer, "_apply_thresholds"):
            pred_class = self.analyzer._apply_thresholds(
                proba_vec, classes, self.thresholds,
                default_class="lateral", global_floor=0.5, min_margin=0.03
            )
        else:
            # fallback simple
            best_idx = int(np.argmax(proba_vec))
            pred_class = classes[best_idx]

        # Clipping + renorm defensivo
        proba = np.clip(proba_vec, 1e-9, 1.0)
        s = proba.sum()
        if not np.isfinite(s) or s <= 0:
            raise RuntimeError("Probabilidades no finitas o suma <= 0.")
        proba = proba / s

        # Dict clase->probabilidad (M1)
        probabilities = {cls: float(round(p, 4)) for cls, p in zip(classes, proba)}
        ordered = sorted(probabilities.values(), reverse=True)
        top1 = ordered[0] if ordered else 0.0
        top2 = ordered[1] if len(ordered) > 1 else 0.0
        margin = top1 - top2


        pred = pred_class
        if pred in ("bullish", "bearish") and getattr(self, "m2_model", None) is not None:
            try:
                # Determinar set de features para M2
                feats_m2 = FEATURES + FEATS_M2

                # Construir vector x_m2 con orden exacto
                # Fuente principal: última fila de df_ready (ya con tus features preparados)
                src_series = df_ready.iloc[-1]



                x_dict = {}
                # soporte para direction_sign / dummies
                dir_sign_val = 1.0 if pred == "bullish" else -1.0
                dir_onehot = {"dir_bullish": 1.0 if pred == "bullish" else 0.0,
                              "dir_bearish": 1.0 if pred == "bearish" else 0.0}

                for f in feats_m2:
                    if f == "direction_sign":
                        x_dict[f] = dir_sign_val
                    elif f in ("dir_bullish", "dir_bearish"):
                        x_dict[f] = dir_onehot.get(f, 0.0)
                    elif f == "p_hat":
                        x_dict[f] = probabilities[pred]
                    elif f == "margin":
                        x_dict[f] = margin
                    elif f in src_series.index:
                        x_dict[f] = float(src_series[f])
                    elif f in df_ready.columns:
                        x_dict[f] = float(df_ready.iloc[-1][f])
                    else:
                        # Si falta, deja NaN para detección
                        x_dict[f] = np.nan

                x_dict['price_change'] = df_ready["price_change"].iloc[-2]
                # Verifica NaNs
                x_vals = [x_dict[f] for f in feats_m2]
                if any([not np.isfinite(v) for v in x_vals]):
                    # puedes cambiar por imputación si quieres continuar igual
                    raise RuntimeError(
                        f"Features M2 con NaN/inf en inferencia: {[feats_m2[i] for i, v in enumerate(x_vals) if not np.isfinite(v)]}")

                X_m2 = np.array([x_vals], dtype="float32")

                # Inferencia M2 (RF no requiere scaler)
                proba_m2 = self.m2_model.predict_proba(X_m2)[0]  # [p0, p1]
                p_meta = float(np.clip(proba_m2[1], 0.0, 1.0))  # prob. de “saldrá bien”
                p_meta_used = round(p_meta, 4)

                # Reemplazar la probabilidad de la dirección predicha por p_meta
                probabilities[pred] = p_meta_used

                # (Opcional) si quieres actualizar margen mostrado:
                ordered = sorted(probabilities.values(), reverse=True)
                top1 = ordered[0] if ordered else 0.0
                top2 = ordered[1] if len(ordered) > 1 else 0.0
                margin = top1 - top2
            except Exception as e:
                # No rompas la predicción si M2 falla: deja M1 tal cual y reporta aviso.
                print(f"[M2] Inferencia omitida: {e}")

        close_px = float(last_meta["close"])

        return {
            "prediction": pred,
            "probabilities": probabilities,
            "timestamp": last_meta['datetime'],
            "last_row_close": close_px,
            "last_row_atr": float(last_meta.get('ATR', np.nan)),
        }

    def get_signal(self, asof_index: int) -> Optional[Dict]:
        """Obtiene la predicción para la vela `asof_index`.
        Por defecto usa el predictor offline. Si `use_offline_predictor=False`, llama a `predict_realtime`.
        """
        if self.use_offline_predictor:
            return self.offline_predict_at_index(asof_index)
        else:
            # Online: el método del usuario podría imprimir; respetamos suppress_predict_stdout
            if self.suppress_predict_stdout:
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    return self.analyzer.predict_realtime(symbol=self.symbol, interval=self.interval)
            return self.analyzer.predict_realtime(symbol=self.symbol, interval=self.interval)

    # --------------------------
    # LÓGICA DE NEGOCIO
    # --------------------------

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        hl = df["high"] - df["low"]
        hc = (df["high"] - prev_close).abs()
        lc = (df["low"] - prev_close).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    def _passes_monitor(self, ts: pd.Timestamp, proba: float) -> bool:
        start, end = self.day_hours
        in_day = start <= ts.time() <= end
        thresh = self.proba_threshold_day if in_day else self.proba_threshold_night
        # return proba is not None and thresh <= float(proba)
        return proba is not None and float(proba) >= thresh < 0.97

    def _tp_sl_from_adapter_or_atr(self, i: int, proba: int, side: Literal["long", "short"],
                                   maybe_tp: Optional[float], maybe_sl: Optional[float]) -> Tuple[float, float]:
        px = float(self.df.loc[i, "close"])

        m0 = self.df.loc[i, "macd"]
        m1 = self.df.loc[i - 1, "macd"]

        s0 = self.df.loc[i, "macd_signal"]
        s1 = self.df.loc[i - 1, "macd_signal"]

        h0 = self.df.loc[i, "macd_hist"]
        h1 = self.df.loc[i - 1, "macd_hist"]
        slope_up = h0 > h1
        slope_down = h0 < h1

        if self.use_adapter_tp_sl_when_missing and (maybe_tp is not None) and (maybe_sl is not None):
            return float(maybe_tp), float(maybe_sl)
        atr = float(self.df.loc[i, "ATR"]) if not np.isnan(self.df.loc[i, "ATR"]) else px * 0.005
        ema_200 = float(self.df.loc[i, "EMA_200"])
        if side == "long":
            # if px < ema_200 and not (h0 > 0 and s0 > s1):
            #     self.atr_mult_tp = 3
            tp = px + self.atr_mult_tp * atr
            sl = px - self.atr_mult_sl * atr
        else:
            # if px > ema_200 and not (h1 < 0 and s0 < s1):
            #     self.atr_mult_tp = 3
            tp = px - self.atr_mult_tp * atr
            sl = px + self.atr_mult_sl * atr

        return tp, sl

    def _size_position(self, px: float, kelly_fraction: Optional[float] = None) -> float:
        k = max(0.0, min(1.0, float(kelly_fraction))) if kelly_fraction is not None else 1.0
        if self.position_mode == "fixed_risk":
            margin = self.equity * self.risk_fraction * k
            notional = margin * self.leverage
        else:
            notional = self.fixed_notional_usdt * k
        qty = max(0.0, notional / px)
        return qty

    def _apply_conflict_policy(self, bar_high: float, bar_low: float, tp: float, sl: float, side: str) -> Optional[str]:
        hit_tp = bar_high >= tp if side == "long" else bar_low <= tp
        hit_sl = bar_low <= sl if side == "long" else bar_high >= sl
        if hit_tp and hit_sl:
            if self.order_conflict_policy == "tp_first":
                return "tp"
            elif self.order_conflict_policy == "sl_first":
                return "sl"
            elif self.order_conflict_policy == "best":
                return "tp"
            else:
                return "sl"
        elif hit_tp:
            return "tp"
        elif hit_sl:
            return "sl"
        return None

    def _try_close_trade_on_bar(self, trade: Trade, i: int) -> None:
        if not trade.is_open():
            return
        bar = self.df.loc[i]
        decision = self._apply_conflict_policy(bar.high, bar.low, trade.tp_price, trade.sl_price, trade.side)
        if decision is not None:
            trade.exit_time = bar["datetime"]
            trade.exit_reason = decision
            trade.exit_price = trade.tp_price if decision == "tp" else trade.sl_price
            trade.closed_with = "taker"

    def _check_and_process_withdrawal(self, current_time: pd.Timestamp) -> bool:
        """
        Verifica si el equity ha alcanzado el objetivo de duplicación y procesa un retiro.
        
        Lógica:
        - Cada vez que equity >= last_withdrawal_target (el doble del punto anterior)
        - capital_total = withdrawn_capital + equity
        - withdrawal_amount = capital_total / 2
        - Se retira esa cantidad, actualizando equity y withdrawn_capital
        - El nuevo objetivo es equity * 2
        
        Args:
            current_time: Timestamp del momento del retiro
            
        Returns:
            True si se procesó un retiro, False si no
        """
        # Si los retiros están deshabilitados, salir
        if not getattr(self, "enable_withdrawals", True):
            return False

        if self.equity < self.last_withdrawal_target:
            return False
        
        # Calcular capital total
        capital_total = self.withdrawn_capital + self.equity
        
        # Calcular monto a retirar (mitad del total)
        withdrawal_amount = capital_total / 2.0
        
        # Guardar estado previo para logging
        equity_before = self.equity
        
        # Actualizar valores
        self.equity -= withdrawal_amount
        self.withdrawn_capital += withdrawal_amount
        
        # Actualizar próximo objetivo (cuando el equity actual se duplique)
        self.last_withdrawal_target = self.equity * 2.0
        
        # Registrar el retiro
        withdrawal_record = {
            'timestamp': current_time,
            'amount': withdrawal_amount,
            'equity_before': equity_before,
            'equity_after': self.equity,
            'total_withdrawn': self.withdrawn_capital,
            'capital_total': capital_total,
            'next_target': self.last_withdrawal_target
        }
        self.withdrawals.append(withdrawal_record)
        
        # Evento de retiro (si hay callback)
        if self.on_event:
            self.on_event("withdrawal", withdrawal_record)
        
        # Log si verbose
        if self.verbose >= 1:
            print(f"💰 RETIRO @ {current_time} | "
                  f"Amount=${withdrawal_amount:.2f} | "
                  f"Equity: ${equity_before:.2f} → ${self.equity:.2f} | "
                  f"Total withdrawn=${self.withdrawn_capital:.2f} | "
                  f"Next target=${self.last_withdrawal_target:.2f}")
        
        return True


    def run(self, test_start: str, test_end: Optional[str] = None) -> Dict:
        """
        Ejecuta el walk-forward entre [test_start, test_end].
        Por defecto **entrena una sola vez** con el histórico previo a `test_start`.
        Si `retrain_each_bar=True`, reentrena en cada barra antes de predecir.
        """


        # Alinear test_start / test_end a la misma tz que el DataFrame
        tz = getattr(self, "tz", None)

        start = pd.to_datetime(test_start)
        if tz is not None:
            start = start.tz_localize(tz) if start.tzinfo is None else start.tz_convert(tz)

        end = None
        if test_end is not None:
            end = pd.to_datetime(test_end)
            if tz is not None:
                end = end.tz_localize(tz) if end.tzinfo is None else end.tz_convert(tz)

        # Filtrado con tipos compatibles
        if end is not None:
            mask = self.df["datetime"].between(start, end)
        else:
            mask = (self.df["datetime"] >= start)

        idxs = self.df.index[mask].tolist()
        if not idxs:
            raise ValueError("Rango de test vacío. Revisa las fechas.")

        # self.analyzer.run_analysis(
        #     symbol="AVAXUSDT",
        #     start_date=self.start_date,
        #     end_date=self.end_date,
        #     interval="5"
        # )
        # self.df = self.analyzer.data

        # self.df = self.analyzer.add_indicators(self.df)
        # self.df = self.analyzer.preparar_features(self.df)

        self.df, features, _ = prepare_features_incremental(
            raw_df=self.df,
            preparar_features_fn=self.analyzer.preparar_features,  # tu propia función
            params=params_to_features,
            cache_dir="feature_cache",
            datetime_col="datetime",
            warmup_bars=150
        )

        # ENTRENAR UNA VEZ con datos previos a test_start
        first_test_i = idxs[0]
        train_slice_initial = self.df.loc[: first_test_i - 1]
        if len(train_slice_initial) >= 200:
            self.fit_model(train_slice_initial)
        else:
            self.logs.append("Aviso: histórico previo insuficiente para entrenar antes del test.")

        current_trade: Optional[Trade] = None
        last_allowed_index = idxs[-1]

        # 1) Prepara el DF UNA sola vez (al cargar datos)
        # dfp = prepare_rule_inputs(self.df, window=4300, min_periods=200)
        # self._prepared_rules_df = dfp
        # 2) Compila reglas UNA sola vez
        # masks = build_rule_masks(dfp)
        # dfp["_rule_bullish_any"] = masks["bullish_any"]
        # dfp["_rule_bearish_any"] = masks["bearish_any"]

        for pos, i in enumerate(idxs):
            just_closed = False
            # Heartbeat de progreso
            ts_here = self.df.loc[i, "datetime"]
            if self.on_event and (pos == 0 or (pos + 1) % self.log_every_bars == 0 or pos == len(idxs) - 1):
                self.on_event("progress", {"i": pos + 1, "total": len(idxs), "time": ts_here,
                                           "equity": self.equity, "trades": len(self.trades)})
            elif self.verbose >= 1 and (pos == 0 or (pos + 1) % self.log_every_bars == 0 or pos == len(idxs) - 1):
                print(f"[{pos + 1}/{len(idxs)}] {ts_here} | equity={self.equity:.2f} | trades={len(self.trades)}")
            # (Opcional) reentrenar por barra
            if self.retrain_each_bar:
                # reentrena en cada vela (modo antiguo)
                train_slice = self.df.loc[: i - 1]
                if self.n_keep_for_fit:
                    train_slice = train_slice.tail(self.n_keep_for_fit)
                if len(train_slice) >= 200:
                    # (opcional) anunciar ventana
                    if getattr(self, "announce_fit", False):
                        n = len(train_slice);
                        t0 = train_slice["datetime"].iloc[0];
                        t1 = train_slice["datetime"].iloc[-1]
                        if self.on_event: self.on_event("fit_start",
                                                        {"phase": "each_bar", "pos": pos + 1, "n": n, "from": t0,
                                                         "to": t1})
                    self.fit_model(train_slice)
                    if getattr(self, "announce_fit", False):
                        if self.on_event: self.on_event("fit_end", {"phase": "each_bar", "pos": pos + 1})

            elif self.retrain_every_bars is not None and self.retrain_every_bars > 0:
                # reentrena cada N velas del período de test (p.ej., cada 144)
                if (pos > 0) and ((pos + 1) % self.retrain_every_bars == 0):
                    # self.analyzer.run_analysis(
                    #     symbol="AVAXUSDT",
                    #     start_date=self.start_date,
                    #     end_date=self.end_date,
                    #     interval="5"
                    # )
                    self.df = self.analyzer.data
                    self.df, features, _ = prepare_features_incremental(
                        raw_df=self.df,
                        preparar_features_fn=self.analyzer.preparar_features,  # tu propia función
                        params=params_to_features,
                        cache_dir="feature_cache",
                        datetime_col="datetime",
                        warmup_bars=150
                    )
                    # Ventana rodante: últimas n_keep_for_fit velas hasta i-1
                    start_idx = max(0, i - (self.n_keep_for_fit or len(self.df)))
                    train_slice = self.df.iloc[start_idx: i]
                    if len(train_slice) >= 200:
                        if getattr(self, "announce_fit", False):
                            n = len(train_slice);
                            t0 = train_slice["datetime"].iloc[0];
                            t1 = train_slice["datetime"].iloc[-1]
                            if self.on_event: self.on_event("fit_start",
                                                            {"phase": "periodic", "pos": pos + 1, "n": n, "from": t0,
                                                             "to": t1})
                        self.fit_model(train_slice)
                        if getattr(self, "announce_fit", False):
                            if self.on_event: self.on_event("fit_end", {"phase": "periodic", "pos": pos + 1})


            # sig = self.get_signal(i)
            # if sig:
            #     pred = sig.get("prediction")
            #     proba = self._extract_proba_for_monitor(sig)
            #     probs = sig.get("probabilities") or {}
            #     # proba_pred = float(probs.get(pred, 0.0))
            #
            #     ordered = sorted(probs.values(), reverse=True)
            #     top1 = ordered[0] if ordered else 0.0
            #     top2 = ordered[1] if len(ordered) > 1 else 0.0
            #     margin = (top1 - top2)
            #
            #     ts = self.df.loc[i, "timestamp"]
            #     # macd = self.df.loc[i, "macd"]
            #     # macd_signal = self.df.loc[i, "macd_signal"]
            #     # h0 = self.df.loc[i, "macd_hist"]
            #     # h1 = self.df.loc[i - 1, "macd_hist"]
            #     # slope = '↑' if h0 > h1 else '↓'
            #     # macd_signal_slope = '↑' if macd > macd_signal else '↓'
            #     side = "long" if pred == "bullish" else "short"
            #     # k = self.df.loc[i, "k"]
            #     # e = self.df.loc[i, "e"]
            #
            #
            #     tp, sl = self._tp_sl_from_adapter_or_atr(
            #         i, proba, side,
            #         sig.get("take_profit") or sig.get("tp"),
            #         sig.get("stop_loss") or sig.get("sl"),
            #     )
            #
            #     # if self.on_event and proba >= 0.6:
            #     #     self.on_event("new_predicction", {
            #     #         "time": ts, "side": pred, "prob": proba, "macd": macd,
            #     #         "macd_signal": macd_signal,
            #     #         "macd_signal_slope": macd_signal_slope,
            #     #         'tp': tp,
            #     #         'sl': sl,
            #     #         'k': k,
            #     #         'e': e,
            #     #         'margin': margin
            #     #     })
            #
            #     print(f"timestamp={ts} | side={pred} | prob={proba:.3f} | tp={tp} | sl={sl} | margin={margin}")
            #     sig = False

            # Si hay trade abierto, intentar cierre en esta barra
            if current_trade is not None and current_trade.is_open():
                self._try_close_trade_on_bar(current_trade, i)
                if not current_trade.is_open():
                    pnl = current_trade.pnl_usdt()
                    fees_paid = current_trade.fees_usdt()
                    self.equity += pnl - fees_paid
                    
                    # Verificar si se debe procesar un retiro
                    ts_here = self.df.loc[i, "datetime"]
                    self._check_and_process_withdrawal(ts_here)
                    
                    # Evento de cierre ANTES de limpiar la variable
                    if self.on_event:
                        dur = (current_trade.exit_time - current_trade.entry_time) if (
                                current_trade.exit_time and current_trade.entry_time) else None
                        self.on_event("close", {"time": current_trade.exit_time, "reason": current_trade.exit_reason,
                                                "price": current_trade.exit_price, "pnl": pnl, "fees": fees_paid,
                                                "equity": self.equity, "entry_time": current_trade.entry_time,
                                                "duration": dur})

                    if self.verbose >= 2:
                        print(
                            f"CLOSE {current_trade.side} @ {current_trade.exit_price:.4f} ({current_trade.exit_reason}) | PnL={pnl:.2f} | fees={fees_paid:.2f} | equity={self.equity:.2f}")
                    self._mark_entry_outcome(current_trade)
                    self._update_rule_stats_on_close(current_trade, pnl)
                    self.trades.append(current_trade)
                    current_trade = None

            if just_closed:
                continue
            # Early exit por señal opuesta con ganancia asegurada
            if (current_trade is not None and current_trade.is_open() and self.early_exit_on_opposite_signal):
                sig_cur = self.get_signal(i)
                # sig_cur = False
                pnl_gross = current_trade.pnl_usdt()
                if sig_cur:
                    probs = sig_cur.get("probabilities") or {}
                    try:
                        px_now = float(self.df.loc[i, "close"])
                    except Exception:
                        px_now = float(current_trade.entry_price)
                    prob_used = None
                    bear_l = probs.get("lateral")

                    if current_trade.side == "long" and current_trade.entry_price > 0:
                        # Use taker fees for both sides as conservative estimate
                        fee_rate = getattr(self.fees, "taker_open", 0.00055)
                        breakeven_price = current_trade.entry_price * (1 + fee_rate) / (1 - fee_rate)
                        
                        bear_p = probs.get("bearish")
                        if (bear_p is not None) and ((float(bear_p) >= self.early_exit_threshold) or (
                                float(bear_l) >= self.early_exit_threshold + 0.2)) and (
                                px_now > breakeven_price):
                            current_trade.exit_time = self.df.loc[i, "datetime"]
                            current_trade.exit_price = px_now
                            current_trade.exit_reason = "early_opposite"
                            prob_used = max(float((probs or {}).get("bearish", 0)),
                                            float((probs or {}).get("lateral", 0)))
                            if self.on_event:
                                dur = (current_trade.exit_time - current_trade.entry_time) if (
                                        current_trade.exit_time and current_trade.entry_time) else None
                                self.on_event("early_opposite", {
                                    "side": current_trade.side,
                                    "time": current_trade.exit_time,
                                    "price": current_trade.exit_price,
                                    "prob": prob_used,
                                    "threshold": self.early_exit_threshold,
                                    "entry_time": current_trade.entry_time,
                                    "duration": dur,
                                    "pnl_gross": pnl_gross
                                })
                    elif current_trade.side == "short" and current_trade.entry_price > 0:
                        fee_rate = getattr(self.fees, "taker_open", 0.00055)
                        breakeven_price = current_trade.entry_price * (1 - fee_rate) / (1 + fee_rate)
                        
                        bull_p = probs.get("bullish")
                        if (bull_p is not None) and ((float(bull_p) >= self.early_exit_threshold) or (
                                float(bear_l) >= self.early_exit_threshold + 0.2)) and (
                                px_now < breakeven_price):
                            current_trade.exit_time = self.df.loc[i, "datetime"]
                            current_trade.exit_price = px_now
                            current_trade.exit_reason = "early_opposite"
                            prob_used = max(float((probs or {}).get("bullish", 0)),
                                            float((probs or {}).get("lateral", 0)))
                            if self.on_event:
                                dur = (current_trade.exit_time - current_trade.entry_time) if (
                                        current_trade.exit_time and current_trade.entry_time) else None
                                self.on_event("early_opposite", {
                                    "side": current_trade.side,
                                    "time": current_trade.exit_time,
                                    "price": current_trade.exit_price,
                                    "prob": prob_used,
                                    "threshold": self.early_exit_threshold,
                                    "entry_time": current_trade.entry_time,
                                    "duration": dur,
                                    "pnl_gross": pnl_gross
                                })

                if current_trade is not None and not current_trade.is_open():
                    pnl = current_trade.pnl_usdt()
                    fees_paid = current_trade.fees_usdt()
                    self.equity += pnl - fees_paid
                    
                    # Verificar si se debe procesar un retiro
                    ts_here = self.df.loc[i, "datetime"]
                    self._check_and_process_withdrawal(ts_here)
                    
                    if self.on_event:
                        dur = (current_trade.exit_time - current_trade.entry_time) if (
                                current_trade.exit_time and current_trade.entry_time) else None
                        self.on_event("close", {"time": current_trade.exit_time, "reason": current_trade.exit_reason,
                                                "price": current_trade.exit_price, "pnl": pnl, "fees": fees_paid,
                                                "equity": self.equity, "entry_time": current_trade.entry_time,
                                                "duration": dur})
                    if self.verbose >= 2:
                        print(
                            f"CLOSE {current_trade.side} @ {current_trade.exit_price:.4f} ({current_trade.exit_reason}) | PnL={pnl:.2f} | fees={fees_paid:.2f} | equity={self.equity:.2f}")
                    self._update_rule_stats_on_close(current_trade, pnl)
                    self.trades.append(current_trade)
                    current_trade = None
                    continue

            # Si no hay trade, evaluar nueva señal
            if current_trade is None:
                sig = self.get_signal(i)
                if not sig:
                    continue
                pred = sig.get("prediction")
                if pred not in {"bullish", "bearish"}:
                    continue
                probs = sig.get("probabilities") or {}
                proba = self._extract_proba_for_monitor(sig)

                ordered = sorted(probs.values(), reverse=True)
                top1 = ordered[0] if ordered else 0.0
                top2 = ordered[1] if len(ordered) > 1 else 0.0
                margin = (top1 - top2)
                # 3) En tu loop: solo consulta
                # gate_order = validate_start_order_fast(dfp, pred, i, margin, proba)
                # if gate_order and gate_order.get("enter"):
                #     rule_name = gate_order.get("rule")  # ← nombre textual de la regla
                #     side_txt = gate_order.get("side")  # 'bullish' o 'bearish'
                #     if verbose >= 2:
                #         print(rule_name)

                if True:

                    ts = self.df.loc[i, "datetime"]
                    side = "long" if pred == "bullish" else "short"
                    if not self._passes_monitor(ts, proba):
                        continue

                    px = float(self.df.loc[i, "close"])  # entrada al cierre de i

                    # TP/SL concretos de esta señal
                    tp, sl = self._tp_sl_from_adapter_or_atr(
                        i, proba, side,
                        sig.get("take_profit") or sig.get("tp"),
                        sig.get("stop_loss") or sig.get("sl"),
                    )

                    # ★ Kelly: si la señal no trae kelly_fraction, lo calculamos aquí
                    kelly = sig.get("kelly_fraction")
                    if (kelly is None):
                        kelly = self._kelly_from_signal(probs, side, px, float(tp), float(sl))

                    # Sizing con Kelly (si kelly es None, _size_position usa 1.0 por defecto)
                    qty = self._size_position(px, kelly)
                    if qty <= 0:
                        continue

                    current_trade = Trade(
                        side=side,
                        entry_time=ts,
                        entry_price=px,
                        tp_price=float(tp),
                        sl_price=float(sl),
                        qty=float(qty),
                        leverage=self.leverage,
                        opened_with="taker",
                        closed_with="taker",
                        entry_proba=proba,
                        entry_pred=pred,
                    )

                    current_trade.entry_index = i
                    # marca info de regla en el trade y en el DF (en la vela de ENTRADA)
                    # rule_name = rule_name if gate_order and gate_order.get("enter") else "no rule"
                    # side_txt = side_txt if gate_order and gate_order.get("enter") else "bearish"
                    rule_name = "no rule"
                    side_txt = "bearish"
                    current_trade.entry_rule = rule_name
                    current_trade.entry_rule_side = side_txt
                    self.df.loc[i, "entry_flag"] = True
                    self.df.loc[i, "entry_rule"] = rule_name
                    self.df.loc[i, "entry_rule_side"] = side_txt
                    self.df.loc[i, "entry_flag"] = True

                    side_txt = None
                    pred_txt = None
                    try:
                        if hasattr(current_trade, "side"):  # 'long'/'short' o similar
                            side_txt = str(current_trade.side).lower()
                        elif hasattr(current_trade, "is_long"):
                            side_txt = "long" if bool(current_trade.is_long) else "short"
                        elif hasattr(current_trade, "direction"):
                            side_txt = str(current_trade.direction).lower()
                    except Exception:
                        pass

                    if side_txt in ("long", "buy"):
                        pred_txt = "bullish"
                    elif side_txt in ("short", "sell"):
                        pred_txt = "bearish"

                    self.df.loc[i, "entry_side"] = side_txt
                    self.df.loc[i, "entry_pred"] = pred_txt

                    # comisión de entrada
                    entry_fee = px * qty * (
                        self.fees.taker_open if current_trade.opened_with == "taker" else self.fees.maker_open)
                    self.equity -= entry_fee

                    # Log / evento (incluye Kelly)
                    if self.on_event:
                        self.on_event("open", {
                            "time": ts, "side": side, "price": px,
                            "tp": float(tp), "sl": float(sl),
                            "qty": float(qty), "prob": proba,
                            "kelly": float(kelly) if kelly is not None else None,
                            "margin": float(margin)
                        })
                    if self.verbose >= 2:
                        print(f"OPEN  {side} @ {px:.4f} | tp={tp:.4f} sl={sl:.4f} | qty={qty:.4f} "
                              f"| prob={proba * 100:.1f} | kelly={kelly if kelly is not None else 1.0:.3f}")
                else:
                    continue

            # Avanzar para cierre, pero no más allá de `test_end` si se pidió
            j_limit = last_allowed_index if self.close_trades_at_test_end else len(self.df) - 1
            j = i + 1
            if False and current_trade is not None and current_trade.is_open() and j <= j_limit:
                self._try_close_trade_on_bar(current_trade, j)
                if not current_trade.is_open():
                    pnl = current_trade.pnl_usdt()
                    fees_paid = current_trade.fees_usdt()
                    self.equity += pnl - fees_paid
                    if self.on_event:
                        dur = (current_trade.exit_time - current_trade.entry_time) if (
                                current_trade.exit_time and current_trade.entry_time) else None
                        self.on_event("close", {"time": current_trade.exit_time, "reason": current_trade.exit_reason,
                                                "price": current_trade.exit_price, "pnl": pnl, "fees": fees_paid,
                                                "equity": self.equity, "entry_time": current_trade.entry_time,
                                                "duration": dur})
                    if self.verbose >= 2:
                        print(
                            f"CLOSE {current_trade.side} @ {current_trade.exit_price:.4f} ({current_trade.exit_reason}) | PnL={pnl:.2f} | fees={fees_paid:.2f} | equity={self.equity:.2f}")
                    self._mark_entry_outcome(current_trade)
                    self._update_rule_stats_on_close(current_trade, pnl)
                    self.trades.append(current_trade)
                    current_trade = None
                    break
                j += 1

        # Si queda abierta al final, cerramos a mercado en la última barra del test
        if current_trade is not None and current_trade.is_open():
            pnl = 1
            last_bar = self.df.loc[last_allowed_index]
            current_trade.exit_time = last_bar["datetime"]
            current_trade.exit_price = float(last_bar["close"])
            current_trade.exit_reason = "close"
            pnl = current_trade.pnl_usdt()
            fees_paid = current_trade.fees_usdt()
            self.equity += pnl - fees_paid
            if self.on_event:
                dur = (current_trade.exit_time - current_trade.entry_time) if (
                        current_trade.exit_time and current_trade.entry_time) else None
                self.on_event("close", {"time": current_trade.exit_time, "reason": current_trade.exit_reason,
                                        "price": current_trade.exit_price, "pnl": pnl, "fees": fees_paid,
                                        "equity": self.equity, "entry_time": current_trade.entry_time,
                                        "duration": dur})
            if self.verbose >= 2:
                print(
                    f"FORCE CLOSE @ period end {current_trade.exit_price:.4f} | PnL={pnl:.2f} | fees={fees_paid:.2f} | equity={self.equity:.2f}")
            self._mark_entry_outcome(current_trade)
            self._update_rule_stats_on_close(current_trade, pnl)
            self.trades.append(current_trade)

        wins = sum(1 for t in self.trades if t.exit_reason == "tp")
        losses = sum(1 for t in self.trades if t.exit_reason == "sl")
        rr = wins / max(1, losses) if (wins + losses) > 0 else None

        proba_summary = self.summarize_by_proba(step=1, include_forced=True)

        if self.verbose >= 1:
            print("=== Resumen por probabilidad (bucket 10%) ===")
            for bucket, s in proba_summary.items():
                print(f"p≈{bucket}%  wins={s['wins']:<4} losses={s['losses']:<4} "
                      f"forced={s['forced']:<4} trades={s['trades']}")

        # === Construir trades_df y resumir ===
        trades_rows = []
        for t in self.trades:
            trades_rows.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "leverage": getattr(t, "leverage", None),
                "exit_reason": t.exit_reason,
                "entry_proba": getattr(t, "entry_proba", None),
                "entry_pred": getattr(t, "entry_pred", None),
                "fees": t.fees_usdt() if hasattr(t, "fees_usdt") else None,
                "pnl": t.pnl_usdt() if hasattr(t, "pnl_usdt") else None,
            })
        trades_df = pd.DataFrame(trades_rows)

        # metrics, trades_df = self._summarize_backtest(trades_df)

        metrics, trades_df = self._summarize_backtest(trades_df, equity_curve=None, initial_equity=self.equity)

        if self.verbose >= 1:
            print("\n=== Resumen Backtest ===")
            for k, v in metrics.items():
                print(f"{k:>18}: {v}")
            
            # Mostrar información de retiros si existen
            if hasattr(self, 'withdrawals') and self.withdrawals:
                print("\n=== Retiros de Capital ===")
                print(f"{'Retiros realizados':>18}: {len(self.withdrawals)}")
                print(f"{'Capital retirado':>18}: ${self.withdrawn_capital:.2f}")
                print(f"{'Equity final':>18}: ${self.equity:.2f}")
                print(f"{'Capital TOTAL':>18}: ${self.equity + self.withdrawn_capital:.2f}")

        # al final de run():
        self._finalize_and_print_rule_stats()
        
        # Preparar datos de retiros para el retorno
        withdrawal_summary = None
        if hasattr(self, 'withdrawals') and self.withdrawals:
            withdrawal_summary = {
                "total_withdrawn": float(self.withdrawn_capital),
                "n_withdrawals": len(self.withdrawals),
                "total_capital": float(self.equity + self.withdrawn_capital),
                "withdrawals": self.withdrawals
            }

        return {
            "final_equity": float(self.equity),
            "n_trades": len(self.trades),
            "wins": wins,
            "losses": losses,
            "win_loss_ratio": float(rr) if rr is not None else None,
            "trades": [asdict(t) for t in self.trades],
            "fees": asdict(self.fees),
            "proba_summary": proba_summary,
            "metrics": metrics,
            "withdrawals": withdrawal_summary
        }

    def _finalize_and_print_rule_stats(self):
        rows = []
        for name, s in self.rule_stats.items():
            used = int(s["used"])
            wins = int(s["wins"])
            losses = int(s["losses"])
            be = int(s["breakeven"])
            attempts_for_wr = max(1, wins + losses)  # no contamos BE para win_rate
            win_rate = wins / attempts_for_wr
            avg_pnl = (s["pnl_sum"] / used) if used > 0 else 0.0
            rows.append({
                "rule": name,
                "side": s["side"],
                "used": used,
                "wins": wins,
                "losses": losses,
                "breakeven": be,
                "win_rate": round(win_rate, 4),
                "avg_pnl": round(avg_pnl, 4),
            })
        if not rows:
            print("[rule-stats] No hay trades abiertos por reglas.")
            self.rule_stats_df = pd.DataFrame()
            return

        df = pd.DataFrame(rows).sort_values(
            by=["side", "used", "win_rate"], ascending=[True, False, False]
        )
        self.rule_stats_df = df
        if self.verbose >= 1:
            print("\n=== REGLAS (desempeño por regla de entrada) ===")
            print(df.to_string(index=False))

    def _kelly_from_signal(self,
                           probs: dict,
                           side: str,
                           entry: float,
                           tp: float,
                           sl: float,
                           rr_min: float = 1e-6,
                           rr_max: float = 100.0) -> Optional[float]:
        """
        Calcula kelly usando p (proba direccional) y b = reward/risk desde TP/SL reales.
        side: 'long' o 'short'.
        probs: dict con claves 'bullish'/'bearish' (calibradas).
        """
        if probs is None:
            return None

        if side == "long":
            p = probs.get("bullish", None)
            reward = max(tp - entry, 1e-12)
            risk = max(entry - sl, 1e-12)
        else:  # short
            p = probs.get("bearish", None)
            reward = max(entry - tp, 1e-12)
            risk = max(sl - entry, 1e-12)

        if p is None:
            return None

        # Sanitiza y limita
        # p = min(max(p, 1e-3), 1 - 1e-3)
        p = 0.58
        rr = reward / risk
        rr = min(max(rr, rr_min), rr_max)

        k = p - (1.0 - p) / rr  # Kelly clásica
        return float(k)

    def _bucketize_proba(self, p: Optional[float], step: int = 10) -> Optional[int]:
        """Convierte p (0..1) a un bucket de porcentaje (0..100) según step (10 -> 0,10,20,...)."""
        if p is None:
            return None
        pct = int(round(100 * float(p)))
        return (pct // step) * step  # 87% -> 80 si step=10

    def summarize_by_proba(self, step: int = 10, include_forced: bool = True) -> Dict[int, Dict[str, int]]:
        """
        Devuelve {bucket_pct: {'wins': x, 'losses': y, 'forced': z, 'trades': n}}, ordenado por 'wins' desc.
        - wins: tp o (cierre forzado con PnL>0 si include_forced=True)
        - losses: sl o (cierre forzado con PnL<=0 si include_forced=True)
        - forced: cierres 'close' (forzados por fin de período)
        """
        summ: Dict[int, Dict[str, int]] = {}
        for t in self.trades:
            b = self._bucketize_proba(t.entry_proba, step=step)
            if b is None:
                continue
            d = summ.setdefault(b, {"wins": 0, "losses": 0, "forced": 0, "trades": 0})
            # clasificar resultado
            if t.exit_reason in ("tp", "sl"):
                win = (t.exit_reason == "tp")
            else:
                # 'close' forzado: decidir por PnL
                win = (t.pnl_usdt() > 0.0)
                if include_forced:
                    d["forced"] += 1
            if win:
                d["wins"] += 1
            else:
                d["losses"] += 1
            d["trades"] += 1

        # ordenar por wins desc
        ordered = dict(sorted(summ.items(), key=lambda kv: kv[1]["wins"], reverse=True))
        return ordered

    def _summarize_backtest(self,
                            trades_df: pd.DataFrame,
                            equity_curve: Optional[pd.Series] = None,
                            initial_equity: Optional[float] = None,
                            auto_build_equity: bool = True,
                            build_r_multiple: bool = True):
        df = trades_df.copy()

        # --- Tiempos
        if {'entry_time', 'exit_time'}.issubset(df.columns):
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])

        # --- PnL bruto si falta
        if 'pnl' not in df.columns and {'entry_price', 'exit_price', 'qty', 'side'}.issubset(df.columns):
            side = df['side'].str.lower()
            pnl_raw = np.where(side.isin(['short', 'bearish']),
                               (df['entry_price'] - df['exit_price']) * df['qty'],
                               (df['exit_price'] - df['entry_price']) * df['qty'])
            df['pnl'] = pnl_raw

        # --- PnL neto
        df['pnl_net'] = df['pnl'] - df['fees'] if 'fees' in df.columns else df['pnl']

        # --- r-multiple (si no existe y hay SL)
        if build_r_multiple and 'r_multiple' not in df.columns:
            if {'entry_price', 'sl_price', 'qty'}.issubset(df.columns):
                risk = (df['entry_price'] - df['sl_price']).abs() * df['qty']
                risk = risk.replace(0, np.nan)
                df['r_multiple'] = df['pnl_net'] / risk

        # --- Clasificación de resultado (usa exit_reason + pnl_net)
        if 'exit_reason' in df.columns:
            reason = df['exit_reason'].astype(str).str.lower()
            df['result'] = np.select(
                [reason.eq('tp'),
                 reason.eq('early_opposite'),
                 reason.eq('sl')],
                ['win', 'win', 'loss'],
                default=np.where(df['pnl_net'] > 0, 'win',
                                 np.where(df['pnl_net'] < 0, 'loss', 'breakeven'))
            )
        else:
            df['result'] = np.where(df['pnl_net'] > 0, 'win',
                                    np.where(df['pnl_net'] < 0, 'loss', 'breakeven'))

        # --- Duración
        if {'entry_time', 'exit_time'}.issubset(df.columns):
            dur = (df['exit_time'] - df['entry_time']).dt.total_seconds()
            df['hold_minutes'] = dur / 60.0

        # --- Conteos
        n_trades = len(df)
        n_closed = int((~df['pnl_net'].isna()).sum())
        wins = int((df['result'] == 'win').sum())
        losses = int((df['result'] == 'loss').sum())
        breakeven = int((df['result'] == 'breakeven').sum())
        winrate = wins / n_closed if n_closed > 0 else np.nan

        # --- Rachas
        def _streak(series, target):
            best = cur = 0
            for x in series:
                if x == target:
                    cur += 1;
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        max_consec_wins = _streak(df['result'].tolist(), 'win')
        max_consec_losses = _streak(df['result'].tolist(), 'loss')

        # --- PnL stats
        gross_profit = df.loc[df['pnl_net'] > 0, 'pnl_net'].sum()
        gross_loss = -df.loc[df['pnl_net'] < 0, 'pnl_net'].sum()  # positivo
        net_profit = gross_profit - gross_loss
        avg_win = df.loc[df['pnl_net'] > 0, 'pnl_net'].mean()
        avg_loss = df.loc[df['pnl_net'] < 0, 'pnl_net'].mean()  # negativo
        payoff_ratio = (avg_win / abs(avg_loss)) if (
                pd.notna(avg_win) and pd.notna(avg_loss) and avg_loss != 0) else np.nan
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)
        expectancy_val = net_profit / n_closed if n_closed > 0 else np.nan

        # --- SQN (si ya calculamos r_multiple arriba)
        if 'r_multiple' in df.columns and df['r_multiple'].notna().any():
            exp_R = float(df['r_multiple'].mean(skipna=True))
            std_R = float(df['r_multiple'].std(skipna=True))
            sqn = (exp_R / (std_R + 1e-12)) * np.sqrt(max(n_closed, 1))
        else:
            exp_R = std_R = sqn = np.nan

        # --- Equity curve / MDD
        def _max_drawdown(series: pd.Series):
            if series is None or len(series) == 0:
                return np.nan
            s = series.astype(float).values
            peak = np.maximum.accumulate(s)
            dd = (s - peak) / (peak + 1e-12)
            return float(dd.min())  # negativo

        mdd = np.nan
        if equity_curve is not None:
            eq = equity_curve.dropna()
            mdd = _max_drawdown(eq)
        elif auto_build_equity and initial_equity is not None and 'exit_time' in df.columns:
            # reconstruir equity con los cierres en orden temporal
            tmp = df.sort_values('exit_time')
            eq = initial_equity + tmp['pnl_net'].cumsum()
            eq.index = tmp['exit_time']
            mdd = _max_drawdown(eq)

        summary = {
            "trades": int(n_trades),
            "cerrados": int(n_closed),
            "wins": int(wins),
            "losses": int(losses),
            "breakeven": int(breakeven),
            "winrate": round(winrate, 4) if pd.notna(winrate) else np.nan,
            "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else profit_factor,
            "payoff_ratio": round(payoff_ratio, 4) if pd.notna(payoff_ratio) else np.nan,
            "expectancy_$": round(expectancy_val, 4) if pd.notna(expectancy_val) else np.nan,
            "expectancy_R": round(exp_R, 4) if pd.notna(exp_R) else np.nan,
            "std_R": round(std_R, 4) if pd.notna(std_R) else np.nan,
            "SQN": round(sqn, 4) if pd.notna(sqn) else np.nan,
            "max_drawdown": round(mdd, 4) if pd.notna(mdd) else np.nan,
            "max_consec_wins": int(max_consec_wins),
            "max_consec_losses": int(max_consec_losses),
            "avg_hold_min": round(float(df['hold_minutes'].mean()), 2) if 'hold_minutes' in df.columns else np.nan,
            "p50_hold_min": round(float(df['hold_minutes'].median()), 2) if 'hold_minutes' in df.columns else np.nan,
            "gross_profit": round(float(gross_profit), 2),
            "gross_loss": round(float(gross_loss), 2),
            "net_profit": round(float(net_profit), 2),
        }
        return summary, df

    def _mark_entry_outcome(self, trade) -> None:
        """Marca en self.df la vela de ENTRADA con el resultado final del trade."""
        entry_idx = getattr(trade, "entry_index", None)
        if entry_idx is None:
            return
        pnl = float(trade.pnl_usdt()) if hasattr(trade, "pnl_usdt") else np.nan
        if np.isnan(pnl):
            # Si no hay PnL calculable, solo registramos el cierre
            self.df.loc[entry_idx, ["entry_exit_time", "entry_exit_reason"]] = [trade.exit_time, trade.exit_reason]
            return

        result = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        self.df.loc[entry_idx, "entry_win"] = (pnl > 0)
        self.df.loc[entry_idx, "entry_result"] = result
        self.df.loc[entry_idx, "entry_exit_time"] = trade.exit_time
        self.df.loc[entry_idx, "entry_exit_reason"] = trade.exit_reason
        self.df.loc[entry_idx, "entry_pnl"] = pnl

    def autogen_rules_from_entries(self,
                                   prepared_df: Optional[pd.DataFrame] = None,
                                   min_support: int = 30,
                                   max_pairs: int = 100,
                                   candidate_cols: Optional[list] = None,
                                   no_entry: bool = True,
                                   no_entry_lift_max: float = 0.9,
                                   max_triples: int = 200,
                                   triples_from_top_k: int = 30) -> dict:
        """
        Genera reglas a partir de ENTRADAS reales (entry_flag=True).
        Además, si no_entry=True, genera reglas de NO ENTRAR cuando la lift <= no_entry_lift_max.

        Devuelve:
        {
          "bullish": [...],
          "bearish": [...],
          "no_entry_bullish": [...],
          "no_entry_bearish": [...]
        }
        """

        if prepared_df is None:
            prepared_df = getattr(self, "_prepared_rules_df", None)
            if prepared_df is None:
                raise RuntimeError("No tengo prepared_df. Pasa uno o ejecuta run() que fije self._prepared_rules_df.")

        # === Trae columnas (sin join para evitar colisiones) ===
        needed = ["entry_flag", "entry_win", "entry_result", "entry_pred"]
        df_join = prepared_df.copy()
        df_right = self.df.reindex(df_join.index)

        for col in needed:
            if col in df_right.columns:
                df_join[col] = df_right[col].values
            else:
                # crea vacías si faltan (las rellenamos abajo si podemos)
                if col == "entry_flag" or col == "entry_win":
                    df_join[col] = False
                else:
                    df_join[col] = pd.NA

        # Inferir entry_pred si falta
        if df_join["entry_pred"].isna().all():
            if "entry_side" in df_right.columns:
                mapped = df_right["entry_side"].map(
                    lambda s: "bullish" if str(s).lower() in ("long", "buy")
                    else ("bearish" if str(s).lower() in ("short", "sell") else pd.NA)
                )
                df_join["entry_pred"] = df_join["entry_pred"].fillna(mapped)
            for col_long, col_short in (("entry_long", "entry_short"), ("signal_long", "signal_short")):
                if col_long in df_right.columns and col_short in df_right.columns:
                    long_s = df_right[col_long]
                    short_s = df_right[col_short]
                    mask_nan = df_join["entry_pred"].isna()
                    df_join.loc[mask_nan & (long_s == True), "entry_pred"] = "bullish"
                    df_join.loc[mask_nan & (short_s == True), "entry_pred"] = "bearish"

        df_entries = df_join[df_join["entry_flag"] == True].copy()
        if df_entries.empty:
            return {"bullish": [], "bearish": [], "no_entry_bullish": [], "no_entry_bearish": []}

        if df_entries["entry_pred"].dropna().empty:
            df_entries["entry_pred"] = "bullish"

        df_entries["entry_pred"] = (
            df_entries["entry_pred"].astype(str).str.lower()
            .replace({"buy": "bullish", "long": "bullish", "sell": "bearish", "short": "bearish"})
        )

        # Win rate base por lado
        base_wr = {}
        for side in ("bullish", "bearish"):
            sub = df_entries[df_entries["entry_pred"] == side]
            base_wr[side] = float((sub["entry_win"] == True).mean()) if len(sub) else None

        # Candidatas
        if candidate_cols is None:
            qcols = [c for c in df_entries.columns if c.endswith("_q")]
            bpool = [
                "vwap_up", "vwap_down", "e_down", "d_down", "e100_up", "e100_down",
                "macd_up", "macd_down", "macd_hist_down", "macd_signal_down", "macd_signal_up",
                "k100_lt_e100", "k100_gt_e100", "k_gt_e", "k_lt_e", "d_lt_e", "macd_gt_signal",
                "e_lt_25", "e100_lt_25", "dist_PP_lt_0", "dist_R1_gt_0", "dist_R2_gt_0",
                "dist_S1_gt_0", "dist_S2_gt_0", "rsi_40_60"
            ]
            bcols = [c for c in bpool if c in df_entries.columns]
            candidate_cols = qcols + bcols

        def _iter_single(df):
            for col in candidate_cols:
                s = df[col]
                if s.dtype.name in ("category", "object"):
                    vals = [v for v in pd.Series(s.dropna().unique()).tolist() if v is not pd.NA]
                    for v in vals:
                        yield f"{col} == {v}", (s == v)
                elif pd.api.types.is_bool_dtype(s):
                    yield f"{col} == True", (s == True)

        def _metrics(mask, df, side):
            sub = df[(df["entry_pred"] == side) & mask]
            support = int(sub.shape[0])
            if support < min_support:
                return None
            wr_base = base_wr[side]
            if wr_base is None or wr_base <= 0:
                return None
            wr = float((sub["entry_win"] == True).mean())
            lift = float(wr / wr_base) if wr_base > 0 else np.nan
            return {"min_support": support, "win_rate": wr, "lift": lift}

        def _as_clause(cond_name: str) -> dict:
            if " == " in cond_name:
                col, val = cond_name.split(" == ", 1)
                val = val.strip().strip('"').strip("'")
                if val.lower() in ("true", "false"):
                    value = (val.lower() == "true")
                else:
                    value = val
                return {"col": col.strip(), "op": "==", "value": value}
            else:
                return {"col": cond_name.strip(), "op": "==", "value": True}

        # Recipientes
        rules_pos = {"bullish": [], "bearish": []}
        rules_neg = {"no_entry_bullish": [], "no_entry_bearish": []}

        singles = list(_iter_single(df_entries))

        # 1) Singles
        for side in ("bullish", "bearish"):
            for name, mask in singles:
                m = _metrics(mask, df_entries, side)
                if not m:
                    continue
                item = {
                    "name": name,
                    "all": [_as_clause(name)],
                    **m
                }
                # Positivas
                if m["lift"] >= 1.0:
                    rules_pos[side].append(item)
                # Negativas (NO ENTRY)
                if no_entry and (m["lift"] <= no_entry_lift_max):
                    key = "no_entry_bullish" if side == "bullish" else "no_entry_bearish"
                    rules_neg[key].append(item)

        # Orden inicial por lift
        for side in ("bullish", "bearish"):
            rules_pos[side].sort(key=lambda r: (r["lift"], r["win_rate"], r["min_support"]), reverse=True)
        for key in ("no_entry_bullish", "no_entry_bearish"):
            rules_neg[key].sort(key=lambda r: (r["lift"], -r["win_rate"], r["min_support"]))  # lift más bajo primero

        # 2) Pares (limitados)
        from itertools import combinations
        for side in ("bullish", "bearish"):
            top_singles = rules_pos[side][:min(50, len(rules_pos[side]))]  # combinar sobre buenas
            evaluated = 0
            for a, b in combinations(top_singles, 2):
                if evaluated >= max_pairs:
                    break
                n1, n2 = a["name"], b["name"]

                def mkmask(df, cond):
                    if " == " in cond:
                        col, val = cond.split(" == ", 1)
                        val = val.strip().strip('"').strip("'")
                        s = df[col]
                        if pd.api.types.is_bool_dtype(s):
                            return s == (val.lower() == "true")
                        return s.astype(str) == val
                    return pd.Series(True, index=df.index)

                mask_pair = mkmask(df_entries, n1) & mkmask(df_entries, n2)
                m = _metrics(mask_pair, df_entries, side)
                if m:
                    item = {
                        "name": f"{n1} AND {n2}",
                        "all": [_as_clause(n1), _as_clause(n2)],
                        **m
                    }
                    if m["lift"] >= 1.0:
                        rules_pos[side].append(item)
                    if no_entry and (m["lift"] <= no_entry_lift_max):
                        key = "no_entry_bullish" if side == "bullish" else "no_entry_bearish"
                        rules_neg[key].append(item)
                evaluated += 1

        # Orden final
        for side in ("bullish", "bearish"):
            rules_pos[side].sort(key=lambda r: (r["lift"], r["win_rate"], r["min_support"]), reverse=True)
        for key in ("no_entry_bullish", "no_entry_bearish"):
            # para no_entry, prioriza lift más bajo (peor) y buen soporte
            rules_neg[key].sort(key=lambda r: (r["lift"], r["min_support"]))

        # === 3) Triples (limitadas) ===
        # Para acotar la explosión combinatoria, combinamos los mejores "singles" por lado.
        # tomamos los "top_singles" que ya usamos para pares, y combinamos de a 3.
        from itertools import combinations

        for side in ("bullish", "bearish"):
            # re-armar singles a partir de df_entries (evita dependencia del orden de rules_pos)
            # Usamos las mejores condiciones individuales segun lift ya calculadas en rules_pos[side]
            # pero nos quedamos solo con los primeros 'triples_from_top_k'
            best_single_names = [r["name"] for r in rules_pos[side] if " AND " not in r["name"]][:triples_from_top_k]

            evaluated = 0
            for n1, n2, n3 in combinations(best_single_names, 3):
                if evaluated >= max_triples:
                    break

                def mkmask(df, cond):
                    if " == " in cond:
                        col, val = cond.split(" == ", 1)
                        val = val.strip().strip('"').strip("'")
                        s = df[col]
                        if pd.api.types.is_bool_dtype(s):
                            return s == (val.lower() == "true")
                        return s.astype(str) == val
                    # por seguridad: no deberíamos caer aquí porque normalizamos arriba
                    return pd.Series(True, index=df.index)

                mask_triple = mkmask(df_entries, n1) & mkmask(df_entries, n2) & mkmask(df_entries, n3)

                # micro-pruning barato: si el soporte potencial ya es bajo, salta
                if mask_triple.sum() < min_support:
                    evaluated += 1
                    continue

                m = _metrics(mask_triple, df_entries, side)
                if m:
                    item = {
                        "name": f"{n1} AND {n2} AND {n3}",
                        "all": [_as_clause(n1), _as_clause(n2), _as_clause(n3)],
                        **m
                    }
                    if m["lift"] >= 1.0:
                        rules_pos[side].append(item)
                    if no_entry and (m["lift"] <= no_entry_lift_max):
                        key = "no_entry_bullish" if side == "bullish" else "no_entry_bearish"
                        rules_neg[key].append(item)

                evaluated += 1

        # Re-orden final tras incluir triples
        for side in ("bullish", "bearish"):
            rules_pos[side].sort(key=lambda r: (r["lift"], r["win_rate"], r["min_support"]), reverse=True)
        for key in ("no_entry_bullish", "no_entry_bearish"):
            rules_neg[key].sort(key=lambda r: (r.get("lift", 999.0), -r.get("min_support", 0)))

        return {
            "bullish": rules_pos["bullish"],
            "bearish": rules_pos["bearish"],
            "no_entry_bullish": rules_neg["no_entry_bullish"],
            "no_entry_bearish": rules_neg["no_entry_bearish"],
        }

    def _normalize_cond_name_to_clause(self, cond_name: str) -> dict:
        """
        Normaliza una condición textual a un dict {col, op, value} en formato deployable.
        Casos:
          - "col == Q1/Q2/Q3/Q4" -> {col, "==", "Qx"}
          - "col == True/False"  -> {col, "==", True/False}
        Si no hay ' == ', asumimos una señal booleana positiva: "flag" -> {flag, "==", True}
        """
        if " == " in cond_name:
            col, val = cond_name.split(" == ", 1)
            col = col.strip()
            val = val.strip().strip('"').strip("'")
            # cast boolean si aplica
            if val.lower() in ("true", "false"):
                value = (val.lower() == "true")
            else:
                value = val
            return {"col": col, "op": "==", "value": value}
        else:
            # Señal booleana afirmada
            return {"col": cond_name.strip(), "op": "==", "value": True}

    def export_rules_deployable(self,
                                rules: dict | None = None,
                                top_k_per_side: int = 100,
                                round_ndigits: int = 6,
                                min_support: int | None = None,
                                min_lift: float | None = None,
                                min_win_rate: float | None = None,
                                # filtros específicos para NO ENTRY:
                                no_entry_top_k: int = 100,
                                no_entry_min_support: int | None = None,
                                no_entry_max_lift: float | None = 0.9,
                                save_path: str | None = None) -> dict:
        """
        Exporta en formato deployable con 4 llaves:
          'bullish', 'bearish', 'no_entry_bullish', 'no_entry_bearish'
        """
        import json, math
        from pathlib import Path

        if rules is None:
            rules = self.autogen_rules_from_entries()

        deploy = {"bullish": [], "bearish": [], "no_entry_bullish": [], "no_entry_bearish": []}

        def _passes_pos_filters(r):
            if min_support is not None and int(r.get("min_support", 0)) < min_support:
                return False
            if min_lift is not None:
                lift = r.get("lift", float("nan"))
                if lift is None or (isinstance(lift, float) and math.isnan(lift)) or lift < min_lift:
                    return False
            if min_win_rate is not None and r.get("win_rate", 0.0) < min_win_rate:
                return False
            return True

        def _passes_neg_filters(r):
            if no_entry_min_support is not None and int(r.get("min_support", 0)) < no_entry_min_support:
                return False
            if no_entry_max_lift is not None:
                lift = r.get("lift", float("nan"))
                if lift is None or (isinstance(lift, float) and math.isnan(lift)) or lift > no_entry_max_lift:
                    return False
            return True

        def _norm_cond(name: str) -> dict:
            if " == " in name:
                col, val = name.split(" == ", 1)
                val = val.strip().strip('"').strip("'")
                if val.lower() in ("true", "false"):
                    value = (val.lower() == "true")
                else:
                    value = val
                return {"col": col.strip(), "op": "==", "value": value}
            return {"col": name.strip(), "op": "==", "value": True}

        # POSITIVAS
        for side in ("bullish", "bearish"):
            candidates = sorted(rules.get(side, []),
                                key=lambda r: (r.get("lift", 0.0), r.get("win_rate", 0.0), r.get("min_support", 0)),
                                reverse=True)
            out = []
            for r in candidates:
                if not _passes_pos_filters(r):
                    continue
                name = r["name"]
                if " AND " in name:
                    conds = [_norm_cond(p.strip()) for p in name.split(" AND ")]
                else:
                    conds = [_norm_cond(name)]
                out.append({
                    "name": name,
                    "all": conds,
                    "min_support": int(r.get("min_support", 0)),
                    "win_rate": round(float(r.get("win_rate", 0.0)), round_ndigits),
                    "lift": round(float(r.get("lift", 0.0)), round_ndigits),
                })
                if len(out) >= top_k_per_side:
                    break
            deploy[side] = out

        # NO ENTRY
        for key in ("no_entry_bullish", "no_entry_bearish"):
            candidates = rules.get(key, [])
            # Para no_entry, ordena por lift ascendente (peor primero), luego mayor soporte
            candidates = sorted(candidates, key=lambda r: (r.get("lift", 999.0), -r.get("min_support", 0)))
            out = []
            for r in candidates:
                if not _passes_neg_filters(r):
                    continue
                name = r["name"]
                if " AND " in name:
                    conds = [_norm_cond(p.strip()) for p in name.split(" AND ")]
                else:
                    conds = [_norm_cond(name)]
                out.append({
                    "name": name,
                    "all": conds,
                    "min_support": int(r.get("min_support", 0)),
                    "win_rate": round(float(r.get("win_rate", 0.0)), round_ndigits),
                    "lift": round(float(r.get("lift", 0.0)), round_ndigits),
                })
                if len(out) >= no_entry_top_k:
                    break
            deploy[key] = out

        if save_path:
            p = Path(save_path)
            content_json = json.dumps(deploy, ensure_ascii=False, indent=2)
            if p.suffix.lower() == ".json":
                p.write_text(content_json)
            elif p.suffix.lower() == ".py":
                p.write_text(
                    "# Auto-generated rules (deployable format)\n# coding: utf-8\n\nrules = " + content_json + "\n")
            else:
                p.write_text(content_json)

        return deploy

    def _update_rule_stats_on_close(self, trade, pnl_usdt: float):
        """Acumula stats por regla cuando un trade se cierra."""
        entry_idx = getattr(trade, "entry_index", None)
        rule_name = getattr(trade, "entry_rule", None)
        side_txt = getattr(trade, "entry_rule_side", None)

        # intenta recuperar de self.df si no vino en el objeto
        if entry_idx is not None:
            if rule_name is None and "entry_rule" in self.df.columns:
                rule_name = self.df.at[entry_idx, "entry_rule"]
            if side_txt is None and "entry_rule_side" in self.df.columns:
                side_txt = self.df.at[entry_idx, "entry_rule_side"]

        # si el trade no fue abierto por una regla (None), no contamos
        if rule_name in (None, ""):
            return

        rs = self.rule_stats[rule_name]
        if side_txt:
            rs["side"] = side_txt

        rs["used"] += 1
        rs["pnl_sum"] += float(pnl_usdt)

        if pnl_usdt > 0:
            rs["wins"] += 1
        elif pnl_usdt < 0:
            rs["losses"] += 1
        else:
            rs["breakeven"] += 1

    def plot_performance(self, 
                        initial_equity: Optional[float] = None,
                        save_html: Optional[str] = None,
                        show_plot: bool = True,
                        height: int = 2400) -> Optional[go.Figure]:
        """
        Genera visualizaciones interactivas del desempeño de la simulación.
        
        Args:
            initial_equity: Capital inicial (si None, usa self.equity final - sum(pnl))
            save_html: Ruta para guardar el gráfico como HTML interactivo
            show_plot: Si True, muestra el gráfico en navegador
            height: Altura total del gráfico en píxeles
            
        Returns:
            Figura de plotly con todos los subplots
        """
        if not self.trades:
            print("⚠️ No hay trades para graficar. Ejecuta run() primero.")
            return None
        
        # Convertir trades a DataFrame
        trades_rows = []
        for t in self.trades:
            trades_rows.append({
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "leverage": getattr(t, "leverage", None),
                "exit_reason": t.exit_reason,
                "entry_proba": getattr(t, "entry_proba", None),
                "entry_pred": getattr(t, "entry_pred", None),
                "fees": t.fees_usdt() if hasattr(t, "fees_usdt") else 0,
                "pnl": t.pnl_usdt() if hasattr(t, "pnl_usdt") else 0,
            })
        
        df_trades = pd.DataFrame(trades_rows)
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
        df_trades['pnl_net'] = df_trades['pnl'] - df_trades['fees']
        df_trades['result'] = df_trades['pnl_net'].apply(
            lambda x: 'win' if x > 0 else ('loss' if x < 0 else 'breakeven')
        )
        df_trades['duration_min'] = (df_trades['exit_time'] - df_trades['entry_time']).dt.total_seconds() / 60
        
        # Calcular equity curve
        if initial_equity is None:
            initial_equity = 1000.0  # Default
        
        df_trades = df_trades.sort_values('exit_time').reset_index(drop=True)
        df_trades['equity'] = initial_equity + df_trades['pnl_net'].cumsum()
        df_trades['cumulative_pnl'] = df_trades['pnl_net'].cumsum()
        
        # Calcular drawdown
        running_max = df_trades['equity'].expanding().max()
        df_trades['drawdown'] = (df_trades['equity'] - running_max) / running_max * 100
        
        # Crear subplots
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=(
                'Curva de Equity', 'Drawdown (%)',
                'PnL por Trade', 'PnL Acumulado',
                'Win Rate por Período (ventana 20 trades)', 'Duración de Trades (minutos)',
                'Distribución de PnL', 'Trades por Exit Reason',
                'Precio con Entradas/Salidas (últimos 500 datos)', 'Probabilidad vs PnL'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}]
            ],
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            row_heights=[0.15, 0.15, 0.15, 0.15, 0.20, 0.20]
        )
        
        # 1. Curva de Equity
        fig.add_trace(
            go.Scatter(
                x=df_trades['exit_time'],
                y=df_trades['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='#2E86AB', width=2),
                hovertemplate='<b>%{x}</b><br>Equity: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Línea de equity inicial
        fig.add_hline(
            y=initial_equity, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.5,
            row=1, col=1
        )
        
        # ====== VISUALIZACIÓN DE RETIROS ======
        if hasattr(self, 'withdrawals') and self.withdrawals:
            # Agregar marcadores verticales para cada retiro
            for withdrawal in self.withdrawals:
                # FIX: Convertir a milisegundos para evitar TypeError en plotly/pandas
                ts_ms = withdrawal['timestamp'].timestamp() * 1000
                fig.add_vline(
                    x=ts_ms,
                    line_dash="dot",
                    line_color="orange",
                    opacity=0.6,
                    row=1, col=1,
                    annotation_text=f"💰${withdrawal['amount']:.0f}",
                    annotation_position="top"
                )
            
            # Crear DataFrame de retiros para agregar línea de capital total
            df_withdrawals = pd.DataFrame(self.withdrawals)
            df_withdrawals['timestamp'] = pd.to_datetime(df_withdrawals['timestamp'])
            
            # Construir serie de capital total (equity + withdrawn) a lo largo del tiempo
            # Necesitamos sincronizar con df_trades para tener puntos en el tiempo
            df_total_capital = df_trades[['exit_time', 'equity']].copy()
            df_total_capital['total_capital'] = df_total_capital['equity']
            
            # Para cada punto en el tiempo, sumar el capital retirado acumulado hasta ese momento
            for idx in df_total_capital.index:
                exit_time = df_total_capital.loc[idx, 'exit_time']
                # Sumar todos los retiros que ocurrieron antes o en este momento
                withdrawn_so_far = df_withdrawals[df_withdrawals['timestamp'] <= exit_time]['amount'].sum()
                df_total_capital.loc[idx, 'total_capital'] += withdrawn_so_far
            
            # Agregar línea de capital total (semi-transparente)
            fig.add_trace(
                go.Scatter(
                    x=df_total_capital['exit_time'],
                    y=df_total_capital['total_capital'],
                    mode='lines',
                    name='Capital Total (Equity + Retirado)',
                    line=dict(color='#FFA500', width=2, dash='dash'),
                    opacity=0.7,
                    hovertemplate='<b>%{x}</b><br>Capital Total: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )

        
        # 2. Drawdown
        fig.add_trace(
            go.Scatter(
                x=df_trades['exit_time'],
                y=df_trades['drawdown'],
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='#A23B72', width=2),
                hovertemplate='<b>%{x}</b><br>DD: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. PnL por Trade
        colors = df_trades['result'].map({'win': '#06D6A0', 'loss': '#EF476F', 'breakeven': '#FFC43D'})
        fig.add_trace(
            go.Bar(
                x=df_trades.index,
                y=df_trades['pnl_net'],
                marker_color=colors,
                name='PnL por Trade',
                hovertemplate='Trade #%{x}<br>PnL: $%{y:.2f}<br>%{customdata}<extra></extra>',
                customdata=df_trades['exit_reason']
            ),
            row=2, col=1
        )
        
        # 4. PnL Acumulado
        fig.add_trace(
            go.Scatter(
                x=df_trades['exit_time'],
                y=df_trades['cumulative_pnl'],
                mode='lines',
                name='PnL Acumulado',
                line=dict(color='#118AB2', width=2),
                fill='tozeroy',
                hovertemplate='<b>%{x}</b><br>PnL Acum: $%{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Win Rate por Período (ventana móvil)
        window = 20
        if len(df_trades) >= window:
            # Convertir a valores numéricos primero (1 para win, 0 para otros)
            df_trades['is_win'] = (df_trades['result'] == 'win').astype(int)
            df_trades['win_roll'] = df_trades['is_win'].rolling(window).mean() * 100
            fig.add_trace(
                go.Scatter(
                    x=df_trades['exit_time'],
                    y=df_trades['win_roll'],
                    mode='lines',
                    name=f'Win Rate (Rolling {window})',
                    line=dict(color='#073B4C', width=2),
                    hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Línea de 50%
            fig.add_hline(
                y=50, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                row=3, col=1
            )
        
        # 6. Duración de Trades
        colors_duration = df_trades['result'].map({'win': '#06D6A0', 'loss': '#EF476F', 'breakeven': '#FFC43D'})
        fig.add_trace(
            go.Scatter(
                x=df_trades.index,
                y=df_trades['duration_min'],
                mode='markers',
                name='Duración',
                marker=dict(
                    size=8,
                    color=colors_duration,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='Trade #%{x}<br>Duración: %{y:.1f} min<br>%{customdata}<extra></extra>',
                customdata=df_trades['result']
            ),
            row=3, col=2
        )
        
        # 7. Distribución de PnL (Histograma)
        fig.add_trace(
            go.Histogram(
                x=df_trades['pnl_net'],
                nbinsx=30,
                name='Distribución PnL',
                marker_color='#2E86AB',
                hovertemplate='PnL Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # 8. Trades por Exit Reason
        exit_counts = df_trades['exit_reason'].value_counts()
        fig.add_trace(
            go.Bar(
                x=exit_counts.index,
                y=exit_counts.values,
                name='Exit Reason',
                marker_color=['#06D6A0' if r == 'tp' else '#EF476F' if r == 'sl' else '#FFC43D' 
                             for r in exit_counts.index],
                hovertemplate='%{x}: %{y} trades<extra></extra>'
            ),
            row=4, col=2
        )
        
        # 9. Precio con Entradas/Salidas
        # Tomar últimos N datos para no saturar el gráfico
        n_candles = min(500, len(self.df))
        df_plot = self.df.tail(n_candles).copy()
        
        fig.add_trace(
            go.Scatter(
                x=df_plot['datetime'],
                y=df_plot['close'],
                mode='lines',
                name='Precio',
                line=dict(color='#073B4C', width=1),
                hovertemplate='<b>%{x}</b><br>Close: %{y:.4f}<extra></extra>'
            ),
            row=5, col=1
        )
        
        # Marcar entradas y salidas
        df_entries = df_trades[df_trades['entry_time'].isin(df_plot['datetime'])]
        df_exits = df_trades[df_trades['exit_time'].isin(df_plot['datetime'])]
        
        # Entradas Long
        entries_long = df_entries[df_entries['side'] == 'long']
        if not entries_long.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries_long['entry_time'],
                    y=entries_long['entry_price'],
                    mode='markers',
                    name='Entry Long',
                    marker=dict(symbol='triangle-up', size=10, color='#06D6A0', line=dict(width=1, color='white')),
                    hovertemplate='<b>Long Entry</b><br>%{x}<br>Price: %{y:.4f}<extra></extra>'
                ),
                row=5, col=1
            )
        
        # Entradas Short
        entries_short = df_entries[df_entries['side'] == 'short']
        if not entries_short.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries_short['entry_time'],
                    y=entries_short['entry_price'],
                    mode='markers',
                    name='Entry Short',
                    marker=dict(symbol='triangle-down', size=10, color='#EF476F', line=dict(width=1, color='white')),
                    hovertemplate='<b>Short Entry</b><br>%{x}<br>Price: %{y:.4f}<extra></extra>'
                ),
                row=5, col=1
            )
        
        # Salidas (TP en verde, SL en rojo, otros en amarillo)
        for reason, color, symbol in [('tp', '#06D6A0', 'star'), ('sl', '#EF476F', 'x'), ('early_opposite', '#FFC43D', 'circle')]:
            exits_filtered = df_exits[df_exits['exit_reason'] == reason]
            if not exits_filtered.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits_filtered['exit_time'],
                        y=exits_filtered['exit_price'],
                        mode='markers',
                        name=f'Exit {reason.upper()}',
                        marker=dict(symbol=symbol, size=8, color=color, line=dict(width=1, color='white')),
                        hovertemplate=f'<b>Exit {reason.upper()}</b><br>%{{x}}<br>Price: %{{y:.4f}}<extra></extra>'
                    ),
                    row=5, col=1
                )
        
        # 10. Probabilidad vs PnL (scatter para ver correlación)
        df_with_proba = df_trades.dropna(subset=['entry_proba'])
        if not df_with_proba.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_with_proba['entry_proba'],
                    y=df_with_proba['pnl_net'],
                    mode='markers',
                    name='Proba vs PnL',
                    marker=dict(
                        size=8,
                        color=df_with_proba['pnl_net'],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="PnL", x=1.15),
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='Proba: %{x:.3f}<br>PnL: $%{y:.2f}<extra></extra>'
                ),
                row=6, col=1
            )
        
        # 11. Win Rate por Bucket de Probabilidad
        if not df_with_proba.empty:
            # Crear buckets de probabilidad
            df_with_proba['proba_bucket'] = pd.cut(
                df_with_proba['entry_proba'], 
                bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['0.0-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
            )
            
            bucket_stats = df_with_proba.groupby('proba_bucket').agg({
                'result': lambda x: (x == 'win').sum() / len(x) * 100,
                'pnl_net': 'count'
            }).reset_index()
            bucket_stats.columns = ['proba_bucket', 'win_rate', 'count']
            
            fig.add_trace(
                go.Bar(
                    x=bucket_stats['proba_bucket'].astype(str),
                    y=bucket_stats['win_rate'],
                    name='Win Rate por Bucket',
                    marker_color='#2E86AB',
                    text=bucket_stats['count'],
                    texttemplate='n=%{text}',
                    textposition='outside',
                    hovertemplate='Bucket: %{x}<br>Win Rate: %{y:.1f}%<br>Trades: %{text}<extra></extra>'
                ),
                row=6, col=2
            )
            
            # Línea de 50%
            fig.add_hline(
                y=50, 
                line_dash="dash", 
                line_color="gray", 
                opacity=0.5,
                row=6, col=2
            )
        
        # Actualizar layout
        # Construir título con información de retiros si existen
        title_parts = [
            f"<b>Reporte de Desempeño - BackTesting</b><br>",
            f"<sup>Trades: {len(self.trades)} | Final Equity: ${df_trades['equity'].iloc[-1]:.2f} | ",
            f"Win Rate: {(df_trades['result'] == 'win').sum() / len(df_trades) * 100:.1f}%"
        ]
        
        if hasattr(self, 'withdrawals') and self.withdrawals:
            total_withdrawn = self.withdrawn_capital
            n_withdrawals = len(self.withdrawals)
            total_capital = df_trades['equity'].iloc[-1] + total_withdrawn
            title_parts.append(f" | 💰 Retirado: ${total_withdrawn:.2f} ({n_withdrawals} retiros) | Capital Total: ${total_capital:.2f}")
        
        title_parts.append("</sup>")
        title_text = "".join(title_parts)
        
        fig.update_layout(
            height=height,
            title_text=title_text,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=10)
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Tiempo", row=1, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        
        fig.update_xaxes(title_text="Tiempo", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Trade #", row=2, col=1)
        fig.update_yaxes(title_text="PnL ($)", row=2, col=1)
        
        fig.update_xaxes(title_text="Tiempo", row=2, col=2)
        fig.update_yaxes(title_text="PnL Acumulado ($)", row=2, col=2)
        
        fig.update_xaxes(title_text="Tiempo", row=3, col=1)
        fig.update_yaxes(title_text="Win Rate (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Trade #", row=3, col=2)
        fig.update_yaxes(title_text="Duración (min)", row=3, col=2)
        
        fig.update_xaxes(title_text="PnL Net ($)", row=4, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=4, col=1)
        
        fig.update_xaxes(title_text="Exit Reason", row=4, col=2)
        fig.update_yaxes(title_text="Count", row=4, col=2)
        
        fig.update_xaxes(title_text="Tiempo", row=5, col=1)
        fig.update_yaxes(title_text="Precio", row=5, col=1)
        
        fig.update_xaxes(title_text="Probabilidad", row=6, col=1)
        fig.update_yaxes(title_text="PnL Net ($)", row=6, col=1)
        
        fig.update_xaxes(title_text="Bucket de Probabilidad", row=6, col=2)
        fig.update_yaxes(title_text="Win Rate (%)", row=6, col=2)
        
        # Guardar si se especificó
        if save_html:
            fig.write_html(save_html)
            print(f"✅ Gráfico guardado en: {save_html}")
        
        # Mostrar si se especificó
        if show_plot:
            fig.show()
        
        return fig


def _now_iso():
    # guardamos en UTC con marca ISO + guardamos hora local aparte
    return datetime.now(timezone.utc).isoformat()


class TradeLogger:
    """
    Logger de trading estilo 'journal':
    - Cada evento se escribe en JSONL (append-only) para auditoría.
    - Mantiene un estado de posiciones abiertas en memoria para control rápido.
    - Permite calcular estadísticas a partir del log.
    """

    def __init__(self, log_path="trade_log.jsonl"):
        self.log_path = log_path
        # open_positions: key=(symbol), value=dict con datos de la posición abierta
        self.open_positions = {}  # { "AVAXUSDT": {...} }
        # para series de equity (si registras PnL intradía)
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", encoding="utf-8") as f:
                pass

    def _write(self, record: dict):
        record["ts_utc"] = _now_iso()
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -------- Señales y órdenes --------
    def log_signal(self, *, symbol, timeframe, side, prob, margin_pp, probs_dict, kelly, rr, take_profit, stop_loss,
                   price, reason, gate):
        """
        Registra la señal que gatilla la operación (antes de enviar orden).
        Devuelve un trade_id para seguir la trazabilidad.
        """
        trade_id = str(uuid.uuid4())
        rec = {
            "event": "signal",
            "trade_id": trade_id,
            "symbol": symbol,
            "timeframe": timeframe,
            "side": side,  # 'long'/'short'
            "prob": prob,
            "margin_pp": margin_pp,  # top1-top2 en puntos porcentuales
            "probs": probs_dict,
            "kelly": kelly,
            "reward_risk": rr,
            "entry_price": price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "gate": gate,  # 'soft' / 'strong'
            "reason": reason  # texto libre: p.ej. "u_soft >= 0.80 & margin_ok"
        }
        self._write(rec)
        return trade_id

    def log_order_attempt(self, *, trade_id, symbol, side, qty, payload):
        self._write({
            "event": "order_attempt",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "payload": payload
        })

    def log_order_ack(self, *, trade_id, symbol, side, qty, broker_raw, order_id):
        """
        Acknowledge de orden aceptada por el broker.
        Marca posición como abierta.
        """
        self._write({
            "event": "order_ack",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_id": order_id,
            "broker_raw": broker_raw
        })
        # Guardamos estado de posición abierta (si no existía)
        # NOTA: asumimos una sola posición por símbolo (one-way). Si usas hedge-mode, habría que separar por side.
        self.open_positions[symbol] = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_id": order_id,
            "opened_utc": _now_iso()
        }

    def log_order_reject(self, *, trade_id, symbol, side, qty, broker_raw):
        """
        Incluye casos como "Cambio de posición no permitido".
        """
        self._write({
            "event": "order_reject",
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "broker_raw": broker_raw
        })

    # -------- Cierres y PnL --------
    def log_close(self, *, symbol, exit_reason, exit_price, pnl_usdt, pnl_r, order_id=None, trade_id=None):
        """
        Registra el cierre de la posición. Puedes llamarlo cuando detectes:
        - Size vuelve a 0 (poll al exchange)
        - Te llega webhook de TP/SL lleno
        - Cierre manual
        """
        # Busca trade_id si no lo pasaron
        if trade_id is None:
            info = self.open_positions.get(symbol)
            trade_id = info["trade_id"] if info else str(uuid.uuid4())

        rec = {
            "event": "close",
            "trade_id": trade_id,
            "symbol": symbol,
            "exit_reason": exit_reason,  # 'tp'/'sl'/'manual'/'error'
            "exit_price": exit_price,
            "pnl_usdt": pnl_usdt,
            "pnl_r": pnl_r,  # PnL en múltiplos de riesgo (R)
            "order_id": order_id
        }
        self._write(rec)
        # Limpia estado en memoria
        if symbol in self.open_positions:
            self.open_positions.pop(symbol, None)

    def log_info(self, **kwargs):
        rec = {"event": "info"}
        rec.update(kwargs)
        self._write(rec)

    # -------- Utilidades --------
    def has_open_position(self, symbol, side=None):
        """
        Retorna True si hay una posición abierta en 'symbol'.
        Si pasas 'side', valida que coincida.
        """
        info = self.open_positions.get(symbol)
        if not info:
            return False
        if side is None:
            return True
        return info.get("side") == side

    # -------- Estadísticas --------
    def compute_stats(self, pretty: bool = False, return_frames: bool = False):
        """
        Lee el JSONL y arma KPIs + breakdowns.
        - pretty=True  → retorna (stats_dict, pretty_text)
        - return_frames=True → además retorna (df_trades, df_side, df_gate)
        - por defecto retorna solo stats_dict

        Requisitos: pandas, numpy.
        """

        def _to_dt(x):
            try:
                return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
            except Exception:
                return pd.NaT

        def _pick(df, base):
            # Devuelve la columna si existe o su variante con sufijo _x/_y tras merge
            for c in (base, base + "_x", base + "_y"):
                if c in df.columns:
                    return c
            return None

        def _streaks(wins_series):
            # wins_series: 1=win, 0=loss
            best = worst = cur_w = cur_l = 0
            for w in wins_series.astype(int).tolist():
                if w == 1:
                    cur_w += 1;
                    best = max(best, cur_w)
                    cur_l = 0
                else:
                    cur_l += 1;
                    worst = max(worst, cur_l)
                    cur_w = 0
            return best, worst

        if not os.path.exists(self.log_path):
            return {"trades": 0}

        # --- Carga JSONL ---
        rows = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        if not rows:
            return {"trades": 0}

        df = pd.DataFrame(rows)

        # --- Filtra eventos relevantes ---
        closes = df[df["event"] == "close"].copy()
        signals = df[df["event"] == "signal"].copy()

        if closes.empty or signals.empty:
            out = {"trades": 0}
            return (out, "No hay trades cerrados.") if pretty else out

        # Dejar la PRIMERA señal por trade_id (origen del trade)
        signals = signals.sort_values("ts_utc").drop_duplicates(subset=["trade_id"], keep="first")

        # Selección robusta de columnas de signal
        sig_cols = ["trade_id", "symbol", "side", "reward_risk", "entry_price",
                    "take_profit", "stop_loss", "prob", "kelly", "gate", "ts_utc"]
        for c in sig_cols:
            if c not in signals.columns:
                signals[c] = np.nan
        signals = signals[sig_cols].rename(columns={"ts_utc": "signal_ts_utc"})

        # Merge close <- signal
        trades = closes.merge(signals, on="trade_id", how="left", suffixes=("", "_sig"))

        # Parseo de timestamps
        trades["close_ts_utc"] = trades["ts_utc"].apply(_to_dt)
        trades["signal_ts_utc"] = trades["signal_ts_utc"].apply(_to_dt)
        trades = trades.sort_values("close_ts_utc")

        # Duración (min)
        trades["duration_min"] = (trades["close_ts_utc"] - trades["signal_ts_utc"]).dt.total_seconds() / 60.0

        # Win/loss
        trades["win"] = (trades["pnl_usdt"] > 0).astype(int)
        sum_wins = trades.loc[trades["win"] == 1, "pnl_usdt"].sum()
        sum_losses = trades.loc[trades["win"] == 0, "pnl_usdt"].abs().sum()
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (np.inf if sum_wins > 0 else 0.0)
        winrate = trades["win"].mean() if len(trades) else 0.0

        # Expectancy en R
        has_r = "pnl_r" in trades.columns and trades["pnl_r"].notna().any()
        avg_r = trades["pnl_r"].mean() if has_r else np.nan
        expectancy_r = avg_r

        # Equity y DD
        trades["cum_pnl"] = trades["pnl_usdt"].cumsum()
        running_max = trades["cum_pnl"].cummax()
        drawdown = trades["cum_pnl"] - running_max
        max_dd = drawdown.min() if not drawdown.empty else 0.0

        # Métricas extra
        avg_win = trades.loc[trades["win"] == 1, "pnl_usdt"].mean()
        avg_loss = trades.loc[trades["win"] == 0, "pnl_usdt"].mean()
        med_pnl = trades["pnl_usdt"].median()
        best_trade = trades.loc[trades["pnl_usdt"].idxmax()] if not trades.empty else None
        worst_trade = trades.loc[trades["pnl_usdt"].idxmin()] if not trades.empty else None
        mean_dur = trades["duration_min"].mean()
        med_dur = trades["duration_min"].median()

        # Rachas
        best_streak, worst_streak = _streaks(trades["win"])

        # Sharpe por trade (no anualizado)
        mu = trades["pnl_usdt"].mean()
        sd = trades["pnl_usdt"].std(ddof=1)
        sharpe_tr = (mu / sd) if sd and not np.isnan(sd) and sd != 0 else np.nan

        total_pnl = trades["pnl_usdt"].sum()

        # Breakdown por side
        if "side" in trades.columns:
            side_grp = trades.groupby("side", dropna=False)["pnl_usdt"].agg(
                trades="count", pnl_total="sum", pnl_avg="mean"
            ).reset_index()
            # winrate por side
            wr_side = trades.groupby("side", dropna=False)["win"].mean().reset_index().rename(
                columns={"win": "winrate"})
            side_grp = side_grp.merge(wr_side, on="side", how="left")
        else:
            side_grp = pd.DataFrame()

        # Breakdown por gate
        if "gate" in trades.columns:
            gate_grp = trades.groupby("gate", dropna=False)["pnl_usdt"].agg(
                trades="count", pnl_total="sum", pnl_avg="mean"
            ).reset_index()
            wr_gate = trades.groupby("gate", dropna=False)["win"].mean().reset_index().rename(
                columns={"win": "winrate"})
            gate_grp = gate_grp.merge(wr_gate, on="gate", how="left")
        else:
            gate_grp = pd.DataFrame()

        # Salida principal (dict)
        out = {
            "trades": int(len(trades)),
            "winrate": float(winrate),
            "profit_factor": float(profit_factor),
            "avg_R": (float(avg_r) if not np.isnan(avg_r) else None),
            "expectancy_R": (float(expectancy_r) if not np.isnan(expectancy_r) else None),
            "avg_win_usdt": (float(avg_win) if not np.isnan(avg_win) else None),
            "avg_loss_usdt": (float(avg_loss) if not np.isnan(avg_loss) else None),
            "median_pnl_usdt": (float(med_pnl) if not np.isnan(med_pnl) else None),
            "max_drawdown_usdt": float(max_dd),
            "total_pnl_usdt": float(total_pnl),
            "mean_duration_min": (float(mean_dur) if not np.isnan(mean_dur) else None),
            "median_duration_min": (float(med_dur) if not np.isnan(med_dur) else None),
            "best_streak_wins": int(best_streak),
            "worst_streak_losses": int(worst_streak),
            "sharpe_per_trade": (float(sharpe_tr) if not np.isnan(sharpe_tr) else None),
            "best_trade": {
                "trade_id": best_trade["trade_id"],
                "pnl_usdt": float(best_trade["pnl_usdt"]),
                "close_ts_utc": str(best_trade["close_ts_utc"])
            } if best_trade is not None else None,
            "worst_trade": {
                "trade_id": worst_trade["trade_id"],
                "pnl_usdt": float(worst_trade["pnl_usdt"]),
                "close_ts_utc": str(worst_trade["close_ts_utc"])
            } if worst_trade is not None else None,
        }

        # Tabla bonita opcional (texto)
        if pretty:
            def f(v):
                if v is None: return "—"
                if isinstance(v, float):
                    return "∞" if np.isinf(v) else f"{v:.2f}"
                return str(v)

            lines = []
            lines.append("═ KPIs Generales ═")
            lines.append(f"Trades cerrados         : {out['trades']}")
            lines.append(f"Win rate                : {out['winrate'] * 100:.2f}%")
            lines.append(
                f"Profit factor           : {('∞' if np.isinf(out['profit_factor']) else f(out['profit_factor']))}")
            lines.append(f"Expectancy (R/trade)    : {f(out['expectancy_R'])}")
            lines.append(f"Avg R                   : {f(out['avg_R'])}")
            lines.append(f"Avg win (USDT)          : {f(out['avg_win_usdt'])}")
            lines.append(f"Avg loss (USDT)         : {f(out['avg_loss_usdt'])}")
            lines.append(f"Median PnL (USDT)       : {f(out['median_pnl_usdt'])}")
            lines.append(f"Max drawdown (USDT)     : {f(out['max_drawdown_usdt'])}")
            lines.append(f"PnL neto total (USDT)   : {f(out['total_pnl_usdt'])}")
            lines.append(f"Duración media (min)    : {f(out['mean_duration_min'])}")
            lines.append(f"Duración mediana (min)  : {f(out['median_duration_min'])}")
            lines.append(f"Mejor racha (wins)      : {out['best_streak_wins']}")
            lines.append(f"Peor racha (losses)     : {out['worst_streak_losses']}")
            lines.append(f"Sharpe por trade        : {f(out['sharpe_per_trade'])}")

            # Best/Worst
            if out["best_trade"]:
                lines.append(
                    f"Best trade              : {out['best_trade']['trade_id']}  ({out['best_trade']['pnl_usdt']:.2f} USDT)")
            if out["worst_trade"]:
                lines.append(
                    f"Worst trade             : {out['worst_trade']['trade_id']} ({out['worst_trade']['pnl_usdt']:.2f} USDT)")

            # Breakdown por side
            if not side_grp.empty:
                lines.append("\n─ Breakdown por side ─")
                lines.append("side   trades   winrate     pnl_total    pnl_avg")
                for _, r in side_grp.iterrows():
                    wr = f"{(r['winrate'] * 100):.1f}%"
                    lines.append(
                        f"{str(r['side']):5}  {int(r['trades']):6d}   {wr:>7}   {r['pnl_total']:.2f}    {r['pnl_avg']:.2f}")

            # Breakdown por gate
            if not gate_grp.empty:
                lines.append("\n─ Breakdown por gate ─")
                lines.append("gate   trades   winrate     pnl_total    pnl_avg")
                for _, r in gate_grp.iterrows():
                    g = str(r['gate']) if pd.notna(r['gate']) else "None"
                    wr = f"{(r['winrate'] * 100):.1f}%"
                    lines.append(f"{g:5}  {int(r['trades']):6d}   {wr:>7}   {r['pnl_total']:.2f}    {r['pnl_avg']:.2f}")

            pretty_text = "\n".join(lines)

        # Retornos según flags
        if pretty and return_frames:
            return out, pretty_text, trades, side_grp, gate_grp
        elif pretty:
            return out, pretty_text
        elif return_frames:
            return out, trades, side_grp, gate_grp
        else:
            return out

    def enrich_open_position(self, symbol, **kwargs):
        """Añade datos como entry_price, tp/sl, fee_rate, etc. a la posición abierta."""
        if symbol in self.open_positions:
            self.open_positions[symbol].update(kwargs)

    def get_open_position(self, symbol):
        return self.open_positions.get(symbol)

    def set_open_position(self, symbol, posdict):
        self.open_positions[symbol] = posdict


def _to_importance_series(importances, available_cols):
    """
    Acepta:
      - pd.Series: index = feature, value = importance
      - pd.DataFrame con columnas ['feature','importance']
    Devuelve:
      - pd.Series alineada a available_cols (faltantes -> 0)
    """
    if isinstance(importances, pd.DataFrame):
        if {'feature', 'importance'}.issubset(importances.columns):
            imp = importances.set_index('feature')['importance']
        else:
            raise ValueError("Si pasas DataFrame de importancias debe tener columnas ['feature','importance'].")
    elif isinstance(importances, pd.Series):
        imp = importances.copy()
    else:
        raise ValueError("importances debe ser pd.Series o pd.DataFrame(['feature','importance']).")

    # Alinear a columnas disponibles; faltantes con 0
    imp = imp.reindex(available_cols).fillna(0.0)
    return imp


def select_features_by_correlation(
        X: pd.DataFrame,
        feature_names: list,
        importances,  # Series o DataFrame (feature, importance)
        corr_threshold: float = 0.90,
        impute: bool = True
):
    """
    Filtra colinealidad en X usando solo columnas en feature_names.
    Mantiene, dentro de cada cluster correlacionado, el feature más importante.

    Retorna:
      X_sel: DataFrame con columnas seleccionadas
      selected_cols: lista de columnas seleccionadas
      dropped_cols: lista de columnas eliminadas por alta correlación
      corr_matrix: matriz de correlación absoluta usada para la selección
    """
    # 1) Subset a features solicitados
    Xf = X.loc[:, [c for c in feature_names if c in X.columns]].copy()

    # 2) Sanitizar: solo numéricas, reemplazar inf, quitar constantes y all-NaN
    Xf = Xf.replace([np.inf, -np.inf], np.nan)

    # Forzar numérico si algo quedó como object (coerce -> NaN)
    for c in Xf.columns:
        if not np.issubdtype(Xf[c].dtype, np.number):
            Xf[c] = pd.to_numeric(Xf[c], errors='coerce')

    # Eliminar columnas constantes
    const_cols = [c for c in Xf.columns if Xf[c].nunique(dropna=True) <= 1]
    if const_cols:
        Xf.drop(columns=const_cols, inplace=True)

    # Imputación simple (opcional) para poder calcular corr
    if impute:
        Xf = Xf.fillna(method='ffill').fillna(method='bfill')

    # Eliminar columnas que quedaron totalmente NaN
    allnan = [c for c in Xf.columns if Xf[c].isna().all()]
    if allnan:
        Xf.drop(columns=allnan, inplace=True)

    # Si después de limpiar no hay suficientes columnas, salir
    if Xf.shape[1] <= 1:
        imp_series = _to_importance_series(importances, Xf.columns)
        return Xf, list(Xf.columns), [], Xf.corr().abs() if Xf.shape[1] > 0 else pd.DataFrame()

    # 3) Matriz de correlación absoluta
    corr = Xf.corr().abs()

    # 4) Orden por importancia (desc)
    imp_series = _to_importance_series(importances, Xf.columns)
    ordered = imp_series.sort_values(ascending=False).index.tolist()

    # 5) Selección greedy: conserva el más importante y elimina sus “vecinos” > umbral
    selected = []
    dropped = set()

    for f in ordered:
        if f in dropped:
            continue
        selected.append(f)
        # Marcar para dropear los altamente correlacionados con f
        highly_corr = corr.loc[f, :][corr.loc[f, :] > corr_threshold].index.tolist()
        # quitarse a sí mismo
        highly_corr = [c for c in highly_corr if c != f]
        dropped.update(highly_corr)

    # 6) Construir resultados
    dropped_cols = [c for c in Xf.columns if c not in selected]
    X_sel = Xf[selected].copy()
    return X_sel, selected, dropped_cols, corr


class StochasticAnalyzer:
    def __init__(self, api_key=None, api_secret=None, atr_mult=4, atr_stop=3, m2_model=None, data=None, early_exit_threshold = 0.95, lookahead = 70):
        self.umbral_strong = None
        self.umbral_soft = None
        self.min_margin = 0.05
        self.data = data
        self.results = []
        self.model = None
        self.m2_model = m2_model
        self.scaler = StandardScaler()
        self.api_key = api_key
        self.api_secret = api_secret
        self.atr_mult = atr_mult
        self.atr_stop = atr_stop
        self.lookahead = lookahead
        self.opposite_exit_threshold = early_exit_threshold
        self.tradelogger = TradeLogger(log_path="trade_log.jsonl")
        self.session = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            recv_window=20000  # Aumentar ventana de tiempo a 20 segundos
        )
        self.session_demo = HTTP(
            api_key=api_key_demo,
            api_secret=api_secret_demo,
            demo=True,
            recv_window=20000  # Aumentar ventana de tiempo a 20 segundos
        )

    def get_data_filename(self, symbol, interval, start_date, end_date):
        """
        Genera el nombre del archivo CSV basado en los parámetros
        """
        return f"{symbol}_{interval}m_{start_date}_{end_date}.csv"

    def load_cached_data(self, filename):
        """
        Carga datos desde archivo CSV si existe
        """
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['datetime'] = pd.to_datetime(df['datetime'])
                print(f"Datos cargados desde cache: {filename} ({len(df)} registros)")
                return df
            except Exception as e:
                print(f"Error cargando cache: {e}")
                return None
        return None

    def save_data_to_cache(self, df, filename):
        """
        Guarda datos en archivo CSV
        """
        try:
            df.to_csv(filename, index=False)
            print(f"Datos guardados en cache: {filename}")
        except Exception as e:
            print(f"Error guardando cache: {e}")

    def get_missing_date_ranges(self, cached_df, start_date, end_date):
        """
        Determina qué rangos de fechas faltan en los datos cacheados
        """
        if cached_df is None or len(cached_df) == 0:
            return [(start_date, end_date)]

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Fechas en cache
        cached_start = cached_df['datetime'].min()
        cached_end = cached_df['datetime'].max()

        missing_ranges = []

        # Verificar si necesitamos datos antes del cache
        if start_dt < cached_start:
            missing_ranges.append((start_date, cached_start.strftime("%Y-%m-%d")))

        # Verificar si necesitamos datos después del cache
        if end_dt > cached_end:
            missing_ranges.append((cached_end.strftime("%Y-%m-%d"), end_date))

        return missing_ranges

    def generate_signature(self, params, api_secret):
        """
        Genera la firma para autenticación con API privada de Bybit
        """
        param_str = '&'.join([f"{key}={params[key]}" for key in sorted(params.keys())])
        return hmac.new(api_secret.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()

    def get_bybit_data_private(
            self,
            symbol: str = "AVAXUSDT",
            interval: str = "5",
            start_date: str | None = "2025-06-01",
            end_date: str | None = "2025-06-02",
            limit: int = 0
    ):
        """
        Descarga velas OHLC desde la API PRIVADA de Bybit.
        ▸ limit == 0  → modo RANGO (usa start_date y end_date).
        ▸ limit >  0  → modo LOOK-BACK (últimas «limit» velas).
        """

        if not self.api_key or not self.api_secret:
            print("API Key y Secret requeridos para API privada → usando API pública")
            return self.get_bybit_data_public(symbol, interval, start_date, end_date, limit)

        url = "https://api.bybit.com/v5/market/kline"
        max_limit = 1000
        all_data = []

        if limit > 0:
            params = {
                "api_key": self.api_key,
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": str(min(limit, max_limit)),
                "timestamp": str(int(time.time() * 1000))
            }
            params["sign"] = self.generate_signature(params, self.api_secret)

            try:
                r = requests.get(url, params=params, timeout=10)
                try:
                    data = r.json()
                except ValueError as e:
                    print(f"Error decoding JSON (private look-back). Status: {r.status_code}, Body: {r.text[:500]}")
                    raise e
                    
                if data["retCode"] == 0 and data["result"]["list"]:
                    for k in data["result"]["list"]:
                        all_data.append({
                            "timestamp": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        })
                else:
                    print(f"Error en respuesta: {data}")
            except Exception as e:
                print(f"Error (look-back): {e}")

        else:
            print(f"Obteniendo datos de {symbol} (API privada) desde {start_date} hasta {end_date}…")
            if not start_date or not end_date:
                raise ValueError("Debes especificar start_date y end_date en modo rango")

            # Calcular número de velas a descargar según el intervalo
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            minutes_per_candle = int(interval)
            num_candles = int((end_ts - start_ts) / (minutes_per_candle * 60 * 1000))

            # Si cabe en una sola petición
            if num_candles <= max_limit:
                params = {
                    "api_key": self.api_key,
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": str(start_ts),
                    "end": str(end_ts),
                    "limit": str(num_candles),
                    "timestamp": str(int(time.time() * 1000))
                }
                params["sign"] = self.generate_signature(params, self.api_secret)
                try:
                    r = requests.get(url, params=params, timeout=10)
                    try:
                        data = r.json()
                    except ValueError as e:
                        print(f"Error decoding JSON (private range). Status: {r.status_code}, Body: {r.text[:500]}")
                        raise e

                    if data["retCode"] == 0 and data["result"]["list"]:
                        for k in data["result"]["list"]:
                            all_data.append({
                                "timestamp": int(k[0]),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5])
                            })
                    else:
                        print(f"Error en respuesta: {data}")
                except Exception as e:
                    print(f"Error (rango simple): {e}")
            else:
                # Usar bucle como antes
                current = start_ts
                while current < end_ts:
                    params = {
                        "api_key": self.api_key,
                        "category": "linear",
                        "symbol": symbol,
                        "interval": interval,
                        "start": str(current),
                        "limit": str(max_limit),
                        "timestamp": str(int(time.time() * 1000))
                    }
                    params["sign"] = self.generate_signature(params, self.api_secret)

                    try:
                        r = requests.get(url, params=params, timeout=10)
                        try:
                            data = r.json()
                        except ValueError as e:
                            print(f"Error decoding JSON (private loop). Status: {r.status_code}, Body: {r.text[:500]}")
                            raise e

                        if data["retCode"] == 0 and data["result"]["list"]:
                            klines = data["result"]["list"]
                            for k in klines:
                                all_data.append({
                                    "timestamp": int(k[0]),
                                    "open": float(k[1]),
                                    "high": float(k[2]),
                                    "low": float(k[3]),
                                    "close": float(k[4]),
                                    "volume": float(k[5])
                                })
                            current = int(klines[-1][0]) + 1
                        else:
                            print(f"Error en respuesta: {data}")
                            break
                        time.sleep(0.1)  # evitar rate-limit
                    except Exception as e:
                        print(f"Error obteniendo datos: {e}")
                        break

        if all_data:
            df = pd.DataFrame(all_data)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            # print(f"Datos obtenidos: {len(df)} velas")
            return df

        print("Sin datos devueltos")
        return None

    def get_bybit_data_public(
            self,
            symbol: str = "AVAXUSDT",
            interval: str = "5",
            start_date: str | None = "2025-06-01",
            end_date: str | None = "2025-06-02",
            limit: int = 0
    ):
        """
        Obtiene datos OHLC de la API pública Bybit.
        ▸ Si limit == 0  ⇒  modo *rango de fechas* (comportamiento original).
        ▸ Si limit >  0  ⇒  modo *look-back*  (últimas «limit» velas).
        """

        url = "https://api.bybit.com/v5/market/kline"
        max_limit = 1000  # tope permitido por Bybit
        all_data = []

        # ────────────────────────────
        # 1)  MODO LOOK-BACK (limit > 0)
        # ────────────────────────────
        if limit > 0:
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, max_limit)
            }
            try:
                r = requests.get(url, params=params, timeout=10)
                r.raise_for_status()
                try:
                    data = r.json()
                except ValueError as e:
                    print(f"Error decoding JSON (look-back). Status: {r.status_code}, Body: {r.text[:500]}")
                    raise e

                if data["retCode"] == 0 and data["result"]["list"]:
                    klines = data["result"]["list"]
                    for k in klines:
                        all_data.append({
                            "timestamp": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5])
                        })
            except Exception as e:
                print(f"Error (look-back): {e}")

        # ────────────────────────────────
        # 2)  MODO RANGO DE FECHAS (limit == 0)
        # ────────────────────────────────
        else:
            print(f"Obteniendo datos de {symbol} (API pública) desde {start_date} hasta {end_date}…")
            if not start_date or not end_date:
                raise ValueError("Debes especificar start_date y end_date en modo rango")

            # Calcular número de velas según el intervalo
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            minutes_per_candle = int(interval)
            num_candles = int((end_ts - start_ts) / (minutes_per_candle * 60 * 1000))

            if num_candles <= max_limit:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": interval,
                    "start": str(start_ts),
                    "end": str(end_ts),
                    "limit": str(num_candles)
                }
                try:
                    r = requests.get(url, params=params, timeout=10)
                    try:
                        data = r.json()
                    except ValueError as e:
                        print(f"Error decoding JSON (range). Status: {r.status_code}, Body: {r.text[:500]}")
                        raise e

                    if data["retCode"] == 0 and data["result"]["list"]:
                        klines = data["result"]["list"]
                        for k in klines:
                            all_data.append({
                                "timestamp": int(k[0]),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5])
                            })
                except Exception as e:
                    print(f"Error (rango simple): {e}")
            else:
                # Si no cabe en una sola petición, usar el bucle clásico
                current = start_ts
                while current < end_ts:
                    params = {
                        "category": "linear",
                        "symbol": symbol,
                        "interval": interval,
                        "start": str(current),
                        "limit": max_limit
                    }
                    try:
                        r = requests.get(url, params=params, timeout=10)
                        try:
                            data = r.json()
                        except ValueError as e:
                            print(f"Error decoding JSON (public loop). Status: {r.status_code}, Body: {r.text[:500]}")
                            raise e

                        if data["retCode"] == 0 and data["result"]["list"]:
                            klines = data["result"]["list"]
                            for k in klines:
                                all_data.append({
                                    "timestamp": int(k[0]),
                                    "open": float(k[1]),
                                    "high": float(k[2]),
                                    "low": float(k[3]),
                                    "close": float(k[4]),
                                    "volume": float(k[5])
                                })
                            current = int(klines[-1][0]) + 1
                        else:
                            break
                        time.sleep(0.1)  # evitar rate-limit
                    except Exception as e:
                        print(f"Error obteniendo datos: {e}")
                        break

        # ────────────────────────────────
        # Construir DataFrame final
        # ────────────────────────────────
        if all_data:
            df = pd.DataFrame(all_data)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            print(f"Datos obtenidos: {len(df)} velas")
            return df

        print("Sin datos devueltos")
        return None

    def get_bybit_data_smart(self, symbol="AVAXUSDT", interval="5", start_date="2025-06-01", end_date="2025-06-02"):
        """
        Método inteligente que maneja cache y descarga solo datos faltantes
        """
        filename = self.get_data_filename(symbol, interval, start_date, end_date)

        # Intentar cargar datos cacheados
        cached_df = self.load_cached_data(filename)

        if cached_df is not None:
            # Verificar si las fechas solicitadas están completamente cubiertas
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            cached_start = cached_df['datetime'].min()
            cached_end = cached_df['datetime'].max()

            if start_dt >= cached_start and end_dt <= cached_end:
                print("Todos los datos ya están en cache, no es necesario descargar")
                # Filtrar solo el rango solicitado
                mask = (cached_df['datetime'] >= start_dt) & (cached_df['datetime'] <= end_dt)
                return cached_df[mask].reset_index(drop=True)

        # Determinar qué rangos necesitamos descargar
        missing_ranges = self.get_missing_date_ranges(cached_df, start_date, end_date)

        if not missing_ranges:
            print("Todos los datos ya están disponibles en cache")
            return cached_df

        print(f"Necesario descargar {len(missing_ranges)} rangos de fechas faltantes")

        # Descargar datos faltantes
        new_data_list = []
        for start_missing, end_missing in missing_ranges:
            print(f"Descargando rango: {start_missing} a {end_missing}")

            # Usar API privada si está disponible, sino API pública
            if self.api_key and self.api_secret:
                new_data = self.get_bybit_data_private(symbol, interval, start_missing, end_missing)
            else:
                new_data = self.get_bybit_data_public(symbol, interval, start_missing, end_missing)

            if new_data is not None and len(new_data) > 0:
                new_data_list.append(new_data)

        # Combinar todos los datos
        all_dataframes = []
        if cached_df is not None:
            all_dataframes.append(cached_df)
        all_dataframes.extend(new_data_list)

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            # Eliminar duplicados basados en timestamp
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(
                drop=True)

            # Guardar datos actualizados en cache
            self.save_data_to_cache(combined_df, filename)

            # Filtrar solo el rango solicitado
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            mask = (combined_df['datetime'] >= start_dt) & (combined_df['datetime'] <= end_dt)
            final_df = combined_df[mask].reset_index(drop=True)

            print(f"Datos finales: {len(final_df)} registros para el período {start_date} - {end_date}")
            return final_df

        print("No se pudieron obtener datos")
        return None

    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """
        Calcula el oscilador estocástico
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def add_indicators(self, df):
        """
        Añade todos los indicadores estocásticos, RSI, ATR y ADX al DataFrame,
        y clasifica la tendencia futura usando un umbral absoluto basado en ATR.
        """
        # Estocástico con período K=100
        k100_raw, _ = self.calculate_stochastic(df['high'], df['low'], df['close'], 100, 1)
        df['k100'] = k100_raw.rolling(window=1).mean()
        df['d100'] = df['k100'].rolling(window=3).mean()
        df['e100'] = df['d100'].rolling(window=57).mean()
        #
        # # Estocástico estándar
        df['k'], df['d'] = self.calculate_stochastic(df['high'], df['low'], df['close'], 14, 3)
        df['e'] = df['d'].rolling(window=40).mean()

        # ATR clásico (ya lo tienes, pero aseguramos consistencia)
        atr_period = 10
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=atr_period).mean()

        # Trend EMA
        for period in [200]:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'trend_ema{period}'] = np.where(df['close'] > df[f'EMA_{period}'], 1, 0)

        def sma(s: pd.Series, n: int) -> pd.Series:
            return s.rolling(n, min_periods=n).mean()

        def ema(data: pd.Series, period: int) -> pd.Series:
            return data.ewm(span=period, adjust=False).mean()

        def calculate(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
            fast_length = 12
            slow_length = 26
            signal_length = 9

            fast_ma = ema(prices.astype("float64"), fast_length)
            slow_ma = ema(prices.astype("float64"), slow_length)
            macd_line = fast_ma - slow_ma
            signal_line = ema(macd_line, signal_length)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram

        eps = 1e-9  # para evitar divisiones por cero
        df["macd"], df["macd_signal"], df["macd_hist"] = calculate(df["close"])
        vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['VWAP'] = vwap
        df['return_1_raw'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['volatility_10'] = df['return_1_raw'].rolling(10).std()
        df['volatility_50'] = df['return_1_raw'].rolling(50).std()
        df['tr'] = np.maximum(df['high'] - df['low'],
                              np.maximum(np.abs(df['high'] - df['close'].shift(1)),
                                         np.abs(df['low'] - df['close'].shift(1))))
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_norm'] = df['atr_14'] / (np.abs(df['close']) + eps)

        df['volatility'] = df['close'].pct_change().rolling(window=20).std()

        # # Volatilidad
        df['return_1_raw'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['volatility_10'] = df['return_1_raw'].rolling(10).std()


        # # RSI robusto
        def compute_rsi(series, window):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window).mean()
            ma_down = down.rolling(window).mean()
            rs = ma_up / (ma_down + eps)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df['rsi_14'] = compute_rsi(df['close'], 14)
        # df['rsi_50'] = compute_rsi(df['close'], 50)
        #
        # # Zonas de soporte y resistencia
        df = add_sr_features(
            df,
            high_col="high",
            low_col="low",
            close_col="close",
            datetime_col="datetime",
            atr_window=14,
            fractal_window=20,
            near_pct=0.005,  # 0.5%
            near_k_atr=0.5,  # 0.5 * ATR
            use_daily_pivots=True
        )

        # Tendencia simple (siguiente vela)
        df['price_change'] = df['close'].shift(-1) - df['close']
        df['trend'] = np.where(df['price_change'] > 0, 'bullish',
                               np.where(df['price_change'] < 0, 'bearish', 'lateral'))
        df.loc[df['price_change'].isna(), 'trend'] = np.nan
        lookahead = self.lookahead

        df = self.assign_future_trend_with_dd_order_sensitive(
            df,
            lookahead=lookahead,
            atr_mult=getattr(self, "atr_mult", 5),
            atr_dd_mult=getattr(self, "atr_stop", 3),
            order_sensitive=getattr(self, "order_sensitive_dd", True),
            drop_helpers=False  # o False si quieres auditar targets/stops
        )

        # df = self.assign_future_trend_with_dd_first_event(
        #     df,
        #     lookahead=lookahead,
        #     atr_mult=self.atr_mult,
        #     atr_dd_mult=getattr(self, "atr_stop", 2.6),
        #     first_event=True,
        #     drop_helpers=False  # o False si quieres auditar targets/stops
        # )

        # Donde no hay futuro, asigna NaN a la tendencia futura

        return df

    def assign_future_trend_with_dd_first_event(self, df,
                                                lookahead=60,
                                                atr_mult=4,
                                                atr_dd_mult=3,
                                                first_event=True,  # Variante A activada por defecto
                                                drop_helpers=True):
        """
        Rotula 'future_trend' usando TP/SL por ATR.

        Variante A (first_event=True):
          - Recorre la ventana futura y decide por el PRIMER evento entre:
              TP_up, TP_down, stop_bull, stop_bear
            Reglas:
              * Si el primer evento es TP_up  -> 'bullish'
              * Si el primer evento es TP_down-> 'bearish'
              * Si el primer evento es un stop (de cualquier lado) -> 'lateral'
          - Las últimas 'lookahead' filas quedan en NaN (no hay futuro completo).

        Modo rápido (first_event=False):
          - 'bullish' si (alcanza TP_up) y NUNCA viola stop_bull en la ventana.
          - 'bearish' si (alcanza TP_down) y NUNCA viola stop_bear en la ventana.
          - Si ambas condiciones fallan -> 'lateral'.

        Requiere: columnas ['high','low','close','ATR'] en df.
        Crea/usa: 'target_up','target_down','stop_bull','stop_bear','future_trend'.
        """

        if atr_mult is None:
            atr_mult = getattr(self, "atr_mult", 1.0)

        L = int(lookahead)
        n = len(df)
        if n <= L:
            raise ValueError(f"DataFrame demasiado corto para lookahead={L} (n={n}).")

        # --- Targets (TP) y Stops (DD) ---
        df['target_up'] = df['close'] + df['ATR'] * float(atr_mult)
        df['target_down'] = df['close'] - df['ATR'] * float(atr_mult)
        df['stop_bull'] = df['close'] - df['ATR'] * float(atr_dd_mult)  # long pierde si cae bajo esto
        df['stop_bear'] = df['close'] + df['ATR'] * float(atr_dd_mult)  # short pierde si sube sobre esto

        # --- Ventanas futuras alineadas (vectorizadas) ---
        future_max_high = df['high'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_low = df['low'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))
        future_max_close = df['close'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_close = df['close'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))

        # Eventos agregados en la ventana (para el modo rápido y para detectar ambigüedades)
        bullish_tp = (future_max_high > df['target_up']) | (future_max_close > df['target_up'])
        bearish_tp = (future_min_low < df['target_down']) | (future_min_close < df['target_down'])
        bullish_dd_breach = (future_min_low < df['stop_bull']) | (future_min_close < df['stop_bull'])
        bearish_dd_breach = (future_max_high > df['stop_bear']) | (future_max_close > df['stop_bear'])

        if not first_event:
            # ---------- Modo rápido (sin orden, conservador) ----------
            bullish_valid = bullish_tp & ~bullish_dd_breach
            bearish_valid = bearish_tp & ~bearish_dd_breach

            df['future_trend'] = np.select(
                [bullish_valid, bearish_valid],
                ['bullish', 'bearish'],
                default='lateral'
            )
        else:
            # ---------- Variante A: primer evento gana ----------
            # Base rápida: si exactamente uno es válido, asigna sin escanear
            bullish_valid = (bullish_tp & ~bullish_dd_breach)
            bearish_valid = (bearish_tp & ~bearish_dd_breach)

            # Asignación base (unívoca)
            future_trend = np.where(bullish_valid, 'bullish',
                                    np.where(bearish_valid, 'bearish', 'lateral')).astype(object)

            # Necesitan escaneo las filas donde NO hay un único válido,
            # pero sí hay algún evento en la ventana:
            any_event = (bullish_tp | bearish_tp | bullish_dd_breach | bearish_dd_breach)
            # (A ^ B) = True si exactamente uno es True. Negamos para coger 0 o 2 verdaderos.
            need_scan = (~(bullish_valid ^ bearish_valid)) & any_event
            idxs = np.where(need_scan.fillna(False).values)[0]

            # Arrays para escanear por tiempo real
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            tgt_up = df['target_up'].values
            tgt_dn = df['target_down'].values
            st_bu = df['stop_bull'].values
            st_be = df['stop_bear'].values

            for i in idxs:
                start, end = i + 1, min(i + 1 + L, n)
                decided = False
                for t in range(start, end):
                    # 1) TPs primero (si ocurren antes, ganan al stop)
                    if (highs[t] > tgt_up[i]) or (closes[t] > tgt_up[i]):
                        future_trend[i] = 'bullish';
                        decided = True;
                        break
                    if (lows[t] < tgt_dn[i]) or (closes[t] < tgt_dn[i]):
                        future_trend[i] = 'bearish';
                        decided = True;
                        break
                    # 2) Si aún no hubo TP, cualquier stop ⇒ lateral
                    if (lows[t] < st_bu[i]) or (closes[t] < st_bu[i]) \
                            or (highs[t] > st_be[i]) or (closes[t] > st_be[i]):
                        future_trend[i] = 'lateral';
                        decided = True;
                        break
                if not decided:
                    future_trend[i] = 'lateral'

            df['future_trend'] = future_trend

        # Últimas L filas -> NaN (no hay futuro completo)
        df.loc[df.index[-L:], 'future_trend'] = np.nan

        # Limpieza opcional
        if drop_helpers:
            df.drop(columns=['target_up', 'target_down', 'stop_bull', 'stop_bear'],
                    inplace=True, errors='ignore')

        # Guarda lookahead para otras partes (train split, etc.)
        self.lookahead = L
        return df

    def assign_future_trend_with_dd_order_sensitive(self, df,
                                                    lookahead=25,
                                                    atr_mult=None,
                                                    atr_dd_mult=1.0,
                                                    order_sensitive=True,
                                                    drop_helpers=True):
        """
        Rotula 'future_trend' usando objetivos (TP) por ATR y un máximo drawdown (stop) por ATR.

        Reglas:
          - Bullish válido si alcanza target_up y NO viola stop_bull dentro de las próximas 'lookahead' velas.
          - Bearish válido si alcanza target_down y NO viola stop_bear dentro de las próximas 'lookahead' velas.
          - Si order_sensitive=True y en una fila se tocan ambos (TP y stop), decide por el evento que ocurre primero.
          - Las últimas 'lookahead' filas quedan en NaN (no hay futuro completo).

        Requiere en df: ['high','low','close','ATR'].
        Crea/usa: 'target_up','target_down','stop_bull','stop_bear','future_trend'.
        """
        if atr_mult is None:
            atr_mult = getattr(self, "atr_mult", 1.0)

        L = int(lookahead)

        # Targets (TP) y Stops (DD permitido)
        df['target_up'] = df['close'] + df['ATR'] * float(atr_mult)
        df['target_down'] = df['close'] - df['ATR'] * float(atr_mult)
        df['stop_bull'] = df['close'] - df['ATR'] * float(atr_dd_mult)  # long pierde si cae bajo esto
        df['stop_bear'] = df['close'] + df['ATR'] * float(atr_dd_mult)  # short pierde si sube sobre esto

        # Ventanas futuras alineadas
        future_max_high = df['high'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_low = df['low'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))
        future_max_close = df['close'].shift(-1).rolling(L, min_periods=L).max().shift(-(L - 1))
        future_min_close = df['close'].shift(-1).rolling(L, min_periods=L).min().shift(-(L - 1))

        # TP hits (como tenías)
        bullish_tp = (future_max_high >= df['target_up']) | (future_max_close >= df['target_up'])
        bearish_tp = (future_min_low <= df['target_down']) | (future_min_close <= df['target_down'])

        # Breach de stop / DD
        bullish_dd_breach = (future_min_low <= df['stop_bull']) | (future_min_close <= df['stop_bull'])
        bearish_dd_breach = (future_max_high >= df['stop_bear']) | (future_max_close >= df['stop_bear'])

        if not order_sensitive:
            # Modo simple (rápido y conservador)
            bullish_valid = bullish_tp & ~bullish_dd_breach
            bearish_valid = bearish_tp & ~bearish_dd_breach

            df['future_trend'] = np.select(
                [bullish_valid, bearish_valid],
                ['bullish', 'bearish'],
                default='lateral'
            )
        else:
            # Modo sensible al orden: resolver conflictos (TP y DD verdaderos a la vez)
            bullish_valid = bullish_tp & ~bullish_dd_breach
            bearish_valid = bearish_tp & ~bearish_dd_breach

            # Base
            future_trend = np.where(bullish_valid, 'bullish',
                                    np.where(bearish_valid, 'bearish', 'lateral')).astype(object)

            conflict_bull = (bullish_tp & bullish_dd_breach).fillna(False).values
            conflict_bear = (bearish_tp & bearish_dd_breach).fillna(False).values

            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            tgt_up = df['target_up'].values
            st_bu = df['stop_bull'].values
            tgt_dn = df['target_down'].values
            st_be = df['stop_bear'].values
            n = len(df)

            # Solo recorremos índices conflictivos (pocos)
            idx_conf_bull = np.where(conflict_bull)[0]
            for i in idx_conf_bull:
                start, end = i + 1, min(i + 1 + L, n)
                first = None
                for t in range(start, end):
                    if (highs[t] >= tgt_up[i]) or (closes[t] >= tgt_up[i]):
                        first = 'tp';
                        break
                    if (lows[t] <= st_bu[i]) or (closes[t] <= st_bu[i]):
                        first = 'sl';
                        break
                future_trend[i] = 'bullish' if first == 'tp' else ('lateral' if first is None else 'lateral')

            idx_conf_bear = np.where(conflict_bear)[0]
            for i in idx_conf_bear:
                start, end = i + 1, min(i + 1 + L, n)
                first = None
                for t in range(start, end):
                    if (lows[t] <= tgt_dn[i]) or (closes[t] <= tgt_dn[i]):
                        first = 'tp';
                        break
                    if (highs[t] >= st_be[i]) or (closes[t] >= st_be[i]):
                        first = 'sl';
                        break
                future_trend[i] = 'bearish' if first == 'tp' else ('lateral' if first is None else 'lateral')

            df['future_trend'] = future_trend

        # Las últimas L filas no tienen futuro completo → NaN
        df.loc[df.index[-L:], 'future_trend'] = np.nan

        # Limpieza opcional
        if drop_helpers:
            df.drop(columns=[
                'target_up', 'target_down', 'stop_bull', 'stop_bear'
            ], inplace=True, errors='ignore')

        # Guarda lookahead en la instancia (útil para excluir en training)
        self.lookahead = L
        return df

    def add_volatility_zones(self, df):
        # Puedes ajustar percentiles o usar valores absolutos si prefieres
        low_vol = df['volatility'].quantile(0.33)
        high_vol = df['volatility'].quantile(0.66)
        df['vol_zone'] = np.select(
            [
                df['volatility'] < low_vol,
                (df['volatility'] >= low_vol) & (df['volatility'] < high_vol),
                df['volatility'] >= high_vol
            ],
            ['low', 'medium', 'high'],
            default='unknown'
        )
        return df

    def analyze_data_quality(self, df):
        """
        Analiza la calidad de los datos y distribución de tendencias
        """
        print("=== ANÁLISIS DE CALIDAD DE DATOS ===")
        print(f"Total de registros: {len(df)}")
        print(f"Registros con future_trend válido: {df['future_trend'].notna().sum()}")

        # Distribución de tendencias
        trend_distribution = df['future_trend'].value_counts()
        print("\n=== DISTRIBUCIÓN DE TENDENCIAS ===")
        for trend, count in trend_distribution.items():
            percentage = (count / len(df)) * 100
            print(f"{trend}: {count} ({percentage:.2f}%)")

        # Estadísticas de price_change_5
        # print("\n=== ESTADÍSTICAS DE CAMBIO DE PRECIO (5 períodos) ===")
        # print(f"Media: {df['price_change_lookahead'].mean():.6f}")
        # print(f"Mediana: {df['price_change_lookahead'].median():.6f}")
        # print(f"Desviación estándar: {df['price_change_lookahead'].std():.6f}")
        # print(f"Min: {df['price_change_lookahead'].min():.6f}")
        # print(f"Max: {df['price_change_lookahead'].max():.6f}")

    def create_combinations(self, df):
        """
        Crea combinaciones mejoradas con zonas específicas y relaciones posicionales
        """
        combinations = []

        # Definir zonas específicas para cada indicador
        zones = {
            'oversold_extreme': (0, 10),  # Sobreventa extrema
            'oversold': (10, 20),  # Sobreventa
            'low': (20, 30),  # Bajo
            'low_neutral': (30, 40),  # Bajo-neutral
            'neutral_low': (40, 50),  # Neutral-bajo
            'neutral_high': (50, 60),  # Neutral-alto
            'high': (60, 70),  # Alto
            'high_neutral': (70, 80),  # Alto-neutral
            'overbought': (80, 90),  # Sobrecompra
            'overbought_extreme': (90, 100)  # Sobrecompra extrema
        }

        # # Relaciones posicionales específicas
        # stoch14_relations = ['k>d>e', 'e>d>k', 'any']  # Reducir relaciones
        # stoch100_relations = ['k>d>e', 'e>d>k', 'any']  # Reducir relaciones
        #
        # # Direcciones de tendencia para E100
        # directions_e100 = ['ascending', 'descending']
        # directions_e14 = ['ascending', 'descending']

        # Generar combinaciones específicas con zonas y relaciones
        # for k14_zone_name, k14_zone in zones.items():
        #     for d14_zone_name, d14_zone in zones.items():
        #         for e14_zone_name, e14_zone in zones.items():
        #             for k100_zone_name, k100_zone in zones.items():
        #                 for d100_zone_name, d100_zone in zones.items():
        #                     for e100_zone_name, e100_zone in zones.items():
        #                         for stoch14_rel in stoch14_relations:
        #                             for stoch100_rel in stoch100_relations:
        #                                 for direction_e100 in directions_e100:
        #                                     for direction_e14 in directions_e14:
        #                                         combination = {
        #                                             # Zonas específicas para estocástico 14
        #                                             'k14_zone': k14_zone_name,
        #                                             'k14_min': k14_zone[0],
        #                                             'k14_max': k14_zone[1],
        #                                             'd14_zone': d14_zone_name,
        #                                             'd14_min': d14_zone[0],
        #                                             'd14_max': d14_zone[1],
        #                                             'e14_zone': e14_zone_name,
        #                                             'e14_min': e14_zone[0],
        #                                             'e14_max': e14_zone[1],
        #
        #                                             # Zonas específicas para estocástico 100
        #                                             'k100_zone': k100_zone_name,
        #                                             'k100_min': k100_zone[0],
        #                                             'k100_max': k100_zone[1],
        #                                             'd100_zone': d100_zone_name,
        #                                             'd100_min': d100_zone[0],
        #                                             'd100_max': d100_zone[1],
        #                                             'e100_zone': e100_zone_name,
        #                                             'e100_min': e100_zone[0],
        #                                             'e100_max': e100_zone[1],
        #
        #                                             # Relaciones posicionales
        #                                             'stoch14_relation': stoch14_rel,
        #                                             'stoch100_relation': stoch100_rel,
        #
        #                                             # Dirección de E100
        #                                             'e100_direction': direction_e100,
        #                                             'e14_direction': direction_e14,
        #
        #                                             'type': 'zone_positional'
        #                                         }
        #                                         combinations.append(combination)
        e14_zones = list(zones.items())
        e100_zones = list(zones.items())
        d100_zones = list(zones.items())

        stoch14_relations = ['k>d>e', 'k<d<e']
        stoch100_relations = ['k>d>e', 'k<d<e']
        directions = ['ascending', 'descending']
        combinations = []
        vol_zones = ['low', 'medium', 'high']
        for e14_name, e14_range in e14_zones:
            for e100_name, e100_range in e100_zones:
                for d100_name, d100_range in d100_zones:
                    for rel14 in stoch14_relations:
                        for rel100 in stoch100_relations:
                            for dir_e14 in directions:
                                for dir_e100 in directions:
                                    for vol_zone in vol_zones:
                                        combinations.append({
                                            'e14_zone': e14_name,
                                            'e14_min': e14_range[0],
                                            'e14_max': e14_range[1],
                                            'e100_zone': e100_name,
                                            'e100_min': e100_range[0],
                                            'e100_max': e100_range[1],
                                            'd100_zone': d100_name,
                                            'd100_min': d100_range[0],
                                            'd100_max': d100_range[1],
                                            'stoch14_relation': rel14,
                                            'stoch100_relation': rel100,
                                            'e14_direction': dir_e14,
                                            'e100_direction': dir_e100,
                                            'vol_zone': vol_zone,  # <-- Nuevo campo
                                            'type': 'simplified'
                                        })
        print(f"Generadas {len(combinations)} combinaciones simplificadas (con d100, e100 y volatilidad)")
        return combinations

    def analyze_combination(self, df, combination):
        if combination.get('type') == 'simplified':
            mask_e14 = (df['e'] >= combination['e14_min']) & (df['e'] < combination['e14_max'])
            mask_e100 = (df['e100'] >= combination['e100_min']) & (df['e100'] < combination['e100_max'])
            mask_d100 = (df['d100'] >= combination['d100_min']) & (df['d100'] < combination['d100_max'])
            mask_e14_dir = (df['e14_direction'] == combination['e14_direction'])
            mask_e100_dir = (df['e100_direction'] == combination['e100_direction'])

            # Relación stoch14
            if combination['stoch14_relation'] == 'k>d>e':
                mask_rel14 = (df['k'] > df['d']) & (df['d'] > df['e'])
            elif combination['stoch14_relation'] == 'k<d<e':
                mask_rel14 = (df['k'] < df['d']) & (df['d'] < df['e'])
            else:
                mask_rel14 = pd.Series([True] * len(df), index=df.index)

            # Relación stoch100
            if combination['stoch100_relation'] == 'k>d>e':
                mask_rel100 = (df['k100'] > df['d100']) & (df['d100'] > df['e100'])
            elif combination['stoch100_relation'] == 'k<d<e':
                mask_rel100 = (df['k100'] < df['d100']) & (df['d100'] < df['e100'])
            else:
                mask_rel100 = pd.Series([True] * len(df), index=df.index)

            mask_vol = (df['vol_zone'] == combination['vol_zone'])
            mask = mask_e14 & mask_e100 & mask_d100 & mask_e14_dir & mask_e100_dir & mask_vol & mask_rel14 & mask_rel100 & \
                   df['future_trend'].notna()

            filtered_data = df[mask]
            if len(filtered_data) < 10:
                return None

            total_samples = len(filtered_data)
            bullish_count = (filtered_data['future_trend'] == 'bullish').sum()
            bearish_count = (filtered_data['future_trend'] == 'bearish').sum()
            lateral_count = (filtered_data['future_trend'] == 'lateral').sum()

            bullish_pct = bullish_count / total_samples * 100
            bearish_pct = bearish_count / total_samples * 100
            lateral_pct = lateral_count / total_samples * 100

            if bullish_pct > 50:
                dominant_trend = 'bullish'
                confidence = bullish_pct
            elif bearish_pct > 50:
                dominant_trend = 'bearish'
                confidence = bearish_pct
            elif lateral_pct > 40:
                dominant_trend = 'lateral'
                confidence = lateral_pct
            else:
                dominant_trend = 'mixed'
                confidence = max(bullish_pct, bearish_pct, lateral_pct)

            result = combination.copy()
            result.update({
                'total_samples': total_samples,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'lateral_count': lateral_count,
                'bullish_pct': bullish_pct,
                'bearish_pct': bearish_pct,
                'lateral_pct': lateral_pct,
                'dominant_trend': dominant_trend,
                'confidence': confidence
            })
            return result
        else:
            return None

    def run_analysis(self, symbol="AVAXUSDT", start_date="2025-06-01", end_date="2025-06-02", interval="5"):
        """
        Ejecuta el análisis completo con cache inteligente
        """
        # Obtener datos usando cache inteligente
        self.data = self.get_bybit_data_smart(symbol, interval, start_date, end_date)
        # después de crear self.data
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])  # sin forzar utc=False

        # forzar awareness y convertir a Bogota (UTC-5)
        if getattr(self.data["datetime"].dt, "tz", None) is None:
            # si tus tiempos representan UTC sin tz, primero localiza como UTC
            self.data["datetime"] = self.data["datetime"].dt.tz_localize("UTC")

        self.data["datetime"] = self.data["datetime"].dt.tz_convert("America/Bogota")

        if self.data is None or len(self.data) == 0:
            print("No se pudieron obtener datos")
            return

        # Añadir indicadores
        self.data = self.add_indicators(self.data)

        self.data = self.add_volatility_zones(self.data)

        # Analizar calidad de datos
        self.analyze_data_quality(self.data)

        # # Crear combinaciones
        # combinations = self.create_combinations(self.data)
        # print(f"\nAnalizando {len(combinations)} combinaciones...")
        #
        # # Analizar cada combinación
        # self.results = []
        # for i, combination in enumerate(combinations):
        #     if i % 1000 == 0:
        #         print(f"Progreso: {i}/{len(combinations)}")
        #
        #     result = self.analyze_combination(self.data, combination)
        #     if result:
        #         self.results.append(result)
        #
        # print(f"Análisis completado. {len(self.results)} combinaciones válidas encontradas.")

    def save_results(self, filename="stochastic_analysis_results.csv"):
        """
        Guarda los resultados en CSV incluyendo información de zonas
        """
        if not self.results:
            print("No hay resultados para guardar")
            return

        df_results = pd.DataFrame(self.results)

        # Descripción legible para todos los casos simplificados
        if 'type' in df_results.columns:
            df_results['pattern_description'] = df_results.apply(
                lambda row: self.create_pattern_description(row) if row.get('type') in ('zone_positional', 'simplified')
                else 'legacy_pattern', axis=1
            )

        df_results = df_results.sort_values('confidence', ascending=False)
        df_results.to_csv(filename, index=False)
        print(f"Resultados guardados en {filename}")

        return df_results

    def create_pattern_description(self, row):
        return (f"E14:{row['e14_zone']}({row['e14_direction']}) | "
                f"E100:{row['e100_zone']}({row['e100_direction']}) | "
                f"D100:{row['d100_zone']} | "
                f"Stoch14:{row.get('stoch14_relation', '?')} | "
                f"Stoch100:{row.get('stoch100_relation', '?')} | "
                f"Vol:{row.get('vol_zone', '?')}")

    def fit_threshold_curve(
            self,
            X_hold,
            y_hold,
            u_vals=None,
            already_scaled=True,
            features=None,
            # === Parámetros Opción A (data-driven) ===
            perc_low=0.40,  # percentil inferior para top1 (en casos de tendencia)
            perc_high=0.90,  # percentil superior para top1 (en casos de tendencia)
            floor=0.50,  # no bajar de 0.50
            ceil=0.70,  # cap base; si None, sin cap base
            step=0.01,  # resolución del grid
            # === Mínimo dinámico de muestras de tendencia ===
            min_trend_samples=None,  # si None, se calculará dinámicamente
            min_abs=300,  # piso absoluto
            min_frac_total=0.03,  # 3% del holdout total
            min_frac_trend=0.10,  # 10% de las muestras de tendencia
            cap_trend_samples=2000,  # tope para no exigir demasiado
            # === Cap dinámico de la cola ===
            tail_min_count=200,  # mínimo absoluto de muestras sobre ceil para liberar el cap
            tail_min_frac=0.02,  # o mínimo % relativo sobre n_trend
            ceil_max=0.90,  # techo máximo cuando se libera el cap
            # === NUEVO: guardas para evitar umbrales demasiado bajos ===
            hard_floor_u=None,  # ej: 0.60 -> descarta u<0.60
            precision_guard=None  # ej: 0.80 -> corta curva por debajo del 1er u con prec>=0.80
    ):
        """
        Calcula trade_rate y precision para umbrales 'u' (top1 >= u y clase top ∈ {bullish,bearish}).
        Si u_vals es None, construye el rango automáticamente con percentiles de top1 en muestras de tendencia.
        Aplica cap dinámico: si hay suficiente masa de probas > ceil, permite subir el tope hasta ceil_max.
        Devuelve un DataFrame con columnas: ['u','trade_rate','precision','n'] y metadatos en df.attrs.
        """

        # === Validación & coerción de X_hold ===
        if X_hold is None:
            raise TypeError("X_hold es None. Pásame el holdout de features.")

        if not already_scaled:
            if features is None:
                features = getattr(self, "features_", None)
                if features is None:
                    raise ValueError("Debes pasar 'features' si already_scaled=False.")
            if hasattr(X_hold, "values"):  # DataFrame/Series
                X_arr = np.asarray(X_hold[features].values if hasattr(X_hold, "__getitem__") else X_hold.values)
            else:
                X_arr = np.asarray(X_hold)
            X_arr = self.scaler.transform(X_arr)
        else:
            X_arr = np.asarray(X_hold.values) if hasattr(X_hold, "values") else np.asarray(X_hold)

        if X_arr.ndim != 2:
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            else:
                raise TypeError(
                    f"Se esperaba array 2D, recibido ndim={X_arr.ndim}, shape={getattr(X_arr, 'shape', None)}")

        # === Modelo listo ===
        mdl = getattr(self, "model", None)
        if mdl is None or not hasattr(mdl, "predict_proba"):
            raise RuntimeError("self.model no está entrenado o no soporta predict_proba().")

        # === Probabilidades y clases top ===
        proba = mdl.predict_proba(X_arr)
        classes = list(mdl.classes_)
        top = proba.max(axis=1)
        top_idx = proba.argmax(axis=1)
        cls_top = np.array([classes[i] for i in top_idx])
        is_trend = np.isin(cls_top, ["bullish", "bearish"])
        top_trend = top[is_trend]

        # === Mínimo dinámico de muestras de tendencia ===
        n_total = int(X_arr.shape[0])
        n_trend = int(top_trend.size)
        if min_trend_samples is None:
            min_trend_samples = max(
                int(min_abs),
                int(min_frac_total * n_total),
                int(min_frac_trend * n_trend)
            )
            min_trend_samples = min(int(min_trend_samples), int(cap_trend_samples))

        # Variables de auditoría
        q_lo = q_hi = None
        tail_count = tail_frac = None
        dynamic_cap_applied = False

        # === Construcción data-driven del rango de umbrales ===
        if u_vals is None:
            if n_trend >= min_trend_samples:
                q_lo, q_hi = np.quantile(top_trend, [perc_low, perc_high])
                u_min = max(float(floor), float(q_lo))

                if ceil is None:
                    # Sin cap base: usa q_hi directamente
                    u_max = float(q_hi)
                else:
                    # Cap base
                    u_max = min(float(ceil), float(q_hi))
                    # ¿Hay suficiente masa en la cola por encima del cap base?
                    if float(q_hi) > float(ceil):
                        tail_count = int((top_trend >= float(ceil)).sum())
                        tail_frac = tail_count / max(1, n_trend)
                        if (tail_count >= int(tail_min_count)) or (tail_frac >= float(tail_min_frac)):
                            # Libera el cap hasta ceil_max o q_hi
                            u_max = min(float(ceil_max), float(q_hi))
                            dynamic_cap_applied = True

                # Si el rango queda degenerado o no finito: fallback
                if (not np.isfinite(u_min)) or (not np.isfinite(u_max)) or (u_max <= u_min):
                    u_min, u_max = float(floor), min(float(ceil) if ceil is not None else float(floor) + 0.20,
                                                     float(floor) + 0.12)
            else:
                # Fallback seguro si no alcanza el mínimo
                u_min, u_max = float(floor), min(float(ceil) if ceil is not None else float(floor) + 0.20,
                                                 float(floor) + 0.12)

            # Grid de umbrales
            u_vals = np.round(np.arange(u_min, u_max + 1e-9, float(step)), 3)
            if u_vals.size < 3:
                u_vals = np.unique(
                    np.round(np.linspace(u_min, max(u_min + 2 * step, u_max), num=3), 3)
                )

        # === Métrica por umbral ===
        rows = []
        y_hold_arr = np.asarray(y_hold)
        for u in u_vals:
            m = (top >= u) & is_trend
            n = int(m.sum())
            if n == 0:
                rows.append((float(u), 0.0, float("nan"), 0))
                continue
            prec = float((y_hold_arr[m] == cls_top[m]).mean())
            trade_rate = n / len(y_hold_arr)
            rows.append((float(u), float(trade_rate), float(prec), n))

        df = pd.DataFrame(rows, columns=["u", "trade_rate", "precision", "n"])

        # === NUEVO: aplicar guardas ===
        # 1) Filtro por suelo duro de u
        if hard_floor_u is not None:
            hard_floor_u = float(hard_floor_u)
            df = df[df["u"] >= hard_floor_u].reset_index(drop=True)

        # 2) Corte por precisión mínima "primera vez que alcanza"
        #    Encuentra el menor u con precision >= precision_guard y descarta u por debajo.
        if precision_guard is not None:
            precision_guard = float(precision_guard)
            df_ok = df[df["precision"] >= precision_guard]
            if not df_ok.empty:
                u_guard = float(df_ok["u"].min())
                df = df[df["u"] >= u_guard].reset_index(drop=True)
                df.attrs["precision_guard_used"] = precision_guard
                df.attrs["u_guard_cut"] = u_guard

        # --- Metadatos (lo que ya tenías) ---
        df.attrs["u_vals"] = u_vals
        df.attrs["n_total"] = n_total
        df.attrs["n_trend"] = n_trend
        df.attrs["min_trend_samples_used"] = int(min_trend_samples)
        df.attrs["range_built_from_percentiles"] = bool(n_trend >= min_trend_samples)
        df.attrs["q_lo"] = None if q_lo is None else float(q_lo)
        df.attrs["q_hi"] = None if q_hi is None else float(q_hi)
        df.attrs["tail_count_over_ceil"] = tail_count
        df.attrs["tail_frac_over_ceil"] = tail_frac
        df.attrs["dynamic_cap_applied"] = bool(dynamic_cap_applied)
        if u_vals is not None and len(u_vals) > 0:
            # actualiza u_min/u_max tras guardas
            if not df.empty:
                df.attrs["u_min"] = float(df["u"].min())
                df.attrs["u_max"] = float(df["u"].max())
            else:
                df.attrs["u_min"] = float(u_vals.min())
                df.attrs["u_max"] = float(u_vals.max())

        return df

    def _build_base_model(self, model_type: str, random_state: int = 42, model_kwargs: Optional[Dict] = None):
        """
        Devuelve un clasificador scikit-compatible según model_type.
        - rf: RandomForestClassifier
        - lgbm: LightGBM (LGBMClassifier)
        - xgb: XGBoost (XGBClassifier)
        """
        model_kwargs = model_kwargs or {}
        model_type = (model_type or "rf").lower()
        if model_type in ("rf", "randomforest", "random_forest"):
            defaults = dict(n_estimators=700, max_features="sqrt", n_jobs=-1, random_state=random_state)
            defaults.update(model_kwargs)
            return RandomForestClassifier(**defaults), "rf"

        if model_type in ("lgbm", "lightgbm"):
            if not _HAS_LGBM:
                raise ImportError("LightGBM no está instalado. Ejecuta: pip install lightgbm")
            # valores razonables; puedes ajustarlos vía model_kwargs
            defaults = dict(
                n_estimators=700,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1
            )
            defaults.update(model_kwargs)
            return LGBMClassifier(**defaults), "lgbm"

        raise ValueError(f"model_type desconocido: {model_type}. Usa: 'rf', 'lgbm' o 'xgb'.")

    def _safe_feature_importance(self, model, X_tr, y_tr, feature_names, plot=True, export_csv=None):
        """
        Obtiene importancias de características (nativas o permutation importance).
        Devuelve DataFrame con ['feature', 'importance'].

        Parámetros:
        - plot: bool -> si True, muestra un gráfico de barras horizontales.
        - export_csv: str -> si no es None, guarda el DataFrame en un CSV con este nombre.
        """
        imp = None
        # Intentos comunes
        for attr in ("feature_importances_",):
            if hasattr(model, attr):
                vals = getattr(model, attr)
                if isinstance(vals, (list, np.ndarray)) and len(vals) == len(feature_names):
                    imp = np.array(vals, dtype=float)
                    break

        # Si no se logró, usar permutation importance
        if imp is None:
            try:
                pi = permutation_importance(model, X_tr, y_tr, n_repeats=5, random_state=42, n_jobs=-1)
                imp = pi.importances_mean
            except Exception:
                imp = np.zeros(len(feature_names), dtype=float)

        df_imp = (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Exportar a CSV si se solicita
        if export_csv:
            df_imp.to_csv(export_csv, index=False)

        # Mostrar gráfico
        if plot:
            plt.figure(figsize=(8, max(6, len(df_imp) * 0.25)))  # alto dinámico
            plt.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
            plt.title("Importancia de características (Random Forest)")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()
            # plt.show(block=False)

        return df_imp

    def _oof_probas_time_series(self, X_array, y_array, base_estimator, n_splits=5, clip_eps=1e-6):
        """
        Genera probabilidades OOF respetando el tiempo, usando TimeSeriesSplit.
        Alinea columnas de probas a las clases globales (rellena con 0 si faltan en algún fold).
        Devuelve: proba_oof (n_samples, K), y_oof (n_samples,), classes_global (list)
        """
        # Asegúrate de que X_array e y_array mantienen el ORDEN temporal
        classes_global = list(pd.Index(y_array).unique().sort_values())
        n_classes = len(classes_global)
        class_to_idx = {c: i for i, c in enumerate(classes_global)}

        proba_oof = np.zeros((len(y_array), n_classes), dtype=float)
        filled = np.zeros(len(y_array), dtype=bool)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_array)):
            est = clone(base_estimator)
            est.fit(X_array[tr_idx], y_array[tr_idx])

            proba_fold = est.predict_proba(X_array[val_idx])  # (n_val, k_fold)
            # Alinear columnas a clases_global
            proba_aligned = np.zeros((len(val_idx), n_classes), dtype=float)
            model_classes = list(est.classes_)
            model_c2i = {c: i for i, c in enumerate(model_classes)}
            for j, c in enumerate(classes_global):
                if c in model_c2i:
                    proba_aligned[:, j] = proba_fold[:, model_c2i[c]]
                else:
                    # Si una clase no aparece en este fold, queda 0 (sin info)
                    proba_aligned[:, j] = 0.0

            # Clipping de estabilidad
            proba_aligned = np.clip(proba_aligned, clip_eps, 1.0 - clip_eps)
            # Re-normaliza por si hubo columnas 0
            proba_sum = proba_aligned.sum(axis=1, keepdims=True)
            proba_sum[proba_sum == 0] = 1.0
            proba_aligned = proba_aligned / proba_sum

            proba_oof[val_idx] = proba_aligned
            filled[val_idx] = True

        # Por si algo quedó sin llenar (no debería)
        if not filled.all():
            missing = np.where(~filled)[0]
            proba_oof[missing] = 1.0 / n_classes

        return proba_oof, y_array, classes_global

    def create_predictive_model(
            self,
            display_graphic: bool = False,
            N_KEEP: int = 40000,
            model_type: str = "rf",
            model_kwargs: Optional[Dict] = None,
            calibrate_method: str = "isotonic",  # "isotonic" o "sigmoid"
            return_metrics: bool = False,
            display_feature_plot: bool = False,
            use_df_parquet: bool = True
    ):
        if self.data is None:
            print("No hay datos para entrenar el modelo")
            return

        # Preparación de dataset y features (tu función existente)
        if use_df_parquet:
            df_model, features, _ = prepare_features_incremental(
                raw_df=self.data,
                preparar_features_fn=self.preparar_features,  # tu propia función
                params=params_to_features,
                cache_dir="feature_cache",
                datetime_col="datetime",
                warmup_bars=150
            )
        else:
            df_model, features = self.preparar_features(self.data)

        df_model["datetime"] = pd.to_datetime(df_model["datetime"])

        # --- Guardas anti-fuga de fechas tras prepare_features_incremental ---
        raw_start = self.data["datetime"].min()
        raw_end = self.data["datetime"].max()
        # raw_start = pd.to_datetime("1690800600", unit='s', utc=True).tz_convert('America/Bogota')

        # Alinear tz del df_model a la de self.data (si hiciera falta)
        tz_raw = getattr(self.data["datetime"].dt, "tz", None)
        tz_mod = getattr(df_model["datetime"].dt, "tz", None)
        if (tz_raw is not None) and (tz_mod is None):
            df_model["datetime"] = df_model["datetime"].dt.tz_localize(tz_raw)
        elif (tz_raw is not None) and (tz_mod is not None) and (str(tz_raw) != str(tz_mod)):
            df_model["datetime"] = df_model["datetime"].dt.tz_convert(tz_raw)

        # Recortar estrictamente al rango del train_slice
        mask_bounds = (df_model["datetime"] >= raw_start) & (df_model["datetime"] <= raw_end)
        n_outside = int((~mask_bounds).sum())
        if n_outside:
            print(f"[LEAK GUARD] recortando {n_outside} filas fuera de {raw_start}..{raw_end}")

        df_model = df_model.loc[mask_bounds].copy()
        df_model.sort_values("datetime", inplace=True)
        df_model.reset_index(drop=True, inplace=True)

        # (opcional) sanity checks
        if not df_model["datetime"].is_monotonic_increasing:
            print("[WARN] datetime no es monótono tras recorte; ordenando por seguridad.")
            df_model = df_model.sort_values("datetime").reset_index(drop=True)

        # Log del rango resultante (para auditar)
        try:
            a, b = df_model["datetime"].iloc[0], df_model["datetime"].iloc[-1]
            print(f"Rango df_model tras recorte: {a} -> {b} (n={len(df_model)})")
        except Exception:
            pass

        L = getattr(self, "lookahead", 25)  # usa self.lookahead o 25 por defecto

        n_total = len(df_model)
        if n_total <= L:
            raise RuntimeError(f"No hay suficientes filas para excluir lookahead={L} (n={n_total}).")

        # Recorte a las últimas N_KEEP
        if n_total > N_KEEP:
            df_model = df_model.iloc[-N_KEEP:].copy()
            n_total = len(df_model)

        if n_total - L < 1000:
            print(f"Advertencia: solo {n_total - L} filas de train tras recorte y lookahead. "
                  f"Considera reducir L o N_KEEP.")

        # Excluir últimas L para evitar fuga de datos
        df_train = df_model.iloc[:-L].copy()
        df_train = df_train[df_train["future_trend"].notna()]
        df_train = df_train.dropna(subset=features + ["future_trend"])

        try:
            first_val = df_train["datetime"].iloc[0]
            last_val = df_train["datetime"].iloc[-1]

            # Si son timestamps de pandas, usar .strftime
            if isinstance(first_val, (pd.Timestamp, datetime)):
                first_date = first_val.strftime("%Y-%m-%d %H:%M:%S")
                last_date = last_val.strftime("%Y-%m-%d %H:%M:%S")
            else:
                # Caso de enteros o floats tipo UNIX timestamp
                first_date = datetime.fromtimestamp(first_val).strftime("%Y-%m-%d %H:%M:%S")
                last_date = datetime.fromtimestamp(last_val).strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"Rango de fechas en self.data inicial de recorte NKEEP y Lookahead: {first_date} -> {last_date} (n={n_total}).")
        except Exception as e:
            print(f"No se pudo obtener rango de fechas: {e}")

        X = df_train[features]
        y = df_train["future_trend"]

        # ===== Métricas auxiliares =====
        def multiclass_brier(y_true_labels, y_prob, classes_text):
            class_to_idx = {c: i for i, c in enumerate(classes_text)}
            y_idx = np.array([class_to_idx[v] for v in y_true_labels])
            Y = np.zeros_like(y_prob)
            Y[np.arange(len(y_true_labels)), y_idx] = 1.0
            return float(np.mean(np.sum((y_prob - Y) ** 2, axis=1)))

        def ece_maxprob(y_true_labels, y_prob, classes_text, n_bins=10):
            class_to_idx = {c: i for i, c in enumerate(classes_text)}
            y_idx = np.array([class_to_idx[v] for v in y_true_labels])
            conf = np.max(y_prob, axis=1)
            pred = np.argmax(y_prob, axis=1)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            binids = np.digitize(conf, bins) - 1
            ece, n = 0.0, len(y_true_labels)
            for i in range(n_bins):
                mask = binids == i
                if np.any(mask):
                    acc = np.mean(pred[mask] == y_idx[mask])
                    avg_conf = np.mean(conf[mask])
                    ece += (np.sum(mask) / n) * abs(acc - avg_conf)
            return float(ece)

        # ===============================
        # Entrenamiento modelo base
        # ===============================
        base_model, model_tag = self._build_base_model(model_type=model_type, model_kwargs=model_kwargs or {})

        # Split + escalado (mantener por consistencia)
        # NOTA: Los árboles no requieren escalado, pero dejamos tu pipeline igual para no romper nada.
        # Para XGB con LabelEncoder, haremos una ruta especial.
        use_label_encoding = (model_tag == "xgb")

        # ===== Ruta normal (RF / LGBM) sin encoding =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        base_model.fit(X_tr, y_tr)

        # ---- Sin calibrar ----
        preds_vanilla = base_model.predict(X_test_scaled)
        proba_vanilla = base_model.predict_proba(X_test_scaled)
        classes = base_model.classes_

        print("Reporte de clasificación de modelo SIN calibrar:")
        print(classification_report(y_test, preds_vanilla, labels=classes))
        print(f"Brier (multiclase): {multiclass_brier(y_test, proba_vanilla, classes):.6f}")
        print(f"Log loss: {log_loss(y_test, proba_vanilla, labels=classes):.6f}")
        print(f"ECE: {ece_maxprob(y_test, proba_vanilla, classes):.6f}")
        print("=" * 50)

        # ---- Calibración ----
        if calibrate_method not in ("isotonic", "sigmoid", "beta"):
            calibrate_method = "isotonic"

        if calibrate_method == "beta":
            # === NEW: OOF + Beta + clipping (time-series aware) ===
            # 1) Generar OOF sobre el TRAIN (X_train_scaled, y_train)
            proba_oof, y_oof, classes_global = self._oof_probas_time_series(
                X_array=X_train_scaled,
                y_array=y_train.to_numpy(),  # respeta orden temporal del df
                base_estimator=base_model,
                n_splits=5,
                clip_eps=1e-6
            )
            class_to_idx = {c: i for i, c in enumerate(classes_global)}
            y_oof_idx = np.array([class_to_idx[v] for v in y_oof], dtype=int)

            # 2) Ajustar calibrador Beta sobre OOF (prueba 'ab'/'bm'/'abm')
            mc_beta = MultiClassBetaCalibrator(parameters="abm", classes_=classes_global)
            mc_beta.fit(proba_oof, y_oof_idx)

            # 3) Re-entrenar el modelo base en TODO el TRAIN y envolver con Beta
            base_model_final = clone(base_model).fit(X_train_scaled, y_train)

            preds_vanilla = base_model_final.predict(X_test_scaled)
            proba_vanilla = base_model_final.predict_proba(X_test_scaled)
            classes = base_model_final.classes_

            print("Reporte de clasificación de modelo SIN calibrar + entrenamiento completo:")
            print(classification_report(y_test, preds_vanilla, labels=classes))
            print(f"Brier (multiclase): {multiclass_brier(y_test, proba_vanilla, classes):.6f}")
            print(f"Log loss: {log_loss(y_test, proba_vanilla, labels=classes):.6f}")
            print(f"ECE: {ece_maxprob(y_test, proba_vanilla, classes):.6f}")
            print("=" * 50)

            self.model = BetaMultiWrapper(base_model_final, mc_beta, classes_=classes_global, label_encoder=None)

            # 4) (Opcional) clipping en inferencia dentro del wrapper (si tu wrapper no lo hace)
            #    Asegúrate que BetaMultiWrapper hace np.clip en predict_proba antes/después del calibrador.
        else:
            self.model = CalibratedClassifierCV(base_model, method=calibrate_method, cv='prefit')
            self.model.fit(X_cal, y_cal)

        # Métricas del calibrado final
        proba_cal = self.model.predict_proba(X_test_scaled)
        preds_cal = self.model.predict(X_test_scaled)
        classes_cal = getattr(self.model, "classes_", base_model.classes_)

        # --------- MÉTRICAS CALIBRADAS (DEVUELTAS) ----------
        report_calib = classification_report(y_test, preds_cal, labels=classes_cal, output_dict=True)
        brier_calib = multiclass_brier(y_test, proba_cal, classes_cal)
        logloss_calib = log_loss(y_test, proba_cal, labels=classes_cal)
        ece_calib = ece_maxprob(y_test, proba_cal, classes_cal)

        print("Reporte de clasificación de modelo CALIBRADO:")
        print(classification_report(y_test, preds_cal, labels=classes_cal))
        print(f"Brier (multiclase): {brier_calib:.6f}")
        print(f"Log loss: {logloss_calib:.6f}")
        print(f"ECE: {ece_calib:.6f}")

        # Selector de umbrales (tu función)
        # u_strong, info_strong, u_soft, info_soft, df_bands = self.select_thresholds_by_band_precision(
        #     X_hold=X_test_scaled, y_hold=y_test,
        #     band_step=0.1, u_min=0.55, u_max=0.95,
        #     min_margin=0.03,
        #     min_n_soft=1000,
        #     min_n_strong=1200,
        #     min_trade_rate_soft=0.10,
        #     min_trade_rate_strong=0.03,
        #     soft_range=(0.83, 0.865),
        #     strong_range=(0.89, 0.96),
        #     soft_start=0.60,
        #     strong_start=0.80,
        #     use_ci_lower=True,
        #     prefer_center_soft=True
        # )

        # Importancias
        # feature_importance = self._safe_feature_importance(base_model, X_tr, y_tr, features)
        feature_importance = self._safe_feature_importance(
            base_model, X_tr, y_tr, features,
            plot=display_feature_plot,
            export_csv="importancias_rf_lag.csv"  # opcional
        )
        print("\nImportancia de características:")
        print(feature_importance)

        # X_sel, selected_cols, dropped_cols, corr_matrix = select_features_by_correlation(
        #     X=df_model,
        #     feature_names=features,
        #     importances=feature_importance,
        #     corr_threshold=0.90
        # )
        # #
        # print("Features seleccionados:", selected_cols)
        # print("Features eliminados:", dropped_cols)

        print(f"\nModelo base: {model_tag.upper()} | calibración: {calibrate_method}")
        # print("u_strong:", u_strong, "| u_soft:", u_soft)
        # print("STRONG:", info_strong)
        # print("SOFT:", info_soft)

        # Gráficas de calibración
        if display_graphic:
            proba_base = proba_vanilla
            proba_calib = proba_cal
            clases = classes_cal  # o classes; ambas deberían coincidir en orden

            plt.figure(figsize=(7, 5))
            for target_class in clases:
                idx = list(clases).index(target_class)
                y_true_bin = np.array([1 if yt == target_class else 0 for yt in y_test])

                ppb_x, ppb_y = calibration_curve(y_true_bin, proba_base[:, idx], n_bins=10, strategy='uniform')
                plt.plot(ppb_x, ppb_y, marker='o', label=f"Sin calibrar '{target_class}'")

                ppc_x, ppc_y = calibration_curve(y_true_bin, proba_calib[:, idx], n_bins=10, strategy='uniform')
                plt.plot(ppc_x, ppc_y, marker='o', linestyle='--', label=f"Calibrado '{target_class}'")

            fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
            plt.xlabel('Probabilidad predicha por el modelo')
            plt.ylabel('Proporción real de acierto')
            plt.title(f'Gráfico de calibración: sin calibrar vs calibrado\nEntrenado: {fecha_actual}')
            plt.legend();
            plt.grid();
            # plt.show(block=False)
            plt.show()

        # Guardar en self
        self.base_model = base_model
        self.feature_importance_ = feature_importance
        self.classes_ = classes_cal
        self.model_tag_ = model_tag

        if return_metrics:
            metrics_dict = {
                "classes": list(classes_cal),
                "report_calib": report_calib,  # dict completo por clase + macro/weighted
                "brier_calib": float(brier_calib),
                "logloss_calib": float(logloss_calib),
                "ece_calib": float(ece_calib),
                "N_KEEP": N_KEEP,
                "model_tag": model_tag,
                "calibrate_method": calibrate_method,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            return self.model, metrics_dict

        return self.model

    def _flatten_report(self, report_dict: dict, prefix: str = "") -> dict:
        """
        Aplana el classification_report (output_dict=True) a un solo nivel.
        Mantiene claves útiles como:
          - precision/recall/f1-score de cada clase ('bearish','bullish','lateral' si aplica)
          - 'macro avg' y 'weighted avg'
        """
        flat = {}
        for k, v in report_dict.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                for mk, mv in v.items():
                    flat[f"{key}_{mk}"] = mv
            else:
                flat[key] = v
        return flat

    def sweep_nkeep(self,
                    start: int = 10_000,
                    stop: int = 160_000,
                    step: int = 10_000,
                    model_type: str = "rf",
                    model_kwargs: Optional[Dict] = None,
                    calibrate_method: str = "isotonic",
                    export_csv_path: Optional[str] = None,
                    sort_by: str = "macro_f1",  # opciones: 'macro_f1', 'logloss', 'brier', 'ece'
                    ascending: Optional[bool] = None):
        """
        Ejecuta entrenamiento y evaluación calibrada para N_KEEP en [start, stop] con 'step'.
        Devuelve un DataFrame con:
          - N_KEEP, brier_calib, logloss_calib, ece_calib
          - precision/recall/f1 por clase (si existen)
          - macro avg y weighted avg
        """

        rows = []
        errs = []

        # Orden de sort por defecto coherente con la métrica:
        if ascending is None:
            # f1 mayor es mejor; logloss/brier/ece menor es mejor
            ascending = (sort_by in ["logloss", "brier", "ece"])

        for nkeep in range(start, stop + 1, step):
            try:
                _model, metrics = self.create_predictive_model(
                    display_graphic=False,
                    N_KEEP=nkeep,
                    model_type=model_type,
                    model_kwargs=(model_kwargs or {}),
                    calibrate_method=calibrate_method,
                    return_metrics=True
                )
                if metrics is None:
                    errs.append((nkeep, "Sin métricas devueltas"))
                    continue

                flat_rep = self._flatten_report(metrics["report_calib"], prefix="rep_")
                row = {
                    "N_KEEP": nkeep,
                    "model_tag": metrics.get("model_tag"),
                    "calibrate_method": metrics.get("calibrate_method"),
                    "brier_calib": metrics.get("brier_calib"),
                    "logloss_calib": metrics.get("logloss_calib"),
                    "ece_calib": metrics.get("ece_calib"),
                    "trained_at": metrics.get("trained_at"),
                    # Atajos útiles:
                    "macro_precision": flat_rep.get("rep_macro avg_precision"),
                    "macro_recall": flat_rep.get("rep_macro avg_recall"),
                    "macro_f1": flat_rep.get("rep_macro avg_f1-score"),
                    "weighted_f1": flat_rep.get("rep_weighted avg_f1-score"),
                }
                # Si quieres por-clase (si existen):
                for cls in metrics["classes"]:
                    for m in ["precision", "recall", "f1-score", "support"]:
                        key = f"rep_{cls}_{m}"
                        if key in flat_rep:
                            row[key] = flat_rep[key]

                rows.append(row)

            except Exception as e:
                errs.append((nkeep, str(e)))

        df = pd.DataFrame(rows)

        # Renombres amigables (opcional)
        if "macro_f1" not in df.columns and "rep_macro avg_f1-score" in df.columns:
            df["macro_f1"] = df["rep_macro avg_f1-score"]

        # Ordenar según preferencia
        if sort_by == "macro_f1":
            df = df.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)
        elif sort_by == "logloss":
            df = df.sort_values(by="logloss_calib", ascending=True).reset_index(drop=True)
        elif sort_by == "brier":
            df = df.sort_values(by="brier_calib", ascending=True).reset_index(drop=True)
        elif sort_by == "ece":
            df = df.sort_values(by="ece_calib", ascending=True).reset_index(drop=True)
        else:
            # Fallback: macro_f1
            df = df.sort_values(by="macro_f1", ascending=False).reset_index(drop=True)

        # Exportar si se pide
        if export_csv_path:
            try:
                df.to_csv(export_csv_path, index=False)
                print(f"[OK] Resultados exportados a: {export_csv_path}")
            except Exception as e:
                print(f"[WARN] No se pudo exportar CSV: {e}")

        # Log de errores (si hubo)
        if errs:
            print("\n[Resumen de errores en el barrido]:")
            for nkeep, msg in errs:
                print(f"  N_KEEP={nkeep}: {msg}")

        return df

    def get_best_calibration_model(self, display_graphic=False,
                                   methods=("isotonic", "sigmoid", "beta", "histbin")):

        # ── Utilidades ──────────────────────────────────────────────────────────────
        def expected_calibration_error(y_true, proba, n_bins=15):
            """
            ECE multiclass (one-vs-rest promedio macro).
            y_true: Series/array de etiquetas (string/int)
            proba: ndarray [n_samples, n_classes] con predict_proba
            """
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            eces = []
            for k in range(len(classes)):
                p = proba[:, k]
                t = y_true_bin[:, k]
                # bins uniformes en probabilidad predicha
                bins = np.linspace(0, 1, n_bins + 1)
                ece = 0.0
                for i in range(n_bins):
                    mask = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else (p >= bins[i]) & (p <= bins[i + 1])
                    if np.any(mask):
                        conf = p[mask].mean()
                        acc = t[mask].mean()
                        w = mask.mean()
                        ece += w * abs(acc - conf)
                eces.append(ece)
            return float(np.mean(eces))

        class HistogramBinningCalibrator:
            """
            Calibrador simple por binning en cuantiles (multiclase via one-vs-rest).
            Regularización: mezcla con la frecuencia global de la clase (alpha).
            """

            def __init__(self, n_bins=20, alpha=0.2, quantile=True):
                self.n_bins = n_bins
                self.alpha = alpha
                self.quantile = quantile
                self.classes_ = None
                self.bin_edges_ = {}
                self.bin_means_ = {}
                self.class_priors_ = {}

            def fit(self, y_true, proba):
                self.classes_ = np.arange(proba.shape[1])
                y_true = np.asarray(y_true)
                for k in self.classes_:
                    p = proba[:, k]
                    t = (y_true == k).astype(int)
                    prior = t.mean()
                    self.class_priors_[k] = prior
                    if self.quantile:
                        edges = np.quantile(p, np.linspace(0, 1, self.n_bins + 1))
                        edges[0], edges[-1] = 0.0, 1.0
                    else:
                        edges = np.linspace(0, 1, self.n_bins + 1)
                    self.bin_edges_[k] = edges
                    means = []
                    for i in range(self.n_bins):
                        mask = (p >= edges[i]) & (p < edges[i + 1]) if i < self.n_bins - 1 else (p >= edges[i]) & (
                                p <= edges[i + 1])
                        if np.any(mask):
                            emp = t[mask].mean()
                            # mezcla con prior para suavizar
                            cal = (1 - self.alpha) * emp + self.alpha * prior
                        else:
                            cal = prior
                        means.append(cal)
                    self.bin_means_[k] = np.array(means, dtype=float)
                return self

            def predict_proba(self, proba):
                out = np.zeros_like(proba, dtype=float)
                for k in self.classes_:
                    p = proba[:, k]
                    edges = self.bin_edges_[k]
                    means = self.bin_means_[k]
                    idx = np.searchsorted(edges, p, side="right") - 1
                    idx = np.clip(idx, 0, len(means) - 1)
                    out[:, k] = means[idx]
                # renormaliza por fila para mantener suma=1 (suave)
                row_sum = out.sum(axis=1, keepdims=True)
                # Evita división por cero: mezcla con proba original si suma casi cero
                mask = row_sum.squeeze() > 1e-12
                out[mask] /= row_sum[mask]
                out[~mask] = proba[~mask]
                return out

        def fit_beta_calibration(y, proba):
            try:
                from betacal import BetaCalibration
            except Exception:
                return None  # no instalado

            classes = np.unique(y)
            y_bin = label_binarize(y, classes=classes)
            models = []

            for k in range(len(classes)):
                bc = BetaCalibration(parameters="abm")  # "abm" suele rendir bien
                bc.fit(proba[:, k].reshape(-1, 1), y_bin[:, k])
                models.append(bc)

            def _predict_one(bc, x):
                # Algunas versiones traen predict_proba; otras solo predict
                if hasattr(bc, "predict_proba"):
                    return bc.predict_proba(x).ravel()
                else:
                    return bc.predict(x).ravel()

            def predict_proba_fn(P):
                out = np.zeros_like(P, dtype=float)
                for k, bc in enumerate(models):
                    out[:, k] = _predict_one(bc, P[:, k].reshape(-1, 1))
                # renormaliza por fila para que sumen 1
                s = out.sum(axis=1, keepdims=True)
                out = np.divide(out, np.maximum(s, 1e-12))
                return out

            return predict_proba_fn

        # ── Tu flujo original (con mejoras) ────────────────────────────────────────
        if self.data is None:
            print("No hay datos para entrenar el modelo")
            return

        # df_model, features = self.preparar_features(self.data)
        df_model, features, _ = prepare_features_incremental(
            raw_df=self.data,
            preparar_features_fn=self.preparar_features,  # tu propia función
            params=params_to_features,
            cache_dir="feature_cache",
            datetime_col="datetime",
            warmup_bars=150
        )

        # --- Guardas anti-fuga de fechas tras prepare_features_incremental ---
        raw_start = self.data["datetime"].min()
        raw_end = self.data["datetime"].max()

        # Alinear tz del df_model a la de self.data (si hiciera falta)
        tz_raw = getattr(self.data["datetime"].dt, "tz", None)
        tz_mod = getattr(df_model["datetime"].dt, "tz", None)
        if (tz_raw is not None) and (tz_mod is None):
            df_model["datetime"] = df_model["datetime"].dt.tz_localize(tz_raw)
        elif (tz_raw is not None) and (tz_mod is not None) and (str(tz_raw) != str(tz_mod)):
            df_model["datetime"] = df_model["datetime"].dt.tz_convert(tz_raw)

        # Recortar estrictamente al rango del train_slice
        mask_bounds = (df_model["datetime"] >= raw_start) & (df_model["datetime"] <= raw_end)
        n_outside = int((~mask_bounds).sum())
        if n_outside:
            print(f"[LEAK GUARD] recortando {n_outside} filas fuera de {raw_start}..{raw_end}")

        df_model = df_model.loc[mask_bounds].copy()
        df_model.sort_values("datetime", inplace=True)
        df_model.reset_index(drop=True, inplace=True)

        # (opcional) sanity checks
        if not df_model["datetime"].is_monotonic_increasing:
            print("[WARN] datetime no es monótono tras recorte; ordenando por seguridad.")
            df_model = df_model.sort_values("datetime").reset_index(drop=True)

        # Log del rango resultante (para auditar)
        try:
            a, b = df_model["datetime"].iloc[0], df_model["datetime"].iloc[-1]
            print(f"Rango df_model tras recorte: {a} -> {b} (n={len(df_model)})")
        except Exception:
            pass

        L = self.lookahead
        if len(df_model) <= L:
            raise RuntimeError(f"No hay suficientes filas para excluir lookahead={L}.")

        df_train = df_model.iloc[:-L].copy()
        df_train = df_train[df_train["future_trend"].notna()]
        df_train = df_train.dropna(subset=features + ["future_trend"])

        X = df_train[features]
        y = df_train["future_trend"]

        # Split principal
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Split interno para calibración (del train; no tocar test)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
        )

        # Entrena RF base una sola vez (prefit)
        base_model = RandomForestClassifier(
            n_estimators=700, max_features="sqrt", n_jobs=-1, random_state=42
        )
        base_model.fit(X_tr, y_tr)

        # Reporte sin calibrar
        preds_vanilla = base_model.predict(X_test_scaled)
        print("Reporte de clasificación de modelo sin calibrar:")
        print(classification_report(y_test, preds_vanilla))

        base_proba_cal = base_model.predict_proba(X_cal)
        base_proba_test = base_model.predict_proba(X_test_scaled)
        classes = base_model.classes_
        y_test_encoded = pd.Series(pd.Categorical(y_test, categories=classes).codes)

        # Métricas helper (multiclase)
        def multiclass_logloss(y_true, proba, labels):
            y_idx = pd.Series(pd.Categorical(y_true, categories=labels).codes).to_numpy()
            # evita valores 0/1 extremos
            proba = np.clip(proba, 1e-12, 1 - 1e-12)
            return log_loss(y_idx, proba, labels=np.arange(len(labels)))

        def multiclass_brier(y_true, proba, labels):
            y_idx = pd.Series(pd.Categorical(y_true, categories=labels).codes).to_numpy()
            y_bin = label_binarize(y_idx, classes=np.arange(len(labels)))
            return np.mean(np.sum((proba - y_bin) ** 2, axis=1))

        # ── Probar calibradores ────────────────────────────────────────────────────
        results = []
        best = {"name": None, "metrics": None, "predict_proba": None, "wrapper": None}

        # 1) Sigmoid / Isotonic con CalibratedClassifierCV (prefit)
        for m in ("sigmoid", "isotonic"):
            if m not in methods:
                continue
            clf = CalibratedClassifierCV(base_model, method=m, cv="prefit")
            clf.fit(X_cal, y_cal)
            p_test = clf.predict_proba(X_test_scaled)
            row = {
                "method": m,
                "log_loss": multiclass_logloss(y_test, p_test, classes),
                "brier": multiclass_brier(y_test, p_test, classes),
                "ece": expected_calibration_error(y_test, p_test, n_bins=15),
            }
            results.append(row)
            if (best["metrics"] is None) or (row["brier"] < best["metrics"]["brier"]):
                best = {"name": m, "metrics": row, "predict_proba": clf.predict_proba, "wrapper": clf}

        # 2) Beta Calibration (si está disponible)
        if "beta" in methods:
            beta_fn = fit_beta_calibration(y_cal.values, base_proba_cal)
            if beta_fn is not None:
                p_test = beta_fn(base_proba_test)
                row = {
                    "method": "beta",
                    "log_loss": multiclass_logloss(y_test, p_test, classes),
                    "brier": multiclass_brier(y_test, p_test, classes),
                    "ece": expected_calibration_error(y_test, p_test, n_bins=15),
                }
                results.append(row)
                if row["brier"] < best["metrics"]["brier"]:
                    # wrap en objeto con interfaz mínima (solo predict_proba)
                    class _BetaWrapper:
                        def __init__(self, fn): self._fn = fn

                        def predict_proba(self, X): return fn(base_model.predict_proba(X))

                    fn = beta_fn
                    beta_wrapper = _BetaWrapper(beta_fn)
                    best = {"name": "beta", "metrics": row,
                            "predict_proba": lambda X: beta_fn(base_model.predict_proba(X)),
                            "wrapper": beta_wrapper}
            else:
                print("Aviso: 'betacal' no está instalado; se omite Beta Calibration.")

        # 3) Histogram/Quantile binning con suavizado
        if "histbin" in methods:
            hist = HistogramBinningCalibrator(n_bins=20, alpha=0.15, quantile=True)
            hist.fit(pd.Series(pd.Categorical(y_cal, categories=classes).codes).values, base_proba_cal)
            p_test = hist.predict_proba(base_proba_test)
            row = {
                "method": "histbin_q20_a0.15",
                "log_loss": multiclass_logloss(y_test, p_test, classes),
                "brier": multiclass_brier(y_test, p_test, classes),
                "ece": expected_calibration_error(y_test, p_test, n_bins=15),
            }
            results.append(row)
            if row["brier"] < best["metrics"]["brier"]:
                class _HistWrapper:
                    def __init__(self, h): self.h = h

                    def predict_proba(self, X): return self.h.predict_proba(base_model.predict_proba(X))

                best = {"name": "histbin_q20_a0.15", "metrics": row,
                        "predict_proba": lambda X: hist.predict_proba(base_model.predict_proba(X)),
                        "wrapper": _HistWrapper(hist)}

        # ── Mostrar tabla de resultados ────────────────────────────────────────────
        print("\n=== Comparación de calibradores (menor es mejor) ===")
        df_res = pd.DataFrame(results).sort_values("brier")
        print(df_res.to_string(index=False))

        # ── Reporte clasificador calibrado elegido ────────────────────────────────
        # Usa el mejor calibrador para predicción de clases (argmax de proba)
        p_best = best["predict_proba"](X_test_scaled)
        y_pred_best = classes[np.argmax(p_best, axis=1)]
        print(f"\nMejor calibrador: {best['name']}")
        print("Métricas del mejor calibrador:", best["metrics"])
        print("\nReporte de clasificación con calibración elegida:")
        print(classification_report(y_test, y_pred_best, labels=classes))

        # Guarda el calibrador elegido
        # Si es CalibratedClassifierCV, lo guardamos directamente; si es beta/hist, creamos un wrapper sencillo.
        self.model = best["wrapper"] if best["wrapper"] is not None else CalibratedClassifierCV(base_model,
                                                                                                method=best["name"],
                                                                                                cv="prefit").fit(X_cal,
                                                                                                                 y_cal)

        # ── Curvas de calibración (opcional) ──────────────────────────────────────
        if display_graphic:
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve

            proba_base = base_proba_test
            proba_best = p_best
            clases = classes

            plt.figure(figsize=(7, 5))
            for target_class in clases:
                idx = list(clases).index(target_class)
                y_true_bin = np.array([1 if yt == target_class else 0 for yt in y_test])

                # Sin calibrar
                prob_pred_base, prob_true_base = calibration_curve(
                    y_true_bin, proba_base[:, idx], n_bins=10, strategy='uniform'
                )
                plt.plot(prob_pred_base, prob_true_base, marker='o', label=f"Sin calibrar '{target_class}'")

                # Calibrado (mejor)
                prob_pred_cal, prob_true_cal = calibration_curve(
                    y_true_bin, proba_best[:, idx], n_bins=10, strategy='uniform'
                )
                plt.plot(prob_pred_cal, prob_true_cal, marker='o', linestyle='--', label=f"Calibrado '{target_class}'")

            fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            plt.plot([0, 1], [0, 1], 'k--', label='Perfecto')
            plt.xlabel('Probabilidad predicha')
            plt.ylabel('Frecuencia empírica')
            plt.title(f'Curvas de calibración (base vs mejor)\nEntrenado: {fecha_actual}')
            plt.legend()
            plt.grid()
            plt.show(block=False)

        return self.model

    def visualize_results(self, top_n=20):
        """
        Visualiza los resultados más importantes
        """
        if not self.results:
            print("No hay resultados para visualizar")
            return

        df_results = pd.DataFrame(self.results)

        # Top combinaciones por confianza
        top_results = df_results.nlargest(top_n, 'confidence')

        # Crear visualizaciones
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Combinaciones por Confianza',
                            'Distribución de Tendencias',
                            'Confianza por Tendencia Dominante',
                            'Samples vs Confianza'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )

        # Gráfico 1: Top combinaciones
        fig.add_trace(
            go.Bar(x=list(range(len(top_results))),
                   y=top_results['confidence'],
                   text=top_results['dominant_trend'],
                   name='Confianza'),
            row=1, col=1
        )

        # Gráfico 2: Distribución de tendencias
        trend_counts = df_results['dominant_trend'].value_counts()
        fig.add_trace(
            go.Pie(labels=trend_counts.index,
                   values=trend_counts.values,
                   name='Tendencias'),
            row=1, col=2
        )

        # Gráfico 3: Box plot de confianza por tendencia
        for trend in df_results['dominant_trend'].unique():
            trend_data = df_results[df_results['dominant_trend'] == trend]
            fig.add_trace(
                go.Box(y=trend_data['confidence'],
                       name=trend),
                row=2, col=1
            )

        # Gráfico 4: Samples vs Confianza
        fig.add_trace(
            go.Scatter(x=df_results['total_samples'],
                       y=df_results['confidence'],
                       mode='markers',
                       text=df_results['dominant_trend'],
                       name='Combinaciones'),
            row=2, col=2
        )

        fig.update_layout(height=800, showlegend=True,
                          title_text="Análisis de Patrones Estocásticos")
        fig.show()

    def evaluate_positional_relation(self, df, relation_type, k_col, d_col, e_col):
        """
        Evalúa relaciones posicionales entre indicadores
        """
        tolerance = 3.0  # Tolerancia para considerar valores "iguales"

        if relation_type == 'k>d>e':
            return (df[k_col] > df[d_col]) & (df[d_col] > df[e_col])
        elif relation_type == 'e>d>k':
            return (df[e_col] > df[d_col]) & (df[d_col] > df[k_col])
        elif relation_type == 'any':
            return pd.Series([True] * len(df), index=df.index)
        else:
            return pd.Series([False] * len(df), index=df.index)

    def evaluate_timeframe_relation(self, df, relation_type):
        """
        Evalúa relaciones entre diferentes marcos temporales
        """
        if relation_type == 'both_bullish':
            return ((df['k'] > df['d']) & (df['d'] > df['e'])) & \
                ((df['k100'] > df['d100']) & (df['d100'] > df['e100']))
        elif relation_type == 'both_bearish':
            return ((df['e'] > df['d']) & (df['d'] > df['k'])) & \
                ((df['e100'] > df['d100']) & (df['d100'] > df['k100']))
        elif relation_type == 'stoch14_bull_100_bear':
            return ((df['k'] > df['d']) & (df['d'] > df['e'])) & \
                ((df['e100'] > df['d100']) & (df['d100'] > df['k100']))
        elif relation_type == 'stoch14_bear_100_bull':
            return ((df['e'] > df['d']) & (df['d'] > df['k'])) & \
                ((df['k100'] > df['d100']) & (df['d100'] > df['e100']))
        elif relation_type == 'convergence':
            # Ambos marcos temporales moviéndose en la misma dirección
            stoch14_bullish = (df['k'] > df['d']) & (df['d'] > df['e'])
            stoch100_bullish = (df['k100'] > df['d100']) & (df['d100'] > df['e100'])
            return (stoch14_bullish & stoch100_bullish) | \
                ((~stoch14_bullish) & (~stoch100_bullish))
        elif relation_type == 'divergence':
            stoch14_bullish = (df['k'] > df['d']) & (df['d'] > df['e'])
            stoch100_bullish = (df['k100'] > df['d100']) & (df['d100'] > df['e100'])
            return (stoch14_bullish & (~stoch100_bullish)) | \
                ((~stoch14_bullish) & stoch100_bullish)
        elif relation_type == 'any':
            return pd.Series([True] * len(df), index=df.index)
        else:
            return pd.Series([False] * len(df), index=df.index)

    def analyze_specific_patterns(self, df):
        """
        Analiza patrones específicos como el ejemplo mencionado
        """
        patterns = []

        # Patrón bajista específico: K14 en zona baja < D14 en zona baja < E14 en zona alta + E100 descendiendo
        bearish_pattern_1 = (
                (df['k'] >= 0) & (df['k'] < 20) &  # K14 en zona sobreventa
                (df['d'] >= 0) & (df['d'] < 20) &  # D14 en zona sobreventa
                (df['e'] >= 60) & (df['e'] < 80) &  # E14 en zona alta
                (df['k'] < df['d']) & (df['d'] < df['e']) &  # K14 < D14 < E14
                (df['e100_direction'] == 'descending') &  # E100 descendiendo
                df['future_trend'].notna()
        )

        # Patrón alcista específico: K14 en zona alta > D14 en zona alta > E14 en zona baja + E100 ascendiendo
        bullish_pattern_1 = (
                (df['k'] >= 80) & (df['k'] <= 100) &  # K14 en zona sobrecompra
                (df['d'] >= 80) & (df['d'] <= 100) &  # D14 en zona sobrecompra
                (df['e'] >= 20) & (df['e'] < 40) &  # E14 en zona baja
                (df['k'] > df['d']) & (df['d'] > df['e']) &  # K14 > D14 > E14
                (df['e100_direction'] == 'ascending') &  # E100 ascendiendo
                df['future_trend'].notna()
        )

        # Analizar patrones
        for pattern_name, pattern_mask in [('bearish_specific', bearish_pattern_1),
                                           ('bullish_specific', bullish_pattern_1)]:
            filtered_data = df[pattern_mask]

            if len(filtered_data) >= 10:
                trend_distribution = filtered_data['future_trend'].value_counts()
                total_samples = len(filtered_data)

                pattern_result = {
                    'pattern_name': pattern_name,
                    'total_samples': total_samples,
                    'trend_distribution': trend_distribution.to_dict(),
                    'accuracy': (trend_distribution.get(pattern_name.split('_')[0], 0) / total_samples) * 100
                }
                patterns.append(pattern_result)

        return patterns

    def get_current_recommendation(self, current_k14, current_d14, current_e14,
                                   current_k100, current_d100, current_e100,
                                   current_e100_direction, current_e14_direction):
        """
        Obtiene recomendación basada en valores actuales de los indicadores
        """
        if not self.results:
            return "No hay resultados de análisis disponibles"

        # Buscar combinaciones que coincidan con los valores actuales
        matching_combinations = []

        for result in self.results:
            if result.get('type') == 'zone_positional':
                # Verificar si los valores actuales coinciden con esta combinación
                k14_match = (result['k14_min'] <= current_k14 < result['k14_max'])
                d14_match = (result['d14_min'] <= current_d14 < result['d14_max'])
                e14_match = (result['e14_min'] <= current_e14 < result['e14_max'])
                k100_match = (result['k100_min'] <= current_k100 < result['k100_max'])
                d100_match = (result['d100_min'] <= current_d100 < result['d100_max'])
                e100_match = (result['e100_min'] <= current_e100 < result['e100_max'])

                direction_e100_match = (result['e100_direction'] == current_e100_direction)
                direction_e14_match = (result['e14_direction'] == current_e14_direction)  # Agregar esta línea

                # Verificar relaciones posicionales
                stoch14_relation_match = self.check_relation_match(
                    current_k14, current_d14, current_e14, result['stoch14_relation']
                )
                stoch100_relation_match = self.check_relation_match(
                    current_k100, current_d100, current_e100, result['stoch100_relation']
                )

                if (k14_match and d14_match and e14_match and k100_match and
                        d100_match and e100_match and direction_e100_match and direction_e14_match and  # Modificar esta línea
                        stoch14_relation_match and stoch100_relation_match):
                    matching_combinations.append(result)

        if not matching_combinations:
            return "No se encontraron patrones coincidentes"

        # Ordenar por confianza y obtener la mejor recomendación
        best_match = max(matching_combinations, key=lambda x: x['confidence'])

        recommendation = {
            'dominant_trend': best_match['dominant_trend'],
            'confidence': best_match['confidence'],
            'total_samples': best_match['total_samples'],
            'distribution': {
                'bullish': best_match['bullish_pct'],
                'bearish': best_match['bearish_pct'],
                'lateral': best_match['lateral_pct']
            }
        }

        return recommendation

    def check_relation_match(self, k_val, d_val, e_val, relation_type):
        """
        Verifica si los valores actuales coinciden con la relación especificada
        """
        tolerance = 3.0

        if relation_type == 'k>d>e':
            return k_val > d_val > e_val
        elif relation_type == 'e>d>k':
            return e_val > d_val > k_val
        elif relation_type == 'any':
            return True
        else:
            return False

    def predict_realtime(self,
                         symbol: str = "AVAXUSDT",
                         interval: str = "5",
                         lookback_bars: int = 400,
                         thresholds: dict = None,
                         default_class: str = "lateral"):
        """
        Actualiza el histórico y predice la última vela usando umbrales por clase.
        Asegura que la última fila NO se use para el entrenamiento (previene data leakage).
        """
        if lookback_bars < 100:
            print(
                f"WARNING: lookback_bars={lookback_bars} es menor que 100.\n"
                "No se podrá calcular correctamente el estocástico 100 y los indicadores largos pueden fallar.\n"
                "Recomiendo usar al menos 110 o 120 para evitar NaN."
            )

        # Umbrales por defecto si no se pasan
        if thresholds is None:
            thresholds = {"bearish": 0.55, "bullish": 0.55, "lateral": 0.40}

        # 1) Descargar datos nuevos (solo OHLCV crudo)
        df_new = self.get_bybit_data_private(symbol, interval, start_date=None, end_date=None, limit=lookback_bars)

        df_new["datetime"] = pd.to_datetime(df_new["datetime"])  # sin forzar utc=False

        # forzar awareness y convertir a Bogota (UTC-5)
        if getattr(df_new["datetime"].dt, "tz", None) is None:
            df_new["datetime"] = df_new["datetime"].dt.tz_localize("UTC")

        df_new["datetime"] = df_new["datetime"].dt.tz_convert("America/Bogota")

        if df_new is None or len(df_new) == 0:
            raise RuntimeError("No se pudieron obtener velas de Bybit")

        # 2) Merge con dataset histórico completo (self.data) ANTES de calcular features
        #    Esto replica el contexto completo que tiene la simulación.
        raw_cols = ["datetime", "open", "high", "low", "close", "volume"]
        if self.data is not None and not self.data.empty:
            # Tomar solo columnas crudas de self.data para evitar conflictos
            existing_raw = self.data[raw_cols].copy()
            # Combinar histórico + nuevas velas, sin duplicados
            df_full = pd.concat([existing_raw, df_new[raw_cols]], ignore_index=True)
            df_full = df_full.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
        else:
            df_full = df_new[raw_cols].copy()

        # 3) Optimización: Usar solo las últimas 2000 velas para calcular indicadores.
        #    Esto es suficiente contexto para EMAs de 200 y otros rolling, y mantiene la velocidad.
        df_context = df_full.tail(3000).copy()
        df_processed = self.add_indicators(df_context)
        df_ready, features = self.preparar_features(df_processed)

        # Actualizar self.data con el dataset procesado (manteniendo solo la ventana relevante)
        self.data = df_ready.copy()

        if df_ready.empty:
            raise RuntimeError("No hay suficientes datos para calcular todas las features necesarias (lags, etc).")
        features = FEATURES
        # 4) Última fila válida (del dataset completo)
        last_row = df_ready[features].iloc[-1:]
        last_meta = df_ready.iloc[-1]

        # 5) Escalar y obtener probabilidades
        last_scaled = self.scaler.transform(last_row)

        # Sanity: el modelo debe exponer classes_
        if not hasattr(self.model, "classes_"):
            raise RuntimeError("El modelo no expone 'classes_' y no se puede alinear proba↔clase.")

        proba = self.model.predict_proba(last_scaled)[0]  # shape: (K,)
        classes = list(self.model.classes_)  # orden fijo del modelo

        # Checks de alineación
        if len(proba) != len(classes):
            raise RuntimeError(f"Dim mismatch: len(proba)={len(proba)} vs len(classes)={len(classes)}")

        # Clipping + renormalización defensiva (por si algún calibrador devolvió extremos)
        proba = np.clip(proba, 1e-9, 1.0)  # no bajo de 0
        s = proba.sum()
        if not np.isfinite(s) or s <= 0:
            raise RuntimeError("Probabilidades no finitas o suma <= 0.")
        proba = proba / s

        # Mapeo clase→probabilidad (garantizado con el orden del modelo)
        probabilities = {cls: float(round(p, 4)) for cls, p in zip(classes, proba)}

        # 6) Predicción con umbrales
        thr = thresholds or getattr(self, "thresholds_", {"bearish": 0.50, "bullish": 0.50, "lateral": 0.38})

        # Validar que las llaves de thresholds coincidan con las clases del modelo
        missing = set(classes) - set(thr.keys())
        if missing:
            raise RuntimeError(f"Faltan umbrales para clases: {missing}. Clases del modelo: {classes}")

        pred_class = self._apply_thresholds(
            proba,
            classes,
            thr,
            default_class=default_class,  # usa el parámetro de la función
            global_floor=0.51,
            min_margin=0.03
        )

        probs = probabilities

        ordered = sorted(probs.values(), reverse=True)
        top1 = ordered[0] if ordered else 0.0
        top2 = ordered[1] if len(ordered) > 1 else 0.0
        margin = (top1 - top2)

        pred = pred_class
        ema_200 = float(last_meta['EMA_200'])
        # prob_min = 0.66
        # if pred_class == 'bullish' and margin < 0.3:
        #     print(f"Fake bullish: Margen insuficiente")
        #     probabilities['bullish'] = min(prob_min, probabilities.get('bullish'))
        #
        # if pred_class == 'bearish' and margin < 0.4:
        #     print(f"Fake bearish: Margen insuficiente)")
        #     probabilities['bearish'] = min(prob_min, probabilities.get('bearish'))
        if pred in ("bullish", "bearish") and getattr(self, "m2_model", None) is not None:
            try:
                # Determinar set de features para M2
                feats_m2 = FEATURES + FEATS_M2

                # Construir vector x_m2 con orden exacto
                # Fuente principal: última fila de df_ready (ya con tus features preparados)
                src_series = df_ready.iloc[-1]



                x_dict = {}
                # soporte para direction_sign / dummies
                dir_sign_val = 1.0 if pred == "bullish" else -1.0
                dir_onehot = {"dir_bullish": 1.0 if pred == "bullish" else 0.0,
                              "dir_bearish": 1.0 if pred == "bearish" else 0.0}

                for f in feats_m2:
                    if f == "direction_sign":
                        x_dict[f] = dir_sign_val
                    elif f in ("dir_bullish", "dir_bearish"):
                        x_dict[f] = dir_onehot.get(f, 0.0)
                    elif f == "p_hat":
                        x_dict[f] = probabilities[pred]
                    elif f == "margin":
                        x_dict[f] = margin
                    elif f in src_series.index:
                        x_dict[f] = float(src_series[f])
                    elif f in df_ready.columns:
                        x_dict[f] = float(df_ready.iloc[-1][f])
                    else:
                        # Si falta, deja NaN para detección
                        x_dict[f] = np.nan

                # Verifica NaNs
                x_dict['price_change'] = df_ready["price_change"].iloc[-2]
                x_vals = [x_dict[f] for f in feats_m2]
                if any([not np.isfinite(v) for v in x_vals]):
                    # puedes cambiar por imputación si quieres continuar igual
                    raise RuntimeError(
                        f"Features M2 con NaN/inf en inferencia: {[feats_m2[i] for i, v in enumerate(x_vals) if not np.isfinite(v)]}")

                X_m2 = np.array([x_vals], dtype="float32")

                # Inferencia M2 (RF no requiere scaler)
                proba_m2 = self.m2_model.predict_proba(X_m2)[0]  # [p0, p1]
                p_meta = float(np.clip(proba_m2[1], 0.0, 1.0))  # prob. de “saldrá bien”
                p_meta_used = round(p_meta, 4)

                # Reemplazar la probabilidad de la dirección predicha por p_meta
                probabilities[pred] = p_meta_used

                # (Opcional) si quieres actualizar margen mostrado:
                ordered = sorted(probabilities.values(), reverse=True)
                top1 = ordered[0] if ordered else 0.0
                top2 = ordered[1] if len(ordered) > 1 else 0.0
                margin = top1 - top2
            except Exception as e:
                # No rompas la predicción si M2 falla: deja M1 tal cual y reporta aviso.
                # if getattr(self, "verbose", 0) > 0:
                print(f"[M2] Inferencia omitida: {e}")


        result = {
            "prediction": pred,
            "probabilities": probabilities,
            "thresholds": thresholds,
            "timestamp": last_meta['datetime'],
            "last_row_atr": float(last_meta['ATR']),
            "last_row_close": float(last_meta['close']),
            "last_row_ema200": ema_200,
            "raw_df": df_ready.tail(lookback_bars)[["datetime", "open", "high", "low", "close", "volume", "macd", "macd_signal", "macd_hist"]].copy()
        }

        return result

    def _pnl_usdt_linear(self, side, entry, exit_, qty, fee_rate=0.00055, include_both_sides=True):
        """
        PnL para perp lineal USDT.
        side: 'long'/'short'
        fee_rate: comisión por lado (p.ej., 0.0006 = 6 bps).
        include_both_sides=True → descuenta entrada y salida (2 * fee_rate).
        """
        if qty is None:
            return None
        sign = 1 if side == "long" else -1
        gross = (exit_ - entry) * qty * sign
        fees = 0.0
        if include_both_sides:
            notional_entry = entry * qty
            notional_exit = exit_ * qty
            fees = fee_rate * (notional_entry + notional_exit)
        else:
            # solo fee de cierre (por si el broker ya te descontó la de entrada)
            notional_exit = exit_ * qty
            fees = fee_rate * notional_exit
        return gross - fees

    def _resolve_tp_sl_order_in_bar(self, symbol, interval_start_dt, interval_end_dt, side, tp, sl):
        """
        Baja a 1m entre interval_start_dt y interval_end_dt para resolver si tocó TP o SL primero.
        Devuelve 'tp' | 'sl' | None (si no se tocó nada).pre
        """
        try:
            # Bybit espera timestamps en ms; convertimos:
            start_date = interval_start_dt.strftime("%Y-%m-%d")
            end_date = interval_end_dt.strftime("%Y-%m-%d")
            # mejor usar modo rango con 1m y recortar por timestamps después
            df_1m = self.get_bybit_data_private(symbol=symbol, interval="1",
                                                start_date=start_date, end_date=end_date, limit=0)
            if df_1m is None or df_1m.empty:
                return None
            # recorta exactamente el rango
            mask = (df_1m["datetime"] >= interval_start_dt) & (df_1m["datetime"] <= interval_end_dt)
            df_1m = df_1m.loc[mask].sort_values("datetime")
            for _, r in df_1m.iterrows():
                high = float(r["high"]);
                low = float(r["low"])
                # orden por prioridad dentro de 1m: si se tocan ambos dentro del MINUTO,
                # asumimos que primero tocó el nivel más cercano al precio de apertura de ese 1m
                # (si quieres ultra-precisión necesitarías trades-ticks)
                open_ = float(r["open"])
                # chequeo “ambos tocados”
                hit_tp = (high >= tp) if side == "long" else (low <= tp)
                hit_sl = (low <= sl) if side == "long" else (high >= sl)
                if hit_tp and not hit_sl:
                    return "tp"
                if hit_sl and not hit_tp:
                    return "sl"
                if hit_tp and hit_sl:
                    # decide por cercanía al open
                    dist_tp = abs(tp - open_)
                    dist_sl = abs(sl - open_)
                    return "tp" if dist_tp < dist_sl else "sl"
            return None
        except Exception:
            return None

    def check_and_close_from_market(self, symbol: str, df_last: pd.DataFrame):
        """
        Verifica si una posición abierta tocó TP/SL en cualquiera de las velas *desde la apertura
        (o desde el último chequeo)*. Si cierra, calcula PnL y lo registra en TradeLogger.
        Compatible con TradeLogger provisto.
        """
        pos = self.tradelogger.get_open_position(symbol)
        if not pos:
            return  # no hay posición

        side = pos.get("side")
        qty = pos.get("qty")
        trade_id = pos.get("trade_id")
        order_id = pos.get("order_id")
        fee_rate = pos.get("fee_rate", getattr(self, "fee_rate", 0.00055))

        # entry/tp/sl deberían haberse inyectado a open_positions con enrich_open_position
        entry_price = float(pos.get("entry_price", 0) or 0)
        tp = float(pos.get("take_profit", 0) or 0)
        sl = float(pos.get("stop_loss", 0) or 0)
        if not side or not qty or entry_price <= 0 or tp <= 0 or sl <= 0:
            # Info insuficiente: evita falsos positivos
            return

        if df_last is None or df_last.empty:
            return

        # ---- Preparar dataframe y deducir intervalo (min) ----
        df_last = df_last.sort_values("datetime").copy()
        try:
            dmed = pd.to_timedelta(df_last["datetime"].diff().dropna().median())
            interval_minutes = int(max(1, round(dmed / pd.Timedelta(minutes=1))))
        except Exception:
            interval_minutes = 5  # fallback razonable

        # ---- Punto de inicio: desde opened_utc y/o last_check_dt de la posición ----
        # TradeLogger guarda opened_utc en log_order_ack; last_check_dt lo guardaremos nosotros.
        opened_utc = pos.get("opened_utc")
        last_check_dt = pos.get("last_check_dt")

        try:
            opened_dt = pd.to_datetime(opened_utc) if opened_utc else None
        except Exception:
            opened_dt = None
        try:
            last_check_dt = pd.to_datetime(last_check_dt) if last_check_dt else None
        except Exception:
            last_check_dt = None

        # Comenzar 1 intervalo antes para cubrir “apertura dentro de vela”
        candidates = []
        if opened_dt is not None:
            candidates.append(opened_dt - pd.Timedelta(minutes=interval_minutes))
        if last_check_dt is not None:
            candidates.append(last_check_dt - pd.Timedelta(minutes=interval_minutes))
        if not candidates:
            # Sin marcas: mirar al menos las últimas 2 velas para no perder un evento reciente
            candidates.append(df_last["datetime"].iloc[-1] - pd.Timedelta(minutes=2 * interval_minutes))

        start_dt = max(min(candidates), df_last["datetime"].iloc[0])

        # ---- Slicing de velas a revisar ----
        df = df_last[df_last["datetime"] >= start_dt].copy()
        if df.empty:
            # Aun así, actualizamos last_check_dt a la última de df_last para no reprocesar en loop
            try:
                self.tradelogger.enrich_open_position(symbol, last_check_dt=str(df_last["datetime"].iloc[-1]))
            except Exception:
                pass
            return

        # ---- Recorrer en orden cronológico y evaluar TP/SL ----
        for _, row in df.iterrows():
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_open_dt = pd.to_datetime(row["datetime"])
            bar_end_dt = bar_open_dt + pd.Timedelta(minutes=interval_minutes)

            hit_tp = (bar_high >= tp) if side == "long" else (bar_low <= tp)
            hit_sl = (bar_low <= sl) if side == "long" else (bar_high >= sl)

            if not hit_tp and not hit_sl:
                continue

            # Si ambos en la misma vela: resolver con 1m
            if hit_tp and hit_sl:
                # Debe existir este método en tu clase; si devuelve None, priorizamos SL
                first = self._resolve_tp_sl_order_in_bar(symbol, bar_open_dt, bar_end_dt, side, tp, sl)
                if first is None:
                    first = "sl"
            else:
                first = "tp" if hit_tp else "sl"

            exit_reason = first
            exit_price = tp if first == "tp" else sl

            # ---- PnL ----
            pnl_usdt = self._pnl_usdt_linear(
                side, entry_price, exit_price, qty, fee_rate=fee_rate, include_both_sides=True
            )

            # PnL en R (opcional)
            pnl_r = None
            try:
                R = abs(entry_price - sl) if side == "long" else abs(sl - entry_price)
                sign = 1 if side == "long" else -1
                pnl_r = ((exit_price - entry_price) * sign) / (R if R != 0 else 1e-6)
            except Exception:
                pass

            # ---- Log de cierre y limpieza ----
            self.tradelogger.log_close(
                symbol=symbol,
                exit_reason=exit_reason,
                exit_price=exit_price,
                pnl_usdt=float(pnl_usdt) if pnl_usdt is not None else None,
                pnl_r=float(pnl_r) if pnl_r is not None else None,
                order_id=order_id,
                trade_id=trade_id
            )
            print(f"\n exit_reason={exit_reason} exit_price={exit_price} \n")
            return  # ya cerramos

        # Si no cerró, actualizar last_check_dt a la última vela procesada
        try:
            self.tradelogger.enrich_open_position(symbol, last_check_dt=str(df["datetime"].iloc[-1]))
        except Exception:
            pass

        # Si no cerró, actualizar last_check_dt a la última vela procesada
        try:
            self.tradelogger.enrich_open_position(symbol, last_check_dt=str(df["datetime"].iloc[-1]))
        except Exception:
            pass
        """
        Usa la última/s velas (5m) ya descargadas para verificar si el TP/SL de la posición abierta se tocó.
        Si cierra, calcula PnL y llama tradelogger.log_close(...).
        """
        pos = self.tradelogger.get_open_position(symbol)
        if not pos:
            return  # no hay posición

        side = pos.get("side")
        entry_price = float(pos.get("entry_price", 0))
        tp = float(pos.get("take_profit", 0))
        sl = float(pos.get("stop_loss", 0))
        qty = pos.get("qty")
        fee_rate = pos.get("fee_rate", getattr(self, "fee_rate", 0.0006))
        trade_id = pos.get("trade_id")
        if not side or not entry_price or not qty or not tp or not sl:
            return  # info insuficiente

        # Asegúrate de tener al menos 1 vela cerrada reciente:
        if df_last is None or df_last.empty:
            return

        # Tomamos la última vela cerrada y, opcionalmente, también la anterior por si la posición se abrió “entre velas”
        # (si quieres, puedes filtrar desde la hora de apertura de la posición)
        df = df_last.sort_values("datetime").tail(2)

        # Recorremos las velas en orden cronológico
        for i in range(len(df)):
            row = df.iloc[i]
            bar_open = float(row["open"])
            bar_high = float(row["high"])
            bar_low = float(row["low"])
            bar_open_dt = pd.to_datetime(row["datetime"])
            # fin del intervalo = open + 5 minutos
            bar_end_dt = bar_open_dt + pd.Timedelta(
                minutes=int(df["datetime"].diff().median() / pd.Timedelta(minutes=1) if len(df) > 1 else 5))

            # ¿tocó niveles en esta vela?
            hit_tp = (bar_high >= tp) if side == "long" else (bar_low <= tp)
            hit_sl = (bar_low <= sl) if side == "long" else (bar_high >= sl)

            if not hit_tp and not hit_sl:
                continue  # nada en esta barra

            # Si ambos en la misma vela, resuelve bajando a 1m
            if hit_tp and hit_sl:
                first = self._resolve_tp_sl_order_in_bar(symbol, bar_open_dt, bar_end_dt, side, tp, sl)
                if first is None:
                    # regla conservadora: prioriza SL (peor caso)
                    first = "sl"
            else:
                first = "tp" if hit_tp else "sl"

            # Determina precio de salida (si quieres precio exacto del nivel, usa tp/sl; si prefieres close real, usa bar_close)
            exit_reason = first
            exit_price = tp if first == "tp" else sl

            # Calcula PnL
            pnl_usdt = self._pnl_usdt_linear(side, entry_price, exit_price, qty, fee_rate=fee_rate,
                                             include_both_sides=True)

            # (Opcional) PnL en R si dejaste reward_risk en la señal:
            pnl_r = None
            # Si guardaste el R “unitario” (= distancia a SL), puedes hacer:
            # R = abs(entry - sl); pnl_r = (exit - entry)/R * sign
            try:
                R = abs(entry_price - sl) if side == "long" else abs(sl - entry_price)
                sign = 1 if side == "long" else -1
                pnl_r = ((exit_price - entry_price) * sign) / (R if R != 0 else 1e-6)
            except Exception:
                pass

            # Loguea cierre y limpia posición
            self.tradelogger.log_close(
                symbol=symbol,
                exit_reason=exit_reason,
                exit_price=exit_price,
                pnl_usdt=float(pnl_usdt) if pnl_usdt is not None else None,
                pnl_r=float(pnl_r) if pnl_r is not None else None,
                order_id=pos.get("order_id"),
                trade_id=trade_id
            )
            print(f"exit_reason={exit_reason} exit_price={exit_price}")
            # Si además cierras en el broker vía API, puedes dispararlo aquí.
            break  # ya cerramos; sal del loop

    def send_email(self, subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = email_sender
        msg['To'] = EMAIL_RECEIVER
        try:
            with smtplib.SMTP(EMAIL_SMTP, EMAIL_PORT) as server:
                server.starttls()
                server.login(user=email_sender, password=email_password)
                server.sendmail(email_sender, EMAIL_RECEIVER, msg.as_string())
            print("Alerta email por enviada ✅")
        except Exception as e:
            print(f"Error enviando email: {e}")

    def get_usdt_balance(self, session_local):
        """
        Obtiene el balance disponible en USDT de la cuenta UNIFIED.

        :param session_local: Objeto de sesión HTTP de PyBit para interactuar con la API de Bybit.
        :return: Balance de USDT como float. Retorna 0 en caso de error.
        """
        try:
            wallet_balance = session_local.get_wallet_balance(accountType="UNIFIED")
            for coin in wallet_balance['result']['list'][0]['coin']:
                if coin['coin'] == 'USDT':
                    usdt_balance = coin['walletBalance']
                    print(f"Balance de USDT disponible: {usdt_balance}")
                    return float(usdt_balance)
        except Exception as e:
            print(f"Error al obtener el balance: {str(e)}")
            return 0

    def calculate_qty(self, cantidad_usd, precio_actual, leverage, symbol):
        """
        Calcula la cantidad del activo a comprar/vender basada en el monto en USD y el apalancamiento.

        :param cantidad_usd: Monto en USD que se desea operar.
        :param precio_actual: Precio actual del activo.
        :param leverage: Apalancamiento a utilizar.
        :param symbol: Símbolo del par de trading (ej. 'BTCUSDT').
        :return: Cantidad del activo ajustada y redondeada según reglas específicas del símbolo y magnitud.
        """
        qty = (cantidad_usd / precio_actual) * leverage
        if (symbol == 'BTCUSDT') and qty < 1:
            qty = round(qty, 3)
        elif 1 <= qty < 100:
            qty = round(qty, 1)
        elif qty > 99:
            qty = round(qty)
        else:
            qty = round(qty, 2)
        print(f"Cantidad calculada: {qty}")
        return qty

    def monitor_predictions(self, symbol="AVAXUSDT", interval="5"):

        def prepare_rule_inputs(df: pd.DataFrame, window: int = 2000, min_periods: int = 200) -> pd.DataFrame:
            """
            Precalcula:
            - Cuartiles rodantes (Q1..Q4) para las columnas necesarias (past-only).
            - Señales derivadas simples (vwap_up, k100<e100, etc.).

            Devuelve un DataFrame con columnas originales + *_q (labels) + señales.
            """
            need_quartiles = [
                "dist_R1", "dist_R2", "dist_S1", "dist_S2",
                "volatility_10", "volatility_50",
                "VWAP", "dist_PP", "atr_norm"
            ]
            # <- CAMBIO MINIMO: columnas opcionales si existen en df
            optional_quartiles = [
                "VWAP", "k", "e", "e100", "k100", "rsi_14", "macd", "macd_signal", "dist_R2", "atr_norm",
                "dist_PP",
            ]

            def rolling_ntile_labels_past_only(
                    s: pd.Series,
                    n: int = 5,
                    window: int = 4300,
                    min_periods: int = 200,
            ) -> pd.Series:
                """
                Discretiza s en n-tiles usando sólo historia (shift(1)) y cuantiles rolling.
                Devuelve etiquetas 'Q1'..'Qn' (category). Sin fuga temporal.
                """
                s_shift = s.shift(1)

                # cortes interiores: p = 1/n, 2/n, ..., (n-1)/n
                probs = [(k / n) for k in range(1, n)]
                # calculamos cada cuantila rolling una sola vez
                cuts = []
                for p in probs:
                    cuts.append(s_shift.rolling(window=window, min_periods=min_periods).quantile(p))

                labels = pd.Series(index=s.index, dtype="object")
                valid_mask = pd.Series(True, index=s.index)
                for c in cuts:
                    valid_mask &= c.notna()

                # Asignación por tramos (Q1..Qn)
                # Caso Q1: s <= cut1
                labels.loc[valid_mask & (s <= cuts[0])] = "Q1"
                # Casos intermedios: cut(k-1) < s <= cut(k)
                for k in range(1, n - 1):
                    lower = cuts[k - 1]
                    upper = cuts[k]
                    labels.loc[valid_mask & (s > lower) & (s <= upper)] = f"Q{k + 1}"
                # Caso Qn: s > cut(n-1)
                labels.loc[valid_mask & (s > cuts[-1])] = f"Q{n}"

                return labels.astype("category")

            out = df.copy()

            # Etiquetas de cuartil rodante (una sola vez por columna)
            for col in need_quartiles:
                out[f"{col}_q"] = rolling_ntile_labels_past_only(out[col], window=window, min_periods=min_periods)

            # <- CAMBIO MINIMO: genera *_q sólo si esas columnas existen
            for col in optional_quartiles:
                if col in out.columns and f"{col}_q" not in out.columns:
                    out[f"{col}_q"] = rolling_ntile_labels_past_only(out[col], window=window, min_periods=min_periods)

            # Señales derivadas (vectorizadas)
            out["vwap_up"] = out["VWAP"].diff().gt(0)  # True si VWAP sube vs vela previa
            out["vwap_down"] = out["VWAP"].diff().lt(0)
            out["e_down"] = out["e"].diff().lt(0)
            out["d_down"] = out["d"].diff().lt(0)

            out["e100_up"] = out["e100"].diff().gt(0)
            out["e100_down"] = out["e100"].diff().lt(0)
            out["macd_up"] = out["macd"].diff().gt(0)
            out["macd_down"] = out["macd"].diff().lt(0)

            out["atr_down"] = out["ATR"].diff().lt(0)
            out["macd_hist_down"] = out["macd_hist"].diff().lt(0)
            out["macd_signal_down"] = out["macd_signal"].diff().lt(0)
            out["macd_signal_up"] = out["macd_signal"].diff().gt(0)



            out["k100_lt_e100"] = out["k100"] < out["e100"]
            out["k100_gt_e100"] = out["k100"] > out["e100"]
            out["k_gt_e"] = out["k"] > out["e"]
            out["k_lt_e"] = out["k"] < out["e"]
            out["d_lt_e"] = out["d"] < out["e"]
            out["d_gt_e"] = out["d"] > out["e"]


            out["macd_gt_signal"] = out["macd"] > out["macd_signal"]
            out["e_lt_25"] = out["e"] < 25
            out["e100_lt_25"] = out["e100"] <= 25
            out["dist_PP_lt_0"] = out["dist_PP"] < 0

            out["dist_R1_gt_0"] = out["dist_R1"] > 0
            out["dist_R2_gt_0"] = out["dist_R2"] > 0

            out["dist_S1_gt_0"] = out["dist_S1"] > 0
            out["dist_S2_gt_0"] = out["dist_S2"] > 0
            out["rsi_40_60"] = out["rsi_14"].between(40, 60, inclusive="left").fillna(False)

            return out

        last_prediction = None
        last_minute_checked = None
        min_margin = getattr(self, "min_margin", 0.05)

        while True:
            now = datetime.now()
            minute = now.minute
            second = now.second

            if (minute % 5 == 4) and (last_minute_checked != minute):
                # Espera cierre de vela
                if second > 0:
                    time.sleep(60 - second)

                # === Recalcula 'now' tras dormir ===
                now = datetime.now()

                # 1) Predicción en vivo
                try:
                    resultado = self.predict_realtime(symbol=symbol, interval=interval)
                    self.check_and_close_from_market(symbol, resultado.get("raw_df"))
                except Exception as e:
                    self.tradelogger.log_info(event="predict_error", error=str(e))
                    print(e)
                    time.sleep(5)
                    continue

                # Preparar reglas para filtrar ordenes
                # dfp = prepare_rule_inputs(self.data, window=4300, min_periods=200)

                pred = resultado.get('prediction')
                probs = resultado.get('probabilities', {}) or {}
                proba_pred = float(probs.get(pred, 0.0))

                ordered = sorted(probs.values(), reverse=True)
                top1 = ordered[0] if ordered else 0.0
                top2 = ordered[1] if len(ordered) > 1 else 0.0
                margin = (top1 - top2)
                margin_ok = (margin >= min_margin)

                # 2) Timestamp a Bogotá
                utc_time = resultado.get('timestamp')
                ts = pd.Timestamp(utc_time)
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                bogota_time = ts.tz_convert('America/Bogota')

                # 3) Cálculos de RR/Kelly (con guardas de ATR)
                atr = resultado.get('last_row_atr')
                close_price = resultado.get('last_row_close')
                ema200 = resultado.get('last_row_ema200')
                data_df = resultado.get("raw_df")

                atr_ok = (atr is not None) and (atr > 0)

                # Evita asignar 'short' por defecto cuando es lateral
                side_prediction = "long" if pred == "bullish" else ("short" if pred == "bearish" else None)
                margin_pp = margin * 100.0

                # Calcula TP/SL solo si es operable
                take_profit = stop_loss = reward_risk = kelly_fraction = None
                eligible = (side_prediction is not None) and margin_ok and atr_ok

                if eligible:
                    profit_factor = self.atr_mult
                    loss_factor = self.atr_stop
                    if side_prediction == "long":
                        take_profit = close_price + (profit_factor * atr)
                        stop_loss = close_price - (loss_factor * atr)
                    else:
                        take_profit = close_price - (profit_factor * atr)
                        stop_loss = close_price + (loss_factor * atr)

                    ganancia = abs(take_profit - close_price)
                    perdida = abs(stop_loss - close_price)
                    reward_risk = ganancia / (perdida if perdida != 0 else 1e-6)
                    p = proba_pred
                    q = 1 - p
                    kelly_fraction = max(0.0, min(p - (q / reward_risk), 0.95))

                # Gate sólo si es elegible
                self.umbral_strong = 0.6
                self.umbral_soft = 0.6

                gate = None
                if eligible:
                    gate = "strong" if self.umbral_strong <= proba_pred < 0.99 else None

                # 1) Log de señal (marca elegibilidad)
                trade_id = self.tradelogger.log_signal(
                    symbol=symbol, timeframe=interval, side=side_prediction, prob=proba_pred,
                    margin_pp=margin_pp, probs_dict=probs, kelly=kelly_fraction, rr=reward_risk,
                    take_profit=take_profit, stop_loss=stop_loss, price=close_price,
                    reason="pred_change" if pred != last_prediction else "no_change_reentry",
                    gate=gate
                )
                # También puedes registrar la elegibilidad explicitamente:
                # self.tradelogger.log_info(event="signal_meta", trade_id=trade_id, eligible=bool(eligible),
                #                           margin_ok=bool(margin_ok), atr_ok=bool(atr_ok))

                # =========================
                # (A) RUTAS DE CIERRE (independientes de eligible/gate)
                # =========================
                is_real_account = False
                # A1) Cierre por LATERAL fuerte
                lateral_close_trigger = (pred == "lateral" and proba_pred >= 0.9)
                if lateral_close_trigger and self.tradelogger.has_open_position(symbol):
                    info_positions = self.tradelogger.open_positions.get(symbol) or {}
                    entry_price = info_positions.get("entry_price")
                    side_position = info_positions.get("side")

                    if side_position == "long" and entry_price is not None and close_price > entry_price:
                        # Get position info before closing
                        pos = self.tradelogger.get_open_position(symbol)
                        qty = pos.get("qty")
                        order_id = pos.get("order_id")
                        trade_id = pos.get("trade_id")
                        fee_rate = pos.get("fee_rate", getattr(self, "fee_rate", 0.00055))

                        # Close at broker
                        self.close_order("sell", real_account=is_real_account)

                        # Calculate PnL
                        pnl_usdt = self._pnl_usdt_linear(
                            side_position, entry_price, close_price, qty,
                            fee_rate=fee_rate, include_both_sides=True
                        )

                        # Update tradelogger state
                        self.tradelogger.log_close(
                            symbol=symbol,
                            exit_reason="manual_lateral",
                            exit_price=close_price,
                            pnl_usdt=float(pnl_usdt) if pnl_usdt is not None else None,
                            pnl_r=None,
                            order_id=order_id,
                            trade_id=trade_id
                        )

                        self.tradelogger.log_info(event="closed_on_lateral", symbol=symbol, prob=proba_pred,
                                                  note="Cierre por lateral>=0.85 con PnL favorable")
                        last_prediction = pred
                        last_minute_checked = minute
                        continue

                    if side_position == "short" and entry_price is not None and close_price < entry_price:
                        # Get position info before closing
                        pos = self.tradelogger.get_open_position(symbol)
                        qty = pos.get("qty")
                        order_id = pos.get("order_id")
                        trade_id = pos.get("trade_id")
                        fee_rate = pos.get("fee_rate", getattr(self, "fee_rate", 0.00055))

                        # Close at broker
                        self.close_order("buy", real_account=is_real_account)

                        # Calculate PnL
                        pnl_usdt = self._pnl_usdt_linear(
                            side_position, entry_price, close_price, qty,
                            fee_rate=fee_rate, include_both_sides=True
                        )

                        # Update tradelogger state
                        self.tradelogger.log_close(
                            symbol=symbol,
                            exit_reason="manual_lateral",
                            exit_price=close_price,
                            pnl_usdt=float(pnl_usdt) if pnl_usdt is not None else None,
                            pnl_r=None,
                            order_id=order_id,
                            trade_id=trade_id
                        )

                        self.tradelogger.log_info(event="closed_on_lateral", symbol=symbol, prob=proba_pred,
                                                  note="Cierre por lateral>=0.80 con PnL favorable")
                        last_prediction = pred
                        last_minute_checked = minute
                        continue

                # A2) Cierre por SEÑAL OPUESTA con proba_pred >= 0.70 (sin gate)
                #     Sólo aplica si la nueva predicción es direccional (bullish/bearish).
                if pred in ("bullish", "bearish") and self.tradelogger.has_open_position(symbol):
                    info_positions = self.tradelogger.open_positions.get(symbol) or {}
                    entry_price = info_positions.get("entry_price")
                    side_position = info_positions.get("side")

                    # Qué lado implica la nueva señal
                    new_side = "long" if pred == "bullish" else "short"
                    is_opposite = (side_position is not None and side_position != new_side)

                    if is_opposite and proba_pred >= self.opposite_exit_threshold:
                        # Criterio de PnL favorable (tal como lo tenías):
                        # Calculate fee-adjusted breakeven price
                        # For LONG: Exit > Entry * (1+fee) / (1-fee)
                        # For SHORT: Exit < Entry * (1-fee) / (1+fee)
                        fee_rate = getattr(self, "fee_rate", 0.00055)
                        
                        if side_position == "long" and entry_price is not None:
                            breakeven_price = entry_price * (1 + fee_rate) / (1 - fee_rate)
                            if close_price > breakeven_price:
                                # Get position info before closing
                                pos = self.tradelogger.get_open_position(symbol)
                                qty = pos.get("qty")
                                order_id = pos.get("order_id")
                                trade_id = pos.get("trade_id")
                                fee_rate = pos.get("fee_rate", getattr(self, "fee_rate", 0.00055))

                                # Close at broker
                                self.close_order("sell", real_account=is_real_account)

                                # Calculate PnL
                                pnl_usdt = self._pnl_usdt_linear(
                                    side_position, entry_price, close_price, qty,
                                    fee_rate=fee_rate, include_both_sides=True
                                )

                                # Update tradelogger state
                                self.tradelogger.log_close(
                                    symbol=symbol,
                                    exit_reason="manual_opposite",
                                    exit_price=close_price,
                                    pnl_usdt=float(pnl_usdt) if pnl_usdt is not None else None,
                                    pnl_r=None,
                                    order_id=order_id,
                                    trade_id=trade_id
                                )

                                self.tradelogger.log_info(event="closed_on_opposite", symbol=symbol, prob=proba_pred,
                                                          note="Cierre por opuesto>=0.85 con PnL favorable")
                                print("Cierre por opuesto>=0.85 con PnL favorable")
                                # No abrimos aún; dejamos que el flujo de entrada lo decida abajo (si aplica).
                        elif side_position == "short" and entry_price is not None:
                            breakeven_price = entry_price * (1 - fee_rate) / (1 + fee_rate)
                            if close_price < breakeven_price:
                                # Get position info before closing
                                pos = self.tradelogger.get_open_position(symbol)
                                qty = pos.get("qty")
                                order_id = pos.get("order_id")
                                trade_id = pos.get("trade_id")
                                fee_rate = pos.get("fee_rate", getattr(self, "fee_rate", 0.00055))

                                # Close at broker
                                self.close_order("buy", real_account=is_real_account)

                                # Calculate PnL
                                pnl_usdt = self._pnl_usdt_linear(
                                    side_position, entry_price, close_price, qty,
                                    fee_rate=fee_rate, include_both_sides=True
                                )

                                # Update tradelogger state
                                self.tradelogger.log_close(
                                    symbol=symbol,
                                    exit_reason="manual_opposite",
                                    exit_price=close_price,
                                    pnl_usdt=float(pnl_usdt) if pnl_usdt is not None else None,
                                    pnl_r=None,
                                    order_id=order_id,
                                    trade_id=trade_id
                                )

                                self.tradelogger.log_info(event="closed_on_opposite", symbol=symbol, prob=proba_pred,
                                                          note="Cierre por opuesto>=0.75 con PnL favorable")

                # 4) Decisión de trade
                if pred != last_prediction:
                    print("\n")
                    print(f"[{bogota_time.strftime('%Y-%m-%d %H:%M:%S')}] NUEVA PREDICCIÓN: {pred.upper()} "
                          f"(prob: {proba_pred * 100:.1f}%) | margen: {margin_pp:.1f} pp")
                    print("Detalle probabilidades:", probs)


                    # =========================
                    # (B) ENTRADA/RE-ENTRADA (requiere eligible + gate)
                    # =========================
                    if eligible and gate in ("soft", "strong"):
                        if self.tradelogger.has_open_position(symbol):
                            self.tradelogger.log_info(note="Señal ignorada: ya hay posición abierta",
                                                      symbol=symbol, side=side_prediction, trade_id=trade_id)
                        else:
                            # validate_order = validate_start_order_fast(dfp, pred)
                            # if validate_order and validate_order.get("enter"):
                            if True:
                                # print(validate_order["rule"])
                                self.start_order(pred, symbol, bogota_time, proba_pred, probs,
                                             atr, df=data_df, close_price=close_price, real_account=is_real_account, ema200=ema200, trade_id=trade_id)
                            # else:
                            #     print(f"Prevent enter {validate_order["rule"]}")
                    else:
                         print("Señal no elegible (lateral, sin margen, sin ATR o bajo umbral).")
                    last_prediction = pred

                else:
                    print(f"[{bogota_time.strftime('%Y-%m-%d %H:%M:%S')}] 'Sin cambios': {pred.upper()} "
                          f"(prob: {proba_pred * 100:.1f}%) | margen: {margin_pp:.1f} pp")
                    print(probs)

                    # Re-entrada solo si sube a strong y sigue elegible
                    if eligible and self.umbral_strong <= proba_pred:
                        if self.tradelogger.has_open_position(symbol):
                            self.tradelogger.log_info(note="Señal ignorada: ya hay posición abierta",
                                                      symbol=symbol, side=side_prediction, trade_id=trade_id)
                        else:
                            # validate_order = validate_start_order_fast(dfp, pred)
                            # if validate_order and validate_order.get("enter"):
                            if True:
                                # print(validate_order["rule"])
                                self.start_order(pred, symbol, bogota_time, proba_pred, probs,
                                             atr, df=data_df, close_price=close_price, real_account=is_real_account, ema200=ema200, trade_id=trade_id)
                            # else:
                            #     print(f"Prevent enter {validate_order["rule"]}")
                    else:
                        print("Señal por debajo del umbral strong o no elegible — no trade.")

                last_minute_checked = minute

            else:
                time.sleep(5)

            # === Recalcula 'now' aquí también antes de ventanas de reinicio ===
            now = datetime.now()

            # Ventanas de reinicio
            if now.hour in {RESTART_HOUR_1 - 1} and now.minute == 56:
                break

            if now.hour == 0 and now.minute == 59:
                today_str_ = f"{now.day:02d}"
                tomorrow_ = now + timedelta(days=1)
                tomorrow_day_ = f"{tomorrow_.day:02d}"
                mont_str_ = f"{now.month:02d}"
                filename_ = f'AVAXUSDT_5m_2022-01-01_2026-{mont_str_}-{today_str_}.csv'
                if os.path.exists(filename_):
                    name, extension_ = os.path.splitext(filename_)
                    new_name_ = name[:-2] + tomorrow_day_ + extension_
                    os.rename(filename_, new_name_)
                    print("Nuevo nombre de archivo:", new_name_)

    def start_order(self, pred, symbol, bogota_time, proba, resultado, atr, df, close_price, real_account, ema200,
                    trade_id=None):
        # @TODO refactorizar close price y atr que salgan de df
        m0 = df.iloc[-1]["macd"]
        m1 = df.iloc[-2]["macd"]

        s0 = df.iloc[-1]["macd_signal"]
        s1 = df.iloc[-2]["macd_signal"]

        h0 = df.iloc[-1]["macd_hist"]
        h1 = df.iloc[-2]["macd_hist"]
        slope_up = h0 > h1
        slope_down = h0 < h1

        if atr is not None and atr > 0:
            if (pred == "bullish") or (pred == "bearish"):
                response = {}
                profit_factor = self.atr_mult
                loss_factor = self.atr_stop

                take_profit = close_price + (profit_factor * atr) if pred == "bullish" else close_price - (
                        profit_factor * atr)
                stop_loss = close_price - (loss_factor * atr) if pred == "bullish" else close_price + (
                        loss_factor * atr)

                if pred == "bullish" and close_price < ema200 and proba < 0.8:
                    # if close_price < ema200 and not (m0 > m1 and h0 > 0 and s0 > s1):
                    #     profit_factor = 3
                    take_profit = close_price + (profit_factor * atr)
                    stop_loss = close_price - (loss_factor * atr)

                if pred == "bearish" and close_price > ema200 and proba < 0.8:
                    # if close_price > ema200 and not (m0 < m1 and h1 < 0 and s0 < s1):
                    #     profit_factor = 3
                    take_profit = close_price - (profit_factor * atr)
                    stop_loss = close_price + (loss_factor * atr)

                # --- CÁLCULO DEL FRAGMENTO DE KELLY ---
                ganancia = abs(take_profit - close_price)
                perdida = abs(stop_loss - close_price)
                reward_risk = ganancia / perdida if perdida != 0 else 0.0001  # protección para no dividir por 0
                # p = proba
                # p = max(0.74, proba - 0.3)
                p = 0.6
                q = 1 - p
                kelly_fraction = p - (q / reward_risk)
                # Limitar valores extremos
                kelly_fraction = max(0.0, min(kelly_fraction, 0.95))

                # Construir el payload (para logs y consistencia)
                payload = {
                    "action": "buy" if pred == "bullish" else "sell",
                    "take_profit": f"{take_profit:.10f}",
                    "stop_loss": f"{stop_loss:.10f}",
                    "price": f"{close_price:.3f}",
                    "symbol": symbol,
                    "kelly_fraction": round(kelly_fraction, 4)
                }

                side = "long" if pred == "bullish" else "short"
                action_str = "buy" if pred == "bullish" else "sell"

                # Tamaño/qty: la calculas tú; si ya la tienes en el webhook de respuesta, igual logueamos el attempt.
                qty_preview = None  # pon aquí tu cálculo si lo tienes (o déjalo None)
                self.tradelogger.log_order_attempt(
                    trade_id=trade_id, symbol=symbol, side=side, qty=qty_preview, payload=payload
                )

                # --- Lógica DIRECTA con PyBit (Strict Copy from main_tesnet.py) ---
                session_local = self.session if real_account else self.session_demo
                # session_local = self.session_demo

                leverage = 15
                success = False
                response_data = {}

                try:
                    usdt_balance = self.get_usdt_balance(session_local)
                    print(f"Balance USDT ({'REAL' if real_account else 'DEMO'}): {usdt_balance}")

                    available_balance = usdt_balance * 0.95
                    order_qty = 0.0
                    metodo = ""
                    
                    # 1. Calcular Qty usando Kelly Fraction (si aplica)
                    if kelly_fraction > 0:
                        order_qty = available_balance * kelly_fraction
                        metodo = "Kelly logic"
                        print(f"Usando fracción de Kelly: {kelly_fraction:.4f}")

                        if order_qty > available_balance:
                            order_qty = available_balance
                            print(f"Orden ajustada al balance disponible: {order_qty:.2f} USDT")

                    qty = self.calculate_qty(order_qty, close_price, leverage, symbol)
                    print(f"Cantidad de la orden: {order_qty:.2f} USDT")
                    print(f"Cantidad en {symbol}: {qty}")

                    # 2. Verificar Posiciones Abiertas
                    positions = session_local.get_open_orders(
                        category="linear",
                        symbol=symbol,
                        openOnly=0,
                        limit=1,
                    )
                    
                    pos_list = positions['result']['list']
                    print(f"Posiciones abiertas: {len(pos_list)}")
                    
                    place_new_order = False
                    
                    if len(pos_list) != 0:
                        new_position_side = pos_list[0]['side'] # "Buy" or "Sell"
                        
                        # Strict logic from main_tesnet.py:
                        target_side_cap = action_str.capitalize()
                        
                        if new_position_side == target_side_cap:
                            # Cerrar usando la 'action' (que es igual al side de la posición actual)
                            self.close_order(action_str, real_account=real_account)
                            place_new_order = True
                        else:
                             print("Cambio de posición no permitido (Strict Logic).")
                             response_data = {
                                 "success": False,
                                 "error": "Cambio de posición no permitido",
                                 "bybit_response": "Cambio de posición no permitido",
                                 "qty": qty
                             }
                             success = False
                    else:
                        # No hay posiciones, abrir directo
                        place_new_order = True

                    if place_new_order:
                        bybit_side = "Buy" if pred == "bullish" else "Sell"
                        bybit_resp = session_local.place_order(
                            category="linear",
                            symbol=symbol,
                            side=bybit_side,
                            orderType="Market",
                            qty=qty,
                            leverage=leverage,
                            reduceOnly=False,
                            closeOnTrigger=False,
                            timeInForce="GoodTillCancel",
                            takeProfit=payload['take_profit'],
                            stopLoss=payload['stop_loss'],
                        )
                        success = True
                        response_data = {
                            "success": True,
                            "bybit_response": {"result": bybit_resp['result']},
                            "qty": qty,
                            "order_amount_usdt": order_qty
                        }
                        print(f"Orden ejecutada en PyBit: {bybit_side} {qty} {symbol}")

                except Exception as e:
                    print(f"Error executing order in PyBit: {e}")
                    success = False
                    response_data = {"success": False, "error": str(e)}

                # --- Fin lógica directa ---

                # Adaptación para logs y email
                if success:
                    bybit_resp = response_data.get('bybit_response', {})
                    order_id = None
                    try:
                         # Ajuste según estructura de respuesta
                         if isinstance(bybit_resp, dict) and 'result' in bybit_resp:
                             order_id = bybit_resp['result']['orderId']
                             print(f"bybit_resp {bybit_resp}")
                             print(f'order id {order_id}')
                    except Exception:
                         pass

                    qty_final = response_data.get("qty")

                    self.tradelogger.log_order_ack(
                        trade_id=trade_id, symbol=symbol, side=side, qty=qty_final, broker_raw=response_data,
                        order_id=order_id
                    )
                    self.tradelogger.enrich_open_position(
                        symbol,
                        entry_price=close_price,
                        take_profit=take_profit,
                        stop_loss=stop_loss,
                        side=side,
                        fee_rate=getattr(self, "fee_rate", 0.00055),
                    )
                    subject = f"ALERTA TRADING {'REAL' if real_account else 'DEMO'}: Tendencia {pred.upper()} >{self.umbral_soft * 100:.2f}%"
                    body = f"""
                    Predicción automática:
                    Vela: {bogota_time.strftime('%Y-%m-%d %H:%M:%S')}
                    Tendencia: {pred.upper()}
                    Probabilidad: {proba * 100:.2f}%
                    Reward/Risk: {reward_risk:.2f}
                    Fragmento Kelly: {kelly_fraction:.4f}
                    Detalle probabilidades: {resultado}
                    Json: {payload}
                    """
                    self.send_email(subject, body)
                else:
                    self.tradelogger.log_order_reject(
                        trade_id=trade_id, symbol=symbol, side=side, qty=response_data.get("qty"),
                        broker_raw=response_data
                    )



    def close_order(self, side, real_account=False):
        """
        Cierra posición enviando orden reduceOnly.
        Strict Copy: Usa 'side' (action) y lo capitaliza para el parametro 'side' de place_order.

        :param side: 'buy' o 'sell' (string lower).
        """
        session_local = self.session if real_account else self.session_demo
        symbol = "AVAXUSDT"

        # main_tesnet logic: side = data['action'].capitalize()
        # if action is 'buy', side='Buy'.
        close_side = side.capitalize()

        try:
            print(f"Closing position ({'REAL' if real_account else 'DEMO'}). Sending {close_side} reduceOnly.")

            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": close_side.capitalize(),
                "orderType": "Market",
                "qty": "0",
                "timeInForce": "GoodTillCancel",
                "reduceOnly": True,
                "closeOnTrigger": True
            }

            response = session_local.place_order(**order_params)
            print("Orden de cierre enviada:", response)

        except Exception as e:
            print("Error enviando orden de cierre:", e)

    def append_new_data(self, new_df):
        """
        Anexa nuevos datos OHLCV al histórico acumulado (self.data), evitando duplicados.
        """
        if new_df is None or new_df.empty:
            return
        if self.data is None or self.data.empty:
            self.data = new_df
        else:
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            self.data = self.data.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    def simular_estrategia(self, start_date="2024-07-01", end_date="2025-07-20",
                           capital_inicial=1000, fee=0.0012, leverage=19,
                           verbose=True, graficar=True):
        """
        Simula la estrategia entre dos fechas con capital inicial, Kelly y apalancamiento.
        Entrena el modelo y scaler una sola vez con la data previa a start_date,
        y predice la probabilidad para cada trade con ese modelo.
        SOLO UN TRADE ABIERTO A LA VEZ.
        Exporta resultados y resumen a CSV.
        """
        df = self.data.copy()
        features = [
            'k100', 'd100', 'e100', 'k', 'd', 'e', 'volatility', 'ATR',
            'RSI_14', 'ADX_14', 'pct_green_last_n', 'pct_red_last_n',
            'avg_volume_last_n', 'pct_volume_change'
        ]

        # 1. Entrenar modelo y scaler solo con datos ANTES de start_date
        train_df = df[df['datetime'] < start_date].dropna(subset=features + ['future_trend'])
        if train_df.empty:
            raise Exception("No hay suficiente data previa al periodo de simulación para entrenar el modelo.")

        X_train = train_df[features]
        y_train = train_df['future_trend']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        model.fit(X_train_scaled, y_train)

        # 2. Filtrar solo el periodo a simular
        df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].reset_index(drop=True)

        capital = capital_inicial
        equity_curve = [capital]
        max_equity = capital
        trades = []

        i = 0
        while i < len(df):
            row = df.iloc[i]
            if row['future_trend'] not in ('bullish', 'bearish'):
                i += 1
                continue
            price = row['close']
            atr = row['ATR']
            if pd.isna(atr) or atr <= 0:
                i += 1
                continue

            # Calcular probabilidad real del modelo en ese momento usando SOLO datos previos
            X_test = row[features].values.reshape(1, -1)
            X_test_scaled = scaler.transform(X_test)
            probs = model.predict_proba(X_test_scaled)[0]
            clases = model.classes_

            pred_class = model.predict(X_test_scaled)[0]  # puede ser 'bullish', 'bearish', 'lateral'
            pred_probas = dict(zip(clases, probs))
            prob = pred_probas.get(pred_class, 0.5)

            if pred_class not in ('bullish', 'bearish'):
                i += 1
                continue

            # Kelly
            reward = 3 * atr
            risk = 2.8 * atr
            rrr = reward / risk if risk > 0 else 1
            q = 1 - prob
            kelly = prob - (q / rrr)
            kelly = max(0, min(kelly, 0.95))
            if kelly < 0.01:
                i += 1
                continue

            pos_size = capital * kelly * leverage
            commission = abs(pos_size * fee)

            # TP/SL
            if pred_class == 'bullish':
                tp = price + reward
                sl = price - risk
            else:
                tp = price - reward
                sl = price + risk

            # Buscar hacia adelante hasta que se alcance TP o SL (sin límite)
            df_future = df.iloc[i + 1:]
            hit_tp = hit_sl = None
            idx_tp = idx_sl = None
            salida_idx = None
            for j, fut in df_future.iterrows():
                if pred_class == "bullish":
                    if fut['high'] >= tp:
                        hit_tp = fut['datetime']
                        idx_tp = j
                        salida_idx = i + 1 + j
                        break
                    if fut['low'] <= sl:
                        hit_sl = fut['datetime']
                        idx_sl = j
                        salida_idx = i + 1 + j
                        break
                else:
                    if fut['low'] <= tp:
                        hit_tp = fut['datetime']
                        idx_tp = j
                        salida_idx = i + 1 + j
                        break
                    if fut['high'] >= sl:
                        hit_sl = fut['datetime']
                        idx_sl = j
                        salida_idx = i + 1 + j
                        break

            # Resultado y cierre
            if (idx_tp is not None) and (idx_sl is not None):
                if idx_tp < idx_sl:
                    pnl = pos_size * (reward / price) - commission
                    resultado = "TP"
                    salida_fecha = hit_tp
                    salida_idx = i + 1 + idx_tp
                else:
                    pnl = -pos_size * (risk / price) - commission
                    resultado = "SL"
                    salida_fecha = hit_sl
                    salida_idx = i + 1 + idx_sl
            elif idx_tp is not None:
                pnl = pos_size * (reward / price) - commission
                resultado = "TP"
                salida_fecha = hit_tp
                salida_idx = i + 1 + idx_tp
            elif idx_sl is not None:
                pnl = -pos_size * (risk / price) - commission
                resultado = "SL"
                salida_fecha = hit_sl
                salida_idx = i + 1 + idx_sl
            else:
                if not df_future.empty:
                    last_close = df_future['close'].iloc[-1]
                    salida_fecha = df_future['datetime'].iloc[-1]
                    salida_idx = len(df)  # no hay más filas
                else:
                    last_close = price
                    salida_fecha = row['datetime']
                    salida_idx = i + 1
                if pred_class == 'bullish':
                    pnl = pos_size * ((last_close - price) / price) - commission
                else:
                    pnl = pos_size * ((price - last_close) / price) - commission
                resultado = "No Exit"

            capital += pnl
            max_equity = max(max_equity, capital)
            drawdown = (max_equity - capital) / max_equity

            trade = {
                "fecha_entrada": row['datetime'],
                "fecha_salida": salida_fecha,
                "pred": pred_class,
                "entry": price,
                "tp": tp,
                "sl": sl,
                "probabilidad": prob,
                "kelly_fragment": kelly,
                "pos_size": pos_size,
                "comision": commission,
                "resultado": resultado,
                "pnl": pnl,
                "capital_post": capital,
                "drawdown": drawdown
            }
            trades.append(trade)
            equity_curve.append(capital)

            # Saltar a la siguiente fila DESPUÉS del cierre del trade
            if salida_idx is None:
                i += 1
            else:
                i = salida_idx + 1

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            print("No hay trades en ese rango.")
            return

        # === ESTADÍSTICAS ===
        total_trades = len(trades_df)
        ganadores = (trades_df['pnl'] > 0).sum()
        perdedores = (trades_df['pnl'] < 0).sum()
        profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(
            trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if perdedores else np.nan
        max_drawdown = trades_df['drawdown'].max()
        final_capital = trades_df['capital_post'].iloc[-1]
        total_profit = final_capital - capital_inicial
        win_rate = ganadores / total_trades * 100
        avg_prob = trades_df['probabilidad'].mean()
        avg_kelly = trades_df['kelly_fragment'].mean()

        resumen = {
            "Capital inicial": capital_inicial,
            "Capital final": round(final_capital, 2),
            "Total profit/loss": round(total_profit, 2),
            "Trades": total_trades,
            "Ganadores": ganadores,
            "Perdedores": perdedores,
            "Win rate %": round(win_rate, 2),
            "Profit factor": round(profit_factor, 2),
            "Max drawdown %": round(100 * max_drawdown, 2),
            "Probabilidad promedio": round(avg_prob, 4),
            "Kelly promedio": round(avg_kelly, 4)
        }

        if verbose:
            print("\n--- RESUMEN SIMULACIÓN ---")
            for k, v in resumen.items():
                print(f"{k}: {v}")

            print(trades_df[
                      ['fecha_entrada', 'fecha_salida', 'pred', 'entry', 'tp', 'sl', 'probabilidad', 'kelly_fragment',
                       'pos_size', 'comision', 'resultado', 'pnl', 'capital_post', 'drawdown']].tail(10))

        # === GUARDAR CSVs ===

        bogota_tz = pytz.timezone('America/Bogota')
        for col in ['fecha_entrada', 'fecha_salida']:
            trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce', utc=True).dt.tz_convert(bogota_tz)
            trades_df[col] = trades_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        trades_filename = f"sim_trades_{start_date}_a_{end_date}.csv"
        resumen_filename = f"sim_resumen_{start_date}_a_{end_date}.csv"

        trades_df.to_csv(trades_filename, index=False)
        if verbose:
            print(f"\nArchivo de operaciones exportado a: {trades_filename}")

        resumen_df = pd.DataFrame([resumen])
        resumen_df.to_csv(resumen_filename, index=False)
        if verbose:
            print(f"Archivo de resumen exportado a: {resumen_filename}")

        # === GRÁFICOS ===
        if graficar:
            plt.figure(figsize=(14, 6))
            plt.subplot(2, 2, 1)
            plt.plot(equity_curve, label="Equity Curve")
            plt.title("Evolución de Capital")
            plt.xlabel("Trade #")
            plt.ylabel("Capital (USDT)")
            plt.grid()
            plt.legend()

            plt.subplot(2, 2, 2)
            plt.hist(trades_df['pnl'], bins=30, color='royalblue', alpha=0.7)
            plt.title("Distribución PnL por Trade")
            plt.xlabel("Profit/Loss")
            plt.ylabel("Frecuencia")

            plt.subplot(2, 2, 3)
            plt.plot(trades_df['drawdown'], label="Drawdown")
            plt.title("Drawdown")
            plt.xlabel("Trade #")
            plt.ylabel("Drawdown (%)")
            plt.grid()
            plt.legend()

            plt.subplot(2, 2, 4)
            trades_df['resultado'].value_counts().plot(kind='bar')
            plt.title("Tipos de Salidas (TP/SL/No Exit)")

            plt.tight_layout()
            plt.show(block=False)

        # resumen, trades_df = self._summarize_backtest(trades_df,
        #                                         equity_curve=equity_df['equity'] if equity_df is not None else None)
        #
        # print("\n=== Resumen Backtest ===")
        # for k, v in resumen.items():
        #     print(f"{k:>16}: {v}")

        return trades_df, resumen


    # def preparar_features(self, df, lags=[1]):
    #     """
    #     Feature set 'trader-like':
    #       - Precio y lags básicos
    #       - Volumen y volatilidad (ATR/volatility, atr_norm)
    #       - Soporte/Resistencia (distancias continuas)
    #       - Tiempo/Sesiones (sin/cos + flags de sesiones)
    #       - Tendencia (EMA ratio + líneas de tendencia normalizadas + distancia)
    #       - Momentum (RSI/ROC/momentum_diff)
    #       - Patrones de velas (doji, pinbar, engulfing)
    #       - Canales de tendencia (slopes high/low + ancho)
    #       - Confluencias (RSI + cercanía a SR)
    #     Retorna: (df, features_final)
    #     """
    #     dfx = df.copy()
    #
    #     eps = 1e-9
    #
    #     # =========================
    #     # CONFIG (ajusta a tu mercado)
    #     # =========================
    #     # Ventanas para tendencia/canales
    #     trend_windows = [100]
    #     channel_window = 50
    #
    #     # Sesiones (UTC) — AJUSTA si tu dataset no está en UTC
    #     # NY Open aproximada (13:30–15:30 UTC), London (07:00–09:00), Asia/Tokyo (23:00–01:00)
    #     session_defs = {
    #         "session_ny_open": [(8, 30), (10, 30)],  # 08:30–10:30 hora local NY
    #         "session_london_open": [(2, 0), (4, 0)],  # 02:00–04:00 hora local NY
    #         "session_asia_open": [(18, 0), (20, 0)]  # 18:00–20:00 hora local NY
    #     }
    #
    #     # Confluencias (umbrales)
    #     rsi_oversold = 30
    #     rsi_overbought = 70
    #     sr_threshold_frac = 0.25  # "cerca de SR" = distancia < 0.25*ATR (ejemplo)
    #
    #     # =========================
    #     # Utilidades
    #     # =========================
    #     def _ensure_datetime(d):
    #         if 'datetime' in d.columns and not pd.api.types.is_datetime64_any_dtype(d['datetime']):
    #             d['datetime'] = pd.to_datetime(d['datetime'], utc=True, errors='coerce')
    #         return d
    #
    #     def _safe_pct_change(s, periods=1):
    #         prev = s.shift(periods)
    #         return (s - prev) / (np.abs(prev) + eps)
    #
    #     def _rsi(series, window=14):
    #         delta = series.diff()
    #         up = delta.clip(lower=0)
    #         down = -delta.clip(upper=0)
    #         ma_up = up.rolling(window).mean()
    #         ma_down = down.rolling(window).mean()
    #         rs = ma_up / (ma_down + eps)
    #         return 100 - (100 / (1 + rs))
    #
    #     def _trend_slope_norm(arr):
    #         y = arr.reshape(-1, 1)
    #         X = np.arange(len(arr), dtype=float).reshape(-1, 1)
    #         reg = LinearRegression().fit(X, y)
    #         slope = float(reg.coef_[0][0])
    #         mean_price = float(np.mean(arr)) if np.isfinite(np.mean(arr)) else np.nan
    #         return slope / (mean_price + eps)
    #
    #     def _trend_r2(arr):
    #         y = arr.reshape(-1, 1)
    #         X = np.arange(len(arr), dtype=float).reshape(-1, 1)
    #         reg = LinearRegression().fit(X, y)
    #         return float(reg.score(X, y))
    #
    #     def _trend_dist_pct(arr):
    #         y = arr.reshape(-1, 1)
    #         X = np.arange(len(arr), dtype=float).reshape(-1, 1)
    #         reg = LinearRegression().fit(X, y)
    #         y_hat_last = float(reg.predict(np.array([[len(arr) - 1]], dtype=float)))
    #         y_last = float(arr[-1])
    #         return (y_last - y_hat_last) / (abs(y_last) + eps)
    #
    #     def _linreg_line(arr):
    #         """Devuelve (coef, intercept) de y ~ a*x + b sobre array 1-D."""
    #         y = arr.reshape(-1, 1)
    #         X = np.arange(len(arr), dtype=float).reshape(-1, 1)
    #         reg = LinearRegression().fit(X, y)
    #         return float(reg.coef_[0][0]), float(reg.intercept_)
    #
    #     dfx = _ensure_datetime(dfx)
    #
    #     # =========================
    #     # Precio / Volumen / Volatilidad / ATR básicos (si faltan)
    #     # =========================
    #     # Asumimos close/high/low/volume existen en df fuente. Si no, fallará arriba en tu pipeline.
    #     # volatility simple (si no existe)
    #     if 'volatility' not in dfx.columns:
    #         dfx['volatility'] = dfx['close'].pct_change().abs().rolling(20).mean()
    #
    #     # ATR clásico (si no existe)
    #     if 'ATR' not in dfx.columns:
    #         tr = np.maximum(dfx['high'] - dfx['low'],
    #                         np.maximum(np.abs(dfx['high'] - dfx['close'].shift(1)),
    #                                    np.abs(dfx['low'] - dfx['close'].shift(1))))
    #         dfx['ATR'] = tr.rolling(14).mean()
    #
    #     if 'atr_norm' not in dfx.columns:
    #         dfx['atr_norm'] = dfx['ATR'] / (np.abs(dfx['close']) + eps)
    #
    #     # Medias de volatilidad
    #     if 'volatility_10' not in dfx.columns:
    #         dfx['volatility_10'] = dfx['close'].pct_change().rolling(10).std()
    #     if 'volatility_50' not in dfx.columns:
    #         dfx['volatility_50'] = dfx['close'].pct_change().rolling(50).std()
    #
    #     # Volume promedio (si no existe)
    #     if 'avg_volume_last_n' not in dfx.columns:
    #         dfx['avg_volume_last_n'] = dfx['volume'].rolling(20).mean()
    #
    #     # RSI, ROC, momentum si faltan
    #     if 'rsi_14' not in dfx.columns:
    #         dfx['rsi_14'] = _rsi(dfx['close'], 14)
    #     if 'rsi_50' not in dfx.columns:
    #         dfx['rsi_50'] = _rsi(dfx['close'], 50)
    #     if 'roc_10' not in dfx.columns:
    #         dfx['roc_10'] = _safe_pct_change(dfx['close'], 10)
    #     if 'roc_30' not in dfx.columns:
    #         dfx['roc_30'] = _safe_pct_change(dfx['close'], 30)
    #     if 'momentum_diff' not in dfx.columns:
    #         dfx['momentum_diff'] = dfx['roc_10'] - dfx['roc_30']
    #
    #     # EMA ratio (si no existe)
    #     if 'ema_ratio' not in dfx.columns:
    #         ema_10 = dfx['close'].ewm(span=10, adjust=False).mean()
    #         ema_50 = dfx['close'].ewm(span=50, adjust=False).mean()
    #         dfx['ema_ratio'] = ema_10 / (ema_50 + eps)
    #
    #     # =========================
    #     # Tiempo / Ciclos y Sesiones
    #     # =========================
    #     if 'datetime' in dfx.columns:
    #         # sin/cos intradía
    #         minutes = dfx['datetime'].dt.hour * 60 + dfx['datetime'].dt.minute
    #         dfx['minute_sin'] = np.sin(2 * np.pi * minutes / 1440.0)
    #         dfx['minute_cos'] = np.cos(2 * np.pi * minutes / 1440.0)
    #         # sin/cos día de la semana (0–6)
    #         dow = dfx['datetime'].dt.dayofweek
    #         dfx['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    #         dfx['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    #
    #         # Flags de sesiones
    #         # Nota: si una sesión cruza medianoche, tratamos ese caso especial
    #         def _in_interval(h, m, start, end):
    #             sh, sm = start;
    #             eh, em = end
    #             start_min = sh * 60 + sm
    #             end_min = eh * 60 + em
    #             t = h * 60 + m
    #             if start_min <= end_min:
    #                 return (t >= start_min) & (t < end_min)
    #             else:
    #                 # cruza medianoche
    #                 return (t >= start_min) | (t < end_min)
    #
    #         hh = dfx['datetime'].dt.hour
    #         mm = dfx['datetime'].dt.minute
    #         for flag, (start, end) in session_defs.items():
    #             dfx[flag] = _in_interval(hh, mm, start, end).astype(int)
    #     else:
    #         # Si no hay datetime, crea columnas nulas para mantener el contrato
    #         dfx['minute_sin'] = 0.0;
    #         dfx['minute_cos'] = 0.0
    #         dfx['dow_sin'] = 0.0;
    #         dfx['dow_cos'] = 0.0
    #         for flag in ["session_ny_open", "session_london_open", "session_asia_open"]:
    #             dfx[flag] = 0
    #
    #     # =========================
    #     # Soportes / Resistencias (se asume ya están en df):
    #     # dist_PP, dist_S1, dist_R1, dist_S2, dist_R2, dist_resist_rl, dist_support_rl
    #     # más dist_*_8h_atr: distancias normalizadas por ATR (ya en tu dataset)
    #     # =========================
    #     dfx = add_sr_features_8h(dfx)  # ← NUEVO (8h)
    #     dfx = add_stoch_extreme_features(dfx)
    #     # =========================
    #     # Líneas de tendencia (close)
    #     # =========================
    #     for w in trend_windows:
    #         dfx[f"trend_slope_norm_{w}"] = (
    #             dfx["close"].rolling(window=w, min_periods=w)
    #             .apply(lambda a: _trend_slope_norm(a), raw=True)
    #         )
    #         dfx[f"trend_r2_{w}"] = (
    #             dfx["close"].rolling(window=w, min_periods=w)
    #             .apply(lambda a: _trend_r2(a), raw=True)
    #         )
    #         dfx[f"trend_dist_pct_{w}"] = (
    #             dfx["close"].rolling(window=w, min_periods=w)
    #             .apply(lambda a: _trend_dist_pct(a), raw=True)
    #         )
    #
    #     # =========================
    #     # Canales de tendencia (regresión sobre highs/lows en ventana)
    #     # =========================
    #     def _channel_feats(high_series, low_series, w):
    #         # slope_high, slope_low y ancho (línea_high - línea_low) en el último punto
    #         hs = high_series.rolling(window=w, min_periods=w).apply(
    #             lambda a: _linreg_line(a)[0], raw=True)
    #         ls = low_series.rolling(window=w, min_periods=w).apply(
    #             lambda a: _linreg_line(a)[0], raw=True)
    #
    #         # reconstruir predicciones al último punto para estimar ancho del canal
    #         def _last_line_val(s, w):
    #             # devuelve y_hat en x=w-1
    #             return s.rolling(window=w, min_periods=w).apply(
    #                 lambda a: _linreg_line(a)[0] * (w - 1) + _linreg_line(a)[1], raw=True)
    #
    #         high_last = _last_line_val(high_series, w)
    #         low_last = _last_line_val(low_series, w)
    #         width = (high_last - low_last) / (np.abs(dfx['close']) + eps)
    #
    #         return hs, ls, width
    #
    #     ch_w = channel_window
    #     ch_high_slope, ch_low_slope, ch_width = _channel_feats(dfx['high'], dfx['low'], ch_w)
    #     dfx['trend_channel_slope_high'] = ch_high_slope
    #     dfx['trend_channel_slope_low'] = ch_low_slope
    #     dfx['trend_channel_width'] = ch_width
    #
    #     # =========================
    #     # Patrones de velas (simples y reproducibles)
    #     # =========================
    #     body = (dfx['close'] - dfx['open']).abs()
    #     range_ = (dfx['high'] - dfx['low']).replace(0, eps)
    #     upper_wick = dfx['high'] - dfx[['close', 'open']].max(axis=1)
    #     lower_wick = dfx[['close', 'open']].min(axis=1) - dfx['low']
    #
    #     # Doji: cuerpo pequeño relativo al rango
    #     dfx['candle_doji'] = (body / range_ < 0.1).astype(int)
    #
    #     # Pin bar: mecha larga en un lado
    #     dfx['candle_pinbar'] = (
    #             ((upper_wick / range_) > 0.6) | ((lower_wick / range_) > 0.6))
    #     dfx['candle_pinbar'] = dfx['candle_pinbar'].astype(int)
    #
    #     # Engulfing: cuerpo actual envuelve al anterior (simplificado)
    #     prev_open = dfx['open'].shift(1);
    #     prev_close = dfx['close'].shift(1)
    #     bull_engulf = (dfx['close'] > dfx['open']) & (prev_close < prev_open) & \
    #                   (dfx['close'] >= prev_open) & (dfx['open'] <= prev_close)
    #     bear_engulf = (dfx['close'] < dfx['open']) & (prev_close > prev_open) & \
    #                   (dfx['open'] >= prev_close) & (dfx['close'] <= prev_open)
    #     dfx['candle_engulfing'] = (bull_engulf | bear_engulf).astype(int)
    #
    #     # =========================
    #     # Confluencias (RSI + cercanía a SR)
    #     # =========================
    #     # "Cerca" = distancia < sr_threshold_frac * ATR
    #     near_S1 = (dfx['dist_S1'].abs() < sr_threshold_frac * (dfx['ATR'] + eps))
    #     near_R1 = (dfx['dist_R1'].abs() < sr_threshold_frac * (dfx['ATR'] + eps))
    #     dfx['rsi14_oversold_near_support'] = ((dfx['rsi_14'] < rsi_oversold) & near_S1).astype(int)
    #     dfx['rsi14_overbought_near_resist'] = ((dfx['rsi_14'] > rsi_overbought) & near_R1).astype(int)
    #
    #     # =========================
    #     # Dummies (si existen en tu dataset)
    #     # =========================
    #     # trend_ema200 / trend_vwap eran categóricas; conviértelas a dummies si están
    #     dummy_cols_base = ['trend_ema200']
    #     for col in dummy_cols_base:
    #         if col in dfx.columns and dfx[col].dtype == 'object':
    #             dfx = pd.get_dummies(dfx, columns=[col])
    #     # aseguremos flags esperados si quieres forzar columnas
    #     for col in ['trend_ema200_up']:
    #         if col not in dfx.columns:
    #             dfx[col] = 0
    #
    #     # =========================
    #     # Lags básicos de OHLC
    #     # =========================
    #     for lag in lags:
    #         dfx[f'close_lag{lag}'] = dfx['close'].shift(lag)
    #         dfx[f'high_lag{lag}'] = dfx['high'].shift(lag)
    #         dfx[f'low_lag{lag}'] = dfx['low'].shift(lag)
    #         dfx[f'e100_lag{lag}'] = dfx['e100'].shift(lag)
    #         dfx[f'volume_lag{lag}'] = dfx['e100'].shift(lag)
    #
    #
    #
    #     # =========================
    #     # New day features
    #     # =========================
    #     dfx = self.add_new_day_features(dfx)
    #
    #     # =========================
    #     # Lista final de features
    #     # =========================
    #
    #
    #     features_price = [
    #         'close', 'high', 'low', 'e100', 'volume'
    #     ]
    #     # === Generar lag features ===
    #     for col in features_price:
    #         for lag in lags:
    #             df[f'{col}_lag{lag}'] = df[col].shift(lag)
    #     lag_feature_names = [f'{col}_lag{lag}' for col in features_price for lag in lags]
    #
    #     features_vol = ['avg_volume_last_n', 'volatility', 'volatility_50', 'atr_norm']
    #     features_sr = ['dist_PP', 'dist_S1', 'dist_R1', 'dist_S2', 'dist_R2', 'dist_resist_rl', 'dist_support_rl',
    #                    'dist_P_8h_atr', 'dist_S1_8h_atr', 'dist_R1_8h_atr', 'dist_S2_8h_atr', 'dist_R2_8h_atr']
    #     features_time = ['minute_sin', 'minute_cos', 'dow_sin', 'dow_cos']
    #     features_trend = ['ema_ratio'] + \
    #                      [f"trend_slope_norm_{w}" for w in trend_windows] + \
    #                      [f"trend_r2_100"] + \
    #                      [f"trend_dist_pct_{w}" for w in trend_windows] + \
    #                      ['trend_channel_slope_high', 'trend_channel_slope_low', 'trend_channel_width']
    #     features_momentum = ['rsi_50', 'roc_30', 'k', 'd']
    #     features_candles = ['candle_engulfing']
    #     features_conflu = ['rsi14_oversold_near_support', 'rsi14_overbought_near_resist']
    #
    #     new_day_feats = ['range_prev_day_abs', 'range_prev_day_pct',
    #                      'overnight_range_abs', 'overnight_range_pct']
    #     if 'ATR' in dfx.columns:
    #         new_day_feats += ['range_prev_day_atr', 'overnight_range_atr']
    #     # Dummies si existen (no obligatorias)
    #     # optional_dummies = [c for c in ['trend_ema200_up'] if c in dfx.columns]
    #
    #     features_final = features_price + features_vol + features_sr + features_time + \
    #                      features_trend + new_day_feats + lag_feature_names + features_candles + features_momentum
    #
    #     # =========================
    #     # Limpieza robusta
    #     # =========================
    #     # Reemplaza inf/-inf por NaN en las columnas de interés y deja que tu pipeline downstream dropee NaNs
    #     dfx[features_final] = dfx[features_final].replace([np.inf, -np.inf], np.nan)
    #
    #     print("Datos preparados para el entrenamiento")
    #
    #     return dfx, features_final

    def preparar_features_trader(self, df, lags=[1]):
        """
        Feature set 'trader-like':
          - Precio y lags básicos
          - Volumen y volatilidad (ATR/volatility, atr_norm)
          - Soporte/Resistencia (distancias continuas)
          - Tiempo/Sesiones (sin/cos + flags de sesiones)
          - Tendencia (EMA ratio + líneas de tendencia normalizadas + distancia)
          - Momentum (RSI/ROC/momentum_diff)
          - Patrones de velas (doji, pinbar, engulfing)
          - Canales de tendencia (slopes high/low + ancho)
          - Confluencias (RSI + cercanía a SR)
        Retorna: (df, features_final)
        """
        dfx = df.copy()

        eps = 1e-9

        # =========================
        # CONFIG (ajusta a tu mercado)
        # =========================
        # Ventanas para tendencia/canales
        trend_windows = [100]
        channel_window = 50

        # Sesiones (UTC) — AJUSTA si tu dataset no está en UTC
        # NY Open aproximada (13:30–15:30 UTC), London (07:00–09:00), Asia/Tokyo (23:00–01:00)
        session_defs = {
            "session_ny_open": [(8, 30), (10, 30)],  # 08:30–10:30 hora local NY
            "session_london_open": [(2, 0), (4, 0)],  # 02:00–04:00 hora local NY
            "session_asia_open": [(18, 0), (20, 0)]  # 18:00–20:00 hora local NY
        }

        # Confluencias (umbrales)
        rsi_oversold = 30
        rsi_overbought = 70
        sr_threshold_frac = 0.25  # "cerca de SR" = distancia < 0.25*ATR (ejemplo)

        # =========================
        # Utilidades
        # =========================
        def _ensure_datetime(d):
            if 'datetime' in d.columns and not pd.api.types.is_datetime64_any_dtype(d['datetime']):
                d['datetime'] = pd.to_datetime(d['datetime'], utc=True, errors='coerce')
            return d

        def _safe_pct_change(s, periods=1):
            prev = s.shift(periods)
            return (s - prev) / (np.abs(prev) + eps)

        def _rsi(series, window=14):
            delta = series.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ma_up = up.rolling(window).mean()
            ma_down = down.rolling(window).mean()
            rs = ma_up / (ma_down + eps)
            return 100 - (100 / (1 + rs))

        def _trend_slope_norm(arr):
            y = arr.reshape(-1, 1)
            X = np.arange(len(arr), dtype=float).reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            slope = float(reg.coef_[0][0])
            mean_price = float(np.mean(arr)) if np.isfinite(np.mean(arr)) else np.nan
            return slope / (mean_price + eps)

        def _trend_r2(arr):
            y = arr.reshape(-1, 1)
            X = np.arange(len(arr), dtype=float).reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            return float(reg.score(X, y))

        def _trend_dist_pct(arr):
            y = arr.reshape(-1, 1)
            X = np.arange(len(arr), dtype=float).reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            y_hat_last = float(reg.predict(np.array([[len(arr) - 1]], dtype=float)))
            y_last = float(arr[-1])
            return (y_last - y_hat_last) / (abs(y_last) + eps)

        def _linreg_line(arr):
            """Devuelve (coef, intercept) de y ~ a*x + b sobre array 1-D."""
            y = arr.reshape(-1, 1)
            X = np.arange(len(arr), dtype=float).reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            return float(reg.coef_[0][0]), float(reg.intercept_)

        dfx = _ensure_datetime(dfx)

        # =========================
        # Precio / Volumen / Volatilidad / ATR básicos (si faltan)
        # =========================
        # Asumimos close/high/low/volume existen en df fuente. Si no, fallará arriba en tu pipeline.
        # volatility simple (si no existe)
        if 'volatility' not in dfx.columns:
            dfx['volatility'] = dfx['close'].pct_change().abs().rolling(20).mean()

        # ATR clásico (si no existe)
        if 'ATR' not in dfx.columns:
            tr = np.maximum(dfx['high'] - dfx['low'],
                            np.maximum(np.abs(dfx['high'] - dfx['close'].shift(1)),
                                       np.abs(dfx['low'] - dfx['close'].shift(1))))
            dfx['ATR'] = tr.rolling(14).mean()

        if 'atr_norm' not in dfx.columns:
            dfx['atr_norm'] = dfx['ATR'] / (np.abs(dfx['close']) + eps)

        # Medias de volatilidad
        if 'volatility_10' not in dfx.columns:
            dfx['volatility_10'] = dfx['close'].pct_change().rolling(10).std()
        if 'volatility_50' not in dfx.columns:
            dfx['volatility_50'] = dfx['close'].pct_change().rolling(50).std()

        # Volume promedio (si no existe)
        if 'avg_volume_last_n' not in dfx.columns:
            dfx['avg_volume_last_n'] = dfx['volume'].rolling(20).mean()

        # RSI, ROC, momentum si faltan
        if 'rsi_14' not in dfx.columns:
            dfx['rsi_14'] = _rsi(dfx['close'], 14)
        if 'rsi_50' not in dfx.columns:
            dfx['rsi_50'] = _rsi(dfx['close'], 50)
        if 'roc_10' not in dfx.columns:
            dfx['roc_10'] = _safe_pct_change(dfx['close'], 10)
        if 'roc_30' not in dfx.columns:
            dfx['roc_30'] = _safe_pct_change(dfx['close'], 30)
        if 'momentum_diff' not in dfx.columns:
            dfx['momentum_diff'] = dfx['roc_10'] - dfx['roc_30']

        # EMA ratio (si no existe)
        if 'ema_ratio' not in dfx.columns:
            ema_10 = dfx['close'].ewm(span=10, adjust=False).mean()
            ema_50 = dfx['close'].ewm(span=50, adjust=False).mean()
            dfx['ema_ratio'] = ema_10 / (ema_50 + eps)

        # =========================
        # Tiempo / Ciclos y Sesiones
        # =========================
        if 'datetime' in dfx.columns:
            # sin/cos intradía
            minutes = dfx['datetime'].dt.hour * 60 + dfx['datetime'].dt.minute
            dfx['minute_sin'] = np.sin(2 * np.pi * minutes / 1440.0)
            dfx['minute_cos'] = np.cos(2 * np.pi * minutes / 1440.0)
            # sin/cos día de la semana (0–6)
            dow = dfx['datetime'].dt.dayofweek
            dfx['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
            dfx['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)

            # sin/cos día del mes (1-31)
            dom = dfx['datetime'].dt.day
            dfx['dom_sin'] = np.sin(2 * np.pi * dom / 31.0)
            dfx['dom_cos'] = np.cos(2 * np.pi * dom / 31.0)

            # Flags de sesiones
            # Nota: si una sesión cruza medianoche, tratamos ese caso especial
            def _in_interval(h, m, start, end):
                sh, sm = start;
                eh, em = end
                start_min = sh * 60 + sm
                end_min = eh * 60 + em
                t = h * 60 + m
                if start_min <= end_min:
                    return (t >= start_min) & (t < end_min)
                else:
                    # cruza medianoche
                    return (t >= start_min) | (t < end_min)

            hh = dfx['datetime'].dt.hour
            mm = dfx['datetime'].dt.minute
            for flag, (start, end) in session_defs.items():
                dfx[flag] = _in_interval(hh, mm, start, end).astype(int)
        else:
            # Si no hay datetime, crea columnas nulas para mantener el contrato
            dfx['minute_sin'] = 0.0;
            dfx['minute_cos'] = 0.0
            dfx['dow_sin'] = 0.0;
            dfx['dow_cos'] = 0.0
            for flag in ["session_ny_open", "session_london_open", "session_asia_open"]:
                dfx[flag] = 0

        # =========================
        # Soportes / Resistencias (se asume ya están en df):
        # dist_PP, dist_S1, dist_R1, dist_S2, dist_R2, dist_resist_rl, dist_support_rl
        # más dist_*_8h_atr: distancias normalizadas por ATR (ya en tu dataset)
        # =========================
        dfx = add_sr_features_8h(dfx)  # ← NUEVO (8h)
        dfx = add_stoch_extreme_features(dfx)
        # =========================
        # Líneas de tendencia (close)
        # =========================
        for w in trend_windows:
            dfx[f"trend_slope_norm_{w}"] = (
                dfx["close"].rolling(window=w, min_periods=w)
                .apply(lambda a: _trend_slope_norm(a), raw=True)
            )
            dfx[f"trend_r2_{w}"] = (
                dfx["close"].rolling(window=w, min_periods=w)
                .apply(lambda a: _trend_r2(a), raw=True)
            )
            dfx[f"trend_dist_pct_{w}"] = (
                dfx["close"].rolling(window=w, min_periods=w)
                .apply(lambda a: _trend_dist_pct(a), raw=True)
            )

        # =========================
        # Canales de tendencia (regresión sobre highs/lows en ventana)
        # =========================
        def _channel_feats(high_series, low_series, w):
            # slope_high, slope_low y ancho (línea_high - línea_low) en el último punto
            hs = high_series.rolling(window=w, min_periods=w).apply(
                lambda a: _linreg_line(a)[0], raw=True)
            ls = low_series.rolling(window=w, min_periods=w).apply(
                lambda a: _linreg_line(a)[0], raw=True)

            # reconstruir predicciones al último punto para estimar ancho del canal
            def _last_line_val(s, w):
                # devuelve y_hat en x=w-1
                return s.rolling(window=w, min_periods=w).apply(
                    lambda a: _linreg_line(a)[0] * (w - 1) + _linreg_line(a)[1], raw=True)

            high_last = _last_line_val(high_series, w)
            low_last = _last_line_val(low_series, w)
            width = (high_last - low_last) / (np.abs(dfx['close']) + eps)

            return hs, ls, width

        ch_w = channel_window
        ch_high_slope, ch_low_slope, ch_width = _channel_feats(dfx['high'], dfx['low'], ch_w)
        dfx['trend_channel_slope_high'] = ch_high_slope
        dfx['trend_channel_slope_low'] = ch_low_slope
        dfx['trend_channel_width'] = ch_width

        # =========================
        # Patrones de velas (simples y reproducibles)
        # =========================
        body = (dfx['close'] - dfx['open']).abs()
        range_ = (dfx['high'] - dfx['low']).replace(0, eps)
        upper_wick = dfx['high'] - dfx[['close', 'open']].max(axis=1)
        lower_wick = dfx[['close', 'open']].min(axis=1) - dfx['low']

        # Doji: cuerpo pequeño relativo al rango
        dfx['candle_doji'] = (body / range_ < 0.1).astype(int)

        # Pin bar: mecha larga en un lado
        dfx['candle_pinbar'] = (
                ((upper_wick / range_) > 0.6) | ((lower_wick / range_) > 0.6))
        dfx['candle_pinbar'] = dfx['candle_pinbar'].astype(int)

        # Engulfing: cuerpo actual envuelve al anterior (simplificado)
        prev_open = dfx['open'].shift(1);
        prev_close = dfx['close'].shift(1)
        bull_engulf = (dfx['close'] > dfx['open']) & (prev_close < prev_open) & \
                      (dfx['close'] >= prev_open) & (dfx['open'] <= prev_close)
        bear_engulf = (dfx['close'] < dfx['open']) & (prev_close > prev_open) & \
                      (dfx['open'] >= prev_close) & (dfx['close'] <= prev_open)
        dfx['candle_engulfing'] = (bull_engulf | bear_engulf).astype(int)

        # =========================
        # Confluencias (RSI + cercanía a SR)
        # =========================
        # "Cerca" = distancia < sr_threshold_frac * ATR
        near_S1 = (dfx['dist_S1'].abs() < sr_threshold_frac * (dfx['ATR'] + eps))
        near_R1 = (dfx['dist_R1'].abs() < sr_threshold_frac * (dfx['ATR'] + eps))
        dfx['rsi14_oversold_near_support'] = ((dfx['rsi_14'] < rsi_oversold) & near_S1).astype(int)
        dfx['rsi14_overbought_near_resist'] = ((dfx['rsi_14'] > rsi_overbought) & near_R1).astype(int)

        # =========================
        # Dummies (si existen en tu dataset)
        # =========================
        # trend_ema200 / trend_vwap eran categóricas; conviértelas a dummies si están
        dummy_cols_base = ['trend_ema200']
        for col in dummy_cols_base:
            if col in dfx.columns and dfx[col].dtype == 'object':
                dfx = pd.get_dummies(dfx, columns=[col])
        # aseguremos flags esperados si quieres forzar columnas
        for col in ['trend_ema200_up']:
            if col not in dfx.columns:
                dfx[col] = 0

        # =========================
        # Lags básicos de OHLC
        # =========================
        for lag in lags:
            dfx[f'close_lag{lag}'] = dfx['close'].shift(lag)
            dfx[f'high_lag{lag}'] = dfx['high'].shift(lag)
            dfx[f'low_lag{lag}'] = dfx['low'].shift(lag)
            dfx[f'e100_lag{lag}'] = dfx['e100'].shift(lag)
            dfx[f'volume_lag{lag}'] = dfx['e100'].shift(lag)

        # =========================
        # New day features
        # =========================
        dfx = self.add_new_day_features(dfx)

        # =========================
        # Lista final de features
        # =========================

        features_price = [
            'close', 'high', 'low', 'e100', 'volume'
        ]
        # === Generar lag features ===
        for col in features_price:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        lag_feature_names = [f'{col}_lag{lag}' for col in features_price for lag in lags]

        features_vol = ['avg_volume_last_n', 'volatility', 'volatility_50', 'atr_norm']
        features_sr = ['dist_PP', 'dist_S1', 'dist_R1', 'dist_S2', 'dist_R2', 'dist_resist_rl', 'dist_support_rl',
                       'dist_P_8h_atr', 'dist_S1_8h_atr', 'dist_R1_8h_atr', 'dist_S2_8h_atr', 'dist_R2_8h_atr']
        features_time = ['minute_sin', 'minute_cos', 'dow_sin', 'dow_cos']
        features_trend = ['ema_ratio'] + \
                         [f"trend_slope_norm_{w}" for w in trend_windows] + \
                         [f"trend_r2_100"] + \
                         [f"trend_dist_pct_{w}" for w in trend_windows] + \
                         ['trend_channel_slope_high', 'trend_channel_slope_low', 'trend_channel_width']
        features_momentum = ['rsi_50', 'roc_30', 'k', 'd']
        features_candles = ['candle_engulfing']
        features_conflu = ['rsi14_oversold_near_support', 'rsi14_overbought_near_resist']

        new_day_feats = ['range_prev_day_abs', 'range_prev_day_pct',
                         'overnight_range_abs', 'overnight_range_pct']
        if 'ATR' in dfx.columns:
            new_day_feats += ['range_prev_day_atr', 'overnight_range_atr']
        # Dummies si existen (no obligatorias)
        # optional_dummies = [c for c in ['trend_ema200_up'] if c in dfx.columns]

        features_final = features_price + features_vol + features_sr + features_time + \
                         features_trend + new_day_feats + lag_feature_names + features_candles + features_momentum

        # =========================
        # Limpieza robusta
        # =========================
        # Reemplaza inf/-inf por NaN en las columnas de interés y deja que tu pipeline downstream dropee NaNs
        dfx[features_final] = dfx[features_final].replace([np.inf, -np.inf], np.nan)

        print("Datos preparados para el entrenamiento")

        return dfx, features_final

    def new_features_complementary(self, dfx):

        eps = 1e-9

        # === NEW: helpers para z-score rolling (contexto/regimen) ===
        def _zscore(s: pd.Series, win: int) -> pd.Series:
            m = s.rolling(win, min_periods=max(10, win // 5)).mean()
            sd = s.rolling(win, min_periods=max(10, win // 5)).std()
            return (s - m) / (sd + eps)

        # === NEW: VWAP (rolling simple y señales derivadas) ===
        if 'VWAP' not in dfx.columns:
            # VWAP clásico acumulado; si prefieres rolling, cambia a rolling( session/window )
            pv = (dfx['close'] * dfx['volume']).cumsum()
            vv = dfx['volume'].replace(0, np.nan).cumsum()
            dfx['VWAP'] = pv / (vv + eps)

        # Dirección de VWAP (drift) y distancia normalizada
        dfx['vwap_up'] = dfx['VWAP'].diff().gt(0).astype(int)
        dfx['dist_to_vwap'] = (dfx['close'] - dfx['VWAP']) / (dfx['ATR'] + eps)

        # === NEW: MACD resumido (señal de cruce) ===
        if 'macd_minus_signal' not in dfx.columns:
            ema_fast = dfx['close'].ewm(span=12, adjust=False).mean()
            ema_slow = dfx['close'].ewm(span=26, adjust=False).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            dfx['macd_minus_signal'] = macd - macd_signal

        # === NEW: régimen de volatilidad (ratio corto/largo) ===
        dfx['vol_ratio'] = dfx['volatility_10'] / (dfx['volatility_50'] + eps)

        # === NEW: z-scores de régimen ===
        dfx['z_atr_norm'] = _zscore(dfx['atr_norm'], win=200)
        dfx['z_volume'] = _zscore(dfx['volume'], win=50)

        # === NEW: “mínima distancia a nivel” normalizada por ATR ===
        # Usa las columnas ya presentes en tu dataset
        _levels = []
        for col in ['dist_R1', 'dist_R2', 'dist_S1', 'dist_S2', 'dist_PP']:
            if col in dfx.columns:
                _levels.append(dfx[col].abs())
        if _levels:
            min_abs_dist = pd.concat(_levels, axis=1).min(axis=1)
            dfx['min_dist_level_atr'] = min_abs_dist / (dfx['ATR'] + eps)
        else:
            dfx['min_dist_level_atr'] = np.nan

        # === NEW: dinámica de tendencia (delta del slope normalizado) ===
        if 'trend_slope_norm_100' in dfx.columns:
            dfx['delta_trend_slope_100'] = dfx['trend_slope_norm_100'] - dfx['trend_slope_norm_100'].shift(1)
        else:
            dfx['delta_trend_slope_100'] = np.nan

        # === NEW: banda neutral RSI (contexto de mean-reversion vs. trend) ===
        dfx['rsi_40_60'] = ((dfx['rsi_14'] >= 40) & (dfx['rsi_14'] <= 60)).astype(int)

        features = [
            'macd_minus_signal',
            'vol_ratio',
            'z_atr_norm',
            'min_dist_level_atr',
            'delta_trend_slope_100',
        ]

        return dfx, features

    def build_mathstat_features(
            self,
            df: pd.DataFrame,
            w_short: int = 12,
            w_mid: int = 24,
            w_long: int = 96,
            entropy_bins: int = 10
    ):
        """
        df: DataFrame con columnas ['datetime','open','high','low','close','volume']
        Retorna: (df_features, feature_names)
        """

        def _rolling_autocorr(x, lag=1):
            # x es un array 1D (ventana). Devuelve autocorr con retardo "lag".
            if len(x) <= lag:
                return np.nan
            x0 = x[:-lag]
            x1 = x[lag:]
            if x0.std(ddof=1) == 0 or x1.std(ddof=1) == 0:
                return 0.0
            return np.corrcoef(x0, x1)[0, 1]

        def _hurst_rs(x):
            # Estimador Hurst simple por R/S en una ventana (rápido y robusto)
            n = len(x)
            if n < 8:
                return np.nan
            y = np.array(x, dtype=float)
            y = y - y.mean()
            z = y.cumsum()
            R = z.max() - z.min()
            S = y.std(ddof=1)
            if S == 0:
                return 0.5
            RS = R / S
            # H ≈ log(R/S) / log(n)
            return np.log(max(RS, 1e-9)) / np.log(n)

        def _linreg_slope_mse(y):
            # y: serie (ventana). Devuelve pendiente y MSE de regresión lineal contra t=[0,1,2,...]
            n = len(y)
            if n < 3:
                return np.nan, np.nan
            t = np.arange(n, dtype=float)
            t_mean = t.mean()
            y_mean = y.mean()
            S_tt = np.sum((t - t_mean) ** 2)
            if S_tt == 0:
                return np.nan, np.nan
            beta1 = np.sum((t - t_mean) * (y - y_mean)) / S_tt
            beta0 = y_mean - beta1 * t_mean
            y_hat = beta0 + beta1 * t
            mse = np.mean((y - y_hat) ** 2)
            return beta1, mse

        def _shannon_entropy(vals, bins=10):
            # Entropía de Shannon de retornos en ventana, con histogramas
            if len(vals) < 5:
                return np.nan
            hist, _ = np.histogram(vals, bins=bins, density=True)
            p = hist[hist > 0]
            if p.size == 0:
                return 0.0
            return -np.sum(p * np.log(p))

        def _permutation_entropy(x, m=3, tau=1):
            """
            Entropía de permutación (Bandt & Pompe), normalizada a [0,1].
            - Usa logaritmos naturales.
            - Normaliza dividiendo por log(m!) -> estable con lgamma.
            """
            n = len(x)
            if n < (m - 1) * tau + 1:
                return np.nan

            # construir patrones
            patterns = {}
            for i in range(n - (m - 1) * tau):
                window = x[i:(i + (m - 1) * tau + 1):tau]
                key = tuple(np.argsort(window))
                patterns[key] = patterns.get(key, 0) + 1

            counts = np.array(list(patterns.values()), dtype=float)
            p = counts / counts.sum()

            # evitar log(0)
            p = p[p > 0]
            if p.size == 0:
                return 0.0

            # denom = ln(m!) de forma estable
            denom = math.lgamma(m + 1)  # ln(m!) == lgamma(m+1)
            if denom == 0:
                return np.nan

            return -np.sum(p * np.log(p)) / denom

        def _rogers_satchell(o, h, l, c):
            # Medida intra-día de Rogers–Satchell en una vela
            return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

        def _garman_klass(o, h, l, c):
            # Varianza GK por vela
            return 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2

        def _parkinson(h, l):
            # Varianza Parkinson por vela
            return (np.log(h / l)) ** 2 / (4 * np.log(2))

        df_math = df.copy()
        # Asegurar orden temporal y tipos
        if 'datetime' in df_math.columns:
            df_math = df_math.sort_values('datetime')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df_math.columns:
                raise ValueError(f"Falta columna requerida: {col}")

        # Retornos
        df_math['ret'] = np.log(df_math['close']).diff()
        df_math['ret_abs'] = df_math['ret'].abs()

        # ATR simple (true range) para normalizaciones
        tr1 = (df_math['high'] - df_math['low']).abs()
        tr2 = (df_math['high'] - df_math['close'].shift()).abs()
        tr3 = (df_math['low'] - df_math['close'].shift()).abs()
        df_math['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df_math['atr_mid'] = df_math['tr'].rolling(w_mid).mean()

        # 1) Retornos y transformaciones
        df_math['ret_norm_atr'] = df_math['ret'] / (df_math['atr_mid'] / df_math['close'].shift())
        df_math['ret_diff2'] = df_math['close'].diff() - df_math['close'].diff().shift()

        # 2) Estadística descriptiva (sobre retornos)
        df_math['ret_mean_s'] = df_math['ret'].rolling(w_short).mean()
        df_math['ret_std_s'] = df_math['ret'].rolling(w_short).std()
        df_math['ret_mean_m'] = df_math['ret'].rolling(w_mid).mean()
        df_math['ret_std_m'] = df_math['ret'].rolling(w_mid).std()

        # skew/kurt (fallback si no está scipy)
        try:
            from scipy.stats import skew, kurtosis
            df_math['ret_skew_m'] = df_math['ret'].rolling(w_mid).apply(lambda x: skew(x, bias=False), raw=False)
            df_math['ret_kurt_m'] = df_math['ret'].rolling(w_mid).apply(lambda x: kurtosis(x, bias=False), raw=False)
        except Exception:
            # Estimadores simples por momentos
            def _skew(x):
                x = np.array(x, dtype=float)
                m = x.mean()
                s = x.std(ddof=1)
                if s == 0: return 0.0
                return np.mean(((x - m) / s) ** 3)

            def _kurt(x):
                x = np.array(x, dtype=float)
                m = x.mean()
                s = x.std(ddof=1)
                if s == 0: return 0.0
                return np.mean(((x - m) / s) ** 4) - 3.0

            df_math['ret_skew_m'] = df_math['ret'].rolling(w_mid).apply(_skew, raw=False)
            df_math['ret_kurt_m'] = df_math['ret'].rolling(w_mid).apply(_kurt, raw=False)

        df_math['range_s'] = (df_math['high'].rolling(w_short).max() - df_math['low'].rolling(w_short).min())

        # 3) Volatilidad y riesgo (agregando estimadores de vela a ventana)
        rs = _rogers_satchell(df_math['open'], df_math['high'], df_math['low'], df_math['close'])
        gk = _garman_klass(df_math['open'], df_math['high'], df_math['low'], df_math['close'])
        pk = _parkinson(df_math['high'], df_math['low'])
        df_math['rs_vol_m'] = pd.Series(rs).rolling(w_mid).mean()
        df_math['gk_vol_m'] = pd.Series(gk).rolling(w_mid).mean()
        df_math['pk_vol_m'] = pd.Series(pk).rolling(w_mid).mean()

        # VaR histórico (percentil 5% de retornos)
        df_math['var_95_m'] = df_math['ret'].rolling(w_mid).apply(lambda x: np.nanpercentile(x, 5), raw=False)

        # 4) Dependencia temporal (autocorr)
        df_math['acf1_s'] = df_math['ret'].rolling(w_short).apply(lambda x: _rolling_autocorr(np.array(x), lag=1),
                                                                  raw=False)
        df_math['acf2_s'] = df_math['ret'].rolling(w_short).apply(lambda x: _rolling_autocorr(np.array(x), lag=2),
                                                                  raw=False)

        # 5) Información y complejidad
        df_math['entropy_s'] = df_math['ret'].rolling(w_short).apply(
            lambda x: _shannon_entropy(np.array(x), bins=entropy_bins),
            raw=False)

        df_math['p_entropy_s'] = df_math['ret'].rolling(w_short).apply(
            lambda x: _permutation_entropy(np.array(x), m=3, tau=1),
            raw=False
        )
        df_math['hurst_m'] = df_math['ret'].rolling(w_mid).apply(lambda x: _hurst_rs(np.array(x)), raw=False)

        # 6) Geometría del precio (pendiente y ruido)
        # Rolling apply no soporta outputs múltiples; hacemos manual con loop ligero sobre índices recientes
        slope_vals = np.full(len(df_math), np.nan, dtype=float)
        mse_vals = np.full(len(df_math), np.nan, dtype=float)

        arr_close = df_math['close'].to_numpy()
        for i in range(w_mid - 1, len(df_math)):
            window = arr_close[i - w_mid + 1: i + 1]
            sl, me = _linreg_slope_mse(window)  # tu helper que retorna (slope, mse)
            slope_vals[i] = sl
            mse_vals[i] = me

        df_math['slope_m'] = slope_vals
        df_math['lin_mse_m'] = mse_vals

        # Distancia a extremos recientes (normalizada)
        roll_max = df_math['close'].rolling(w_mid).max()
        roll_min = df_math['close'].rolling(w_mid).min()
        df_math['dist_to_max_m'] = (df_math['close'] - roll_max) / roll_max
        df_math['dist_to_min_m'] = (df_math['close'] - roll_min) / roll_min

        # 7) Volumen / microestructura simple
        df_math['volume'] = df_math['volume'].astype(float)

        # Volumen solo en velas alcistas como **Series**, no ndarray
        up_vol = df_math['volume'].where(df_math['close'] > df_math['open'], 0.0)

        df_math['vol_rel_m'] = df_math['volume'] / df_math['volume'].rolling(w_mid).mean()
        df_math['vol_std_m'] = df_math['volume'].rolling(w_mid).std() / df_math['volume'].rolling(w_mid).mean()
        df_math['up_vol_ratio'] = (
                up_vol.rolling(w_mid).sum()
                / (df_math['volume'].rolling(w_mid).sum() + 1e-9)
        )
        # Correlación rolling precio-volumen
        df_math['corr_close_vol_m'] = df_math['close'].rolling(w_mid).corr(df_math['volume'])

        # Variance ratio (corta vs larga) sobre retornos
        var_s = df_math['ret'].rolling(w_short).var()
        var_l = df_math['ret'].rolling(w_long).var()
        df_math['var_ratio'] = var_s / (var_l + 1e-9)

        # Número de cruces por la media (oscilación local)
        ma_m = df_math['close'].rolling(w_mid).mean()
        sign_series = np.sign(df_math['close'] - ma_m)
        df_math['cross_ma_m'] = (
            (sign_series * sign_series.shift(1) < 0)
            .astype(int).rolling(w_mid).sum()
        )

        # Limpieza final
        feature_cols = [
            'ret', 'ret_diff2',
            'ret_mean_s', 'ret_skew_m', 'ret_kurt_m', 'range_s',
            'var_95_m',
            'acf1_s', 'acf2_s', 'p_entropy_s', 'hurst_m',
            'lin_mse_m',
            'vol_rel_m', 'vol_std_m', 'up_vol_ratio', 'corr_close_vol_m', 'cross_ma_m'
        ]

        # df_features = df_math[feature_cols].copy()
        return df_math, feature_cols

    def drop_highly_correlated_features(
            self,
            df: pd.DataFrame,
            threshold: float = 0.9,
            method: str = "spearman",  # 'pearson' | 'spearman' | 'kendall'
            coerce_objects: bool = True,  # intenta convertir object->numeric (coerce)
            min_non_na_frac: float = 0.0,  # descarta columnas con demasiados NaN si quieres (>0.0)
            verbose: bool = True,
            return_info: bool = False
    ) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Elimina automáticamente features numéricas altamente correlacionadas,
        ignorando/intentando convertir columnas no numéricas y preservando el resto del DF.

        - threshold: |corr| a partir del cual se considera redundante
        - method:    'pearson'/'spearman'/'kendall'
        - coerce_objects: si True, intenta pd.to_numeric(errors='coerce') en columnas 'object'/'string'
        - min_non_na_frac: si >0, descarta columnas numéricas con fracción de no-NaN < este valor
        - return_info: si True, devuelve (df_filtrado, info_dict)
        """

        info = {
            "numeric_used": [],
            "non_numeric_ignored": [],
            "coerced_to_numeric": [],
            "dropped_constant": [],
            "dropped_low_coverage": [],
            "dropped_correlated": [],
            "corr_threshold": threshold,
            "method": method
        }

        # 1) Intento de coerción para objetos si procede
        work = df.copy()
        if coerce_objects:
            obj_cols = work.select_dtypes(include=["object", "string"]).columns.tolist()
            for c in obj_cols:
                before_na = work[c].isna().sum()
                coerced = pd.to_numeric(work[c], errors="coerce")
                after_na = coerced.isna().sum()
                # Si aumentaron los NaN, igual la dejamos (se usará solo si queda numérica y con cobertura suficiente)
                if not coerced.equals(work[c]):
                    info["coerced_to_numeric"].append(c)
                work[c] = coerced

        # 2) Seleccionar solo columnas numéricas (booleans incluidos)
        num_df = work.select_dtypes(include=[np.number, "bool"]).copy()
        non_numeric_cols = [c for c in df.columns if c not in num_df.columns]
        info["non_numeric_ignored"] = non_numeric_cols

        # 3) Sanitizar: reemplazar inf/-inf por NaN
        num_df = num_df.replace([np.inf, -np.inf], np.nan)

        # 4) (Opcional) cobertura mínima de datos no-NaN
        if min_non_na_frac > 0.0:
            keep_mask = (num_df.notna().mean(axis=0) >= min_non_na_frac)
            low_cov_cols = num_df.columns[~keep_mask].tolist()
            if low_cov_cols:
                info["dropped_low_coverage"] = low_cov_cols
                num_df = num_df.loc[:, keep_mask]

        # 5) Eliminar constantes (varianza cero)
        # DEBUG: Ver qué se está marcando como constante
        if verbose:
            print("\n[DEBUG] Análisis de constantes:")
        
        constant_cols = []
        for col in num_df.columns:
            unique_with_nan = num_df[col].nunique(dropna=False)
            unique_no_nan = num_df[col].nunique(dropna=True)
            variance = num_df[col].var()
            nan_count = num_df[col].isna().sum()
            
            # Una columna es constante si tiene ≤1 valor único (excluyendo NaN) O varianza 0
            is_constant = (unique_no_nan <= 1) or (variance == 0 or pd.isna(variance))
            
            if is_constant:
                constant_cols.append(col)
                if verbose:
                    print(f"  🔴 {col}:")
                    print(f"     - Únicos (con NaN): {unique_with_nan}")
                    print(f"     - Únicos (sin NaN): {unique_no_nan}")
                    print(f"     - Varianza: {variance}")
                    print(f"     - NaN count: {nan_count}/{len(num_df)}")
                    sample_vals = num_df[col].dropna().head(10).tolist()
                    print(f"     - Sample values: {sample_vals}")

        
        if constant_cols:
            info["dropped_constant"] = constant_cols
            num_df = num_df.drop(columns=constant_cols, errors="ignore")
        else:
            print("No constants")

        # Si queda <2 columnas numéricas, no hay correlación que calcular
        if num_df.shape[1] < 2:
            if verbose:
                print("[INFO] Menos de 2 columnas numéricas tras limpieza; no se elimina por correlación.")
            filtered = pd.concat([num_df, df[non_numeric_cols]], axis=1)
            filtered = filtered[df.columns]  # respetar orden original
            if return_info:
                info["numeric_used"] = num_df.columns.tolist()
                return filtered, info
            return filtered

        # 6) (Opcional) imputación mínima para corr: fila a fila (no obligatoria)
        # Para Spearman/Kendall, pandas maneja NaN internamente fila a fila. No imputamos por defecto.

        # 7) Matriz de correlación absoluta
        corr = num_df.corr(method=method).abs()

        # 8) Triángulo superior para no duplicar pares
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # 9) Marcar a eliminar por correlación
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        info["dropped_correlated"] = to_drop
        info["numeric_used"] = num_df.columns.tolist()

        if verbose:
            if info["non_numeric_ignored"]:
                print(f"[INFO] Ignoradas (no numéricas): {len(info['non_numeric_ignored'])}")
            if info["coerced_to_numeric"]:
                print(f"[INFO] Convertidas a numéricas (coerce): {len(info['coerced_to_numeric'])}")
            if info["dropped_constant"]:
                print(f"[INFO] Constantes eliminadas: {len(info['dropped_constant'])}")
                print(info["dropped_constant"])
            if info["dropped_low_coverage"]:
                print(f"[INFO] Eliminadas por baja cobertura: {len(info['dropped_low_coverage'])}")
            print(f"[INFO] Eliminadas por alta correlación (> {threshold}): {len(to_drop)}")
            for col in to_drop:
                partners = upper.index[upper[col] > threshold].tolist()
                if partners:
                    print(f"   - {col} correlacionada con {partners}")

        # 10) Reconstruir DF completo (quitando solo las numéricas redundantes)
        num_df = num_df.drop(columns=to_drop, errors="ignore")
        # filtered = pd.concat([num_df, df[non_numeric_cols]], axis=1)
        # filtered = filtered[df.columns]  # mantener orden original

        # Imprimir columnas que pasaron el filtro
        if verbose:
            kept_numeric_cols = num_df.columns.tolist()
            print(f"\n[INFO] ✅ Columnas numéricas que PASARON el filtro: {len(kept_numeric_cols)}")
            print(f"{kept_numeric_cols}")

        if return_info:
            return info
        # return filtered

    def build_regime_features(self, df: pd.DataFrame,
                              n_hmm_components: int = 4,
                              random_state: int = 42) -> pd.DataFrame:
        """
            Construye features de régimen de mercado:
            - market_regime  (KMeans)
            - hmm_state      (GaussianHMM)
            """

        df_regime = df.copy()

        # ======================================
        # 1) KMEANS MARKET REGIME
        # ======================================
        regime_features = [
            "volatility_50",
            "range_prev_day_pct",
            "hurst_m",
            "rsi_50",
            "vol_ratio",
            "up_vol_ratio",
        ]

        missing_reg = [c for c in regime_features if c not in df_regime.columns]
        if missing_reg:
            raise ValueError(f"Faltan columnas para KMeans: {missing_reg}")

        X_reg = df_regime[regime_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        km = KMeans(n_clusters=3, random_state=random_state, n_init=10)
        df_regime["market_regime"] = km.fit_predict(X_reg)

        # ======================================
        # 2) HMM MARKET STATE
        # ======================================
        hmm_cols = ["ret", "volatility_50", "volume"]

        missing_hmm = [c for c in hmm_cols if c not in df_regime.columns]
        if missing_hmm:
            raise ValueError(f"Faltan columnas para HMM: {missing_hmm}")

        X_hmm = df_regime[hmm_cols].copy()
        X_hmm = X_hmm.replace([np.inf, -np.inf], np.nan)

        # máscara de filas completamente válidas (sin NaNs)
        mask_valid = X_hmm.notna().all(axis=1)

        if mask_valid.sum() < n_hmm_components * 10:
            # muy pocos datos limpios para entrenar
            df_regime["hmm_state"] = np.nan
            print("[HMM] Muy pocos datos válidos para entrenar HMM, se deja hmm_state = NaN.")
            return df_regime

        X_hmm_valid = X_hmm.loc[mask_valid].values

        # debug opcional
        # print("Filas válidas HMM:", X_hmm_valid.shape[0])

        # escalado
        scaler = StandardScaler()
        X_hmm_valid_scaled = scaler.fit_transform(X_hmm_valid)

        # chequeo extra: no debería haber NaNs aquí
        # print("NaNs en X_hmm_valid_scaled:", np.isnan(X_hmm_valid_scaled).sum())
        # print("Infs en X_hmm_valid_scaled:", np.isinf(X_hmm_valid_scaled).sum())

        hmm = GaussianHMM(
            n_components=n_hmm_components,
            covariance_type="diag",
            n_iter=200,
            random_state=random_state
        )

        # 🔴 IMPORTANTE: aquí usamos X_hmm_valid_scaled, NO df[...] crudo
        hmm.fit(X_hmm_valid_scaled)

        # 🔧 FIX: Normalizar transmat_ para evitar filas con suma 0
        # Esto ocurre cuando algunos estados nunca tienen transiciones observadas
        transmat = hmm.transmat_.copy()
        row_sums = transmat.sum(axis=1, keepdims=True)
        
        # Para filas con suma 0, distribuir uniformemente
        zero_rows = (row_sums == 0).flatten()
        if zero_rows.any():
            print(f"[HMM] Warning: {zero_rows.sum()} estados sin transiciones observadas. Normalizando...")
            transmat[zero_rows] = 1.0 / n_hmm_components  # Distribución uniforme
            row_sums = transmat.sum(axis=1, keepdims=True)
        
        # Normalizar todas las filas para asegurar suma = 1
        hmm.transmat_ = transmat / (row_sums + 1e-12)

        hidden_states = hmm.predict(X_hmm_valid_scaled)

        # inicializamos con NaN
        df_regime["hmm_state"] = np.nan
        # llenamos solo donde había datos válidos
        df_regime.loc[mask_valid, "hmm_state"] = hidden_states.astype(float)


        df_regime["buy_pressure"] = df_regime["volume"] * (df_regime["close"] > df_regime["open"]).astype(int)
        df_regime["sell_pressure"] = df_regime["volume"] * (df_regime["close"] < df_regime["open"]).astype(int)

        df_regime["vol_imbalance"] = (df_regime["buy_pressure"] - df_regime["sell_pressure"]) / (df_regime["volume"] + 1e-9)

        df_regime["signed_volume"] = np.sign(df_regime["close"].diff()) * df_regime["volume"]
        df_regime["signed_volume_ema"] = df_regime["signed_volume"].ewm(span=30).mean()

        df_regime["upper_wick"] = df_regime["high"] - np.maximum(df_regime["close"], df_regime["open"])
        df_regime["lower_wick"] = np.minimum(df_regime["close"], df_regime["open"]) - df_regime["low"]
        df_regime["wick_imbalance"] = df_regime["upper_wick"] - df_regime["lower_wick"]

        df_regime["micro_slope"] = (df_regime["close"] - df_regime["close"].shift(3)) / df_regime["close"].shift(3)

        df_regime["ret_ema_20"] = df_regime["ret"].ewm(span=20).mean()
        df_regime["ret_ema_50"] = df_regime["ret"].ewm(span=50).mean()

        df_regime["volatility_ema_30"] = df_regime["ret"].abs().ewm(span=30).mean()

        df_regime["dist_S2_ema"] = df_regime["dist_S2"].ewm(span=40).mean()

        df_regime["shock_ema"] = ((df_regime["ret"].abs() > df_regime["ret"].abs().rolling(200).mean() * 2).astype(int)).ewm(span=50).mean()

        df_regime["trend_slope_ema"] = df_regime["delta_trend_slope_100"].ewm(span=30).mean()

        def rolling_rsquared(series):
            y = series.values
            x = np.arange(len(y))
            x = (x - x.mean()) / (x.std() + 1e-9)
            y = (y - y.mean()) / (y.std() + 1e-9)

            # regresion lineal simple
            beta = np.sum(x * y) / (np.sum(x ** 2) + 1e-9)
            y_hat = beta * x
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2) + 1e-9
            return 1 - ss_res / ss_tot

        df_regime["trend_r2_50"] = (
            df_regime["close"]
            .rolling(50, min_periods=50)
            .apply(rolling_rsquared, raw=False)
        )

        df_regime["trend_acc"] = df_regime["micro_slope"].diff()

        df_regime["whipsaw"] = (df_regime["ATR"] / df_regime["volatility_50"]).ewm(span=30).mean()

        # Limpieza final
        regime_cols = [
            'market_regime', 'hmm_state', 'buy_pressure', 'sell_pressure',
            'signed_volume_ema', 'upper_wick', 'lower_wick', 'wick_imbalance', 'micro_slope', 'ret_ema_20',
            'shock_ema', 'trend_slope_ema', 'trend_r2_50', 'trend_acc', 'whipsaw'
        ]

        # df_features = df_math[feature_cols].copy()
        return df_regime, regime_cols

    def preparar_features(
            self,
            df_final: pd.DataFrame,
            w_short: int = 12,
            w_mid: int = 24,
            w_long: int = 96,
            entropy_bins: int = 10
    ):
        """
            df: DataFrame con columnas ['datetime','open','high','low','close','volume']
            Retorna: (df_features, feature_names)
            """

        df_final, trader_cols = self.preparar_features_trader(df_final)
        df_final, complementary_cols = self.new_features_complementary(df_final)
        df_final, math_cols = self.build_mathstat_features(df_final)

        df_final, regime_cols = self.build_regime_features(df_final)



        # features_trader = ['dist_S2', 'dist_R2', 'close', 'dist_PP', 'volatility_50', 'minute_sin', 'minute_cos',
        #                    'overnight_range_pct', 'trend_slope_norm_100', 'dist_S2_8h_atr', 'avg_volume_last_n',
        #                    'dist_R2_8h_atr', 'overnight_range_abs', 'range_prev_day_pct', 'volume_lag1',
        #                    'dist_R1_8h_atr', 'range_prev_day_abs', 'overnight_range_atr', 'dist_P_8h_atr',
        #                    'trend_r2_100', 'range_prev_day_atr', 'trend_channel_slope_low',
        #                    'ema_ratio', 'trend_dist_pct_100', 'rsi_50', 'dist_resist_rl', 'roc_30', 'dist_support_rl',
        #                    'd', 'dow_sin', 'volume', 'dow_cos', 'candle_engulfing']

        # features_trader = ['dist_S2', 'dist_R2', 'close', 'dist_PP', 'volatility_50', 'minute_sin', 'minute_cos',
        #                    'overnight_range_pct', 'dist_S2_8h_atr', 'avg_volume_last_n',
        #                    'dist_R2_8h_atr', 'overnight_range_abs', 'range_prev_day_pct', 'volume_lag1',
        #                    'range_prev_day_abs',
        #                    'overnight_range_atr', 'dist_P_8h_atr',
        #                    'range_prev_day_atr', 'trend_channel_slope_low',
        #                    'ema_ratio', 'rsi_50', 'dist_resist_rl', 'roc_30', 'dist_support_rl',
        #                    'd', 'dow_sin', 'volume', 'dow_cos']


        features_final = FEATURES

        # Reemplaza inf/-inf por NaN en las columnas de interés y deja que tu pipeline downstream dropee NaNs
        df_final[features_final] = df_final[features_final].replace([np.inf, -np.inf], np.nan)

        # self.drop_highly_correlated_features(df_final[features_final], threshold=0.9)

        print("Datos preparados para el entrenamiento")

        return df_final, features_final

    def add_obos_features(self, df, k_col="k", d_col="d", rsi_col="rsi_50",
                          lookback_pct=200, near_atr=0.25):
        out = df.copy()

        k = out[k_col].astype(float)
        d = out[d_col].astype(float)
        rsi = out[rsi_col].astype(float)

        # Umbrales adaptativos por percentil rodante (evita rigidez 80/20)
        k_p80 = k.rolling(lookback_pct, min_periods=50).quantile(0.80)
        k_p20 = k.rolling(lookback_pct, min_periods=50).quantile(0.20)
        rsi_p70 = rsi.rolling(lookback_pct, min_periods=50).quantile(0.70)
        rsi_p30 = rsi.rolling(lookback_pct, min_periods=50).quantile(0.30)

        out["k_overb"] = (k >= k_p80).astype(int)
        out["k_overs"] = (k <= k_p20).astype(int)
        out["rsi_overb"] = (rsi >= rsi_p70).astype(int)
        out["rsi_overs"] = (rsi <= rsi_p30).astype(int)

        # Persistencia (tiempo en extremo) y reentrada (cruz de vuelta)
        out["k_overb_run"] = out["k_overb"].groupby((out["k_overb"] != out["k_overb"].shift()).cumsum()).cumcount() * \
                             out["k_overb"]
        out["k_overs_run"] = out["k_overs"].groupby((out["k_overs"] != out["k_overs"].shift()).cumsum()).cumcount() * \
                             out["k_overs"]

        out["k_reenter_below"] = ((k.shift(1) >= k_p80.shift(1)) & (k < k_p80)).astype(int)  # salida de sobrecompra
        out["k_reenter_above"] = ((k.shift(1) <= k_p20.shift(1)) & (k > k_p20)).astype(int)  # salida de sobreventa

        # Sobreextensión: z-score y distancia a EMA en ATR
        if "EMA_20" not in out:
            out["EMA_20"] = out["close"].ewm(span=20, adjust=False).mean()
        if "ATR" not in out:
            hl = out["high"] - out["low"]
            hc = (out["high"] - out["close"].shift()).abs()
            lc = (out["low"] - out["close"].shift()).abs()
            out["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        eps = 1e-12
        out["boll_ma"] = out["close"].rolling(20).mean()
        out["boll_std"] = out["close"].rolling(20).std()
        out["z_close_20"] = (out["close"] - out["boll_ma"]) / (out["boll_std"] + eps)
        out["dist_ema20_atr"] = (out["close"] - out["EMA_20"]) / (out["ATR"] + eps)

        # Confluencias con S/R (re-usa tus distancias ya calculadas)
        # Si ya tienes nearest_res_atr / nearest_sup_atr, mejor. Si no, usa las min de tus columnas.
        sr_res_cols = [c for c in out.columns if c.lower().startswith("dist_r") and c.endswith("_atr")]
        sr_sup_cols = [c for c in out.columns if c.lower().startswith("dist_s") and c.endswith("_atr")]
        out["nearest_res_atr"] = out[sr_res_cols].min(axis=1) if sr_res_cols else np.nan
        out["nearest_sup_atr"] = out[sr_sup_cols].min(axis=1) if sr_sup_cols else np.nan

        out["near_res_flag"] = (out["nearest_res_atr"] <= near_atr).astype(int)
        out["near_sup_flag"] = (out["nearest_sup_atr"] <= near_atr).astype(int)

        # Interacciones muy útiles para reversión
        out["overb_near_res"] = ((out["k_overb"] | out["rsi_overb"]) & out["near_res_flag"]).astype(int)
        out["overs_near_sup"] = ((out["k_overs"] | out["rsi_overs"]) & out["near_sup_flag"]).astype(int)

        return out

    def add_divergence_features(self, df, rsi_col="rsi_50", k_col="k", win=20):
        out = df.copy()
        # picos/vales recientes (muy simple)
        out["price_hh"] = (out["high"] >= out["high"].rolling(win).max().shift(1)).astype(int)
        out["price_ll"] = (out["low"] <= out["low"].rolling(win).min().shift(1)).astype(int)

        rsi = out[rsi_col].astype(float)
        k = out[k_col].astype(float)
        out["rsi_lh"] = (rsi <= rsi.rolling(win).max().shift(1)).astype(int)
        out["rsi_hl"] = (rsi >= rsi.rolling(win).min().shift(1)).astype(int)
        out["k_lh"] = (k <= k.rolling(win).max().shift(1)).astype(int)
        out["k_hl"] = (k >= k.rolling(win).min().shift(1)).astype(int)

        # flags de divergencia
        out["bear_div_rsi"] = (out["price_hh"] & out["rsi_lh"]).astype(int)
        out["bear_div_k"] = (out["price_hh"] & out["k_lh"]).astype(int)
        out["bull_div_rsi"] = (out["price_ll"] & out["rsi_hl"]).astype(int)
        out["bull_div_k"] = (out["price_ll"] & out["k_hl"]).astype(int)
        return out

    def add_htf_obos(self, df, freq="1H", k_col="k", rsi_col="rsi_50"):
        out = df.copy().set_index("datetime")
        agg = out[[k_col, rsi_col]].resample(freq).last().rename(columns={
            k_col: f"{k_col}_{freq}", rsi_col: f"{rsi_col}_{freq}"
        })
        out = out.join(agg, how="left").ffill().reset_index()
        # flags HTF
        out[f"{k_col}_{freq}_p80"] = out[f"{k_col}_{freq}"].rolling(200).quantile(0.80)
        out[f"{k_col}_{freq}_p20"] = out[f"{k_col}_{freq}"].rolling(200).quantile(0.20)
        out[f"{k_col}_{freq}_overb"] = (out[f"{k_col}_{freq}"] >= out[f"{k_col}_{freq}_p80"]).astype(int)
        out[f"{k_col}_{freq}_overs"] = (out[f"{k_col}_{freq}"] <= out[f"{k_col}_{freq}_p20"]).astype(int)
        return out

    def add_new_day_features(self, dfx: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        import pandas as pd

        eps = 1e-9
        dfx = dfx.copy()

        # --- 0) Nombres esperados ---
        base_feats = [
            'range_prev_day_abs', 'range_prev_day_pct',
            'gap_open_abs', 'gap_open_pct',
            'overnight_range_abs', 'overnight_range_pct'
        ]
        atr_feats = ['range_prev_day_atr', 'overnight_range_atr']

        # --- 1) Asegurar datetime y claves auxiliares ---
        if 'datetime' in dfx.columns and not pd.api.types.is_datetime64_any_dtype(dfx['datetime']):
            dfx['datetime'] = pd.to_datetime(dfx['datetime'], errors='coerce')
        dfx['__date'] = dfx['datetime'].dt.floor('D')
        dfx['__hour'] = dfx['datetime'].dt.hour
        dfx['__minute'] = dfx['datetime'].dt.minute

        # --- 2) Agregados diarios ---
        daily = (
            dfx.groupby('__date', as_index=True)
            .agg(day_open=('open', 'first'),
                 day_high=('high', 'max'),
                 day_low=('low', 'min'),
                 day_close=('close', 'last'))
            .sort_index()
        )
        daily['range_prev_day_abs'] = (daily['day_high'] - daily['day_low']).shift(1)
        daily['range_prev_day_pct'] = daily['range_prev_day_abs'] / (daily['day_close'].shift(1).abs() + eps)

        # --- 3) Gap de apertura ---
        daily['gap_open_abs'] = daily['day_open'] - daily['day_close'].shift(1)
        daily['gap_open_pct'] = daily['gap_open_abs'] / (daily['day_close'].shift(1).abs() + eps)

        # --- 4) Volatilidad overnight (20:00–06:59) ---
        overnight_flag = (dfx['__hour'] >= 20) | (dfx['__hour'] < 7)
        overnight_date = np.where(dfx['__hour'] >= 20,
                                  dfx['__date'] + pd.Timedelta(days=1),
                                  dfx['__date'])
        dfx['__overnight_date'] = pd.to_datetime(overnight_date)

        overnight = (
            dfx.loc[overnight_flag]
            .groupby('__overnight_date', as_index=True)
            .agg(overnight_high=('high', 'max'),
                 overnight_low=('low', 'min'),
                 overnight_close=('close', 'last'))
            .sort_index()
        )
        overnight['overnight_range_abs'] = overnight['overnight_high'] - overnight['overnight_low']

        # Normalización por close del día previo
        overnight = overnight.join(
            daily[['day_close']].shift(1).rename(columns={'day_close': 'prev_close'}),
            how='left'
        )
        overnight['overnight_range_pct'] = overnight['overnight_range_abs'] / (overnight['prev_close'].abs() + eps)

        # --- 5) Evitar sufijos: borrar si ya existen antes de merge ---
        cols_to_drop = [c for c in base_feats + atr_feats if c in dfx.columns]
        if cols_to_drop:
            dfx.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Merge diario → __date
        dfx = dfx.merge(
            daily[['range_prev_day_abs', 'range_prev_day_pct', 'gap_open_abs', 'gap_open_pct']],
            left_on='__date', right_index=True, how='left', suffixes=('', '_dup1')
        )

        # Merge overnight → __date (overnight se atribuye al día actual)
        dfx = dfx.merge(
            overnight[['overnight_range_abs', 'overnight_range_pct']],
            left_on='__date', right_index=True, how='left', suffixes=('', '_dup2')
        )

        # Si por algún motivo quedaron duplicados con sufijos, preferimos las columnas sin sufijo
        for c in base_feats:
            dup1, dup2 = f'{c}_dup1', f'{c}_dup2'
            if c not in dfx.columns:
                # Si no quedó la "limpia", intenta rescatar alguna duplicada
                if dup1 in dfx.columns:
                    dfx.rename(columns={dup1: c}, inplace=True)
                elif dup2 in dfx.columns:
                    dfx.rename(columns={dup2: c}, inplace=True)
            # Borra restos
            for dup in (dup1, dup2):
                if dup in dfx.columns:
                    dfx.drop(columns=[dup], inplace=True, errors='ignore')

        # --- 6) Normalización extra por ATR (si existe ATR) ---
        if 'ATR' in dfx.columns:
            # Asegura que existan bases (si merge no trajo nada, crea NaN para evitar KeyError)
            for c in base_feats:
                if c not in dfx.columns:
                    dfx[c] = np.nan
            dfx['range_prev_day_atr'] = dfx['range_prev_day_abs'] / (dfx['ATR'] + eps)
            dfx['overnight_range_atr'] = dfx['overnight_range_abs'] / (dfx['ATR'] + eps)

        # --- 7) Limpieza auxiliares ---
        dfx.drop(columns=['__date', '__hour', '__minute', '__overnight_date'], errors='ignore', inplace=True)

        # --- 8) Garantizar columnas y sanitizar ---
        # (Asegura existencia aunque todo haya sido NaN por faltantes de merge)
        ensure_cols = base_feats + (atr_feats if 'ATR' in dfx.columns else [])
        for c in ensure_cols:
            if c not in dfx.columns:
                dfx[c] = np.nan

        dfx[ensure_cols] = dfx[ensure_cols].replace([np.inf, -np.inf], np.nan)

        return dfx

    def _apply_thresholds(self,
                          proba_row,
                          classes,
                          thresholds,
                          default_class="lateral",
                          global_floor=0.50,  # confianza mínima global
                          min_margin=0.03):  # diferencia mínima top1 - top2

        # 1) Candidatas que superan su umbral por clase
        accepted = [(cls, float(p)) for cls, p in zip(classes, proba_row)
                    if float(p) >= float(thresholds.get(cls, 0.33))]

        if not accepted:
            return default_class

        # 2) Ordenar por prob
        accepted.sort(key=lambda x: x[1], reverse=True)
        top_cls, top_p = accepted[0]

        # 3) Confianza mínima global
        if global_floor is not None and top_p < global_floor:
            return default_class

        # 4) Margen vs. segunda mejor probabilidad (sobre TODAS las clases)
        sorted_all = sorted([float(p) for p in proba_row], reverse=True)
        second_p = sorted_all[1] if len(sorted_all) > 1 else 0.0
        if min_margin is not None and (top_p - second_p) < min_margin:
            return default_class

        return top_cls

    def simulate_thresholds(self, X_test_scaled, y_test, thresholds):
        """
        Evalúa la matriz de confusión y reporte de clasificación usando umbrales por clase.
        Requiere que self.model ya esté calibrado y entrenado.
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Primero entrena/calibra el modelo (self.model).")

        proba = self.model.predict_proba(X_test_scaled)
        classes = list(self.model.classes_)
        y_pred = [self._apply_thresholds(p, classes, thresholds) for p in proba]

        labels = ["bearish", "bullish", "lateral"]  # orden consistente
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print("=== Umbrales usados ===")
        print(thresholds)
        print("\nMatriz de confusión con umbrales:\n", cm)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        print("\nMatriz normalizada por fila:\n", np.round(cm_norm, 3))
        print("\nReporte de clasificación con umbrales:")
        print(classification_report(y_test, y_pred, labels=labels))

    def wilson_ci(self, k, n, z=1.96):
        if n == 0: return (np.nan, np.nan)
        p = k / n
        denom = 1 + z ** 2 / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
        return center - half, center + half

    def audit_bands(self, model, X_hold_scaled, y_hold, classes,
                    bins=np.arange(0.55, 0.91, 0.05),
                    min_margin=None, session_mask=None, session_name="all"):
        proba = model.predict_proba(X_hold_scaled)
        top = proba.max(axis=1)
        idx = proba.argmax(axis=1)
        cls_top = np.array([classes[i] for i in idx])
        is_trend = np.isin(cls_top, ["bullish", "bearish"])
        top2 = np.partition(proba, -2, axis=1)[:, -2]
        margin = top - top2

        y = np.asarray(y_hold)

        if session_mask is None:
            session_mask = np.ones_like(top, dtype=bool)

        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (top >= lo) & (top < hi) & is_trend & session_mask
            if min_margin is not None:
                m &= (margin >= min_margin)
            n = int(m.sum())
            if n == 0:
                rows.append((session_name, f"[{lo:.2f},{hi:.2f})", 0, np.nan, np.nan, np.nan))
                continue
            hits = int((y[m] == cls_top[m]).sum())
            prec = hits / n
            ci_lo, ci_hi = self.wilson_ci(hits, n)
            rows.append((session_name, f"[{lo:.2f},{hi:.2f})", n, prec, ci_lo, ci_hi))
        return pd.DataFrame(rows, columns=["session", "band", "n", "precision", "ci_lo", "ci_hi"])

    def _wilson_ci(self, k, n, z=1.96):
        if n <= 0: return (np.nan, np.nan)
        p = k / n
        denom = 1 + z ** 2 / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
        return center - half, center + half

    def select_thresholds_by_band_precision(
            self,
            X_hold,
            y_hold,
            already_scaled=True,
            features=None,
            band_step=0.05,
            u_min=0.55,
            u_max=0.95,
            min_margin=0.03,
            # --- pisos separados por sesión ---
            min_n_soft=1000,
            min_n_strong=1200,
            min_trade_rate_soft=0.10,
            min_trade_rate_strong=0.03,
            # objetivos (IC inferior de Wilson)
            soft_range=(0.83, 0.865),
            strong_range=(0.89, 0.91),
            soft_start=0.60,
            strong_start=0.80,
            min_gap=0.03,
            use_ci_lower=True,
            prefer_center_soft=True  # solo aplicamos “prefer center” al modo soft
    ):

        def _wilson_ci(k, n, z=1.96):
            if n <= 0: return (np.nan, np.nan)
            p = k / n
            denom = 1 + z ** 2 / n
            center = (p + z * z / (2 * n)) / denom
            half = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
            return center - half, center + half

        # --- coerción y modelo ---
        if not already_scaled:
            if features is None:
                features = getattr(self, "features_", None)
                if features is None:
                    raise ValueError("Debes pasar 'features' si already_scaled=False.")
            X_arr = np.asarray(X_hold[features].values if hasattr(X_hold, "__getitem__") else X_hold)
            X_arr = self.scaler.transform(X_arr)
        else:
            X_arr = np.asarray(X_hold.values) if hasattr(X_hold, "values") else np.asarray(X_hold)
        if X_arr.ndim != 2:
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
            else:
                raise TypeError(f"Se esperaba array 2D, recibido ndim={X_arr.ndim}")

        mdl = getattr(self, "model", None)
        if mdl is None or not hasattr(mdl, "predict_proba"):
            raise RuntimeError("self.model no está entrenado o no soporta predict_proba().")

        proba = mdl.predict_proba(X_arr)
        classes = list(mdl.classes_)
        top = proba.max(axis=1)
        idx = proba.argmax(axis=1)
        cls_top = np.array([classes[i] for i in idx])
        is_trend = np.isin(cls_top, ["bullish", "bearish"])
        top2 = np.partition(proba, -2, axis=1)[:, -2]
        margin = top - top2
        y = np.asarray(y_hold)

        # --- construir bandas ---
        bins = np.arange(u_min, u_max + 1e-9, band_step)
        rows = []
        N = len(y)
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (top >= lo) & (top < hi) & is_trend & (margin >= float(min_margin))
            n = int(m.sum())
            if n == 0:
                rows.append((lo, hi, 0, np.nan, np.nan, np.nan, 0.0));
                continue
            hits = int((y[m] == cls_top[m]).sum())
            prec = hits / n
            ci_lo, ci_hi = _wilson_ci(hits, n)
            trade_rate = n / N
            rows.append((lo, hi, n, prec, ci_lo, ci_hi, trade_rate))
        dfb = pd.DataFrame(rows, columns=["u_lo", "u_hi", "n", "precision", "ci_lo", "ci_hi", "trade_rate"])

        key = "ci_lo" if use_ci_lower else "precision"
        reason_soft = "normal"
        reason_strong = "normal"

        # ---------- SOFT: centro del rango + pisos ----------
        lo_s, hi_s = map(float, soft_range)
        target_s = 0.5 * (lo_s + hi_s)
        cand_s = dfb[dfb["u_lo"] >= float(soft_start)].copy()
        feas_s = cand_s[(cand_s["n"] >= int(min_n_soft)) &
                        (cand_s["trade_rate"] >= float(min_trade_rate_soft)) &
                        (cand_s[key] >= lo_s)].copy()
        if not feas_s.empty:
            if prefer_center_soft:
                feas_s = feas_s.assign(dist=(feas_s[key] - target_s).abs())
                feas_s = feas_s.sort_values(by=["dist", "u_lo", "n"], ascending=[True, True, False])
            else:
                feas_s = feas_s.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
            soft_row = feas_s.iloc[0]
        else:
            # fallback soft: mejor ci_lo, luego n/trade
            fb_s = cand_s[cand_s["n"] > 0].copy()
            fb_s = fb_s.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
            soft_row = fb_s.iloc[0]
            reason_soft = "fallback"

        u_soft = float(np.round(soft_row["u_lo"], 3))

        # ---------- STRONG: volumen primero + pisos ----------
        lo_h, hi_h = map(float, strong_range)
        cand_h = dfb[dfb["u_lo"] >= float(strong_start)].copy()

        # 1) Cumple CI_min + pisos ⇒ elegir por mayor n/trade_rate (prioriza volumen)
        feas_h = cand_h[(cand_h["n"] >= int(min_n_strong)) &
                        (cand_h["trade_rate"] >= float(min_trade_rate_strong)) &
                        (cand_h[key] >= lo_h)].copy()
        if not feas_h.empty:
            feas_h = feas_h.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
            strong_row = feas_h.iloc[0]
        else:
            # 2) Relajar CI (pero mantener pisos) y elegir por volumen
            relax_h = cand_h[(cand_h["n"] >= int(min_n_strong)) &
                             (cand_h["trade_rate"] >= float(min_trade_rate_strong))].copy()
            if not relax_h.empty:
                relax_h = relax_h.sort_values(by=[key, "n", "trade_rate", "u_lo"],
                                              ascending=[False, False, False, True])
                strong_row = relax_h.iloc[0]
                reason_strong = "relaxed_quality_kept_floors"
            else:
                # 3) Último recurso: bajar u hasta encontrar la mejor banda por n/trade (pero ≥ strong_start)
                fb_h = cand_h[cand_h["n"] > 0].copy()
                fb_h = fb_h.sort_values(by=["n", "trade_rate", key, "u_lo"], ascending=[False, False, False, True])
                strong_row = fb_h.iloc[0]
                reason_strong = "fallback_volume_first"

        u_strong = float(np.round(strong_row["u_lo"], 3))

        # ---------- BUMP opcional de strong si ≤ 0.75 ----------
        # (se hace ANTES de forzar gap para que el gap considere el bump)
        if u_strong <= 0.75:
            u_bumped = float(np.round(u_strong + 0.05, 3))
            # buscar fila exacta para el bump
            m_b = np.isclose(dfb["u_lo"].values, u_bumped)
            if m_b.any():
                row_b = dfb.loc[m_b].iloc[0]
                # validar pisos + calidad mínima strong
                floors_ok = (int(row_b["n"]) >= int(min_n_strong)) and \
                            (float(row_b["trade_rate"]) >= float(min_trade_rate_strong))
                quality_ok = (float(row_b[key]) >= float(lo_h))
                if floors_ok and quality_ok:
                    u_strong = u_bumped
                    strong_row = row_b
                    reason_strong = "forced_bump_if_<=0.75"
                else:
                    reason_strong = "forced_bump_if_<=0.75_rejected_by_floors_or_quality"
            else:
                reason_strong = "forced_bump_if_<=0.75_no_band"

        # --- Enforce gap: soft < strong ---
        if u_soft >= u_strong:
            below = dfb[(dfb["u_lo"] < u_strong) &
                        (dfb["u_lo"] >= soft_start) &
                        (dfb["n"] >= int(min_n_soft)) &
                        (dfb["trade_rate"] >= float(min_trade_rate_soft))].copy()
            if not below.empty:
                below = below.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
                soft_row = below.iloc[0]
                u_soft = float(np.round(min(soft_row["u_lo"], u_strong - float(min_gap)), 3))
                reason_soft = reason_soft if reason_soft != "normal" else "adjusted_below_strong"
            else:
                u_soft = float(np.round(max(float(soft_start), u_strong - float(min_gap)), 3))
                # re-sync soft_row con la banda de u_soft (si existe)
                m = np.isclose(dfb["u_lo"].values, u_soft)
                if m.any():
                    soft_row = dfb.loc[m].iloc[0]
                reason_soft = reason_soft if reason_soft != "normal" else "forced_gap"

        # --- Guard post-selección STRONG: nunca devolver algo por debajo de pisos (tras bump/gap) ---
        if (int(strong_row["n"]) < int(min_n_strong)) or (
                float(strong_row["trade_rate"]) < float(min_trade_rate_strong)):
            cands = dfb[(dfb["u_lo"] >= float(strong_start)) &
                        (dfb["n"] >= int(min_n_strong)) &
                        (dfb["trade_rate"] >= float(min_trade_rate_strong)) &
                        (dfb[key] >= float(lo_h))].copy()
            if not cands.empty:
                cands = cands.sort_values(by=[key, "n", "trade_rate", "u_lo"], ascending=[False, False, False, True])
                strong_row = cands.iloc[0]
                u_strong = float(np.round(strong_row["u_lo"], 3))
                reason_strong = (
                        reason_strong + "_adjusted_for_floors") if reason_strong != "normal" else "adjusted_for_floors"
            else:
                # último recurso: fija mínimo razonable manteniendo gap
                u_strong = float(np.round(max(0.75, u_soft + float(min_gap)), 3))
                # re-sync strong_row si existe banda
                m = np.isclose(dfb["u_lo"].values, u_strong)
                if m.any():
                    strong_row = dfb.loc[m].iloc[0]
                reason_strong = (reason_strong + "_forced_min_u75") if reason_strong != "normal" else "forced_min_u75"

        # --- RE-SYNC FINAL: asegura que las métricas correspondan a los u elegidos ---
        m_str = np.isclose(dfb["u_lo"].values, u_strong)
        if m_str.any():
            strong_row = dfb.loc[m_str].iloc[0]
        m_soft = np.isclose(dfb["u_lo"].values, u_soft)
        if m_soft.any():
            soft_row = dfb.loc[m_soft].iloc[0]

        # persistir
        self.umbral_soft = u_soft
        self.umbral_strong = u_strong

        self.threshold_policy_ = {
            "method": "band_precision_selector_strong_max_volume",
            "params": {
                "band_step": float(band_step), "u_min": float(u_min), "u_max": float(u_max),
                "min_margin": float(min_margin),
                "min_n_soft": int(min_n_soft), "min_n_strong": int(min_n_strong),
                "min_trade_rate_soft": float(min_trade_rate_soft),
                "min_trade_rate_strong": float(min_trade_rate_strong),
                "soft_range": tuple(map(float, soft_range)), "strong_range": tuple(map(float, strong_range)),
                "soft_start": float(soft_start), "strong_start": float(strong_start),
                "min_gap": float(min_gap), "use_ci_lower": bool(use_ci_lower)
            },
            "soft": {
                "reason": reason_soft,
                "chosen": {
                    "u": u_soft, "n": int(soft_row["n"]), "precision": float(soft_row["precision"]),
                    "ci": [float(soft_row["ci_lo"]), float(soft_row["ci_hi"])],
                    "band": [float(soft_row["u_lo"]), float(soft_row["u_hi"])],
                    "trade_rate": float(soft_row["trade_rate"])
                }
            },
            "strong": {
                "reason": reason_strong,
                "chosen": {
                    "u": u_strong, "n": int(strong_row["n"]), "precision": float(strong_row["precision"]),
                    "ci": [float(strong_row["ci_lo"]), float(strong_row["ci_hi"])],
                    "band": [float(strong_row["u_lo"]), float(strong_row["u_hi"])],
                    "trade_rate": float(strong_row["trade_rate"])
                }
            }
        }
        return u_strong, self.threshold_policy_["strong"], u_soft, self.threshold_policy_["soft"], dfb


class LabeledCalibratedModel:
    """
    Envuelve un CalibratedClassifierCV entrenado con y numérico (0..K-1)
    y expone predict/predict_proba con clases en texto.
    """

    def __init__(self, calibrated_model, label_encoder: LabelEncoder):
        self._calib = calibrated_model
        self.le = label_encoder
        # Exponer clases en texto, no numéricas
        self.classes_ = self.le.classes_

    def predict_proba(self, X):
        return self._calib.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.le.inverse_transform(idx)

    # opcional: acceder al estimador base si lo necesitas
    @property
    def base_estimator(self):
        return self._calib.base_estimator


def is_allowed_time():
    now = datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # Bloqueo desde sábado 17:00 (5pm) hasta lunes 06:00 (6am)
    if (weekday == 5 and hour >= 17) or (weekday == 6) or (weekday == 0 and hour < 4):
        return False
    return True


def multiclass_brier(y_true_labels: np.ndarray, proba: np.ndarray, classes_text: List) -> float:
    """
    Brier score multiclass: promedio de (p_k - y_k)^2 sobre K clases.
    y_true_labels: labels en el mismo dominio que classes_text
    proba: shape (n, K) alineada con classes_text
    """
    # mapa a índices 0..K-1 según classes_text
    idx_map = {c: i for i, c in enumerate(classes_text)}
    y_idx = np.array([idx_map[c] for c in y_true_labels])
    y_onehot = np.zeros_like(proba)
    y_onehot[np.arange(len(y_true_labels)), y_idx] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def ece_maxprob(y_true_labels: np.ndarray, proba: np.ndarray, classes_text: List, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE): bucketiza por p_max.
    """
    idx_map = {c: i for i, c in enumerate(classes_text)}
    y_idx = np.array([idx_map[c] for c in y_true_labels])
    p_max = proba.max(axis=1)
    y_hat = proba.argmax(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (p_max >= b0) & (p_max < b1) if b1 < 1.0 else (p_max >= b0) & (p_max <= b1)
        if not np.any(mask):
            continue
        conf = p_max[mask].mean()
        acc = (y_hat[mask] == y_idx[mask]).mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def print_block_metrics(title: str,
                        y_test_labels: np.ndarray,
                        proba: np.ndarray,
                        classes_text: List):
    """
    Imprime el bloque de métricas exactamente como lo quieres.
    y_test_labels: etiquetas en el dominio 'classes_text'
    proba: shape (n,K) ordenada como classes_text
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(classes_text)
    y_test_enc = le.transform(y_test_labels)  # 0..K-1
    preds_labels = np.array(classes_text)[proba.argmax(axis=1)]

    print(title)
    print(classification_report(y_test_labels, preds_labels, labels=classes_text))
    print(f"Brier (multiclase): {multiclass_brier(y_test_labels, proba, classes_text):.6f}")
    print(f"Log loss: {log_loss(y_test_enc, proba, labels=np.arange(len(classes_text))):.6f}")
    print(f"ECE: {ece_maxprob(y_test_labels, proba, classes_text):.6f}")


class IdentityCalibrator:
    """No calibra; sólo para interfaz homogénea."""

    def __init__(self, classes_):
        self.classes_ = np.array(classes_)

    def fit(self, *args, **kwargs):
        return self

    def predict_proba(self, P):
        return P


class IsotonicWrapper:
    """
    Calibración isotónica binaria sobre p(1). Mantiene salida shape (n,2).
    """

    def __init__(self, classes_):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.classes_ = np.array(classes_)

    def fit(self, P, y):
        # P: (n,2) alineado a self.classes_
        idx1 = int(np.where(self.classes_ == 1)[0][0])
        p1 = P[:, idx1]
        self.iso.fit(p1, y)
        return self

    def predict_proba(self, P):
        idx1 = int(np.where(self.classes_ == 1)[0][0])
        p1 = P[:, idx1]
        p1c = self.iso.predict(p1)
        p1c = np.clip(p1c, 1e-6, 1 - 1e-6)
        p0c = 1.0 - p1c
        # Reconstituye (n,2) en el mismo orden de self.classes_
        out = np.zeros_like(P)
        out[:, idx1] = p1c
        out[:, 1 - idx1] = p0c
        return out


class CalibratedMetaModel:
    """
    En producción:
      - Si calibrator == 'none' → usa rf_final.predict_proba alineado.
      - Si 'beta' o 'isotonic' → aplica calibrador a proba raw alineada.
      - Si 'platt' → usa directamente el CalibratedClassifierCV (ignora rf_final).
    """

    def __init__(self, rf_final, classes_, mode="none",
                 beta_calib=None, iso_calib=None, platt_ccv=None):
        self.rf_final = rf_final
        self.classes_ = np.array(classes_)
        self.mode = mode
        self.beta_calib = beta_calib
        self.iso_calib = iso_calib
        self.platt_ccv = platt_ccv

    def _align_clip_norm(self, P, model_classes):
        order = np.array([np.where(model_classes == c)[0][0] for c in self.classes_])
        P = P[:, order]
        P = np.clip(P, 1e-6, 1 - 1e-6)
        P /= P.sum(axis=1, keepdims=True)
        return P

    def predict_proba(self, X):
        if self.mode == "platt":
            # Platt: usar directamente el ccv (ya está entrenado con prefit)
            P = self.platt_ccv.predict_proba(X)
            return self._align_clip_norm(P, self.platt_ccv.classes_)
        else:
            # Raw del RF final
            P = self.rf_final.predict_proba(X)
            P = self._align_clip_norm(P, self.rf_final.classes_)
            # Aplicar calibración si corresponde
            if self.mode == "beta" and self.beta_calib is not None:
                P = self.beta_calib.predict_proba(P)
                P = np.clip(P, 1e-6, 1 - 1e-6);
                P /= P.sum(axis=1, keepdims=True)
            elif self.mode == "isotonic" and self.iso_calib is not None:
                P = self.iso_calib.predict_proba(P)
                P = np.clip(P, 1e-6, 1 - 1e-6);
                P /= P.sum(axis=1, keepdims=True)
            return P


class PlattWrapper:
    """Sigmoid/Platt binaria sobre p(1) vía CalibratedClassifierCV estilo 'cv=__prefit'."""

    def __init__(self, base_clf, base_classes):
        self.base = base_clf
        self.classes_ = np.array(base_classes)
        self.ccv = None

    def fit(self, X, y, X_val=None, y_val=None):
        # Entrena un CalibratedClassifierCV con 'prefit' sobre holdout
        if X_val is None or y_val is None:
            raise ValueError("Se requiere holdout para Platt.")
        self.ccv = CalibratedClassifierCV(self.base, method='sigmoid', cv='prefit')
        self.ccv.fit(X_val, y_val)
        return self

    def predict_proba(self, X):
        return self.ccv.predict_proba(X)


def _build_Xy_strict(
        dfm: pd.DataFrame,
        required_features: List[str],
        target_col: str,
        strict: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    dfm = dfm.copy()

    missing = [c for c in required_features if c not in dfm.columns]
    if missing:
        msg = f"[META] Faltan {len(missing)} features requeridas: {missing}"
        if strict:
            raise ValueError(msg)
        else:
            print("[WARN]", msg, "→ se excluirán del set.")
            required_features = [c for c in required_features if c in dfm.columns]

    feats = list(required_features)

    # direction → numérico (si aparece)
    if "direction" in feats:
        dfm["direction_sign"] = (
            dfm["direction"].map({"bullish": 1.0, "bearish": -1.0}).astype("float32")
        )
        feats[feats.index("direction")] = "direction_sign"

    for c in feats:
        if not pd.api.types.is_numeric_dtype(dfm[c]):
            dfm[c] = pd.to_numeric(dfm[c], errors="coerce")

    nan_report = dfm[feats].isna().sum().sort_values(ascending=False)
    if (nan_report > 0).any():
        print("[META][NaNs por feature]\n", nan_report[nan_report > 0].head(20))

    before = len(dfm)
    dfx = dfm.dropna(subset=feats + [target_col]).copy()
    after = len(dfx)
    if after == 0:
        print(f"[META] Dropna eliminó todas las filas ({before} -> 0). Revisa NaNs en features/target.")
    else:
        print(f"[META] Dropna: {before} -> {after} filas (quedaron {after / before:.1%}).")

    X = dfx[feats].astype("float32").to_numpy()
    y = dfx[target_col].astype(int).to_numpy()
    return X, y, feats, dfx


# ============================================================
#               FUNCIÓN PRINCIPAL CORREGIDA
# ============================================================

def train_meta_model_rf_beta_locked(
        df_meta: pd.DataFrame,
        features_final: List[str],
        target_col: str = "y_meta",
        N_KEEP: int = 40000,
        n_splits: int = 5,
        embargo: int = 5,
        random_state: int = 42,
        min_train_size: int = 5000,
        strict_features: bool = True,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
        calibrator: str = "beta",  # 'beta' | 'isotonic' | 'platt' | 'none'
        abandon_if_worse: bool = True,
        tol_logloss: float = 1e-6,
) -> Tuple[object, Dict]:
    """
    Entrena M2 (binario) con features bloqueadas.
    - Evalúa con split temporal (holdout) SIN fuga.
    - Calibra con OOF + embargo si es posible; si no, con sub-holdout interno del train.
    - Si la calibración empeora logloss en el holdout de evaluación, se descarta.

    Devuelve: (modelo_envuelto, metrics)
      - modelo_envuelto: objeto con .predict_proba(X) que aplica calibración si 'útil'.
    """
    eps = 1e-6

    if "datetime" not in df_meta.columns:
        raise ValueError("df_meta debe incluir 'datetime'.")

    # 0) Orden + corte temporal
    def _fmt_ts(ts):
        if ts is None:
            return "NA"
        # Si tiene tz, muéstrala; si no, sin tz
        try:
            if getattr(ts, "tzinfo", None):
                return ts.strftime("%Y-%m-%d %H:%M:%S%z")
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(ts)

    dfm0 = df_meta.sort_values("datetime").reset_index(drop=True).copy()
    print(f"[META] Total filas iniciales: {len(dfm0)}")

    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
        tz_df = getattr(dfm0["datetime"].dt, "tz", None)
        if tz_df is not None and as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize(tz_df)
        elif tz_df is not None and as_of_ts.tzinfo is not None and str(as_of_ts.tzinfo) != str(tz_df):
            as_of_ts = as_of_ts.tz_convert(tz_df)

        before_cut = len(dfm0)
        dfm0 = dfm0[dfm0["datetime"] <= as_of_ts].copy()
        print(f"[META] Corte temporal as_of={as_of_ts} → {before_cut} -> {len(dfm0)} filas")
        if dfm0.empty:
            raise RuntimeError("[META] No hay datos previos a 'as_of'.")

        # Fechas tras el recorte
        start_ts = dfm0["datetime"].iloc[0]
        end_ts = dfm0["datetime"].iloc[-1]
        print(f"[META] Rango tras recorte: start={_fmt_ts(start_ts)} | end={_fmt_ts(end_ts)}")

    else:
        if dfm0.empty:
            raise RuntimeError("[META] DataFrame vacío.")
        # Fechas sin recorte
        start_ts = dfm0["datetime"].iloc[0]
        end_ts = dfm0["datetime"].iloc[-1]
        print(f"[META] Rango sin recorte: start={_fmt_ts(start_ts)} | end={_fmt_ts(end_ts)}")

    # 1) Etiqueta binaria si no existe
    if target_col not in dfm0.columns:
        dfm0[target_col] = np.where(
            dfm0["result"].eq("tp"),
            1,
            np.where(dfm0["result"].eq("sl"), 0, (dfm0["pnl_R"].fillna(0.0) > 0).astype(int)),
        )

    # 2) Filtrar direcciones
    before_dir = len(dfm0)
    dfm1 = dfm0[dfm0["direction_sign"].isin([1.0, -1.0])].copy()
    print(f"[META] Filtro direction: {before_dir} -> {len(dfm1)}")

    # 3) Recorte N_KEEP
    if len(dfm1) > N_KEEP:
        dfm1 = dfm1.iloc[-N_KEEP:].copy()
        print(f"[META] Recorte N_KEEP={N_KEEP}: quedaron {len(dfm1)} filas")

    if len(dfm1) < min_train_size:
        print(f"[META][WARN] n={len(dfm1)} < min_train_size={min_train_size}. Intento continuar...")

    # 4) X, y con features bloqueadas
    X_all, y_all, feats_used, dfx = _build_Xy_strict(
        dfm=dfm1,
        required_features=features_final,
        target_col=target_col,
        strict=strict_features,
    )

    n = len(y_all)
    classes_ = np.array(sorted(np.unique(y_all)))  # [0,1]
    print(f"[META] Tras dropna → n={n} | clases={classes_.tolist()}")
    if n == 0:
        raise RuntimeError("[META] No hay muestras después de dropna.")
    if len(classes_) < 2:
        raise RuntimeError(f"[META] Clases insuficientes: {classes_.tolist()} (requiere 0 y 1).")

    # 5) Split temporal para evaluación (80/20; puedes cambiar regla si quieres)
    split_eval = int(np.floor(0.8 * n))
    if split_eval == 0 or split_eval == n:
        split_eval = max(1, int(np.floor(0.7 * n)))
    if split_eval == 0 or split_eval == n:
        raise RuntimeError("[META] Dataset demasiado pequeño para holdout.")

    X_tr, y_tr = X_all[:split_eval], y_all[:split_eval]
    X_ho, y_ho = X_all[split_eval:], y_all[split_eval:]
    estimators = 700
    # 6) Entrena modelo sólo con TRAIN para EVALUAR sin fuga
    rf_eval = RandomForestClassifier(
        n_estimators=estimators, max_features="sqrt", min_samples_leaf=1,
        n_jobs=-1, random_state=random_state, class_weight='balanced'
    ).fit(X_tr, y_tr)

    # 7) Probabilidades RAW en holdout de evaluación (alinear+clip+renorm)
    def align_clip_norm(P, model_classes, classes_target):
        order = np.array([np.where(model_classes == c)[0][0] for c in classes_target])
        P = P[:, order]
        P = np.clip(P, eps, 1 - eps)
        P /= P.sum(axis=1, keepdims=True)
        return P

    P_raw_eval = align_clip_norm(rf_eval.predict_proba(X_ho), rf_eval.classes_, classes_)

    # 8) OOF para calibración (si posible)
    can_oof = (n_splits >= 2) and (len(y_tr) >= n_splits)
    proba_oof = None
    # if can_oof:
    # if False:
    #     print(f"[META] OOF con KFold n_splits={n_splits} sobre TRAIN (n={len(y_tr)})")
    #     kf = KFold(n_splits=n_splits, shuffle=False)
    #     proba_oof = np.full((len(y_tr), len(classes_)), np.nan, dtype=float)
    #     idx_all = np.arange(len(y_tr))
    #
    #     for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(idx_all)):
    #         # embargo temporal
    #         if embargo > 0:
    #             te_start, te_end = te_idx.min(), te_idx.max()
    #             keep_left = tr_idx < (te_start - embargo)
    #             keep_right = tr_idx > (te_end + embargo)
    #             tr_idx = tr_idx[keep_left | keep_right]
    #
    #         if len(tr_idx) == 0 or len(te_idx) == 0:
    #             print(f"[META][OOF] Fold {fold_idx}: skip (train={len(tr_idx)}, test={len(te_idx)})")
    #             continue
    #
    #         rf = RandomForestClassifier(
    #             n_estimators=estimators, max_features="sqrt", min_samples_leaf=1,
    #             n_jobs=-1, random_state=random_state + 100 + fold_idx, class_weight='balanced'
    #         ).fit(X_tr[tr_idx], y_tr[tr_idx])
    #
    #         P = rf.predict_proba(X_tr[te_idx])
    #         P = align_clip_norm(P, rf.classes_, classes_)
    #         proba_oof[te_idx, :] = P
    #
    #     if np.isnan(proba_oof).any():
    #         print("[META][OOF] Huecos en OOF → completando con modelo train-full.")
    #         rf_tmp = RandomForestClassifier(
    #             n_estimators=estimators, max_features="sqrt", min_samples_leaf=1,
    #             n_jobs=-1, random_state=random_state + 999, class_weight='balanced'
    #         ).fit(X_tr, y_tr)
    #         P_full = align_clip_norm(rf_tmp.predict_proba(X_tr), rf_tmp.classes_, classes_)
    #         mask_nan = np.isnan(proba_oof).any(axis=1)
    #         proba_oof[mask_nan, :] = P_full[mask_nan]

    # 9) Fallback para calibración con sub-holdout dentro de TRAIN
    #    (evita usar X_ho para entrenar calibrador)
    split_cal = int(np.floor(0.8 * len(y_tr)))
    if split_cal == 0 or split_cal == len(y_tr):
        split_cal = max(1, int(np.floor(0.7 * len(y_tr))))
    X_cal_tr, y_cal_tr = X_tr[:split_cal], y_tr[:split_cal]
    X_cal_va, y_cal_va = X_tr[split_cal:], y_tr[split_cal:]

    # Para beta/isotonic si no hubo OOF:
    if proba_oof is None and len(y_cal_va) > 0:
        rf_for_cal = RandomForestClassifier(
            n_estimators=estimators, max_features="sqrt", min_samples_leaf=1,
            n_jobs=-1, random_state=random_state + 200, class_weight='balanced'
        ).fit(X_cal_tr, y_cal_tr)
        P_cal_va = align_clip_norm(rf_for_cal.predict_proba(X_cal_va), rf_for_cal.classes_, classes_)
    else:
        P_cal_va = None  # se usará proba_oof

    # 10) Construir calibradores
    beta_calib = None
    iso_calib = None
    platt_ccv = None

    if calibrator == "beta":
        try:
            # Requiere tu clase externa:
            # from yourmodule import MultiClassBetaCalibrator
            beta_calib = MultiClassBetaCalibrator(parameters="abm", classes_=classes_)
            if proba_oof is not None:
                beta_calib.fit(proba_oof, y_tr)
            else:
                if P_cal_va is None:
                    raise RuntimeError("Sin OOF y sin holdout de calibración para Beta.")
                beta_calib.fit(P_cal_va, y_cal_va)
        except Exception as e:
            print(f"[META][WARN] Beta falló ({e}); se usará 'none'.")
            beta_calib = None
            calibrator_used = "none"
        else:
            calibrator_used = "beta"

    elif calibrator == "isotonic":
        try:
            iso_calib = IsotonicWrapper(classes_)
            if proba_oof is not None:
                iso_calib.fit(proba_oof, y_tr)
            else:
                if P_cal_va is None:
                    raise RuntimeError("Sin OOF y sin holdout de calibración para Isotonic.")
                iso_calib.fit(P_cal_va, y_cal_va)
        except Exception as e:
            print(f"[META][WARN] Isotonic falló ({e}); se usará 'none'.")
            iso_calib = None
            calibrator_used = "none"
        else:
            calibrator_used = "isotonic"

    elif calibrator == "platt":
        try:
            # Base prefit y ccv entrenado con validación de calibración (no con el holdout de evaluación)
            rf_prefit = RandomForestClassifier(
                n_estimators=estimators, max_features="sqrt", min_samples_leaf=1,
                n_jobs=-1, random_state=random_state + 300
            ).fit(X_cal_tr, y_cal_tr)
            platt_ccv = CalibratedClassifierCV(rf_prefit, method='sigmoid', cv='prefit')
            platt_ccv.fit(X_cal_va, y_cal_va)  # usa sólo la parte de train reservada para calibración
        except Exception as e:
            print(f"[META][WARN] Platt falló ({e}); se usará 'none'.")
            platt_ccv = None
            calibrator_used = "none"
        else:
            calibrator_used = "platt"
    else:
        calibrator_used = "none"

    # 11) Evaluación: aplicar calibración al holdout de evaluación
    if calibrator_used == "beta" and beta_calib is not None:
        P_cal_eval = beta_calib.predict_proba(P_raw_eval)
        P_cal_eval = np.clip(P_cal_eval, eps, 1 - eps);
        P_cal_eval /= P_cal_eval.sum(axis=1, keepdims=True)
    elif calibrator_used == "isotonic" and iso_calib is not None:
        P_cal_eval = iso_calib.predict_proba(P_raw_eval)
        P_cal_eval = np.clip(P_cal_eval, eps, 1 - eps);
        P_cal_eval /= P_cal_eval.sum(axis=1, keepdims=True)
    elif calibrator_used == "platt" and platt_ccv is not None:
        P_cal_eval = platt_ccv.predict_proba(X_ho)
        P_cal_eval = align_clip_norm(P_cal_eval, platt_ccv.classes_, classes_)
    else:
        P_cal_eval = P_raw_eval.copy()

    # 12) Métricas en holdout
    ll_raw = log_loss(y_ho, P_raw_eval, labels=classes_)
    ll_cal = log_loss(y_ho, P_cal_eval, labels=classes_)

    use_cal = True
    if abandon_if_worse and (ll_cal > ll_raw + tol_logloss):
        print(
            f"[META][INFO] Calibración '{calibrator_used}' empeoró logloss (raw={ll_raw:.6f} < cal={ll_cal:.6f}). Me quedo con RAW.")
        use_cal = False

    preds_eval = ((P_cal_eval if use_cal else P_raw_eval)[:, int(np.where(classes_ == 1)[0][0])] >= 0.5).astype(int)

    # 13) Reentrena modelo final en TODO el set para deploy
    rf_final = RandomForestClassifier(
        n_estimators=700, max_features="sqrt", min_samples_leaf=1,
        n_jobs=-1, random_state=random_state
    ).fit(X_all, y_all)

    # Para Platt en producción, ajustamos un ccv sobre el mismo holdout de calibración (X_cal_va, y_cal_va)
    # pero con base rf_final (prefit). Esto es lo más consistente posible sin fugar al holdout de evaluación.
    platt_ccv_final = None
    if use_cal and calibrator_used == "platt" and len(y_cal_va) > 0:
        try:
            platt_ccv_final = CalibratedClassifierCV(rf_final, method='sigmoid', cv='prefit')
            platt_ccv_final.fit(X_cal_va, y_cal_va)
        except Exception as e:
            print(f"[META][WARN] Platt final falló ({e}); usaré RAW en producción.")
            platt_ccv_final = None
            use_cal = False
            calibrator_used = "none"

    # 14) Empaquetar el modelo para producción
    if use_cal:
        model_deployed = CalibratedMetaModel(
            rf_final=rf_final,
            classes_=classes_,
            mode=calibrator_used,
            beta_calib=beta_calib if calibrator_used == "beta" else None,
            iso_calib=iso_calib if calibrator_used == "isotonic" else None,
            platt_ccv=platt_ccv_final if calibrator_used == "platt" else None,
        )
    else:
        model_deployed = CalibratedMetaModel(
            rf_final=rf_final,
            classes_=classes_,
            mode="none"
        )

    # 15) Reporte/métricas
    metrics = {
        "classes": classes_.tolist(),
        "n_samples": int(n),
        "N_KEEP": int(N_KEEP),
        "n_splits": int(n_splits),
        "features_used": feats_used,
        "missing_features": [c for c in features_final if c not in feats_used],
        "report_hold": classification_report(y_ho, preds_eval, labels=classes_, output_dict=True),
        "logloss_hold_raw": float(ll_raw),
        "logloss_hold_cal": float(ll_cal),
        "calibrator_requested": calibrator,
        "calibrator_used": calibrator_used if use_cal else "none",
        "calibration_used": bool(use_cal),
        "trained_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "as_of": str(as_of) if as_of is not None else None,
        "train_datetime_min": str(dfx["datetime"].min()) if "datetime" in dfx.columns else None,
        "train_datetime_max": str(dfx["datetime"].max()) if "datetime" in dfx.columns else None,
    }
    print_block_metrics("RAW", y_ho, P_raw_eval, classes_)
    print_block_metrics("CAL", y_ho, P_cal_eval, classes_)

    print("[M2] Features usadas (bloqueadas):", len(feats_used))
    if metrics["missing_features"]:
        print("[M2][WARN] Faltaron:", metrics["missing_features"])
    print(
        f"LogLoss RAW={metrics['logloss_hold_raw']:.6f} | CAL={metrics['logloss_hold_cal']:.6f} | used_cal={metrics['calibration_used']} ({metrics['calibrator_used']})")

    return model_deployed, metrics


def recortar_rango(df_model: pd.DataFrame,
                   as_of: str | pd.Timestamp | None = None,
                   since: str | pd.Timestamp | None = None,
                   since_inclusive: bool = True) -> pd.DataFrame:
    """
    Recorta df_model por fecha:
      - since: fecha inicial (>= si since_inclusive, de lo contrario >)
      - as_of: fecha final (<=)
    Respeta/convierte zona horaria del DF.
    """

    def _fmt_ts(ts, target_tz: str | None = "America/Bogota") -> str:
        """Devuelve un string legible del timestamp, respetando TZ si existe."""
        try:
            # Pandas Timestamp: ts.tz es None si es naive
            if getattr(ts, "tz", None) is not None:
                if target_tz:
                    try:
                        return ts.tz_convert(target_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
                    except Exception:
                        # Si no se puede convertir, al menos formatea con su tz actual
                        return ts.strftime("%Y-%m-%d %H:%M:%S %Z")
                else:
                    return ts.strftime("%Y-%m-%d %H:%M:%S %Z")
            else:
                # Naive
                return ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            # Fallback robusto
            return str(ts)

    df_model = df_model.sort_values("datetime").reset_index(drop=True).copy()
    print(f"[META] Total filas iniciales: {len(df_model)}")

    # Asegura dtype datetime (por si acaso)
    if not pd.api.types.is_datetime64_any_dtype(df_model["datetime"]):
        df_model["datetime"] = pd.to_datetime(df_model["datetime"])

    tz_df = getattr(df_model["datetime"].dt, "tz", None)

    # Normaliza 'as_of'
    as_of_ts = None
    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
        if tz_df is not None and as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize(tz_df)
        elif tz_df is not None and as_of_ts.tzinfo is not None and str(as_of_ts.tzinfo) != str(tz_df):
            as_of_ts = as_of_ts.tz_convert(tz_df)

    # Normaliza 'since'
    since_ts = None
    if since is not None:
        since_ts = pd.to_datetime(since)
        if tz_df is not None and since_ts.tzinfo is None:
            since_ts = since_ts.tz_localize(tz_df)
        elif tz_df is not None and since_ts.tzinfo is not None and str(since_ts.tzinfo) != str(tz_df):
            since_ts = since_ts.tz_convert(tz_df)

    # Aplica filtros
    before_cut = len(df_model)
    mask = pd.Series(True, index=df_model.index)

    if since_ts is not None:
        mask &= df_model["datetime"] >= since_ts if since_inclusive else df_model["datetime"] > since_ts
    if as_of_ts is not None:
        mask &= df_model["datetime"] <= as_of_ts

    df_model = df_model[mask].copy()
    print(f"[META] Corte temporal since={since_ts} | as_of={as_of_ts} → {before_cut} -> {len(df_model)} filas")

    if df_model.empty:
        raise RuntimeError("[META] No hay datos en el rango solicitado (since/as_of).")

    # Fechas tras el recorte
    start_ts = df_model["datetime"].iloc[0]
    end_ts = df_model["datetime"].iloc[-1]
    print(f"[META] Rango tras recorte: start={_fmt_ts(start_ts)} | end={_fmt_ts(end_ts)}")

    return df_model


def _oof_probas_time_series(X_array, y_array, base_estimator, n_splits=5, clip_eps=1e-6):
    """
    Genera probabilidades OOF respetando el tiempo, usando TimeSeriesSplit.
    Alinea columnas de probas a las clases globales (rellena con 0 si faltan en algún fold).
    Devuelve: proba_oof (n_samples, K), y_oof (n_samples,), classes_global (list)
    """
    # Asegúrate de que X_array e y_array mantienen el ORDEN temporal
    classes_global = list(pd.Index(y_array).unique().sort_values())
    n_classes = len(classes_global)
    class_to_idx = {c: i for i, c in enumerate(classes_global)}

    proba_oof = np.zeros((len(y_array), n_classes), dtype=float)
    filled = np.zeros(len(y_array), dtype=bool)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_array)):
        est = clone(base_estimator)
        est.fit(X_array[tr_idx], y_array[tr_idx])

        proba_fold = est.predict_proba(X_array[val_idx])  # (n_val, k_fold)
        # Alinear columnas a clases_global
        proba_aligned = np.zeros((len(val_idx), n_classes), dtype=float)
        model_classes = list(est.classes_)
        model_c2i = {c: i for i, c in enumerate(model_classes)}
        for j, c in enumerate(classes_global):
            if c in model_c2i:
                proba_aligned[:, j] = proba_fold[:, model_c2i[c]]
            else:
                # Si una clase no aparece en este fold, queda 0 (sin info)
                proba_aligned[:, j] = 0.0

        # Clipping de estabilidad
        proba_aligned = np.clip(proba_aligned, clip_eps, 1.0 - clip_eps)
        # Re-normaliza por si hubo columnas 0
        proba_sum = proba_aligned.sum(axis=1, keepdims=True)
        proba_sum[proba_sum == 0] = 1.0
        proba_aligned = proba_aligned / proba_sum

        proba_oof[val_idx] = proba_aligned
        filled[val_idx] = True

    # Por si algo quedó sin llenar (no debería)
    if not filled.all():
        missing = np.where(~filled)[0]
        proba_oof[missing] = 1.0 / n_classes

    return proba_oof, y_array, classes_global


def _safe_feature_importance(model, X_tr, y_tr, feature_names, plot=True, export_csv=None):
    """
    Obtiene importancias de características (nativas o permutation importance).
    Devuelve DataFrame con ['feature', 'importance'].

    Parámetros:
    - plot: bool -> si True, muestra un gráfico de barras horizontales.
    - export_csv: str -> si no es None, guarda el DataFrame en un CSV con este nombre.
    """
    imp = None
    # Intentos comunes
    for attr in ("feature_importances_",):
        if hasattr(model, attr):
            vals = getattr(model, attr)
            if isinstance(vals, (list, np.ndarray)) and len(vals) == len(feature_names):
                imp = np.array(vals, dtype=float)
                break

    # Si no se logró, usar permutation importance
    if imp is None:
        try:
            pi = permutation_importance(model, X_tr, y_tr, n_repeats=5, random_state=42, n_jobs=-1)
            imp = pi.importances_mean
        except Exception:
            imp = np.zeros(len(feature_names), dtype=float)

    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Exportar a CSV si se solicita
    if export_csv:
        df_imp.to_csv(export_csv, index=False)

    # Mostrar gráfico
    if plot:
        plt.figure(figsize=(8, max(6, len(df_imp) * 0.25)))  # alto dinámico
        plt.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
        plt.title("Importancia de características (Random Forest)")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show(block=False)

    return df_imp


def _build_base_model(model_type: str, random_state: int = 42, model_kwargs: Optional[Dict] = None):
    """
    Devuelve un clasificador scikit-compatible según model_type.
    - rf: RandomForestClassifier
    - lgbm: LightGBM (LGBMClassifier)
    - xgb: XGBoost (XGBClassifier)
    """
    model_kwargs = model_kwargs or {}
    model_type = (model_type or "rf").lower()
    if model_type in ("rf", "randomforest", "random_forest"):
        defaults = dict(n_estimators=1000, max_features="sqrt", n_jobs=-1, random_state=random_state)
        defaults.update(model_kwargs)
        return RandomForestClassifier(**defaults), "rf"

    if model_type in ("lgbm", "lightgbm"):
        if not _HAS_LGBM:
            raise ImportError("LightGBM no está instalado. Ejecuta: pip install lightgbm")
        # valores razonables; puedes ajustarlos vía model_kwargs
        defaults = dict(
            n_estimators=700,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1
        )
        defaults.update(model_kwargs)
        return LGBMClassifier(**defaults), "lgbm"

    raise ValueError(f"model_type desconocido: {model_type}. Usa: 'rf', 'lgbm' o 'xgb'.")

def create_predictive_model_m2(
        df_model,
        features,
        display_graphic: bool = False,
        N_KEEP: int = 40000,
        model_type: str = "rf",
        model_kwargs: Optional[Dict] = None,
        calibrate_method: str = "isotonic",  # "isotonic" o "sigmoid"
        return_metrics: bool = False,
        display_feature_plot: bool = False,
        as_of: Optional[Union[str, pd.Timestamp]] = None,
):
    scaler = StandardScaler()
    if as_of is not None:
        as_of_ts = pd.to_datetime(as_of)
        tz_df = getattr(df_model["datetime"].dt, "tz", None)
        if tz_df is not None and as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize(tz_df)
        elif tz_df is not None and as_of_ts.tzinfo is not None and str(as_of_ts.tzinfo) != str(tz_df):
            as_of_ts = as_of_ts.tz_convert(tz_df)
        before_cut = len(df_model)
        df_model = df_model[df_model["datetime"] <= as_of_ts].copy()
        print(f"[META] Corte temporal as_of={as_of_ts} → {before_cut} -> {len(df_model)} filas")
        if df_model.empty:
            raise RuntimeError("[META] No hay datos previos a 'as_of'.")


    # Alinear tz del df_model a la de self.data (si hiciera falta)
    tz_raw = getattr(df_model["datetime"].dt, "tz", None)
    tz_mod = getattr(df_model["datetime"].dt, "tz", None)
    if (tz_raw is not None) and (tz_mod is None):
        df_model["datetime"] = df_model["datetime"].dt.tz_localize(tz_raw)
    elif (tz_raw is not None) and (tz_mod is not None) and (str(tz_raw) != str(tz_mod)):
        df_model["datetime"] = df_model["datetime"].dt.tz_convert(tz_raw)


    try:
        n_total = len(df_model)
        first_val = df_model["datetime"].iloc[0]
        last_val = df_model["datetime"].iloc[-1]

        # Si son timestamps de pandas, usar .strftime
        if isinstance(first_val, (pd.Timestamp, datetime)):
            first_date = first_val.strftime("%Y-%m-%d %H:%M:%S")
            last_date = last_val.strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Caso de enteros o floats tipo UNIX timestamp
            first_date = datetime.fromtimestamp(first_val).strftime("%Y-%m-%d %H:%M:%S")
            last_date = datetime.fromtimestamp(last_val).strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"Rango de fechas en self.data inicial de recorte NKEEP y Lookahead: {first_date} -> {last_date} (n={n_total}).")
    except Exception as e:
        print(f"No se pudo obtener rango de fechas: {e}")


    # 1) Etiqueta binaria si no existe
    if "y_meta" not in df_model.columns:
        # Solo positivo (1) cuando es 'tp', 0 en caso contrario (incluyendo 'sl')
        df_model["y_meta"] = np.where(df_model["result"].eq("tp"), 1, 0)


    # Excluir últimas L para evitar fuga de datos
    df_train = df_model.copy()
    df_train = df_train[df_train["y_meta"].notna()]
    df_train = df_train.dropna(subset=features + ["y_meta"])


    X = df_train[features]
    y = df_train["y_meta"]

    # ===== Métricas auxiliares =====
    def multiclass_brier(y_true_labels, y_prob, classes_text):
        class_to_idx = {c: i for i, c in enumerate(classes_text)}
        y_idx = np.array([class_to_idx[v] for v in y_true_labels])
        Y = np.zeros_like(y_prob)
        Y[np.arange(len(y_true_labels)), y_idx] = 1.0
        return float(np.mean(np.sum((y_prob - Y) ** 2, axis=1)))

    def ece_maxprob(y_true_labels, y_prob, classes_text, n_bins=10):
        class_to_idx = {c: i for i, c in enumerate(classes_text)}
        y_idx = np.array([class_to_idx[v] for v in y_true_labels])
        conf = np.max(y_prob, axis=1)
        pred = np.argmax(y_prob, axis=1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.digitize(conf, bins) - 1
        ece, n = 0.0, len(y_true_labels)
        for i in range(n_bins):
            mask = binids == i
            if np.any(mask):
                acc = np.mean(pred[mask] == y_idx[mask])
                avg_conf = np.mean(conf[mask])
                ece += (np.sum(mask) / n) * abs(acc - avg_conf)
        return float(ece)

    def safe_log_loss(y_true, y_pred, labels=None):
        y_pred = np.asarray(y_pred)

        # Caso: proba binaria (n_samples, 1) pero se pasan dos labels
        if y_pred.ndim == 2 and y_pred.shape[1] == 1 and labels is not None and len(labels) == 2:
            # Interpretamos la columna como p(clase positiva = labels[1])
            return log_loss(y_true, y_pred.ravel(), labels=labels)

        # Caso normal: dejamos que sklearn haga su magia
        if labels is None:
            return log_loss(y_true, y_pred)
        else:
            return log_loss(y_true, y_pred, labels=labels)

    # ===============================
    # Entrenamiento modelo base
    # ===============================
    base_model, model_tag = _build_base_model(model_type=model_type, model_kwargs=model_kwargs or {})

    # ===== Ruta normal (RF / LGBM) sin encoding =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    base_model.fit(X_tr, y_tr)

    # ---- Sin calibrar ----
    preds_vanilla = base_model.predict(X_test_scaled)
    proba_vanilla = base_model.predict_proba(X_test_scaled)
    classes = base_model.classes_

    print("Reporte de clasificación de modelo SIN calibrar:")
    print(classification_report(y_test, preds_vanilla, labels=classes))
    print(f"Brier (multiclase): {multiclass_brier(y_test, proba_vanilla, classes):.6f}")

    # print(f"Log loss: {log_loss(y_test, proba_vanilla, labels=classes):.6f}")
    # print(f"Log loss: {safe_log_loss(y_test, proba_vanilla):.6f}")

    print(f"ECE: {ece_maxprob(y_test, proba_vanilla, classes):.6f}")
    print("=" * 50)

    # ---- Calibración ----
    if calibrate_method not in ("isotonic", "sigmoid", "beta"):
        calibrate_method = "isotonic"

    if calibrate_method == "beta":
        # === NEW: OOF + Beta + clipping (time-series aware) ===
        # 1) Generar OOF sobre el TRAIN (X_train_scaled, y_train)
        proba_oof, y_oof, classes_global = _oof_probas_time_series(
            X_array=X_train_scaled,
            y_array=y_train.to_numpy(),  # respeta orden temporal del df
            base_estimator=base_model,
            n_splits=5,
            clip_eps=1e-6
        )
        class_to_idx = {c: i for i, c in enumerate(classes_global)}
        y_oof_idx = np.array([class_to_idx[v] for v in y_oof], dtype=int)

        # 2) Ajustar calibrador Beta sobre OOF (prueba 'ab'/'bm'/'abm')
        mc_beta = MultiClassBetaCalibrator(parameters="abm", classes_=classes_global)
        mc_beta.fit(proba_oof, y_oof_idx)

        # 3) Re-entrenar el modelo base en TODO el TRAIN y envolver con Beta
        base_model_final = clone(base_model).fit(X_train_scaled, y_train)

        preds_vanilla = base_model_final.predict(X_test_scaled)
        proba_vanilla = base_model_final.predict_proba(X_test_scaled)
        classes = base_model_final.classes_

        print("Reporte de clasificación de modelo SIN calibrar + entrenamiento completo:")
        print(classification_report(y_test, preds_vanilla, labels=classes))
        print(f"Brier (multiclase): {multiclass_brier(y_test, proba_vanilla, classes):.6f}")
        print(f"Log loss: {log_loss(y_test, proba_vanilla, labels=classes):.6f}")
        print(f"ECE: {ece_maxprob(y_test, proba_vanilla, classes):.6f}")
        print("=" * 50)

        model = BetaMultiWrapper(base_model_final, mc_beta, classes_=classes_global, label_encoder=None)

        # 4) (Opcional) clipping en inferencia dentro del wrapper (si tu wrapper no lo hace)
        #    Asegúrate que BetaMultiWrapper hace np.clip en predict_proba antes/después del calibrador.
    else:
        model = CalibratedClassifierCV(base_model, method=calibrate_method, cv='prefit')
        model.fit(X_cal, y_cal)

    # Métricas del calibrado final
    proba_cal = model.predict_proba(X_test_scaled)
    preds_cal = model.predict(X_test_scaled)
    classes_cal = getattr(model, "classes_", base_model.classes_)

    # --------- MÉTRICAS CALIBRADAS (DEVUELTAS) ----------
    report_calib = classification_report(y_test, preds_cal, labels=classes_cal, output_dict=True)
    brier_calib = multiclass_brier(y_test, proba_cal, classes_cal)
    # logloss_calib = log_loss(y_test, proba_cal, labels=classes_cal)
    # logloss_calib = safe_log_loss(y_test, proba_cal)
    ece_calib = ece_maxprob(y_test, proba_cal, classes_cal)

    print("Reporte de clasificación de modelo CALIBRADO:")
    print(classification_report(y_test, preds_cal, labels=classes_cal))
    print(f"Brier (multiclase): {brier_calib:.6f}")
    # print(f"Log loss: {logloss_calib:.6f}")
    print(f"ECE: {ece_calib:.6f}")


    # Importancias
    feature_importance = _safe_feature_importance(base_model, X_tr, y_tr, features, display_feature_plot)
    print("\nImportancia de características:")
    print(feature_importance)

    print(f"\nModelo base: {model_tag.upper()} | calibración: {calibrate_method}")


    # Gráficas de calibración
    if display_graphic:
        proba_base = proba_vanilla
        proba_calib = proba_cal
        clases = classes_cal  # o classes; ambas deberían coincidir en orden

        plt.figure(figsize=(7, 5))
        for target_class in clases:
            idx = list(clases).index(target_class)
            y_true_bin = np.array([1 if yt == target_class else 0 for yt in y_test])

            ppb_x, ppb_y = calibration_curve(y_true_bin, proba_base[:, idx], n_bins=10, strategy='uniform')
            plt.plot(ppb_x, ppb_y, marker='o', label=f"Sin calibrar '{target_class}'")

            ppc_x, ppc_y = calibration_curve(y_true_bin, proba_calib[:, idx], n_bins=10, strategy='uniform')
            plt.plot(ppc_x, ppc_y, marker='o', linestyle='--', label=f"Calibrado '{target_class}'")

        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
        plt.xlabel('Probabilidad predicha por el modelo')
        plt.ylabel('Proporción real de acierto')
        plt.title(f'Gráfico de calibración: sin calibrar vs calibrado\nEntrenado: {fecha_actual}')
        plt.legend();
        plt.grid();
        plt.show(block=False)

    # model = base_model

    if return_metrics:
        metrics_dict = {
            "classes": list(classes_cal),
            "report_calib": report_calib,  # dict completo por clase + macro/weighted
            "brier_calib": float(brier_calib),
            # "logloss_calib": float(logloss_calib),
            "ece_calib": float(ece_calib),
            "N_KEEP": N_KEEP,
            "model_tag": model_tag,
            "calibrate_method": calibrate_method,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return model, metrics_dict

    return model


if __name__ == "__main__":
    now = datetime.now()
    # ===========Script automatizado===========
    # today_str = f"{datetime.now().day:02d}"
    # tomorrow = datetime.now() + timedelta(days=1)
    # tomorrow_day = f"{tomorrow.day:02d}"
    # mont_str = f"{datetime.now().month:02d}"
    # filename = f'AVAXUSDT_5m_2022-01-01_2026-{mont_str}-{today_str}.csv'
    # if os.path.exists(filename):
    #     nombre, extension = os.path.splitext(filename)
    #     new_name = nombre[:-2] + tomorrow_day + extension
    #     os.rename(filename, new_name)
    #     print("Nuevo nombre de archivo:", new_name)
    #
    # end_date = f"2026-{mont_str}-{tomorrow_day}"
    # # end_date = f"2026-01-01"
    #
    # filename_tomorrow = f'AVAXUSDT_5m_2022-01-01_{end_date}.csv'
    # if os.path.exists(filename_tomorrow):
    #     print(f"iniciando script a las {now.hour}:{now.minute}:{now.second} horas")
    #     analyzer = StochasticAnalyzer(
    #         api_key=api_key,
    #         api_secret=api_secret,
    #         m2_model=None,
    #         atr_mult=4,
    #         atr_stop=3,
    #         lookahead=50
    #     )
    #     analyzer.run_analysis(
    #         symbol="AVAXUSDT",
    #         start_date="2022-01-01",
    #         end_date=end_date,
    #         interval="5",
    #     )
    #     print("KPIs:", analyzer.tradelogger.compute_stats(True, True))
    #     df = analyzer.data
    #
    #     #===============================Creacion del dataset M1 Output===============================================
    #     print("Iniciando M1 Output dataframe")
    #     builder = M1OutputCacheBuilder(
    #         analyzer,
    #         df,
    #         proba_threshold=0.60,  # único umbral
    #         atr_period=14,
    #         atr_mult_tp=4.0,
    #         atr_mult_sl=3.0,
    #         offline_lookback_bars=200,
    #         n_keep_for_fit=122_000,
    #         # cache=CacheConfig(cache_path="m1_outputs_2026.parquet"),
    #         cache=None,
    #     )
    #     today_str = datetime.now().day
    #     tomorrow = datetime.now() + timedelta(days=1)
    #     tomorrow_day = tomorrow.day
    #     mont_str = int(datetime.now().month)
    #
    #     end_limit = datetime(2026, mont_str, tomorrow_day)  # último test_end deseado (exclusivo)
    #     # start = end_limit - relativedelta(years=1)
    #     start = end_limit - relativedelta(years=1)
    #
    #     chunks = []
    #     while start < end_limit:
    #         end = start + relativedelta(months=1)
    #         if end > end_limit:
    #             end = end_limit  # por si el último tramo fuese parcial
    #         ts, te = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    #         print(f">>> Ejecutando ventana {ts} -> {te}")
    #         meta_i = builder.build_and_cache(test_start=ts, test_end=te, verbose=1, export_parquet=True)
    #         chunks.append(meta_i)
    #         start = end  # avanza a la siguiente ventana
    #
    #     # Unir todos los resultados en un solo DataFrame
    #     m1_output = pd.concat(chunks, ignore_index=True)
    #     # FIX: Eliminamos duplicados por si los chunks se solapan
    #     if "datetime" in m1_output.columns:
    #         m1_output = m1_output.drop_duplicates(subset=["datetime"], keep="first")
    #     else:
    #          m1_output = m1_output.drop_duplicates()
    #     # (opcional) Ordenar por columna temporal si existe
    #     if "datetime" in m1_output.columns:
    #         m1_output = m1_output.sort_values("datetime").reset_index(drop=True)
    #     print("M1 Output listo")
    #     print(m1_output.tail())
    #
    #     #==============================Creacion de modelo predictivo 2 con output M1========================================
    #     features = FEATURES + FEATS_M2
    #     # print("Creando modelo predictivo 2 con output M1")
    #     df_cache = m1_output
    #     # df_cache = pd.read_parquet("m1_outputs_2026.parquet")
    #     # Contar los valores de la columna result
    #     result_counts = df_cache['result'].value_counts()
    #     print("\nDistribución de resultados:")
    #     print(result_counts)
    #     # Calcular winrate (porcentaje de tp sobre el total)
    #     total_trades = len(df_cache[df_cache['result'].notna()])
    #     tp_count = len(df_cache[df_cache['result'] == 'tp'])
    #     sl_count = len(df_cache[df_cache['result'] == 'sl'])
    #     winrate = (tp_count / total_trades * 100) if total_trades > 0 else 0
    #     print(f"\n{'=' * 50}")
    #     print(f"ESTADÍSTICAS DE TRADING")
    #     print(f"{'=' * 50}")
    #     print(f"Total de operaciones: {total_trades}")
    #     print(f"Take Profit (TP): {tp_count} ({tp_count / total_trades * 100:.2f}%)")
    #     print(f"Stop Loss (SL): {sl_count} ({sl_count / total_trades * 100:.2f}%)")
    #     print(f"Winrate: {winrate:.2f}%")
    #     print(f"{'=' * 50}\n")
    #     model_m2 = create_predictive_model_m2(
    #         df_model=df_cache,
    #         features=features,
    #         model_type='lightgbm',
    #         model_kwargs=None,
    #         calibrate_method="sigmoid",
    #         display_graphic=False,
    #         return_metrics=False
    #     )
    #     analyzer2 = StochasticAnalyzer(
    #         api_key=api_key,
    #         api_secret=api_secret,
    #         atr_mult=4,
    #         atr_stop=3,
    #         m2_model=model_m2,
    #         data=df,
    #         early_exit_threshold=0.75,
    #         lookahead=50
    #     )
    #
    #     print("Creando modelo predictivo 1")
    #     analyzer2.create_predictive_model(
    #         model_type="lgbm",
    #         calibrate_method="sigmoid",
    #         N_KEEP=122_000,
    #         display_graphic=False,
    #         model_kwargs={
    #             "objective": "multiclass",
    #             "num_class": 3,  # ajusta según tus clases reales
    #             "n_estimators": 1000,  # más árboles + LR más bajo
    #             "learning_rate": 0.03,  # más bajo para mejorar logloss
    #             "num_leaves": 96,  # complejidad del árbol
    #             "max_depth": -1,  # deja que LGBM lo maneje
    #             "min_child_samples": 200,  # regularización clave con 220k filas
    #
    #             "subsample": 0.7,  # antes 0.9, bajamos para más robustez
    #             "subsample_freq": 1,
    #             "colsample_bytree": 0.7,  # menos columnas por árbol, menos overfit
    #
    #             "reg_lambda": 15.0,  # L2 fuerte → baja variancia, mejora logloss
    #             "reg_alpha": 0.1,  # L1 ligera
    #
    #             "min_split_gain": 0.0,
    #             "random_state": 42,
    #             "n_jobs": -1
    #         },
    #         display_feature_plot=False,
    #         use_df_parquet=True,
    #     )
    #     print("Iniciando predicciones")
    #     analyzer2.monitor_predictions()
    #     print("\nFinalizando primer ciclo")
    #     while True:
    #         now = datetime.now()
    #         if now.hour in {RESTART_HOUR_1} and now.minute == 0:
    #             print("Reiniciando script programado a las ", now.hour, " horas")
    #             now = datetime.now()
    #             #
    #             analyzer = StochasticAnalyzer(api_key=api_key, api_secret=api_secret, m2_model=None, atr_mult=4, atr_stop=3)
    #
    #             today_str = f"{datetime.now().day:02d}"
    #             tomorrow = datetime.now() + timedelta(days=1)
    #             tomorrow_day = f"{tomorrow.day:02d}"
    #             mont_str = f"{datetime.now().month:02d}"
    #             filename = f'AVAXUSDT_5m_2022-01-01_2026-{mont_str}-{today_str}.csv'
    #             if os.path.exists(filename):
    #                 nombre, extension = os.path.splitext(filename)
    #                 new_name = nombre[:-2] + tomorrow_day + extension
    #                 os.rename(filename, new_name)
    #                 print("Nuevo nombre de archivo:", new_name)
    #
    #             end_date = f"2026-{mont_str}-{tomorrow_day}"
    #             filename_tomorrow = f'AVAXUSDT_5m_2022-01-01_{end_date}.csv'
    #             if os.path.exists(filename_tomorrow):
    #                 print(f"iniciando script a las {now.hour}:{now.minute}:{now.second} horas")
    #                 analyzer = StochasticAnalyzer(
    #                     api_key=api_key,
    #                     api_secret=api_secret,
    #                     m2_model=None,
    #                     atr_mult=4,
    #                     atr_stop=3,
    #                     lookahead=50
    #                 )
    #                 analyzer.run_analysis(
    #                     symbol="AVAXUSDT",
    #                     start_date="2022-01-01",
    #                     end_date=end_date,
    #                     interval="5",
    #                 )
    #                 print("KPIs:", analyzer.tradelogger.compute_stats(True, True))
    #                 df = analyzer.data
    #             # ===========M2===========
    #                 builder = M1OutputCacheBuilder(
    #                     analyzer,
    #                     df,
    #                     proba_threshold=0.60,  # único umbral
    #                     atr_period=14,
    #                     atr_mult_tp=4.0,
    #                     atr_mult_sl=3.0,
    #                     offline_lookback_bars=200,
    #                     n_keep_for_fit=122_000,
    #                     cache=None,
    #                 )
    #                 today_str = datetime.now().day
    #                 tomorrow = datetime.now() + timedelta(days=1)
    #                 tomorrow_day = tomorrow.day
    #                 mont_str = int(datetime.now().month)
    #
    #                 end_limit = datetime(2026, mont_str, tomorrow_day)  # último test_end deseado (exclusivo)
    #                 start = end_limit - relativedelta(months=1)
    #
    #                 chunks = []
    #                 while start < end_limit:
    #                     end = start + relativedelta(months=1)
    #                     if end > end_limit:
    #                         end = end_limit  # por si el último tramo fuese parcial
    #                     ts, te = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    #                     print(f">>> Ejecutando ventana {ts} -> {te}")
    #                     meta_i = builder.build_and_cache(test_start=ts, test_end=te, verbose=1)
    #                     chunks.append(meta_i)
    #                     start = end  # avanza a la siguiente ventana
    #
    #                 # Unir todos los resultados en un solo DataFrame
    #                 m1_output = pd.concat(chunks, ignore_index=True)
    #                 # FIX: Eliminamos duplicados por si los chunks se solapan
    #                 if "datetime" in m1_output.columns:
    #                     m1_output = m1_output.drop_duplicates(subset=["datetime"], keep="first")
    #                 else:
    #                     m1_output = m1_output.drop_duplicates()
    #
    #                 # (opcional) Ordenar por columna temporal si existe
    #                 if "datetime" in m1_output.columns:
    #                     m1_output = m1_output.sort_values("datetime").reset_index(drop=True)
    #
    #                 print(m1_output.tail())
    #                 features = FEATURES + FEATS_M2
    #
    #                 print("Creando modelo predictivo 2 con output M1")
    #                 # df_cache = pd.read_parquet("m1_outputs_2year_dec7.parquet")
    #                 df_cache = m1_output
    #                 result_counts = df_cache['result'].value_counts()
    #                 print("\nDistribución de resultados:")
    #                 print(result_counts)
    #                 # Calcular winrate (porcentaje de tp sobre el total)
    #                 total_trades = len(df_cache[df_cache['result'].notna()])
    #                 tp_count = len(df_cache[df_cache['result'] == 'tp'])
    #                 sl_count = len(df_cache[df_cache['result'] == 'sl'])
    #                 winrate = (tp_count / total_trades * 100) if total_trades > 0 else 0
    #                 print(f"\n{'=' * 50}")
    #                 print(f"ESTADÍSTICAS DE TRADING")
    #                 print(f"{'=' * 50}")
    #                 print(f"Total de operaciones: {total_trades}")
    #                 print(f"Take Profit (TP): {tp_count} ({tp_count / total_trades * 100:.2f}%)")
    #                 print(f"Stop Loss (SL): {sl_count} ({sl_count / total_trades * 100:.2f}%)")
    #                 print(f"Winrate: {winrate:.2f}%")
    #                 print(f"{'=' * 50}\n")
    #                 model_m2 = create_predictive_model_m2(
    #                     df_model=df_cache,
    #                     features=features,
    #                     model_type='lightgbm',
    #                     model_kwargs=None,
    #                     calibrate_method="sigmoid",
    #                     display_graphic=False,
    #                     return_metrics=False
    #                 )
    #                 print("Creando modelo predictivo 1")
    #                 analyzer2 = StochasticAnalyzer(
    #                     api_key=api_key,
    #                     api_secret=api_secret,
    #                     atr_mult=4,
    #                     atr_stop=3,
    #                     m2_model=model_m2,
    #                     data=df,
    #                     early_exit_threshold=0.75,
    #                     lookahead=50
    #                 )
    #
    #                 analyzer2.create_predictive_model(
    #                     model_type="lgbm",
    #                     calibrate_method="sigmoid",
    #                     N_KEEP=122_000,
    #                     display_graphic=False,
    #                     model_kwargs={
    #                         "objective": "multiclass",
    #                         "num_class": 3,  # ajusta según tus clases reales
    #                         "n_estimators": 1000,  # más árboles + LR más bajo
    #                         "learning_rate": 0.03,  # más bajo para mejorar logloss
    #                         "num_leaves": 96,  # complejidad del árbol
    #                         "max_depth": -1,  # deja que LGBM lo maneje
    #                         "min_child_samples": 200,  # regularización clave con 220k filas
    #
    #                         "subsample": 0.7,  # antes 0.9, bajamos para más robustez
    #                         "subsample_freq": 1,
    #                         "colsample_bytree": 0.7,  # menos columnas por árbol, menos overfit
    #
    #                         "reg_lambda": 15.0,  # L2 fuerte → baja variancia, mejora logloss
    #                         "reg_alpha": 0.1,  # L1 ligera
    #
    #                         "min_split_gain": 0.0,
    #                         "random_state": 42,
    #                         "n_jobs": -1
    #                     },
    #                     display_feature_plot=False,
    #                     use_df_parquet=False
    #                 )
    #                 print("Iniciando predicciones")
    #                 analyzer2.monitor_predictions()
    #             else:
    #                 print(f"El archivo {filename_tomorrow} no existe")
    #         time.sleep(60)
    # else:
    #     print(f"El archivo {filename} no existe")

    # =====================================TESTING==================================

    today_str = f"{datetime.now().day:02d}"
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_day = f"{tomorrow.day:02d}"
    mont_str = f"{datetime.now().month:02d}"
    filename = f'AVAXUSDT_5m_2022-01-01_2026-{mont_str}-{today_str}.csv'
    if os.path.exists(filename):
        nombre, extension = os.path.splitext(filename)
        new_name = nombre[:-2] + tomorrow_day + extension
        os.rename(filename, new_name)
        print("Nuevo nombre de archivo:", new_name)

    end_date = f"2026-{mont_str}-{tomorrow_day}"
    # end_date = f"2026-01-01"

    filename_tomorrow = f'AVAXUSDT_5m_2022-01-01_{end_date}.csv'
    if os.path.exists(filename_tomorrow):
        print(f"iniciando script a las {now.hour}:{now.minute}:{now.second} horas")
        analyzer = StochasticAnalyzer(api_key=api_key_roge, api_secret=api_secret_roge, m2_model=None, atr_mult=4,
                                      atr_stop=3, lookahead=50)
        analyzer.run_analysis(
            symbol="AVAXUSDT",
            start_date="2022-01-01",
            end_date=end_date,
            interval="5",
        )
        print("KPIs:", analyzer.tradelogger.compute_stats(True, True))
        df = analyzer.data

        # #=============================Creacion del Dataset M1===============================================
        print("Creando Dataset M1")
        builder = M1OutputCacheBuilder(
            analyzer,
            df,
            proba_threshold=0.60,  # único umbral
            atr_period=14,
            atr_mult_tp=4.0,
            atr_mult_sl=3.0,
            offline_lookback_bars=200,
            n_keep_for_fit=122_000,
            cache=CacheConfig(cache_path="m1_outputs_12month_2025.parquet"),
            # cache=CacheConfig(None),
        )
        today_str = datetime.now().day
        tomorrow = datetime.now() + timedelta(days=1)
        tomorrow_day = tomorrow.day
        mont_str = int(datetime.now().month)

        start = datetime(2024, 1, 1)  # cambia a diario
        end_limit = datetime(2025, 1, 1)  # último test_end deseado (exclusivo)

        chunks = []
        while start < end_limit:
            end = start + relativedelta(months=1)
            if end > end_limit:
                end = end_limit  # por si el último tramo fuese parcial
            ts, te = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            print(f">>> Ejecutando ventana {ts} -> {te}")
            meta_i = builder.build_and_cache(test_start=ts, test_end=te, verbose=1, export_parquet=True)
            chunks.append(meta_i)
            start = end  # avanza a la siguiente ventana
        # Unir todos los resultados en un solo DataFrame
        m1_output = pd.concat(chunks, ignore_index=True)
        # FIX: Eliminamos duplicados por si los chunks se solapan
        if "datetime" in m1_output.columns:
            m1_output = m1_output.drop_duplicates(subset=["datetime"], keep="first")
        else:
            m1_output = m1_output.drop_duplicates()
        # (opcional) Ordenar por columna temporal si existe
        if "datetime" in m1_output.columns:
            m1_output = m1_output.sort_values("datetime").reset_index(drop=True)
        print("M1 Output listo")
        print(m1_output.tail())
        #         # # #
        #
        #
        print("Creando modelo predictivo 2 con output M1")
        # df_cache = pd.read_parquet("m1_outputs_12month_2026.parquet")
        df_cache = m1_output
        print("df_cache_parquet")
        # print(len(df_cache_parquet))
        print("df_cache")
        print(len(df_cache))
        # Calcular winrate basado en la columna result
        # Contar los valores de la columna result
        result_counts = df_cache['result'].value_counts()
        print("\nDistribución de resultados:")
        print(result_counts)
        # Calcular winrate (porcentaje de tp sobre el total)
        total_trades = len(df_cache[df_cache['result'].notna()])
        tp_count = len(df_cache[df_cache['result'] == 'tp'])
        sl_count = len(df_cache[df_cache['result'] == 'sl'])
        winrate = (tp_count / total_trades * 100) if total_trades > 0 else 0
        print(f"\n{'=' * 50}")
        print(f"ESTADÍSTICAS DE TRADING")
        print(f"{'=' * 50}")
        print(f"Total de operaciones: {total_trades}")
        print(f"Take Profit (TP): {tp_count} ({tp_count / total_trades * 100:.2f}%)")
        print(f"Stop Loss (SL): {sl_count} ({sl_count / total_trades * 100:.2f}%)")
        print(f"Winrate: {winrate:.2f}%")
        print(f"{'=' * 50}\n")
        # print(m1_output.iloc[-1])

        features = FEATURES + FEATS_M2

        model_m2 = create_predictive_model_m2(
            df_model=df_cache,
            features=features,
            model_type='lightgbm',
            model_kwargs=None,
            calibrate_method="sigmoid",
            display_graphic=False,
            return_metrics=False
        )


        # # ==================== simulator
        def logger(event, payload):
            if event == "progress":
                print(f"[{payload['i']}/{payload['total']}] {payload['time']} | "
                      f"equity={payload['equity']:.2f} | trades={payload['trades']}")
            # elif event == "open":
            #     print(f"OPEN  {payload['side']} @ {payload['price']:.4f} | "
            #           f"tp={payload['tp']:.4f} sl={payload['sl']:.4f} | "
            #           f"proba={payload['prob'] * 100:.1f} | "
            #           f"margin={payload['margin']:.2f} | "
            #           f"kelly={payload['kelly'] if payload['kelly'] is not None else 1.0:.3f}")
            # elif event == "close":
            #     dur = payload.get("duration")
            #     print(f"CLOSE {payload['reason']} @ {payload['price']:.4f} | "
            #           f"PnL={payload['pnl']:.2f} | "
            #           f"equity={payload['equity']:.2f} | "
            #           f"in={payload['entry_time']} out={payload['time']} "
            #           f"dur={dur}")
            # elif event == "early_opposite":
            #     print(
            #         f"EARLY_OPPOSITE {payload['side']} @ {payload['price']:.4f} "
            #         f"| prob={payload['prob']:.3f} | thr={payload['threshold']:.2f} "
            #         f"| in={payload['entry_time']} out={payload['time']} "
            #         f"| dur={payload['duration']} pnl_gross={payload['pnl_gross']:.2f}"
            #     )
            # elif event == "withdrawal":
            #     print(
            #         f"💰 RETIRO DETECTADO ({payload['timestamp']}) | "
            #         f"Monto: ${payload['amount']:.2f} | "
            #         f"Total Retirado: ${payload['total_withdrawn']:.2f} | "
            #         f"Capital Total (Eq+Ret): ${payload['capital_total']:.2f}"
            #     )
        # # #
        # print("Iniciando simulacion sin m2")
        # sim = TradeSimulator(
        #     analyzer,
        #     df,
        #     initial_equity=30,
        #     leverage=1000,
        #     risk_fraction=0.02,
        #     on_event=logger,  # <— activar callback
        #     verbose=1,  # 0: silencio, 1: heartbeat cada N barras, 2: +OPEN/CLOSE
        #     log_every_bars=288,
        #     proba_threshold_day=0.6,
        #     proba_threshold_night=0.6,
        #     n_keep_for_fit=122_000,
        #     atr_mult_tp=4,
        #     atr_mult_sl=3,
        #     retrain_each_bar=False,
        #     retrain_every_bars=288,
        #     suppress_fit_stdout=False,
        #     early_exit_threshold=0.75,
        #     n_estimators=1000,
        #     start_date="2023-01-01",
        #     end_date=end_date,
        #     m2_model=None,
        #     enable_withdrawals=False
        # )
        # # # #
        # results = sim.run(test_start="2025-02-27", test_end="2025-02-28")
        # print("DBG sim.run returned", results["final_equity"])
        # sim.plot_performance(
        #     initial_equity=40,  # Capital inicial
        #     save_html="backtest_report_sin_m2.html",  # Guardar como HTML (opcional)
        #     show_plot=True,  # Mostrar en navegador
        #     height=2400  # Altura en píxeles
        # )

        print("Iniciando simulacion con m2")
        sim = TradeSimulator(
            analyzer,
            df,
            initial_equity=30,
            leverage=1000,
            risk_fraction=0.02,
            on_event=logger,  # <— activar callback
            verbose=1,  # 0: silencio, 1: heartbeat cada N barras, 2: +OPEN/CLOSE
            log_every_bars=288,
            proba_threshold_day=0.6,
            proba_threshold_night=0.6,
            n_keep_for_fit=122_000,
            atr_mult_tp=4,
            atr_mult_sl=3,
            retrain_each_bar=False,
            retrain_every_bars=288,
            suppress_fit_stdout=False,
            early_exit_threshold=0.75,
            n_estimators=1000,
            start_date="2023-01-01",
            end_date=end_date,
            m2_model=model_m2,
            enable_withdrawals=False,
            m2_df=df_cache
        )
        # # #
        results = sim.run(test_start="2025-01-01", test_end="2025-02-01")
        print("DBG sim.run returned", results["final_equity"])
        sim.plot_performance(
            initial_equity=40,  # Capital inicial
            save_html="backtest_report_con_m2_2026.html",  # Guardar como HTML (opcional)
            show_plot=True,  # Mostrar en navegador
            height=2400  # Altura en píxeles
        )
        # # #
        # # rules_raw = sim.autogen_rules_from_entries(
        # #     min_support=5,  # exige soporte sólido
        # #     max_pairs=200,  # pares
        # #     max_triples=150,  # triples
        # #     triples_from_top_k=50,  # solo combina las 25 mejores singles
        # #     no_entry=True,
        # #     no_entry_lift_max=0.9
        # # )
        # #
        # # # 2) Exportar en formato deployable (filtrando un poco)
        # #  deployable = sim.export_rules_deployable(
        # #     rules=rules_raw,
        # #     top_k_per_side=60,
        # #     min_support=10,
        # #     min_lift=1.2,
        # #     min_win_rate=0.7,
        # #     no_entry_top_k=90,
        # #     no_entry_min_support=5,
        # #     no_entry_max_lift=0.9,
        # #     save_path="rules_autogen_220k_6months.py"
        # # )
        # #
        # # # 3) Ya puedes importar 'rules' desde rules_autogen.py en tu rules_engine.py, o cargar el JSON.
        # # print(deployable["bullish"][:3])
        # # print(deployable["bearish"][:3])
        #
        # # =========== get bet best dataset trough step forward testing
        # from math import isfinite
        #
        # mejor_n = None
        # mejor_equity = float("-inf")
        # equities_por_n = {}
        #
        # for n in range(100_000, 140_001, 2000):
        #     sim = TradeSimulator(
        #             analyzer,
        #             df,
        #             initial_equity=30,
        #             leverage=1000,
        #             risk_fraction=0.02,
        #             on_event=logger,  # <— activar callback
        #             verbose=1,  # 0: silencio, 1: heartbeat cada N barras, 2: +OPEN/CLOSE
        #             log_every_bars=288,
        #             proba_threshold_day=0.6,
        #             proba_threshold_night=0.6,
        #             n_keep_for_fit=n,
        #             atr_mult_tp=4,
        #             atr_mult_sl=3,
        #             retrain_each_bar=False,
        #             retrain_every_bars=288,
        #             suppress_fit_stdout=False,
        #             early_exit_threshold=0.75,
        #             n_estimators=1000,
        #             start_date="2023-01-01",
        #             end_date=end_date,
        #             m2_model=None,
        #             enable_withdrawals=False
        #     )
        #
        #     try:
        #         results = sim.run(test_start="2025-03-07", test_end="2025-03-10")
        #         final_winrate = results['metrics']['winrate']
        #         print("DBG sim.run returned", results["final_equity"])
        #         print("wins", results["wins"])
        #         print("losses", results["losses"])
        #         print("max_consec_wins", results["metrics"]["max_consec_wins"])
        #         print("max_consec_losses", results["metrics"]["max_consec_losses"])
        #     except Exception as e:
        #         print(f"[WARN] n_keep_for_fit={n} falló: {e}")
        #         final_winrate = None
        #
        #     equities_por_n[n] = final_winrate
        #     print(f"n_keep_for_fit={n} -> winrate={final_winrate}")
        #
        #     if final_winrate is not None and isfinite(final_winrate) and final_winrate > mejor_equity:
        #         mejor_equity = final_winrate
        #         mejor_n = n
        #
        # # Resumen final
        # print("\nResumen:")
        # for n, eq in sorted(equities_por_n.items()):
        #     print(f"  n={n:5d}  winrate={eq}")
        #
        # if mejor_n is not None:
        #     print(f"\nMejor n_keep_for_fit = {mejor_n} con winrate = {mejor_equity}")
        # else:
        #     print("\nNo hubo corridas válidas (todos devolvieron None/NaN o fallaron).")

        # # # =========== get bet best dataset testing estimators
        # from math import isfinite
        #
        # mejor_n = None
        # mejor_equity = float("-inf")
        # equities_por_n = {}
        #
        # for n in range(70000, 200001, 10000):
        #     sim = TradeSimulator(
        #         analyzer,
        #         df,
        #         initial_equity=30,
        #         leverage=1000,
        #         risk_fraction=0.02,
        #         on_event=logger,  # <— activar callback
        #         verbose=1,  # 0: silencio, 1: heartbeat cada N barras, 2: +OPEN/CLOSE
        #         log_every_bars=288,
        #         proba_threshold_day=0.6,
        #         proba_threshold_night=0.6,
        #         n_keep_for_fit=n,
        #         atr_mult_tp=4,
        #         atr_mult_sl=3,
        #         retrain_each_bar=False,
        #         retrain_every_bars=288,
        #         suppress_fit_stdout=False,
        #         early_exit_threshold=0.75,
        #         n_estimators=1000,
        #         start_date="2023-01-01",
        #         end_date=end_date,
        #         m2_model=None,
        #         enable_withdrawals=False
        #     )
        #
        #     try:
        #         results = sim.run(test_start="2025-01-30", test_end="2025-02-04")
        #         final_winrate = results['metrics']['winrate']
        #         print("DBG sim.run returned", results["final_equity"])
        #         print("wins", results["wins"])
        #         print("losses", results["losses"])
        #         print("max_consec_wins", results["metrics"]["max_consec_wins"])
        #         print("max_consec_losses", results["metrics"]["max_consec_losses"])
        #     except Exception as e:
        #         print(f"[WARN] Estimators={n} falló: {e}")
        #         final_winrate = None
        #
        #     equities_por_n[n] = final_winrate
        #     print(f"Estimators={n} -> winrate={final_winrate}")
        #
        #     if final_winrate is not None and isfinite(final_winrate) and final_winrate > mejor_equity:
        #         mejor_equity = final_winrate
        #         mejor_n = n
        #
        # # Resumen final
        # print("\nResumen:")
        # for n, eq in sorted(equities_por_n.items()):
        #     print(f"  n={n:5d}  winrate={eq}")
        #
        # if mejor_n is not None:
        #     print(f"\nMejor n_keep_for_fit = {mejor_n} con winrate = {mejor_equity}")
        # else:
        #     print("\nNo hubo corridas válidas (todos devoFlvieron None/NaN o fallaron).")
        # #
        # # # =========== get_best_calibration_model
        # #     start=65_000,
        # #     start=65_000,
        # # analyzer.get_best_calibration_model(display_graphic=True)
        # #
        # # # =============best dataset size:
        # # df_results = analyzer.sweep_nkeep(
        # #     start=65_000,        # analyzer.get_best_calibration_model(display_graphic=True)
        # #     stop=105_000,
        # #     step=1000,
        # #     model_type="rf",
        # #     model_kwargs={"n_estimators": 700, "max_features": "sqrt", "random_state": 42, "n_jobs": -1},
        # #     calibrate_method="beta",
        # #     export_csv_path="bestDataset_65k-105k_beta_700trees_trader_features.csv",
        # #     sort_by="logloss"  # alternativas: 'logloss', 'brier', 'ece', 'macro_f1'
        # # )
        # # print(df_results.head(10))


    else:
        print(f"El archivo {filename} no existe")
