# -*- coding: utf-8 -*-
"""Train XAU/USD 15-minute and next-session forecasts from recent intraday bars.

The command-line entry point runs xau_15m_kaggle_train.py (the active neural models: LSTM, GRU,
Transformer, Transformer-MT, SAG-LSTM) on the latest 20,000 15-minute rows.

Run:
    python train_predict.py --data XAU_15m_data.csv --rows 20000
"""

import ast
import io
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FILE_2014_2026 = DATA_DIR / "XAU_USD Historical Data  2014-2026.csv"
FILE_1995_2013 = DATA_DIR / "XAU_USD Historical Data 1995-2013.csv"
FILE_1980_1994 = Path("C:/Users/User/Downloads/XAU_USD Historical Data 1980_1994.csv")

MODELS_DIR = OUT_DIR / "models"            # saved models for --predict (no retraining)
FORECAST_LOG = OUT_DIR / "forecast_log.csv"  # every next-day forecast made, to compare with what happened


def next_trading_day(date):
    """Next weekday after `date` (the forecast target)."""
    return (pd.Timestamp(date) + pd.offsets.BDay(1)).normalize()

TEST_SIZE = 0.10          # backtest hold-out (Phase A only)
VAL_SIZE = 0.10           # validation slice of the Phase A training data (GA fitness)
BATCH_SIZE = 128
EPOCHS = 84               # scaled from 57 based on 47% dataset size increase (8250 -> 12109 rows)
LEARNING_RATE = 0.001
RANDOM_SEED = 42
HIT_TOLERANCE = 0.005     # prediction counted as a "hit" if within 0.5% of actual
WINDOW_SIZE = 20          # days of history seen by the sliding-window models
EEMD_WINDOW = 64          # trailing days decomposed by EEMD for each forecast day (causal: no future data)
EEMD_TRIALS = 20          # noise-added EMD runs averaged per decomposition
EEMD_NOISE = 0.2          # white-noise std as a fraction of the segment's std (Wu & Huang default)

# Regularization for the neural models
WEIGHT_DECAY = 1e-3       # AdamW decoupled weight decay (L2-style penalty on the weights)
GRAD_CLIP = 1.0           # max gradient norm

# Every neural model also classifies tomorrow's close as Up/Down: loss = MSE(returns) + CLS_WEIGHT * BCE(up/down)
CLS_WEIGHT = 0.5

# Ensembles use only the ENSEMBLE_TOP_K models with the lowest average $ RMSE on the VALIDATION period
# (choosing them on the test period would make the ensembles' test scores look better than they are)
ENSEMBLE_TOP_K = 4

# Market regimes (Gaussian mixture on backward-looking indicators; fitted on training rows only)
REGIME_NAMES = ["Bearish", "Sideways", "Bullish"]
REGIME_COLUMNS = ["Regime_Bearish", "Regime_Sideways", "Regime_Bullish"]
REGIME_INPUTS = ["SMA_30_Distance", "EMA_30_Distance", "Momentum_Ratio", "Volatility_14", "ATR_Ratio", "Return_20"]
MIN_REFIT_EPOCHS = 3      # the 100% refit trains for max(best epoch, this)

# LightGBM (one regressor per target, early-stopped on validation)
LGB_PARAMS = dict(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                  min_split_gain=0.0, random_state=RANDOM_SEED, verbose=-1)

# Chronos-Bolt (pretrained time-series foundation model, used zero-shot)
CHRONOS_MODEL = "amazon/chronos-bolt-small"
CHRONOS_CONTEXT = 512     # days of history given to Chronos for each forecast
CHRONOS_BATCH = 512

# Disabled after the backtest (they ranked 10th-17th by average RMSE). Set to True to use them again.
USE_LIGHTGBM = False
USE_CHRONOS = False

# Genetic algorithm (SAG-LSTM hyperparameters)
POPULATION_SIZE = 5
GENERATIONS = 5
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.7
HIDDEN_SIZE_CHOICES = [64, 128, 256]
LEARNING_RATE_RANGE = [0.0001, 0.01]
EPOCHS_CHOICES = [5, 10, 15]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. LOAD DATA
# ============================================================

# Latest daily bars not yet in the CSV export (newest first, same format as investing.com)
NEW_FINANCIAL_DATA = """Date,Price,Open,High,Low,Change %
9/30/2026,"4,157.34","4,182.80","4,219.46","4,147.47",-0.61%

9/29/2026,"4,182.80","4,113.87","4,185.42","4,113.57",+1.34%

9/28/2026,"4,127.38","4,277.90","4,280.56","4,110.95",-3.73%

9/25/2026,"4,287.25","4,279.12","4,316.81","4,254.45",+0.20%

9/24/2026,"4,278.52","4,287.38","4,304.95","4,244.63",-0.21%

9/23/2026,"4,287.51","4,357.02","4,371.45","4,274.73",-1.56%

9/22/2026,"4,355.48","4,343.57","4,378.25","4,291.42",+0.27%

9/21/2026,"4,343.56","4,383.44","4,387.51","4,322.63",-0.79%

9/18/2026,"4,378.17","4,342.73","4,399.77","4,334.34",0.85%

9/17/2026,"4,341.30","4,263.97","4,382.63","4,257.55",1.82%

9/16/2026,"4,263.50","4,294.50","4,368.04","4,235.08",-0.71%

9/15/2026,"4,287.60","4,288.20","4,317.39","4,261.51",-0.02%

9/14/2026,"4,288.29","4,332.70","4,355.65","4,253.30",-1.39%

9/11/2026,"4,348.72","4,319.31","4,402.70","4,295.55",0.73%

"""


def clean_price_frame(frame):
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    for col in ["Price", "Open", "High", "Low"]:
        frame[col] = frame[col].astype(str).str.replace(",", "").astype(float)
    frame["Change %"] = frame["Change %"].astype(str).str.replace("%", "").astype(float)
    return frame[["Date", "Price", "Open", "High", "Low", "Change %"]]


def load_data():
    df_new = clean_price_frame(pd.read_csv(io.StringIO(NEW_FINANCIAL_DATA)))
    df_2014_2026 = clean_price_frame(pd.read_csv(FILE_2014_2026))
    df_1995_2013 = clean_price_frame(pd.read_csv(FILE_1995_2013))
    df_1980_1994 = clean_price_frame(pd.read_csv(FILE_1980_1994))

    # New rows first so that they win when a date is duplicated (e.g. 9/11/2026)
    df = pd.concat([df_new, df_2014_2026, df_1995_2013, df_1980_1994], ignore_index=True)
    n_before = len(df)
    df = df.drop_duplicates(subset="Date", keep="first")
    print(f"Removed {n_before - len(df)} duplicate date row(s)")

    df = df.sort_values("Date").reset_index(drop=True)
    print("Rows:", len(df), "| Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
    return df


# ============================================================
# 2. FEATURE ENGINEERING (same as original notebook)
# ============================================================

def build_features(df):
    new_df = df.copy()

    new_df["Previous_Day_Price"] = new_df["Price"].shift(1)
    
    # This is a known-at-the-close feature. Only the target columns below use
    # shift(-1), so the final available bar remains usable for inference.
    new_df["Return"] = new_df["Price"].pct_change()

    new_df["Log_Return"] = np.log(new_df["Price"] / new_df["Previous_Day_Price"])

    for w in (7, 14, 30):
        new_df[f"SMA_{w}"] = new_df["Price"].rolling(window=w).mean()
        new_df[f"EMA_{w}"] = new_df["Price"].ewm(span=w, adjust=False).mean()

    new_df["Volatility_7"] = new_df["Return"].rolling(window=7).std()
    new_df["Volatility_14"] = new_df["Return"].rolling(window=14).std()

    # RSI (14)
    delta = new_df["Price"].diff()
    avg_gain = delta.clip(lower=0).rolling(window=14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    new_df["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    # MACD
    ema_12 = new_df["Price"].ewm(span=12, adjust=False).mean()
    ema_26 = new_df["Price"].ewm(span=26, adjust=False).mean()
    new_df["MACD"] = ema_12 - ema_26
    new_df["MACD_Signal"] = new_df["MACD"].ewm(span=9, adjust=False).mean()
    new_df["MACD_Histogram"] = new_df["MACD"] - new_df["MACD_Signal"]

    # ATR (14)
    previous_close = new_df["Price"].shift(1)
    true_range = pd.concat([
        new_df["High"] - new_df["Low"],
        abs(new_df["High"] - previous_close),
        abs(new_df["Low"] - previous_close),
    ], axis=1).max(axis=1)
    new_df["ATR"] = true_range.rolling(window=14).mean()

    # Bollinger Bands (20)
    bb_middle = new_df["Price"].rolling(window=20).mean()
    bb_std = new_df["Price"].rolling(window=20).std()
    new_df["BB_Middle"] = bb_middle
    new_df["BB_Upper"] = bb_middle + 2 * bb_std
    new_df["BB_Lower"] = bb_middle - 2 * bb_std

    new_df["Momentum"] = new_df["Price"].diff(periods=7)
    new_df["Return_20"] = new_df["Price"].pct_change(20)  # used by the regime detector

    new_df = new_df.iloc[30:].reset_index(drop=True)

    # Cyclical date features
    month = new_df["Date"].dt.month
    dow = new_df["Date"].dt.dayofweek
    doy = new_df["Date"].dt.dayofyear
    new_df["Month_Sin"] = np.sin(2 * np.pi * month / 12)
    new_df["Month_Cos"] = np.cos(2 * np.pi * month / 12)
    new_df["DayOfWeek_Sin"] = np.sin(2 * np.pi * dow / 7)
    new_df["DayOfWeek_Cos"] = np.cos(2 * np.pi * dow / 7)
    new_df["DayOfYear_Sin"] = np.sin(2 * np.pi * doy / 365)
    new_df["DayOfYear_Cos"] = np.cos(2 * np.pi * doy / 365)

    # Scale-invariant transforms
    new_df["Open_Return"] = new_df["Open"].pct_change()

    new_df["High_Return"] = new_df["High"].pct_change()

    new_df["Low_Return"] = new_df["Low"].pct_change()
    
    new_df["Previous_Day_Return"] = new_df["Previous_Day_Price"].pct_change()

    for col in ["SMA_7", "SMA_14", "SMA_30", "EMA_7", "EMA_14", "EMA_30", "BB_Middle"]:
        new_df[f"{col}_Distance"] = (new_df["Price"] - new_df[col]) / new_df[col]

    new_df["MACD_Relative"] = new_df["MACD"] / new_df["Price"]
    new_df["MACD_Signal_Relative"] = new_df["MACD_Signal"] / new_df["Price"]
    new_df["MACD_Hist_Relative"] = new_df["MACD_Histogram"] / new_df["Price"]
    new_df["ATR_Ratio"] = new_df["ATR"] / new_df["Price"]
    new_df["BB_Position"] = (new_df["Price"] - new_df["BB_Lower"]) / (new_df["BB_Upper"] - new_df["BB_Lower"])
    new_df["BB_Width"] = (new_df["BB_Upper"] - new_df["BB_Lower"]) / new_df["BB_Middle"]
    new_df["Momentum_Ratio"] = new_df["Momentum"] / new_df["Price"]

    new_df["RSI_Raw"] = new_df["RSI"]
    new_df["RSI"] = new_df["RSI"] / 100
    return new_df


SCALED_FEATURE_COLUMNS = [
    "Volatility_7", "Volatility_14",
    "SMA_7_Distance", "SMA_14_Distance", "SMA_30_Distance",
    "EMA_7_Distance", "EMA_14_Distance", "EMA_30_Distance",
    "BB_Middle_Distance",
    "MACD_Relative", "MACD_Signal_Relative", "MACD_Hist_Relative",
    "ATR_Ratio", "BB_Width", "Momentum_Ratio",
]
UNSCALED_FEATURE_COLUMNS = [
    "Return", "Log_Return", "RSI",
    "Month_Sin", "Month_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos", "DayOfYear_Sin", "DayOfYear_Cos",
    "BB_Position", "Open_Return", "High_Return", "Low_Return", "Previous_Day_Return",
]
# Base features (known before regimes are fitted); the regime probabilities are appended last, so in every
# model input the final 3 values of each day are P(Bearish), P(Sideways), P(Bullish)
BASE_FEATURE_COLUMNS = SCALED_FEATURE_COLUMNS + UNSCALED_FEATURE_COLUMNS
UNSCALED_FEATURE_COLUMNS = UNSCALED_FEATURE_COLUMNS + REGIME_COLUMNS
FEATURE_COLUMNS = SCALED_FEATURE_COLUMNS + UNSCALED_FEATURE_COLUMNS


def fit_regimes(frame):
    """
    Gaussian mixture with 3 components on standardized REGIME_INPUTS of `frame` (training rows only).
    Components are named by their average 20-day return: lowest = Bearish, middle = Sideways, highest = Bullish.
    """
    from sklearn.mixture import GaussianMixture

    scaler = StandardScaler().fit(frame[REGIME_INPUTS])
    gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=RANDOM_SEED)
    labels = gmm.fit_predict(scaler.transform(frame[REGIME_INPUTS]))
    mean_ret = [frame["Return_20"].values[labels == k].mean() for k in range(3)]
    order = np.argsort(mean_ret)  # component index for Bearish, Sideways, Bullish
    return {"scaler": scaler, "gmm": gmm, "order": order}


def regime_probabilities(regimes, frame):
    """(rows, 3) probabilities in REGIME_NAMES order."""
    proba = regimes["gmm"].predict_proba(regimes["scaler"].transform(frame[REGIME_INPUTS]))
    return proba[:, regimes["order"]]


def add_regimes(frame, regimes):
    return frame.assign(**dict(zip(REGIME_COLUMNS, regime_probabilities(regimes, frame).T)))

# ============================================================
# 3. NEXT-DAY TARGETS
# ============================================================
#
# Every target is the next day's value relative to TODAY'S CLOSE:
#     Next_X = X(t+1) / Price(t) - 1
# so predicted price = Price(t) * (1 + predicted return)
# ============================================================

PRICE_TYPES = ["Open", "High", "Low", "Close"]
SOURCE_COLUMN = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Price"}
TARGET_COLUMNS = [f"Next_{p}" for p in PRICE_TYPES]


def add_targets(new_df):
    for p in PRICE_TYPES:
        new_df[f"Next_{p}"] = new_df[SOURCE_COLUMN[p]].shift(-1) / new_df["Price"] - 1
    return new_df


# ============================================================
# 4. MODELS (unchanged from original notebook)
# ============================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        return self.fc(output[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, output_size=1, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(d_model // 2, output_size))

    def forward(self, x):
        x = self.positional_encoding(self.input_projection(x))
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class UpDownMixin:
    """Adds the joint regression + Up/Down classification loss to a model that implements forward_all(x)."""

    def multitask_loss(self, x, y):
        returns, up_logit = self.forward_all(x)
        up = (y[:, self.close_index] > self.up_threshold).float()
        return F.mse_loss(returns, y) + self.cls_weight * F.binary_cross_entropy_with_logits(up_logit, up)


class WithUpDownHead(UpDownMixin, nn.Module):
    """
    Gives any regression model an Up/Down classification output: the wrapped model is built with one extra
    output unit, which is the logit of P(tomorrow's close > today's close). forward() returns only the
    O/H/L/C returns, so the rest of the pipeline is unchanged.
    """

    def __init__(self, base, output_size, cls_weight=CLS_WEIGHT):
        super().__init__()
        self.base, self.output_size, self.cls_weight = base, output_size, cls_weight
        self.close_index = output_size - 1
        self.up_threshold = 0.0  # scaled value of a 0% close return; set from the target scaler before training

    def forward_all(self, x):
        out = self.base(x)
        return out[:, :self.output_size], out[:, self.output_size]

    def forward(self, x):
        return self.forward_all(x)[0]


class TransformerMultiTask(UpDownMixin, nn.Module):
    """
    Multi-task, regime-gated Transformer (same encoder as TimeSeriesTransformer).

    Two outputs learned together:
      * Regression:     next-day Open/High/Low/Close returns, from 3 regime-specific expert heads
                        (Bearish / Sideways / Bullish) blended by the day's regime probabilities
      * Classification: probability that tomorrow's close is above today's close (Up vs Down)
    Loss = MSE(returns) + CLS_WEIGHT * BCE(up/down). The regime probabilities are the last 3 input values.
    """

    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, output_size=4, n_regimes=3, dropout=0.1,
                 cls_weight=CLS_WEIGHT):
        super().__init__()
        self.n_regimes, self.cls_weight = n_regimes, cls_weight
        self.close_index = output_size - 1
        self.up_threshold = 0.0  # scaled value of a 0% close return; set from the target scaler before training
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def head(out):
            return nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d_model // 2, out))

        self.regime_heads = nn.ModuleList([head(output_size) for _ in range(n_regimes)])
        self.direction_head = head(1)

    def forward_all(self, x):
        h = self.encoder(self.positional_encoding(self.input_projection(x)))[:, -1, :]
        gate = x[:, -1, -self.n_regimes:]                                    # regime probabilities
        experts = torch.stack([head(h) for head in self.regime_heads], dim=1)  # (batch, regimes, outputs)
        returns = (gate.unsqueeze(-1) * experts).sum(dim=1)
        return returns, self.direction_head(h).squeeze(-1)

    def forward(self, x):
        return self.forward_all(x)[0]


def up_probability(model, X):
    """P(tomorrow's close > today's close) from a multi-task model; X is an array or a DataLoader."""
    model.eval()
    with torch.no_grad():
        if isinstance(X, DataLoader):
            logits = [model.forward_all(xb.to(device))[1].cpu().numpy() for xb, _ in X]
            return 1 / (1 + np.exp(-np.concatenate(logits)))
        logits = model.forward_all(torch.tensor(X, dtype=torch.float32).to(device))[1].cpu().numpy()
        return 1 / (1 + np.exp(-logits))


class PatchTST(nn.Module):
    def __init__(self, input_size, window_size=20, patch_length=5, stride=5, d_model=128, nhead=8,
                 num_layers=3, output_size=1, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = ((window_size - patch_length) // stride) + 1
        self.patch_projection = nn.Linear(patch_length * input_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(self.num_patches * d_model, d_model), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, output_size))

    def forward(self, x):
        batch_size = x.size(0)
        patches = [
            x[:, start:start + self.patch_length, :].reshape(batch_size, -1)
            for start in range(0, self.window_size - self.patch_length + 1, self.stride)
        ]
        patches = self.patch_projection(torch.stack(patches, dim=1))
        patches = patches + self.position_embedding[:, :patches.size(1)]
        return self.head(self.transformer(patches))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        residual = self.skip(x)
        x = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        x = torch.sigmoid(self.gate(x)) * x
        return self.norm(x + residual)


class SAG_LSTM(nn.Module):
    """Self-Attention Gated LSTM."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.attention_linear = nn.Linear(hidden_size, 1)
        self.grn = GatedResidualNetwork(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(torch.tanh(self.attention_linear(lstm_out)), dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        return self.fc(self.grn(context_vector))


class CompactTFT(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.variable_selector = nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())
        self.feature_projection = nn.Linear(input_size, d_model)
        self.grn = GatedResidualNetwork(input_size=d_model, hidden_size=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(d_model // 2, output_size))

    def forward(self, x):
        x = x * self.variable_selector(x)
        x = self.grn(self.feature_projection(x))
        x = self.temporal_encoder(x)
        return self.output_head(x[:, -1, :])


class CNN_BiLSTM_Attention(nn.Module):
    """
    CNN-BiLSTM-Attention (Input -> Conv2D -> BiLSTM -> Attention -> Dense -> Output).

    The input window (WINDOW_SIZE days x features) is treated as a one-channel image:
      * Conv2D extracts local patterns across neighbouring days and neighbouring features
      * BiLSTM reads the convolved sequence forwards (A_t) and backwards (B_t), both
        halves are concatenated per time step (Y_t = f(w5*A_t + w6*B_t) in the paper)
      * attention scores every day in the window and builds a weighted context vector
      * dense layers map the context vector to the 4 next-day returns
    """

    def __init__(self, input_size, window_size, conv_filters=32, kernel_size=3, hidden_size=64,
                 num_layers=1, output_size=1, dropout=0.2):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_filters, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # pool over features only, keep every time step
            nn.Dropout(dropout),
        )
        conv_features = conv_filters * (input_size // 2)
        self.bilstm = nn.LSTM(conv_features, hidden_size, num_layers, batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.attention_score = nn.Linear(2 * hidden_size, 1, bias=False)
        self.dense = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(hidden_size, output_size))
        self.last_attention = None

    def forward(self, x):
        # x: (batch, window, features) -> (batch, 1, window, features)
        x = self.conv(x.unsqueeze(1))                        # (batch, filters, window, features/2)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)       # (batch, window, filters * features/2)
        h, _ = self.bilstm(x)                                # (batch, window, 2 * hidden)
        scores = self.attention_score(torch.tanh(self.attention(h)))
        weights = torch.softmax(scores, dim=1)               # (batch, window, 1)
        self.last_attention = weights.detach().squeeze(-1)
        context = torch.sum(h * weights, dim=1)              # (batch, 2 * hidden)
        return self.dense(context)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, values):
        scores = self.score(torch.tanh(
            self.query_projection(query).unsqueeze(2) + self.key_projection(values).unsqueeze(1)
        )).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, values)
        return context, attention_weights


class LSTMAttentionLSTM(nn.Module):
    """Encoder-attention-decoder model for the next-day OHLC returns and direction logit."""

    def __init__(self, input_size, hidden_size, output_steps, output_size=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_steps = output_steps
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.decoder = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.last_attention_weights = None

    def forward(self, x):
        encoder_outputs, (hidden, _) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).expand(-1, self.output_steps, -1)
        context, attention_weights = self.attention(decoder_input, encoder_outputs)
        decoder_input = torch.cat([decoder_input, context], dim=-1)
        decoder_outputs, _ = self.decoder(decoder_input)
        predictions = self.fc(self.dropout(decoder_outputs))
        self.last_attention_weights = attention_weights.detach()
        return predictions[:, 0, :] if self.output_steps == 1 else predictions


class LSTMTransformer(nn.Module):
    """
    LSTM-Transformer (encoder-decoder).

    Encoder: Input -> linear embedding + positional encoding -> Multi-Head Attention -> Add&Norm
             -> FFN -> Add&Norm   (standard post-norm Transformer encoder layer)
    Decoder: LSTM over the encoder output -> Cross-Attention (queries = LSTM states,
             keys/values = encoder output) -> Add&Norm -> FFN -> Add&Norm
    Output:  Linear on the last time step. The paper's Softmax is for classification; this is a
             regression of 4 next-day returns, so the output is left linear.
    """

    def __init__(self, input_size, d_model=64, nhead=4, num_encoder_layers=2, lstm_hidden=64,
                 output_size=1, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   dropout=dropout, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.decoder_lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True)
        self.lstm_to_model = nn.Linear(lstm_hidden, d_model) if lstm_hidden != d_model else nn.Identity()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, output_size)

    def forward(self, x):
        # Encoder
        memory = self.encoder(self.positional_encoding(self.input_projection(x)))   # (batch, window, d_model)
        # Decoder
        q, _ = self.decoder_lstm(memory)
        q = self.lstm_to_model(q)
        attn, _ = self.cross_attention(q, memory, memory)
        h = self.norm1(q + self.dropout(attn))                  # Add & Norm
        h = self.norm2(h + self.dropout(self.ffn(h)))           # FFN -> Add & Norm
        return self.output(h[:, -1, :])


class CausalConvBlock(nn.Module):
    """TCN residual block: two dilated causal 1-D convolutions + residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # padding is applied on both sides; chopping the right side keeps each output causal (no look-ahead)
        out = self.dropout(torch.relu(self.conv1(x)[..., :-self.pad]))
        out = self.dropout(torch.relu(self.conv2(out)[..., :-self.pad]))
        return torch.relu(out + self.downsample(x))


class EEMD_TCN_GRU(nn.Module):
    """
    EEMD-TCN-GRU.

    The close-price window is decomposed by EEMD (done beforehand, see eemd_components):
      * IMF 1  = highest-frequency component          -> TCN branch
      * IMF 2  = remaining lower-frequency signal     -> GRU branch
                 (sum of the other IMFs + residue, so IMF 1 + IMF 2 = original signal)
    Each branch also receives the engineered features of every day in the window.
    The two branch outputs are added (the "Add" node) to give the prediction.

    Input x: (batch, window, 2 + features) with channel 0 = IMF 1, channel 1 = IMF 2.
    """

    def __init__(self, input_size, tcn_channels=32, tcn_levels=4, kernel_size=3, gru_hidden=64,
                 output_size=1, dropout=0.2):
        super().__init__()
        blocks, in_ch = [], input_size + 1
        for level in range(tcn_levels):  # dilations 1, 2, 4, 8 -> receptive field covers the window
            blocks.append(CausalConvBlock(in_ch, tcn_channels, kernel_size, 2 ** level, dropout))
            in_ch = tcn_channels
        self.tcn = nn.Sequential(*blocks)
        self.tcn_out = nn.Linear(tcn_channels, output_size)
        self.gru = nn.GRU(input_size + 1, gru_hidden, batch_first=True)
        self.gru_out = nn.Linear(gru_hidden, output_size)

    def forward(self, x):
        imf1, imf2, feats = x[..., 0:1], x[..., 1:2], x[..., 2:]
        tcn_h = self.tcn(torch.cat([imf1, feats], dim=-1).transpose(1, 2))   # (batch, channels, window)
        gru_h, _ = self.gru(torch.cat([imf2, feats], dim=-1))                # (batch, window, hidden)
        return self.tcn_out(tcn_h[:, :, -1]) + self.gru_out(gru_h[:, -1, :])  # Add


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class HoltWintersDecomposition(nn.Module):
    """
    Multiplicative Holt-Winters decomposition with learnable smoothing (alpha = level, gamma = seasonal),
    one pair per input channel. Requires strictly positive inputs.
        L_t = alpha * X_t / S_{t-1} + (1 - alpha) * L_{t-1}
        S_t = gamma * X_t / L_t     + (1 - gamma) * S_{t-1}
        Y_t = X_t / (L_t * S_t)     (deseasonalized, de-levelled series)
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha_raw = nn.Parameter(torch.zeros(channels))
        self.gamma_raw = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha, gamma = torch.sigmoid(self.alpha_raw), torch.sigmoid(self.gamma_raw)
        level, seasonal = x[:, 0, :], torch.ones_like(x[:, 0, :])
        levels, seasonals = [level], [seasonal]
        for t in range(1, x.size(1)):
            current = x[:, t, :]
            level = alpha * current / (seasonal + self.eps) + (1 - alpha) * level
            seasonal = gamma * current / (level + self.eps) + (1 - gamma) * seasonal
            levels.append(level)
            seasonals.append(seasonal)
        levels, seasonals = torch.stack(levels, dim=1), torch.stack(seasonals, dim=1)
        return x / (levels * seasonals + self.eps), levels, seasonals


class HelformerAttentionBlock(nn.Module):
    """Multi-head self-attention with input/output projections and Add & Norm."""

    def __init__(self, input_dim, num_heads=4, head_size=16, dropout=0.1):
        super().__init__()
        hidden = num_heads * head_size
        self.input_projection = nn.Linear(input_dim, hidden)
        self.attention = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.output_projection = nn.Linear(hidden, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        h = self.input_projection(x)
        attn, _ = self.attention(h, h, h)
        return self.norm(x + self.dropout(self.output_projection(attn)))


class HelformerLSTMBlock(nn.Module):
    """LSTM (replacing the Transformer FFN) -> Mish -> projection -> Add & Norm."""

    def __init__(self, input_dim, hidden_size=30, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.mish = Mish()
        self.projection = nn.Linear(hidden_size, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.norm(x + self.dropout(self.projection(self.mish(h))))


class Helformer(nn.Module):
    """
    Helformer: Holt-Winters decomposition -> Multi-head attention -> LSTM block -> Dense.

    Adapted to this data:
      * Holt-Winters is multiplicative, so it is applied to the POSITIVE price channels only:
        the window's Open/High/Low/Close divided by the last day's close (values around 1.0).
        The standardized features (often <= 0) would make its divisions blow up.
      * The deseasonalized prices are centred as (Y - 1) * 100 and concatenated with the
        29 standardized features before the attention block.
      * The dense layer outputs the 4 next-day returns (Open, High, Low, Close).

    Input x: (batch, window, 4 + features), channels 0-3 = O/H/L/C relative to the last close.
    """

    def __init__(self, input_size, price_channels=4, num_heads=4, head_size=16, lstm_hidden=30,
                 output_size=1, dropout=0.1):
        super().__init__()
        self.price_channels = price_channels
        model_dim = price_channels + input_size
        self.decomposition = HoltWintersDecomposition(price_channels)
        self.attention = HelformerAttentionBlock(model_dim, num_heads, head_size, dropout)
        self.lstm_block = HelformerLSTMBlock(model_dim, lstm_hidden, dropout)
        self.fc = nn.Linear(model_dim, output_size)

    def forward(self, x):
        prices, feats = x[..., :self.price_channels], x[..., self.price_channels:]
        deseasonalized, _, _ = self.decomposition(prices)
        h = torch.cat([(deseasonalized - 1.0) * 100.0, feats], dim=-1)
        h = self.lstm_block(self.attention(h))
        return self.fc(h[:, -1, :])


# Models that take a WINDOW_SIZE-day sliding window instead of a single day's features
WINDOW_MODELS = {"CNN-BiLSTM-Attn", "LSTM-Transformer", "LSTM-Attention-LSTM"}
# Models that take the sliding window plus the 2 EEMD components
EEMD_MODELS = {"EEMD-TCN-GRU"}


# Models that take the sliding window plus the window's O/H/L/C relative to the last close
PRICE_WINDOW_MODELS = {"Helformer"}


# Neural models used for training and prediction. The commented-out ones were disabled after the
# backtest (they ranked 10th-17th by average RMSE); uncomment a name here AND in build_models to use it again.
ACTIVE_NEURAL_MODELS = [
    # "CNN-BiLSTM-Attn",
    # "LSTM-Transformer",
    # "LSTM-Attention-LSTM",
    # "EEMD-TCN-GRU",
    # "Helformer",
    "LSTM",
    "GRU",
    "Transformer",
    # "PatchTST",
    # "TFT",
    "Transformer-MT",
    "SAG-LSTM",
]


def input_kind(name):
    if name in EEMD_MODELS:
        return "eemd"
    if name in PRICE_WINDOW_MODELS:
        return "prices"
    return "window" if name in WINDOW_MODELS else "row"


def build_models(input_size, output_size, sag_hidden_size):
    """
    Fresh, identically-seeded instances of all neural models. Each predicts the O/H/L/C returns and also
    classifies tomorrow's close as Up/Down (base models are built with one extra output for the Up logit).
    """
    set_seed()
    regression_size = output_size
    output_size = regression_size + 1
    models = {
        # Disabled after the backtest (ranked 10th-17th by average RMSE):
        # "CNN-BiLSTM-Attn": CNN_BiLSTM_Attention(input_size=input_size, window_size=WINDOW_SIZE,
        #                                         output_size=output_size),
        "LSTM-Transformer": LSTMTransformer(input_size=input_size, output_size=output_size),
        "LSTM-Attention-LSTM": LSTMAttentionLSTM(input_size=input_size, hidden_size=128, output_steps=1,
                              output_size=output_size),
        # "EEMD-TCN-GRU": EEMD_TCN_GRU(input_size=input_size, output_size=output_size),
        # "Helformer": Helformer(input_size=input_size, output_size=output_size),
        "LSTM": LSTMModel(input_size=input_size, output_size=output_size),
        "GRU": GRUModel(input_size=input_size, output_size=output_size),
        "Transformer": TimeSeriesTransformer(input_size=input_size, output_size=output_size),
        # Sequence length is 1 (single row input), so PatchTST uses one patch of length 1
        # "PatchTST": PatchTST(input_size=input_size, window_size=1, patch_length=1, stride=1,
        #                      output_size=output_size),
        # "TFT": CompactTFT(input_size=input_size, output_size=output_size),
        "SAG-LSTM": SAG_LSTM(input_size=input_size, hidden_size=sag_hidden_size, output_size=output_size),
    }
    models = {name: WithUpDownHead(m, regression_size) for name, m in models.items()}
    # Transformer-MT has its own classification head (and regime-gated regression heads)
    models["Transformer-MT"] = TransformerMultiTask(input_size=input_size, output_size=regression_size)
    return {name: models[name] for name in ACTIVE_NEURAL_MODELS}


# ============================================================
# 5. TRAINING / PREDICTION
# ============================================================

def make_loader(X, y, shuffle):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_model(model, train_loader, val_loader, epochs, learning_rate, target_scaler, verbose=True, name=""):
    """
    Train with MSE on scaled returns. val_loader may be None (full-data refit).
    Validation metrics are recorded, but do not change the configured training duration or weights.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "best_epoch": epochs}

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if hasattr(model, "multitask_loss"):
                loss = model.multitask_loss(X_batch, y_batch)
            else:
                loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()
        history["train_loss"].append(total / len(train_loader))

        msg = f"  [{name}] Epoch {epoch + 1}/{epochs} | train loss {history['train_loss'][-1]:.5f}"

        if val_loader is not None:
            model.eval()
            total, preds, actuals = 0.0, [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    out = model(X_batch)
                    total += criterion(out, y_batch).item()
                    preds.append(out.cpu().numpy())
                    actuals.append(y_batch.cpu().numpy())
            history["val_loss"].append(total / len(val_loader))
            preds_raw = target_scaler.inverse_transform(np.concatenate(preds))
            actuals_raw = target_scaler.inverse_transform(np.concatenate(actuals))
            history["val_rmse"].append(float(np.sqrt(mean_squared_error(actuals_raw, preds_raw))))
            msg += f" | val loss {history['val_loss'][-1]:.5f} | val RMSE(return) {history['val_rmse'][-1]:.5f}"

        if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
            print(msg)
    return model, history


def predict_loader(model, loader):
    """Predictions (scaled) for every batch of a DataLoader, in order."""
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(X.to(device)).cpu().numpy() for X, _ in loader])


def fit_lightgbm(X_tr, y_tr, X_val, y_val):
    """One LGBMRegressor per target using the configured tree count; return Close loss curves."""
    import lightgbm as lgb

    models, best, curves = [], [], ([], [])
    for j in range(y_tr.shape[1]):
        evals = {}
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_tr, y_tr[:, j], eval_set=[(X_tr, y_tr[:, j]), (X_val, y_val[:, j])],
                  eval_names=["train", "valid"], eval_metric="l2",
                  callbacks=[lgb.record_evaluation(evals)])
        models.append(model)
        best.append(LGB_PARAMS["n_estimators"])
        if j == len(PRICE_TYPES) - 1:
            curves = (evals["train"]["l2"], evals["valid"]["l2"])
    return models, best, curves


def fit_lightgbm_classifier(X_tr, up_tr, X_val, up_val):
    """LGBMClassifier using the configured tree count for tomorrow's close Up (1) / Down (0)."""
    import lightgbm as lgb

    params = {k: v for k, v in LGB_PARAMS.items()}
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, up_tr, eval_set=[(X_val, up_val)], eval_names=["valid"], eval_metric="binary_logloss")
    return clf, LGB_PARAMS["n_estimators"]


def predict_lightgbm(models, X):
    return np.column_stack([m.predict(X, num_iteration=m.best_iteration_) for m in models])


_chronos_pipeline = None


def chronos_returns(df, dates):
    """
    Zero-shot Chronos-Bolt forecasts of the next day's Open/High/Low/Close for each date, each series forecast
    from its own last CHRONOS_CONTEXT values up to and including that date. Returned as returns vs that
    day's close, shape (len(dates), 4) in PRICE_TYPES order.
    """
    global _chronos_pipeline
    from chronos import BaseChronosPipeline

    if _chronos_pipeline is None:
        _chronos_pipeline = BaseChronosPipeline.from_pretrained(CHRONOS_MODEL, device_map=str(device),
                                                                torch_dtype=torch.float32)
    pos_of = {d: i for i, d in enumerate(df["Date"])}
    series = [torch.tensor(df[SOURCE_COLUMN[p]].values, dtype=torch.float32) for p in PRICE_TYPES]
    positions = [pos_of[pd.Timestamp(d)] for d in dates]
    contexts = [s[max(0, q - CHRONOS_CONTEXT + 1):q + 1] for q in positions for s in series]
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    chunks = []
    for start in range(0, len(contexts), CHRONOS_BATCH):
        q, _ = _chronos_pipeline.predict_quantiles(contexts[start:start + CHRONOS_BATCH], prediction_length=1,
                                                   quantile_levels=levels)
        chunks.append(q[:, 0, :].numpy())
    quantiles = np.concatenate(chunks).reshape(len(positions), len(PRICE_TYPES), len(levels))
    closes = df["Price"].values[positions]
    returns = quantiles[:, :, levels.index(0.5)] / closes[:, None] - 1
    # P(up) = 1 - F(today's close), F interpolated from the Close forecast's quantiles
    close_q = np.sort(quantiles[:, PRICE_TYPES.index("Close"), :], axis=1)
    p_up = np.array([1 - np.interp(c, qs, levels, left=0.05, right=0.95) for c, qs in zip(closes, close_q)])
    return returns, p_up


def classification_metrics(p_up, actual_up):
    """Up/Down classification scores for probabilities p_up against actual up days."""
    pred = p_up > 0.5
    tp = np.sum(pred & actual_up)
    return {
        "accuracy": float(np.mean(pred == actual_up) * 100),
        "auc": float(roc_auc_score(actual_up, p_up)) if 0 < actual_up.mean() < 1 and np.ptp(p_up) > 0 else None,
        "precision_up": float(tp / pred.sum() * 100) if pred.sum() else None,
        "recall_up": float(tp / actual_up.sum() * 100) if actual_up.sum() else None,
        "pred_up_share": float(pred.mean() * 100),
    }


def fit_stacking(val_returns, y_val, names):
    """
    Per target, a convex blend (weights >= 0, sum 1) of the models' validation returns plus a "Naive"
    zero-return column, minimising squared error. Returns {target: {name: weight}}.
    """
    from scipy.optimize import minimize

    weights = {}
    for j, p in enumerate(PRICE_TYPES):
        A = np.column_stack([val_returns[n][:, j] for n in names] + [np.zeros(len(y_val))])
        y = y_val[:, j]
        k = A.shape[1]
        res = minimize(lambda w: np.mean((A @ w - y) ** 2), np.full(k, 1 / k), method="SLSQP",
                       bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"ftol": 1e-14, "maxiter": 500})
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        weights[p] = {n: float(v) for n, v in zip(names + ["Naive"], w)}
    return weights


def blend_returns(returns, weights):
    """Apply {target: {name: w}} (with "Naive" = 0 return) to {name: (n, 4) returns}."""
    n = len(next(iter(returns.values())))
    out = np.zeros((n, len(PRICE_TYPES)))
    for j, p in enumerate(PRICE_TYPES):
        for name, w in weights[p].items():
            if name != "Naive":
                out[:, j] += w * returns[name][:, j]
    return out


def predict_model(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()


# ============================================================
# 6. GENETIC ALGORITHM FOR SAG-LSTM
# ============================================================

def create_individual():
    return [
        int(np.random.choice(HIDDEN_SIZE_CHOICES)),
        float(np.random.uniform(*LEARNING_RATE_RANGE)),
        int(np.random.choice(EPOCHS_CHOICES)),
    ]


def mutate_individual(individual):
    mutated = list(individual)
    if np.random.rand() < MUTATION_RATE:
        fresh = create_individual()
        idx = np.random.randint(len(mutated))
        mutated[idx] = fresh[idx]
    return mutated


def crossover_individuals(parent1, parent2):
    if np.random.rand() < CROSSOVER_RATE:
        point = np.random.randint(1, len(parent1))
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return list(parent1), list(parent2)


def run_genetic_algorithm(input_size, output_size, train_loader, val_loader, target_scaler):
    cache = {}

    def fitness(ind):
        key = (ind[0], round(ind[1], 6), ind[2])
        if key not in cache:
            torch.manual_seed(RANDOM_SEED)
            model = SAG_LSTM(input_size=input_size, hidden_size=ind[0], output_size=output_size)
            _, hist = train_model(model, train_loader, val_loader, ind[2], ind[1], target_scaler, verbose=False)
            cache[key] = hist["val_rmse"][-1] if hist["val_rmse"] else float("inf")
        return cache[key]

    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_individual, best_fitness = None, float("inf")
    ga_log = []

    for generation in range(GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            if fit < best_fitness:
                best_fitness, best_individual = fit, list(ind)
        ga_log.append({"generation": generation + 1, "best_fitness": best_fitness,
                       "mean_fitness": float(np.mean(fitnesses))})
        print(f"  GA generation {generation + 1}/{GENERATIONS}: best val RMSE {best_fitness:.6f} "
              f"(hidden={best_individual[0]}, lr={best_individual[1]:.5f}, epochs={best_individual[2]})")

        ranked = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda p: p[0])]
        elite = ranked[:POPULATION_SIZE // 2]
        new_population = [list(ind) for ind in elite]
        while len(new_population) < POPULATION_SIZE:
            p1 = elite[np.random.randint(len(elite))]
            p2 = elite[np.random.randint(len(elite))]
            c1, c2 = crossover_individuals(p1, p2)
            new_population.append(mutate_individual(c1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate_individual(c2))
        population = new_population

    return best_individual, best_fitness, ga_log


# ============================================================
# 7. HELPERS
# ============================================================

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), SCALED_FEATURE_COLUMNS),
            ("passthrough", "passthrough", UNSCALED_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )


def to_sequence(X):
    """(rows, features) -> (rows, seq_len=1, features)"""
    return np.asarray(X, dtype=np.float32)[:, np.newaxis, :]


def make_windows(X_scaled, end_indices):
    """(rows, features) -> (len(end_indices), WINDOW_SIZE, features); each window ends at (and includes) its row."""
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    return np.stack([X_scaled[i - WINDOW_SIZE + 1:i + 1] for i in end_indices])


def eemd_components(df, dates):
    """
    Causal EEMD for every date in `dates`: decompose the trailing EEMD_WINDOW closes (as log price
    relative to that day's close) and keep the last WINDOW_SIZE days of
      IMF 1 = ensemble-averaged first intrinsic mode function (high frequency)
      IMF 2 = segment - IMF 1 (everything slower: the other IMFs + residue)
    Decomposing the whole series at once would leak future prices into past IMFs, so each day
    is decomposed using only its own history. Results are cached per date in outputs/eemd_cache.npz.

    Returns (len(dates), WINDOW_SIZE, 2) float32; rows without enough history are NaN.
    """
    from PyEMD import EMD

    cache_path = OUT_DIR / "eemd_cache.npz"
    params = np.array([EEMD_WINDOW, WINDOW_SIZE, EEMD_TRIALS, EEMD_NOISE])
    cache = {}
    if cache_path.exists():
        z = np.load(cache_path)
        if np.array_equal(z["params"], params):
            cache = {(d, s): c for d, s, c in zip(z["dates"], z["segsum"], z["comps"])}

    closes = df["Price"].values
    pos_of = {d: i for i, d in enumerate(df["Date"])}
    emd = EMD()
    out = np.full((len(dates), WINDOW_SIZE, 2), np.nan, dtype=np.float32)
    new_keys, new_comps, t0 = [], [], time.time()
    for n, date in enumerate(dates):
        p = pos_of[date]
        if p < EEMD_WINDOW - 1:
            continue
        raw = closes[p - EEMD_WINDOW + 1:p + 1]
        key = (str(pd.Timestamp(date).date()), float(np.round(raw.sum(), 4)))  # segsum detects edited history
        if key in cache:
            out[n] = cache[key]
            continue
        seg = np.log(raw / raw[-1])
        rng = np.random.default_rng(RANDOM_SEED + p)
        imf1 = np.zeros_like(seg)
        for _ in range(EEMD_TRIALS):
            imfs = emd.emd(seg + rng.normal(0, EEMD_NOISE * seg.std(), seg.size), max_imf=1)
            imf1 += imfs[0]
        imf1 /= EEMD_TRIALS
        comp = np.stack([imf1, seg - imf1], axis=-1)[-WINDOW_SIZE:].astype(np.float32)
        out[n] = comp
        new_keys.append(key)
        new_comps.append(comp)
        if len(new_keys) % 1000 == 0:
            print(f"  EEMD: {len(new_keys)} new windows decomposed ({time.time() - t0:.0f}s)")

    if new_keys:
        cache.update(dict(zip(new_keys, new_comps)))
        keys = list(cache)
        np.savez(cache_path, params=params, dates=np.array([k[0] for k in keys]),
                 segsum=np.array([k[1] for k in keys]), comps=np.stack([cache[k] for k in keys]))
        print(f"  EEMD: decomposed {len(new_keys)} windows in {time.time() - t0:.0f}s "
              f"({len(dates) - len(new_keys)} from cache)")
    return out


def eemd_inputs(comps, X_scaled, indices, scale):
    """Concatenate scaled EEMD components with the feature window: (n, WINDOW_SIZE, 2 + features)."""
    return np.concatenate([comps[indices] / scale, make_windows(X_scaled, indices)], axis=-1).astype(np.float32)


def price_window_inputs(full_seq, X_scaled, indices):
    """
    (n, WINDOW_SIZE, 4 + features): the window's Open/High/Low/Close divided by the close of the
    window's last day (strictly positive, ~1.0), followed by the scaled feature window.
    """
    ohlc = full_seq[["Open", "High", "Low", "Price"]].values.astype(np.float32)
    rel = make_windows(ohlc, indices) / ohlc[np.asarray(indices), 3][:, None, None]
    return np.concatenate([rel, make_windows(X_scaled, indices)], axis=-1).astype(np.float32)


def returns_to_prices(returns, base_close):
    """returns: (n, 4) in PRICE_TYPES order, base_close: (n,) -> dict of price arrays with OHLC consistency."""
    prices = base_close[:, None] * (1 + returns)
    o, h, l, c = prices[:, 0], prices[:, 1], prices[:, 2], prices[:, 3]
    h = np.maximum.reduce([h, o, c])
    l = np.minimum.reduce([l, o, c])
    return {"Open": o, "High": h, "Low": l, "Close": c}


def score(actual_prices, pred_prices, base_close):
    """Per-target metrics in $ plus direction accuracy of the close."""
    metrics = {}
    for p in PRICE_TYPES:
        a, f = actual_prices[p], pred_prices[p]
        metrics[p] = {
            "MAE": float(mean_absolute_error(a, f)),
            "RMSE": float(np.sqrt(mean_squared_error(a, f))),
            "MAPE": float(np.mean(np.abs((f - a) / a)) * 100),
            "HitRate": float(np.mean(np.abs((f - a) / a) <= HIT_TOLERANCE) * 100),
        }
    actual_dir = np.sign(actual_prices["Close"] - base_close)
    pred_dir = np.sign(pred_prices["Close"] - base_close)
    metrics["DirectionAccuracy"] = float(np.mean(actual_dir == pred_dir) * 100)
    metrics["AvgRMSE"] = float(np.mean([metrics[p]["RMSE"] for p in PRICE_TYPES]))
    return metrics


def leaderboard_table(backtest_metrics):
    """Models ranked by average RMSE over O/H/L/C on the backtest hold-out ($ errors)."""
    rows = []
    for name, m in backtest_metrics.items():
        row = {"Model": name,
               "Avg RMSE": np.mean([m[p]["RMSE"] for p in PRICE_TYPES]),
               "Avg MAE": np.mean([m[p]["MAE"] for p in PRICE_TYPES])}
        for p in PRICE_TYPES:
            row[f"{p} RMSE"] = m[p]["RMSE"]
            row[f"{p} MAE"] = m[p]["MAE"]
        row["Direction %"] = m["DirectionAccuracy"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("Avg RMSE").reset_index(drop=True)
    table.insert(0, "Rank", range(1, len(table) + 1))
    return table


def print_leaderboard(backtest_metrics):
    print("\n" + "=" * 70 + "\nMODELS SORTED BY RMSE (backtest hold-out, $ errors)\n" + "=" * 70)
    print(leaderboard_table(backtest_metrics).to_string(index=False, float_format="%.2f"))


# ============================================================
# 8. MAIN
# ============================================================

def export_backtest_ohlc():
    """
    Rebuild the hold-out Open/High/Low/Close predictions (and P(up)) of every model and ensemble from the
    stored outputs (model_predictions.npz + results.json), next to the actual prices -> outputs/backtest_ohlc.csv.
    No model is trained. The rows are checked against results.json so a stale store cannot be misaligned.
    """
    with open(OUT_DIR / "results.json") as f:
        R = json.load(f)
    z = np.load(OUT_DIR / "model_predictions.npz")
    df = load_data()
    feat_df = add_targets(build_features(df))
    labelled = feat_df.dropna(subset=BASE_FEATURE_COLUMNS + REGIME_INPUTS + TARGET_COLUMNS).reset_index(drop=True)
    split_idx = int(len(labelled) * (1 - TEST_SIZE))
    test_df = labelled.iloc[split_idx:]
    b = R["backtest"]
    if len(test_df) != b["test_rows"] or str(test_df["Date"].iloc[0].date()) != b["test_start"]:
        raise SystemExit("Stored predictions do not match the current data; run a full training first.")

    next_date = dict(zip(feat_df["Date"], feat_df["Date"].shift(-1)))
    base = test_df["Price"].values
    out = pd.DataFrame({"Date": test_df["Date"].values,
                        "Target_Date": [next_date[d] for d in test_df["Date"]],
                        "Close_t": base})
    for p in PRICE_TYPES:
        out[f"Actual {p}"] = base * (1 + test_df[f"Next_{p}"].values)

    test_ret = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("test__")}
    test_up = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("testup__")}
    members, inv_w, stack_w = R["ensemble_members"], R["ensemble_weights"], R["stacking_weights"]
    test_ret.update({
        "Ensemble (mean)": np.mean([test_ret[n] for n in members], axis=0),
        "Ensemble (RMSE-weighted)": np.sum([inv_w[n] * test_ret[n] for n in members], axis=0),
        "Ensemble (stacked)": blend_returns(test_ret, stack_w),
    })
    close_w = stack_w["Close"]
    test_up.update({
        "Ensemble (mean)": np.mean([test_up[n] for n in members], axis=0),
        "Ensemble (RMSE-weighted)": np.sum([inv_w[n] * test_up[n] for n in members], axis=0),
        "Ensemble (stacked)": close_w["Naive"] * 0.5 + np.sum([close_w[n] * test_up[n] for n in members], axis=0),
    })

    columns, worst = {}, 0.0
    for name, ret in test_ret.items():
        prices = returns_to_prices(ret, base)
        for p in PRICE_TYPES:
            columns[f"{name} | {p}"] = prices[p]
        if name in test_up:
            columns[f"{name} | P(up)"] = test_up[name]
        if name in b["metrics"]:  # sanity check against the stored scores
            avg_rmse = np.mean([np.sqrt(np.mean((prices[p] - out[f"Actual {p}"].values) ** 2)) for p in PRICE_TYPES])
            worst = max(worst, abs(avg_rmse - b["metrics"][name]["AvgRMSE"]))
    out = pd.concat([out, pd.DataFrame(columns)], axis=1)
    out.to_csv(OUT_DIR / "backtest_ohlc.csv", index=False)
    print(f"Saved {OUT_DIR / 'backtest_ohlc.csv'}: {len(out)} days x {len(test_ret)} models "
          f"(max difference vs stored RMSE: {worst:.6f})")


def compute_market_stats(df, infer_row):
    """Indicator snapshot of the last available day, for the UI."""
    last = infer_row.iloc[0]
    last_date = df["Date"].max()
    last_year = df[df["Date"] > last_date - pd.Timedelta(days=365)]
    daily_ret = df["Price"].pct_change().dropna()
    return {
        "RSI_14": float(last["RSI_Raw"]),
        "ATR_14": float(last["ATR"]),
        "ATR_pct": float(last["ATR_Ratio"] * 100),
        "Volatility_7_pct": float(last["Volatility_7"] * 100),
        "Volatility_14_pct": float(last["Volatility_14"] * 100),
        "Annualized_Vol_30d_pct": float(daily_ret.tail(30).std() * np.sqrt(252) * 100),
        "BB_Position": float(last["BB_Position"]),
        "BB_Upper": float(last["BB_Upper"]),
        "BB_Lower": float(last["BB_Lower"]),
        "BB_Middle": float(last["BB_Middle"]),
        "SMA_7": float(last["SMA_7"]), "SMA_14": float(last["SMA_14"]), "SMA_30": float(last["SMA_30"]),
        "EMA_7": float(last["EMA_7"]), "EMA_14": float(last["EMA_14"]), "EMA_30": float(last["EMA_30"]),
        "MACD": float(last["MACD"]), "MACD_Signal": float(last["MACD_Signal"]),
        "Momentum_7": float(last["Momentum"]),
        "High_52w": float(last_year["High"].max()), "Low_52w": float(last_year["Low"].min()),
        "Return_1d_pct": float(last["Return"] * 100),
        "Return_5d_pct": float((df["Price"].iloc[-1] / df["Price"].iloc[-6] - 1) * 100),
        "Return_20d_pct": float((df["Price"].iloc[-1] / df["Price"].iloc[-21] - 1) * 100),
        "Return_YTD_pct": float((df["Price"].iloc[-1] /
                                 df[df["Date"] < pd.Timestamp(last_date.year, 1, 1)]["Price"].iloc[-1] - 1) * 100),
        "Avg_Daily_Range_20d": float((df["High"] - df["Low"]).tail(20).mean()),
        "Up_Days_Pct_All": float((daily_ret > 0).mean() * 100),
    }


# ============================================================
# SAVED MODELS: refit once and save, then predict new days without training
# ============================================================

def _model_file(name):
    return MODELS_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".pt")


def prepare_frames(df, regimes=None):
    """Features, regime probabilities and the contiguous day sequence (last row = the day to forecast from)."""
    feat_df = add_targets(build_features(df))
    infer_row = feat_df.iloc[[-1]]
    labelled = feat_df.dropna(subset=BASE_FEATURE_COLUMNS + REGIME_INPUTS + TARGET_COLUMNS).reset_index(drop=True)
    first_valid = feat_df[BASE_FEATURE_COLUMNS + REGIME_INPUTS].notna().all(axis=1).idxmax()
    full_seq = feat_df.loc[first_valid:].reset_index(drop=True)
    assert len(full_seq) == len(labelled) + 1 and full_seq["Date"].iloc[-1] == df["Date"].max()
    assert not infer_row[BASE_FEATURE_COLUMNS + REGIME_INPUTS].isnull().any(axis=None), "NaN in the last day's features"
    if regimes is None:
        regimes = fit_regimes(labelled)
    labelled, full_seq, infer_row = (add_regimes(f, regimes) for f in (labelled, full_seq, infer_row))
    return labelled, full_seq, infer_row, regimes


def inference_inputs(df, full_seq, infer_row, preprocessor, eemd_scale):
    """Inputs of every kind for the last day, built with the saved (already fitted) preprocessing."""
    Xs = preprocessor.transform(full_seq[FEATURE_COLUMNS])
    n = len(full_seq) - 1
    comps = eemd_components(df, full_seq["Date"].values)
    return {
        "row": to_sequence(preprocessor.transform(infer_row[FEATURE_COLUMNS])),
        "window": make_windows(Xs, [n]),
        "prices": price_window_inputs(full_seq, Xs, [n]),
        "eemd": eemd_inputs(comps, Xs, [n], eemd_scale),
    }, comps


def save_artifacts(models, preprocessor, target_scaler, regimes, eemd_scale, lgb_regressors, lgb_classifier,
                   trained_through, sag_hidden):
    MODELS_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        torch.save(model.state_dict(), _model_file(name))
    with open(MODELS_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump({"preprocessor": preprocessor, "target_scaler": target_scaler, "regimes": regimes,
                     "eemd_scale": eemd_scale, "lgb_regressors": lgb_regressors, "lgb_classifier": lgb_classifier,
                     "trained_through": str(pd.Timestamp(trained_through).date()), "sag_hidden": int(sag_hidden),
                     "feature_columns": FEATURE_COLUMNS, "model_names": list(models),
                     "saved_at": pd.Timestamp.now().isoformat(timespec="seconds")}, f)
    print(f"Saved {len(models)} neural models + LightGBM + preprocessing to {MODELS_DIR}")


def _lightgbm_trees_from_results_or_log(R):
    if R.get("lightgbm_trees"):
        return R["lightgbm_trees"]["regression"], R["lightgbm_trees"]["classifier"]
    log = (OUT_DIR / "train_log.txt").read_text(encoding="utf-8", errors="ignore")
    reg = re.findall(r"Refitting LightGBM on all data \(trees per target (\{.*?\})\)", log)
    clf = re.findall(r"Up/Down classifier: (\d+) trees", log)
    if not reg or not clf:
        raise SystemExit("LightGBM tree counts not found; run a full training instead.")
    return ast.literal_eval(reg[-1]), int(clf[-1])


def refit_and_save():
    """
    One-time refit of the final (100% data) models on EXACTLY the data behind outputs/results.json, with the
    same epochs / hyperparameters, then save them. Verifies the refit reproduces the stored forecasts.
    """
    import lightgbm as lgb

    with open(OUT_DIR / "results.json") as f:
        R = json.load(f)
    trained_through = pd.Timestamp(R["last_date"])
    df = load_data()
    df = df[df["Date"] <= trained_through].reset_index(drop=True)
    print(f"Refitting the final models on data through {trained_through.date()} (same data as results.json)")

    labelled, full_seq, infer_row, regimes = prepare_frames(df)
    pre = make_preprocessor()
    X_all = to_sequence(pre.fit_transform(labelled[FEATURE_COLUMNS]))
    tscaler = StandardScaler()
    y_all = tscaler.fit_transform(labelled[TARGET_COLUMNS].values)
    Xs = pre.transform(full_seq[FEATURE_COLUMNS])
    all_idx = np.arange(WINDOW_SIZE - 1, len(labelled))
    comps = eemd_components(df, full_seq["Date"].values)
    has_comps = ~np.isnan(comps).any(axis=(1, 2))
    all_idx_e = all_idx[has_comps[all_idx]]
    eemd_scale = comps[all_idx_e].reshape(-1, 2).std(axis=0)
    loaders = {
        "row": make_loader(X_all, y_all, shuffle=False),
        "window": make_loader(make_windows(Xs, all_idx), y_all[all_idx], shuffle=False),
        "eemd": make_loader(eemd_inputs(comps, Xs, all_idx_e, eemd_scale), y_all[all_idx_e], shuffle=False),
        "prices": make_loader(price_window_inputs(full_seq, Xs, all_idx), y_all[all_idx], shuffle=False),
    }
    inputs, _ = inference_inputs(df, full_seq, infer_row, pre, eemd_scale)
    last_close = float(infer_row["Price"].iloc[0])
    sag_hidden = R["ga"]["best_hidden_size"]

    trained, diffs = {}, {}
    for name, model in build_models(len(FEATURE_COLUMNS), len(TARGET_COLUMNS), sag_hidden).items():
        epochs, lr = R["training"][name]["epochs"], R["training"][name]["learning_rate"]
        print(f"  {name}: {epochs} epochs")
        model.up_threshold = float(-tscaler.mean_[3] / tscaler.scale_[3])
        model, _ = train_model(model, loaders[input_kind(name)], None, epochs, lr, tscaler, verbose=False, name=name)
        trained[name] = model
        prices = returns_to_prices(tscaler.inverse_transform(predict_model(model, inputs[input_kind(name)])),
                                   np.array([last_close]))
        diffs[name] = max(abs(float(prices[p][0]) - R["predictions"][name][p]) for p in PRICE_TYPES)

    reg_trees, clf_trees = _lightgbm_trees_from_results_or_log(R)
    y_full = labelled[TARGET_COLUMNS].values
    lgb_regs = [lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": reg_trees[p]}).fit(labelled[FEATURE_COLUMNS],
                                                                                   y_full[:, j])
                for j, p in enumerate(PRICE_TYPES)]
    lgb_clf = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": clf_trees}).fit(
        labelled[FEATURE_COLUMNS], labelled["Next_Close"].values > 0)
    prices = returns_to_prices(np.column_stack([m.predict(infer_row[FEATURE_COLUMNS]) for m in lgb_regs]),
                               np.array([last_close]))
    diffs["LightGBM"] = max(abs(float(prices[p][0]) - R["predictions"]["LightGBM"][p]) for p in PRICE_TYPES)

    save_artifacts(trained, pre, tscaler, regimes, eemd_scale, lgb_regs, lgb_clf, trained_through, sag_hidden)
    R["lightgbm_trees"] = {"regression": reg_trees, "classifier": clf_trees}
    R["models_trained_through"] = str(trained_through.date())
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(R, f, indent=2)
    print("\nReproduction check (max $ difference vs the stored forecast for "
          f"{R['target_date']}, per model):")
    for name, d in diffs.items():
        print(f"  {name:18s} {d:10.4f}")


def _log_forecasts(made_on, target_date, predictions, classification):
    rows = [{"made_on": made_on, "target_date": target_date, "model": name,
             **{p: v[p] for p in PRICE_TYPES},
             "p_up": (classification.get(name) or {}).get("p_up_next_day")}
            for name, v in predictions.items()]
    new = pd.DataFrame(rows)
    if FORECAST_LOG.exists():
        old = pd.read_csv(FORECAST_LOG, dtype={"made_on": str, "target_date": str})
        new = pd.concat([old[~((old["made_on"] == made_on) & old["model"].isin(new["model"]))], new])
    new.to_csv(FORECAST_LOG, index=False)


def _live_track(df):
    """Every logged forecast whose target day is now known, with its errors."""
    if not FORECAST_LOG.exists():
        return []
    log = pd.read_csv(FORECAST_LOG, dtype={"made_on": str, "target_date": str})
    by_date = df.set_index(df["Date"].dt.strftime("%Y-%m-%d"))
    out = []
    for _, r in log.iterrows():
        if r["target_date"] not in by_date.index or r["made_on"] not in by_date.index:
            continue
        a, prev_close = by_date.loc[r["target_date"]], float(by_date.loc[r["made_on"], "Price"])
        actual = {"Open": float(a["Open"]), "High": float(a["High"]), "Low": float(a["Low"]), "Close": float(a["Price"])}
        went_up = actual["Close"] > prev_close
        out.append({
            "made_on": r["made_on"], "target_date": r["target_date"], "model": r["model"],
            **{f"pred_{p}": float(r[p]) for p in PRICE_TYPES}, **{f"actual_{p}": actual[p] for p in PRICE_TYPES},
            "avg_abs_error": float(np.mean([abs(r[p] - actual[p]) for p in PRICE_TYPES])),
            "close_error": float(r["Close"] - actual["Close"]),
            "price_direction_right": bool((r["Close"] > prev_close) == went_up),
            "p_up": None if pd.isna(r["p_up"]) else float(r["p_up"]),
            "updown_call_right": None if pd.isna(r["p_up"]) else bool((r["p_up"] > 0.5) == went_up),
            "prev_close": prev_close,
        })
    return out


def predict_next_day():
    """
    Forecast the day after the latest data with the SAVED models (no training): runs the preprocessing and
    feature steps on the new data, predicts every model and ensemble, and updates outputs/results.json.
    """
    import lightgbm  # noqa: F401  (needed to unpickle the LightGBM models)

    if not (MODELS_DIR / "artifacts.pkl").exists():
        raise SystemExit("No saved models: run `python train_predict.py --save-models` once first.")
    with open(MODELS_DIR / "artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    with open(OUT_DIR / "results.json") as f:
        R = json.load(f)
    assert art["feature_columns"] == FEATURE_COLUMNS, "feature list changed since the models were saved"

    # keep the forecast currently shown (e.g. made on 22 Sep for 23 Sep) in the log before replacing it
    _log_forecasts(R["last_date"], R["target_date"], R["predictions"], R.get("classification") or {})

    df = load_data()
    last_date = df["Date"].max()
    target_date = next_trading_day(last_date)
    print(f"Latest data {last_date.date()} -> forecasting {target_date.date()} with models trained through "
          f"{art['trained_through']} (no retraining)")
    labelled, full_seq, infer_row, _ = prepare_frames(df, art["regimes"])
    inputs, comps = inference_inputs(df, full_seq, infer_row, art["preprocessor"], art["eemd_scale"])
    tscaler = art["target_scaler"]
    last_close = float(infer_row["Price"].iloc[0])

    predictions, next_up = {}, {}
    attention_dates = [str(d.date()) for d in full_seq["Date"].iloc[-WINDOW_SIZE:]]
    for name, model in build_models(len(FEATURE_COLUMNS), len(TARGET_COLUMNS), art["sag_hidden"]).items():
        model.load_state_dict(torch.load(_model_file(name), map_location=device))
        model.to(device).eval()
        if name not in art.get("model_names", []):
            continue
        X = inputs[input_kind(name)]
        prices = returns_to_prices(tscaler.inverse_transform(predict_model(model, X)), np.array([last_close]))
        predictions[name] = {p: float(prices[p][0]) for p in PRICE_TYPES}
        next_up[name] = float(up_probability(model, X)[0])
        if name == "CNN-BiLSTM-Attn" and R.get("cnn_bilstm_attention"):
            R["cnn_bilstm_attention"].update(dates=attention_dates,
                                             weights=model.base.last_attention[0].cpu().numpy().tolist())
        if name == "Helformer" and R.get("helformer"):
            dec = model.base.decomposition
            with torch.no_grad():
                x_prices = torch.tensor(X[:, :, :4]).to(device)
                deseason, lvl, seas = dec(x_prices)
            R["helformer"].update(dates=attention_dates, close_pct=((x_prices[0, :, 3] - 1) * 100).tolist(),
                                  level_pct=((lvl[0, :, 3] - 1) * 100).tolist(), seasonal=seas[0, :, 3].tolist(),
                                  deseasonalized_pct=((deseason[0, :, 3] - 1) * 100).tolist())
    if R.get("eemd"):
        last_comp = comps[len(full_seq) - 1]
        R["eemd"].update(dates=attention_dates, imf1_pct=(last_comp[:, 0] * 100).tolist(),
                         imf2_pct=(last_comp[:, 1] * 100).tolist())

    X_row = infer_row[FEATURE_COLUMNS]
    prices = returns_to_prices(np.column_stack([m.predict(X_row) for m in art["lgb_regressors"]]),
                               np.array([last_close]))
    predictions["LightGBM"] = {p: float(prices[p][0]) for p in PRICE_TYPES}
    next_up["LightGBM"] = float(art["lgb_classifier"].predict_proba(X_row)[0, 1])

    chronos_ret, chronos_up = chronos_returns(df, [last_date])  # pretrained, zero-shot
    prices = returns_to_prices(chronos_ret, np.array([last_close]))
    predictions["Chronos-Bolt"] = {p: float(prices[p][0]) for p in PRICE_TYPES}
    next_up["Chronos-Bolt"] = float(chronos_up[0])

    # Ensembles: same members and weights as the last training run
    members, inv_w, stack_w = R["ensemble_members"], R["ensemble_weights"], R["stacking_weights"]
    ens = {
        "Ensemble (mean)": {p: float(np.mean([predictions[n][p] for n in members])) for p in PRICE_TYPES},
        "Ensemble (RMSE-weighted)": {p: float(sum(inv_w[n] * predictions[n][p] for n in members))
                                     for p in PRICE_TYPES},
        "Ensemble (stacked)": {p: float(sum(w * (last_close if n == "Naive" else predictions[n][p])
                                            for n, w in stack_w[p].items())) for p in PRICE_TYPES},
    }
    for e in ens.values():
        e["High"] = max(e["High"], e["Open"], e["Close"])
        e["Low"] = min(e["Low"], e["Open"], e["Close"])
    predictions.update(ens)
    close_w = stack_w["Close"]
    next_up.update({
        "Ensemble (mean)": float(np.mean([next_up[n] for n in members])),
        "Ensemble (RMSE-weighted)": float(sum(inv_w[n] * next_up[n] for n in members)),
        "Ensemble (stacked)": float(close_w["Naive"] * 0.5 + sum(close_w[n] * next_up[n] for n in members)),
    })

    # ---- update what the UI shows (backtest scores stay as they were) ----
    recent = df.tail(120)
    daily_ret = df["Price"].pct_change().dropna()
    last = infer_row.iloc[0]
    R.update({
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "last_date": str(last_date.date()), "target_date": str(target_date.date()), "n_rows": int(len(df)),
        "last_bar": {"Open": float(last["Open"]), "High": float(last["High"]), "Low": float(last["Low"]),
                     "Close": last_close},
        "predictions": predictions,
        "market_stats": compute_market_stats(df, infer_row),
        "recent_ohlc": {"Date": [str(d.date()) for d in recent["Date"]], "Open": recent["Open"].tolist(),
                        "High": recent["High"].tolist(), "Low": recent["Low"].tolist(),
                        "Close": recent["Price"].tolist()},
        "daily_returns_pct": (daily_ret.tail(1000) * 100).round(4).tolist(),
        "models_trained_through": art["trained_through"],
        "prediction_mode": "saved models, no retraining",
    })
    for name, p in next_up.items():
        R.setdefault("classification", {}).setdefault(name, {})["p_up_next_day"] = p
    if R.get("regimes"):
        tail = full_seq.tail(250)
        R["regimes"]["current"] = dict(zip(REGIME_NAMES, regime_probabilities(art["regimes"], infer_row)[0]
                                           .round(4).tolist()))
        R["regimes"]["timeline"] = {
            "Date": [str(d.date()) for d in tail["Date"]], "Close": tail["Price"].round(2).tolist(),
            "Regime": [REGIME_NAMES[k] for k in regime_probabilities(art["regimes"], tail).argmax(axis=1)]}

    _log_forecasts(str(last_date.date()), str(target_date.date()), predictions, R["classification"])
    R["live_track"] = _live_track(df)
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(R, f, indent=2)

    print(f"\nPredictions for {target_date.date()} (last close {last_close:,.2f} on {last_date.date()}):")
    table = pd.DataFrame(predictions).T[PRICE_TYPES]
    table["P(up)"] = pd.Series(next_up)
    print(table.to_string(float_format="%.2f"))
    for t in R["live_track"]:
        if t["model"] in ("Ensemble (stacked)", "Ensemble (mean)"):
            print(f"  {t['model']} forecast for {t['target_date']}: close {t['pred_Close']:,.2f} vs actual "
                  f"{t['actual_Close']:,.2f} (error {t['close_error']:+.2f}), Up/Down call "
                  f"{'right' if t['updown_call_right'] else 'wrong'}")
    print(f"\nSaved {OUT_DIR / 'results.json'} and {FORECAST_LOG}")


def main(only=None):
    """only: optional list of model names to (re)train; every other model's results are
    taken from the existing outputs/results.json and outputs/backtest_predictions.csv."""
    set_seed()
    print("Using device:", device)

    df = load_data()
    assert df["Date"].is_unique
    last_date = df["Date"].max()
    TARGET_DATE = next_trading_day(last_date)
    print("Last available date:", last_date.date(), "-> predicting", TARGET_DATE.date())

    feat_df = add_targets(build_features(df))

    # Inference row = the last day (its targets are unknown)
    infer_row = feat_df.iloc[[-1]]
    assert infer_row["Date"].iloc[0] == last_date
    assert not infer_row[BASE_FEATURE_COLUMNS].isnull().any(axis=None), "NaN in inference features"

    labelled = feat_df.dropna(subset=BASE_FEATURE_COLUMNS + REGIME_INPUTS + TARGET_COLUMNS).reset_index(drop=True)
    print(f"Features: {len(FEATURE_COLUMNS)} | Targets: {len(TARGET_COLUMNS)} | Labelled rows: {len(labelled)}")

    # Contiguous day sequence (labelled rows + inference row) used to build sliding windows
    first_valid = feat_df[BASE_FEATURE_COLUMNS + REGIME_INPUTS].notna().all(axis=1).idxmax()
    full_seq = feat_df.loc[first_valid:].reset_index(drop=True)
    assert len(full_seq) == len(labelled) + 1 and full_seq["Date"].iloc[-1] == last_date, \
        "labelled rows are not contiguous; sliding windows would be misaligned"

    INPUT_SIZE, OUTPUT_SIZE = len(FEATURE_COLUMNS), len(TARGET_COLUMNS)

    old = None
    if only:
        with open(OUT_DIR / "results.json") as f:
            old = json.load(f)
        if old["last_date"] != str(last_date.date()) or old["target_date"] != str(TARGET_DATE.date()):
            raise SystemExit(f"outputs/results.json is for {old['last_date']} -> {old['target_date']}, but the data "
                             f"is {last_date.date()} -> {TARGET_DATE.date()}. Run a full training instead of --only.")
        print(f"Training only: {', '.join(only)} (other models are loaded from outputs/results.json)")

    # --------------------------------------------------------
    # PHASE A: BACKTEST (first 90% train, last 10% hold-out)
    # --------------------------------------------------------
    print("\n" + "=" * 70 + "\nPHASE A: BACKTEST ON LAST 10%\n" + "=" * 70)

    split_idx = int(len(labelled) * (1 - TEST_SIZE))
    # 1-day embargo: row split_idx-1's target is the close of row split_idx (first test day)
    fit_df = labelled.iloc[:split_idx - 1]
    test_df = labelled.iloc[split_idx:]
    val_start = int(len(fit_df) * (1 - VAL_SIZE))
    tr_df, val_df = fit_df.iloc[:val_start], fit_df.iloc[val_start:]
    print(f"Train {len(tr_df)} | Val {len(val_df)} | Test {len(test_df)} "
          f"({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")

    # Market regimes, fitted on the training rows only
    regimes_a = fit_regimes(tr_df)
    tr_df, val_df, test_df = (add_regimes(f, regimes_a) for f in (tr_df, val_df, test_df))
    labelled, full_seq = add_regimes(labelled, regimes_a), add_regimes(full_seq, regimes_a)
    test_regime = regime_probabilities(regimes_a, test_df).argmax(axis=1)
    print("Regimes (train rows): " + ", ".join(
        f"{n} {np.mean(regime_probabilities(regimes_a, tr_df).argmax(axis=1) == k) * 100:.0f}%"
        for k, n in enumerate(REGIME_NAMES)))

    pre_a = make_preprocessor()
    X_tr = to_sequence(pre_a.fit_transform(tr_df[FEATURE_COLUMNS]))
    X_val = to_sequence(pre_a.transform(val_df[FEATURE_COLUMNS]))
    X_te = to_sequence(pre_a.transform(test_df[FEATURE_COLUMNS]))

    tscaler_a = StandardScaler()
    y_tr = tscaler_a.fit_transform(tr_df[TARGET_COLUMNS].values)
    y_val = tscaler_a.transform(val_df[TARGET_COLUMNS].values)

    train_loader = make_loader(X_tr, y_tr, shuffle=False)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    # Sliding-window inputs for the window models (same scalers, same rows; the first
    # WINDOW_SIZE-1 training rows are skipped because they have no full history window)
    Xs_a = pre_a.transform(full_seq[FEATURE_COLUMNS])
    y_lab_a = tscaler_a.transform(labelled[TARGET_COLUMNS].values)
    tr_idx = np.arange(WINDOW_SIZE - 1, val_start)
    val_idx = np.arange(val_start, split_idx - 1)
    te_idx = np.arange(split_idx, len(labelled))
    data_a = {
        "row": (train_loader, val_loader, X_te),
        "window": (make_loader(make_windows(Xs_a, tr_idx), y_lab_a[tr_idx], shuffle=False),
                   make_loader(make_windows(Xs_a, val_idx), y_lab_a[val_idx], shuffle=False),
                   make_windows(Xs_a, te_idx)),
    }

    # EEMD components (only computed when an EEMD model is trained)
    train_eemd = not only or any(m in EEMD_MODELS for m in only)
    if train_eemd:
        print("\nDecomposing close prices with causal EEMD ...")
        comps = eemd_components(df, full_seq["Date"].values)
        has_comps = ~np.isnan(comps).any(axis=(1, 2))
        tr_idx_e = tr_idx[has_comps[tr_idx]]
        scale_a = comps[tr_idx_e].reshape(-1, 2).std(axis=0)
        data_a["eemd"] = (
            make_loader(eemd_inputs(comps, Xs_a, tr_idx_e, scale_a), y_lab_a[tr_idx_e], shuffle=False),
            make_loader(eemd_inputs(comps, Xs_a, val_idx, scale_a), y_lab_a[val_idx], shuffle=False),
            eemd_inputs(comps, Xs_a, te_idx, scale_a),
        )

    train_prices = not only or any(m in PRICE_WINDOW_MODELS for m in only)
    if train_prices:
        data_a["prices"] = (
            make_loader(price_window_inputs(full_seq, Xs_a, tr_idx), y_lab_a[tr_idx], shuffle=False),
            make_loader(price_window_inputs(full_seq, Xs_a, val_idx), y_lab_a[val_idx], shuffle=False),
            price_window_inputs(full_seq, Xs_a, te_idx),
        )

    if only and "SAG-LSTM" not in only:
        ga = old["ga"]
        sag_hidden, sag_lr, sag_epochs = ga["best_hidden_size"], ga["best_learning_rate"], ga["best_epochs"]
        best_fit, ga_log = ga["best_val_rmse"], ga["log"]
    else:
        print("\nRunning Genetic Algorithm for SAG-LSTM ...")
        set_seed()
        best_params, best_fit, ga_log = run_genetic_algorithm(INPUT_SIZE, OUTPUT_SIZE, train_loader, val_loader,
                                                              tscaler_a)
        sag_hidden, sag_lr, sag_epochs = best_params
    print(f"GA best: hidden={sag_hidden}, lr={sag_lr:.5f}, epochs={sag_epochs}")

    train_cfg = {name: (EPOCHS, LEARNING_RATE) for name in
                 ["CNN-BiLSTM-Attn", "LSTM-Transformer", "LSTM-Attention-LSTM", "EEMD-TCN-GRU", "Helformer",
                  "LSTM", "GRU", "Transformer", "Transformer-MT",
                  "PatchTST", "TFT"]}
    train_cfg["SAG-LSTM"] = (sag_epochs, sag_lr)
    train_cfg["LightGBM"] = (LGB_PARAMS["n_estimators"], LGB_PARAMS["learning_rate"])
    train_cfg["Chronos-Bolt"] = (0, 0.0)  # pretrained, no training

    base_close_test = test_df["Price"].values
    # actual next-day prices reconstructed from the targets
    actual_test = {p: base_close_test * (1 + test_df[f"Next_{p}"].values) for p in PRICE_TYPES}

    backtest_metrics, histories = {}, {}
    val_up, test_up = {}, {}              # P(tomorrow's close > today's close) on validation / test
    y_up_val, y_up_test = val_df["Next_Close"].values > 0, test_df["Next_Close"].values > 0
    val_returns, test_returns = {}, {}   # predicted next-day returns (rows, 4) on validation / test
    for name, model in build_models(INPUT_SIZE, OUTPUT_SIZE, sag_hidden).items():
        if only and name not in only:
            continue
        epochs, lr = train_cfg[name]
        print(f"\nTraining {name} (epochs={epochs}, lr={lr:.5f})")
        t0 = time.time()
        tl, vl, Xt = data_a[input_kind(name)]
        if hasattr(model, "up_threshold"):
            model.up_threshold = float(-tscaler_a.mean_[3] / tscaler_a.scale_[3])
        model, hist = train_model(model, tl, vl, epochs, lr, tscaler_a, name=name)
        histories[name] = {"backtest": hist, "train_seconds_backtest": time.time() - t0}
        val_returns[name] = tscaler_a.inverse_transform(predict_loader(model, vl))
        test_returns[name] = tscaler_a.inverse_transform(predict_model(model, Xt))
        val_up[name], test_up[name] = up_probability(model, vl), up_probability(model, Xt)
        cm = classification_metrics(test_up[name], y_up_test)
        print(f"  Up/Down: test accuracy {cm['accuracy']:.1f}% (AUC {cm['auc'] or float('nan'):.3f}; "
              f"always-up baseline {y_up_test.mean() * 100:.1f}%)")
        train_cfg[name] = (max(MIN_REFIT_EPOCHS, epochs), lr)  # epochs used for the 100% refit

    y_val_raw = val_df[TARGET_COLUMNS].values

    if not only or "LightGBM" in only:
        print("\nTraining LightGBM (one model per target, fixed tree count)")
        t0 = time.time()
        lgb_models, lgb_best, lgb_curves = fit_lightgbm(tr_df[FEATURE_COLUMNS], tr_df[TARGET_COLUMNS].values,
                                                        val_df[FEATURE_COLUMNS], y_val_raw)
        val_returns["LightGBM"] = predict_lightgbm(lgb_models, val_df[FEATURE_COLUMNS])
        test_returns["LightGBM"] = predict_lightgbm(lgb_models, test_df[FEATURE_COLUMNS])
        lgb_clf, lgb_clf_best = fit_lightgbm_classifier(tr_df[FEATURE_COLUMNS], tr_df["Next_Close"].values > 0,
                                                        val_df[FEATURE_COLUMNS], y_up_val)
        val_up["LightGBM"] = lgb_clf.predict_proba(val_df[FEATURE_COLUMNS], num_iteration=lgb_clf.best_iteration_)[:, 1]
        test_up["LightGBM"] = lgb_clf.predict_proba(test_df[FEATURE_COLUMNS],
                                                    num_iteration=lgb_clf.best_iteration_)[:, 1]
        print(f"  Up/Down classifier: {lgb_clf_best} trees, test accuracy "
              f"{classification_metrics(test_up['LightGBM'], y_up_test)['accuracy']:.1f}%")
        train_cfg["LightGBM"] = (lgb_best[-1], LGB_PARAMS["learning_rate"])
        histories["LightGBM"] = {"backtest": {"train_loss": lgb_curves[0], "val_loss": lgb_curves[1],
                                              "best_epoch": lgb_best[-1]},
                                 "train_seconds_backtest": time.time() - t0}
        print(f"  best trees per target {dict(zip(PRICE_TYPES, lgb_best))} ({time.time() - t0:.0f}s)")

    if not only or "Chronos-Bolt" in only:
        print(f"\nForecasting with {CHRONOS_MODEL} (zero-shot, context {CHRONOS_CONTEXT} days)")
        t0 = time.time()
        val_returns["Chronos-Bolt"], val_up["Chronos-Bolt"] = chronos_returns(df, val_df["Date"].values)
        test_returns["Chronos-Bolt"], test_up["Chronos-Bolt"] = chronos_returns(df, test_df["Date"].values)
        histories["Chronos-Bolt"] = {"backtest": {"train_loss": [], "val_loss": [], "best_epoch": 0},
                                     "train_seconds_backtest": time.time() - t0}
        print(f"  {len(val_df) + len(test_df)} days x 4 series forecast in {time.time() - t0:.0f}s")

    for name, ret in test_returns.items():
        backtest_metrics[name] = score(actual_test, returns_to_prices(ret, base_close_test), base_close_test)

    # Naive baseline: tomorrow's O/H/L/C = today's close
    naive = {p: base_close_test.copy() for p in PRICE_TYPES}
    backtest_metrics["Naive (today's close)"] = score(actual_test, naive, base_close_test)
    # direction is undefined for a zero-change forecast
    backtest_metrics["Naive (today's close)"]["DirectionAccuracy"] = None
    if only:
        # keep the stored scores of the models that were not retrained, in the usual model order
        merged = {}
        for name in list(train_cfg) + ["Naive (today's close)"]:
            if name in backtest_metrics:
                merged[name] = backtest_metrics[name]
            elif name in old["backtest"]["metrics"]:
                merged[name] = old["backtest"]["metrics"][name]
        backtest_metrics = merged

    # Validation/test predictions of every model are stored so --only runs can rebuild the ensembles
    store_path = OUT_DIR / "model_predictions.npz"
    if only:
        if not store_path.exists():
            raise SystemExit("outputs/model_predictions.npz not found: run a full training once before using --only.")
        z = np.load(store_path)
        stores = {"val": val_returns, "test": test_returns, "valup": val_up, "testup": test_up}
        for key in z.files:
            kind, name = key.split("__", 1)
            target = stores[kind]
            expected = len(val_df) if kind.startswith("val") else len(test_df)
            if name not in target and name in train_cfg and len(z[key]) == expected:
                target[name] = z[key]
    np.savez(store_path, **{f"val__{n}": v for n, v in val_returns.items()},
             **{f"test__{n}": t for n, t in test_returns.items()},
             **{f"valup__{n}": v for n, v in val_up.items()},
             **{f"testup__{n}": t for n, t in test_up.items()})

    # ---- Ensembles (weights come from the VALIDATION period; the test period stays unseen) ----
    candidates = [n for n in train_cfg if n in val_returns and n in test_returns]
    base_close_val = val_df["Price"].values
    actual_val = {p: base_close_val * (1 + y_val_raw[:, j]) for j, p in enumerate(PRICE_TYPES)}
    val_avg_rmse = {}
    for n in candidates:
        prices = returns_to_prices(val_returns[n], base_close_val)
        val_avg_rmse[n] = float(np.mean([np.sqrt(np.mean((prices[p] - actual_val[p]) ** 2)) for p in PRICE_TYPES]))
    ens_names = sorted(candidates, key=val_avg_rmse.get)[:ENSEMBLE_TOP_K]
    print(f"\nEnsemble members (top {ENSEMBLE_TOP_K} by validation avg RMSE): "
          + ", ".join(f"{n} (${val_avg_rmse[n]:.2f})" for n in ens_names))
    val_rmse = {n: float(np.sqrt(np.mean((val_returns[n] - y_val_raw) ** 2))) for n in ens_names}
    inv = np.array([1 / val_rmse[n] for n in ens_names])
    inv_rmse_weights = {n: float(w) for n, w in zip(ens_names, inv / inv.sum())}
    stacking_weights = fit_stacking(val_returns, y_val_raw, ens_names)

    ensemble_test_returns = {
        "Ensemble (mean)": np.mean([test_returns[n] for n in ens_names], axis=0),
        "Ensemble (RMSE-weighted)": np.sum([inv_rmse_weights[n] * test_returns[n] for n in ens_names], axis=0),
        "Ensemble (stacked)": blend_returns(test_returns, stacking_weights),
    }
    for name, ret in ensemble_test_returns.items():
        backtest_metrics[name] = score(actual_test, returns_to_prices(ret, base_close_test), base_close_test)

    # Ensemble P(up): same members and weights; the stacked ensemble gives its "Naive" share a neutral 0.5
    def ensemble_up(ups):
        close_w = stacking_weights["Close"]
        return {
            "Ensemble (mean)": np.mean([ups[n] for n in ens_names], axis=0),
            "Ensemble (RMSE-weighted)": np.sum([inv_rmse_weights[n] * np.asarray(ups[n]) for n in ens_names], axis=0),
            "Ensemble (stacked)": close_w["Naive"] * 0.5 + np.sum([close_w[n] * np.asarray(ups[n])
                                                                   for n in ens_names], axis=0),
        }

    test_up_all = {**{n: test_up[n] for n in train_cfg if n in test_up}, **ensemble_up(test_up)}
    val_up_all = {**{n: val_up[n] for n in train_cfg if n in val_up}, **ensemble_up(val_up)}
    classification = {}
    for name, p_up in test_up_all.items():
        classification[name] = classification_metrics(p_up, y_up_test)
        classification[name]["val_accuracy"] = classification_metrics(val_up_all[name], y_up_val)["accuracy"]
    classification["Always up (baseline)"] = {"accuracy": float(y_up_test.mean() * 100),
                                              "val_accuracy": float(y_up_val.mean() * 100),
                                              "auc": 0.5, "precision_up": float(y_up_test.mean() * 100),
                                              "recall_up": 100.0, "pred_up_share": 100.0}
    print("\nUp/Down classification on the hold-out (sorted by accuracy):")
    print(pd.DataFrame(classification).T[["accuracy", "auc", "precision_up", "recall_up", "pred_up_share",
                                          "val_accuracy"]].sort_values("accuracy", ascending=False)
          .to_string(float_format="%.2f"))

    regime_rmse = {}
    for name, ret in [("Naive (today's close)", np.zeros((len(test_df), 4)))] + list(test_returns.items()) \
            + list(ensemble_test_returns.items()):
        prices = returns_to_prices(ret, base_close_test)
        regime_rmse[name] = {
            REGIME_NAMES[k]: float(np.mean([np.sqrt(np.mean((prices[p][test_regime == k]
                                                             - actual_test[p][test_regime == k]) ** 2))
                                            for p in PRICE_TYPES])) if np.any(test_regime == k) else None
            for k in range(3)}

    print("\nStacking weights (validation-fitted, per target):")
    print(pd.DataFrame(stacking_weights).loc[lambda t: (t > 0.005).any(axis=1)].to_string(float_format="%.3f"))
    print_leaderboard(backtest_metrics)

    bt_df = pd.DataFrame({"Date": test_df["Date"].values, "Close_t": base_close_test,
                          "Actual_Next_Close": actual_test["Close"]})
    # the forecast is for the next trading day; store the date the forecast refers to
    bt_df["Target_Date"] = labelled["Date"].shift(-1).iloc[split_idx:].values
    for name, ret in list(test_returns.items()) + list(ensemble_test_returns.items()):
        bt_df[name] = returns_to_prices(ret, base_close_test)["Close"]
    bt_df.to_csv(OUT_DIR / "backtest_predictions.csv", index=False)

    # --------------------------------------------------------
    # PHASE B: REFIT ON 100% OF THE DATA AND PREDICT THE NEXT DAY
    # --------------------------------------------------------
    print("\n" + "=" * 70 + "\nPHASE B: REFIT ON 100% OF DATA\n" + "=" * 70)

    regimes_b = fit_regimes(labelled)
    labelled, full_seq, infer_row = (add_regimes(f, regimes_b) for f in (labelled, full_seq, infer_row))

    pre_b = make_preprocessor()
    X_all = to_sequence(pre_b.fit_transform(labelled[FEATURE_COLUMNS]))
    X_inf = to_sequence(pre_b.transform(infer_row[FEATURE_COLUMNS]))
    tscaler_b = StandardScaler()
    y_all = tscaler_b.fit_transform(labelled[TARGET_COLUMNS].values)
    full_loader = make_loader(X_all, y_all, shuffle=False)
    Xs_b = pre_b.transform(full_seq[FEATURE_COLUMNS])
    all_idx = np.arange(WINDOW_SIZE - 1, len(labelled))
    data_b = {
        "row": (full_loader, X_inf),
        "window": (make_loader(make_windows(Xs_b, all_idx), y_all[all_idx], shuffle=False),
                   make_windows(Xs_b, [len(labelled)])),
    }
    eemd_view = None
    if train_eemd:
        all_idx_e = all_idx[has_comps[all_idx]]
        scale_b = comps[all_idx_e].reshape(-1, 2).std(axis=0)
        data_b["eemd"] = (make_loader(eemd_inputs(comps, Xs_b, all_idx_e, scale_b), y_all[all_idx_e], shuffle=False),
                          eemd_inputs(comps, Xs_b, [len(labelled)], scale_b))
        # decomposition of the window used for the forecast, as % of the last close, for the UI
        last_comp = comps[len(labelled)]
        eemd_view = {"dates": [str(d.date()) for d in full_seq["Date"].iloc[-WINDOW_SIZE:]],
                     "imf1_pct": (last_comp[:, 0] * 100).tolist(), "imf2_pct": (last_comp[:, 1] * 100).tolist(),
                     "eemd_window": EEMD_WINDOW, "trials": EEMD_TRIALS}
    if train_prices:
        data_b["prices"] = (make_loader(price_window_inputs(full_seq, Xs_b, all_idx), y_all[all_idx], shuffle=False),
                            price_window_inputs(full_seq, Xs_b, [len(labelled)]))
    helformer_view = None
    attention_dates = [str(d.date()) for d in full_seq["Date"].iloc[-WINDOW_SIZE:]]
    attention_weights = None
    print(f"Training rows: {len(labelled)} ({labelled['Date'].min().date()} to {labelled['Date'].max().date()})")

    last_close = float(infer_row["Price"].iloc[0])
    predictions = {}
    next_up = {}   # P(close up) for the forecast day
    final_models = {}
    for name, model in build_models(INPUT_SIZE, OUTPUT_SIZE, sag_hidden).items():
        if only and name not in only:
            continue
        epochs, lr = train_cfg[name]
        print(f"\nRefitting {name} on all data (epochs={epochs}, lr={lr:.5f})")
        t0 = time.time()
        tl, Xi = data_b[input_kind(name)]
        if hasattr(model, "up_threshold"):
            model.up_threshold = float(-tscaler_b.mean_[3] / tscaler_b.scale_[3])
        model, hist = train_model(model, tl, None, epochs, lr, tscaler_b, name=name)
        next_up[name] = float(up_probability(model, Xi)[0])
        histories[name]["full"] = hist
        histories[name]["train_seconds_full"] = time.time() - t0
        pred_returns = tscaler_b.inverse_transform(predict_model(model, Xi))
        if name == "CNN-BiLSTM-Attn":
            attention_weights = model.base.last_attention[0].cpu().numpy().tolist()
        if name == "Helformer":
            dec = model.base.decomposition
            with torch.no_grad():
                x_prices = torch.tensor(Xi[:, :, :4]).to(device)
                deseason, lvl, seas = dec(x_prices)
            helformer_view = {
                "alpha": torch.sigmoid(dec.alpha_raw).tolist(), "gamma": torch.sigmoid(dec.gamma_raw).tolist(),
                "channels": ["Open", "High", "Low", "Close"], "dates": attention_dates,
                "close_pct": ((x_prices[0, :, 3] - 1) * 100).tolist(),
                "level_pct": ((lvl[0, :, 3] - 1) * 100).tolist(),
                "seasonal": seas[0, :, 3].tolist(),
                "deseasonalized_pct": ((deseason[0, :, 3] - 1) * 100).tolist(),
            }
        prices = returns_to_prices(pred_returns, np.array([last_close]))
        predictions[name] = {p: float(prices[p][0]) for p in PRICE_TYPES}
        final_models[name] = model

    if not only or "LightGBM" in only:
        import lightgbm as lgb

        print(f"\nRefitting LightGBM on all data (trees per target {dict(zip(PRICE_TYPES, lgb_best))})")
        t0 = time.time()
        y_full = labelled[TARGET_COLUMNS].values
        lgb_full, full_curve = [], []
        for j, n_trees in enumerate(lgb_best):
            evals = {}
            m = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": n_trees})
            m.fit(labelled[FEATURE_COLUMNS], y_full[:, j], eval_set=[(labelled[FEATURE_COLUMNS], y_full[:, j])],
                  eval_names=["train"], eval_metric="l2", callbacks=[lgb.record_evaluation(evals)])
            lgb_full.append(m)
            if j == len(PRICE_TYPES) - 1:
                full_curve = evals["train"]["l2"]
        pred_returns = np.column_stack([m.predict(infer_row[FEATURE_COLUMNS]) for m in lgb_full])
        prices = returns_to_prices(pred_returns, np.array([last_close]))
        predictions["LightGBM"] = {p: float(prices[p][0]) for p in PRICE_TYPES}
        clf_full = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": lgb_clf_best})
        clf_full.fit(labelled[FEATURE_COLUMNS], labelled["Next_Close"].values > 0)
        next_up["LightGBM"] = float(clf_full.predict_proba(infer_row[FEATURE_COLUMNS])[0, 1])
        lgb_trees = {"regression": dict(zip(PRICE_TYPES, [int(t) for t in lgb_best])), "classifier": int(lgb_clf_best)}
        histories["LightGBM"]["full"] = {"train_loss": full_curve}
        histories["LightGBM"]["train_seconds_full"] = time.time() - t0

    if not only or "Chronos-Bolt" in only:
        t0 = time.time()
        pred_returns, chronos_up = chronos_returns(df, [last_date])
        next_up["Chronos-Bolt"] = float(chronos_up[0])
        prices = returns_to_prices(pred_returns, np.array([last_close]))
        predictions["Chronos-Bolt"] = {p: float(prices[p][0]) for p in PRICE_TYPES}
        histories["Chronos-Bolt"]["full"] = {"train_loss": []}
        histories["Chronos-Bolt"]["train_seconds_full"] = time.time() - t0

    if only:
        # merge with the stored predictions of the models that were not retrained
        predictions = {name: predictions[name] if name in predictions else old["predictions"][name]
                       for name in train_cfg if name in predictions or name in old["predictions"]}
        if attention_weights is None and old.get("cnn_bilstm_attention"):
            attention_weights = old["cnn_bilstm_attention"]["weights"]
        if eemd_view is None:
            eemd_view = old.get("eemd")
        if helformer_view is None:
            helformer_view = old.get("helformer")
        for name, info in (old.get("classification") or {}).items():
            if name in train_cfg and name not in next_up and info.get("p_up_next_day") is not None:
                next_up[name] = info["p_up_next_day"]
    # Ensembles are linear in returns, so they can be built directly from the predicted prices
    mean_ens = {p: float(np.mean([predictions[n][p] for n in ens_names])) for p in PRICE_TYPES}
    weighted_ens = {p: float(sum(inv_rmse_weights[n] * predictions[n][p] for n in ens_names)) for p in PRICE_TYPES}
    stacked_ens = {p: float(sum(w * (last_close if n == "Naive" else predictions[n][p])
                                for n, w in stacking_weights[p].items())) for p in PRICE_TYPES}
    for ens in (mean_ens, weighted_ens, stacked_ens):
        ens["High"] = max(ens["High"], ens["Open"], ens["Close"])
        ens["Low"] = min(ens["Low"], ens["Open"], ens["Close"])
    predictions["Ensemble (mean)"] = mean_ens
    predictions["Ensemble (RMSE-weighted)"] = weighted_ens
    predictions["Ensemble (stacked)"] = stacked_ens

    next_up.update({n: float(v) for n, v in ensemble_up(next_up).items()})
    for name, p in next_up.items():
        if name in classification:
            classification[name]["p_up_next_day"] = p
    print(f"\nP(close up on {TARGET_DATE.date()}): "
          + ", ".join(f"{n} {p:.1%}" for n, p in next_up.items()))

    print(f"\nPredictions for {TARGET_DATE.date()} (last close {last_close:,.2f} on {last_date.date()}):")
    print(pd.DataFrame(predictions).T[PRICE_TYPES].to_string(float_format="%.2f"))
    print_leaderboard(backtest_metrics)

    market_stats = compute_market_stats(df, infer_row)
    last = infer_row.iloc[0]
    daily_ret = df["Price"].pct_change().dropna()

    training = {
        name: {"epochs": train_cfg[name][0], "learning_rate": train_cfg[name][1],
               "backtest_train_loss": h["backtest"]["train_loss"],
               "backtest_val_loss": h["backtest"]["val_loss"],
               "best_epoch": h["backtest"].get("best_epoch"),
               "full_train_loss": h["full"]["train_loss"],
               "train_seconds": h["train_seconds_backtest"] + h["train_seconds_full"]}
        for name, h in histories.items()
    }
    if only:
        training = {name: training[name] if name in training else old["training"][name]
                    for name in train_cfg if name in training or name in old["training"]}

    # Regime summary for the UI (Phase B mixture, fitted on all labelled rows)
    all_regime = regime_probabilities(regimes_b, labelled).argmax(axis=1)
    tail = full_seq.tail(250)
    regime_view = {
        "names": REGIME_NAMES,
        "inputs": REGIME_INPUTS,
        "current": dict(zip(REGIME_NAMES, regime_probabilities(regimes_b, infer_row)[0].round(4).tolist())),
        "profile": {
            REGIME_NAMES[k]: {
                "share_of_days_pct": float(np.mean(all_regime == k) * 100),
                "avg_20d_return_pct": float(labelled["Return_20"].values[all_regime == k].mean() * 100),
                "avg_daily_vol_pct": float(labelled["Volatility_14"].values[all_regime == k].mean() * 100),
                "next_day_up_pct": float(np.mean(labelled["Next_Close"].values[all_regime == k] > 0) * 100),
                "avg_next_day_return_pct": float(labelled["Next_Close"].values[all_regime == k].mean() * 100),
            } for k in range(3)},
        "timeline": {"Date": [str(d.date()) for d in tail["Date"]], "Close": tail["Price"].round(2).tolist(),
                     "Regime": [REGIME_NAMES[k] for k in regime_probabilities(regimes_b, tail).argmax(axis=1)]},
        "test_days": {REGIME_NAMES[k]: int(np.sum(test_regime == k)) for k in range(3)},
        "backtest_avg_rmse": regime_rmse,
    }

    recent = df.tail(120)
    results = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "first_date": str(df["Date"].min().date()),
        "last_date": str(last_date.date()),
        "target_date": str(TARGET_DATE.date()),
        "n_rows": int(len(df)),
        "n_training_rows_full": int(len(labelled)),
        "n_features": INPUT_SIZE,
        "features": FEATURE_COLUMNS,
        "last_bar": {"Open": float(last["Open"]), "High": float(last["High"]),
                     "Low": float(last["Low"]), "Close": last_close},
        "predictions": predictions,
        "ensemble_weights": inv_rmse_weights,
        "stacking_weights": stacking_weights,
        "backtest": {
            "train_rows": int(len(tr_df)), "val_rows": int(len(val_df)), "test_rows": int(len(test_df)),
            "test_start": str(test_df["Date"].min().date()), "test_end": str(test_df["Date"].max().date()),
            "val_start": str(val_df["Date"].min().date()), "val_end": str(val_df["Date"].max().date()),
            "metrics": backtest_metrics,
        },
        "ga": {"best_hidden_size": int(sag_hidden), "best_learning_rate": float(sag_lr),
               "best_epochs": int(sag_epochs), "best_val_rmse": float(best_fit), "log": ga_log},
        "training": training,
        "market_stats": market_stats,
        "eemd": eemd_view,
        "helformer": helformer_view,
        "classification": classification,
        "lightgbm_trees": lgb_trees if (not only or "LightGBM" in only) else old.get("lightgbm_trees"),
        "models_trained_through": str(last_date.date()),
        "ensemble_members": ens_names,
        "ensemble_selection": {"top_k": ENSEMBLE_TOP_K, "criterion": "lowest average $ RMSE on the validation period",
                               "val_avg_rmse": val_avg_rmse},
        "regimes": regime_view,
        "cnn_bilstm_attention": {"window_size": WINDOW_SIZE, "dates": attention_dates,
                                 "weights": attention_weights},
        "recent_ohlc": {
            "Date": [str(d.date()) for d in recent["Date"]],
            "Open": recent["Open"].tolist(), "High": recent["High"].tolist(),
            "Low": recent["Low"].tolist(), "Close": recent["Price"].tolist(),
        },
        "daily_returns_pct": (daily_ret.tail(1000) * 100).round(4).tolist(),
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'results.json'} and {OUT_DIR / 'backtest_predictions.csv'}")
    export_backtest_ohlc()
    if not only:  # save the final models so new days can be predicted with --predict (no retraining)
        save_artifacts(final_models, pre_b, tscaler_b, regimes_b, scale_b, lgb_full, clf_full, last_date, sag_hidden)
        _log_forecasts(str(last_date.date()), str(TARGET_DATE.date()), predictions, classification)


if __name__ == "__main__":
    from xau_15m_kaggle_train import main as train_15m_models

    train_15m_models()
