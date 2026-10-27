# -*- coding: utf-8 -*-

"""
XAU/USD 15-minute SAG-LSTM training
-----------------------------------

ONLY SAG_LSTM is used.

Predicts:
    1. Next 15-minute High
    2. Next 15-minute Low
    3. Next day Open
    4. Next day High
    5. Next day Low
    6. Next day Close
    7. Next 15-minute Close

Targets are log-returns relative to the current Close.

Dataset:
    XAU_15m_data.csv

Works with:
    Kaggle / Colab / Local
"""



import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

ROWS = 494236

WINDOW = 32
# 32 x 15 minutes = 8 hours of historical data

DAY_BARS = 96
# 96 x 15 minutes = 24 hours

EPOCHS = 54

BATCH_SIZE = 8192

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-3

GRAD_CLIP = 1.0

SEED = 42


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "n15_High",
    "n15_Low",
    "nd_Open",
    "nd_High",
    "nd_Low",
    "nd_Close",
    "n15_Close"
]


# ============================================================
# REGIME FEATURES
# ============================================================

REGIME_INPUTS = [
    "SMA_30_Distance",
    "EMA_30_Distance",
    "Momentum_Ratio",
    "Volatility_14",
    "ATR_Ratio",
    "Return_20"
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUT_DIR = (
    Path("/kaggle/working")
    if Path("/kaggle/working").exists()
    else Path.cwd() / "outputs"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# GATED RESIDUAL NETWORK
# ============================================================

class GatedResidualNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0.1
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.gate = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.skip = (
            nn.Linear(
                input_size,
                hidden_size
            )
            if input_size != hidden_size
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

    def forward(self, x):

        residual = self.skip(x)

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        gate = torch.sigmoid(
            self.gate(x)
        )

        x = gate * x

        x = x + residual

        x = self.norm(x)

        return x


# ============================================================
# SAG-LSTM
# ============================================================

class SAG_LSTM(nn.Module):
    """
    Self-Attention Gated LSTM

    Structure:

        Input
          ↓
        LSTM
          ↓
        Self-Attention
          ↓
        Context Vector
          ↓
        Gated Residual Network
          ↓
        Linear
          ↓
        7 predictions
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        output_size=1,
        dropout=0.2
    ):

        super().__init__()

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0
            )
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention_linear = nn.Linear(
            hidden_size,
            1
        )

        # ----------------------------------------------------
        # Gated Residual Network
        # ----------------------------------------------------

        self.grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # ----------------------------------------------------
        # Final output layer
        # ----------------------------------------------------

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):

        # ====================================================
        # 1. LSTM
        # ====================================================

        lstm_out, _ = self.lstm(x)

        # Shape:
        # [batch, sequence_length, hidden_size]

        # ====================================================
        # 2. SELF ATTENTION
        # ====================================================

        attention_scores = self.attention_linear(
            lstm_out
        )

        attention_scores = torch.tanh(
            attention_scores
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        # ====================================================
        # 3. CONTEXT VECTOR
        # ====================================================

        context_vector = torch.sum(
            lstm_out * attention_weights,
            dim=1
        )

        # Shape:
        # [batch, hidden_size]

        # ====================================================
        # 4. GATED RESIDUAL NETWORK
        # ====================================================

        context_vector = self.grn(
            context_vector
        )

        # ====================================================
        # 5. FINAL OUTPUT
        # ====================================================

        output = self.fc(
            context_vector
        )

        return output


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path, rows):

    df = pd.read_csv(
        path,
        sep=";"
    )

    # Expected columns:
    # Date;Open;High;Low;Close;Volume

    df.columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    # --------------------------------------------------------
    # Convert Date
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%Y.%m.%d %H:%M"
    )

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numerical_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    df[numerical_columns] = (
        df[numerical_columns]
        .apply(pd.to_numeric)
    )

    # --------------------------------------------------------
    # Sort data
    # --------------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .drop_duplicates("Date")
        .tail(rows)
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(df)} | "
        f"{df.Date.iloc[0]} -> "
        f"{df.Date.iloc[-1]}"
    )

    return df


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(df):

    d = df.copy()

    # --------------------------------------------------------
    # Basic price references
    # --------------------------------------------------------

    c = d["Close"]

    prev = d["Close"].shift(1)

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    d["Return"] = c.pct_change()

    d["Log_Return"] = np.log(
        c / prev
    )

    # --------------------------------------------------------
    # Open / High / Low returns
    # --------------------------------------------------------

    for col in [
        "Open",
        "High",
        "Low"
    ]:

        d[f"{col}_Return"] = np.log(
            d[col] / prev
        )

    # --------------------------------------------------------
    # High-Low range
    # --------------------------------------------------------

    d["HL_Range"] = (
        d["High"] - d["Low"]
    ) / c

    # --------------------------------------------------------
    # SMA / EMA
    # --------------------------------------------------------

    for w in [
        7,
        14,
        30
    ]:

        d[f"SMA_{w}_Distance"] = (
            c /
            c.rolling(w).mean()
            - 1
        )

        d[f"EMA_{w}_Distance"] = (
            c /
            c.ewm(
                span=w,
                adjust=False
            ).mean()
            - 1
        )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    d["Volatility_7"] = (
        d["Return"]
        .rolling(7)
        .std()
    )

    d["Volatility_14"] = (
        d["Return"]
        .rolling(14)
        .std()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = c.diff()

    gain = (
        delta
        .clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss

    d["RSI"] = (
        100 - 100 / (1 + rs)
    ) / 100

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    macd = (
        c.ewm(
            span=12,
            adjust=False
        ).mean()
        -
        c.ewm(
            span=26,
            adjust=False
        ).mean()
    )

    sig = (
        macd
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    d["MACD_Relative"] = (
        macd / c
    )

    d["MACD_Signal_Relative"] = (
        sig / c
    )

    d["MACD_Hist_Relative"] = (
        (macd - sig) / c
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - prev).abs(),
            (d["Low"] - prev).abs()
        ],
        axis=1
    ).max(axis=1)

    d["ATR_Ratio"] = (
        tr
        .rolling(14)
        .mean()
        / c
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    mid = (
        c
        .rolling(20)
        .mean()
    )

    sd = (
        c
        .rolling(20)
        .std()
    )

    d["BB_Width"] = (
        4 * sd / mid
    )

    d["BB_Position"] = (
        c - (mid - 2 * sd)
    ) / (4 * sd)

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    d["Momentum_Ratio"] = (
        c.diff(7) / c
    )

    # --------------------------------------------------------
    # 20-period return
    # --------------------------------------------------------

    d["Return_20"] = (
        c.pct_change(20)
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    d["Log_Volume"] = (
        np.log1p(d["Volume"])
        -
        np.log1p(d["Volume"])
        .rolling(
            96,
            min_periods=1
        )
        .mean()
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    hour = (
        d["Date"].dt.hour
        +
        d["Date"].dt.minute / 60
    )

    dow = d["Date"].dt.dayofweek

    d["Hour_Sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    d["Hour_Cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    d["DayOfWeek_Sin"] = np.sin(
        2 * np.pi * dow / 7
    )

    d["DayOfWeek_Cos"] = np.cos(
        2 * np.pi * dow / 7
    )

    # --------------------------------------------------------
    # Indicator warm-up
    # --------------------------------------------------------

    d = (
        d
        .iloc[30:]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Clean NaN / Inf
    # --------------------------------------------------------

    d = d.replace(
        [np.inf, -np.inf],
        np.nan
    )

    d = d.ffill().fillna(0)

    return d


# ============================================================
# BASE FEATURES
# ============================================================

BASE_FEATURES = [

    "Return",
    "Log_Return",

    "Open_Return",
    "High_Return",
    "Low_Return",

    "HL_Range",

    "SMA_7_Distance",
    "SMA_14_Distance",
    "SMA_30_Distance",

    "EMA_7_Distance",
    "EMA_14_Distance",
    "EMA_30_Distance",

    "Volatility_7",
    "Volatility_14",

    "RSI",

    "MACD_Relative",
    "MACD_Signal_Relative",
    "MACD_Hist_Relative",

    "ATR_Ratio",

    "BB_Width",
    "BB_Position",

    "Momentum_Ratio",

    "Log_Volume",

    "Hour_Sin",
    "Hour_Cos",

    "DayOfWeek_Sin",
    "DayOfWeek_Cos"
]


# ============================================================
# MAKE TARGETS
# ============================================================

def make_targets(d):

    """
    Targets are log returns relative to
    the Close of bar t.

    Outputs:

        0 = next 15m High
        1 = next 15m Low
        2 = next day Open
        3 = next day High
        4 = next day Low
        5 = next day Close
        6 = next 15m Close
    """

    O = d["Open"].values

    H = d["High"].values

    L = d["Low"].values

    C = d["Close"].values

    n = len(d)

    Y = np.full(
        (
            n,
            len(TARGETS)
        ),
        np.nan
    )

    for t in range(
        n - DAY_BARS
    ):

        f = slice(
            t + 1,
            t + 1 + DAY_BARS
        )

        future_values = np.array(
            [
                H[t + 1],
                L[t + 1],

                O[t + 1],

                H[f].max(),
                L[f].min(),

                C[t + DAY_BARS],

                C[t + 1]
            ]
        )

        Y[t] = np.log(
            future_values / C[t]
        )

    return Y


# ============================================================
# CREATE WINDOWS
# ============================================================

def windows(X, ends):

    return np.stack(
        [
            X[
                e - WINDOW + 1:
                e + 1
            ]
            for e in ends
        ]
    )


# ============================================================
# DATALOADER
# ============================================================

def loader(
    X,
    y,
    shuffle
):

    ds = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        ),
        torch.tensor(
            y,
            dtype=torch.float32
        )
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


# ============================================================
# PREDICT
# ============================================================

def predict(model, X):

    model.eval()

    predictions = []

    with torch.no_grad():

        for i in range(
            0,
            len(X),
            1024
        ):

            xb = torch.tensor(
                X[i:i + 1024],
                dtype=torch.float32
            ).to(device)

            pred = model(xb)

            predictions.append(
                pred.cpu().numpy()
            )

    return np.concatenate(
        predictions
    )


# ============================================================
# RETURNS -> PRICES
# ============================================================

def to_prices(
    ret,
    base_close
):

    return (
        np.exp(ret)
        *
        base_close[:, None]
    )


# ============================================================
# FIX OHLC
# ============================================================

def fix_ohlc(p):

    # --------------------------------------------------------
    # Next 15m
    # --------------------------------------------------------

    n15 = {
        "High": p["n15_High"],
        "Low": p["n15_Low"],
        "Close": p["n15_Close"]
    }

    n15["High"] = max(
        n15["High"],
        n15["Close"]
    )

    n15["Low"] = min(
        n15["Low"],
        n15["Close"]
    )

    # --------------------------------------------------------
    # Next day
    # --------------------------------------------------------

    nd = {
        "Open": p["nd_Open"],
        "High": p["nd_High"],
        "Low": p["nd_Low"],
        "Close": p["nd_Close"]
    }

    nd["High"] = max(
        nd["High"],
        nd["Open"],
        nd["Close"]
    )

    nd["Low"] = min(
        nd["Low"],
        nd["Open"],
        nd["Close"]
    )

    return {

        "next_15m": {
            k: round(float(v), 2)
            for k, v in n15.items()
        },

        "next_day": {
            k: round(float(v), 2)
            for k, v in nd.items()
        }
    }


# ============================================================
# SAVE TEST REPORT
# ============================================================

def save_test_report(
    dates,
    actual,
    predicted,
    metrics,
    targets,
    out_dir
):

    names = {

        "n15_High":
            "Next 15m High",

        "n15_Low":
            "Next 15m Low",

        "n15_Close":
            "Next 15m Close",

        "nd_Open":
            "Next day Open",

        "nd_High":
            "Next day High",

        "nd_Low":
            "Next day Low",

        "nd_Close":
            "Next day Close"
    }

    col = {
        t: j
        for j, t in enumerate(targets)
    }

    # ========================================================
    # ACTUAL VS PREDICTED TABLE
    # ========================================================

    table = pd.DataFrame({
        "Date": dates
    })

    for t in targets:

        table[
            f"actual_{t}"
        ] = actual[
            :, col[t]
        ]

        table[
            f"pred_{t}"
        ] = predicted[
            :, col[t]
        ]

        table[
            f"err_{t}"
        ] = (
            table[f"pred_{t}"]
            -
            table[f"actual_{t}"]
        )

    table.to_csv(
        out_dir /
        "test_actual_vs_predicted.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        {
            "MAE": metrics["mae"],
            "Direction_Accuracy":
                metrics["direction_accuracy"]
        }
    )

    summary.to_csv(
        out_dir /
        "test_summary_mae.csv"
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    bias = pd.DataFrame({

        "mean_error_pred_minus_actual": {
            t:
            table[f"err_{t}"].mean()
            for t in targets
        },

        "MAE": {
            t:
            table[f"err_{t}"].abs().mean()
            for t in targets
        },

        "MAPE_%": {
            t:
            (
                table[f"err_{t}"].abs()
                /
                table[f"actual_{t}"]
            ).mean()
            * 100

            for t in targets
        }
    })

    bias.to_csv(
        out_dir /
        "test_error_analysis.csv"
    )

    print(
        "\nSAG_LSTM error on TEST data:"
    )

    print(
        bias.round(3).to_string()
    )

    # ========================================================
    # FIRST / LAST TEST ROWS
    # ========================================================

    print(
        "\nFirst / last 5 TEST rows:"
    )

    show = (
        ["Date"]
        +
        [
            f"{p}_{t}"
            for t in targets
            for p in (
                "actual",
                "pred"
            )
        ]
    )

    print(
        pd.concat(
            [
                table[show].head(5),
                table[show].tail(5)
            ]
        )
        .round(2)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # ACTUAL VS PREDICTED GRAPHS
    # ========================================================

    groups = {

        "next_15m": [
            "n15_High",
            "n15_Low",
            "n15_Close"
        ],

        "next_day": [
            "nd_Open",
            "nd_High",
            "nd_Low",
            "nd_Close"
        ]
    }

    for gname, tg in groups.items():

        for suffix, sl in [
            (
                "full_test",
                slice(None)
            ),
            (
                "last_300_bars",
                slice(-300, None)
            )
        ]:

            fig, axes = plt.subplots(
                len(tg),
                1,
                figsize=(
                    14,
                    3.2 * len(tg)
                ),
                sharex=True
            )

            axes = np.atleast_1d(
                axes
            )

            for ax, t in zip(
                axes,
                tg
            ):

                ax.plot(
                    table["Date"].iloc[sl],
                    table[
                        f"actual_{t}"
                    ].iloc[sl],
                    label="Actual",
                    lw=1.2
                )

                ax.plot(
                    table["Date"].iloc[sl],
                    table[
                        f"pred_{t}"
                    ].iloc[sl],
                    label="SAG_LSTM",
                    lw=1.0
                )

                ax.set_title(
                    f"{names[t]} - "
                    f"Actual vs SAG_LSTM "
                    f"({suffix.replace('_', ' ')})"
                )

                ax.set_ylabel(
                    "Price"
                )

                ax.grid(
                    alpha=0.3
                )

                ax.legend(
                    loc="upper left"
                )

            fig.tight_layout()

            fig.savefig(
                out_dir /
                f"test_{gname}_{suffix}.png",
                dpi=110
            )

            plt.close(fig)

    # ========================================================
    # SCATTER PLOTS
    # ========================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 8)
    )

    for ax, t in zip(
        axes.ravel(),
        targets
    ):

        a = table[
            f"actual_{t}"
        ]

        p = table[
            f"pred_{t}"
        ]

        ax.scatter(
            a,
            p,
            s=4,
            alpha=0.4
        )

        lim = [
            min(
                a.min(),
                p.min()
            ),
            max(
                a.max(),
                p.max()
            )
        ]

        ax.plot(
            lim,
            lim,
            "r--",
            lw=1
        )

        ax.set_title(
            names[t]
        )

        ax.set_xlabel(
            "Actual"
        )

        ax.set_ylabel(
            "Predicted"
        )

    # Hide unused subplot
    for ax in axes.ravel()[
        len(targets):
    ]:

        ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        out_dir /
        "test_scatter_actual_vs_predicted.png",
        dpi=110
    )

    plt.close(fig)

    print(
        "\nSaved test tables and graphs to:",
        out_dir
    )


# ============================================================
# MAIN
# ============================================================

def main(argv=None):

    global EPOCHS

    # ========================================================
    # ARGUMENTS
    # ========================================================

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default="//kaggle/input/datasets/ferdeenbakhtkag/xau-15m-data/XAU_15m_data.csv"
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=ROWS
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    args, _ = ap.parse_known_args(
        argv
    )

    EPOCHS = int(
        args.epochs
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SEED
    # ========================================================

    set_seed()

    print(
        "Device:",
        device
    )

    print(
        "Model: SAG_LSTM"
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    d = build_features(
        load_data(
            args.data,
            args.rows
        )
    )

    # ========================================================
    # TARGETS
    # ========================================================

    Y = make_targets(
        d
    )

    n = len(d)

    # ========================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    labelled = np.arange(
        WINDOW - 1,
        n - DAY_BARS
    )

    n_tr = int(
        len(labelled) * 0.8
    )

    n_va = int(
        len(labelled) * 0.1
    )

    tr = labelled[
        :n_tr
    ]

    # Leave DAY_BARS gap between
    # train and validation

    va = labelled[
        n_tr + DAY_BARS:
        n_tr + n_va
    ]

    # Leave DAY_BARS gap between
    # validation and test

    te = labelled[
        n_tr + n_va + DAY_BARS:
    ]

    print(
        f"train {len(tr)} | "
        f"val {len(va)} | "
        f"test {len(te)}"
    )

    # ========================================================
    # MARKET REGIMES
    # ========================================================

    print(
        "\nFitting market regimes..."
    )

    reg_scaler = StandardScaler().fit(
        d.loc[
            tr,
            REGIME_INPUTS
        ]
    )

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=3,
        random_state=SEED
    ).fit(
        reg_scaler.transform(
            d.loc[
                tr,
                REGIME_INPUTS
            ]
        )
    )

    # --------------------------------------------------------
    # Training regime labels
    # --------------------------------------------------------

    lab = gmm.predict(
        reg_scaler.transform(
            d.loc[
                tr,
                REGIME_INPUTS
            ]
        )
    )

    # --------------------------------------------------------
    # Order regimes by Return_20
    #
    # Lowest  = Bearish
    # Middle  = Sideways
    # Highest = Bullish
    # --------------------------------------------------------

    order = np.argsort(
        [
            d[
                "Return_20"
            ].values[tr][
                lab == k
            ].mean()

            for k in range(3)
        ]
    )

    # --------------------------------------------------------
    # Regime probabilities
    # --------------------------------------------------------

    proba = gmm.predict_proba(
        reg_scaler.transform(
            d[
                REGIME_INPUTS
            ]
        )
    )[:, order]

    print(
        "Market regimes created:"
    )

    print(
        "0 = Bearish"
    )

    print(
        "1 = Sideways"
    )

    print(
        "2 = Bullish"
    )

    # ========================================================
    # FEATURE SCALING
    # ========================================================

    print(
        "\nScaling features..."
    )

    feat_scaler = StandardScaler().fit(
        d.loc[
            tr,
            BASE_FEATURES
        ]
    )

    base_scaled = np.clip(
        feat_scaler.transform(
            d[
                BASE_FEATURES
            ]
        ),
        -8,
        8
    )

    # ========================================================
    # ADD REGIME PROBABILITIES
    # ========================================================

    feats = np.hstack(
        [
            base_scaled,
            proba
        ]
    ).astype(
        np.float32
    )

    print(
        "Total input features:",
        feats.shape[1]
    )

    # ========================================================
    # TARGET SCALING
    # ========================================================

    print(
        "\nScaling targets..."
    )

    y_scaler = StandardScaler().fit(
        Y[tr]
    )

    Ys = y_scaler.transform(
        np.nan_to_num(Y)
    )

    # ========================================================
    # CREATE WINDOWS
    # ========================================================

    print(
        "\nCreating windows..."
    )

    Xtr = windows(
        feats,
        tr
    )

    Xva = windows(
        feats,
        va
    )

    Xte = windows(
        feats,
        te
    )

    ytr = Ys[
        tr
    ]

    yva = Ys[
        va
    ]

    # ========================================================
    # CURRENT PRICE
    # ========================================================

    close = d[
        "Close"
    ].values

    last_close = close[-1]

    # Latest window
    x_last = feats[
        -WINDOW:
    ][None]

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_dl = loader(
        Xtr,
        ytr,
        shuffle=False
    )

    val_dl = loader(
        Xva,
        yva,
        shuffle=False
    )

    print(
        "\nTrain batches:",
        len(train_dl)
    )

    print(
        "Validation batches:",
        len(val_dl)
    )

    # ========================================================
    # BUILD SAG-LSTM
    # ========================================================

    print(
        "\nBuilding SAG_LSTM..."
    )

    model = SAG_LSTM(
        input_size=feats.shape[1],
        hidden_size=128,
        num_layers=2,
        output_size=len(TARGETS),
        dropout=0.2
    ).to(device)

    print(
        model
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # BEST MODEL
    # ========================================================

    best_val_loss = float(
        "inf"
    )

    best_state = None

    # ========================================================
    # TRAINING
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING SAG_LSTM TRAINING"
    )

    print(
        "=" * 70
    )

    training_start = time.time()

    for ep in range(
        EPOCHS
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        total_train_loss = 0.0

        for xb, yb in train_dl:

            xb = xb.to(
                device,
                non_blocking=True
            )

            yb = yb.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            pred = model(
                xb
            )

            # ------------------------------------------------
            # MSE loss
            # ------------------------------------------------

            loss = F.mse_loss(
                pred,
                yb
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            optimizer.step()

            total_train_loss += (
                loss.item()
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        validation_losses = []

        with torch.no_grad():

            for xb, yb in val_dl:

                xb = xb.to(
                    device,
                    non_blocking=True
                )

                yb = yb.to(
                    device,
                    non_blocking=True
                )

                pred = model(
                    xb
                )

                val_loss = F.mse_loss(
                    pred,
                    yb
                )

                validation_losses.append(
                    val_loss.item()
                )

        vl = float(
            np.mean(
                validation_losses
            )
        )

        train_loss = (
            total_train_loss
            /
            len(train_dl)
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            np.isfinite(vl)
            and
            vl < best_val_loss
        ):

            best_val_loss = vl

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

        # ====================================================
        # PRINT
        # ====================================================

        if (
            ep == 0
            or
            (ep + 1) % 5 == 0
            or
            ep == EPOCHS - 1
        ):

            print(
                f"Epoch "
                f"{ep + 1:3d}/{EPOCHS} | "
                f"Train Loss: "
                f"{train_loss:.6f} | "
                f"Val Loss: "
                f"{vl:.6f}"
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    training_time = (
        time.time()
        -
        training_start
    )

    print(
        "\nTraining completed in "
        f"{training_time:.0f} seconds"
    )

    # ========================================================
    # CHECK BEST MODEL
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "No valid validation loss "
            "was produced. "
            "Check the model output "
            "or validation data."
        )

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_state
    )

    model.to(
        device
    )

    print(
        "Best validation loss:",
        best_val_loss
    )

    # ========================================================
    # SAVE SAG-LSTM MODEL
    # ========================================================

    model_path = (
        OUT_DIR /
        "xau15m_SAG_LSTM.pt"
    )

    torch.save(
        {
            "model": best_state,

            "targets": TARGETS,

            "window": WINDOW,

            "hidden_size": 128,

            "num_layers": 2,

            "dropout": 0.2,

            "input_size": feats.shape[1]
        },
        model_path
    )

    print(
        "\nSaved model to:",
        model_path
    )

    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    print(
        "\nGenerating validation predictions..."
    )

    pred_va_scaled = predict(
        model,
        Xva
    )

    pred_va_returns = (
        y_scaler.inverse_transform(
            pred_va_scaled
        )
    )

    pred_va_prices = to_prices(
        pred_va_returns,
        close[va]
    )

    actual_va_prices = to_prices(
        Y[va],
        close[va]
    )

    # ========================================================
    # VALIDATION MAE
    # ========================================================

    val_mae = np.abs(
        pred_va_prices
        -
        actual_va_prices
    ).mean()

    print(
        f"SAG_LSTM validation "
        f"$MAE: {val_mae:.2f}"
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    print(
        "\nGenerating test predictions..."
    )

    pred_te_scaled = predict(
        model,
        Xte
    )

    pred_te_returns = (
        y_scaler.inverse_transform(
            pred_te_scaled
        )
    )

    pred_te_prices = to_prices(
        pred_te_returns,
        close[te]
    )

    # ========================================================
    # ACTUAL TEST PRICES
    # ========================================================

    actual_te_prices = to_prices(
        Y[te],
        close[te]
    )

    # ========================================================
    # NAIVE MODEL
    # ========================================================

    naive_te = np.repeat(
        close[te][:, None],
        len(TARGETS),
        axis=1
    )

    # ========================================================
    # METRICS
    # ========================================================

    mae = np.abs(
        pred_te_prices
        -
        actual_te_prices
    ).mean(
        axis=0
    )

    # --------------------------------------------------------
    # Direction accuracy
    #
    # Uses next 15m Close
    # --------------------------------------------------------

    predicted_direction = (
        pred_te_prices[:, -1]
        >
        close[te]
    )

    actual_direction = (
        actual_te_prices[:, -1]
        >
        close[te]
    )

    direction_accuracy = float(
        np.mean(
            predicted_direction
            ==
            actual_direction
        )
    )

    metrics = {

        "model": "SAG_LSTM",

        "mae": dict(
            zip(
                TARGETS,
                mae.round(3).tolist()
            )
        ),

        "direction_accuracy":
            direction_accuracy
    }

    # ========================================================
    # PRINT TEST METRICS
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "TEST RESULTS - SAG_LSTM"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Target':20s}"
        f"{'MAE':>15s}"
    )

    print(
        "-" * 40
    )

    for target, value in zip(
        TARGETS,
        mae
    ):

        print(
            f"{target:20s}"
            f"{value:15.2f}"
        )

    print(
        "-" * 40
    )

    print(
        "15m Direction Accuracy:",
        f"{direction_accuracy:.3f}"
    )

    print(
        "15m Direction Accuracy:",
        f"{direction_accuracy * 100:.2f}%"
    )

    # ========================================================
    # NAIVE MAE
    # ========================================================

    naive_mae = np.abs(
        naive_te
        -
        actual_te_prices
    ).mean(
        axis=0
    )

    print(
        "\nNaive MAE:"
    )

    for target, value in zip(
        TARGETS,
        naive_mae
    ):

        print(
            f"{target:20s}"
            f"{value:15.2f}"
        )

    # ========================================================
    # BACKTEST CSV
    # ========================================================

    backtest = pd.DataFrame(
        {
            "Date":
                d.Date.values[te],

            "Close":
                close[te]
        }
    )

    # --------------------------------------------------------
    # Actual columns
    # --------------------------------------------------------

    for j, target in enumerate(
        TARGETS
    ):

        backtest[
            f"actual_{target}"
        ] = actual_te_prices[
            :, j
        ]

    # --------------------------------------------------------
    # SAG-LSTM predictions
    # --------------------------------------------------------

    for j, target in enumerate(
        TARGETS
    ):

        backtest[
            f"pred_{target}"
        ] = pred_te_prices[
            :, j
        ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    backtest_path = (
        OUT_DIR /
        "backtest_15m_SAG_LSTM.csv"
    )

    backtest.to_csv(
        backtest_path,
        index=False
    )

    print(
        "\nSaved backtest to:",
        backtest_path
    )

    # ========================================================
    # FINAL LATEST PREDICTION
    # ========================================================

    print(
        "\nGenerating latest prediction..."
    )

    latest_pred_scaled = predict(
        model,
        x_last
    )

    latest_pred_returns = (
        y_scaler.inverse_transform(
            latest_pred_scaled
        )
    )

    latest_pred_prices = to_prices(
        latest_pred_returns,
        np.array([
            last_close
        ])
    )[0]

    # ========================================================
    # FIX OHLC
    # ========================================================

    latest_prediction = fix_ohlc(
        dict(
            zip(
                TARGETS,
                latest_pred_prices
            )
        )
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result = {

        "model":
            "SAG_LSTM",

        "data_rows":
            int(len(d)),

        "last_bar_time":
            str(
                d.Date.iloc[-1]
            ),

        "last_close":
            float(
                last_close
            ),

        "window":
            WINDOW,

        "epochs":
            EPOCHS,

        "hidden_size":
            128,

        "num_layers":
            2,

        "dropout":
            0.2,

        "validation_mae":
            float(val_mae),

        "test_metrics":
            metrics,

        "prediction":
            latest_prediction
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        OUT_DIR /
        "prediction_15m_SAG_LSTM.json"
    )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        "\nSaved prediction JSON to:",
        json_path
    )

    # ========================================================
    # SAVE TEST REPORT
    # ========================================================

    save_test_report(
        d.Date.values[te],
        actual_te_prices,
        pred_te_prices,
        metrics,
        TARGETS,
        OUT_DIR
    )

    # ========================================================
    # FINAL PREDICTION DISPLAY
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL SAG_LSTM PREDICTION"
    )

    print(
        "=" * 80
    )

    print(
        "From:",
        result["last_bar_time"]
    )

    print(
        "Last Close:",
        round(
            last_close,
            2
        )
    )

    print(
        "\nNext 15-minute:"
    )

    print(
        json.dumps(
            result[
                "prediction"
            ]["next_15m"],
            indent=2
        )
    )

    print(
        "\nNext 24-hour:"
    )

    print(
        json.dumps(
            result[
                "prediction"
            ]["next_day"],
            indent=2
        )
    )

    print(
        "\n15m Direction Accuracy:",
        f"{direction_accuracy * 100:.2f}%"
    )

    print(
        "\nAll outputs saved to:"
    )

    print(
        OUT_DIR
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()