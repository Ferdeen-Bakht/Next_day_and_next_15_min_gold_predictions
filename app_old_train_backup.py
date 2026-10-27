# -*- coding: utf-8 -*-
"""XAU/USD 15-minute training on the latest 20,000 bars (standalone: Kaggle / Colab / local).

Trains the neural models of train_predict.py (LSTM, GRU, Transformer, Transformer-MT, SAG-LSTM)
and predicts, from the newest bars:
  * next 15-minute bar: High / Low / Close
  * next day (next 96 bars = 24h of 15m bars): Open / High / Low / Close
No daily-data training. Every target is a log-return relative to the last close.

Kaggle:  python xau_15m_kaggle_train.py --data /kaggle/input/<dataset>/XAU_15m_data.csv
Local:   python xau_15m_kaggle_train.py --data XAU_15m_data.csv --rows 20000
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

ROWS = 80000
WINDOW = 32               # 15m bars of history per sample (8 hours)
DAY_BARS = 96             # 96 x 15min = next "day"
EPOCHS = 70
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
CLS_WEIGHT = 0.5          # loss = MSE(returns) + CLS_WEIGHT * BCE(next 15m close up/down)
ENSEMBLE_TOP_K = 3        # ensemble = mean of the K models with the lowest validation $ MAE
SEED = 42
MODEL_NAMES = ["LSTM", "GRU", "Transformer", "Transformer-MT", "SAG-LSTM"]

# order matters: the LAST output is the Up/Down reference (next 15m close)
TARGETS = ["n15_High", "n15_Low", "nd_Open", "nd_High", "nd_Low", "nd_Close", "n15_Close"]
REGIME_INPUTS = ["SMA_30_Distance", "EMA_30_Distance", "Momentum_Ratio", "Volatility_14", "ATR_Ratio", "Return_20"]

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "outputs"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# MODELS (copied unchanged from train_predict.py)
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












def build_models(input_size, output_size):
    set_seed()
    n = output_size + 1   # +1 unit = Up/Down logit
    base = {
        "LSTM": LSTMModel(input_size, output_size=n),
        "GRU": GRUModel(input_size, output_size=n),
        "Transformer": TimeSeriesTransformer(input_size, output_size=n),
        "SAG-LSTM": SAG_LSTM(input_size, hidden_size=128, output_size=n),
    }

    models = {k: WithUpDownHead(m, output_size) for k, m in base.items()}
    models["Transformer-MT"] = TransformerMultiTask(input_size, output_size=output_size)
    return {k: models[k] for k in MODEL_NAMES}

# ============================================================
# DATA + FEATURES
# ============================================================

def load_data(path, rows):
    df = pd.read_csv(path, sep=";")          # file is "Date;Open;High;Low;Close;Volume"
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y.%m.%d %H:%M")
    df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric)
    df = df.sort_values("Date").drop_duplicates("Date").tail(rows).reset_index(drop=True)
    print(f"Rows: {len(df)} | {df.Date.iloc[0]} -> {df.Date.iloc[-1]}")
    return df


def build_features(df):
    d = df.copy()
    c, prev = d["Close"], d["Close"].shift(1)
    d["Return"] = c.pct_change()
    d["Log_Return"] = np.log(c / prev)
    for col in ["Open", "High", "Low"]:
        d[f"{col}_Return"] = np.log(d[col] / prev)
    d["HL_Range"] = (d["High"] - d["Low"]) / c
    for w in (7, 14, 30):
        d[f"SMA_{w}_Distance"] = c / c.rolling(w).mean() - 1
        d[f"EMA_{w}_Distance"] = c / c.ewm(span=w, adjust=False).mean() - 1
    d["Volatility_7"] = d["Return"].rolling(7).std()
    d["Volatility_14"] = d["Return"].rolling(14).std()
    delta = c.diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = (100 - 100 / (1 + gain / loss)) / 100
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    d["MACD_Relative"], d["MACD_Signal_Relative"], d["MACD_Hist_Relative"] = macd / c, sig / c, (macd - sig) / c
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev).abs(), (d["Low"] - prev).abs()], axis=1).max(axis=1)
    d["ATR_Ratio"] = tr.rolling(14).mean() / c
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    d["BB_Width"] = 4 * sd / mid
    d["BB_Position"] = (c - (mid - 2 * sd)) / (4 * sd)
    d["Momentum_Ratio"] = c.diff(7) / c
    d["Return_20"] = c.pct_change(20)
    d["Log_Volume"] = np.log1p(d["Volume"]) - np.log1p(d["Volume"]).rolling(96, min_periods=1).mean()
    hour = d["Date"].dt.hour + d["Date"].dt.minute / 60
    dow = d["Date"].dt.dayofweek
    d["Hour_Sin"], d["Hour_Cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    d["DayOfWeek_Sin"], d["DayOfWeek_Cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d = d.iloc[30:].reset_index(drop=True)          # indicator warm-up
    return d.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


BASE_FEATURES = ["Return", "Log_Return", "Open_Return", "High_Return", "Low_Return", "HL_Range",
                 "SMA_7_Distance", "SMA_14_Distance", "SMA_30_Distance", "EMA_7_Distance", "EMA_14_Distance",
                 "EMA_30_Distance", "Volatility_7", "Volatility_14", "RSI", "MACD_Relative",
                 "MACD_Signal_Relative", "MACD_Hist_Relative", "ATR_Ratio", "BB_Width", "BB_Position",
                 "Momentum_Ratio", "Log_Volume", "Hour_Sin", "Hour_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos"]


def make_targets(d):
    """Log-return vs the close of bar t. Last DAY_BARS rows have no complete future -> NaN."""
    O, H, L, C = (d[k].values for k in ["Open", "High", "Low", "Close"])
    n = len(d)
    Y = np.full((n, len(TARGETS)), np.nan)
    for t in range(n - DAY_BARS):
        f = slice(t + 1, t + 1 + DAY_BARS)
        Y[t] = np.log(np.array([H[t + 1], L[t + 1], O[t + 1], H[f].max(), L[f].min(), C[t + DAY_BARS], C[t + 1]]) / C[t])
    return Y


def windows(X, ends):
    return np.stack([X[e - WINDOW + 1: e + 1] for e in ends])


def loader(X, y, shuffle):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.tensor(X[i:i + 1024], dtype=torch.float32).to(device)).cpu().numpy()
                               for i in range(0, len(X), 1024)])


def to_prices(ret, base_close):
    return np.exp(ret) * base_close[:, None]


def fix_ohlc(p):
    """p: dict target->price. Enforce High >= Open/Close and Low <= Open/Close."""
    n15 = {"High": p["n15_High"], "Low": p["n15_Low"], "Close": p["n15_Close"]}
    n15["High"], n15["Low"] = max(n15["High"], n15["Close"]), min(n15["Low"], n15["Close"])
    nd = {"Open": p["nd_Open"], "High": p["nd_High"], "Low": p["nd_Low"], "Close": p["nd_Close"]}
    nd["High"] = max(nd["High"], nd["Open"], nd["Close"]); nd["Low"] = min(nd["Low"], nd["Open"], nd["Close"])
    return {"next_15m": {k: round(float(v), 2) for k, v in n15.items()},
            "next_day": {k: round(float(v), 2) for k, v in nd.items()}}


def save_test_report(dates, actual, preds, metrics, targets, out_dir):
    """Table + graphs of actual vs predicted on the test set (the last ~10% of the data).

    actual / preds: dict-like (rows, len(targets)) arrays of prices, columns in `targets` order.
    """
    names = {"n15_High": "Next 15m High", "n15_Low": "Next 15m Low", "n15_Close": "Next 15m Close",
             "nd_Open": "Next day Open", "nd_High": "Next day High", "nd_Low": "Next day Low",
             "nd_Close": "Next day Close"}
    col = {t: j for j, t in enumerate(targets)}

    # ---- table: every test row, actual vs predicted (ensemble) + error ----
    table = pd.DataFrame({"Date": dates})
    for t in targets:
        table[f"actual_{t}"] = actual[:, col[t]]
        table[f"pred_{t}"] = preds["Ensemble"][:, col[t]]
        table[f"err_{t}"] = table[f"pred_{t}"] - table[f"actual_{t}"]
    table.to_csv(out_dir / "test_actual_vs_predicted.csv", index=False)

    # ---- table: MAE / bias per target and model (Naive = price stays at last close) ----
    summary = pd.DataFrame({m: v["mae"] for m, v in metrics.items()}).T
    summary["15m_dir_acc"] = [metrics[m]["dir_acc_15m"] for m in summary.index]
    summary.to_csv(out_dir / "test_summary_mae.csv")
    bias = pd.DataFrame({"mean_error(pred-actual)": {t: table[f"err_{t}"].mean() for t in targets},
                         "MAE": {t: table[f"err_{t}"].abs().mean() for t in targets},
                         "MAPE_%": {t: (table[f"err_{t}"].abs() / table[f"actual_{t}"]).mean() * 100 for t in targets}})
    bias.to_csv(out_dir / "test_ensemble_error.csv")
    print("\nEnsemble error on TEST data:")
    print(bias.round(3).to_string())
    print("\nFirst / last 5 TEST rows (actual vs predicted):")
    show = ["Date"] + [f"{p}_{t}" for t in targets for p in ("actual", "pred")]
    print(pd.concat([table[show].head(5), table[show].tail(5)]).round(2).to_string(index=False))

    # ---- graphs ----
    groups = {"next_15m": ["n15_High", "n15_Low", "n15_Close"],
              "next_day": ["nd_Open", "nd_High", "nd_Low", "nd_Close"]}
    for gname, tg in groups.items():
        for suffix, sl in (("full_test", slice(None)), ("last_300_bars", slice(-300, None))):
            fig, axes = plt.subplots(len(tg), 1, figsize=(14, 3.2 * len(tg)), sharex=True)
            for ax, t in zip(np.atleast_1d(axes), tg):
                ax.plot(table["Date"].iloc[sl], table[f"actual_{t}"].iloc[sl], label="Actual", color="tab:blue", lw=1.2)
                ax.plot(table["Date"].iloc[sl], table[f"pred_{t}"].iloc[sl], label="Predicted", color="tab:orange", lw=1.0)
                ax.set_title(f"{names[t]} - actual vs predicted ({suffix.replace('_', ' ')})")
                ax.set_ylabel("Price"); ax.grid(alpha=0.3); ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(out_dir / f"test_{gname}_{suffix}.png", dpi=110)
            plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))   # scatter: perfect prediction lies on the diagonal
    for ax, t in zip(axes.ravel(), targets):
        a, p = table[f"actual_{t}"], table[f"pred_{t}"]
        ax.scatter(a, p, s=4, alpha=0.4)
        lim = [min(a.min(), p.min()), max(a.max(), p.max())]
        ax.plot(lim, lim, "r--", lw=1); ax.set_title(names[t]); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    for ax in axes.ravel()[len(targets):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out_dir / "test_scatter_actual_vs_predicted.png", dpi=110); plt.close(fig)
    print("\nSaved test tables + graphs (test_*.csv / test_*.png) to", out_dir)


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    global EPOCHS

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default="/kaggle/input/datasets/ferdeenbakhtkag/xau-15m-data/XAU_15m_data.csv"
    )
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)

    args, _ = ap.parse_known_args(argv)

    EPOCHS = int(args.epochs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    set_seed()

    print("device:", device)

    # Load data
    d = build_features(
        load_data(args.data, args.rows)
    )

    Y = make_targets(d)

    n = len(d)

    # Train / validation / test split
    labelled = np.arange(
        WINDOW - 1,
        n - DAY_BARS
    )

    n_tr = int(len(labelled) * 0.8)
    n_va = int(len(labelled) * 0.1)

    tr = labelled[:n_tr]

    va = labelled[
        n_tr + DAY_BARS:
        n_tr + n_va
    ]

    te = labelled[
        n_tr + n_va + DAY_BARS:
    ]

    print(
        f"train {len(tr)} | "
        f"val {len(va)} | "
        f"test {len(te)}"
    )

    # ---------------------------------------------------------
    # Market regimes
    # GMM fitted only on training rows
    # Regime probabilities are the LAST 3 inputs
    # ---------------------------------------------------------

    reg_scaler = StandardScaler().fit(
        d.loc[tr, REGIME_INPUTS]
    )

    gmm = GaussianMixture(
        3,
        covariance_type="full",
        n_init=3,
        random_state=SEED
    ).fit(
        reg_scaler.transform(
            d.loc[tr, REGIME_INPUTS]
        )
    )

    lab = gmm.predict(
        reg_scaler.transform(
            d.loc[tr, REGIME_INPUTS]
        )
    )

    # Order regimes by Return_20:
    # lowest = bearish
    # middle = sideways
    # highest = bullish
    order = np.argsort([
        d["Return_20"].values[tr][lab == k].mean()
        for k in range(3)
    ])

    proba = gmm.predict_proba(
        reg_scaler.transform(
            d[REGIME_INPUTS]
        )
    )[:, order]

    # ---------------------------------------------------------
    # Feature scaling
    # ---------------------------------------------------------

    feat_scaler = StandardScaler().fit(
        d.loc[tr, BASE_FEATURES]
    )

    base_scaled = np.clip(
        feat_scaler.transform(
            d[BASE_FEATURES]
        ),
        -8,
        8
    )

    # Add regime probabilities as final 3 features
    feats = np.hstack([
        base_scaled,
        proba
    ]).astype(np.float32)

    # ---------------------------------------------------------
    # Target scaling
    # ---------------------------------------------------------

    y_scaler = StandardScaler().fit(
        Y[tr]
    )

    Ys = y_scaler.transform(
        np.nan_to_num(Y)
    )

    # ---------------------------------------------------------
    # Create windows
    # ---------------------------------------------------------

    Xtr = windows(feats, tr)
    Xva = windows(feats, va)
    Xte = windows(feats, te)

    ytr = Ys[tr]
    yva = Ys[va]

    # ---------------------------------------------------------
    # Current / last price
    # ---------------------------------------------------------

    close = d["Close"].values

    last_close = close[-1]

    x_last = feats[-WINDOW:][None]

    # Scaled value corresponding to 0% return
    up_thr = float(
        y_scaler.transform(
            np.zeros(
                (1, len(TARGETS))
            )
        )[0, -1]
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_dl = loader(
        Xtr,
        ytr,
        False
    )

    val_dl = loader(
        Xva,
        yva,
        False
    )

    # ---------------------------------------------------------
    # Build models
    # ---------------------------------------------------------

    models = build_models(
        feats.shape[1],
        len(TARGETS)
    )

    nT = len(TARGETS)

    preds_te = {}
    preds_last = {}
    val_mae = {}

    # =========================================================
    # Train each model
    # =========================================================

    for name in args.models:

        t0 = time.time()

        print(f"\nTraining {name}...")

        model = models[name].to(device)

        model.up_threshold = up_thr

        # Optimizer
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # Reset best model for every model
        best = float("inf")
        best_state = None

        # =====================================================
        # Epoch loop
        # =====================================================

        for ep in range(EPOCHS):

            # -------------------------------------------------
            # Training
            # -------------------------------------------------

            model.train()

            tot = 0.0

            for xb, yb in train_dl:

                xb = xb.to(device)
                yb = yb.to(device)

                opt.zero_grad()

                loss = model.multitask_loss(
                    xb,
                    yb
                )

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                opt.step()

                tot += loss.item()

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            model.eval()

            with torch.no_grad():

                val_losses = []

                for xb, yb in val_dl:

                    xb = xb.to(device)
                    yb = yb.to(device)

                    output = model(xb)

                    loss = F.mse_loss(
                        output,
                        yb
                    )

                    val_losses.append(
                        loss.item()
                    )

                vl = float(
                    np.mean(val_losses)
                )

            # -------------------------------------------------
            # Save best validation model
            # -------------------------------------------------

            if np.isfinite(vl) and vl < best:

                best = vl

                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }

            # -------------------------------------------------
            # Print progress
            # -------------------------------------------------

            if (
                ep == 0
                or (ep + 1) % 5 == 0
                or ep == EPOCHS - 1
            ):

                print(
                    f"  [{name}] "
                    f"epoch {ep + 1}/{EPOCHS} "
                    f"train {tot / len(train_dl):.4f} "
                    f"val {vl:.4f}"
                )

        # =====================================================
        # Check best model
        # =====================================================

        if best_state is None:

            raise RuntimeError(
                f"[{name}] No valid validation loss was produced. "
                f"Check for NaN/Inf in the model output "
                f"or validation data."
            )

        # Restore best model
        model.load_state_dict(
            best_state
        )

        # Save model
        torch.save(
            {
                "model": best_state,
                "targets": TARGETS,
                "window": WINDOW
            },
            OUT_DIR / f"xau15m_{name}.pt"
        )

        # =====================================================
        # Validation predictions
        # =====================================================

        pv = to_prices(
            y_scaler.inverse_transform(
                predict(model, Xva)[:, :nT]
            ),
            close[va]
        )

        actual_va = to_prices(
            Y[va],
            close[va]
        )

        val_mae[name] = float(
            np.abs(
                pv - actual_va
            ).mean()
        )

        # =====================================================
        # Test predictions
        # =====================================================

        preds_te[name] = to_prices(
            y_scaler.inverse_transform(
                predict(model, Xte)[:, :nT]
            ),
            close[te]
        )

        # =====================================================
        # Latest prediction
        # =====================================================

        preds_last[name] = to_prices(
            y_scaler.inverse_transform(
                predict(model, x_last)[:, :nT]
            ),
            np.array([last_close])
        )[0]

        print(
            f"  [{name}] done in "
            f"{time.time() - t0:.0f}s | "
            f"val $MAE {val_mae[name]:.2f}"
        )

    # =========================================================
    # Ensemble
    # =========================================================

    top = sorted(
        val_mae,
        key=val_mae.get
    )[:ENSEMBLE_TOP_K]

    preds_te["Ensemble"] = np.mean(
        [
            preds_te[m]
            for m in top
        ],
        axis=0
    )

    preds_last["Ensemble"] = np.mean(
        [
            preds_last[m]
            for m in top
        ],
        axis=0
    )

    # =========================================================
    # Test actual values
    # =========================================================

    actual_te = to_prices(
        Y[te],
        close[te]
    )

    # Naive prediction:
    # assumes future price stays equal to current close
    naive_te = np.repeat(
        close[te][:, None],
        nT,
        axis=1
    )

    print(
        f"\nTEST $MAE per target "
        f"(Naive = 'price stays at last close'); "
        f"ensemble = {top}"
    )

    print(
        f"{'model':16s}"
        + "".join(
            f"{t:>12s}"
            for t in TARGETS
        )
        + "   15m-dir-acc"
    )

    # =========================================================
    # Metrics
    # =========================================================

    metrics = {}

    for name, p in {
        "Naive": naive_te,
        **preds_te
    }.items():

        mae = np.abs(
            p - actual_te
        ).mean(axis=0)

        if name != "Naive":

            acc = float(
                np.mean(
                    (p[:, -1] > close[te])
                    ==
                    (actual_te[:, -1] > close[te])
                )
            )

        else:

            acc = float("nan")

        metrics[name] = {
            "mae": dict(
                zip(
                    TARGETS,
                    mae.round(3).tolist()
                )
            ),
            "dir_acc_15m": acc
        }

        print(
            f"{name:16s}"
            + "".join(
                f"{v:12.2f}"
                for v in mae
            )
            + f"   {acc:.3f}"
        )

    # =========================================================
    # Final result
    # =========================================================

    result = {
        "data_rows": int(len(d)),

        "last_bar_time": str(
            d.Date.iloc[-1]
        ),

        "last_close": float(
            last_close
        ),

        "ensemble_members": top,

        "test_metrics": metrics,

        "ensemble": fix_ohlc(
            dict(
                zip(
                    TARGETS,
                    preds_last["Ensemble"]
                )
            )
        ),

        "per_model": {
            m: fix_ohlc(
                dict(
                    zip(
                        TARGETS,
                        preds_last[m]
                    )
                )
            )
            for m in args.models
        }
    }

    # =========================================================
    # Save JSON
    # =========================================================

    with open(
        OUT_DIR / "prediction_15m.json",
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    # =========================================================
    # Save backtest CSV
    # =========================================================

    pd.DataFrame(
        {
            "Date": d.Date.values[te],
            "Close": close[te],

            **{
                f"actual_{t}": actual_te[:, j]
                for j, t in enumerate(TARGETS)
            },

            **{
                f"ens_{t}": preds_te["Ensemble"][:, j]
                for j, t in enumerate(TARGETS)
            }
        }
    ).to_csv(
        OUT_DIR / "backtest_15m.csv",
        index=False
    )

    # =========================================================
    # Test-set report: actual vs predicted (table + graphs)
    # =========================================================

    save_test_report(d.Date.values[te], actual_te, preds_te, metrics, TARGETS, OUT_DIR)

    # =========================================================
    # Final prediction
    # =========================================================

    print(
        "\nPREDICTION from",
        result["last_bar_time"],
        "| last close",
        last_close
    )

    print(
        json.dumps(
            result["ensemble"],
            indent=2
        )
    )

    print(
        "saved to",
        OUT_DIR
    )


if __name__ == "__main__":
    main()
