# -*- coding: utf-8 -*-
"""XAU/USD 15-minute training on the latest 20,000 bars (standalone: Kaggle / Colab / local).

Trains the neural models of train_predict.py (LSTM, GRU, Transformer, Transformer-MT, SAG-LSTM, LSTM-Transformer,
LSTM-Attention-LSTM, CNN-BiLSTM-Attn, PatchTST, TFT)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROWS = 20000
WINDOW = 32               # 15m bars of history per sample (8 hours)
DAY_BARS = 96             # 96 x 15min = next "day"
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0
CLS_WEIGHT = 0.5          # loss = MSE(returns) + CLS_WEIGHT * BCE(next 15m close up/down)
ENSEMBLE_TOP_K = 3        # ensemble = mean of the K models with the lowest validation $ MAE
SEED = 42
MODEL_NAMES = ["LSTM", "GRU", "Transformer", "Transformer-MT", "SAG-LSTM",
               "LSTM-Transformer", "LSTM-Attention-LSTM", "CNN-BiLSTM-Attn", "PatchTST", "TFT"]

# order matters: the LAST output is the Up/Down reference (next 15m close)
TARGETS = ["n15_High", "n15_Low", "nd_Open", "nd_High", "nd_Low", "nd_Close", "n15_Close"]
REGIME_INPUTS = ["SMA_30_Distance", "EMA_30_Distance", "Momentum_Ratio", "Volatility_14", "ATR_Ratio", "Return_20"]

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent / "outputs"
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


def build_models(input_size, output_size):
    set_seed()
    n = output_size + 1   # +1 unit = Up/Down logit
    base = {
        "LSTM": LSTMModel(input_size, output_size=n),
        "GRU": GRUModel(input_size, output_size=n),
        "Transformer": TimeSeriesTransformer(input_size, output_size=n),
        "SAG-LSTM": SAG_LSTM(input_size, hidden_size=128, output_size=n),
        "LSTM-Transformer": LSTMTransformer(input_size, output_size=n),
        "LSTM-Attention-LSTM": LSTMAttentionLSTM(input_size, hidden_size=128, output_steps=1, output_size=n),
        "CNN-BiLSTM-Attn": CNN_BiLSTM_Attention(input_size, window_size=WINDOW, output_size=n),
        "PatchTST": PatchTST(input_size, window_size=WINDOW, patch_length=8, stride=8, output_size=n),
        "TFT": CompactTFT(input_size, output_size=n),
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


# ============================================================
# MAIN
# ============================================================

def main(argv=None):
    global EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="XAU_15m_data.csv")
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    args, _ = ap.parse_known_args(argv)
    EPOCHS = args.epochs
    OUT_DIR.mkdir(exist_ok=True)
    set_seed()
    print("device:", device)

    d = build_features(load_data(args.data, args.rows))
    Y = make_targets(d)
    n = len(d)
    labelled = np.arange(WINDOW - 1, n - DAY_BARS)           # sample = window ending at bar t
    n_tr, n_va = int(len(labelled) * 0.8), int(len(labelled) * 0.1)
    tr = labelled[:n_tr]
    va = labelled[n_tr + DAY_BARS: n_tr + n_va]               # DAY_BARS gaps: targets look that far forward
    te = labelled[n_tr + n_va + DAY_BARS:]
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    # market regimes (GMM fitted on training rows only); probabilities are the LAST 3 inputs
    reg_scaler = StandardScaler().fit(d.loc[tr, REGIME_INPUTS])
    gmm = GaussianMixture(3, covariance_type="full", n_init=3, random_state=SEED).fit(reg_scaler.transform(d.loc[tr, REGIME_INPUTS]))
    lab = gmm.predict(reg_scaler.transform(d.loc[tr, REGIME_INPUTS]))
    order = np.argsort([d["Return_20"].values[tr][lab == k].mean() for k in range(3)])
    proba = gmm.predict_proba(reg_scaler.transform(d[REGIME_INPUTS]))[:, order]

    feat_scaler = StandardScaler().fit(d.loc[tr, BASE_FEATURES])
    feats = np.hstack([np.clip(feat_scaler.transform(d[BASE_FEATURES]), -8, 8), proba]).astype(np.float32)
    y_scaler = StandardScaler().fit(Y[tr])
    Ys = y_scaler.transform(np.nan_to_num(Y))

    Xtr, Xva, Xte = windows(feats, tr), windows(feats, va), windows(feats, te)
    ytr, yva = Ys[tr], Ys[va]
    close = d["Close"].values
    last_close = close[-1]
    x_last = feats[-WINDOW:][None]
    up_thr = float(y_scaler.transform(np.zeros((1, len(TARGETS))))[0, -1])   # scaled value of a 0% return

    train_dl, val_dl = loader(Xtr, ytr, True), loader(Xva, yva, False)
    models = build_models(feats.shape[1], len(TARGETS))
    nT = len(TARGETS)
    preds_te, preds_last, val_mae = {}, {}, {}
    for name in args.models:
        model = models[name].to(device)
        model.up_threshold = up_thr
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best, best_state, t0 = 1e9, None, time.time()
        for ep in range(EPOCHS):
            model.train()
            tot = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = model.multitask_loss(xb, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step(); tot += loss.item()
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([F.mse_loss(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_dl]))
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ep == 0 or (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
                print(f"  [{name}] epoch {ep + 1}/{EPOCHS} train {tot / len(train_dl):.4f} val {vl:.4f}")
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "targets": TARGETS, "window": WINDOW}, OUT_DIR / f"xau15m_{name}.pt")

        pv = to_prices(y_scaler.inverse_transform(predict(model, Xva)[:, :nT]), close[va])
        val_mae[name] = float(np.abs(pv - to_prices(Y[va], close[va])).mean())
        preds_te[name] = to_prices(y_scaler.inverse_transform(predict(model, Xte)[:, :nT]), close[te])
        preds_last[name] = to_prices(y_scaler.inverse_transform(predict(model, x_last)[:, :nT]), np.array([last_close]))[0]
        print(f"  [{name}] done in {time.time() - t0:.0f}s | val $MAE {val_mae[name]:.2f}")

    # ensemble = mean of the top-K models by validation $ MAE
    top = sorted(val_mae, key=val_mae.get)[:ENSEMBLE_TOP_K]
    preds_te["Ensemble"] = np.mean([preds_te[m] for m in top], axis=0)
    preds_last["Ensemble"] = np.mean([preds_last[m] for m in top], axis=0)

    actual_te = to_prices(Y[te], close[te])
    naive_te = np.repeat(close[te][:, None], nT, axis=1)
    print(f"\nTEST $MAE per target (Naive = 'price stays at last close'); ensemble = {top}")
    print(f"{'model':16s}" + "".join(f"{t:>12s}" for t in TARGETS) + "   15m-dir-acc")
    metrics = {}
    for name, p in {"Naive": naive_te, **preds_te}.items():
        mae = np.abs(p - actual_te).mean(axis=0)
        acc = float(np.mean((p[:, -1] > close[te]) == (actual_te[:, -1] > close[te]))) if name != "Naive" else float("nan")
        metrics[name] = {"mae": dict(zip(TARGETS, mae.round(3).tolist())), "dir_acc_15m": acc}
        print(f"{name:16s}" + "".join(f"{v:12.2f}" for v in mae) + f"   {acc:.3f}")

    result = {"data_rows": int(len(d)), "last_bar_time": str(d.Date.iloc[-1]), "last_close": float(last_close),
              "ensemble_members": top, "test_metrics": metrics,
              "ensemble": fix_ohlc(dict(zip(TARGETS, preds_last["Ensemble"]))),
              "per_model": {m: fix_ohlc(dict(zip(TARGETS, preds_last[m]))) for m in args.models}}
    with open(OUT_DIR / "prediction_15m.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame({"Date": d.Date.values[te], "Close": close[te],
                  **{f"actual_{t}": actual_te[:, j] for j, t in enumerate(TARGETS)},
                  **{f"ens_{t}": preds_te["Ensemble"][:, j] for j, t in enumerate(TARGETS)}}
                 ).to_csv(OUT_DIR / "backtest_15m.csv", index=False)
    print("\nPREDICTION from", result["last_bar_time"], "| last close", last_close)
    print(json.dumps(result["ensemble"], indent=2))
    print("saved to", OUT_DIR)


if __name__ == "__main__":
    main()
