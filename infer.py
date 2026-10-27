# -*- coding: utf-8 -*-
"""
SAG-LSTM inference - NO retraining. Loads the artifacts written by kaggle_xau_continue.py
(or train_local.py) and predicts for any start bar.

Artifact folder (model_dir):
    model.pt        SAG_LSTM weights + sizes
    pipeline.npz    feature / target scalers + regime GMM as plain arrays (no pickle, so it loads
                    on any scikit-learn version, e.g. trained on Kaggle, used locally)
    meta.json       split dates, train_end, metrics

CLI:
    python infer.py --model-dir outputs/sag_final --start "2026-01-29 12:00"
    python infer.py --model-dir outputs/sag_final --date 2026-01-29 --csv day.csv
    python infer.py --model-dir outputs/sag_final            # latest bar

Shared by app.py and the training scripts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import kaggle_xau_train as kx

HERE = Path(__file__).resolve().parent
STEP = pd.Timedelta(minutes=15)


# ------------------------------------------------------------
# pipeline fit / export / import
# ------------------------------------------------------------

def fit_pipeline(d, Y, tr):
    """Regime GMM + feature scaler + target scaler fitted on rows `tr` only (same as kaggle_xau_train.py)."""
    reg_scaler = StandardScaler().fit(d.loc[tr, kx.REGIME_INPUTS])
    gmm = GaussianMixture(n_components=3, covariance_type="full", n_init=3,
                          random_state=kx.SEED).fit(reg_scaler.transform(d.loc[tr, kx.REGIME_INPUTS]))
    lab = gmm.predict(reg_scaler.transform(d.loc[tr, kx.REGIME_INPUTS]))
    order = np.argsort([d["Return_20"].values[tr][lab == k].mean() for k in range(3)])  # bear, side, bull
    feat_scaler = StandardScaler().fit(d.loc[tr, kx.BASE_FEATURES])
    y_scaler = StandardScaler().fit(Y[tr])
    return {"reg_scaler": reg_scaler, "gmm": gmm, "order": order,
            "feat_scaler": feat_scaler, "y_scaler": y_scaler}


def export_pipeline(path, pipe):
    arrs = {}
    for name in ("reg_scaler", "feat_scaler", "y_scaler"):
        s = pipe[name]
        arrs[f"{name}_mean"], arrs[f"{name}_scale"], arrs[f"{name}_var"] = s.mean_, s.scale_, s.var_
    g = pipe["gmm"]
    arrs.update(gmm_weights=g.weights_, gmm_means=g.means_, gmm_covariances=g.covariances_,
                gmm_prec_chol=g.precisions_cholesky_, order=np.asarray(pipe["order"]))
    np.savez(path, **arrs)


def _scaler(mean, scale, var):
    s = StandardScaler()
    s.mean_, s.scale_, s.var_ = mean, scale, var
    s.n_features_in_, s.n_samples_seen_ = len(mean), 1
    return s


def import_pipeline(path):
    z = np.load(path)
    g = GaussianMixture(n_components=len(z["gmm_weights"]), covariance_type="full")
    g.weights_, g.means_, g.covariances_ = z["gmm_weights"], z["gmm_means"], z["gmm_covariances"]
    g.precisions_cholesky_ = z["gmm_prec_chol"]
    g.n_features_in_ = g.means_.shape[1]
    g.converged_, g.n_iter_ = True, 0
    return {**{n: _scaler(z[f"{n}_mean"], z[f"{n}_scale"], z[f"{n}_var"])
               for n in ("reg_scaler", "feat_scaler", "y_scaler")},
            "gmm": g, "order": z["order"]}


def make_features(d, pipe):
    """Scaled base features + Bearish/Sideways/Bullish probabilities, one row per bar of `d`."""
    reg = pipe["reg_scaler"].transform(d[kx.REGIME_INPUTS].values)
    proba = pipe["gmm"].predict_proba(reg)[:, pipe["order"]]
    base = np.clip(pipe["feat_scaler"].transform(d[kx.BASE_FEATURES].values), -8, 8)
    return np.hstack([base, proba]).astype(np.float32)


# ------------------------------------------------------------
# load / predict
# ------------------------------------------------------------

def load_bundle(model_dir):
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    model = kx.SAG_LSTM(input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"],
                        num_layers=ckpt["num_layers"], output_size=len(ckpt["targets"]),
                        dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["model"])
    model.to(kx.device).eval()
    meta = json.loads((model_dir / "meta.json").read_text())
    return model, import_pipeline(model_dir / "pipeline.npz"), meta


def load_features(data_path, rows):
    return kx.build_features(kx.load_data(str(data_path), rows))


def actuals(d, t):
    """Actual prices for the same 7 targets; NaN where the future is not in the data yet."""
    n = len(d)
    out = dict.fromkeys(kx.TARGETS, np.nan)
    if t + 1 < n:
        out["n15_High"], out["n15_Low"], out["n15_Close"] = d.High[t + 1], d.Low[t + 1], d.Close[t + 1]
    if t + kx.DAY_BARS < n:
        f = slice(t + 1, t + 1 + kx.DAY_BARS)
        out.update(nd_Open=d.Open[t + 1], nd_High=d.High[f].max(), nd_Low=d.Low[f].min(),
                   nd_Close=d.Close[t + kx.DAY_BARS])
    return out


def predict_rows(model, pipe, feats, close, idx):
    """Raw predicted prices (n, 7) for start bars `idx`, anchored on each bar's Close."""
    idx = np.asarray(idx)
    X = kx.windows(feats, idx)
    ret = pipe["y_scaler"].inverse_transform(kx.predict(model, X))
    return kx.to_prices(ret, close[idx])


def fixed_pred(raw_row):
    """fix_ohlc(): High >= Open/Close, Low <= Open/Close. Returns a flat dict keyed by TARGETS."""
    f = kx.fix_ohlc(dict(zip(kx.TARGETS, raw_row)))
    return {**{f"n15_{k}": v for k, v in f["next_15m"].items()},
            **{f"nd_{k}": v for k, v in f["next_day"].items()}}


def forecast_at(model, pipe, feats, d, t):
    raw = predict_rows(model, pipe, feats, d["Close"].values, [t])[0]
    return fixed_pred(raw), actuals(d, t)


def find_index(d, ts):
    ts = pd.Timestamp(ts)
    hit = np.where(d.Date.values == np.datetime64(ts))[0]
    if len(hit) == 0:
        raise SystemExit(f"No bar at {ts} in the data (weekend / holiday / gap).")
    t = int(hit[0])
    if t < kx.WINDOW - 1:
        raise SystemExit("Not enough history before that bar.")
    return t


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(HERE / "outputs" / "sag_final"))
    ap.add_argument("--data", default=None, help="CSV (default: meta data file next to this script)")
    ap.add_argument("--start", default=None, help="start bar, e.g. '2026-01-29 12:00'")
    ap.add_argument("--date", default=None, help="predict every bar of this date, e.g. 2026-01-29")
    ap.add_argument("--csv", default=None, help="write the --date table here")
    a = ap.parse_args()

    model, pipe, meta = load_bundle(a.model_dir)
    d = load_features(a.data or HERE / meta["data"], meta["rows"])
    feats = make_features(d, pipe)

    if a.date:
        idx = np.where((d.Date.dt.date == pd.Timestamp(a.date).date()) & (np.arange(len(d)) >= kx.WINDOW - 1))[0]
        raw = predict_rows(model, pipe, feats, d["Close"].values, idx)
        rows = []
        for i, r in zip(idx, raw):
            p, ac = fixed_pred(r), actuals(d, i)
            row = {"Start": d.Date[i], "Minute_of_day": d.Date[i].hour * 60 + d.Date[i].minute,
                   "Start_Close": d.Close[i]}
            row.update({f"pred_{k}": v for k, v in p.items()})
            row.update({f"actual_{k}": v for k, v in ac.items()})
            rows.append(row)
        out = pd.DataFrame(rows)
        print(out.round(2).to_string(index=False))
        if a.csv:
            out.to_csv(a.csv, index=False)
            print("saved", a.csv)
        return

    t = find_index(d, a.start) if a.start else len(d) - 1
    p, ac = forecast_at(model, pipe, feats, d, t)
    ts, c0 = d.Date[t], float(d.Close[t])
    print(f"Start {ts:%Y-%m-%d %H:%M}  (minute {ts.hour * 60 + ts.minute} of day)   start Close {c0:,.2f}")
    print(f"{'target':10s} {'time':>22s} {'pred':>10s} {'d$':>8s} {'d%':>8s} {'actual':>10s} {'err':>8s}")
    when = {"n15": f"{ts + STEP:%m-%d %H:%M}", "nd_Open": f"{ts + STEP:%m-%d %H:%M}",
            "nd_Close": f"{ts + STEP * kx.DAY_BARS:%m-%d %H:%M}"}
    for k in ["n15_High", "n15_Low", "n15_Close", "nd_Open", "nd_High", "nd_Low", "nd_Close"]:
        w = when.get(k, when.get(k[:3], f"{ts + STEP:%m-%d %H:%M}->{ts + STEP * kx.DAY_BARS:%m-%d %H:%M}"))
        a_ = ac[k]
        print(f"{k:10s} {w:>22s} {p[k]:10.2f} {p[k] - c0:+8.2f} {(p[k] / c0 - 1) * 100:+7.3f}% "
              f"{'' if np.isnan(a_) else format(a_, '10.2f'):>10s} "
              f"{'' if np.isnan(a_) else format(p[k] - a_, '+8.2f'):>8s}")


if __name__ == "__main__":
    main()
