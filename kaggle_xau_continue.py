# -*- coding: utf-8 -*-
"""
Iteration 2 on Kaggle: continue SAG-LSTM training on ALL rows up to --train-end.

Upload next to this file: kaggle_xau_train.py and infer.py (both are imported).

What it does
    * loads the iteration-1 weights (--resume) and keeps training from them
    * rebuilds the iteration-1 scalers / regime model (first 80% of the labelled rows, same as
      kaggle_xau_train.py) so the resumed weights see exactly the same input scaling
    * trains on every labelled row whose next-day target ends on/before --train-end
    * bars after --train-end (e.g. 29 Jan 2026) are never trained on -> honest out-of-sample test
    * measures epoch time and computes how many epochs fit in --time-budget-hours
    * holds out the last --val-bars training bars for early stopping (set 0 to train on everything)
    * writes /kaggle/working/sag_final/ {model.pt, model_last.pt, pipeline.npz, meta.json, backtest.csv}

Kaggle notebook:
    !python kaggle_xau_continue.py --resume /kaggle/input/<ds>/xau15m_SAG_LSTM.pt \
        --train-end "2026-01-28 23:45"
Then download sag_final/ and put it in outputs/sag_final/ locally (see infer.py / app.py).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import kaggle_xau_train as kx
from infer import export_pipeline, fit_pipeline, make_features, predict_rows

device = kx.device


def cuda_sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_val(model, dl):
    model.eval()
    with torch.no_grad():
        return float(np.mean([F.mse_loss(model(xb.to(device)), yb.to(device)).item() for xb, yb in dl]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/datasets/ferdeenbakhtkag/xau-15m-data/XAU_15m_data.csv")
    ap.add_argument("--rows", type=int, default=kx.ROWS, help="must equal iteration 1 when resuming")
    ap.add_argument("--resume", default=None, help="iteration-1 checkpoint (xau15m_SAG_LSTM.pt)")
    ap.add_argument("--train-end", default="2026-01-28 23:45")
    ap.add_argument("--epochs", type=int, default=100, help="maximum extra epochs")
    ap.add_argument("--time-budget-hours", type=float, default=10.0, help="Kaggle session limit is 12h")
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 when resuming, else 1e-3")
    ap.add_argument("--val-bars", type=int, default=2000, help="last N training bars used for early stopping; 0=off")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out = Path(a.out_dir) if a.out_dir else kx.OUT_DIR / "sag_final"
    out.mkdir(parents=True, exist_ok=True)
    lr = a.lr or (1e-4 if a.resume else kx.LEARNING_RATE)
    kx.set_seed()
    print("Device:", device, "| resume:", a.resume, "| lr:", lr)

    # ---------------- data / split ----------------
    d = kx.build_features(kx.load_data(a.data, a.rows))
    Y = kx.make_targets(d)
    n = len(d)
    labelled = np.arange(kx.WINDOW - 1, n - kx.DAY_BARS)

    cut = int(np.searchsorted(d.Date.values, np.datetime64(pd.Timestamp(a.train_end)), side="right")) - 1
    trainable = labelled[labelled + kx.DAY_BARS <= cut]       # next-day horizon ends inside training period
    te = labelled[labelled > cut]                             # out-of-sample test
    if a.val_bars > 0:
        va = trainable[-a.val_bars:]
        tr = trainable[: -a.val_bars - kx.DAY_BARS]           # gap so val targets don't overlap train
    else:
        va, tr = np.array([], dtype=int), trainable
    print(f"train {len(tr)} ({d.Date[tr[0]]} -> {d.Date[tr[-1]]}) | val {len(va)} | test {len(te)}"
          + (f" ({d.Date[te[0]]} -> {d.Date[te[-1]]})" if len(te) else ""))

    # ---------------- pipeline (iteration-1 scaling when resuming) ----------------
    fit_rows = labelled[: int(len(labelled) * 0.8)] if a.resume else tr
    pipe = fit_pipeline(d, Y, fit_rows)
    feats = make_features(d, pipe)
    Ys = pipe["y_scaler"].transform(np.nan_to_num(Y))

    train_dl = kx.loader(kx.windows(feats, tr), Ys[tr], shuffle=False)
    val_dl = kx.loader(kx.windows(feats, va), Ys[va], shuffle=False) if len(va) else None

    # ---------------- model ----------------
    model = kx.SAG_LSTM(input_size=feats.shape[1], hidden_size=128, num_layers=2,
                        output_size=len(kx.TARGETS), dropout=0.2).to(device)
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print("Loaded iteration-1 weights from", a.resume)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=kx.WEIGHT_DECAY)

    def ckpt(state):
        return {"model": state, "targets": kx.TARGETS, "window": kx.WINDOW, "hidden_size": 128,
                "num_layers": 2, "dropout": 0.2, "input_size": feats.shape[1]}

    snap = lambda: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    best_val, best_state, best_ep = (run_val(model, val_dl) if val_dl else math.inf), snap(), 0
    start_val = best_val
    print(f"Validation loss at start (iteration-1 weights): {start_val:.6f}" if val_dl else "No validation set")

    # ---------------- training with time budget ----------------
    budget = a.time_budget_hours * 3600
    t0, ep_times, bad, max_epochs = time.time(), [], 0, a.epochs
    for ep in range(1, a.epochs + 1):
        e0 = time.time()
        model.train()
        tl = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), kx.GRAD_CLIP)
            opt.step()
            tl += loss.item()
        cuda_sync()
        tl /= len(train_dl)
        vl = run_val(model, val_dl) if val_dl else float("nan")
        ep_times.append(time.time() - e0)

        if val_dl and vl < best_val:
            best_val, best_state, best_ep, bad = vl, snap(), ep, 0
        else:
            bad += 1
        torch.save(ckpt(snap()), out / "model_last.pt")      # crash-safe: always the latest epoch

        per_epoch = float(np.mean(ep_times))
        fit_more = int((budget - (time.time() - t0)) // per_epoch)
        max_epochs = min(a.epochs, ep + fit_more)
        print(f"Epoch {ep:3d} | train {tl:.6f} | val {vl:.6f} | best {best_val:.6f}@{best_ep} | "
              f"{per_epoch:.0f}s/epoch | budget allows {fit_more} more (cap {max_epochs})", flush=True)

        if val_dl and bad >= a.patience:
            print(f"Early stop: no val improvement for {a.patience} epochs")
            break
        if fit_more < 1:
            print("Time budget reached")
            break

    last_state = snap()
    trained_epochs = ep
    final_state = best_state if val_dl else last_state
    print(f"Trained {trained_epochs} extra epochs | saved weights = "
          + (f"best-val epoch {best_ep}" if val_dl else "last epoch"))
    model.load_state_dict(final_state)
    torch.save(ckpt(final_state), out / "model.pt")
    torch.save(ckpt(last_state), out / "model_last.pt")
    export_pipeline(out / "pipeline.npz", pipe)

    # ---------------- out-of-sample test ----------------
    dt = lambda i: str(d.Date.iloc[i])
    close = d["Close"].values
    meta = {"mode": "final", "train_end": a.train_end, "rows": a.rows, "data": Path(a.data).name,
            "data_first": dt(0), "data_last": dt(n - 1), "resumed_from": a.resume, "lr": lr,
            "extra_epochs_trained": trained_epochs, "best_epoch": best_ep, "epoch_seconds": float(np.mean(ep_times)),
            "val_loss_start": None if not val_dl else start_val, "val_loss_best": None if not val_dl else best_val,
            "train": [dt(tr[0]), dt(tr[-1])], "val": [dt(va[0]), dt(va[-1])] if len(va) else None,
            "test": [dt(te[0]), dt(te[-1])] if len(te) else None}
    if len(te):
        pt = predict_rows(model, pipe, feats, close, te)
        at = kx.to_prices(Y[te], close[te])
        mae = np.abs(pt - at).mean(axis=0)
        naive = np.abs(close[te][:, None] - at).mean(axis=0)
        meta["test_mae"] = dict(zip(kx.TARGETS, mae.round(3).tolist()))
        meta["test_naive_mae"] = dict(zip(kx.TARGETS, naive.round(3).tolist()))
        meta["direction_accuracy"] = float(np.mean((pt[:, -1] > close[te]) == (at[:, -1] > close[te])))
        bt = pd.DataFrame({"Date": d.Date.values[te], "Close": close[te]})
        for j, t in enumerate(kx.TARGETS):
            bt[f"actual_{t}"] = at[:, j]
        for j, t in enumerate(kx.TARGETS):
            bt[f"pred_{t}"] = pt[:, j]
        bt.to_csv(out / "backtest.csv", index=False)
        print("\nOUT-OF-SAMPLE MAE (after train-end):")
        for t in kx.TARGETS:
            print(f"  {t:10s} model {meta['test_mae'][t]:9.2f} | naive {meta['test_naive_mae'][t]:9.2f}")
        print(f"  15m direction accuracy {meta['direction_accuracy']:.3f}")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
