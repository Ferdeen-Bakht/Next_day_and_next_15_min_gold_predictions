# -*- coding: utf-8 -*-
"""
Local SAG-LSTM training (same model / features / targets / split as kaggle_xau_train.py).

Two iterations:
    --mode eval   80/10/10 chronological split with best-epoch selection on validation.
                  Test block is never trained on -> this is the run you report accuracy from.
    --mode final  trains on all labelled rows (or up to --train-end) for a fixed number of epochs.
                  For the real iteration 2 use kaggle_xau_continue.py (resumes iteration 1). Production model; no honest accuracy claim.

Artifacts go to outputs/sag_<mode>/ and are read by app.py.

    python train_local.py --mode eval  --rows 100000 --epochs 54
    python train_local.py --mode final --rows 100000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import kaggle_xau_train as kx
from infer import export_pipeline, fit_pipeline, make_features

HERE = Path(__file__).resolve().parent
device = kx.device


def run_epochs(model, train_dl, val_dl, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=kx.LEARNING_RATE, weight_decay=kx.WEIGHT_DECAY)
    best, best_state, best_ep = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), kx.GRAD_CLIP)
            opt.step()
            tl += loss.item()
        tl /= len(train_dl)

        vl = float("nan")
        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([F.mse_loss(model(xb.to(device)), yb.to(device)).item()
                                    for xb, yb in val_dl]))
            if np.isfinite(vl) and vl < best:
                best, best_ep = vl, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {ep + 1:3d}/{epochs} | train {tl:.6f} | val {vl:.6f} | {time.time() - t0:.0f}s",
              flush=True)

    if val_dl is None:                       # final mode: keep the last epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_ep = epochs
    if best_state is None:
        raise RuntimeError("No finite validation loss was produced.")
    model.load_state_dict(best_state)
    return best_state, best_ep, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "XAU_15m_data.csv"))
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--mode", choices=["eval", "final"], default="eval")
    ap.add_argument("--train-end", default=None,
                    help="final mode: last bar whose data may be used, e.g. '2026-01-28 23:45'. "
                         "Later bars become an out-of-sample test.")
    args = ap.parse_args()

    out = HERE / "outputs" / f"sag_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or kx.EPOCHS

    kx.set_seed()
    print("Device:", device, "| mode:", args.mode, "| epochs:", epochs)

    d = kx.build_features(kx.load_data(args.data, args.rows))
    Y = kx.make_targets(d)
    n = len(d)
    labelled = np.arange(kx.WINDOW - 1, n - kx.DAY_BARS)

    if args.mode == "eval":
        n_tr, n_va = int(len(labelled) * 0.8), int(len(labelled) * 0.1)
        tr = labelled[:n_tr]
        va = labelled[n_tr + kx.DAY_BARS: n_tr + n_va]
        te = labelled[n_tr + n_va + kx.DAY_BARS:]
    elif args.train_end:
        cut = int(np.searchsorted(d.Date.values, np.datetime64(pd.Timestamp(args.train_end)), side="right")) - 1
        # a row's next-day target looks DAY_BARS ahead: keep its horizon inside the training period
        tr = labelled[labelled + kx.DAY_BARS <= cut]
        va = np.array([], dtype=int)
        te = labelled[labelled > cut]
    else:
        tr, va, te = labelled, np.array([], dtype=int), np.array([], dtype=int)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    pipe = fit_pipeline(d, Y, tr)
    y_scaler = pipe["y_scaler"]
    feats = make_features(d, pipe)
    Ys = y_scaler.transform(np.nan_to_num(Y))

    train_dl = kx.loader(kx.windows(feats, tr), Ys[tr], shuffle=False)
    val_dl = kx.loader(kx.windows(feats, va), Ys[va], shuffle=False) if len(va) else None

    model = kx.SAG_LSTM(input_size=feats.shape[1], hidden_size=128, num_layers=2,
                        output_size=len(kx.TARGETS), dropout=0.2).to(device)
    state, best_ep, best_val = run_epochs(model, train_dl, val_dl, epochs)

    torch.save({"model": state, "targets": kx.TARGETS, "window": kx.WINDOW, "hidden_size": 128,
                "num_layers": 2, "dropout": 0.2, "input_size": feats.shape[1]}, out / "model.pt")
    export_pipeline(out / "pipeline.npz", pipe)

    dt = lambda i: str(d.Date.iloc[i])
    meta = {
        "mode": args.mode, "train_end": args.train_end, "rows": args.rows, "data": Path(args.data).name,
        "data_first": dt(0), "data_last": dt(n - 1),
        "epochs": epochs, "best_epoch": best_ep,
        "best_val_loss": None if val_dl is None else best_val,
        "train": [dt(tr[0]), dt(tr[-1])],
        "val": [dt(va[0]), dt(va[-1])] if len(va) else None,
        "test": [dt(te[0]), dt(te[-1])] if len(te) else None,
    }

    if len(te):
        close = d["Close"].values

        def prices(idx):
            p = y_scaler.inverse_transform(kx.predict(model, kx.windows(feats, idx)))
            return kx.to_prices(p, close[idx])

        if len(va):
            pv, av = prices(va), kx.to_prices(Y[va], close[va])
            meta["validation_mae"] = float(np.abs(pv - av).mean())
        pt, at = prices(te), kx.to_prices(Y[te], close[te])
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

        print("\nTEST MAE (SAG_LSTM vs naive = start close):")
        for t in kx.TARGETS:
            print(f"  {t:10s} {meta['test_mae'][t]:9.2f}   naive {meta['test_naive_mae'][t]:9.2f}")
        print(f"  15m direction accuracy: {meta['direction_accuracy']:.3f}")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nSaved artifacts to:", out)


if __name__ == "__main__":
    main()
