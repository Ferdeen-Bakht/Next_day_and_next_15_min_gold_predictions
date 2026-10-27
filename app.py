# -*- coding: utf-8 -*-
"""
XAU/USD SAG-LSTM viewer (Streamlit).

Pick a start bar -> see predicted next-15m (High/Low/Close) and next-day (Open/High/Low/Close)
prices relative to the Close of the start bar, next to the actual values when they exist.

    python train_local.py --mode eval --rows 100000     # iteration 1 (honest test split)
    python train_local.py --mode final --rows 100000    # iteration 2 (production, all data)
    streamlit run app.py
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import kaggle_xau_train as kx
from infer import actuals, fixed_pred, load_bundle, load_features, make_features, predict_rows

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
STEP = pd.Timedelta(minutes=15)

st.set_page_config(page_title="XAU/USD SAG-LSTM", layout="wide")

LABELS = {"n15_High": "Next 15m High", "n15_Low": "Next 15m Low", "n15_Close": "Next 15m Close",
          "nd_Open": "Next day Open", "nd_High": "Next day High", "nd_Low": "Next day Low",
          "nd_Close": "Next day Close"}


# ------------------------------------------------------------
# cached loaders
# ------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def load_d(data_name, rows):
    return load_features(HERE / data_name, rows)


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_dir, _d_key):
    d = load_d(*_d_key)
    model, pipe, _ = load_bundle(model_dir)
    return model, pipe, make_features(d, pipe)


def status_of(ts, meta):
    def inside(rng):
        return rng and pd.Timestamp(rng[0]) <= ts <= pd.Timestamp(rng[1])
    if inside(meta["train"]):
        return "Used in TRAINING", "error"
    if inside(meta.get("val")):
        return "Used in VALIDATION (epoch selection)", "warning"
    if inside(meta.get("test")):
        return "Out-of-sample TEST bar (never trained on)", "success"
    return "Outside the train/val/test ranges (e.g. 96-bar gap or unlabelled tail)", "info"


# ------------------------------------------------------------
# sidebar: model + start time
# ------------------------------------------------------------

st.title("XAU/USD · SAG-LSTM · next 15 min & next day")

models = {p.parent.name: p.parent for p in sorted(OUT.glob("sag_*/meta.json"))}
if not models:
    st.error("No trained model found. Run `python train_local.py --mode eval --rows 100000` first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    names = {"sag_eval": "Iteration 1 · eval (80/10/10 split)",
             "sag_final": "Iteration 2 · final (all data)"}
    key = st.radio("Trained model", list(models), format_func=lambda k: names.get(k, k))
    meta = json.loads((models[key] / "meta.json").read_text())
    st.caption(f"Data {meta['data']} · last {meta['rows']:,} bars\n\n"
               f"{meta['data_first']} → {meta['data_last']}")
    st.caption(f"Train: {meta['train'][0]} → {meta['train'][1]}")
    if meta.get("val"):
        st.caption(f"Val: {meta['val'][0]} → {meta['val'][1]}")
    if meta.get("test"):
        st.caption(f"Test: {meta['test'][0]} → {meta['test'][1]}")
    if meta["mode"] == "final" and meta.get("train_end"):
        st.info(f"Trained only up to {meta['train_end']} (targets included). "
                "Bars after that are out-of-sample.")
    elif meta["mode"] == "final":
        st.warning("Trained on all labelled bars, so there is no honest accuracy check on them. "
                   "Use the eval model's numbers for accuracy.")

d_key = (meta["data"], meta["rows"])
d = load_d(*d_key)
model, pipe, feats = load_model(str(models[key]), d_key)

first_ok = kx.WINDOW - 1
dates = d.Date
with st.sidebar:
    st.header("Start time")
    default_ts = pd.Timestamp(meta["test"][1]) if meta.get("test") else dates.iloc[-1]
    if st.session_state.get("model_key") != key:      # new model -> reset to its default start bar
        st.session_state.update(model_key=key, day=default_ts.date(), hhmm=default_ts.strftime("%H:%M"))
    if st.button("Jump to last test bar" if meta.get("test") else "Jump to latest bar"):
        st.session_state.update(day=default_ts.date(), hhmm=default_ts.strftime("%H:%M"))
    day = st.date_input("Date", min_value=dates.iloc[first_ok].date(), max_value=dates.iloc[-1].date(),
                        key="day")
    day_idx = np.where((dates.dt.date == day) & (np.arange(len(d)) >= first_ok))[0]
    if len(day_idx) == 0:
        st.warning("No bars on that date (weekend / holiday).")
        st.stop()
    hhmm_opts = [dates.iloc[i].strftime("%H:%M") for i in day_idx]
    if st.session_state.get("hhmm") not in hhmm_opts:
        st.session_state["hhmm"] = hhmm_opts[-1]
    hhmm = st.selectbox("Time (HH:MM)", hhmm_opts, key="hhmm")
    t = int(day_idx[hhmm_opts.index(hhmm)])

# ------------------------------------------------------------
# prediction (all prices are anchored to the start bar's Close)
# ------------------------------------------------------------

start_ts = dates.iloc[t]
start_close = float(d.Close[t])
pred = fixed_pred(predict_rows(model, pipe, feats, d["Close"].values, [t])[0])   # High>=O/C, Low<=O/C
act = actuals(d, t)

label, kind = status_of(start_ts, meta)
getattr(st, kind)(f"Start bar {start_ts:%Y-%m-%d %H:%M}: {label}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Start time", f"{start_ts:%H:%M}", f"{start_ts:%Y-%m-%d}", delta_color="off")
c2.metric("Start time in minutes", f"{start_ts.hour * 60 + start_ts.minute} min",
          f"{start_ts.hour}h {start_ts.minute:02d}m since midnight", delta_color="off")
c3.metric("Start price (Close)", f"${start_close:,.2f}")
c4.metric("Predicted next 15m Close", f"${pred['n15_Close']:,.2f}",
          f"{pred['n15_Close'] - start_close:+.2f}  ({(pred['n15_Close'] / start_close - 1) * 100:+.3f}%)")

n15_t = f"{start_ts + STEP:%m-%d %H:%M}"
nd_span = f"{start_ts + STEP:%m-%d %H:%M} → {start_ts + STEP * kx.DAY_BARS:%m-%d %H:%M}"
when = {"n15_High": n15_t, "n15_Low": n15_t, "n15_Close": n15_t,
        "nd_Open": f"{start_ts + STEP:%m-%d %H:%M}", "nd_High": nd_span, "nd_Low": nd_span,
        "nd_Close": f"{start_ts + STEP * kx.DAY_BARS:%m-%d %H:%M}"}


def table(keys):
    rows = []
    for k in keys:
        a = act[k]
        rows.append({
            "Target": LABELS[k], "Time": when[k],
            "Start price": round(start_close, 2),
            "Predicted": round(pred[k], 2),
            "Δ vs start $": round(pred[k] - start_close, 2),
            "Δ vs start %": round((pred[k] / start_close - 1) * 100, 3),
            "Actual": None if np.isnan(a) else round(float(a), 2),
            "Error (pred − actual)": None if np.isnan(a) else round(pred[k] - float(a), 2),
        })
    return pd.DataFrame(rows)


st.subheader(f"Next 15 minutes  (bar {start_ts + STEP:%H:%M}–{start_ts + 2 * STEP:%H:%M})")
st.dataframe(table(["n15_High", "n15_Low", "n15_Close"]), hide_index=True, use_container_width=True)
st.subheader(f"Next day  (24h window {nd_span})")
st.dataframe(table(["nd_Open", "nd_High", "nd_Low", "nd_Close"]), hide_index=True, use_container_width=True)
if np.isnan(act["nd_Close"]):
    st.caption("Actual values are blank where the data ends before the horizon is complete.")

# ------------------------------------------------------------
# chart
# ------------------------------------------------------------

lo, hi = max(0, t - 64), min(len(d), t + kx.DAY_BARS + 1)
v = d.iloc[lo:hi]
fig = go.Figure(go.Candlestick(x=v.Date, open=v.Open, high=v.High, low=v.Low, close=v.Close,
                               name="Actual", increasing_line_color="#2e7d32", decreasing_line_color="#c62828"))
fig.add_vline(x=start_ts, line_dash="dot", line_color="gray")
fig.add_annotation(x=start_ts, y=1, yref="paper", text=f"start {start_ts:%H:%M}", showarrow=False)
fig.add_hline(y=start_close, line_dash="dot", line_color="gray")
t1, t96 = start_ts + STEP, start_ts + STEP * kx.DAY_BARS
pts = [("n15 High", t1, pred["n15_High"], "triangle-up"), ("n15 Low", t1, pred["n15_Low"], "triangle-down"),
       ("n15 Close", t1, pred["n15_Close"], "circle"), ("day Open", t1, pred["nd_Open"], "square"),
       ("day Close", t96, pred["nd_Close"], "diamond")]
for name, x_, y_, sym in pts:
    fig.add_trace(go.Scatter(x=[x_], y=[y_], mode="markers", name=f"Pred {name}",
                             marker=dict(size=11, symbol=sym, line=dict(width=1, color="black"))))
fig.add_shape(type="line", x0=t1, x1=t96, y0=pred["nd_High"], y1=pred["nd_High"],
              line=dict(color="#1565c0", dash="dash"))
fig.add_shape(type="line", x0=t1, x1=t96, y0=pred["nd_Low"], y1=pred["nd_Low"],
              line=dict(color="#ef6c00", dash="dash"))
fig.add_annotation(x=t96, y=pred["nd_High"], text="pred day High", showarrow=False, yshift=10)
fig.add_annotation(x=t96, y=pred["nd_Low"], text="pred day Low", showarrow=False, yshift=-10)
fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                  legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# iteration-1 accuracy (the only honest numbers)
# ------------------------------------------------------------

ev = models.get("sag_eval")
if ev is not None:
    em = json.loads((ev / "meta.json").read_text())
    with st.expander("Iteration 1 test accuracy (out-of-sample, the number to report)"):
        st.dataframe(pd.DataFrame({"SAG-LSTM MAE $": em["test_mae"],
                                   "Naive MAE $ (start close)": em["test_naive_mae"]}).round(2),
                     use_container_width=True)
        st.write(f"15m direction accuracy: **{em['direction_accuracy'] * 100:.2f}%** · "
                 f"validation MAE ${em['validation_mae']:.2f} · best epoch {em['best_epoch']}")
