# -*- coding: utf-8 -*-
# Build a daily snapshot from SQLite: returns + latest price + 52w hi/lo

import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "movers.db"

def load_snapshot():
    with sqlite3.connect(DB) as conn:
        ret = pd.read_sql_query("SELECT * FROM returns", conn, parse_dates=["asof"])
        px  = pd.read_sql_query("SELECT symbol, date, adj_close FROM prices", conn, parse_dates=["date"])
    if ret.empty or px.empty:
        return pd.DataFrame()

    # latest price per symbol
    last_px = px.sort_values(["symbol","date"]).groupby("symbol", as_index=False).tail(1)
    last_px = last_px.rename(columns={"date":"as_of", "adj_close":"last_price"})
    last_px = last_px[["symbol","as_of","last_price"]]

    # 52w window = 252 trading days
    px = px.sort_values(["symbol","date"])
    px["rn"] = px.groupby("symbol")["date"].rank(method="first")
    hi = px.groupby("symbol").tail(252).groupby("symbol")["adj_close"].max().rename("hi_52w")
    lo = px.groupby("symbol").tail(252).groupby("symbol")["adj_close"].min().rename("lo_52w")
    rng = pd.concat([hi, lo], axis=1).reset_index()

    # merge
    snap = ret.sort_values(["symbol","asof"]).groupby("symbol", as_index=False).tail(1)
    snap = snap.merge(last_px, on=["symbol"], how="left").merge(rng, on="symbol", how="left")

    # distances
    snap["pct_from_hi"] = (snap["last_price"] / snap["hi_52w"] - 1.0) * 100.0
    snap["pct_from_lo"] = (snap["last_price"] / snap["lo_52w"] - 1.0) * 100.0

    # arrange columns
    cols = [
        "symbol","asof","last_price",
        "r_1d","r_1w","r_1m","r_3m","r_6m","r_1y",
        "vol_z","hi_52w","lo_52w","pct_from_hi","pct_from_lo"
    ]
    snap = snap[cols]
    return snap
