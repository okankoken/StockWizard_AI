# -*- coding: utf-8 -*-
# StockWizard_AI - Compute returns and snapshot (overwrite tables)
# /home/train/StockWizard_AI/fetch/compute.py

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "movers.db"

WINDOWS = {
    "r_1d": 1,
    "r_1w": 5,
    "r_1m": 21,
    "r_3m": 63,
    "r_6m": 126,
    "r_1y": 252,
}

def ensure_tables(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS returns;")
    cur.execute("""
    CREATE TABLE returns(
        symbol TEXT NOT NULL,
        asof   TEXT NOT NULL,
        r_1d REAL, r_1w REAL, r_1m REAL, r_3m REAL, r_6m REAL, r_1y REAL,
        vol_z REAL,
        PRIMARY KEY(symbol, asof)
    );
    """)
    cur.execute("DROP TABLE IF EXISTS snapshot;")
    cur.execute("""
    CREATE TABLE snapshot(
        symbol TEXT PRIMARY KEY,
        asof   TEXT NOT NULL,
        last_price REAL,
        r_1d REAL, r_1w REAL, r_1m REAL, r_3m REAL, r_6m REAL, r_1y REAL,
        vol_z REAL,
        hi_52w REAL, lo_52w REAL,
        pct_from_hi REAL, pct_from_lo REAL
    );
    """)
    con.commit()

def _calendar_lookback_return(df: pd.DataFrame, days: int, base_col: str = "close") -> float | None:
    """
    Go back by N calendar days; pick the last trading day on/before that date,
    then return (last / past - 1). Uses 'close' to align with Yahoo performance.
    """
    if df.empty or base_col not in df:
        return None
    x = df.dropna(subset=[base_col]).copy()
    x["date"] = pd.to_datetime(x["date"])
    last_date = x["date"].max()
    base_date = last_date - pd.Timedelta(days=days)
    past = x.loc[x["date"] <= base_date]
    if past.empty:
        return None
    past_px = past.sort_values("date").iloc[-1][base_col]
    last_px = x.loc[x["date"] == last_date, base_col].iloc[0]
    if past_px in (None, 0) or pd.isna(past_px):
        return None
    return (last_px / past_px) - 1.0

def compute_for_symbol(df: pd.DataFrame, meta_row: dict | None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date").copy()

    # pre-fill bar-based returns via adj_close (kept for continuity; r_1d will be overridden)
    for k, n in WINDOWS.items():
        df[k] = df["adj_close"] / df["adj_close"].shift(n) - 1.0

    # 30-day volume z-score
    vol_mean = df["volume"].rolling(30).mean()
    vol_std  = df["volume"].rolling(30).std().replace(0, np.nan)
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std

    # last row scaffold
    last = df.iloc[-1:].copy()
    cols = ["symbol","date"] + list(WINDOWS.keys()) + ["vol_z"]
    last = last[cols].rename(columns={"date": "asof"})
    last["asof"] = pd.to_datetime(last["asof"]).dt.strftime("%Y-%m-%d")
    last["vol_z"] = last["vol_z"].fillna(0.0)

    # r_1d: align with Yahoo => last_price / prev_close - 1 (from symbols meta); else fallback bars
    if meta_row:
        prev_close = meta_row.get("prev_close")
        price_now  = meta_row.get("last_price")
        if price_now is None or (isinstance(price_now, float) and np.isnan(price_now)):
            price_now = float(df["close"].iloc[-1])
        if prev_close not in (None, 0) and not (isinstance(prev_close, float) and np.isnan(prev_close)):
            last["r_1d"] = price_now / prev_close - 1.0
        else:
            last["r_1d"] = df["close"].iloc[-1] / df["close"].shift(1).iloc[-1] - 1.0
    else:
        last["r_1d"] = df["close"].iloc[-1] / df["close"].shift(1).iloc[-1] - 1.0

    # other windows: calendar lookback using close (Yahoo-like)
    last["r_1w"] = _calendar_lookback_return(df, 7,   "close")
    last["r_1m"] = _calendar_lookback_return(df, 30,  "close")
    last["r_3m"] = _calendar_lookback_return(df, 90,  "close")
    last["r_6m"] = _calendar_lookback_return(df, 180, "close")
    last["r_1y"] = _calendar_lookback_return(df, 365, "close")

    return last

def build_snapshot(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    # last price per symbol (close)
    last_px = (
        prices.sort_values(["symbol","date"])
              .groupby("symbol", as_index=False)
              .tail(1)
              .rename(columns={"date":"asof_px", "close":"last_price"})[["symbol","asof_px","last_price"]]
    )

    # 52w range by last 365 calendar days (close-based)
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    cutoff = prices["date"].max() - pd.Timedelta(days=365)
    tail365 = prices.loc[prices["date"] >= cutoff]
    rng = tail365.groupby("symbol")["close"].agg(hi_52w="max", lo_52w="min").reset_index()

    # latest returns per symbol
    snap = returns.sort_values(["symbol","asof"]).groupby("symbol", as_index=False).tail(1)

    snap = snap.merge(last_px, on="symbol", how="left").merge(rng, on="symbol", how="left")

    snap["pct_from_hi"] = (snap["last_price"] / snap["hi_52w"] - 1.0) * 100.0
    snap["pct_from_lo"] = (snap["last_price"] / snap["lo_52w"] - 1.0) * 100.0

    cols = [
        "symbol","asof","last_price",
        "r_1d","r_1w","r_1m","r_3m","r_6m","r_1y",
        "vol_z","hi_52w","lo_52w","pct_from_hi","pct_from_lo"
    ]
    return snap[cols]

def main():
    with sqlite3.connect(str(DB)) as con:
        prices = pd.read_sql_query("SELECT * FROM prices", con, parse_dates=["date"])
        if prices.empty:
            print("No prices in DB. Run fetch/prices.py first.")
            return

        # read symbols meta (prev_close, last_price)
        try:
            symbols_meta = pd.read_sql_query("SELECT * FROM symbols", con)
            meta_map = {r["symbol"]: dict(r) for _, r in symbols_meta.iterrows()}
        except Exception:
            meta_map = {}

        ensure_tables(con)

        out = []
        for sym, g in prices.groupby("symbol", sort=False):
            r = compute_for_symbol(g, meta_map.get(sym))
            if not r.empty:
                out.append(r)
        if not out:
            print("No returns computed.")
            return

        ret = pd.concat(out, ignore_index=True)

        # write returns
        rows = ret[["symbol","asof","r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","vol_z"]].values.tolist()
        con.executemany("""
        INSERT INTO returns(symbol,asof,r_1d,r_1w,r_1m,r_3m,r_6m,r_1y,vol_z)
        VALUES (?,?,?,?,?,?,?,?,?)
        """, rows)

        # write snapshot
        snap = build_snapshot(prices, ret)
        srows = snap.values.tolist()
        con.executemany("""
        INSERT INTO snapshot(symbol,asof,last_price,r_1d,r_1w,r_1m,r_3m,r_6m,r_1y,vol_z,hi_52w,lo_52w,pct_from_hi,pct_from_lo)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, srows)

        con.commit()
        print(f"Computed returns for {len(ret)} symbols and snapshot rows {len(snap)} (asof={ret['asof'].iloc[0]}).")

if __name__ == "__main__":
    main()
