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
    # returns'i sifirla
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
    # snapshot'i sifirla
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

def compute_for_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date").copy()

    # getiriler (adj_close üzerinden)
    for k, n in WINDOWS.items():
        df[k] = df["adj_close"] / df["adj_close"].shift(n) - 1.0

    # 30 günlük hacim z-skoru
    vol_mean = df["volume"].rolling(30).mean()
    vol_std  = df["volume"].rolling(30).std()
    vol_std_safe = vol_std.replace(0, np.nan)
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std_safe

    last = df.iloc[-1:].copy()
    cols = ["symbol","date"] + list(WINDOWS.keys()) + ["vol_z"]
    last = last[cols].rename(columns={"date": "asof"})
    last["asof"] = pd.to_datetime(last["asof"]).dt.strftime("%Y-%m-%d")
    last["vol_z"] = last["vol_z"].fillna(0.0)
    return last

def build_snapshot(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    # sembol bazinda SON fiyat
    last_px = (
        prices.sort_values(["symbol","date"])
              .groupby("symbol", as_index=False)
              .tail(1)
              .rename(columns={"date":"asof_px", "adj_close":"last_price"})[["symbol","asof_px","last_price"]]
    )

    # 52 haftalik aralik: her sembol için son 252 satiri alip min/max
    tail252 = (
        prices.sort_values(["symbol","date"])
              .groupby("symbol", as_index=False, group_keys=False)
              .tail(252)
    )
    rng = tail252.groupby("symbol")["adj_close"].agg(hi_52w="max", lo_52w="min").reset_index()

    # returns'tan son satir (her sembol)
    snap = returns.sort_values(["symbol","asof"]).groupby("symbol", as_index=False).tail(1)

    # asof çakismasini önlemek için last_px'te asof'u 'asof_px' yaptik; merge güvenli
    snap = snap.merge(last_px, on="symbol", how="left").merge(rng, on="symbol", how="left")

    # mesafeler (yüzde)
    snap["pct_from_hi"] = (snap["last_price"] / snap["hi_52w"] - 1.0) * 100.0
    snap["pct_from_lo"] = (snap["last_price"] / snap["lo_52w"] - 1.0) * 100.0

    cols = [
        "symbol","asof","last_price",
        "r_1d","r_1w","r_1m","r_3m","r_6m","r_1y",
        "vol_z","hi_52w","lo_52w","pct_from_hi","pct_from_lo"
    ]
    # 'asof' returns'tan geldi; 'asof_px' sadece diagnostik amaçli, disari vermiyoruz
    return snap[cols]

def main():
    with sqlite3.connect(str(DB)) as con:
        prices = pd.read_sql_query("SELECT * FROM prices", con, parse_dates=["date"])
        if prices.empty:
            print("No prices in DB. Run fetch/prices.py first.")
            return

        ensure_tables(con)

        out = []
        for sym, g in prices.groupby("symbol", sort=False):
            r = compute_for_symbol(g)
            if not r.empty:
                out.append(r)
        if not out:
            print("No returns computed.")
            return

        ret = pd.concat(out, ignore_index=True)

        # returns yaz
        rows = ret[["symbol","asof","r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","vol_z"]].values.tolist()
        con.executemany("""
        INSERT INTO returns(symbol,asof,r_1d,r_1w,r_1m,r_3m,r_6m,r_1y,vol_z)
        VALUES (?,?,?,?,?,?,?,?,?)
        """, rows)

        # snapshot yaz
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
