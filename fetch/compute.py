# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "movers.db"

WINDOWS = {
    "r_1d":   1,
    "r_1w":   5,
    "r_1m":   21,
    "r_3m":   63,
    "r_6m":   126,
    "r_1y":   252,
}

def ensure_returns_table(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS returns(
        symbol TEXT NOT NULL,
        asof   TEXT NOT NULL,
        r_1d REAL, r_1w REAL, r_1m REAL, r_3m REAL, r_6m REAL, r_1y REAL,
        vol_z REAL,
        PRIMARY KEY(symbol, asof)
    );
    """)
    conn.commit()

def compute_for_symbol(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date").copy()

    # pct changes over adj_close
    for k, n in WINDOWS.items():
        df[k] = df["adj_close"] / df["adj_close"].shift(n) - 1.0

    # volume z-score (30d) with safe std
    vol_mean = df["volume"].rolling(30).mean()
    vol_std  = df["volume"].rolling(30).std()
    vol_std_safe = vol_std.replace(0, np.nan)  # 0 bolmeyi engelle
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std_safe

    # last row (day close)
    last = df.iloc[-1:].copy()
    keep = ["symbol","date"] + list(WINDOWS.keys()) + ["vol_z"]
    last = last[keep].rename(columns={"date": "asof"})

    # asof -> 'YYYY-MM-DD' string (sqlite TEXT)
    last["asof"] = pd.to_datetime(last["asof"]).dt.strftime("%Y-%m-%d")
    # vol_z NaN ise 0 yap (istege bagli)
    last["vol_z"] = last["vol_z"].fillna(0.0)

    return last

def main():
    with sqlite3.connect(DB) as conn:
        ensure_returns_table(conn)
        prices = pd.read_sql_query("SELECT * FROM prices", conn, parse_dates=["date"])
        if prices.empty:
            print("No prices in DB. Run fetch/prices.py first.")
            return

        out = []
        for sym, g in prices.groupby("symbol", sort=False):
            r = compute_for_symbol(g)
            if not r.empty:
                out.append(r)
        if not out:
            print("No returns computed.")
            return

        ret = pd.concat(out, ignore_index=True)

        # Upsert
        rows = ret[["symbol","asof","r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","vol_z"]].values.tolist()
        conn.executemany("""
        INSERT INTO returns(symbol,asof,r_1d,r_1w,r_1m,r_3m,r_6m,r_1y,vol_z)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol,asof) DO UPDATE SET
          r_1d=excluded.r_1d, r_1w=excluded.r_1w, r_1m=excluded.r_1m,
          r_3m=excluded.r_3m, r_6m=excluded.r_6m, r_1y=excluded.r_1y,
          vol_z=excluded.vol_z;
        """, rows)
        conn.commit()
        print(f"Computed returns for {len(ret)} symbols (asof={ret['asof'].iloc[0]}).")

if __name__ == "__main__":
    main()
