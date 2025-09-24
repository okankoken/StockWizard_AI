# /home/train/StockWizard_AI/setup.sh
#!/usr/bin/env bash
set -euo pipefail

BASE="/home/train/StockWizard_AI"

echo ">> Creating folders under $BASE"
mkdir -p "$BASE"/{data,fetch,ui}

echo ">> Writing requirements.txt"
cat > "$BASE/requirements.txt" <<'TXT'
yfinance==0.2.43
pandas==2.2.2
numpy==1.26.4
streamlit==1.37.1
TXT

echo ">> Writing data/symbols_bist100.txt (seed list, edit as you wish)"
cat > "$BASE/data/symbols_bist100.txt" <<'TXT'
ASELS.IS
THYAO.IS
BIMAS.IS
SISE.IS
FROTO.IS
KCHOL.IS
EREGL.IS
TOASO.IS
YKBNK.IS
HALKB.IS
TXT

echo ">> Writing fetch/prices.py"
cat > "$BASE/fetch/prices.py" <<'PY'
# -*- coding: utf-8 -*-
import sqlite3, time
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "movers.db"
SYMS = ROOT / "data" / "symbols_bist100.txt"

DB.parent.mkdir(parents=True, exist_ok=True)

def ensure_prices_table(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS prices(
        symbol TEXT NOT NULL,
        date   TEXT NOT NULL,
        open   REAL, high REAL, low REAL, close REAL, adj_close REAL,
        volume REAL,
        PRIMARY KEY (symbol, date)
    );
    """)
    conn.commit()

def load_symbols():
    with open(SYMS, "r", encoding="utf-8") as f:
        syms = [s.strip() for s in f if s.strip()]
    return sorted(set(syms))

def download_prices(symbol: str, period="2y"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d", auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["open","high","low","close","volume"]]
    # adj close separately (auto_adjust=True)
    df_adj = ticker.history(period=period, interval="1d", auto_adjust=True)
    if not df_adj.empty and "Close" in df_adj.columns:
        df["adj_close"] = df_adj["Close"].values
    else:
        df["adj_close"] = df["close"].values
    df["symbol"] = symbol
    df["date"] = df.index.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)

def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return
    rows = df[["symbol","date","open","high","low","close","adj_close","volume"]].values.tolist()
    conn.executemany("""
    INSERT INTO prices(symbol,date,open,high,low,close,adj_close,volume)
    VALUES (?,?,?,?,?,?,?,?)
    ON CONFLICT(symbol,date) DO UPDATE SET
      open=excluded.open, high=excluded.high, low=excluded.low,
      close=excluded.close, adj_close=excluded.adj_close, volume=excluded.volume;
    """, rows)
    conn.commit()

def main():
    syms = load_symbols()
    with sqlite3.connect(DB) as conn:
        ensure_prices_table(conn)
        for i, s in enumerate(syms, 1):
            try:
                df = download_prices(s, period="2y")
                upsert_prices(conn, df)
                print(f"[{i}/{len(syms)}] {s}: {len(df)} rows")
                time.sleep(0.8)
            except Exception as e:
                print(f"ERR {s}: {e}")

if __name__ == "__main__":
    main()
PY

echo ">> Writing fetch/compute.py"
cat > "$BASE/fetch/compute.py" <<'PY'
# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path
import pandas as pd

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
    # percentage changes over adj_close
    for k, n in WINDOWS.items():
        df[k] = df["adj_close"] / df["adj_close"].shift(n) - 1.0
    # 30-day volume z-score
    vol_roll = df["volume"].rolling(30)
    df["vol_z"] = (df["volume"] - vol_roll.mean()) / vol_roll.std()
    last = df.iloc[-1:].copy()
    keep = ["symbol","date"] + list(WINDOWS.keys()) + ["vol_z"]
    last = last[keep].rename(columns={"date":"asof"})
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
        print(f"Computed returns for {len(ret)} symbols (asof={ret['asof'].iloc[0].date()}).")

if __name__ == "__main__":
    main()
PY

echo ">> Writing ui/app.py"
cat > "$BASE/ui/app.py" <<'PY'
# -*- coding: utf-8 -*-
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "movers.db"

st.set_page_config(page_title="StockWizard_AI", layout="wide")

@st.cache_data(ttl=600)
def load_returns():
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql_query("SELECT * FROM returns", conn, parse_dates=["asof"])
    return df

@st.cache_data(ttl=600)
def load_prices(symbol):
    with sqlite3.connect(DB) as conn:
        q = "SELECT date, adj_close, volume FROM prices WHERE symbol=? ORDER BY date"
        df = pd.read_sql_query(q, conn, params=[symbol], parse_dates=["date"])
    return df

@st.cache_data(ttl=600)
def load_symbols():
    with sqlite3.connect(DB) as conn:
        df = pd.read_sql_query("SELECT DISTINCT symbol FROM prices ORDER BY symbol", conn)
    return df["symbol"].tolist()

@st.cache_data(ttl=600)
def load_xu100(period="1y"):
    t = yf.Ticker("^XU100")
    hist = t.history(period=period, interval="1d", auto_adjust=True)
    hist.index = hist.index.tz_localize(None)
    return hist

st.title("🪄 StockWizard_AI")
st.caption("Realized moves only · Not investment advice")

window_map = {
    "1 day": "r_1d",
    "1 week": "r_1w",
    "1 month": "r_1m",
    "3 months": "r_3m",
    "6 months": "r_6m",
    "1 year": "r_1y",
}
sel_label = st.sidebar.selectbox("Change window", list(window_map.keys()), index=0)
metric_col = window_map[sel_label]
direction = st.sidebar.radio("Direction", ["Top Gainers", "Top Losers"], index=0)
limit = st.sidebar.slider("How many?", 5, 30, 10)
min_volz = st.sidebar.slider("Min volume z-score (anomaly)", -2.0, 3.0, -999.0, 0.1)

ret = load_returns()
if ret.empty:
    st.warning("No returns yet. Run fetch/prices.py then fetch/compute.py")
    st.stop()

ret = ret[~ret[metric_col].isna()].copy()
if min_volz > -900:
    ret = ret[ret["vol_z"] >= min_volz]

ret["rank"] = ret[metric_col].rank(ascending=(direction=="Top Losers"), method="first")
ret = ret.sort_values("rank")

left, right = st.columns([2, 1])

with left:
    st.subheader(f"{direction} · {sel_label}")
    show = ret[["symbol", metric_col, "vol_z", "asof"]].head(limit).copy()
    show = show.rename(columns={metric_col: "return", "vol_z": "volume_z", "asof": "as_of"})
    show["return"] = (show["return"] * 100).round(2)
    show["volume_z"] = show["volume_z"].round(2)
    st.dataframe(show, use_container_width=True)

with right:
    st.subheader("Symbol details")
    symbols = load_symbols()
    default_sym = show["symbol"].iloc[0] if not show.empty else (symbols[0] if symbols else None)
    idx = symbols.index(default_sym) if (symbols and default_sym in symbols) else 0
    sym = st.selectbox("Pick a symbol", symbols, index=idx if symbols else 0)
    if sym:
        p = load_prices(sym)
        if not p.empty:
            p["ret_1m"] = p["adj_close"].pct_change(21) * 100
            st.line_chart(p.set_index("date")[["adj_close"]])
            st.caption("Adj Close (auto-adjusted)")
            st.line_chart(p.set_index("date")[["ret_1m"]])
            st.caption("Approx. 1M return (%)")
        else:
            st.info("No price history for this symbol.")
    st.divider()
    st.caption("⚠️ This is not investment advice.")
PY

echo ">> Writing run.sh"
cat > "$BASE/run.sh" <<'SH'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python fetch/prices.py
python fetch/compute.py

streamlit run ui/app.py
SH
chmod +x "$BASE/run.sh"

echo "✅ Setup complete.
Next steps:
  cd $BASE
  chmod +x setup.sh
  ./run.sh
"

