# -*- coding: utf-8 -*-
# StockWizard_AI - Fetch via Yahoo "chart" JSON (no yfinance), incremental upsert

import os, time, random, sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "movers.db"
SYMS = ROOT / "data" / "symbols_bist100.txt"

# Tunables via env
MAX_SYMBOLS_PER_RUN = int(os.environ.get("RUN_MAX_SYMBOLS", "5"))    # how many symbols to process this run
INIT_DAYS           = int(os.environ.get("RUN_INIT_DAYS", "800"))    # initial lookback window
LOOKBACK_PAD        = int(os.environ.get("RUN_LOOKBACK_PAD", "12"))  # overlap days on resume
RETRIES             = int(os.environ.get("RUN_RETRIES", "6"))
BASE_BACKOFF        = float(os.environ.get("RUN_BASE_BACKOFF", "6.0"))
BACKOFF_JITTER      = float(os.environ.get("RUN_BACKOFF_JITTER", "2.0"))
USER_AGENT          = os.environ.get("RUN_UA", "Mozilla/5.0")

def ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS prices(
        symbol TEXT NOT NULL,
        date   TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
        PRIMARY KEY (symbol, date)
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date);")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fetch_state(
        id INTEGER PRIMARY KEY CHECK (id=1),
        cursor INTEGER NOT NULL
    );
    """)
    cur = conn.execute("SELECT cursor FROM fetch_state WHERE id=1").fetchone()
    if cur is None:
        conn.execute("INSERT INTO fetch_state(id, cursor) VALUES (1, 0);")
    conn.commit()

def load_symbols() -> list[str]:
    symbols = []
    with open(SYMS, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().upper()
            if not s:
                continue
            if not s.endswith(".IS"):
                s += ".IS"
            symbols.append(s)
    return sorted(set(symbols))

def get_cursor(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT cursor FROM fetch_state WHERE id=1").fetchone()[0])

def set_cursor(conn: sqlite3.Connection, val: int):
    conn.execute("UPDATE fetch_state SET cursor=? WHERE id=1", (val,))
    conn.commit()

def latest_date_for(conn: sqlite3.Connection, symbol: str) -> str | None:
    row = conn.execute("SELECT MAX(date) FROM prices WHERE symbol=?", (symbol,)).fetchone()
    return row[0] if row and row[0] else None

def epoch_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def fetch_chart(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": epoch_utc(start_date),
        "period2": epoch_utc(end_date),
        "interval": "1d",
        "events": "div,split",
        "includePrePost": "false",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code == 429:
        raise RuntimeError("429 Too Many Requests")
    r.raise_for_status()
    js = r.json()
    res = (js.get("chart", {}) or {}).get("result") or []
    if not res:
        return pd.DataFrame()
    res = res[0]
    ts = res.get("timestamp", []) or []
    ind = res.get("indicators", {}) or {}
    quote = (ind.get("quote") or [{}])[0]
    adj   = (ind.get("adjclose") or [{}])[0]
    rows = []
    for i, t in enumerate(ts):
        def take(arr):
            return arr[i] if isinstance(arr, list) and i < len(arr) else None
        o = take(quote.get("open"))
        h = take(quote.get("high"))
        l = take(quote.get("low"))
        c = take(quote.get("close"))
        v = take(quote.get("volume"))
        ac = take(adj.get("adjclose")) if adj else None
        if c is None:
            continue
        if ac is None:
            ac = c
        d = datetime.fromtimestamp(t, timezone.utc).date().isoformat()
        rows.append([symbol, d, o, h, l, c, ac, v])
    df = pd.DataFrame(rows, columns=["symbol","date","open","high","low","close","adj_close","volume"])
    # drop rows where close/adj_close are NaN
    df = df.dropna(subset=["close","adj_close"], how="any")
    return df

def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    rows = df[["symbol","date","open","high","low","close","adj_close","volume"]].values.tolist()
    conn.executemany("""
        INSERT INTO prices(symbol,date,open,high,low,close,adj_close,volume)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol,date) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low,
          close=excluded.close, adj_close=excluded.adj_close, volume=excluded.volume;
    """, rows)
    conn.commit()
    return len(rows)

def exp_backoff(attempt: int):
    sleep_s = min(BASE_BACKOFF * (2 ** (attempt - 1)), 300) + random.uniform(0.0, BACKOFF_JITTER)
    time.sleep(sleep_s)

def process_symbol(conn: sqlite3.Connection, symbol: str) -> int:
    today = datetime.now(timezone.utc).date()
    last = latest_date_for(conn, symbol)
    if last:
        start = datetime.strptime(last, "%Y-%m-%d").date() - timedelta(days=LOOKBACK_PAD)
    else:
        start = today - timedelta(days=INIT_DAYS)
    end = today + timedelta(days=1)  # include today

    for attempt in range(1, RETRIES + 1):
        try:
            df = fetch_chart(symbol,
                             datetime.combine(start, datetime.min.time()),
                             datetime.combine(end,   datetime.min.time()))
            n = upsert_prices(conn, df)
            print(f"{symbol}: upserted {n} rows")
            return n
        except Exception as e:
            print(f"{symbol}: attempt {attempt}/{RETRIES} -> {e}")
            exp_backoff(attempt)
    print(f"{symbol}: failed after retries")
    return 0

def main():
    syms = load_symbols()
    if not syms:
        print("No symbols found.")
        return
    with sqlite3.connect(DB) as conn:
        ensure_schema(conn)
        cur = get_cursor(conn)
        total = 0
        for i in range(MAX_SYMBOLS_PER_RUN):
            idx = (cur + i) % len(syms)
            total += process_symbol(conn, syms[idx])
        set_cursor(conn, (cur + MAX_SYMBOLS_PER_RUN) % len(syms))
        print(f"Done. Upserted rows: {total}")

if __name__ == "__main__":
    main()
