# -*- coding: utf-8 -*-
# StockWizard_AI - Fetch via Yahoo "chart" JSON (no yfinance), full reset on each run
# /home/train/StockWizard_AI/fetch/prices.py

import os, time, random, sqlite3, re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB = DATA_DIR / "movers.db"
SYMS = DATA_DIR / "symbols_bist100.txt"

# Tunables via env
MAX_SYMBOLS_PER_RUN = int(os.environ.get("RUN_MAX_SYMBOLS", "100000"))
INIT_DAYS           = int(os.environ.get("RUN_INIT_DAYS", "800"))
RETRIES             = int(os.environ.get("RUN_RETRIES", "5"))
BASE_BACKOFF        = float(os.environ.get("RUN_BASE_BACKOFF", "4.0"))
BACKOFF_JITTER      = float(os.environ.get("RUN_BACKOFF_JITTER", "1.5"))
USER_AGENT          = os.environ.get("RUN_UA", "Mozilla/5.0")

class SymbolNotFoundError(Exception):
    pass

def epoch_utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def reset_db():
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS prices(
        symbol TEXT NOT NULL,
        date   TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
        PRIMARY KEY (symbol, date)
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date);")
    # NEW: symbols table for names
    cur.execute("""
    CREATE TABLE IF NOT EXISTS symbols(
        symbol TEXT PRIMARY KEY,
        short_name TEXT,
        long_name  TEXT
    );
    """)
    con.commit()
    con.close()

def read_text_smart(path: Path) -> str:
    encs = ["utf-8-sig", "cp1254", "iso-8859-9", "latin-1"]
    for enc in encs:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("latin-1", errors="replace")

def load_symbols() -> list[str]:
    raw = read_text_smart(SYMS) if SYMS.exists() else ""
    symbols = []
    for line in raw.splitlines():
        s = re.sub(r"[^A-Za-z0-9\.]", "", line.strip().upper())
        if not s:
            continue
        if not s.endswith(".IS"):
            s += ".IS"
        symbols.append(s)
    return sorted(set(symbols))

def fetch_chart(symbol: str, start_date: datetime, end_date: datetime):
    """
    Returns (df, meta_names) where df is OHLCV and meta_names is dict(short_name,long_name).
    """
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
    if r.status_code == 404:
        raise SymbolNotFoundError(f"404 Not Found for {symbol}")
    r.raise_for_status()

    js = r.json()
    ch = js.get("chart", {}) or {}
    if ch.get("error"):
        code = (ch["error"].get("code") or "").lower()
        msg  = ch["error"].get("description") or ch["error"].get("message") or str(ch["error"])
        if "not" in code or "no data" in code or "bad" in code:
            raise SymbolNotFoundError(f"{symbol}: {msg}")
        raise RuntimeError(f"{symbol}: chart.error -> {msg}")

    res = (ch.get("result") or [])
    if not res:
        return pd.DataFrame(), {}

    res = res[0]
    meta = res.get("meta", {}) or {}
    short_name = meta.get("shortName")
    long_name  = meta.get("longName")

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
    df = df.dropna(subset=["close","adj_close"], how="any")
    return df, {"short_name": short_name, "long_name": long_name}

def upsert_prices(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    rows = df[["symbol","date","open","high","low","close","adj_close","volume"]].values.tolist()
    con.executemany("""
        INSERT INTO prices(symbol,date,open,high,low,close,adj_close,volume)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol,date) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low,
          close=excluded.close, adj_close=excluded.adj_close, volume=excluded.volume;
    """, rows)
    con.commit()
    return len(rows)

def upsert_symbol(con: sqlite3.Connection, symbol: str, names: dict):
    if not names:
        return
    con.execute("""
        INSERT INTO symbols(symbol, short_name, long_name)
        VALUES (?,?,?)
        ON CONFLICT(symbol) DO UPDATE SET
          short_name=COALESCE(excluded.short_name, short_name),
          long_name =COALESCE(excluded.long_name, long_name);
    """, (symbol, names.get("short_name"), names.get("long_name")))
    con.commit()

def fetch_all(symbols: list[str]) -> int:
    total = 0
    with sqlite3.connect(str(DB)) as con:
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=INIT_DAYS)
        end   = today + timedelta(days=1)
        for i, sym in enumerate(symbols[:MAX_SYMBOLS_PER_RUN], 1):
            for attempt in range(1, RETRIES + 1):
                try:
                    df, meta_names = fetch_chart(
                        sym,
                        datetime.combine(start, datetime.min.time()),
                        datetime.combine(end,   datetime.min.time()),
                    )
                    n = upsert_prices(con, df)
                    upsert_symbol(con, sym, meta_names)
                    print(f"{sym}: rows {n}")
                    total += n
                    break
                except SymbolNotFoundError as e:
                    print(f"{sym}: skip -> {e}")
                    break
                except Exception as e:
                    sleep_s = min(BASE_BACKOFF * (2 ** (attempt - 1)), 300) + random.uniform(0.0, BACKOFF_JITTER)
                    print(f"{sym}: attempt {attempt}/{RETRIES} -> {e} (sleep {sleep_s:.1f}s)")
                    time.sleep(sleep_s)
    return total

def main():
    ensure_dirs()
    reset_db()
    symbols = load_symbols()
    if not symbols:
        print("No symbols file. Put symbols_bist100.txt under data/")
        return
    t0 = time.perf_counter()
    total = fetch_all(symbols)
    dt = time.perf_counter() - t0
    print(f"DONE. Inserted rows: {total}")
    m, s = divmod(int(dt), 60); h, m = divmod(m, 60)
    print(f"Elapsed: {h:d}:{m:02d}:{s:02d}")

if __name__ == "__main__":
    main()
