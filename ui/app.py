# -*- coding: utf-8 -*-
# StockWizard_AI - Streamlit UI (Dashboard + Chat)
# Notes: ASCII comments only; code UTF-8.

# ensure local imports from this folder work
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import yfinance as yf
import unicodedata
import re
import os
from snapshot import load_snapshot
try:
    from news import news_search
except Exception:
    def news_search(*args, **kwargs):
        return []


# News search helper (Google CSE)
try:
    from ui.news import news_search
except Exception:
    def news_search(*args, **kwargs):
        return []

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

# Title
st.title("StockWizard_AI")
st.caption("Realized moves only. This is not investment advice.")

# Tabs
tab1, tab2 = st.tabs(["Dashboard", "Chat"])

# Dashboard tab
with tab1:
    st.subheader("Market snapshot")
    snap = load_snapshot()
    if snap.empty:
        st.warning("No snapshot yet. Run fetch/prices.py then fetch/compute.py")
        st.stop()

    # filter optional: show only symbols that exist in DB
    symbols_all = load_symbols()
    snap = snap[snap["symbol"].isin(symbols_all)].copy()

    # format for view
    view = snap.copy()
    for c in ["r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","pct_from_hi","pct_from_lo","vol_z"]:
        view[c] = view[c].astype(float)

    # sorting controls
    sort_by = st.sidebar.selectbox(
        "Sort by",
        ["r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","pct_from_hi","pct_from_lo","vol_z","last_price"],
        index=1
    )
    ascending = st.sidebar.checkbox("Ascending", value=False)
    topn = st.sidebar.slider("How many rows", 5, 100, 30)

    view = view.sort_values(sort_by, ascending=ascending).head(topn)

    # pretty percent columns
    def fmt_pct(x):
        return None if pd.isna(x) else f"{x*100:.2f}%" if abs(x) < 10 else f"{x*100:.1f}%"

    pct_cols = ["r_1d","r_1w","r_1m","r_3m","r_6m","r_1y"]
    for c in pct_cols:
        view[c] = (view[c] * 100).round(2)
    view["pct_from_hi"] = view["pct_from_hi"].round(2)
    view["pct_from_lo"] = view["pct_from_lo"].round(2)
    view["vol_z"] = view["vol_z"].round(2)
    view["last_price"] = view["last_price"].round(3)

    st.dataframe(
        view[[
            "symbol","last_price","r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","vol_z","hi_52w","lo_52w","pct_from_hi","pct_from_lo","asof"
        ]],
        use_container_width=True
    )


# Chat tab
with tab2:
    # Message memory
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello. Ask about movers or a symbol. Example: 'top 5 losers today', 'THYAO.IS 1y', 'THYAO.IS news'."}
        ]

    # Show messages
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    user_msg = st.chat_input("Type your question...")
    if user_msg:
        st.session_state["messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        resp = ""

        # Helpers
        def ascii_fold(s: str) -> str:
            # Fold Turkish characters to ASCII
            return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

        def infer_window(t: str) -> str:
            # Map common phrases to returns column
            if ("1 yil" in t) or ("1y" in t) or ("1 year" in t) or ("yillik" in t) or ("yillik" in t):
                return "r_1y"
            if ("6 ay" in t) or ("6a" in t) or ("6 months" in t):
                return "r_6m"
            if ("3 ay" in t) or ("3a" in t) or ("3 months" in t) or ("ceyrek" in t) or ("Ã§eyrek" in t):
                return "r_3m"
            if ("1 ay" in t) or ("1a" in t) or ("1 month" in t) or ("aylik" in t) or ("aylik" in t):
                return "r_1m"
            if ("hafta" in t) or ("1w" in t) or ("week" in t):
                return "r_1w"
            if ("bugun" in t) or ("bugÃ¼n" in t) or ("today" in t) or ("gun" in t) or ("gÃ¼n" in t):
                return "r_1d"
            return "r_1d"

        t = ascii_fold(user_msg).lower()

        # 1) News intent
        if "haber" in t or "news" in t:
            tokens = re.split(r"\s+", user_msg.upper())
            sym = next((tok for tok in tokens if tok.endswith(".IS")), None)
            q = sym if sym else user_msg
            items = news_search(q, num=5)
            if items:
                lines = []
                for it in items:
                    title = it.get("title") or ""
                    link = it.get("link") or ""
                    src = it.get("source") or ""
                    lines.append(f"- **{title}**\n  {link}\n  _{src}_")
                resp = "\n".join(lines)
            else:
                resp = "No news or missing API key."
        else:
            # 2) Performance movers (symbol vs list)
            col = infer_window(t)

            # Direction inference (default up)
            gain_trigs = ["artan", "artmis", "yukselen", "gainers", "up", "pozitif"]
            loss_trigs = ["dusen", "eksi", "kaybeden", "losers", "down", "negatif"]
            if any(w in t for w in loss_trigs):
                direction = "down"
            elif any(w in t for w in gain_trigs):
                direction = "up"
            else:
                direction = "up"

            # Limit (default 10)
            mnum = re.search(r"(\d+)", t)
            limit = int(mnum.group(1)) if mnum else 10

            # Symbol detection
            tokens = re.split(r"\s+", user_msg.upper())
            sym = next((tok for tok in tokens if tok.endswith(".IS")), None)

            if sym:
                # Per-symbol summary from returns
                df = load_returns()
                row = df[df["symbol"] == sym].tail(1)
                if row.empty:
                    resp = f"No summary for {sym}."
                else:
                    r = row.iloc[0]
                    vals = []
                    for k, name in [("r_1d","1d"),("r_1w","1w"),("r_1m","1m"),("r_3m","3m"),("r_6m","6m"),("r_1y","1y")]:
                        v = r.get(k)
                        if pd.notna(v):
                            vals.append(f"{name}: {round(float(v)*100,2)}%")
                    resp = f"**{sym}** (as of {r['asof'].date()}): " + ", ".join(vals)
            else:
                # Use snapshot for richer context (price, vol_z, 52w hi lo)
                snap = load_snapshot()
                if snap.empty:
                    resp = "No snapshot yet. Run fetch/prices.py then fetch/compute.py"
                else:
                    df = snap.copy()
                    df = df[~df[col].isna()].copy()
                    df["rank"] = df[col].rank(ascending=(direction == "down"), method="first")
                    df = df.sort_values("rank").head(limit)
                    if df.empty:
                        resp = "No result."
                    else:
                        lines = [f"Top {limit} {'losers' if direction=='down' else 'gainers'} for {col}:"]
                        for _, rr in df.iterrows():
                            pct = round(float(rr[col]) * 100, 2)
                            price = round(float(rr["last_price"]), 3) if pd.notna(rr["last_price"]) else None
                            vz = round(float(rr["vol_z"]), 2) if pd.notna(rr["vol_z"]) else None
                            hi = round(float(rr["hi_52w"]), 2) if pd.notna(rr["hi_52w"]) else None
                            lo = round(float(rr["lo_52w"]), 2) if pd.notna(rr["lo_52w"]) else None
                            lines.append(f"- {rr['symbol']}: {pct}%  price={price}  vol_z={vz}  52wHi/Lo={hi}/{lo}")
                        resp = "\n".join(lines)

        with st.chat_message("assistant"):
            st.markdown(resp)
        st.session_state["messages"].append({"role": "assistant", "content": resp})
