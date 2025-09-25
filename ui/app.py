# -*- coding: utf-8 -*-
# StockWizard_AI - Streamlit UI (Dashboard + Explorer + Chat), reads snapshot table
# /home/train/StockWizard_AI/ui/app.py

import sys, os, re, sqlite3, unicodedata
from pathlib import Path
import pandas as pd
import streamlit as st
import requests
AI_SERVER_URL = os.environ.get("AI_SERVER_URL", "http://localhost:8008")


EMOJI = {
    "search": "\U0001F50D",
    "chart_up": "\U0001F4C8",
    "newspaper": "\U0001F4F0",
    "warning": "\u26A0",
    "check": "\u2705",
    "cross": "\u274C",
    "fire": "\U0001F525"
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB = DATA_DIR / "movers.db"

def _connect_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DB))

def table_exists(name: str) -> bool:
    try:
        with _connect_db() as conn:
            cur = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,))
            return cur.fetchone() is not None
    except Exception:
        return False

# news search (safe fallback)
try:
    from news import news_search
except Exception:
    def news_search(*args, **kwargs):
        return []

# Türkçe baslik + ince tema dokunusu
st.set_page_config(page_title="StockWizard_AI - BIST", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
:root {
  --app-font: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans', 'DejaVu Sans', Arial, sans-serif;
}
html, body, [class*="st-"], * { font-family: var(--app-font) !important; }
</style>
""", unsafe_allow_html=True)



st.markdown("""
<style>
/* Baslik için hafif degrade */
h1, .st-emotion-cache-10trblm, .st-emotion-cache-1v0mbdj {
  background: linear-gradient(90deg,#7c3aed,#06b6d4);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
/* Metrigi güçlendir */
div[data-testid="stMetricValue"] { font-weight: 800; }
/* Kenar bosluklari biraz toparla */
.block-container { padding-top: 1rem; }
/* Tab basliklarini belirginlestir */
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)



st.title("StockWizard_AI")
st.caption("Sadece gerçeklesmis hareketler. Bu bir yatirim tavsiyesi degildir.")


# optional health box
with st.expander("DB Health", expanded=False):
    st.write("DB path:", str(DB))
    st.write("DB exists:", DB.exists())
    try:
        with _connect_db() as c:
            tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        st.write("Tables:", tables)
        st.success("SQLite open OK")
    except Exception as e:
        st.error(f"SQLite open error: {e}")

# ---- Data loaders ----
@st.cache_data(ttl=300)
def load_snapshot():
    if not DB.exists() or not table_exists("snapshot"):
        return pd.DataFrame()
    with _connect_db() as conn:
        return pd.read_sql_query("SELECT * FROM snapshot", conn, parse_dates=["asof"])

@st.cache_data(ttl=300)
def load_symbols():
    if not DB.exists() or not table_exists("prices"):
        return []
    with _connect_db() as conn:
        df = pd.read_sql_query("SELECT DISTINCT symbol FROM prices ORDER BY symbol", conn)
    return df["symbol"].dropna().astype(str).tolist()

@st.cache_data(ttl=300)
def load_prices(symbol: str):
    if not DB.exists() or not table_exists("prices"):
        return pd.DataFrame()
    with _connect_db() as conn:
        q = "SELECT date, adj_close, volume FROM prices WHERE symbol=? ORDER BY date"
        return pd.read_sql_query(q, conn, params=[symbol], parse_dates=["date"])


# ===== Pano (Dashboard) =====
tab_dash, tab_explorer, tab_chat = st.tabs(["Hisseler", "Grafik", "Chatbot"])

with tab_dash:
    if not DB.exists():
        st.warning("Veritabani bulunamadi. Önce veri çekme scriptlerini çalistirin.")
        st.stop()
    if not table_exists("snapshot"):
        st.warning("'snapshot' tablosu yok. Komut: python -m fetch.compute")
        st.stop()

    st.subheader("Piyasa Özeti")

    snap = load_snapshot()
    if snap.empty:
        st.info("Snapshot bos. Fetch scriptlerini çalistirin.")
        st.stop()

    # sayisal kolonlari temizle
    num_cols = ["r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","vol_z","last_price","hi_52w","lo_52w","pct_from_hi","pct_from_lo"]
    for c in num_cols:
        snap[c] = pd.to_numeric(snap[c], errors="coerce")

    # Siralama (Türkçe etiketler -> kolon eslemesi)
    sort_map = {
        "1G Getiri %": "r_1d",
        "1H Getiri %": "r_1w",
        "1A Getiri %": "r_1m",
        "3A Getiri %": "r_3m",
        "6A Getiri %": "r_6m",
        "1Y Getiri %": "r_1y",
        "Zirveye Mesafe %": "pct_from_hi",
        "Dipten Uzaklik %": "pct_from_lo",
        "Hacim Z-Skoru": "vol_z",
        "Son Fiyat ?": "last_price",
    }
    sort_label = st.sidebar.selectbox("Sirala", list(sort_map.keys()), index=1)
    sort_by = sort_map[sort_label]
    ascending = st.sidebar.checkbox("Artan sirala", value=False)
    topn = st.sidebar.slider("Kaç satir gösterilsin?", 1, 600, 30)

    view = snap.sort_values(sort_by, ascending=ascending).head(topn).copy()

    # Yüzdeleri %'ye çevir
    for c in ["r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","pct_from_hi","pct_from_lo"]:
        view[c] = (view[c].astype(float) * 100.0).round(2)
    view["vol_z"] = view["vol_z"].round(2)
    view["last_price"] = view["last_price"].round(2)

    # Türkçe kolon basliklari + formatlar
    from streamlit import column_config as cc
    st.dataframe(
        view[[
            "symbol","last_price","r_1d","r_1w","r_1m","r_3m","r_6m","r_1y","vol_z",
            "hi_52w","lo_52w","pct_from_hi","pct_from_lo","asof"
        ]],
        use_container_width=True,
        column_config={
            "symbol": cc.TextColumn("Sembol"),
            "last_price": cc.NumberColumn("Son Fiyat", format="\u20ba %.3f"),
            "r_1d": cc.NumberColumn("1G Getiri (%)", format="%.2f%%"),
            "r_1w": cc.NumberColumn("1H Getiri (%)", format="%.2f%%"),
            "r_1m": cc.NumberColumn("1A Getiri (%)", format="%.2f%%"),
            "r_3m": cc.NumberColumn("3A Getiri (%)", format="%.2f%%"),
            "r_6m": cc.NumberColumn("6A Getiri (%)", format="%.2f%%"),
            "r_1y": cc.NumberColumn("1Y Getiri (%)", format="%.2f%%"),
            "vol_z": cc.NumberColumn("Hacim Z-Skoru"),
            "hi_52w": cc.NumberColumn("52H Zirve", format="\u20ba %.3f"),
            "lo_52w": cc.NumberColumn("52H Dip", format="\u20ba %.3f"),
            "pct_from_hi": cc.NumberColumn("Zirveye Mesafe (%)", format="%.2f%%"),
            "pct_from_lo": cc.NumberColumn("Dipten Uzaklik (%)", format="%.2f%%"),
            "asof": cc.DatetimeColumn("Tarih"),
        }
    )


# ===== Kesif (Explorer) =====
with tab_explorer:
    if not DB.exists():
        st.warning("Veritabani yok. Önce veri çekin.")
        st.stop()
    if not table_exists("prices"):
        st.warning("'prices' tablosu yok. Komut: python -m fetch.prices")
        st.stop()

    st.subheader(f"{EMOJI['search']} Sembol Kesfi")

    symbols = load_symbols()
    if not symbols:
        st.info("Veritabaninda sembol bulunamadi.")
        st.stop()

    c1, c2, c3 = st.columns([2,2,1])
    with c1:
        symbol = st.selectbox("Sembol seçin", options=symbols, index=0)
    with c2:
        window = st.selectbox("Zaman araligi", ["1A","3A","6A","1Y","3Y","MAKS"], index=2)
    with c3:
        show_volume = st.checkbox("Hacmi göster", value=True)

    df = load_prices(symbol)
    if df.empty:
        st.warning(f"{symbol} için fiyat bulunamadi.")
        st.stop()

    df = df.sort_values("date")
    max_date = df["date"].max()

    def cut(df_in, months=None, years=None):
        if months:
            start = max_date - pd.DateOffset(months=months)
        elif years:
            start = max_date - pd.DateOffset(years=years)
        else:
            start = df_in["date"].min()
        return df_in[df_in["date"] >= start].copy()

    if window == "1A":
        wdf = cut(df, months=1)
    elif window == "3A":
        wdf = cut(df, months=3)
    elif window == "6A":
        wdf = cut(df, months=6)
    elif window == "1Y":
        wdf = cut(df, years=1)
    elif window == "3Y":
        wdf = cut(df, years=3)
    else:
        wdf = df.copy()

    lc, rc = st.columns([3,1])
    with lc:
        st.markdown(f"**{symbol} – Düzeltilmis Kapanis (Adj Close)**")
        st.line_chart(wdf.set_index("date")["adj_close"])
        if show_volume:
            st.markdown("**Hacim**")
            st.bar_chart(wdf.set_index("date")["volume"])

    with rc:
        last_price = float(wdf["adj_close"].iloc[-1])
        first_price = float(wdf["adj_close"].iloc[0]) if len(wdf) > 0 else None
        chg = (last_price / first_price - 1.0) * 100.0 if first_price else None
    
        st.metric(label="Se\u00e7ili aral\u0131k getirisi", value=(f"{chg:.2f}%" if chg is not None else "N/A"))
        st.metric(label="Son fiyat (\u20ba)", value=f"\u20ba {last_price:.3f}")  # <-- TL simgesi
        st.metric(label="Kay\u0131t say\u0131s\u0131", value=len(wdf))

    with st.container():
        if st.button(f"{EMOJI['newspaper']} Bu sembolde haber ara"):
            with st.spinner("Haberler getiriliyor..."):
                try:
                    items = news_search(symbol, num=5) or []
                except Exception as e:
                    st.warning(f"Haber hatasi: {e}")
                    items = []
            if items:
                for it in items:
                    title = (it.get("title") or "").strip()
                    link  = (it.get("link") or "").strip()
                    src   = (it.get("source") or "").strip()
                    if link.startswith("http"):
                        st.markdown(f"- **[{title}]({link})**  \n  _{src}_")
                    else:
                        st.markdown(f"- **{title}**  \n  _{src}_")
            else:
                st.info("Haber bulunamadi (veya API anahtari yok).")


# ===== Chat =====
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Merhaba \U0001F44B  Sembol ve sorunu yaz.\nOrn: 'THYAO.IS bu hafta ne yaptı?', 'en çok yükselenler', 'SASA.IS haber'."
            }
        ]

    # Geçmişi göster
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Sorunuzu yaz\u0131n...")
    if user_msg:
        st.session_state["messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Basit sembol çıkarımı (opsiyonel)
        import re
        toks = re.split(r"\s+", user_msg.upper())
        sym = next((t for t in toks if t.endswith(".IS")), None)

        # Backend'e gönder
        try:
            with st.spinner("Yan\u0131t haz\u0131rlan\u0131yor..."):
                r = requests.post(
                    f"{AI_SERVER_URL}/chat",
                    json={"message": user_msg, "symbol": sym},
                    timeout=30
                )
                if r.ok:
                    reply = r.json().get("reply", "")
                else:
                    reply = f"Sunucu hatas\u0131: {r.status_code} {r.text}"
        except Exception as e:
            reply = f"Ba\u011flant\u0131 hatas\u0131: {e}"

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        
        
        