# /home/train/StockWizard_AI/backend/ai_server.py
# -*- coding: utf-8 -*-

# LLM chat proxy: keeps API keys server-side, returns model reply in Turkish

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import sqlite3
from pathlib import Path
import datetime as dt
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
import traceback, logging
import re
from datetime import datetime

APP = FastAPI(title="StockWizard_AI Chat API")

# CORS middleware'i APP taniminin HEMEN ALTINA ekle:
APP.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://0.0.0.0:8501",
        # gerekirse dis IP/alan adin:
        # "http://YOUR_HOST_OR_IP:8501",
        "*",  # gelistirirken serbest birakmak için; prod'da daralt
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HERE = Path(__file__).resolve().parent           # .../StockWizard_AI/backend
ROOT = HERE.parent                               # .../StockWizard_AI
DB = (ROOT / "data" / "movers.db").resolve()     # .../StockWizard_AI/data/movers.db

load_dotenv(ROOT / ".env")

# (opsiyonel) Ortam degiskeni ile override imkani:
DB = Path(os.environ.get("SW_DB_PATH", str(DB))).resolve()

print(">> AI Server DB path:", DB)               # baslarken path'i gör

# ====== Config ======
MODEL_BACKEND = os.getenv("LLM_BACKEND", "gemini")  # "gemini" or "openai"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

class ChatIn(BaseModel):
    message: str
    symbol: Optional[str] = None

class ChatOut(BaseModel):
    reply: str

def db_context(symbol: str | None, user_text: str, topn: int = 10) -> str:
    # sembol özeti + genel tepe 5 düşen/yükselen (1g)
    import sqlite3
    ctx = []

    with sqlite3.connect(str(DB)) as c:
        if symbol:
            r = c.execute("""
                SELECT asof, r_1d,r_1w,r_1m,r_3m,r_6m,r_1y
                FROM returns WHERE symbol=? ORDER BY asof DESC LIMIT 1
            """, (symbol,)).fetchone()
            if r:
                asof, *vals = r
                ctx.append(f"Sembol {symbol} son veriler (asof {asof}): "
                           f"1g={vals[0]:.6f}, 1h={vals[1]:.6f}, 1a={vals[2]:.6f}, "
                           f"3a={vals[3]:.6f}, 6a={vals[4]:.6f}, 1y={vals[5]:.6f}.")

        top_up  = _fetch_snapshot_top("r_1d", "up", 5)
        top_dn  = _fetch_snapshot_top("r_1d", "down", 5)
        if top_up:
            s = ", ".join([f"{sym}:{r:.4f}" for sym,_,r in top_up])
            ctx.append(f"Bugün en çok yükselen 5: {s}.")
        if top_dn:
            s = ", ".join([f"{sym}:{r:.4f}" for sym,_,r in top_dn])
            ctx.append(f"Bugün en çok düşen 5: {s}.")

    ctx.append(f"Kullanıcı sorusu: {user_text}")
    return "\n".join(ctx)

def build_prompt(user_text: str, symbol: str | None) -> str:
    return SYS_INSTR + "\n\n" + db_context(symbol, user_text, topn=10)

def build_prompt(user_msg: str, symbol: Optional[str]) -> str:
    ctx = db_context(symbol, user_msg, topn=10)
    return f"""Sen finans asistanısın. Cevapları TÜRKÇE ver.
Elindeki tablo alanları: symbol, last_price (TL), r_1d, r_1w, r_1m, r_3m, r_6m, r_1y, vol_z, hi_52w, lo_52w, pct_from_hi, pct_from_lo, asof.
Kurallar:
- Yüzdeleri % işaretiyle ve 2 ondalık basamakla yaz: 3.41%
- Fiyatları '₺' simgesiyle yaz: ₺12.345
- Rakam uydurma. Dönem tabloda yoksa: 'Bu dönem için verim yok' de.

{ctx}

[soru]
{user_msg}
"""

def call_gemini(prompt: str) -> str:
    import os, requests, json

    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        return "Sunucu: GOOGLE_API_KEY tanimsiz."

    # Önce env'deki modeli dene; olmazsa alternatiflere düs
    first = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")
    candidates = [first, "gemini-1.5-flash-8b", "gemini-1.5-flash-001"]

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for model in candidates:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={key}"
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 404:
                # Bu sürüme erisim yok  siradakine dene
                continue
            if not resp.ok:
                return f"Sunucu hata: {resp.status_code} {resp.text}"
            data = resp.json()
            return (data["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except Exception as e:
            return f"Sunucu hata: {e}"

    return "Sunucu: Uygun Gemini modeli bulunamadi."



def call_openai(prompt: str) -> str:
    # pip install openai
    from openai import OpenAI
    if not OPENAI_API_KEY:
        return "Sunucu: OPENAI_API_KEY tan\u0131ms\u0131z."
    client = OpenAI(api_key=OPENAI_API_KEY)
    chat = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":"You are a helpful Turkish market assistant."},
            {"role":"user","content":prompt},
        ],
        temperature=0.3,
    )
    return chat.choices[0].message.content.strip()

log = logging.getLogger("stockwizard_ai")
logging.basicConfig(level=logging.INFO)

class ChatIn(BaseModel):
    message: str
    symbol: Optional[str] = None

@APP.get("/health")
def health():
    try:
        with sqlite3.connect(str(DB)) as c:
            t = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            tables = [x[0] for x in t]
            snap = c.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0] if "snapshot" in tables else 0
    except Exception as e:
        tables, snap = [f"ERR: {e}"], 0
    return {
        "db_path": str(DB),
        "tables": tables,
        "snapshot_rows": snap,
        "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b"),
        "key_prefix": (os.getenv("GOOGLE_API_KEY","")[:6] + "***") if os.getenv("GOOGLE_API_KEY") else "NONE",
    }




@APP.post("/chat")
def chat(inp: ChatIn):
    try:
        # 1) hızlı yol
        fast = simple_answer(inp.message, inp.symbol)
        if fast:
            return {"reply": fast}

        # 2) LLM prompt’u (Türkçe, net talimat)
        prompt = build_prompt(inp.message, inp.symbol)   # mevcut fonksiyonun
        reply = call_gemini(prompt)
        return {"reply": reply}
    except Exception as e:
        import traceback, logging
        logging.getLogger("stockwizard_ai").error("CHAT ERROR: %s\n%s", e, traceback.format_exc())
        return {"reply": f"Sunucu hata: {e}"}


if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=8008)
    
    
    
def mask(s, show=6):
    return (s[:show] + "" + str(len(s)) + "ch") if s else "(empty)"

print(">> LLM_BACKEND:", MODEL_BACKEND)
print(">> GOOGLE_API_KEY:", mask(GOOGLE_API_KEY))
print(">> GEMINI_MODEL:", GEMINI_MODEL)

# --- DB baglam yardimcilari ---
import sqlite3
from typing import Optional

def _connect_db():
    return sqlite3.connect(str(DB))

WINDOW_MAP = {
    "1g": "r_1d",
    "1h": "r_1w",
    "1a": "r_1m",
    "3a": "r_3m",
    "6a": "r_6m",
    "1y": "r_1y",
}

def infer_window(text: str) -> str:
    t = text.lower()
    if "1y" in t or "1 yil" in t or "yillik" in t or "yillik" in t: return "1y"
    if "6a" in t or "6 ay" in t: return "6a"
    if "3a" in t or "3 ay" in t or "çeyrek" in t or "ceyrek" in t: return "3a"
    if "1a" in t or "1 ay" in t or "aylik" in t or "aylik" in t: return "1a"
    if "hafta" in t or "1w" in t or "week" in t: return "1h"
    return "1g"

def db_context(symbol: Optional[str], user_text: str, topn: int = 10) -> str:
    """
    Kucuk ve hedefli baglam: (a) sembol ozetleri, (b) en cok artan/azalan listeleri.
    """
    win_key = infer_window(user_text)           # örn: "1y"
    col = WINDOW_MAP[win_key]                   # örn: "r_1y"

    ctx_lines = [f"[baglam] zaman_penceresi={win_key} (kolon={col})"]

    with _connect_db() as conn:
        # 1) Sembol özeti
        if symbol:
            q = "SELECT * FROM snapshot WHERE symbol=?"
            row = conn.execute(q, (symbol,)).fetchone()
            if row:
                cols = [c[0] for c in conn.execute("PRAGMA table_info(snapshot)")]
                rec = dict(zip(cols, row))
                ctx_lines.append("[sembol_ozet]")
                # TL, % ve temel alanlar:
                def pct(v): 
                    try: return f"{float(v)*100:.2f}%"
                    except: return "NA"
                def tl(v):
                    try: return f"{float(v):.3f} ?"
                    except: return "NA"

                ctx_lines += [
                    f"symbol={rec.get('symbol')}",
                    f"asof={rec.get('asof')}",
                    f"fiyat={tl(rec.get('last_price'))}",
                    f"r_1g={pct(rec.get('r_1d'))}",
                    f"r_1h={pct(rec.get('r_1w'))}",
                    f"r_1a={pct(rec.get('r_1m'))}",
                    f"r_3a={pct(rec.get('r_3m'))}",
                    f"r_6a={pct(rec.get('r_6m'))}",
                    f"r_1y={pct(rec.get('r_1y'))}",
                    f"vol_z={rec.get('vol_z')}",
                    f"hi_52w={rec.get('hi_52w')}",
                    f"lo_52w={rec.get('lo_52w')}",
                    f"pct_from_hi={(rec.get('pct_from_hi'))}",
                    f"pct_from_lo={(rec.get('pct_from_lo'))}",
                ]
            else:
                ctx_lines.append(f"[sembol_ozet] {symbol} icin kayit yok.")

        # 2) Movers listeleri (top gainers/losers)  snapshot tablosundan
        # up/down niyetini kaba belirle
        txt = user_text.lower()
        direction = "down" if any(k in txt for k in ["dus", "düs", "eksi", "down", "loser", "kaybeden"]) else "up"

        # seçili kolon yoksa düsme
        cols = [r[1] for r in conn.execute("PRAGMA table_info(snapshot)")]
        if col in cols:
            asc = "ASC" if direction == "down" else "DESC"
            q = f"""
                SELECT symbol, last_price, {col} AS ret, vol_z, hi_52w, lo_52w
                FROM snapshot
                WHERE {col} IS NOT NULL
                ORDER BY ret {asc}
                LIMIT ?
            """
            rows = conn.execute(q, (topn,)).fetchall()
            if rows:
                ctx_lines.append(f"[movers] direction={direction} topn={topn} kolon={col}")
                for s, lp, r, vz, hi, lo in rows:
                    try:
                        p_ret = f"{float(r)*100:.2f}%"
                    except:
                        p_ret = "NA"
                    try:
                        lp_tl = f"{float(lp):.3f} ?"
                    except:
                        lp_tl = "NA"
                    ctx_lines.append(f"{s} | getiri={p_ret} | fiyat={lp_tl} | vol_z={vz} | 52wHi/Lo={hi}/{lo}")

    return "\n".join(ctx_lines)


def _fetch_snapshot_top(col: str, direction: str, limit: int = 5) -> list[tuple]:
    """snapshot tablosundan col'a göre en iyi/kötü 'limit' satırı getirir."""
    asc = 1 if direction == "down" else 0
    q = f"""
        SELECT symbol, last_price, {col}
        FROM snapshot
        WHERE {col} IS NOT NULL
        ORDER BY {col} {'ASC' if asc else 'DESC'}
        LIMIT ?
    """
    with sqlite3.connect(str(DB)) as c:
        return c.execute(q, (limit,)).fetchall()

def simple_answer(user_text: str, symbol: str | None) -> str | None:
    """
    Çok sık sorulanları anında DB'den döndür.
    Döndüyse string (cevap), döndürmediyse None (LLM'e bırak).
    """
    t = user_text.lower()

    # 0) Selam / smalltalk
    if any(w in t for w in ["selam", "merhaba", "nasılsın", "naber", "iyi misin"]):
        return "Merhaba! Yardımcı olabileceğim bir hisse ya da dönem sorusu var mı? Örn: **'bugün en çok yükselen 5 hisse'** ya da **'THYAO.IS 1a'**."

    # 1) "bugün/1g en çok yükselen/düşen"
    if "bugün" in t or "1g" in t or "1 g" in t or "gun" in t or "gün" in t:
        direction = "down" if any(w in t for w in ["düşen", "dusen", "düşük", "kaybeden", "loser"]) else "up"
        topn_match = re.search(r"(\d+)", t)
        topn = int(topn_match.group(1)) if topn_match else 5
        rows = _fetch_snapshot_top("r_1d", direction, topn)
        if not rows:
            return "Bugün için veri bulamadım."
        lines = [f"**Bugün en çok {'düşen' if direction=='down' else 'yükselen'} {topn} hisse:**"]
        for sym, px, r in rows:
            lines.append(f"- {sym}: {r*100:.2f}%  • ₺{px:.3f}")
        return "\n".join(lines)

    # 2) Genel “en çok yükselen/düşen” (pencere tespiti)
    def pick_col(tt: str) -> str:
        if "1y" in tt or "yıl" in tt or "yillik" in tt: return "r_1y"
        if "6a" in tt or "6 ay" in tt: return "r_6m"
        if "3a" in tt or "3 ay" in tt: return "r_3m"
        if "1a" in tt or "1 ay" in tt or "aylık" in tt: return "r_1m"
        if "hafta" in tt or "1w" in tt: return "r_1w"
        return "r_1d"

    if any(w in t for w in ["yükselen", "yukselen", "gainer", "artış", "artan", "loser", "düşen", "dusen", "eksi"]):
        col = pick_col(t)
        direction = "down" if any(w in t for w in ["düşen", "dusen", "loser", "eksi"]) else "up"
        topn_match = re.search(r"(\d+)", t)
        topn = int(topn_match.group(1)) if topn_match else 5
        rows = _fetch_snapshot_top(col, direction, topn)
        if not rows:
            return "Veri bulamadım."
        period_map = {"r_1d":"1g","r_1w":"1h","r_1m":"1a","r_3m":"3a","r_6m":"6a","r_1y":"1y"}
        lines = [f"**{period_map.get(col,col)} dönemi en çok {'düşen' if direction=='down' else 'yükselen'} {topn} hisse:**"]
        for sym, px, r in rows:
            lines.append(f"- {sym}: {r*100:.2f}%  • ₺{px:.3f}")
        return "\n".join(lines)

    # 3) Belirli sembol ve dönem
    if symbol:
        col = pick_col(t)
        with sqlite3.connect(str(DB)) as c:
            row = c.execute(f"""
                SELECT asof, r_1d, r_1w, r_1m, r_3m, r_6m, r_1y
                FROM returns WHERE symbol=? ORDER BY asof DESC LIMIT 1
            """, (symbol,)).fetchone()
        if row:
            asof, r1d,r1w,r1m,r3m,r6m,r1y = row
            return (f"**{symbol}** (asof {asof}): "
                    f"1g {r1d*100:.2f}%, 1h {r1w*100:.2f}%, 1a {r1m*100:.2f}%, "
                    f"3a {r3m*100:.2f}%, 6a {r6m*100:.2f}%, 1y {r1y*100:.2f}%")

    # 4) Hicbiri uymadıysa LLM’e bırak
    return None