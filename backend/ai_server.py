# /home/train/StockWizard_AI/backend/ai_server.py
# -*- coding: utf-8 -*-
# StockWizard_AI – Chat API (Gemini REST v1), DB-aware answers + fallback + news

import os, re, json, logging, traceback, sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import requests

# -----------------------------------------------------------------------------
# App + CORS
# -----------------------------------------------------------------------------
APP = FastAPI(title="StockWizard_AI Chat API")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://0.0.0.0:8501",
        "*",  # geliştirirken geniş; prod'da daralt
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Paths / ENV
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
load_dotenv(ROOT / ".env")

DEFAULT_DB = (ROOT / "data" / "movers.db").resolve()
DB = Path(os.environ.get("SW_DB_PATH", str(DEFAULT_DB))).resolve()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-8b")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID", "")  # haberler için

log = logging.getLogger("stockwizard_ai")
logging.basicConfig(level=logging.INFO)

def _mask(s: str, show: int = 6) -> str:
    return (s[:show] + f"...({len(s)}ch)") if s else "(empty)"

print(">> AI Server DB path:", DB)
print(">> GOOGLE_API_KEY:", _mask(GOOGLE_API_KEY))
print(">> GEMINI_MODEL:", GEMINI_MODEL)

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def _connect_db():
    return sqlite3.connect(str(DB))

def _table_exists(name: str) -> bool:
    try:
        with _connect_db() as c:
            r = c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        return r is not None
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Intent helpers
# -----------------------------------------------------------------------------
WINDOW_MAP = {
    "1g": "r_1d",
    "1h": "r_1w",
    "1a": "r_1m",
    "3a": "r_3m",
    "6a": "r_6m",
    "1y": "r_1y",
}
PERIOD_LABEL = {"r_1d":"1g","r_1w":"1h","r_1m":"1a","r_3m":"3a","r_6m":"6a","r_1y":"1y"}
FALLBACK_ORDER = ["r_1y","r_6m","r_3m","r_1m","r_1w","r_1d"]

def _normalize(t: str) -> str:
    t = t.lower()
    repl = {
        "bugün":"bugun", "gün":"gun", "günü":"gun",
        "yıllık":"yillik", "yıl":"yil",
        "düşen":"dusen", "yükselen":"yukselen", "artış":"artis",
        "değişmeyen":"degismeyen", "değismeyen":"degismeyen",
        "yatay":"yatay",
    }
    for k,v in repl.items():
        t = t.replace(k, v)
    return t

def infer_window(text: str) -> str:
    t = _normalize(text)
    if any(k in t for k in ["1y","yil","yillik","yılda","yilda"]): return "1y"
    if any(k in t for k in ["6a","6 ay"]):                          return "6a"
    if any(k in t for k in ["3a","3 ay","ceyrek","çeyrek"]):        return "3a"
    if any(k in t for k in ["1a","1 ay","aylik","ayın"]):           return "1a"
    if any(k in t for k in ["hafta","1w","week"]):                  return "1h"
    return "1g"

def infer_direction(text: str) -> str:
    t = _normalize(text)
    if any(k in t for k in ["degismeyen","sabit","yatay","flat","0 degisim","0 değişim"]):
        return "flat"
    return "down" if any(k in t for k in ["dusen","eksi","down","loser","kaybeden","azalan"]) else "up"

def extract_topn(text: str, default: int = 5) -> int:
    """
    Metindeki TÜM sayıları ve bazı Türkçe sayı kelimelerini toplar, en büyüğünü seçer.
    '1 yılda en çok düşen 9 hisse' -> 9
    """
    cand: List[int] = []
    # tüm rakamlar
    for n in re.findall(r"\d+", text):
        try:
            cand.append(int(n))
        except Exception:
            pass
    # bazı Türkçe sözcükler
    words = {
        "iki":2, "üç":3, "uc":3, "dört":4, " dort":4, "beş":5, "bes":5,
        "altı":6, "alti":6, "yedi":7, "sekiz":8, "dokuz":9, "on":10,
        "onbir":11, "on bir":11, "oniki":12, "on iki":12, "onüç":13, "on uc":13,
    }
    t = _normalize(text)
    for w, v in words.items():
        if w in t:
            cand.append(v)
    n = max(cand) if cand else default
    return max(1, min(200, n))

# -----------------------------------------------------------------------------
# Data queries
# -----------------------------------------------------------------------------
def _fetch_snapshot_top(col: str, direction: str, limit: int) -> List[Tuple[str, float, float]]:
    if not _table_exists("snapshot"):
        return []
    asc = "ASC" if direction == "down" else "DESC"
    q = f"""
        SELECT symbol, last_price, {col} AS ret
        FROM snapshot
        WHERE {col} IS NOT NULL
        ORDER BY ret {asc}
        LIMIT ?
    """
    with _connect_db() as c:
        return c.execute(q, (limit,)).fetchall()

def _fetch_zero_change(col: str, tol: float, limit: int) -> List[Tuple[str, float, float]]:
    """|ret| <= tol (ör. 0.0005 ~ 0.05%)"""
    if not _table_exists("snapshot"):
        return []
    q = f"""
        SELECT symbol, last_price, {col} AS ret
        FROM snapshot
        WHERE {col} IS NOT NULL AND ABS({col}) <= ?
        ORDER BY ABS({col}) ASC
        LIMIT ?
    """
    with _connect_db() as c:
        return c.execute(q, (tol, limit)).fetchall()

def _fetch_exact_top(col: str, direction: str, topn: int) -> List[Tuple[str, float, float, str]]:
    """İstenen kolondan doldur; eksikse FALLBACK_ORDER ile tamamla. (symbol uniq)."""
    order = [col] + [k for k in FALLBACK_ORDER if k != col]
    seen, out, need = set(), [], topn
    for c in order:
        if need <= 0:
            break
        rows = _fetch_snapshot_top(c, direction, max(need, 50))
        for s, px, r in rows:
            if s in seen:
                continue
            seen.add(s)
            out.append((s, px, r, c))
            need -= 1
            if need == 0:
                break
    return out[:topn]

def _fetch_symbol_snapshot(symbol: str) -> Optional[Dict[str, Any]]:
    if not _table_exists("snapshot"):
        return None
    with _connect_db() as c:
        row = c.execute("SELECT * FROM snapshot WHERE symbol=? LIMIT 1", (symbol,)).fetchone()
        if not row:
            return None
        cols = [r[1] for r in c.execute("PRAGMA table_info(snapshot)")]
        return dict(zip(cols, row))

def _fetch_symbol_returns(symbol: str) -> Optional[Dict[str, Any]]:
    if not _table_exists("returns"):
        return None
    with _connect_db() as c:
        row = c.execute("""
            SELECT asof, r_1d,r_1w,r_1m,r_3m,r_6m,r_1y
            FROM returns WHERE symbol=? ORDER BY asof DESC LIMIT 1
        """, (symbol,)).fetchone()
    if not row:
        return None
    asof, r1d,r1w,r1m,r3m,r6m,r1y = row
    return {"asof":asof,"r_1d":r1d,"r_1w":r1w,"r_1m":r1m,"r_3m":r3m,"r_6m":r6m,"r_1y":r1y}

def _fetch_symbol_names(symbol: str) -> Dict[str, Optional[str]]:
    if not _table_exists("symbols"):
        return {"short_name": None, "long_name": None}
    with _connect_db() as c:
        row = c.execute("SELECT short_name, long_name FROM symbols WHERE symbol=?",(symbol,)).fetchone()
    return {"short_name": (row[0] if row else None), "long_name": (row[1] if row else None)}

# -----------------------------------------------------------------------------
# News
# -----------------------------------------------------------------------------
def search_news(q: str, num: int = 5) -> List[Dict[str,str]]:
    """Google CSE ile basit haber araması."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    try:
        r = requests.get(url, params={
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": q,
            "num": min(10, num),
            "gl": "tr",
            "lr": "lang_tr",
        }, timeout=20)
        if not r.ok:
            return []
        j = r.json()
        items = []
        for it in j.get("items", []):
            items.append({"title": it.get("title",""), "link": it.get("link",""), "source": it.get("displayLink","")})
        return items
    except Exception:
        return []

# -----------------------------------------------------------------------------
# LLM prompt
# -----------------------------------------------------------------------------
SYS_INSTR = (
    "Sen finans asistanısın. Cevapları TÜRKÇE ver.\n"
    "- Yüzdeleri % işareti ve 2 ondalıkla yaz: 3.41%\n"
    "- Fiyatları ₺ ile yaz: ₺12.345\n"
    "- Veri yoksa uydurma; 'Bu dönem için verim yok.' de."
)

def db_context(symbol: Optional[str], user_text: str, topn: int = 10) -> str:
    win_key = infer_window(user_text)
    col     = WINDOW_MAP[win_key]
    direction = infer_direction(user_text)
    ctx = [f"[baglam] pencere={win_key} kolon={col} direction={direction}"]

    if symbol:
        snap = _fetch_symbol_snapshot(symbol)
        ret  = _fetch_symbol_returns(symbol)
        names = _fetch_symbol_names(symbol)
        if snap:
            def pct(v):
                try: return f"{float(v)*100:.2f}%"
                except: return "NA"
            def tl(v):
                try: return f"₺{float(v):.3f}"
                except: return "NA"
            ctx.append("[sembol_ozet]")
            ctx += [
                f"symbol={snap.get('symbol')}",
                f"short_name={names.get('short_name')}",
                f"asof={snap.get('asof')}",
                f"fiyat={tl(snap.get('last_price'))}",
                f"r_1g={pct(snap.get('r_1d'))}",
                f"r_1h={pct(snap.get('r_1w'))}",
                f"r_1a={pct(snap.get('r_1m'))}",
                f"r_3a={pct(snap.get('r_3m'))}",
                f"r_6a={pct(snap.get('r_6m'))}",
                f"r_1y={pct(snap.get('r_1y'))}",
                f"pct_from_hi={snap.get('pct_from_hi')}",
                f"pct_from_lo={snap.get('pct_from_lo')}",
            ]
        elif ret:
            ctx.append("[sembol_ozet] snapshot yok; returns mevcut.")
            ctx.append(json.dumps(ret, ensure_ascii=False))
        else:
            ctx.append(f"[sembol_ozet] {symbol} için kayıt yok.")

    # movers
    if direction == "flat":
        rows = _fetch_zero_change(col, 0.0005, topn)  # ~0.05%
        if rows:
            ctx.append(f"[movers_flat] topn={topn} kolon={col} tol=0.05%")
            for s, lp, r in rows:
                ctx.append(f"{s} | getiri={float(r)*100:.2f}% | fiyat=₺{float(lp):.3f}")
    else:
        rows = _fetch_exact_top(col, direction, topn)
        if rows:
            ctx.append(f"[movers] topn={topn} direction={direction} ana_kolon={col}")
            for s, lp, r, used in rows:
                ctx.append(f"{s} | getiri={float(r)*100:.2f}% | fiyat=₺{float(lp):.3f} | kolon={used}")

    ctx.append(f"[kullanici_soru] {user_text}")
    return "\n".join(ctx)

def build_prompt(user_msg: str, symbol: Optional[str]) -> str:
    n = extract_topn(user_msg, 10)
    ctx = db_context(symbol, user_msg, topn=n)
    return (
        f"{SYS_INSTR}\n"
        f"- Kullanıcı adet belirttiyse tam olarak {n} madde döndür.\n"
        f"- Sıralı madde işaretleri kullan.\n\n"
        f"{ctx}\n\n[soru]\n{user_msg}"
    )

def call_gemini(prompt: str) -> str:
    key = GOOGLE_API_KEY
    if not key:
        return "Sunucu: GOOGLE_API_KEY tanımsız."
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 768}
    }
    for model in [os.getenv("GEMINI_MODEL", GEMINI_MODEL), "gemini-1.5-flash-8b", "gemini-1.5-flash-001"]:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={key}"
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 404:
                continue
            if not r.ok:
                return f"Sunucu hata: {r.status_code} {r.text}"
            j = r.json()
            return (j["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except Exception as e:
            return f"Sunucu hata: {e}"
    return "Sunucu: Uygun Gemini modeli bulunamadı."

# -----------------------------------------------------------------------------
# Fast answers (deterministic)
# -----------------------------------------------------------------------------
def simple_answer(user_text: str, symbol: Optional[str]) -> Optional[str]:
    t = _normalize(user_text)
    tl = "₺"

    # selam
    if any(w in t for w in ["selam","merhaba","naber","nasilsin","nasılsın"]):
        return "Merhaba! Örn: **'bugün en çok düşen 10'**, **'1y en çok yükselen 7'**, **'THYAO.IS haber'**, **'SASA.IS 6a'**."

    # haber
    if "haber" in t or "news" in t:
        m = re.findall(r"\b[A-Z0-9]{2,}\.IS\b", user_text.upper())
        q = (m[0] if m else user_text) + " BIST"
        items = search_news(q, num=extract_topn(user_text, 5))
        if not items:
            return "Haber bulunamadı ya da anahtarlar eksik."
        lines = [f"**Haberler ({q}):**"]
        for it in items:
            title = it.get("title","").strip()
            link  = it.get("link","").strip()
            src   = it.get("source","").strip()
            lines.append(f"- [{title}]({link})  — _{src}_")
        return "\n".join(lines)

    # bugün kısayolu -> 1g
    if any(k in t for k in ["bugun","gun","1g"]):
        topn = extract_topn(t, 5)
        direction = infer_direction(t)
        if direction == "flat":
            rows = _fetch_zero_change("r_1d", 0.0005, topn)
            if not rows: return "Bugün için değişimi ~0 olan hisse yok gibi."
            head = f"**Bugün değişimi ~0 olan {topn} hisse:**"
            return "\n".join([head] + [f"- {s}: {r*100:.2f}% • {tl}{px:.3f}" for s, px, r in rows])

        rows = _fetch_exact_top("r_1d", direction, topn)
        if not rows: return "Bugün için veri bulamadım."
        head = f"**Bugün en çok {'düşen' if direction=='down' else 'yükselen'} {topn} hisse:**"
        return "\n".join([head] + [f"- {s}: {r*100:.2f}% • {tl}{px:.3f}" for s, px, r, used in rows])

    # genel “yükselen/düşen/flat” + istenen pencere
    if any(w in t for w in ["yukselen","yükselen","artan","dusen","düşen","loser","eksi","azalan","degismeyen","sabit","yatay"]):
        win_key = infer_window(t); col = WINDOW_MAP[win_key]; direction = infer_direction(t); topn = extract_topn(t,5)
        if direction == "flat":
            rows = _fetch_zero_change(col, 0.0005, topn)
            if not rows: return f"{win_key} için değişimi ~0 olan hisse bulunamadı."
            head = f"**{win_key} dönemi değişimi ~0 olan {topn} hisse:**"
            return "\n".join([head] + [f"- {s}: {r*100:.3f}% • {tl}{px:.3f}" for s, px, r in rows])

        rows = _fetch_exact_top(col, direction, topn)
        if not rows: return "Veri bulamadım."
        head = f"**{win_key} dönemi en çok {'düşen' if direction=='down' else 'yükselen'} {topn} hisse:**"
        return "\n".join([head] + [f"- {s}: {r*100:.2f}% • {tl}{px:.3f}" for s, px, r, used in rows])

    # sembol özeti
    if symbol:
        r = _fetch_symbol_returns(symbol)
        names = _fetch_symbol_names(symbol)
        nm = f" ({names['short_name']})" if names.get("short_name") else ""
        if r:
            return (f"**{symbol}{nm}** (asof {r['asof']}): "
                    f"1g {r['r_1d']*100:.2f}%, 1h {r['r_1w']*100:.2f}%, 1a {r['r_1m']*100:.2f}%, "
                    f"3a {r['r_3m']*100:.2f}%, 6a {r['r_6m']*100:.2f}%, 1y {r['r_1y']*100:.2f}%")

    return None  # LLM'e bırak

# -----------------------------------------------------------------------------
# Schemas & endpoints
# -----------------------------------------------------------------------------
class ChatIn(BaseModel):
    message: str
    symbol: Optional[str] = None

class ChatOut(BaseModel):
    reply: str

@APP.get("/health")
def health():
    try:
        with _connect_db() as c:
            t = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            tables = [x[0] for x in t]
            snap = c.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0] if "snapshot" in tables else 0
    except Exception as e:
        tables, snap = [f"ERR: {e}"], 0
    return {
        "db_path": str(DB),
        "tables": tables,
        "snapshot_rows": snap,
        "model": GEMINI_MODEL,
        "key_prefix": (GOOGLE_API_KEY[:6] + "***") if GOOGLE_API_KEY else "NONE",
    }

@APP.post("/chat", response_model=ChatOut)
def chat(inp: ChatIn):
    try:
        fast = simple_answer(inp.message, inp.symbol)
        if fast:
            return {"reply": fast}
        prompt = build_prompt(inp.message, inp.symbol)
        reply = call_gemini(prompt)
        return {"reply": reply}
    except Exception as e:
        log.error("CHAT ERROR: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"reply": f"Sunucu hata: {e}"})

if __name__ == "__main__":
    uvicorn.run(APP, host="0.0.0.0", port=8008)
