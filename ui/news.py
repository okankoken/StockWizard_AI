# -*- coding: utf-8 -*-
# Google Custom Search helper (ASCII-only comments)
# /home/train/StockWizard_AI/ui/news.py

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = (os.getenv("GOOGLE_CSE_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
CSE_ID  = (os.getenv("GOOGLE_CSE_ID") or "").strip()

def _call(params):
    url = "https://www.googleapis.com/customsearch/v1"
    return requests.get(url, params=params, timeout=20)

def news_search(query: str, num: int = 5, days: int = 7):
    """
    Return list of dicts: {title, link, snippet, source}
    On error, return [{"error": "..."}]
    """
    if not query or not API_KEY or not CSE_ID:
        return [{"error": f"API_KEY/CSE_ID missing. API_KEY_OK={bool(API_KEY)} CSE_ID_OK={bool(CSE_ID)}"}]

    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": max(1, min(int(num), 10)),
        "gl": "tr",
        "lr": "lang_tr",
        "dateRestrict": f"d{max(1, int(days))}",  # some engines may ignore this
    }

    try:
        r = _call(params)
        # Some engine configs reject dateRestrict; retry without it.
        if r.status_code == 400 and "dateRestrict" in (r.text or ""):
            params.pop("dateRestrict", None)
            r = _call(params)

        if not r.ok:
            return [{"error": f"{r.status_code} {r.text[:200]}"}]

        js = r.json()
        items = js.get("items", []) or []
        out = []
        for it in items:
            out.append({
                "title":   (it.get("title") or "").strip(),
                "link":    (it.get("link") or "").strip(),
                "snippet": (it.get("snippet") or "").strip(),
                "source":  (it.get("displayLink") or "").lower().strip(),
            })
        if not out:
            return [{"error": "No results."}]
        return out
    except Exception as e:
        return [{"error": f"Exception: {e}"}]
