# -*- coding: utf-8 -*-
# Google Custom Search helper (ASCII comments only)

import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# load .env from project root explicitly
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.environ.get("GOOGLE_API_KEY")
CSE_ID  = os.environ.get("GOOGLE_CSE_ID")

def news_search(query: str, num: int = 5, days: int = 7):
    # returns list of {title, link, snippet, source}
    if not API_KEY or not CSE_ID or not query:
        return []
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": max(1, min(num, 10)),
        "safe": "off",
        "lr": "lang_tr",
        "gl": "tr",
        "dateRestrict": f"d{max(1, days)}",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        items = js.get("items", []) or []
        out = []
        for it in items:
            out.append({
                "title": it.get("title"),
                "link": it.get("link"),
                "snippet": it.get("snippet"),
                "source": (it.get("displayLink") or "").lower(),
            })
        return out
    except Exception:
        return []
