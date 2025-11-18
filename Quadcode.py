from __future__ import annotations
import os
import sys
import time
import random
import json
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import numpy as np
import feedparser
from dateutil import parser as dateparser
import yfinance as yf

# ============ Configuration ============
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-ai/deepseek-v3.1"
REQUEST_TIMEOUT = 60
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# ============ API Key ============
def get_nvidia_key() -> str:           #This function is used to get the API key from build.nvidia
    key = os.getenv("NVIDIA_API_KEY")
    if key:
        return key
    key = input("Enter NVIDIA API key: ").strip()
    if not key:
        raise RuntimeError("NVIDIA API key missing.")
    return key

# ============ Backoff for HTTP errors ============
def backoff_post(url, headers, payload, max_retries=6):
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            wait = (2**attempt) + random.random()
            print(f"[Network error] Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return r

        if r.status_code in (429, 500, 502, 503, 504):
            wait = (2**attempt) + random.random()
            print(f"[{r.status_code}] Retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError("Max retries exceeded calling NVIDIA API.")

# ============ Data Fetching ============
def fetch_yfinance_history(symbol, days=180):             #This fetches all the data for the past 6 months which will be used to analyse the past trends
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)

    ticker = yf.Ticker(symbol)
    hist = ticker.history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
    )

    if hist.empty:
        raise RuntimeError(f"No yfinance data for {symbol}")

    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    hist.set_index("Date", inplace=True)

    return pd.DataFrame({
        "o": hist["Open"],
        "h": hist["High"],
        "l": hist["Low"],
        "c": hist["Close"],
        "v": hist["Volume"].fillna(0)
    })

def fetch_yfinance_quote(symbol):  #Fetches the most recent closing price of a stock using the yfinance library.
    try:
        last = yf.Ticker(symbol).history(period="2d")
        close = float(last["Close"].iloc[-1])
        return {"last_close": close}
    except:
        return {"last_close": None}

def fetch_google_news_rss(query, max_items=8):        # Fetches the news data, from google news, about the company stock selected by the user to predict fluctuations in the stock prices
    url = GOOGLE_NEWS_RSS.format(query=requests.utils.quote(query))
    feed = feedparser.parse(url)

    out = []
    for e in feed.entries[:max_items]:
        try:
            ts = dateparser.parse(e.published).isoformat()
        except:
            ts = e.get("published", "")
        out.append({
            "title": e.get("title", ""),
            "link": e.get("link", ""),
            "published": ts,
            "summary": e.get("summary", "")
        })
    return out

# ============ Indicators ============
def compute_technicals(df):
    df2 = df.copy()
    df2["returns"] = df2["c"].pct_change()

    out = {}
    out["last_close"] = float(df2["c"].iloc[-1])
    out["returns_1d"] = float(df2["returns"].iloc[-1])
    out["returns_7d"] = float(df2["c"].iloc[-1] / df2["c"].iloc[-8] - 1) if len(df2) > 8 else None
    out["vol_30d"] = float(df2["returns"].tail(30).std() * np.sqrt(252))

    df2["ma20"] = df2["c"].rolling(20).mean()
    df2["ma50"] = df2["c"].rolling(50).mean()
    out["ma20"] = None if pd.isna(df2["ma20"].iloc[-1]) else float(df2["ma20"].iloc[-1])
    out["ma50"] = None if pd.isna(df2["ma50"].iloc[-1]) else float(df2["ma50"].iloc[-1])

    out["ma20_vs_price"] = ("above" if out["ma20"] and out["last_close"] > out["ma20"] else "below") if out["ma20"] else "unknown"
    out["ma20_vs_ma50"] = ("above" if (out["ma20"] and out["ma50"] and out["ma20"] > out["ma50"]) else "below") if (out["ma20"] and out["ma50"]) else "unknown"

    return out

def summarize_price_history(df):
    start = df.index[0]
    end = df.index[-1]
    return {
        "data_start": start.isoformat(),
        "data_end": end.isoformat(),
        "period_days": (end - start).days,
        "min": float(df["c"].min()),
        "max": float(df["c"].max()),
        "mean": float(df["c"].mean()),
        "std": float(df["c"].std()),
        "last": float(df["c"].iloc[-1]),
        "pct_change_period": float(df["c"].iloc[-1] / df["c"].iloc[0] - 1)
    }

# ============ Prompt Construction ============
def build_prompt(symbol, name_hint, price_summary, techs, news_items):
    p = []

    # --- Core identity / metadata ---
    p.append(f"Ticker: {symbol}")
    p.append(f"Company: {name_hint or 'Unknown'}")

    # Historical data range
    p.append(
        f"Historical Data Range (UTC): {price_summary['data_start']} to {price_summary['data_end']} "
        f"({price_summary['period_days']} days)"
    )

    # --- Price summary ---
    p.append("\nPRICE SUMMARY:")
    p.append(f"- Last close: {price_summary['last']:.2f}")
    p.append(f"- Low:  {price_summary['min']:.2f}")
    p.append(f"- High: {price_summary['max']:.2f}")
    p.append(f"- Mean: {price_summary['mean']:.2f}")
    p.append(f"- Std deviation: {price_summary['std']:.2f}")
    p.append(f"- % change over period: {price_summary['pct_change_period']*100:.2f}%")

    # --- Technical Indicators ---
    p.append("\nTECHNICAL INDICATORS:")
    p.append(f"- MA20: {techs.get('ma20')}")
    p.append(f"- MA50: {techs.get('ma50')}")
    p.append(f"- MA20 vs Price: {techs.get('ma20_vs_price')}")
    p.append(f"- MA20 vs MA50: {techs.get('ma20_vs_ma50')}")
    p.append(f"- 1-day return: {techs.get('returns_1d')*100:.2f}%")
    p.append(f"- Annualized vol (30d): {techs.get('vol_30d'):.5f}")

    # --- News ---
    p.append("\nRECENT NEWS HEADLINES:")       #Check if the news about the company selected by the user is present or not
    if not news_items:
        p.append("- None found")
    else:
        for n in news_items:
            p.append(f"- [{n['published']}] {n['title']}")

    # --- Instructions for the model ---
    p.append("\nINSTRUCTIONS TO THE ASSISTANT:")
    p.append(
        "You are a financial research assistant. Use the historical data, technical indicators, "
        "and news sentiment to produce a disciplined investment assessment."
    )

    p.append(
        "Your final output MUST include exactly the following components in this exact order:\n"
        "1. Recommendation: BUY / HOLD / SELL"
    )

    # STRICT REQUIRED FORMAT
    p.append("2. Aggression Score (0-100): <integer>")

    p.append("\nAGGRESSION SCORE RULES:")
    p.append("0   = Buy aggressively")
    p.append("100 = Sell aggressively")
    p.append("Do not add any text on the same line as the integer. Example:")
    p.append("Aggression Score (0-100): 20")

    p.append(
        "If you are uncertain or the data is mixed, output:\n"
        "Aggression Score (0-100): 50"
    )

    p.append("\nAFTER these two required lines, include:")
    p.append("- One quantitative reason based on price or technicals")
    p.append("- One qualitative reason from recent news sentiment")
    p.append("- One key risk to the thesis")

    p.append("\nDo NOT output JSON. Do NOT output code. Follow the exact format.")

    return "\n".join(p)


# ============ NVIDIA call ============
def call_nvidia_chat(prompt, model=DEFAULT_MODEL):
    key = get_nvidia_key()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert financial analyst."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 900,
        "stream": False
    }

    r = backoff_post(BASE_URL, headers, payload)
    j = r.json()
    return j["choices"][0]["message"]["content"], j

# ============ NEW PARSER: aggression scale ============
def parse_decision(text):
    """
    Robust extraction:
      - recommendation: "BUY"|"HOLD"|"SELL" or None
      - aggression_score: integer 0..100 or None

    Strategy:
      1. Strictly match the exact required line "Aggression Score (0-100): <int>" (multiline).
      2. Try other numeric patterns (x/100, x out of 100, x%, bare integer).
      3. If still nothing, use explicit-word heuristics (SELL->90, BUY->10, HOLD->50).
      4. Do NOT return 0 as a silent default; return None if we truly can't infer a value.
    """
    if not text:
        return {"recommendation": None, "aggression_score": None}

    # Keep original text
    t = text

    # 1) Recommendation (word-boundary)
    rec_m = re.search(r"\b(buy|hold|sell)\b", t, flags=re.IGNORECASE)
    recommendation = rec_m.group(1).upper() if rec_m else None

    # Helper: clamp to 0..100
    def clamp_int(v):
        try:
            vi = int(round(float(v)))
        except Exception:
            return None
        if vi < 0:
            vi = 0
        if vi > 100:
            vi = 100
        return vi

    score = None

    # 2) Strict exact-line match (multiline). This is highest priority.
    m = re.search(r"(?im)^[ \t]*Aggression\s+Score\s*\(0-100\)\s*:\s*([0-9]{1,3})[ \t]*$", t, flags=re.MULTILINE)
    if m:
        score = clamp_int(m.group(1))

    # 3) Other numeric patterns if strict line not found
    if score is None:
        patterns = [
            r"(?i)aggression\s*score[^0-9\n\r]*([0-9]{1,3})",      # "Aggression Score: 20"
            r"(?i)([0-9]{1,3})\s*/\s*100\b",                       # "20/100"
            r"(?i)([0-9]{1,3})\s+out of\s+100\b",                 # "20 out of 100"
            r"(?i)([0-9]{1,3})\s*percent\b",                      # "20 percent"
            r"(?i)([0-9]{1,3})\s*%\b",                            # "20%"
        ]
        for p in patterns:
            mm = re.search(p, t)
            if mm:
                val = clamp_int(mm.group(1))
                if val is not None:
                    score = val
                    break

    # 4) If still not found, look for any standalone integer 0..100 (conservative)
    if score is None:
        mm = re.search(r"\b([0-9]{1,3})\b", t)
        if mm:
            val = clamp_int(mm.group(1))
            # Attempt to be conservative: accept only if context near 'aggression' or line starts with the numeric
            if val is not None:
                # if the numeric appears on a line that includes 'Aggression' or is on its own line, accept it
                # check the containing line
                line = mm.string.splitlines()[0] if isinstance(mm.string, str) else None
                # simpler: accept the standalone integer if it's the only token on its line
                lines = t.splitlines()
                for ln in lines:
                    if re.fullmatch(r"\s*" + re.escape(mm.group(1)) + r"\s*", ln):
                        score = val
                        break
                # otherwise accept it (less strict) because we've failed other matches
                if score is None:
                    score = val

    # 5) Heuristic fallback from wording if numeric absent
    if score is None:
        # Prefer strong phrasing
        if re.search(r"(?i)\b(strong sell|sell aggressively|sell strongly|high conviction sell|recommend selling)\b", t):
            score = 90
        elif re.search(r"(?i)\b(strong buy|buy aggressively|buy strongly|high conviction buy|recommend buying)\b", t):
            score = 10
        elif re.search(r"(?i)\b(hold|neutral|wait and see|watch)\b", t):
            score = 50
        else:
            # map the simple recommendation word if present
            if recommendation == "SELL":
                score = 90
            elif recommendation == "BUY":
                score = 10
            elif recommendation == "HOLD":
                score = 50
            else:
                score = None

    return {"recommendation": recommendation, "aggression_score": score}

# ============ Main analysis ============
def analyze_with_nvidia(symbol, days=180):
    symbol = symbol.upper().strip()

    df = fetch_yfinance_history(symbol, days)
    price = summarize_price_history(df)
    tech = compute_technicals(df)
    quote = fetch_yfinance_quote(symbol)

    # company name
    try:
        info = yf.Ticker(symbol).info
        name_hint = info.get("longName") or info.get("shortName")
    except:
        name_hint = None

    # news
    news_items = []
    for q in (name_hint, symbol, f"{symbol} stock"):
        if not q:
            continue
        for item in fetch_google_news_rss(q):
            if item["title"] not in {n["title"] for n in news_items}:
                news_items.append(item)
        if len(news_items) >= 8:
            break

    prompt = build_prompt(symbol, name_hint, price, tech, news_items)
    assistant_text, raw = call_nvidia_chat(prompt)

    return {
        "symbol": symbol,
        "price_summary": price,
        "technical_indicators": tech,
        "news": news_items,
        "assistant_text": assistant_text,
        "parsed": parse_decision(assistant_text),
        "raw": raw
    }

# ============ CLI printing ============
def pretty_print(result):
    print("\n=== PRICE SUMMARY ===")
    print(json.dumps(result["price_summary"], indent=2))

    print("\n=== TECHNICALS ===")
    print(json.dumps(result["technical_indicators"], indent=2))

    print("\n=== NEWS ===")
    for n in result["news"]:
        print(f"- [{n['published']}] {n['title']}")

    print("\n=== NVIDIA ANALYSIS ===")
    print(result["assistant_text"])

    print("\n=== PARSED OUTPUT ===")
    print(json.dumps(result["parsed"], indent=2))

# ============ Main entry ============
def main():
    if len(sys.argv) >= 2:
        symbol = sys.argv[1]
    else:
        symbol = input("Ticker: ").strip()

    days = int(sys.argv[2]) if len(sys.argv) >= 3 else 180

    result = analyze_with_nvidia(symbol, days)
    pretty_print(result)

if __name__ == "__main__":
    main()
