#!/usr/bin/env python3
"""
Livesport match -> structured JSON -> (optional) LLM report -> Markdown

Install:
  pip install playwright pandas python-dotenv requests beautifulsoup4 lxml
  playwright install chromium

Run (debug scrape only):
  python livesport_pipeline.py "URL" --no-llm --headed

Run (with LLM article):
  python livesport_pipeline.py "URL" --lang en
  python livesport_pipeline.py "URL" --lang cs
  python livesport_pipeline.py "URL" --lang de

Env (.env) for LLM:
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-4o-mini
  OPENAI_BASE_URL=https://api.openai.com/v1

Outputs:
  out/match_data.json
  out/match_events.csv
  out/report.md
  out/xhr_dump/   (captured XHR/fetch bodies)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


# -----------------------------
# Data schema
# -----------------------------

@dataclass
class MatchEvent:
    minute: str
    side: str            # home | away | unknown
    event_type: str      # goal | yellow_card | substitution | disallowed_goal | ...
    player: Optional[str]
    detail: Optional[str]
    raw_text: str


@dataclass
class MatchData:
    source_url: str
    match_id: Optional[str]
    home_team: str
    away_team: str
    score_home: Optional[int]
    score_away: Optional[int]
    kickoff_or_date_text: Optional[str]
    competition_text: Optional[str]
    events: List[MatchEvent]


# -----------------------------
# Utilities
# -----------------------------

def slugify(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().lower()
    s = s.replace(" vs ", " vs ").replace(":", "-")
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_.")
    return s[:max_len] if len(s) > max_len else s


def build_match_slug(match: MatchData) -> str:
    # Try to include date/time text if present, plus match id for uniqueness
    dt = slugify(match.kickoff_or_date_text or "")
    home = slugify(match.home_team)
    away = slugify(match.away_team)
    mid = slugify(match.match_id or "")
    core = f"{dt}_{home}-vs-{away}".strip("_")
    if mid:
        core = f"{core}_{mid}"
    return core or (mid or "match")

def ensure_out_dir(path: str = "out") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_int(x: str) -> Optional[int]:
    x = (x or "").strip()
    if not x:
        return None
    m = re.search(r"-?\d+", x)
    return int(m.group(0)) if m else None


def extract_match_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    mid = qs.get("mid", [None])[0]
    if mid and re.fullmatch(r"[A-Za-z0-9]{6,16}", mid):
        return mid
    return None


def first_text(locator, default: str = "") -> str:
    try:
        if locator.count() == 0:
            return default
        return clean_text(locator.first.inner_text())
    except Exception:
        return default


# -----------------------------
# Consent handling
# -----------------------------

def try_accept_cookies(page) -> None:
    candidates = [r"Souhlasím", r"Přijmout", r"Accept", r"I agree", r"Rozumím", r"OK"]
    for patt in candidates:
        try:
            btn = page.get_by_role("button", name=re.compile(patt, re.I))
            if btn.count() > 0:
                btn.first.click(timeout=1500)
                return
        except Exception:
            pass


# -----------------------------
# Event parsing
# -----------------------------

DOM_EVENT_ROW_SELECTORS = [
    ".smv__incidentRow",
    "[class*='smv__incidentRow']",
]

def classify_event_type(icon_href: Optional[str], raw: str) -> str:
    raw_l = (raw or "").lower()
    if icon_href:
        icon_l = icon_href.lower()
        if "own-goal" in icon_l or "owngoal" in icon_l:
            return "own_goal"
        if "penalty" in icon_l and "miss" in icon_l:
            return "penalty_missed"
        if "penalty" in icon_l:
            return "penalty"
        if "goal" in icon_l:
            return "goal"
        if "yellow" in icon_l:
            return "yellow_card"
        if "red" in icon_l:
            return "red_card"
        if "sub" in icon_l or "substitution" in icon_l:
            return "substitution"
        if "var" in icon_l:
            return "var"

    # CZ/EN text fallback
    if "neuznan" in raw_l:
        return "disallowed_goal"
    if "žlut" in raw_l or "yellow" in raw_l:
        return "yellow_card"
    if "červen" in raw_l or "red" in raw_l:
        return "red_card"
    if "stříd" in raw_l or "substitution" in raw_l:
        return "substitution"
    if "penalt" in raw_l:
        return "penalty"
    if "gól" in raw_l or "goal" in raw_l:
        return "goal"
    return "unknown"


def parse_events_from_html_like(html_text: str) -> List[MatchEvent]:
    soup = BeautifulSoup(html_text, "lxml")
    rows = soup.select(".smv__incidentRow, [class*='smv__incidentRow']")
    events: List[MatchEvent] = []

    for row in rows:
        cls = " ".join(row.get("class", [])).lower()
        side = "home" if "home" in cls else "away" if "away" in cls else "unknown"

        minute_el = row.select_one(".smv__timeBox") or row.select_one("[class*='time']")
        minute = clean_text(minute_el.get_text(" ", strip=True)) if minute_el else ""

        icon_href = None
        use = row.select_one("svg use")
        if use:
            icon_href = use.get("href") or use.get("xlink:href")

        player_el = row.select_one(".smv__playerName")
        player = clean_text(player_el.get_text(" ", strip=True)) if player_el else None
        if player == "":
            player = None

        assist_el = row.select_one(".smv__assist")
        assist = clean_text(assist_el.get_text(" ", strip=True)) if assist_el else ""
        detail = f"assist: {assist}" if assist else None

        raw_text = clean_text(row.get_text(" ", strip=True))
        events.append(MatchEvent(
            minute=minute,
            side=side,
            event_type=classify_event_type(icon_href, raw_text),
            player=player,
            detail=detail,
            raw_text=raw_text,
        ))

    return events


def _best_desc(values: List[str]) -> str:
    vals = [v for v in values if v and v.strip()]
    if not vals:
        return ""
    return max(vals, key=len)


def parse_events_from_flashscore_feed(feed_text: str) -> List[MatchEvent]:
    """
    Parses x/feed/df_sui_1_<mid> format.
    """
    starts = [m.start() for m in re.finditer(r"~III[A-Z]*÷", feed_text)]
    if not starts:
        return []
    starts.append(len(feed_text))

    events: List[MatchEvent] = []

    for i in range(len(starts) - 1):
        block = feed_text[starts[i]:starts[i + 1]]
        parts = [p for p in block.split("¬") if "÷" in p]
        if not parts:
            continue

        ordered: List[tuple[str, str]] = []
        data: Dict[str, List[str]] = {}

        for p in parts:
            k, v = p.split("÷", 1)
            v = v.rstrip("|")

            if k.endswith("X") and k not in ("INX", "IOX"):
                k = k[:-1]

            ordered.append((k, v))
            data.setdefault(k, []).append(v)

        minute = (data.get("IB") or [""])[0]
        side_val = (data.get("IA") or [""])[0]
        side = "home" if side_val == "1" else "away" if side_val == "2" else "unknown"

        desc = _best_desc(data.get("ICT", []))
        desc = desc.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
        desc = clean_text(re.sub(r"<[^>]+>", "", desc))

        current_player: Optional[str] = None
        main_type: Optional[str] = None
        main_player: Optional[str] = None
        assist: Optional[str] = None
        sub_out: Optional[str] = None
        sub_in: Optional[str] = None
        reason = (data.get("IL") or [None])[0]

        for k, v in ordered:
            if k == "IF":
                current_player = v
            elif k == "IK":
                typ = v

                if "Asistence" in typ:
                    assist = current_player
                    continue

                if "Střídání - Out" in typ:
                    sub_out = current_player
                    continue

                if "Střídání" in typ and "Out" not in typ:
                    sub_in = current_player
                    if main_type is None:
                        main_type = "substitution"
                        main_player = sub_out or current_player
                    continue

                if "Neuznan" in typ:
                    if main_type is None:
                        main_type = "disallowed_goal"
                        main_player = current_player
                    continue

                if "Žlut" in typ:
                    if main_type is None:
                        main_type = "yellow_card"
                        main_player = current_player
                    continue

                if "Červen" in typ:
                    if main_type is None:
                        main_type = "red_card"
                        main_player = current_player
                    continue

                if "Gól" in typ:
                    if main_type is None:
                        main_type = "goal"
                        main_player = current_player
                    continue

        if not main_type:
            if sub_in or sub_out:
                main_type = "substitution"
                main_player = sub_out
            else:
                continue

        detail_parts: List[str] = []
        if main_type == "goal" and assist:
            detail_parts.append(f"assist: {assist}")
        if main_type == "substitution":
            if sub_in:
                detail_parts.append(f"sub_in: {sub_in}")
            if sub_out:
                detail_parts.append(f"sub_out: {sub_out}")
        if reason and main_type in ("yellow_card", "red_card"):
            detail_parts.append(f"reason: {reason}")

        detail = "; ".join(detail_parts) if detail_parts else None
        raw_text = desc or clean_text(" ".join(
            f"{k}={v}" for (k, v) in ordered if k in ("IF", "IK")
        ))

        events.append(MatchEvent(
            minute=minute,
            side=side,
            event_type=main_type,
            player=main_player,
            detail=detail,
            raw_text=raw_text,
        ))

    return events


def parse_events_from_dom(page) -> List[MatchEvent]:
    events: List[MatchEvent] = []

    for sel in DOM_EVENT_ROW_SELECTORS:
        try:
            page.wait_for_selector(sel, timeout=6000)
        except Exception:
            continue

        rows = page.locator(sel)
        if rows.count() == 0:
            continue

        for i in range(rows.count()):
            row = rows.nth(i)

            cls = (row.get_attribute("class") or "").lower()
            side = "home" if "home" in cls else "away" if "away" in cls else "unknown"

            minute = first_text(row.locator(".smv__timeBox"), default="") \
                     or first_text(row.locator("[class*='time']"), default="")

            icon_href = None
            try:
                use_el = row.locator("svg use")
                if use_el.count() > 0:
                    icon_href = use_el.first.get_attribute("href") or use_el.first.get_attribute("xlink:href")
            except Exception:
                pass

            player = first_text(row.locator(".smv__playerName"), default="") or None
            assist = first_text(row.locator(".smv__assist"), default="")
            detail = f"assist: {assist}" if assist else None

            try:
                raw_text = clean_text(row.inner_text())
            except Exception:
                raw_text = clean_text(row.text_content() or "")

            events.append(MatchEvent(
                minute=minute or "",
                side=side,
                event_type=classify_event_type(icon_href, raw_text),
                player=player if player else None,
                detail=detail,
                raw_text=raw_text,
            ))

        if events:
            return events

    return []


# -----------------------------
# XHR collector
# -----------------------------

class XHRCollector:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.items: List[Dict[str, Any]] = []
        os.makedirs(self.out_dir, exist_ok=True)

    def attach(self, page) -> None:
        def handler(resp):
            try:
                if resp.request.resource_type not in ("xhr", "fetch"):
                    return
                body = resp.body()
                if not body:
                    return

                url = resp.url
                ct = (resp.headers.get("content-type") or "").lower()
                self.items.append({"url": url, "content_type": ct, "body": body})

                safe = re.sub(r"[^a-zA-Z0-9]+", "_", url)[:180]
                ext = "json" if "json" in ct else "html" if "html" in ct else "txt"
                with open(os.path.join(self.out_dir, f"{safe}.{ext}"), "wb") as f:
                    f.write(body)
            except Exception:
                pass

        page.on("response", handler)

    def extract_events(self) -> List[MatchEvent]:
        # HTML incidents
        for it in self.items:
            t = it["body"].decode("utf-8", errors="ignore")
            if "smv__incidentRow" in t:
                ev = parse_events_from_html_like(t)
                if ev:
                    return ev

        # Flashscore feed format
        for it in self.items:
            t = it["body"].decode("utf-8", errors="ignore")
            if "~III" in t and "IK÷" in t and "IB÷" in t:
                ev = parse_events_from_flashscore_feed(t)
                if ev:
                    return ev

        # JSON-wrapped strings
        for it in self.items:
            t = it["body"].decode("utf-8", errors="ignore")
            if "json" not in (it.get("content_type") or ""):
                continue
            try:
                obj = json.loads(t)
            except Exception:
                continue

            stack = [obj]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
                elif isinstance(x, str):
                    if "smv__incidentRow" in x:
                        ev = parse_events_from_html_like(x)
                        if ev:
                            return ev
                    if "~III" in x and "IK÷" in x and "IB÷" in x:
                        ev = parse_events_from_flashscore_feed(x)
                        if ev:
                            return ev

        return []


# -----------------------------
# Navigation
# -----------------------------

TEAM_SELECTORS = [
    ".duelParticipant__home .participant__participantName",
    ".duelParticipant__home .participant__participantNameWrapper",
    ".duelParticipant__home",
]
AWAY_SELECTORS = [
    ".duelParticipant__away .participant__participantName",
    ".duelParticipant__away .participant__participantNameWrapper",
    ".duelParticipant__away",
]

def goto_match_summary(page, url: str) -> None:
    base = url.split("#", 1)[0]
    page.goto(base, wait_until="domcontentloaded", timeout=60000)
    try_accept_cookies(page)

    page.evaluate("() => { location.hash = '#/prehled-zapasu/prehled-zapasu'; }")
    try:
        page.wait_for_function("() => location.hash.includes('prehled-zapasu')", timeout=15000)
    except Exception:
        pass

    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass

    for _ in range(10):
        page.mouse.wheel(0, 1600)
        page.wait_for_timeout(250)


# -----------------------------
# Scrape match
# -----------------------------

def scrape_match(url: str, headed: bool) -> MatchData:
    match_id = extract_match_id(url)
    ensure_out_dir("out")
    dump_dir = os.path.join("out", "xhr_dump")
    os.makedirs(dump_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        context = browser.new_context(
            locale="cs-CZ",
            timezone_id="Europe/Prague",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            ),
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

        page = context.new_page()
        collector = XHRCollector(dump_dir)
        collector.attach(page)

        goto_match_summary(page, url)

        home_team = ""
        away_team = ""
        for sel in TEAM_SELECTORS:
            t = first_text(page.locator(sel), default="")
            if t and len(t) < 80:
                home_team = t
                break
        for sel in AWAY_SELECTORS:
            t = first_text(page.locator(sel), default="")
            if t and len(t) < 80:
                away_team = t
                break
        home_team = home_team or "UNKNOWN_HOME"
        away_team = away_team or "UNKNOWN_AWAY"

        score_home = norm_int(first_text(page.locator(".detailScore__home"), default=""))
        score_away = norm_int(first_text(page.locator(".detailScore__away"), default=""))
        if score_home is None or score_away is None:
            wrapper = first_text(page.locator(".detailScore__wrapper"), default="")
            m = re.search(r"(\d+)\s*[-:]\s*(\d+)", wrapper)
            if m:
                score_home, score_away = int(m.group(1)), int(m.group(2))

        competition_text = first_text(page.locator(".tournamentHeader__country"), default="") or None
        kickoff_or_date_text = first_text(page.locator(".duelParticipant__startTime"), default="") or None

        events = parse_events_from_dom(page)
        if not events:
            events = collector.extract_events()

        browser.close()

    return MatchData(
        source_url=url,
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        score_home=score_home,
        score_away=score_away,
        kickoff_or_date_text=kickoff_or_date_text,
        competition_text=competition_text,
        events=events,
    )


# -----------------------------
# Persist outputs
# -----------------------------

def save_json(match: MatchData, out_dir: str, slug: str) -> str:
    path = os.path.join(out_dir, f"match_data__{slug}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**asdict(match), "events": [asdict(e) for e in match.events]}, f, ensure_ascii=False, indent=2)
    return path


def save_events_csv(match: MatchData, out_dir: str, slug: str) -> str:
    path = os.path.join(out_dir, f"match_events__{slug}.csv")
    df = pd.DataFrame([asdict(e) for e in match.events])
    if df.empty:
        df = pd.DataFrame(columns=["home_team", "away_team", "minute", "side", "event_type", "player", "detail", "raw_text"])
    else:
        df.insert(0, "home_team", match.home_team)
        df.insert(1, "away_team", match.away_team)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_report_md(md: str, out_dir: str, slug: str, lang: str) -> str:
    path = os.path.join(out_dir, f"report__{slug}__{slugify(lang)}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md.strip() + "\n")
    return path


def make_local_report(match: MatchData) -> str:
    title = f"# {match.home_team} vs {match.away_team} — {match.score_home if match.score_home is not None else '?'}:{match.score_away if match.score_away is not None else '?'}"
    tl = "\n".join(f"- {e.minute} — {e.side} — {e.raw_text}" for e in match.events) if match.events else "- Not available from source."
    return f"{title}\n\nSource: {match.source_url}\n\n## Timeline\n\n{tl}\n"


# -----------------------------
# LLM (OpenAI Responses API)
# -----------------------------

def env_any(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def build_llm_prompt(match: MatchData, lang: str = "en") -> Dict[str, str]:
    payload = {
        "match": {
            "home_team": match.home_team,
            "away_team": match.away_team,
            "score_home": match.score_home,
            "score_away": match.score_away,
            "kickoff_or_date_text": match.kickoff_or_date_text,
            "competition_text": match.competition_text,
            "events": [asdict(e) for e in match.events],
            "source_url": match.source_url,
        }
    }

    instructions = (
        "You are a football journalist.\n"
        f"Write the article in: {lang}.\n\n"
        "Style requirements:\n"
        "- Football journalist tone: vivid, match-day atmosphere, momentum swings, but professional.\n"
        "- Do NOT invent facts. Use ONLY the provided JSON.\n"
        "- If something is missing (stadium, attendance, scorers, etc.), explicitly say it’s not available from source.\n"
        "- Mention the final score early.\n"
        "- Reflect the event order (goals/cards/subs/VAR) in the narrative.\n\n"
        "Format requirements (Markdown):\n"
        "1) Headline (one line)\n"
        "2) Lede paragraph (2–3 sentences)\n"
        "3) Body (2–3 short paragraphs)\n"
        "4) 'Key moments' bullet list (3–6 items)\n"
        "5) 'Timeline' section listing ALL events in order as: minute — team — description\n\n"
        "Length requirement:\n"
        "- Target 280–330 words total.\n"
    )

    return {
        "instructions": instructions,
        "input": json.dumps(payload, ensure_ascii=False),
    }


def _extract_output_text(resp_json: dict) -> str:
    if isinstance(resp_json, dict) and resp_json.get("output_text"):
        return resp_json["output_text"]
    out: List[str] = []
    for item in resp_json.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and c.get("text"):
                out.append(c["text"])
    return "\n".join(out).strip()


def _parse_retry_after_seconds(resp: requests.Response) -> Optional[float]:
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            return None
    ra_ms = resp.headers.get("retry-after-ms")
    if ra_ms:
        try:
            return float(ra_ms) / 1000.0
        except ValueError:
            return None
    return None


def call_openai_responses_with_retry(instructions: str, input_text: str) -> str:
    base_url = env_any("OPENAI_BASE_URL", default="https://api.openai.com/v1").rstrip("/")
    api_key = env_any("OPENAI_API_KEY", default="")
    model = env_any("OPENAI_MODEL", default="gpt-4o-mini")

    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var.")

    url = f"{base_url}/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "temperature": 0.6,
        "max_output_tokens": 900,
    }

    max_retries = 6
    base_sleep = 1.0

    for attempt in range(max_retries + 1):
        r = requests.post(url, headers=headers, json=body, timeout=120)

        if r.status_code < 400:
            return _extract_output_text(r.json())

        err_msg = ""
        try:
            j = r.json()
            if isinstance(j, dict) and isinstance(j.get("error"), dict):
                err_msg = j["error"].get("message") or ""
        except Exception:
            pass

        print(f"[LLM] HTTP {r.status_code} (attempt {attempt}/{max_retries})")
        if err_msg:
            print(f"[LLM] message={err_msg}")

        if r.status_code == 429 and "quota" in err_msg.lower():
            raise requests.HTTPError(f"429 quota/billing: {err_msg}", response=r)

        if r.status_code in (429, 500, 503, 408):
            if attempt == max_retries:
                r.raise_for_status()
            ra = _parse_retry_after_seconds(r)
            sleep_s = ra if ra is not None else (base_sleep * (2 ** attempt) + random.random() * 0.25)
            time.sleep(sleep_s)
            continue

        r.raise_for_status()

    raise RuntimeError("Unreachable")


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM call (debug)")
    ap.add_argument("--headed", action="store_true", help="Show browser UI")
    ap.add_argument(
        "--lang",
        default="en",
        help="Output language for the article (e.g., en, cs, de, es, fr). Default: en",
    )
    args = ap.parse_args()

    out_dir = ensure_out_dir("out")

    print("[1/2] Scraping match page...")
    match = scrape_match(args.url, headed=args.headed)

    # Build slug for filenames
    match_slug = build_match_slug(match)

    print("[2/2] Saving structured outputs...")
    print("  -", save_json(match, out_dir, match_slug))
    print("  -", save_events_csv(match, out_dir, match_slug))

    if args.no_llm:
        local_md = make_local_report(match)
        print("  -", save_report_md(local_md, out_dir, match_slug, "local"))
        return 0

    prompt = build_llm_prompt(match, lang=args.lang)
    md = call_openai_responses_with_retry(prompt["instructions"], prompt["input"])
    print("  -", save_report_md(md, out_dir, match_slug, args.lang))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())