"""
scrape_factcheckers.py
======================
Stage 1 data ingestion (Sections 3.4.2.1 and 3.4.6): collects published
verdicts directly from the Dubawa and Africa Check archives using Requests +
BeautifulSoup, to supplement the Google Fact Check Tools API.

  Dubawa       -> WordPress REST API (dubawa.org/wp-json/wp/v2/posts). Each
                  fact-check has "Claim:" and "Verdict:" labels in the body.
  Africa Check -> Nigeria listing (africacheck.org/fact-checks?field_country_value=NG)
                  then each article page. Verdict is read from ClaimReview
                  structured data, a rating element, or the rating image.

Rows whose verdict or claim cannot be read reliably are NOT guessed. They go
to data/raw/needs_review_<site>.csv so a team member can open the article and
fill the 'verdict' column by hand from the published article (no claim is
labelled without a published verdict). preprocess.py automatically uses the
rows whose verdict has been filled in and ignores the rest.

Be polite: keep the delay (default 1.5 s) and respect each site's terms of use;
attribution is preserved in the 'source' and 'url' columns.

Usage:
  python scrape_factcheckers.py --site dubawa --max-pages 20
  python scrape_factcheckers.py --site africacheck --max-pages 30
  python scrape_factcheckers.py --site both
"""

import argparse
import csv
import json
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "FakeNewsDetectorNG/1.0 (academic project; contact: project team)"}
FIELDNAMES = ["claim_id", "claim_text", "verdict", "source", "language_tag", "date", "url"]
REVIEW_FIELDS = ["url", "title", "claim_guess", "verdict_guess", "verdict", "source", "date"]

VERDICT_WORDS = [
    "mostly false", "mostly true", "mostly correct", "partly false", "partly true",
    "half true", "false", "true", "misleading", "incorrect", "correct", "unproven",
    "unverifiable", "unverified", "exaggerated", "understated", "fake", "scam",
    "fabricated", "manipulated", "satire",
]
_VERDICT_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in VERDICT_WORDS) + r")\b", re.I)
NIGERIA_RE = re.compile(
    r"\b(nigeria|nigerian|nigerians|naija|lagos|abuja|kano|port harcourt|ibadan|kaduna|"
    r"enugu|inec|tinubu|buhari|naira|cbn|efcc|ncdc|boko haram)\b", re.I)


def _get(session, url, **kw):
    r = session.get(url, headers=UA, timeout=25, **kw)
    r.raise_for_status()
    return r


def _first_verdict(text):
    m = _VERDICT_RE.search(text or "")
    return m.group(1).title() if m else ""


# ---------------------------------------------------------------------------
# Dubawa
# ---------------------------------------------------------------------------
def parse_dubawa_post(title_html, content_html):
    """Return (claim, verdict) from a Dubawa post. Empty strings if not found."""
    soup = BeautifulSoup(content_html or "", "html.parser")
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    claim, verdict = "", ""
    m = re.search(r"Claim\s*:\s*(.+?)(?=\n\s*\n|Verdict\s*:|Full\s*Text|$)", text, re.I | re.S)
    if m:
        claim = " ".join(m.group(1).split())
    m = re.search(r"Verdict\s*:\s*([^\n]{1,200})", text, re.I)
    if m:
        verdict = _first_verdict(m.group(1))
    title = BeautifulSoup(title_html or "", "html.parser").get_text(" ").strip()
    if not verdict:
        m = re.match(r"^\s*(" + "|".join(re.escape(w) for w in VERDICT_WORDS) + r")\s*[!:.\-]",
                     title, re.I)
        if m:
            verdict = m.group(1).title()
    return claim, verdict, title, text


def scrape_dubawa(max_pages=10, per_page=50, delay=1.5, nigeria_only=True, search=None):
    session = requests.Session()
    rows, review = [], []
    for page in range(1, max_pages + 1):
        params = {"per_page": per_page, "page": page,
                  "_fields": "id,link,title,date,content"}
        if search:
            params["search"] = search
        try:
            r = _get(session, "https://dubawa.org/wp-json/wp/v2/posts", params=params)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 400:
                break  # past the last page
            raise
        posts = r.json()
        if not posts:
            break
        for p in posts:
            claim, verdict, title, body = parse_dubawa_post(
                (p.get("title") or {}).get("rendered", ""),
                (p.get("content") or {}).get("rendered", ""))
            if nigeria_only and not NIGERIA_RE.search(f"{title} {body}"):
                continue
            if "claim" not in body.lower() and not verdict:
                continue  # not a fact-check (news, training, reports)
            link, date = p.get("link", ""), (p.get("date") or "")[:10]
            if claim and verdict:
                rows.append({"claim_id": link, "claim_text": claim, "verdict": verdict,
                             "source": "Dubawa", "language_tag": "", "date": date, "url": link})
            else:
                review.append({"url": link, "title": title, "claim_guess": claim or title,
                               "verdict_guess": verdict, "verdict": "", "source": "Dubawa",
                               "date": date})
        total_pages = int(r.headers.get("X-WP-TotalPages", page))
        print(f"  Dubawa page {page}/{total_pages}: {len(rows)} usable, {len(review)} to review")
        if page >= total_pages:
            break
        time.sleep(delay)
    return rows, review


# ---------------------------------------------------------------------------
# Africa Check
# ---------------------------------------------------------------------------
AC_LIST = ("https://africacheck.org/fact-checks?field_article_type_value=All"
           "&field_rated_value=All&field_country_value=NG&sort_bef_combine=created_DESC&page={page}")
AC_LINK_RE = re.compile(r"^/(?:index\.php/|index\.ph%70/)?fact-checks/[a-z0-9\-]+/[a-z0-9\-]+/?$")


def parse_africacheck_listing(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("?")[0].split("#")[0]
        if href.startswith("https://africacheck.org"):
            href = href[len("https://africacheck.org"):]
        if AC_LINK_RE.match(href):
            href = href.replace("/index.ph%70", "").replace("/index.php", "")
            full = urljoin("https://africacheck.org", href)
            if full not in links:
                links.append(full)
    return links


def _claimreview_from_jsonld(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if node.get("@type") == "ClaimReview":
                    rating = node.get("reviewRating") or {}
                    return (node.get("claimReviewed") or "",
                            rating.get("alternateName") or rating.get("name") or "")
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend(node)
    return "", ""


def parse_africacheck_article(html_text):
    """Return (claim, verdict, title, date)."""
    soup = BeautifulSoup(html_text, "html.parser")
    title = (soup.find("h1").get_text(" ", strip=True) if soup.find("h1") else "")
    date = ""
    t = soup.find("time")
    if t is not None:
        date = (t.get("datetime") or t.get_text(strip=True))[:10]

    claim, verdict = _claimreview_from_jsonld(soup)
    verdict = _first_verdict(verdict) if verdict else ""

    if not verdict:  # rating element (class names containing rating/verdict)
        for el in soup.find_all(class_=re.compile(r"(rating|verdict)", re.I)):
            verdict = _first_verdict(el.get_text(" ", strip=True))
            if verdict:
                break
    if not verdict:  # rating image, e.g. ..._False.png / alt="Incorrect"
        for img in soup.find_all("img"):
            hay = f"{img.get('alt', '')} {os.path.basename(img.get('src', ''))}"
            hay = re.sub(r"[_\-]", " ", hay)
            if re.search(r"rating|verdict|rated|false|incorrect|correct|misleading", hay, re.I):
                verdict = _first_verdict(hay)
                if verdict:
                    break

    if not claim:  # "IN SHORT:" summary or meta description
        for el in soup.find_all(["p", "div", "strong"]):
            txt = el.get_text(" ", strip=True)
            if txt.upper().startswith("IN SHORT"):
                claim = re.sub(r"^IN SHORT\s*:?\s*", "", txt, flags=re.I)
                break
    if not claim:
        meta = soup.find("meta", attrs={"name": "description"}) or \
            soup.find("meta", attrs={"property": "og:description"})
        claim = meta.get("content", "") if meta else ""
    return " ".join(claim.split()), verdict, title, date


def scrape_africacheck(max_pages=10, delay=1.5):
    session = requests.Session()
    rows, review, seen = [], [], set()
    for page in range(max_pages):
        r = _get(session, AC_LIST.format(page=page))
        links = [l for l in parse_africacheck_listing(r.text) if l not in seen]
        if not links:
            break
        for link in links:
            seen.add(link)
            time.sleep(delay)
            try:
                art = _get(session, link)
            except requests.RequestException as e:
                print(f"    skip {link}: {e}")
                continue
            claim, verdict, title, date = parse_africacheck_article(art.text)
            is_report = "/reports/" in link  # reports check many claims at once
            if claim and verdict and not is_report:
                rows.append({"claim_id": link, "claim_text": claim, "verdict": verdict,
                             "source": "Africa Check", "language_tag": "", "date": date,
                             "url": link})
            else:
                review.append({"url": link, "title": title, "claim_guess": claim or title,
                               "verdict_guess": verdict, "verdict": "",
                               "source": "Africa Check", "date": date})
        print(f"  Africa Check page {page}: {len(rows)} usable, {len(review)} to review")
        time.sleep(delay)
    return rows, review


# ---------------------------------------------------------------------------
def _write(path, rows, fields):
    existing = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    seen, merged = set(), []
    for r in existing + rows:
        if r.get("url") in seen:
            continue
        seen.add(r.get("url"))
        merged.append(r)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    return len(merged)


def run(site="both", max_pages=10, delay=1.5, raw_dir="data/raw"):
    if site in ("dubawa", "both"):
        rows, review = scrape_dubawa(max_pages=max_pages, delay=delay)
        n = _write(os.path.join(raw_dir, "dubawa_claims.csv"), rows, FIELDNAMES)
        m = _write(os.path.join(raw_dir, "needs_review_dubawa.csv"), review, REVIEW_FIELDS)
        print(f"Dubawa: {n} labelled claims saved, {m} need manual review")
    if site in ("africacheck", "both"):
        rows, review = scrape_africacheck(max_pages=max_pages, delay=delay)
        n = _write(os.path.join(raw_dir, "africacheck_claims.csv"), rows, FIELDNAMES)
        m = _write(os.path.join(raw_dir, "needs_review_africacheck.csv"), review, REVIEW_FIELDS)
        print(f"Africa Check: {n} labelled claims saved, {m} need manual review")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["dubawa", "africacheck", "both"], default="both")
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--raw-dir", default="data/raw")
    a = ap.parse_args()
    run(a.site, a.max_pages, a.delay, a.raw_dir)
