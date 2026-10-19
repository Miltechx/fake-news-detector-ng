"""
fetch_factcheck_api.py
======================
Stage 1 data ingestion (Section 3.4.2.1): pulls fact-checked claims from the
free Google Fact Check Tools API, restricted to Dubawa and Africa Check
(the study's single documented source), and appends them to
data/raw/api_claims.csv. Safe to re-run: rows are deduplicated.

Also exposes lookup_claim(), used by the app to cross-reference a submitted
claim against existing fact-checker verdicts (FR8).

Get a free key: Google Cloud Console -> enable "Fact Check Tools API" ->
Credentials -> Create API key.

Usage:
  export GOOGLE_FACTCHECK_API_KEY=your_key
  python fetch_factcheck_api.py                     # default Nigerian topic queries
  python fetch_factcheck_api.py --queries "fuel subsidy" "INEC" --max-pages 10
"""

import argparse
import csv
import os
import sys
import time

import requests
from requests.adapters import HTTPAdapter, Retry

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FIELDNAMES = ["claim_id", "claim_text", "verdict", "source", "language_tag", "date", "url"]
PUBLISHER_SITES = ["dubawa.org", "africacheck.org"]
DEFAULT_QUERIES = [
    "Nigeria", "Nigerian", "Lagos", "Abuja", "INEC", "Tinubu", "naira", "CBN",
    "fuel subsidy", "Boko Haram", "bandits", "Kano", "Port Harcourt", "NCDC",
    "vaccine Nigeria", "election Nigeria", "Peter Obi", "Atiku", "EFCC", "Nigerian army",
]


def make_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers["User-Agent"] = "FakeNewsDetectorNG/1.0 (academic project)"
    return session


def _is_allowed_publisher(review, sites):
    pub = review.get("publisher") or {}
    hay = f"{pub.get('site', '')} {pub.get('name', '')} {review.get('url', '')}".lower()
    keys = [s.lower() for s in sites] + ["dubawa", "africa check", "africacheck"]
    return any(k in hay for k in keys)


def _rows_from_response(data, sites):
    rows = []
    for claim in data.get("claims", []) or []:
        text = (claim.get("text") or "").strip()
        for review in claim.get("claimReview", []) or []:
            if sites and not _is_allowed_publisher(review, sites):
                continue
            rating = (review.get("textualRating") or "").strip()
            publisher = (review.get("publisher") or {}).get("name") or "Unknown"
            url = review.get("url") or ""
            if text and rating:
                rows.append({
                    "claim_id": url or text[:80],
                    "claim_text": text,
                    "verdict": rating,
                    "source": publisher,
                    "language_tag": "",  # assigned by the language-ID pass in preprocess.py
                    "date": (review.get("reviewDate") or claim.get("claimDate") or "")[:10],
                    "url": url,
                })
    return rows


def fetch_claims(api_key, query=None, site=None, language=None, max_pages=5,
                 page_size=50, session=None):
    session = session or make_session()
    rows, token = [], None
    for page in range(max_pages):
        params = {"key": api_key, "pageSize": page_size}
        if query:
            params["query"] = query
        if site:
            params["reviewPublisherSiteFilter"] = site
        if language:
            params["languageCode"] = language
        if token:
            params["pageToken"] = token
        resp = session.get(API_URL, params=params, timeout=20)
        if resp.status_code == 400 and not query:
            break  # API requires a query for this combination
        resp.raise_for_status()
        data = resp.json()
        rows.extend(_rows_from_response(data, [site] if site else PUBLISHER_SITES))
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(0.4)
    return rows


def lookup_claim(text, api_key, timeout=8, max_results=5):
    """FR8: find existing Dubawa / Africa Check verdicts for a submitted claim.
    Returns a list of dicts; raises on network/API failure (caller degrades gracefully)."""
    query = " ".join(str(text).split()[:30])[:250]
    resp = requests.get(API_URL, params={"key": api_key, "query": query, "pageSize": 20},
                        timeout=timeout)
    resp.raise_for_status()
    rows = _rows_from_response(resp.json(), PUBLISHER_SITES)
    return rows[:max_results]


def load_existing(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        key = (r.get("claim_id"), r.get("source"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def run(api_key, queries=None, max_pages=5, out_path="data/raw/api_claims.csv"):
    if not api_key:
        raise ValueError("No API key. Set GOOGLE_FACTCHECK_API_KEY or pass --api-key.")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    existing = load_existing(out_path)
    session = make_session()
    new_rows = []
    for site in PUBLISHER_SITES:
        for q in (queries or DEFAULT_QUERIES):
            try:
                got = fetch_claims(api_key, query=q, site=site, max_pages=max_pages,
                                   session=session)
            except requests.HTTPError as e:
                print(f"  [{site}] '{q}': HTTP error {e}")
                continue
            print(f"  [{site}] '{q}': {len(got)} rows")
            new_rows.extend(got)
    merged = dedupe(existing + new_rows)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    added = len(merged) - len(existing)
    print(f"Fetched {len(new_rows)} rows, {added} new. Total in {out_path}: {len(merged)}")
    return added


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("GOOGLE_FACTCHECK_API_KEY"))
    ap.add_argument("--queries", nargs="*", default=None)
    ap.add_argument("--max-pages", type=int, default=5)
    ap.add_argument("--out", default="data/raw/api_claims.csv")
    a = ap.parse_args()
    if not a.api_key:
        sys.exit("Error: provide --api-key or set GOOGLE_FACTCHECK_API_KEY.")
    run(a.api_key, a.queries, a.max_pages, a.out)
