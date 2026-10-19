"""
preprocess.py
=============
Stage 1-2 of the pipeline (Chapter Three, Sections 3.4.2.1 - 3.4.2.4).

Takes one or more raw claim CSVs (sample, Google Fact Check API, Dubawa and
Africa Check scrapes, or a hand-collected file) and produces a clean,
labelled, deduplicated dataset split 80/10/10 into train / val / test.

Steps, in the order the chapter describes them:
  1. Label harmonisation   True -> Real ; False / Misleading / Mostly False -> Fake ;
                           Unverifiable / Unproven / unclear -> dropped (3.4.2.3)
  2. Language identification pass (English vs Nigerian Pidgin / code-switched)
  3. Topical category + group tags used for bias auditing (3.4.5, 3.4.7):
     category, region (geopolitical zone), party_ref, religion_ref, ethnic_ref
  4. Duplicate removal: exact (hash of normalised text) and near-duplicate
     (TF-IDF cosine > 0.90 within the same category); conflicting verdicts
     for the same claim are excluded entirely (3.4.2.3)
  5. Cleaning: HTML, URLs, emojis, personal data (emails, phone numbers,
     @handles) removed; Nigeria-adapted stopwords; stemming applied to
     English records only (3.4.2.4)
  6. 80/10/10 split stratified on label (+ source where possible).
     SMOTE is NOT applied here - it is applied to the training partition
     only, inside train_model.py, so nothing synthetic leaks into val/test.

Usage:
  python preprocess.py                         # all CSVs in data/raw/
  python preprocess.py --inputs data/raw/api_claims.csv data/raw/dubawa_claims.csv
  python preprocess.py --include-sample        # keep the sample file even if real data exists
"""

import argparse
import glob
import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
NEAR_DUP_THRESHOLD = 0.90
SAMPLE_FILENAME = "sample_claims.csv"

# ---------------------------------------------------------------------------
# 1. Label harmonisation
# ---------------------------------------------------------------------------
# Edit this table if your sources use other rating words. None = excluded.
VERDICT_MAP = {
    # Real
    "true": "Real", "correct": "Real", "mostly true": "Real",
    "mostly correct": "Real", "verified": "Real", "accurate": "Real",
    "real": "Real",
    # Fake (Misleading maps to Fake - justified in Section 3.4.2.3)
    "false": "Fake", "fake": "Fake", "misleading": "Fake",
    "mostly false": "Fake", "incorrect": "Fake", "exaggerated": "Fake",
    "fabricated": "Fake", "manipulated": "Fake", "altered": "Fake",
    "doctored": "Fake", "scam": "Fake", "hoax": "Fake",
    "partly false": "Fake", "partially false": "Fake",
    "false context": "Fake", "misleading context": "Fake",
    # Excluded - no defensible direction
    "unverifiable": None, "unverified": None, "unproven": None,
    "unclear": None, "insufficient evidence": None, "investigating": None,
    "satire": None, "explainer": None, "checked": None, "understated": None,
    "half true": None, "partly true": None, "mixed": None, "no evidence": None,
    "unrated": None,
}
_VERDICT_KEYS = sorted(VERDICT_MAP, key=len, reverse=True)


def _norm_verdict(verdict) -> str:
    v = str(verdict if verdict is not None else "").strip().lower()
    v = re.sub(r"[\s_\-]+", " ", v)
    return v.strip(" !.:;,\"'")


def harmonize_label(verdict):
    """Map a fact-checker rating (a word or a full sentence) to Real / Fake / None."""
    v = _norm_verdict(verdict)
    if not v or v == "nan":
        return None
    if v in VERDICT_MAP:
        return VERDICT_MAP[v]
    # Ratings often arrive as sentences: "False. The photo is from 2019..."
    for key in _VERDICT_KEYS:
        if re.match(rf"^{re.escape(key)}(?![a-z])", v):
            return VERDICT_MAP[key]
    return None


# ---------------------------------------------------------------------------
# 2. Language identification (English vs Nigerian Pidgin / code-switched)
# ---------------------------------------------------------------------------
PIDGIN_STRONG = {
    "dey", "wetin", "abeg", "una", "wahala", "sabi", "pikin", "comot", "naija",
    "wey", "dem", "sef", "oya", "haba", "don", "dis", "dat", "pipu", "pipo",
    "tok", "shey", "shebi", "abi", "wan", "katakata", "gbege", "yarn", "nna",
    "sey", "kpatakpata", "na", "unu", "ehn", "nawa",
}
PIDGIN_WEAK = {"oga", "sha", "chop", "waka", "fit", "okada", "kuku", "jare", "o"}
PIDGIN_PATTERNS = [
    r"\bfor (?:whole|di|dis|dat)\b",
    r"\bfit (?!the\b|a\b|an\b|in\b|into\b|for\b|with\b|and\b|to\b)[a-z]+",
    r"\bgo (?:dey|don)\b",
]


def detect_language(text) -> str:
    """Return 'pcm' (Pidgin / code-switched) or 'en' (standard Nigerian English)."""
    t = str(text).lower()
    tokens = re.findall(r"[a-z']+", t)
    score = sum(1.0 for w in tokens if w in PIDGIN_STRONG)
    score += sum(0.5 for w in tokens if w in PIDGIN_WEAK)
    score += sum(1.0 for p in PIDGIN_PATTERNS if re.search(p, t))
    return "pcm" if score >= 1.0 else "en"


# ---------------------------------------------------------------------------
# 3. Category and bias-audit group tags
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "politics": [
        "inec", "election", "elections", "president", "presidency", "governor",
        "senate", "senator", "house of reps", "apc", "pdp", "labour party",
        "nnpp", "tinubu", "atiku", "peter obi", "obi", "buhari", "jonathan",
        "shettima", "vote", "votes", "voting", "bvas", "irev", "minister",
        "assembly", "campaign", "ballot", "polling", "lawmaker", "politician",
        "party", "tribunal", "candidate", "aso rock", "wike",
    ],
    "health": [
        "covid", "covid19", "coronavirus", "vaccine", "vaccines", "malaria",
        "cure", "cures", "disease", "hospital", "doctor", "doctors", "cancer",
        "virus", "health", "drug", "drugs", "ncdc", "world health organization", "lassa", "cholera",
        "monkeypox", "mpox", "diabetes", "hiv", "aids", "herbal", "medicine",
        "pregnant", "pregnancy", "polio", "nafdac", "infection", "treatment",
    ],
    "security": [
        "boko haram", "bandit", "bandits", "kidnap", "kidnapped", "kidnapping",
        "attack", "attacked", "army", "police", "killed", "killing", "terrorist",
        "terrorists", "ipob", "herdsmen", "military", "gunmen", "explosion",
        "bomb", "iswap", "soldiers", "insurgent", "insurgents", "dss", "efcc",
        "arrest", "arrested", "shooting", "unknown gunmen", "esn",
    ],
    "entertainment": [
        "bbnaija", "big brother", "nollywood", "actor", "actress", "singer",
        "music", "davido", "wizkid", "burna boy", "celebrity", "movie", "film",
        "football", "super eagles", "afcon", "osimhen", "tiktok star",
        "comedian", "album", "concert", "award", "musician", "rapper",
    ],
}
CATEGORY_PRIORITY = ["politics", "health", "security", "entertainment"]

ZONE_KEYWORDS = {
    "North-Central": ["benue", "kogi", "kwara", "nasarawa", "niger state",
                      "plateau state", "fct", "abuja", "jos", "ilorin",
                      "makurdi", "lokoja", "minna", "lafia"],
    "North-East": ["adamawa", "bauchi", "borno", "gombe", "taraba", "yobe",
                   "maiduguri", "yola", "jalingo", "damaturu"],
    "North-West": ["jigawa", "kaduna", "kano", "katsina", "kebbi", "sokoto",
                   "zamfara", "zaria", "dutse", "birnin kebbi", "gusau"],
    "South-East": ["abia", "anambra", "ebonyi", "enugu", "imo", "owerri",
                   "onitsha", "awka", "abakaliki", "umuahia", "nnewi"],
    "South-South": ["akwa ibom", "bayelsa", "cross river", "delta state",
                    "edo", "rivers state", "port harcourt", "uyo", "calabar",
                    "benin city", "warri", "asaba", "yenagoa", "niger delta",
                    "delta"],
    "South-West": ["ekiti", "lagos", "ogun", "ondo", "osun", "oyo", "ibadan",
                   "abeokuta", "akure", "osogbo", "ikeja", "lekki",
                   "ado ekiti"],
}
PARTY_KEYWORDS = {
    "APC": ["apc", "all progressives congress"],
    "PDP": ["pdp", "peoples democratic party"],
    "LP": ["labour party", "obidient", "obidients"],
    "NNPP": ["nnpp", "kwankwaso"],
    "APGA": ["apga"],
    "ADC": ["adc", "african democratic congress"],
}
RELIGION_KEYWORDS = {
    "Christian": ["christian", "christians", "church", "churches", "pastor",
                  "bishop", "catholic", "pentecostal", "priest"],
    "Muslim": ["muslim", "muslims", "islam", "islamic", "mosque", "imam",
               "sharia", "hijab", "jumat", "emir", "ramadan"],
}
ETHNIC_KEYWORDS = {
    "Igbo": ["igbo", "ndigbo", "biafra", "biafran"],
    "Yoruba": ["yoruba", "oduduwa"],
    "Hausa": ["hausa"],
    "Fulani": ["fulani"],
    "Ijaw": ["ijaw", "izon"],
    "Tiv": ["tiv"],
    "Kanuri": ["kanuri"],
    "Other": ["ogoni", "ibibio", "efik", "itsekiri", "urhobo", "isoko", "nupe",
              "idoma", "igala", "berom"],
}
GROUP_COLUMNS = ["language_tag", "category", "source", "region", "party_ref",
                 "religion_ref", "ethnic_ref"]


def _count_hits(text, keywords):
    return sum(1 for k in keywords if re.search(rf"\b{re.escape(k)}\b", text))


def _plain(text) -> str:
    t = html.unescape(str(text)).lower()
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^\w\s]", " ", t).replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def tag_category(text) -> str:
    t = _plain(text)
    scores = {c: _count_hits(t, kws) for c, kws in CATEGORY_KEYWORDS.items()}
    best = max(scores.values())
    if best == 0:
        return "other"
    for c in CATEGORY_PRIORITY:
        if scores[c] == best:
            return c
    return "other"


def _multi_tag(text, table, none_value="None"):
    t = _plain(text)
    hits = [name for name, kws in table.items() if _count_hits(t, kws) > 0]
    if not hits:
        return none_value
    return hits[0] if len(hits) == 1 else "Multiple"


def tag_region(text) -> str:
    t = _plain(text)
    t = re.sub(r"\bniger republic\b", " ", t)
    hits = [z for z, kws in ZONE_KEYWORDS.items() if _count_hits(t, kws) > 0]
    if not hits:
        return "National/Unspecified"
    return hits[0] if len(hits) == 1 else "Multiple"


def tag_groups(text) -> dict:
    return {
        "category": tag_category(text),
        "region": tag_region(text),
        "party_ref": _multi_tag(text, PARTY_KEYWORDS),
        "religion_ref": _multi_tag(text, RELIGION_KEYWORDS),
        "ethnic_ref": _multi_tag(text, ETHNIC_KEYWORDS),
    }


# ---------------------------------------------------------------------------
# 5. Cleaning
# ---------------------------------------------------------------------------
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "nor",
    "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
}
# NOTE: negations ("no", "not", "never", "don't") are deliberately kept -
# they flip the meaning of a claim.

# Nigerian Pidgin function words. Content words like "wahala", "sabi",
# "tok" are deliberately NOT included - they carry meaning.
PIDGIN_STOPWORDS = {
    "na", "dey", "dis", "dat", "dem", "una", "unu", "abeg", "sha", "sef",
    "wetin", "wey", "don", "fit", "wan", "shey", "abi", "nna", "biko",
    "chai", "e", "im", "pipu", "pipo",
}
STOPWORDS = ENGLISH_STOPWORDS | PIDGIN_STOPWORDS
# Pidgin content words are never stemmed even inside English records
PROTECTED_FROM_STEMMING = PIDGIN_STRONG | PIDGIN_WEAK | {"naira", "bvas", "inec"}

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?:\+?234|\b0)[789][01]\d[\s-]?\d{3}[\s-]?\d{4}\b")
_URL = re.compile(r"https?://\S+|www\.\S+")
_HANDLE = re.compile(r"@\w+")


def scrub_pii(text) -> str:
    """Remove personal data (emails, phone numbers, @handles). Used before logging."""
    t = str(text)
    t = _EMAIL.sub("[email]", t)
    t = _PHONE.sub("[phone]", t)
    t = _HANDLE.sub("[user]", t)
    return t


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer  # no corpus download needed
    return PorterStemmer()


@lru_cache(maxsize=50000)
def _stem(word):
    return _stemmer().stem(word)


def clean_text(text, lang=None, stem=True) -> str:
    """Clean one claim. The same function is used at training time and in the app."""
    raw = html.unescape(str(text if text is not None else ""))
    if lang is None:
        lang = detect_language(raw)
    t = raw.lower()
    t = re.sub(r"<[^>]+>", " ", t)          # HTML tags
    t = _URL.sub(" ", t)                     # URLs
    t = _EMAIL.sub(" ", t)                   # emails (personal data)
    t = _PHONE.sub(" ", t)                   # phone numbers (personal data)
    t = _HANDLE.sub(" ", t)                  # @handles (personal data)
    t = re.sub(r"[^\w\s]", " ", t)           # punctuation AND emojis
    t = t.replace("_", " ")
    t = re.sub(r"\b\d+\b", " ", t)           # standalone numbers (keeps 5g, covid19)
    tokens = []
    for w in t.split():
        if w in STOPWORDS:
            continue
        if len(w) < 2 or (len(w) == 2 and not any(c.isdigit() for c in w)):
            continue
        if stem and lang == "en" and w not in PROTECTED_FROM_STEMMING:
            w = _stem(w)
        tokens.append(w)
    return " ".join(tokens)


def normalise_for_dedup(text) -> str:
    return _plain(text)


# ---------------------------------------------------------------------------
# 4. Duplicate and conflict removal
# ---------------------------------------------------------------------------
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def remove_duplicates(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    df = df.copy()
    df["_norm"] = df["claim_text"].apply(normalise_for_dedup)
    df["_hash"] = df["_norm"].apply(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest())

    # Pass 1: exact duplicates (same normalised text)
    keep_idx, exact_removed, conflict_removed = [], 0, 0
    conflicts = []
    for _, grp in df.groupby("_hash", sort=False):
        if grp["label"].nunique() > 1:
            conflict_removed += len(grp)
            conflicts.append(grp["claim_text"].iloc[0][:120])
            continue
        keep_idx.append(grp.index[0])
        exact_removed += len(grp) - 1
    df = df.loc[sorted(keep_idx)].copy()

    # Pass 2: near-duplicates within the same category (cosine > threshold)
    near_removed, near_conflict_removed = 0, 0
    keep_idx = []
    for _, grp in df.groupby("category", sort=False):
        idx = list(grp.index)
        if len(idx) < 2:
            keep_idx.extend(idx)
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(grp["_norm"])
        sim = (X @ X.T).tocoo()
        parent = list(range(len(idx)))
        for i, j, s in zip(sim.row, sim.col, sim.data):
            if i < j and s > NEAR_DUP_THRESHOLD:
                ri, rj = _find(parent, i), _find(parent, j)
                if ri != rj:
                    parent[rj] = ri
        clusters = {}
        for k in range(len(idx)):
            clusters.setdefault(_find(parent, k), []).append(idx[k])
        for members in clusters.values():
            if len(members) == 1:
                keep_idx.append(members[0])
                continue
            if df.loc[members, "label"].nunique() > 1:
                near_conflict_removed += len(members)
                conflicts.append(df.loc[members[0], "claim_text"][:120])
                continue
            keep_idx.append(members[0])
            near_removed += len(members) - 1
    df = df.loc[sorted(keep_idx)].drop(columns=["_norm", "_hash"])

    log["exact_duplicates_removed"] = int(exact_removed)
    log["near_duplicates_removed"] = int(near_removed)
    log["conflicting_verdicts_removed"] = int(conflict_removed + near_conflict_removed)
    log["conflicting_claims_examples"] = conflicts[:20]
    return df


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
OPTIONAL_COLS = ["claim_id", "source", "language_tag", "category", "date", "url"]


def resolve_inputs(inputs, raw_dir, include_sample):
    if inputs:
        paths = []
        for p in inputs:
            paths.extend(sorted(glob.glob(p)) or [p])
        return paths
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    real = [p for p in paths if os.path.basename(p) != SAMPLE_FILENAME]
    if real and not include_sample:
        print(f"Real data found - excluding {SAMPLE_FILENAME}. Use --include-sample to keep it.")
        return real
    return paths


def load_raw(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            print(f"Skipping missing/empty file: {p}")
            continue
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        if os.path.basename(p).startswith("needs_review"):
            # Scraper review files: only rows a team member has filled in are used
            if "claim_text" not in df.columns and "claim_guess" in df.columns:
                df = df.rename(columns={"claim_guess": "claim_text"})
            df = df[df.get("verdict", pd.Series(dtype=str)).astype(str).str.strip() != ""]
            if df.empty:
                print(f"Skipping {p}: no verdicts filled in yet")
                continue
        if "verdict" not in df.columns and "label" in df.columns:
            df["verdict"] = df["label"]
        missing = {"claim_text", "verdict"} - set(df.columns)
        if missing:
            raise ValueError(f"{p} is missing required columns: {missing}")
        for c in OPTIONAL_COLS:
            if c not in df.columns:
                df[c] = ""
        if (df["source"] == "").all():
            df["source"] = "Unknown"
        df["source_file"] = os.path.basename(p)
        frames.append(df[["claim_text", "verdict", "source_file"] + OPTIONAL_COLS])
        print(f"Loaded {len(df):5d} rows from {p}")
    if not frames:
        raise ValueError("No input rows found. Add CSVs to data/raw/ first.")
    return pd.concat(frames, ignore_index=True)


def _standardise_source(s) -> str:
    s0 = str(s).strip()
    low = s0.lower()
    if "dubawa" in low:
        return "Dubawa"
    if "africa check" in low or "africacheck" in low:
        return "Africa Check"
    return s0 or "Unknown"


# ---------------------------------------------------------------------------
# 6. Split
# ---------------------------------------------------------------------------
def stratified_split(df, log):
    def _split(data, test_size, key):
        return train_test_split(data, test_size=test_size,
                                stratify=data[key] if key else None,
                                random_state=RANDOM_STATE)

    df = df.copy()
    df["_strat_ls"] = df["label"] + "|" + df["source"]
    for key, desc in [("_strat_ls", "label + source"), ("label", "label"), (None, "random")]:
        try:
            train, temp = _split(df, 0.20, key)
            val, test = _split(temp, 0.50, key)
            log["split_stratified_on"] = desc
            print(f"Split stratified on: {desc}")
            return [d.drop(columns=["_strat_ls"]) for d in (train, val, test)]
        except ValueError:
            continue
    raise ValueError("Could not split the data - too few rows.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(inputs=None, raw_dir="data/raw", outdir="data", include_sample=False, stem=True):
    os.makedirs(outdir, exist_ok=True)
    log = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    paths = resolve_inputs(inputs, raw_dir, include_sample)
    log["input_files"] = [os.path.basename(p) for p in paths]
    log["is_sample_only"] = all(os.path.basename(p) == SAMPLE_FILENAME for p in paths)

    df = load_raw(paths)
    log["rows_loaded"] = int(len(df))

    df["claim_text"] = df["claim_text"].astype(str).str.strip()
    df = df[df["claim_text"].str.len() > 0].copy()
    df["source"] = df["source"].apply(_standardise_source)

    # 1. Label harmonisation
    df["label"] = df["verdict"].apply(harmonize_label)
    unmapped = df.loc[df["label"].isna(), "verdict"].str.lower().value_counts().head(25)
    log["dropped_unverifiable_or_unmapped"] = int(df["label"].isna().sum())
    log["top_dropped_verdicts"] = {str(k)[:60]: int(v) for k, v in unmapped.items()}
    df = df.dropna(subset=["label"]).copy()
    print(f"Label harmonisation: dropped {log['dropped_unverifiable_or_unmapped']} rows "
          f"(unverifiable / unmapped). Remaining: {len(df)}")
    if len(df) == 0:
        raise ValueError("No rows left after label harmonisation. Check VERDICT_MAP.")

    # 2. Language identification pass
    df["language_tag_given"] = df["language_tag"].str.strip().str.lower()
    df["language_tag"] = df["claim_text"].apply(detect_language)
    given = df[df["language_tag_given"].isin(["en", "pcm"])]
    if len(given):
        agree = float((given["language_tag_given"] == given["language_tag"]).mean())
        log["language_id_agreement_with_given_tags"] = round(agree, 4)
        log["language_id_agreement_n"] = int(len(given))
        print(f"Language-ID agreement with provided tags: {agree:.1%} (n={len(given)})")

    # 3. Category and group tags (a provided category is kept)
    tags = df["claim_text"].apply(tag_groups).apply(pd.Series)
    provided_cat = df["category"].str.strip().str.lower()
    df["category"] = np.where(provided_cat.isin(list(CATEGORY_KEYWORDS) + ["other"]),
                              provided_cat, tags["category"])
    for c in ["region", "party_ref", "religion_ref", "ethnic_ref"]:
        df[c] = tags[c]

    # 4. Duplicates and conflicting verdicts
    before = len(df)
    df = remove_duplicates(df, log)
    print(f"Deduplication: {before} -> {len(df)} "
          f"(exact {log['exact_duplicates_removed']}, near {log['near_duplicates_removed']}, "
          f"conflicting verdicts {log['conflicting_verdicts_removed']})")

    # 5. Cleaning
    df["clean_text"] = [clean_text(t, lang=l, stem=stem)
                        for t, l in zip(df["claim_text"], df["language_tag"])]
    empty = int((df["clean_text"].str.strip() == "").sum())
    df = df[df["clean_text"].str.strip() != ""].copy()
    log["empty_after_cleaning_removed"] = empty

    log["final_rows"] = int(len(df))
    log["label_counts"] = df["label"].value_counts().to_dict()
    log["language_counts"] = df["language_tag"].value_counts().to_dict()
    log["category_counts"] = df["category"].value_counts().to_dict()
    log["source_counts"] = df["source"].value_counts().to_dict()
    log["region_counts"] = df["region"].value_counts().to_dict()
    print("\nLabels:", log["label_counts"])
    print("Languages:", log["language_counts"])
    print("Categories:", log["category_counts"])
    print("Sources:", log["source_counts"])

    if df["label"].nunique() < 2:
        raise ValueError("Only one class left - need both Real and Fake claims.")

    # 6. Split
    train, val, test = stratified_split(df, log)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part = part.assign(split=name)
        part.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    full = pd.concat([train.assign(split="train"), val.assign(split="val"),
                      test.assign(split="test")])
    os.makedirs(outdir, exist_ok=True)
    full.to_csv(os.path.join(outdir, "claims_clean.csv"), index=False)
    log["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}
    with open(os.path.join(outdir, "preprocess_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"\ntrain={len(train)} val={len(val)} test={len(test)}")
    print(f"Saved {outdir}/train.csv, val.csv, test.csv, claims_clean.csv, preprocess_log.json")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", help="CSV files or globs (default: data/raw/*.csv)")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--include-sample", action="store_true")
    ap.add_argument("--no-stem", action="store_true", help="disable English stemming")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    run(a.inputs, a.raw_dir, a.outdir, a.include_sample, stem=not a.no_stem)
