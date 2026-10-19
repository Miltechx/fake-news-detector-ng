"""
app.py - Fake News Detector for Nigerian Online Media (Streamlit)
=================================================================
Stage 5 / Presentation tier (Sections 3.3.2, 3.3.5 - 3.3.7).

  Check a claim   FR1 text, FR2 URL, FR3 preprocessing, FR4 label + 0-1 score,
                  FR7 pasted WhatsApp text, FR8 cross-reference, FR9 logging,
                  OFR1 LIME explanation, NFR6 graceful degradation
  Search claims   FR6 keyword / topic / source search
  Admin           FR10 retrain with new data, FR11 disaggregated reports,
                  NFR8 log-based auditing, NFR10 data refresh

Run locally:   streamlit run app.py
Secrets (optional), in .streamlit/secrets.toml or Streamlit Cloud settings:
  GOOGLE_FACTCHECK_API_KEY = "..."   enables live fact-checker cross-reference
  ADMIN_PASSWORD = "..."             enables the Admin tab
"""

import contextlib
import hmac
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

from preprocess import clean_text, detect_language, scrub_pii, tag_groups

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "data"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = BASE / "logs"
LOG_DB = LOG_DIR / "classifications.db"
THRESHOLD = 0.5  # P(Real) >= 0.5 -> Real (fixed decision threshold, FR4)

MODEL_FILES = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
LANG_NAMES = {"en": "English", "pcm": "Nigerian Pidgin / code-switched"}

st.set_page_config(page_title="Fake News Detector - Nigeria", layout="centered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_secret(name):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, "")


def _mtime(path):
    return path.stat().st_mtime if path.exists() else 0


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts(_stamp):
    vec_path = MODEL_DIR / "vectorizer.pkl"
    if not vec_path.exists():
        return None
    vectorizer = joblib.load(vec_path)
    models = {}
    for key in MODEL_FILES:
        p = MODEL_DIR / f"{key}.pkl"
        if p.exists():
            models[key] = joblib.load(p)
    best = (MODEL_DIR / "best_model_name.txt")
    best = best.read_text().strip() if best.exists() else next(iter(models), None)
    meta_path = MODEL_DIR / "training_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {"vectorizer": vectorizer, "models": models, "best": best, "meta": meta}


@st.cache_data(show_spinner=False)
def load_dataset(_stamp):
    p = DATA_DIR / "claims_clean.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, keep_default_na=False)


@st.cache_resource(show_spinner=False)
def dataset_matrix(_stamp, _vectorizer):
    ds = load_dataset(_mtime(DATA_DIR / "claims_clean.csv"))
    if ds.empty:
        return None
    return _vectorizer.transform(ds["clean_text"].astype(str))


def p_real(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index("Real")]


# ---- FR9 logging ----
def _db():
    LOG_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(LOG_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, input_type TEXT,
        submitted_text TEXT, url TEXT, language_tag TEXT, category TEXT, region TEXT,
        party_ref TEXT, religion_ref TEXT, ethnic_ref TEXT, model TEXT,
        label TEXT, score_p_real REAL, explanation TEXT)""")
    return con


def log_event(row):
    try:
        with contextlib.closing(_db()) as con:
            con.execute("""INSERT INTO classifications (created_at, input_type, submitted_text,
                url, language_tag, category, region, party_ref, religion_ref, ethnic_ref,
                model, label, score_p_real, explanation)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (row["created_at"], row["input_type"], row["submitted_text"],
                         row["url"], row["language_tag"], row["category"], row["region"],
                         row["party_ref"], row["religion_ref"], row["ethnic_ref"],
                         row["model"], row["label"], row["score_p_real"], row["explanation"]))
            con.commit()
    except sqlite3.Error as e:
        st.caption(f"Could not write to the classification log: {e}")


def read_log():
    if not LOG_DB.exists():
        return pd.DataFrame()
    with contextlib.closing(_db()) as con:
        return pd.read_sql_query("SELECT * FROM classifications ORDER BY id DESC", con)


# ---- FR2 URL input ----
def fetch_article(url):
    from bs4 import BeautifulSoup
    r = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0 (FakeNewsDetectorNG)"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
        t.decompose()

    def meta(prop):
        m = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
        return (m.get("content") or "").strip() if m else ""

    title = meta("og:title") or (soup.h1.get_text(" ", strip=True) if soup.h1 else "") or \
        (soup.title.get_text(strip=True) if soup.title else "")
    desc = meta("og:description") or meta("description")
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paras = [p for p in paras if len(p) > 60][:3]
    return title, desc, paras


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
art = load_artifacts(_mtime(MODEL_DIR / "vectorizer.pkl"))
st.title("Fake News Detector")
st.caption("Nigerian online media - English and Nigerian Pidgin. "
           "TF-IDF with Naive Bayes, Logistic Regression and Random Forest.")

if art is None or not art["models"]:
    st.error("No trained model found in models/. Run `python preprocess.py` then "
             "`python train_model.py`, and commit the models/ folder.")
    st.stop()

meta = art["meta"]
if meta.get("is_sample_only"):
    st.warning("Demo mode: these models were trained on the small sample file only, so "
               "predictions are not meaningful yet. Retrain on the full Dubawa / Africa Check "
               "dataset before evaluation.")

def run_check(mode, claim_in, url_in, model_key, show_lime, cross_ref):
    text, input_type = claim_in.strip(), "text"
    if mode == "Article URL":
        input_type = "url"
        if not url_in.strip().lower().startswith(("http://", "https://")):
            st.warning("Enter a full URL starting with http:// or https://")
            return
        try:
            with st.spinner("Fetching article..."):
                title, desc, paras = fetch_article(url_in.strip())
        except requests.RequestException as e:
            st.error(f"Could not fetch that page ({e.__class__.__name__}). "
                     "Paste the headline as text instead.")
            return
        text = " ".join(x for x in [title, desc] if x) or " ".join(paras)
        with st.expander("Text extracted from the article"):
            st.write(f"**Headline:** {title or '-'}")
            st.write(f"**Summary:** {desc or '-'}")
            for p in paras:
                st.write(p)

    if not text:
        st.warning("Please enter a claim.")
        return

    lang = detect_language(text)
    cleaned = clean_text(text, lang=lang)
    if not cleaned:
        st.warning("No usable content words after preprocessing. Try a longer claim.")
        return

    vec = art["vectorizer"].transform([cleaned])
    model = art["models"][model_key]
    score = float(p_real(model, vec)[0])
    label = "Real" if score >= THRESHOLD else "Fake"
    groups = tag_groups(text)

    st.divider()
    r1, r2 = st.columns(2)
    r1.metric("Prediction", "Likely genuine" if label == "Real" else "Likely fake")
    r2.metric("Credibility score", f"{score:.2f}")
    st.progress(score)
    st.caption("0 = highly likely fake, 1 = highly likely genuine. Threshold 0.5. "
               "This is a model score, not a verdict - see the Brier score in Admin "
               "for how well calibrated it is.")
    st.caption(f"Language: {LANG_NAMES.get(lang, lang)}  |  Topic: {groups['category']}  |  "
               f"Region: {groups['region']}  |  Model: {MODEL_FILES[model_key]}")

    with st.expander("Compare all three classifiers"):
        rows = []
        for k, m in art["models"].items():
            s = float(p_real(m, vec)[0])
            rows.append({"Classifier": MODEL_FILES[k],
                         "Prediction": "Real" if s >= THRESHOLD else "Fake",
                         "Credibility score": round(s, 3)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # FR8 - cross-reference with existing fact-checker verdicts
    if cross_ref:
        st.subheader("Existing fact-checks")
        ds = load_dataset(_mtime(DATA_DIR / "claims_clean.csv"))
        X_ds = dataset_matrix(_mtime(DATA_DIR / "claims_clean.csv"), art["vectorizer"])
        found = False
        if X_ds is not None and X_ds.shape[0]:
            sims = (X_ds @ vec.T).toarray().ravel()
            top = np.argsort(sims)[::-1][:3]
            hits = [(i, sims[i]) for i in top if sims[i] >= 0.35]
            if hits:
                found = True
                st.write("Similar claims already checked by Dubawa / Africa Check:")
                for i, s in hits:
                    row = ds.iloc[i]
                    link = f" - [source]({row['url']})" if str(row.get("url", "")).startswith("http") else ""
                    st.write(f"- **{row['verdict']}** ({row['source']}, similarity {s:.2f}): "
                             f"{row['claim_text']}{link}")
        api_key = get_secret("GOOGLE_FACTCHECK_API_KEY")
        if api_key:
            try:
                from fetch_factcheck_api import lookup_claim
                live = lookup_claim(text, api_key)
                if live:
                    found = True
                    st.write("Live results from the Google Fact Check Tools API:")
                    for r_ in live:
                        st.write(f"- **{r_['verdict']}** ({r_['source']}): "
                                 f"{r_['claim_text']} - [read]({r_['url']})")
            except Exception:
                st.info("Live fact-check lookup is unavailable right now. "
                        "The model result above is unaffected.")
        if not found:
            st.caption("No matching fact-check found. The model prediction stands on its own.")

    # OFR1 - LIME explanation
    explanation = []
    if show_lime:
        try:
            from lime.lime_text import LimeTextExplainer
            with st.spinner("Generating explanation..."):
                explainer = LimeTextExplainer(class_names=list(model.classes_),
                                              random_state=42)

                def predict_fn(texts):
                    return model.predict_proba(art["vectorizer"].transform(
                        [clean_text(t, lang=lang) for t in texts]))

                target = list(model.classes_).index(label)
                n_feat = min(10, max(3, len(text.split())))
                exp = explainer.explain_instance(text, predict_fn, num_features=n_feat,
                                                 labels=(target,), num_samples=1000)
                explanation = exp.as_list(label=target)
            st.subheader("Why the model decided this")
            if explanation:
                df_exp = pd.DataFrame(explanation, columns=["Word", "Weight"])
                df_exp["Effect"] = np.where(df_exp["Weight"] > 0,
                                            f"pushes towards {label}",
                                            f"pushes away from {label}")
                st.dataframe(df_exp.round(3), hide_index=True, use_container_width=True)
                st.bar_chart(df_exp.set_index("Word")["Weight"], horizontal=True)
            else:
                st.caption("No influential words found.")
        except ImportError:
            st.info("LIME is not installed, so no explanation is shown.")

    log_event({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_type": input_type, "submitted_text": scrub_pii(text)[:2000],
        "url": url_in.strip(), "language_tag": lang, **groups,
        "model": MODEL_FILES[model_key], "label": label, "score_p_real": score,
        "explanation": json.dumps(explanation),
    })
    st.caption("Decision-support tool only. It does not remove or censor content; "
               "check important claims with Dubawa or Africa Check.")


tab_check, tab_search, tab_about, tab_admin = st.tabs(
    ["Check a claim", "Search claims", "About", "Admin"])

# ---------------------------------------------------------------------------
# Check a claim
# ---------------------------------------------------------------------------
with tab_check:
    model_keys = list(art["models"])
    default_idx = model_keys.index(art["best"]) if art["best"] in model_keys else 0

    mode = st.radio("Input", ["Paste text", "Article URL"], horizontal=True,
                    help="Text can come from anywhere, including WhatsApp forwards.")
    with st.form("check_form"):
        if mode == "Paste text":
            claim_in = st.text_area("Claim, headline or message", height=130,
                                    placeholder="e.g. Dem say CBN don recall all old naira notes")
            url_in = ""
        else:
            url_in = st.text_input("Article URL", placeholder="https://...")
            claim_in = ""
        c1, c2 = st.columns(2)
        model_key = c1.selectbox("Classifier", model_keys, index=default_idx,
                                 format_func=lambda k: MODEL_FILES[k] +
                                 (" (best on validation)" if k == art["best"] else ""))
        show_lime = c2.checkbox("Explain with LIME", value=True)
        cross_ref = c2.checkbox("Cross-reference fact-checkers", value=True)
        submitted = st.form_submit_button("Check claim", type="primary")

    if submitted:
        run_check(mode, claim_in, url_in, model_key, show_lime, cross_ref)

# ---------------------------------------------------------------------------
# Search claims (FR6)
# ---------------------------------------------------------------------------
with tab_search:
    ds = load_dataset(_mtime(DATA_DIR / "claims_clean.csv"))
    st.write("Search claims already fact-checked by Dubawa and Africa Check, "
             "and claims checked through this app.")
    q = st.text_input("Keyword", placeholder="e.g. naira, INEC, vaccine")
    include_app = st.checkbox("Include claims checked through this app")
    if ds.empty:
        st.info("No dataset loaded yet.")
    else:
        f1, f2, f3 = st.columns(3)
        cats = f1.multiselect("Topic", sorted(ds["category"].unique()))
        srcs = f2.multiselect("Source", sorted(ds["source"].unique()))
        labs = f3.multiselect("Label", ["Real", "Fake"])
        res = ds.copy()
        if q:
            res = res[res["claim_text"].str.contains(q, case=False, regex=False)]
        if cats:
            res = res[res["category"].isin(cats)]
        if srcs:
            res = res[res["source"].isin(srcs)]
        if labs:
            res = res[res["label"].isin(labs)]
        st.caption(f"{len(res)} fact-checked claims")
        st.dataframe(res[["claim_text", "label", "verdict", "source", "category", "date", "url"]]
                     .head(300), hide_index=True, use_container_width=True,
                     column_config={"url": st.column_config.LinkColumn("url")})
    if include_app:
        lg = read_log()
        if not lg.empty:
            if q:
                lg = lg[lg["submitted_text"].str.contains(q, case=False, regex=False)]
            st.caption(f"{len(lg)} claims checked through the app")
            st.dataframe(lg[["created_at", "submitted_text", "label", "score_p_real",
                             "category", "model"]].head(300),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("No claims have been checked through the app yet.")

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(f"""
This prototype gives a fast first-line credibility assessment of claims circulating in
Nigerian online media, in English and Nigerian Pidgin. It is a **decision-support tool**
for the public and for fact-checkers - not an automated content-removal system.

**How it works:** the claim is cleaned (URLs, emojis and personal data removed,
Nigeria-adapted stopwords), its language is identified, it is turned into TF-IDF
features and scored by the selected classifier. LIME then shows which words drove
the result.

**Training data:** published verdicts from Dubawa and Africa Check (via their archives
and the Google Fact Check Tools API), harmonised to Real (True) and Fake (False,
Misleading). Unverifiable claims are excluded. All credit for the verdicts belongs to
those organisations.

**Current models:** trained {meta.get('trained_at', 'n/a')} on {meta.get('n_train', '?')}
training claims ({meta.get('n_val', '?')} validation, {meta.get('n_test', '?')} test).

**Limitations:** the model judges wording, not facts. New topics it has not seen, satire,
and very short claims can be misjudged. No personal data is stored; submitted text is
scrubbed of emails, phone numbers and handles before logging.
""")

# ---------------------------------------------------------------------------
# Admin (FR10, FR11, NFR8, NFR10)
# ---------------------------------------------------------------------------
with tab_admin:
    admin_pw = get_secret("ADMIN_PASSWORD")
    if not admin_pw:
        st.info("Admin is disabled. Set ADMIN_PASSWORD in the app secrets to enable it.")
    else:
        if not st.session_state.get("admin_ok"):
            pw = st.text_input("Admin password", type="password")
            if pw and hmac.compare_digest(pw, admin_pw):
                st.session_state["admin_ok"] = True
                st.rerun()
            elif pw:
                st.error("Wrong password.")
        if st.session_state.get("admin_ok"):
            st.subheader("Model performance")
            ms = MODEL_DIR / "metrics_summary.csv"
            if ms.exists():
                st.dataframe(pd.read_csv(ms), hide_index=True, use_container_width=True)
                st.download_button("Download metrics summary", ms.read_bytes(),
                                   "metrics_summary.csv", "text/csv")
            figdir = MODEL_DIR / "figures"
            if figdir.exists():
                for img in sorted(figdir.glob("*.png")):
                    st.image(str(img), use_container_width=True)

            st.subheader("Disaggregated performance (bias audit)")
            dg = MODEL_DIR / "disaggregated_report.csv"
            if dg.exists():
                d = pd.read_csv(dg, keep_default_na=False)
                a1, a2, a3 = st.columns(3)
                gcol = a1.selectbox("Group by", sorted(d["group_column"].unique()))
                split = a2.selectbox("Split", ["test", "val"])
                mdl = a3.selectbox("Model", sorted(d["model"].unique()))
                view = d[(d["group_column"] == gcol) & (d["split"] == split) & (d["model"] == mdl)]
                st.dataframe(view.drop(columns=["group_column", "split", "model"]),
                             hide_index=True, use_container_width=True)
                st.caption("false_positive_rate = genuine claims wrongly flagged as Fake. "
                           "Groups with fewer than 10 claims are flagged as small samples.")
                st.download_button("Download full disaggregated report", dg.read_bytes(),
                                   "disaggregated_report.csv", "text/csv")

            st.subheader("Classification log")
            lg = read_log()
            if lg.empty:
                st.caption("No classifications logged yet.")
            else:
                gcol2 = st.selectbox("Summarise log by",
                                     ["language_tag", "category", "region", "party_ref",
                                      "religion_ref", "ethnic_ref", "model", "input_type"])
                summ = lg.groupby([gcol2, "label"]).size().unstack(fill_value=0)
                summ["mean_score_p_real"] = lg.groupby(gcol2)["score_p_real"].mean().round(3)
                st.dataframe(summ, use_container_width=True)
                st.dataframe(lg.head(200), hide_index=True, use_container_width=True)
                st.download_button("Download log (CSV)", lg.to_csv(index=False).encode(),
                                   "classification_log.csv", "text/csv")

            st.subheader("Refresh data and retrain")
            st.caption("On Streamlit Community Cloud, files written here are lost when the app "
                       "restarts. For a permanent update, retrain in Colab and push the new "
                       "models/ and data/ folders to GitHub.")
            raw_files = sorted(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
            if raw_files:
                st.dataframe(pd.DataFrame([{"file": f.name,
                                            "rows": max(0, sum(1 for _ in open(f, encoding="utf-8")) - 1)}
                                           for f in raw_files]),
                             hide_index=True, use_container_width=True)
            up = st.file_uploader("Add newly labelled claims (CSV with claim_text, verdict, "
                                  "source columns)", type="csv")
            if up is not None:
                try:
                    new = pd.read_csv(up, dtype=str, keep_default_na=False)
                    missing = {"claim_text", "verdict"} - set(new.columns)
                    if missing:
                        st.error(f"Missing columns: {missing}")
                    else:
                        RAW_DIR.mkdir(parents=True, exist_ok=True)
                        name = f"uploaded_{datetime.now():%Y%m%d_%H%M%S}.csv"
                        new.to_csv(RAW_DIR / name, index=False)
                        st.success(f"Saved {len(new)} rows as data/raw/{name}")
                except Exception as e:
                    st.error(f"Could not read file: {e}")

            b1, b2 = st.columns(2)
            api_key = get_secret("GOOGLE_FACTCHECK_API_KEY")
            if b1.button("Fetch new claims from the API", disabled=not api_key):
                from fetch_factcheck_api import run as fetch_run
                buf = io.StringIO()
                with st.spinner("Fetching..."), contextlib.redirect_stdout(buf):
                    try:
                        fetch_run(api_key, out_path=str(RAW_DIR / "api_claims.csv"))
                    except Exception as e:
                        print(f"Error: {e}")
                st.code(buf.getvalue()[-4000:])
            if b2.button("Rebuild dataset and retrain", type="primary"):
                import preprocess
                import train_model
                buf = io.StringIO()
                with st.spinner("Preprocessing and training - this can take a few minutes..."), \
                        contextlib.redirect_stdout(buf):
                    try:
                        preprocess.run(raw_dir=str(RAW_DIR), outdir=str(DATA_DIR))
                        train_model.run(str(DATA_DIR), str(MODEL_DIR))
                        ok = True
                    except Exception as e:
                        print(f"Error: {e}")
                        ok = False
                st.code(buf.getvalue()[-6000:])
                if ok:
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success("Retrained. The app now uses the new models.")
