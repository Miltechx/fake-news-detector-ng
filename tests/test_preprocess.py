import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import (clean_text, detect_language, harmonize_label,  # noqa: E402
                        remove_duplicates, scrub_pii, tag_groups)


# ---- label harmonisation (3.4.2.3) ----
def test_harmonize_real():
    for v in ["True", "VERIFIED", "correct", "Mostly True", "mostly-true", "Mostly Correct"]:
        assert harmonize_label(v) == "Real"


def test_harmonize_fake():
    for v in ["False", "misleading", "EXAGGERATED", "mostly_false", "Fake", "Incorrect",
              "False!", "Partly False"]:
        assert harmonize_label(v) == "Fake"


def test_harmonize_sentence_ratings():
    assert harmonize_label("False. The photo is from 2019 in Ghana") == "Fake"
    assert harmonize_label("Misleading: the figure is outdated") == "Fake"
    assert harmonize_label("True - INEC confirmed this") == "Real"


def test_harmonize_drops_unverifiable():
    for v in ["Unverifiable", "Unclear", "", "Unproven", "Half true", "Satire", None]:
        assert harmonize_label(v) is None


def test_harmonize_does_not_confuse_prefixes():
    assert harmonize_label("Truecaller app") is None  # 'true' must be a whole word


# ---- cleaning (3.4.2.4) ----
def test_clean_preserves_numbers_in_words():
    out = clean_text("covid19 and 5g rollout")
    assert "covid19" in out.split() and "5g" in out.split()


def test_clean_drops_standalone_digits():
    assert "2023" not in clean_text("the year 2023 was")


def test_clean_removes_html_and_emojis():
    out = clean_text("<p>Fake news \U0001F621 here</p>")
    assert "<" not in out and "\U0001F621" not in out
    assert "pfake" not in out and "herep" not in out


def test_clean_removes_urls_and_personal_data():
    out = clean_text("Call 08031234567 or mail john@example.com see https://x.co/a @handle")
    for bad in ["0803", "john", "example", "https", "handle"]:
        assert bad not in out


def test_clean_keeps_negation():
    assert "not" in clean_text("INEC did not announce results", lang="en").split()


def test_clean_keeps_pidgin_content_words():
    out = clean_text("Wahala dey for Lagos, sabi pipu")
    assert "wahala" in out
    assert "sabi" in out


def test_stemming_only_for_english():
    assert "elect" in clean_text("elections", lang="en")
    assert "elections" in clean_text("elections", lang="pcm")


def test_scrub_pii():
    s = scrub_pii("Reach me on +2348031234567 or a@b.com @me")
    assert "[phone]" in s and "[email]" in s and "[user]" in s


# ---- language identification ----
def test_detect_language():
    assert detect_language("Wahala dey for Lagos") == "pcm"
    assert detect_language("Fuel subsidy don finish for Nigeria") == "pcm"
    assert detect_language("Hot water fit cure malaria") == "pcm"
    assert detect_language("The president signed the budget yesterday") == "en"


# ---- group tags ----
def test_tag_groups():
    g = tag_groups("APC governor in Kano says vaccines cause infertility")
    assert g["region"] == "North-West"
    assert g["party_ref"] == "APC"
    assert g["category"] in {"politics", "health"}
    assert tag_groups("Nothing specific here")["region"] == "National/Unspecified"


# ---- duplicates and conflicting verdicts ----
def test_dedup_exact_and_conflict():
    df = pd.DataFrame({
        "claim_text": ["INEC bans BVAS", "inec bans bvas!", "Naira is gone", "naira is GONE",
                       "Lagos floods"],
        "label": ["Fake", "Fake", "Fake", "Real", "Real"],
        "category": ["politics", "politics", "other", "other", "other"],
    })
    log = {}
    out = remove_duplicates(df, log)
    assert log["exact_duplicates_removed"] == 1
    assert log["conflicting_verdicts_removed"] == 2
    assert len(out) == 2


def test_near_duplicates_removed():
    df = pd.DataFrame({
        "claim_text": ["The CBN has recalled all old naira notes permanently today",
                       "The CBN has recalled all old naira notes permanently today now",
                       "Drinking hot water cures malaria"],
        "label": ["Fake", "Fake", "Fake"],
        "category": ["other", "other", "other"],
    })
    log = {}
    out = remove_duplicates(df, log)
    assert log["near_duplicates_removed"] == 1
    assert len(out) == 2
