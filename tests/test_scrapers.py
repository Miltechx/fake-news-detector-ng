import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_factcheck_api import _rows_from_response  # noqa: E402
from scrape_factcheckers import (parse_africacheck_article,  # noqa: E402
                                 parse_africacheck_listing, parse_dubawa_post)


def test_parse_dubawa_post():
    content = ("<p><strong>Claim:</strong> A Facebook post claims Lagos has banned okada.</p>"
               "<p><strong>Verdict:</strong> False! The ban covers only some LGAs.</p>"
               "<h3>Full Text:</h3><p>...</p>")
    claim, verdict, title, _ = parse_dubawa_post("False! Lagos did not ban okada", content)
    assert claim.startswith("A Facebook post claims Lagos")
    assert verdict == "False"


def test_parse_dubawa_title_fallback():
    claim, verdict, _, _ = parse_dubawa_post("Misleading! Viral video of flood", "<p>Text</p>")
    assert verdict == "Misleading"


def test_parse_africacheck_listing():
    html = ('<a href="/fact-checks/fact-checks/no-nigeria-did-not-ban-x">a</a>'
            '<a href="https://africacheck.org/fact-checks/reports/some-report">b</a>'
            '<a href="/about-us">c</a>')
    links = parse_africacheck_listing(html)
    assert "https://africacheck.org/fact-checks/fact-checks/no-nigeria-did-not-ban-x" in links
    assert len(links) == 2


def test_parse_africacheck_jsonld():
    html = ('<html><h1>Title</h1><script type="application/ld+json">'
            '{"@graph":[{"@type":"ClaimReview","claimReviewed":"Nigeria has 300m people",'
            '"reviewRating":{"alternateName":"Incorrect"}}]}</script></html>')
    claim, verdict, title, _ = parse_africacheck_article(html)
    assert claim == "Nigeria has 300m people" and verdict == "Incorrect" and title == "Title"


def test_parse_africacheck_rating_image():
    html = ('<html><h1>T</h1><p>IN SHORT: A viral claim says X.</p>'
            '<img src="/files/Rating_Misleading.png" alt=""></html>')
    claim, verdict, _, _ = parse_africacheck_article(html)
    assert claim == "A viral claim says X." and verdict == "Misleading"


def test_api_publisher_filter():
    data = {"claims": [{"text": "Claim A", "claimReview": [
        {"publisher": {"name": "Dubawa", "site": "dubawa.org"}, "textualRating": "False",
         "url": "https://dubawa.org/a"},
        {"publisher": {"name": "Other", "site": "other.com"}, "textualRating": "True",
         "url": "https://other.com/a"}]}]}
    rows = _rows_from_response(data, ["dubawa.org", "africacheck.org"])
    assert len(rows) == 1 and rows[0]["source"] == "Dubawa"
