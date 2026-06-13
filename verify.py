"""
verify.py
---------
Hybrid fake-news assessment that combines TWO signals:

  1. CONTENT signal  (the thesis ML model): a TF-IDF + supervised classifier
     judges whether the text *reads like* fake / sensational news.

  2. EVIDENCE signal (web verification): the claim is searched on the web,
     and we check whether reputable outlets corroborate it.

The two signals are merged into a single verdict:
     CREDIBLE / DISPUTED-OR-FAKE / UNVERIFIED
together with the list of sources, so the user can judge for themselves.

No API key is required — search uses DuckDuckGo via the `ddgs` package.

This maps to the thesis literature (Section 4.4, Figure 4.3): a hybrid of
content-based detection and evidence/fact-checking based detection.
"""

import os
import re
from urllib.parse import urlparse

import joblib

from preprocessing import clean_text

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# Outlets and fact-checkers we treat as relatively reliable corroboration.
REPUTABLE = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "npr.org", "aljazeera.com", "dw.com",
    "bloomberg.com", "ft.com", "economist.com", "cnn.com", "abcnews.go.com",
    "cbsnews.com", "nbcnews.com", "time.com", "forbes.com", "politico.com",
    "nature.com", "science.org", "who.int", "europa.eu", "un.org",
    # Albanian / regional outlets
    "top-channel.tv", "topchannel.tv", "ata.gov.al", "balkanweb.com",
    "reporter.al", "exit.al", "euronews.al",
}
FACT_CHECKERS = {
    "snopes.com", "politifact.com", "factcheck.org", "fullfact.org",
    "apnews.com", "reuters.com", "afp.com", "leadstories.com",
}


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def is_url(text: str) -> bool:
    text = text.strip()
    return text.startswith("http://") or text.startswith("https://")


def extract_from_url(url: str) -> str:
    """Download a web page and return its main article text. Best-effort."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.strip()) > 40:
                return text.strip()
    except Exception:
        pass

    # Fallback: requests + BeautifulSoup
    try:
        import requests
        from bs4 import BeautifulSoup
        html = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(p for p in paragraphs if len(p) > 40).strip()
    except Exception as exc:  # noqa: BLE001
        return f"__ERROR__ Could not read the URL: {exc}"


def build_query(text: str, max_words: int = 16) -> str:
    """
    Turn the input into a short, search-friendly query.

    For a long article we use the first informative sentence (usually the
    headline / lede); for a short claim we use it as-is.
    """
    text = text.strip()
    # First reasonably long sentence
    sentences = re.split(r"(?<=[.!?])\s+", text)
    candidate = next((s for s in sentences if len(s.split()) >= 4), text)
    words = candidate.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)


# --------------------------------------------------------------------------- #
# Signal 1 — ML content/style model
# --------------------------------------------------------------------------- #
_MODEL = None
_VECTORIZER = None
_MODEL_NAME = None


def _load_model():
    global _MODEL, _VECTORIZER, _MODEL_NAME
    if _MODEL is None:
        mp = os.path.join(MODELS_DIR, "model.joblib")
        vp = os.path.join(MODELS_DIR, "vectorizer.joblib")
        if os.path.exists(mp) and os.path.exists(vp):
            _MODEL = joblib.load(mp)
            _VECTORIZER = joblib.load(vp)
            np = os.path.join(MODELS_DIR, "model_name.txt")
            _MODEL_NAME = open(np).read().strip() if os.path.exists(np) else "model"
    return _MODEL, _VECTORIZER, _MODEL_NAME


def ml_style_score(text: str):
    """Return P(fake) in [0,1] from the content model, or None if unavailable."""
    model, vectorizer, _ = _load_model()
    if model is None:
        return None
    import numpy as np
    X = vectorizer.transform([clean_text(text)])
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0][1])
    if hasattr(model, "decision_function"):
        margin = float(model.decision_function(X)[0])
        return float(1.0 / (1.0 + np.exp(-margin)))
    return float(model.predict(X)[0])


# --------------------------------------------------------------------------- #
# Signal 2 — Web evidence
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def search_web(query: str, max_results: int = 8):
    """
    Search the web + news for the query. Returns a list of dicts:
        {title, url, snippet, domain, reputable, fact_checker}
    Requires internet. Returns [] (and never crashes) on failure.
    """
    results = []
    seen = set()
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            raw = []
            try:
                raw += list(ddgs.news(query, max_results=max_results))
            except Exception:
                pass
            try:
                raw += list(ddgs.text(query, max_results=max_results))
            except Exception:
                pass

            for r in raw:
                url = r.get("url") or r.get("href") or ""
                if not url or url in seen:
                    continue
                seen.add(url)
                dom = _domain(url)
                results.append({
                    "title": r.get("title", "").strip(),
                    "url": url,
                    "snippet": (r.get("body") or r.get("excerpt") or "").strip(),
                    "domain": dom,
                    "reputable": dom in REPUTABLE,
                    "fact_checker": dom in FACT_CHECKERS,
                })
    except Exception:
        return []
    return results


# --------------------------------------------------------------------------- #
# Combine into a verdict
# --------------------------------------------------------------------------- #
def assess(raw_input: str):
    """
    Main entry point. Accepts article text, a short claim, or a URL.
    Returns a result dict consumed by the Streamlit app.
    """
    note = ""
    if is_url(raw_input):
        text = extract_from_url(raw_input)
        if text.startswith("__ERROR__"):
            return {"error": text.replace("__ERROR__", "").strip()}
        note = "Text was extracted automatically from the URL."
    else:
        text = raw_input.strip()

    if len(text.split()) < 3:
        return {"error": "Please provide more text (at least a full sentence)."}

    query = build_query(text)
    ml = ml_style_score(text)            # PRIMARY signal (the thesis ML model)
    sources = search_web(query)          # SUPPORTING cross-check
    reputable_hits = [s for s in sources if s["reputable"]]
    fact_hits = [s for s in sources if s["fact_checker"]]
    n_rep = len(reputable_hits)

    # ======================================================================= #
    # DECISION = ML model first, web evidence as a cross-check.
    #
    # The Machine Learning classifier is the core of the thesis, so it makes
    # the primary prediction (FAKE vs REAL). The web search is then used only
    # to CONFIRM or FLAG that prediction with external evidence.
    # ======================================================================= #
    if ml is None:
        # Model not trained yet -> fall back to evidence only.
        ml_label = None
        verdict = "MODEL NOT TRAINED"
        reason = ("The ML model has not been trained yet, so only the web "
                  "cross-check ran. Train it with `python train.py ...` to get "
                  "the model-based prediction (the core of the thesis).")
        confidence = "n/a"
    else:
        ml_label = "FAKE" if ml >= 0.5 else "REAL"
        # Confidence of the model itself (distance from the 0.5 boundary).
        margin = abs(ml - 0.5) * 2          # 0 .. 1
        conf_word = ("high" if margin >= 0.6 else
                     "medium" if margin >= 0.3 else "low")

        if ml_label == "FAKE":
            verdict = "FAKE"
            reason = (f"The ML model classifies this as FAKE "
                      f"(fake-probability {ml*100:.0f}%).")
            if n_rep >= 2:
                # Strong external contradiction -> soften, trust evidence.
                verdict = "FAKE? (evidence disagrees)"
                reason += (f" However, {n_rep} reputable outlets cover this topic, "
                           f"which contradicts the model — the model may be reacting "
                           f"to sensational style. Read the sources before deciding.")
                confidence = "model: " + conf_word + " · evidence disagrees"
            elif n_rep == 0:
                reason += (" No reputable outlet corroborates it either, which is "
                           "consistent with the model.")
                confidence = "model: " + conf_word + " · evidence agrees"
            else:
                reason += (f" {n_rep} reputable outlet(s) mention the topic — check "
                           "them below.")
                confidence = "model: " + conf_word
        else:  # ml_label == "REAL"
            verdict = "REAL"
            reason = (f"The ML model classifies this as REAL "
                      f"(fake-probability {ml*100:.0f}%).")
            if n_rep >= 2:
                verdict = "REAL (confirmed)"
                reason += (f" {n_rep} reputable outlets also cover this topic, which "
                           "supports the model.")
                confidence = "model: " + conf_word + " · evidence agrees"
            elif n_rep == 0:
                reason += (" But no reputable outlet was found to confirm it, so "
                           "treat it with some caution.")
                confidence = "model: " + conf_word + " · no corroboration"
            else:
                reason += (f" {n_rep} reputable outlet(s) found — see below.")
                confidence = "model: " + conf_word

    if fact_hits:
        reason += (f" Note: {len(fact_hits)} fact-checking site(s) appear in the "
                   "results — check them directly below.")

    return {
        "input_kind": "url" if is_url(raw_input) else "text",
        "note": note,
        "query": query,
        "ml_fake_prob": ml,
        "ml_label": ml_label,
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
        "sources": sources,
        "reputable_hits": reputable_hits,
        "fact_hits": fact_hits,
        "extracted_text": text if is_url(raw_input) else None,
    }


if __name__ == "__main__":
    # offline self-test of the non-network parts
    print("is_url:", is_url("https://example.com/news"))
    print("query:", build_query(
        "Scientists at NASA confirm that aliens control the weather. "
        "The agency released documents today."))
    print("ml score (demo model):",
          ml_style_score("SHOCKING miracle pill melts fat instantly doctors hate it"))
    print("domain:", _domain("https://www.reuters.com/world/abc"))
