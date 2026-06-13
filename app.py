"""
app.py
------
Interactive Streamlit application for HYBRID fake-news verification.

It combines:
  * the thesis ML model (TF-IDF + supervised classifier) -> content/style signal
  * live web search of reputable outlets                  -> evidence signal
and shows a verdict plus the sources it found, so the user can judge.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

from verify import assess

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

VERDICT_STYLE = {
    "REAL (confirmed)":          ("success", "✅"),
    "REAL":                      ("success", "✅"),
    "FAKE":                      ("error",   "🚩"),
    "FAKE? (evidence disagrees)":("warning", "⚠️"),
    "MODEL NOT TRAINED":         ("info",    "❔"),
}


def main():
    st.set_page_config(page_title="Fake News Verifier", page_icon="📰", layout="centered")
    st.title("📰 Fake News Verifier")
    st.caption("Hybrid system — ML content model (TF-IDF) is the core; web search is a cross-check")

    has_model = os.path.exists(os.path.join(MODELS_DIR, "model.joblib"))
    if not has_model:
        st.info(
            "No trained ML model found yet, so only the web-evidence check will run. "
            "Train the content model with `python train.py ...` to enable both signals."
        )

    st.sidebar.markdown("### How to use")
    st.sidebar.markdown(
        "Paste **one** of the following, then press **Verify**:\n\n"
        "- the full text of an article\n"
        "- a short claim / headline\n"
        "- a link (URL) to a news page\n\n"
        "The tool searches the web, checks reputable outlets, and returns a "
        "verdict with the sources it found."
    )
    st.sidebar.warning(
        "Needs an internet connection for the web search. "
        "Predictions are statistical — always read the sources yourself."
    )

    user_input = st.text_area(
        "News text, claim, or URL",
        height=200,
        placeholder="Paste an article, a headline/claim, or a https://... link",
    )

    if st.button("Verify", type="primary"):
        if not user_input.strip():
            st.warning("Please paste some text, a claim, or a URL first.")
            return

        with st.spinner("Searching the web and analyzing..."):
            result = assess(user_input)

        if "error" in result:
            st.error(result["error"])
            return

        # ---- ML model prediction (PRIMARY — the thesis core) ---------------
        ml = result.get("ml_fake_prob")
        if ml is not None:
            label = result.get("ml_label")
            st.subheader(f"🧠 Machine Learning prediction: {label}")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Fake probability", f"{ml*100:.0f}%")
                st.progress(min(max(ml, 0.0), 1.0))
            with c2:
                st.metric("Real probability", f"{(1-ml)*100:.0f}%")
            st.caption("This prediction comes from the TF-IDF + supervised "
                       "classifier trained in train.py (the core of the thesis).")

        # ---- Combined verdict banner (model + evidence) --------------------
        style, icon = VERDICT_STYLE.get(result["verdict"], ("info", "❔"))
        banner = getattr(st, style)
        banner(f"{icon}  **Verdict: {result['verdict']}**  ·  {result['confidence']}")
        st.write(result["reason"])

        st.markdown("##### 🔎 Web cross-check (supporting evidence)")
        st.metric("Reputable sources found", len(result["reputable_hits"]))

        if result.get("note"):
            st.caption(result["note"])
        st.caption(f"Search query used: “{result['query']}”")

        # ---- Reputable sources ---------------------------------------------
        if result["reputable_hits"]:
            st.subheader("Reputable coverage")
            for s in result["reputable_hits"]:
                tag = " · 🧪 fact-checker" if s["fact_checker"] else ""
                st.markdown(f"**[{s['title'] or s['domain']}]({s['url']})** "
                            f"— *{s['domain']}*{tag}")
                if s["snippet"]:
                    st.caption(s["snippet"][:300])

        # ---- All other results ---------------------------------------------
        other = [s for s in result["sources"] if not s["reputable"]]
        if other:
            with st.expander(f"Other results found ({len(other)})"):
                for s in other:
                    st.markdown(f"[{s['title'] or s['domain']}]({s['url']}) "
                                f"— *{s['domain']}*")
                    if s["snippet"]:
                        st.caption(s["snippet"][:200])

        if not result["sources"]:
            st.info("No web results were returned. Check your internet connection, "
                    "or try rephrasing the claim into a few clear keywords.")

        # ---- Extracted text (URL mode) -------------------------------------
        if result.get("extracted_text"):
            with st.expander("Article text extracted from the URL"):
                st.write(result["extracted_text"][:3000])

    st.markdown("---")
    st.caption(
        "Educational decision-support tool. It does not give final truth — it "
        "gathers evidence and a model signal to help you verify. Always confirm "
        "important claims with multiple independent, trusted sources."
    )


if __name__ == "__main__":
    main()
