"""
train.py
--------
End-to-end training pipeline for the Fake News Detection system.

Steps (matching the thesis methodology):
    1. Load dataset  (Chapter I  - data collection & labeling)
    2. Clean text    (preprocessing.py)
    3. TF-IDF        (Chapter I  - vectorization)
    4. Train/test split, stratified 80/20  (Section 2.4)
    5. Train several supervised models      (Section 2.5):
           Logistic Regression, Multinomial Naive Bayes,
           Linear SVM, Random Forest, KNN
    6. Evaluate with accuracy / precision / recall / F1 / ROC-AUC  (Section 2.3)
    7. Save the best model + the fitted vectorizer  -> models/
    8. Save confusion matrix & ROC curve figures    -> outputs/
       (these fill the figure placeholders in Chapter V)

USAGE
-----
Option A - ISOT / Kaggle "Fake and Real News" dataset (two files):
    python train.py --fake data/Fake.csv --true data/True.csv

Option B - a single labeled CSV:
    python train.py --csv data/news.csv --text-col text --label-col label
    (label may be 0/1, or strings like "fake"/"real", "FAKE"/"REAL")

Run `python train.py --help` for all options.
"""

import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from preprocessing import clean_text

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
OUTPUTS_DIR = os.path.join(HERE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(args) -> pd.DataFrame:
    """Return a DataFrame with two columns: 'text' (str) and 'label' (0=real, 1=fake)."""
    if args.fake and args.true:
        fake = pd.read_csv(args.fake)
        true = pd.read_csv(args.true)

        def join_cols(df):
            # ISOT files have 'title' and 'text'; combine them if both exist.
            cols = [c for c in ("title", "text") if c in df.columns]
            if not cols:  # fall back to first text-like column
                cols = [df.columns[0]]
            return df[cols].fillna("").agg(" ".join, axis=1)

        fake_text = join_cols(fake)
        true_text = join_cols(true)
        df = pd.DataFrame(
            {
                "text": pd.concat([fake_text, true_text], ignore_index=True),
                "label": [1] * len(fake_text) + [0] * len(true_text),  # 1 = fake
            }
        )

    elif args.csv:
        raw = pd.read_csv(args.csv)
        if args.text_col not in raw.columns:
            raise ValueError(
                f"Text column '{args.text_col}' not found. "
                f"Available columns: {list(raw.columns)}"
            )
        text = raw[args.text_col].fillna("")
        label = _normalize_labels(raw[args.label_col])
        df = pd.DataFrame({"text": text, "label": label})

    else:
        raise SystemExit(
            "No dataset provided.\n"
            "  Use  --fake Fake.csv --true True.csv   (two-file ISOT format)\n"
            "  or   --csv news.csv --text-col text --label-col label"
        )

    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)
    # Shuffle so the two classes are mixed.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Map various label encodings to 1 = fake, 0 = real."""
    fake_words = {"fake", "false", "f", "1", "fake news"}
    real_words = {"real", "true", "t", "0", "reliable"}

    def to_int(v):
        s = str(v).strip().lower()
        if s in fake_words:
            return 1
        if s in real_words:
            return 0
        # numeric fallback
        try:
            return int(float(s))
        except ValueError:
            raise ValueError(f"Cannot interpret label value: {v!r}")

    return series.map(to_int)


# --------------------------------------------------------------------------- #
# 2-7. Train & evaluate
# --------------------------------------------------------------------------- #
def build_models():
    """The supervised classifiers discussed in Section 2.5 of the thesis."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def get_scores(model, X_test):
    """Return probability/score for the positive (fake) class for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def main():
    parser = argparse.ArgumentParser(description="Train fake news detector.")
    parser.add_argument("--fake", help="CSV of fake articles (ISOT Fake.csv)")
    parser.add_argument("--true", help="CSV of real articles (ISOT True.csv)")
    parser.add_argument("--csv", help="Single labeled CSV")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--label-col", default="label", help="Label column name")
    parser.add_argument("--max-features", type=int, default=20000,
                        help="Max TF-IDF vocabulary size")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction held out for testing")
    args = parser.parse_args()

    print("Loading data ...")
    df = load_data(args)
    n_fake = int((df["label"] == 1).sum())
    n_real = int((df["label"] == 0).sum())
    print(f"  {len(df)} articles total  ({n_fake} fake, {n_real} real)")

    print("Cleaning text (this can take a minute on large datasets) ...")
    df["clean"] = df["text"].map(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"],
        df["label"],
        test_size=args.test_size,
        stratify=df["label"],   # stratified split (Section 2.4)
        random_state=42,
    )

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True,
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    results = []
    best = {"name": None, "model": None, "f1": -1.0, "scores": None, "pred": None}

    print("\nTraining & evaluating models")
    print("-" * 72)
    for name, model in build_models().items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        scores = get_scores(model, Xte)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        auc = roc_auc_score(y_test, scores) if scores is not None else float("nan")

        results.append((name, acc, prec, rec, f1, auc))
        print(f"{name:<26} acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}")

        if f1 > best["f1"]:
            best.update(name=name, model=model, f1=f1, scores=scores, pred=pred)

    print("-" * 72)
    print(f"Best model: {best['name']}  (F1 = {best['f1']:.4f})\n")

    # ---- Save the metrics table (for Chapter V) ----------------------------
    metrics_df = pd.DataFrame(
        results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    ).sort_values("F1", ascending=False)
    metrics_path = os.path.join(OUTPUTS_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics table -> {metrics_path}")

    # ---- Detailed report for the best model --------------------------------
    print("\nClassification report (best model):")
    print(classification_report(y_test, best["pred"],
                                target_names=["REAL", "FAKE"], zero_division=0))

    # ---- Confusion matrix figure (Section 2.3) -----------------------------
    cm = confusion_matrix(y_test, best["pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion Matrix - {best['name']}")
    fig.tight_layout()
    cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {cm_path}")

    # ---- ROC curve figure (Section 2.3) ------------------------------------
    if best["scores"] is not None:
        fpr, tpr, _ = roc_curve(y_test, best["scores"])
        auc = roc_auc_score(y_test, best["scores"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"{best['name']} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"Saved ROC curve -> {roc_path}")

    # ---- Persist model + vectorizer ----------------------------------------
    joblib.dump(best["model"], os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    with open(os.path.join(MODELS_DIR, "model_name.txt"), "w") as fh:
        fh.write(best["name"])
    print(f"\nSaved model + vectorizer -> {MODELS_DIR}/")
    print("Done. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
