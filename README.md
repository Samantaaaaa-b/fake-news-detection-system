# Fake News Detection Using Machine Learning

Implementim i plotë i sistemit të përshkruar në tezë: **TF-IDF + klasifikues të mësimit
të mbikëqyrur (scikit-learn) + aplikacion interaktiv Streamlit**. Kodi përputhet me
metodologjinë e tezës dhe mund të përdoret për të mbushur **Kapitullin V**.

## Struktura

```
fake-news-detection/
├── preprocessing.py     # pastrimi i tekstit me NLTK (tokenizim, stopwords, lemmatizim)
├── train.py             # trajnimi + vlerësimi i modeleve, ruan modelin më të mirë
├── app.py               # aplikacioni Streamlit (deployment interaktiv)
├── requirements.txt
├── data/                # vendos këtu dataset-in
├── models/              # model.joblib + vectorizer.joblib (krijohen nga train.py)
└── outputs/             # metrics.csv, confusion_matrix.png, roc_curve.png
```

## 1. Instalimi

```bash
pip install -r requirements.txt
```

(NLTK i shkarkon vetë `stopwords`, `wordnet`, `punkt` herën e parë.)

## 2. Dataset-i

Sistemi është trajnuar mbi **ISOT "Fake and Real News" Dataset** (~44,900 artikuj:
21,417 realë nga Reuters + 23,481 fake) — pikërisht dataset-i që citohet te teza
(Ahmed, Traore & Saad, 2017).

Mënyra më e lehtë për ta marrë (pa llogari Kaggle):

```bash
python download_data.py
```

Kjo shkarkon `Fake.csv` dhe `True.csv` te dosja `data/`.

> Modeli te `models/` është trajnuar tashmë mbi këtë dataset të plotë, pra aplikacioni
> punon menjëherë. Dataset-in e shkarkon vetëm nëse do ta ritrajnosh vetë.

Sistemi punon edhe me një CSV të vetëm me kolonë teksti + etikete (`label` mund të jetë
`0/1` ose `real/fake`).

## 3. Trajnimi

```bash
# Formati ISOT (dy skedarë)
python train.py --fake data/Fake.csv --true data/True.csv

# ose një CSV i vetëm
python train.py --csv data/sample_news.csv --text-col text --label-col label
```

Trajnon pesë modele — **Logistic Regression, Naïve Bayes, Linear SVM, Random Forest, KNN**
(seksioni 2.5 i tezës) — i vlerëson me **accuracy / precision / recall / F1 / ROC-AUC**
(seksioni 2.3), ruan modelin më të mirë në `models/` dhe grafikët (matrica e konfuzionit
dhe kurba ROC) në `outputs/`.

## 4. Aplikacioni

```bash
streamlit run app.py
```

Ngarkon modelin e trajnuar, e lejon përdoruesin të ngjisë një artikull, aplikon të njëjtin
preprocessing si në trajnim dhe shfaq parashikimin (FAKE / REAL) me një vlerë besueshmërie.

## Si lidhet me tezën

| Hapi i tezës | Skedari |
|---|---|
| Preprocessing (Kap. I, 2.1) | `preprocessing.py` |
| Vektorizim TF-IDF (Kap. I) | `train.py` |
| Ndarje 80/20 e stratifikuar (2.4) | `train.py` |
| Modelet e mbikëqyrura (2.5) | `train.py` |
| Metrikat e vlerësimit (2.3) | `train.py` → `outputs/metrics.csv`, grafikët |
| Deployment interaktiv | `app.py` |

## Verifikim hibrid (kërkim + fact-check në internet)

Përveç modelit ML, sistemi ka edhe një **shtresë verifikimi me dëshmi** (`verify.py`):
kur i jep një lajm, ai:

1. e merr tekstin (nga teksti i ngjitur, ose e nxjerr vetë nga një **link/URL**);
2. ndërton një pyetje kërkimi dhe **kërkon në internet** (DuckDuckGo, pa API key);
3. sheh sa **burime të besueshme** (Reuters, AP, BBC, Guardian, fact-checkers, etj.)
   e mbulojnë lajmin;
4. e kombinon me sinjalin ML dhe jep një **verdikt** + listën e burimeve:
   `LIKELY CREDIBLE / WEAKLY SUPPORTED / MIXED SIGNALS / UNVERIFIED / LIKELY FAKE`.

Kjo është një **qasje hibride** (përmbajtje + dëshmi) që lidhet me seksionin 4.4 dhe
Figurën 4.3 të tezës (qasjet e ndryshme për detektimin e fake news).

Përdorimi është i njëjti aplikacion:

```bash
streamlit run app.py
```

Te kutia mund të ngjisësh: **tekst të plotë**, **një pretendim/titull të shkurtër**,
ose **një link (https://...)**. Kërkon **lidhje interneti** për kërkimin online.

> Modeli ML mund të mungojë (nëse s'e ke trajnuar ende) — atëherë punon vetëm shtresa
> e verifikimit online. Të dyja bashkë japin rezultatin më të mirë.

## Shënim

Mjet edukativ vendimmarrjeje. Nuk jep "të vërtetën përfundimtare" — mbledh dëshmi dhe një
sinjal nga modeli që të të ndihmojë të verifikosh. Verifiko gjithmonë pretendimet e
rëndësishme me disa burime të pavarura e të besueshme.
