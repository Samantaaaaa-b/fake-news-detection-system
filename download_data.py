"""
download_data.py
----------------
Downloads the ISOT "Fake and Real News" dataset (the one cited in the thesis:
Ahmed, Traore & Saad, 2017) straight from a public mirror — no Kaggle account
needed. Saves Fake.csv and True.csv into ./data/.

Run:
    python download_data.py
Then train:
    python train.py --fake data/Fake.csv --true data/True.csv
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/main/data/"
FILES = ["Fake.csv", "True.csv"]

os.makedirs("data", exist_ok=True)
for name in FILES:
    dest = os.path.join("data", name)
    print(f"Downloading {name} ... (this is a large file, please wait)")
    urllib.request.urlretrieve(BASE + name, dest)
    size = os.path.getsize(dest) / 1e6
    print(f"  saved {dest}  ({size:.0f} MB)")
print("\nDone. Now run:  python train.py --fake data/Fake.csv --true data/True.csv")
