"""PALM (reference side only; no FE): extract, read the label tables, image sizes, disc-mask presence and polarity."""
import zipfile, json, collections
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); Z = ROOT / "data/eval/palm/PALM.zip"; EX = ROOT / "data/eval/palm/extracted"
if not (EX / "PALM").exists():
    with zipfile.ZipFile(Z) as z:
        for m in z.infolist():
            if "__MACOSX" not in m.filename and not m.filename.endswith(".DS_Store"): z.extract(m, EX)
    print("extracted to", EX)
P = EX / "PALM"
for split in ("Training", "Validation", "Testing"):
    print(f"\n===== {split}")
    for x in sorted((P / split).glob("*.xlsx")):
        df = pd.read_excel(x); print(f"--- {x.name}: {df.shape[0]} rows, columns {list(df.columns)}"); print(df.head(3).to_string()[:600])
    imgs = sorted((P / split / "Images").glob("*")); masks = {m.stem: m for m in (P / split / "Disc Masks").glob("*")}
    sizes = collections.Counter(Image.open(i).size for i in imgs); print("image sizes:", dict(sizes))
    print("images without disc mask:", len([i for i in imgs if i.stem not in masks]), "| by prefix:", dict(collections.Counter(i.stem[0] for i in imgs if i.stem not in masks)))
    print("prefix counts:", dict(collections.Counter(i.stem[0] for i in imgs)))
    for i in [i for i in imgs if i.stem in masks][:3]:
        a = np.asarray(Image.open(masks[i.stem]).convert("L")); vals, cnt = np.unique(a, return_counts=True)
        print(f"  mask {masks[i.stem].name}: size {a.shape[::-1]} vs image {Image.open(i).size}, values {dict(zip(vals.tolist()[:6], (cnt / a.size).round(4).tolist()[:6]))}")
