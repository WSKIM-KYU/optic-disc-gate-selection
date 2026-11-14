"""PALM near-duplicate audit (no FE model): 32x32 grey thumbnails of the FE box crop, z-normalised; correlation with
REFUGE1 images inside SMDG (same centre and devices as PALM), the FE v1.4 training cache (all images), and within PALM
(same-eye groups for a cluster-bootstrap sensitivity analysis). Duplicate threshold 0.985 as in the IDRiD-RFMiD audit."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
from fe_external_check import box, crop
Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "results/fe/palm_ref"; OUT.mkdir(parents=True, exist_ok=True)
def vec(g):
    x = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel(); x -= x.mean(); n = np.linalg.norm(x); return x / n if n > 0 else x
def feat(p):
    im = Image.open(p).convert("RGB"); return vec(crop(im, box(im), Image.BILINEAR).convert("L"))
palm = sorted((ROOT / "data/eval/palm/extracted/PALM").glob("*/Images/*.jpg")); P = np.stack([feat(p) for p in palm]); print("PALM", len(palm), flush=True)
ref = []
for p in sorted((ROOT / "data/fe/smdg").rglob("*")):
    if p.is_file() and "refuge1" in p.name.lower() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        a = np.asarray(Image.open(p).convert("L").resize((64, 64)))
        if len(np.unique(a)) > 8: ref.append(p)          # skip binary masks
R = np.stack([feat(p) for p in ref]); print("REFUGE1 images in SMDG", len(ref), flush=True)
d = np.load(ROOT / "results/fe/fe_cache_v14.npz"); X = d["img"]; meta = json.load(open(ROOT / "results/fe/fe_meta_v14.json"))
Cc = np.stack([vec(Image.fromarray(np.asarray(x)).convert("L")) for x in X]); print("FE cache", len(Cc), flush=True)
out = {"n_palm": len(palm), "n_refuge1": len(ref), "n_fe_cache": len(Cc), "threshold": 0.985}
for name, M, labels in (("refuge1", R, [str(p.relative_to(ROOT)) for p in ref]), ("fe_cache", Cc, [f"{m['dataset']}/{m['split']}/{m.get('id', i)}" for i, m in enumerate(meta)])):
    S = P @ M.T; best = S.max(1); arg = S.argmax(1)
    out[name] = {"max_corr_p50_p99_max": [float(np.percentile(best, 50)), float(np.percentile(best, 99)), float(best.max())],
                 "n_ge_0p985": int((best >= 0.985).sum()), "n_ge_0p95": int((best >= 0.95).sum()),
                 "top10": [[palm[i].name, labels[arg[i]], float(best[i])] for i in np.argsort(-best)[:10]]}
    print(name, out[name]["max_corr_p50_p99_max"], ">=0.985:", out[name]["n_ge_0p985"], flush=True)
S = P @ P.T; np.fill_diagonal(S, -1); parent = list(range(len(palm)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i, j in zip(*np.where(np.triu(S >= 0.95, 1))): parent[find(i)] = find(j)
groups = {}
for i in range(len(palm)): groups.setdefault(find(i), []).append(palm[i].stem)
sizes = [len(g) for g in groups.values()]
out["within_palm"] = {"pairs_ge_0p985": int((np.triu(S, 1) >= 0.985).sum()), "pairs_ge_0p95": int((np.triu(S, 1) >= 0.95).sum()),
                      "clusters_at_0p95": len(groups), "cluster_size_counts": {str(k): sizes.count(k) for k in sorted(set(sizes))}}
json.dump({**out, "clusters_0p95": list(groups.values())}, open(OUT / "dup_audit.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items()}, indent=1))
