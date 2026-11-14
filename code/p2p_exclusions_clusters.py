"""PALM exclusion list and same-eye proxy clusters, fixed before any FE output on PALM (no FE model).
 - excluded from the primary analysis: PALM images whose 32x32 thumbnail correlation (FE box crop, z-normalised) with any
   REFUGE1 image in SMDG is >= 0.985 (same centre and devices; REFUGE1-val was used in an earlier FE evaluation and
   REFUGE1-train is named in the v1.3 data lock); none reached 0.985 against the FE v1.4 training cache (dup_audit.json);
 - clusters: connected components of PALM pairs with correlation >= 0.985 (proxy for images of the same eye), used only for a
   cluster-bootstrap sensitivity interval."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
from fe_external_check import box, crop
Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "results/fe/palm_ref"
if (OUT / "exclusions_clusters.json").exists(): sys.exit("already built")
def vec(g):
    x = np.asarray(g.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel(); x -= x.mean(); n = np.linalg.norm(x); return x / n if n > 0 else x
def feat(p): im = Image.open(p).convert("RGB"); return vec(crop(im, box(im), Image.BILINEAR).convert("L"))
palm = sorted((ROOT / "data/eval/palm/extracted/PALM").glob("*/Images/*.jpg")); P = np.stack([feat(p) for p in palm])
ref = [p for p in sorted((ROOT / "data/fe/smdg").rglob("*")) if p.is_file() and "refuge1" in p.name.lower() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
       and len(np.unique(np.asarray(Image.open(p).convert("L").resize((64, 64))))) > 8]
R = np.stack([feat(p) for p in ref]); S = P @ R.T; best = S.max(1); arg = S.argmax(1)
excl = {palm[i].stem: {"refuge1_match": str(ref[arg[i]].relative_to(ROOT)), "corr": float(best[i])} for i in np.where(best >= 0.985)[0]}
W = P @ P.T; np.fill_diagonal(W, -1); parent = list(range(len(palm)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
for i, j in zip(*np.where(np.triu(W >= 0.985, 1))): parent[find(i)] = find(j)
cl = {}
for i in range(len(palm)): cl.setdefault(find(i), []).append(palm[i].stem)
cluster_of = {s: k for k, (_, members) in enumerate(sorted(cl.items())) for s in members}
sizes = [len(m) for m in cl.values()]
out = {"rule": __doc__.strip(), "n_palm": len(palm), "n_refuge1_images": len(ref), "excluded": excl, "n_excluded": len(excl),
       "cluster_of": cluster_of, "n_clusters": len(cl), "cluster_size_counts": {str(k): sizes.count(k) for k in sorted(set(sizes))},
       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
json.dump(out, open(OUT / "exclusions_clusters.json", "w"), indent=1)
print({k: out[k] for k in ("n_palm", "n_refuge1_images", "n_excluded", "n_clusters", "cluster_size_counts")})
