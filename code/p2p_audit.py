"""EXPLORATORY audit after the locked PALM run (reviewer checks); changes nothing in the locked results.
(1) P3 sensitivity to the same-eye cluster definition: image-level Wilson and Clopper-Pearson, locked cluster bootstrap with five
    random seeds, largest cluster split into single images, image-level bootstrap, and analysis without the largest cluster;
    where the failed accepted images sit; whether the largest cluster is among the excluded duplicates.
(2) Images without a disc mask accepted by each choice, with the predicted frame drawn on the R1m crop for visual review."""
import json, csv, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import beta
from PIL import Image, ImageDraw
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
import p2p_palm_eval as E
import p2c_crop as CR
C = E.C; Image.MAX_IMAGE_PIXELS = None
OUT = ROOT / "results/fe/palm_audit"
A = json.load(open(E.AF)); ref = {r["stem"]: r for r in csv.DictReader(open(E.REFD / "reference_palm.csv"))}
ec = json.load(open(E.REFD / "exclusions_clusters.json")); cl = ec["cluster_of"]; excl = set(ec["excluded"]); size = Counter(cl.values())
big = max(size, key=size.get); rows = [a for a in A["rows"] if not a["refuge_dup"]]
def cp(k, n): return 1.0 if n == 0 or k >= n else float(beta.ppf(0.975, k + 1, n - k))
def boot(fails, keys, B=10000, seed=20261114):
    g = {}
    for f, k in zip(fails, keys): g.setdefault(k, []).append(f)
    f = np.array([sum(v) for v in g.values()]); m = np.array([len(v) for v in g.values()]); nc = len(g)
    if f.sum() == 0: return round(cp(0, nc), 4)
    rng = np.random.default_rng(seed); bs = [f[i].sum() / m[i].sum() for i in (rng.integers(0, nc, nc) for _ in range(B))]
    return round(float(np.percentile(bs, 97.5)), 4)
members = [s for s, c in cl.items() if c == big]
out = {"note": __doc__.strip(), "largest_cluster": {"size": size[big], "members_in_excluded_duplicates": sum(s in excl for s in members),
       "members_included": sum(1 for a in rows if cl.get(a["stem"]) == big),
       "members_included_with_mask": sum(1 for a in rows if cl.get(a["stem"]) == big and ref[a["stem"]]["has_mask"] == "True")}}
for tr, m in (("research", "fe_v14_seed42"), ("product", "fe_v14p_seed42")):
    acc = [a for a in rows if ref[a["stem"]]["has_mask"] == "True" and a["R1m_tta"][m]["valid"]]
    fails = [bool(E.score_row(a, ref[a["stem"]])["R1m_tta"][m]["fail"]) for a in acc]; stems = [a["stem"] for a in acc]; k, n = sum(fails), len(acc)
    k0 = [cl.get(s, s) for s in stems]; ksplit = [s if cl.get(s) == big else cl.get(s, s) for s in stems]
    keep = [i for i, s in enumerate(stems) if cl.get(s) != big]
    out[tr] = {"model": m, "accepted": n, "failures": k, "accepted_in_largest_cluster": sum(cl.get(s) == big for s in stems),
               "failed_images_cluster_size": [(s, size[cl.get(s)]) for s, f in zip(stems, fails) if f],
               "wilson95": [round(x, 4) for x in C.wilson(k, n)], "clopper_pearson_image_upper": round(cp(k, n), 4),
               "cluster_boot_locked_def_5_seeds": [boot(fails, k0, seed=s) for s in (20261114, 1, 2, 3, 4)],
               "cluster_boot_largest_split": boot(fails, ksplit), "image_level_boot": boot(fails, stems),
               "without_largest_cluster": {"accepted": len(keep), "failures": sum(fails[i] for i in keep),
                                           "cluster_boot": boot([fails[i] for i in keep], [k0[i] for i in keep])}}
nm = [a for a in rows if ref[a["stem"]]["has_mask"] != "True"]; out["no_mask"] = {"included_images": len(nm)}; tiles = []
for tr, m in (("research", "fe_v14_seed42"), ("product", "fe_v14p_seed42")):
    accm = [a for a in nm if a["R1m_tta"][m]["valid"]]
    out["no_mask"][tr] = [{"stem": a["stem"], "device": a["device"], "pm_label": ref[a["stem"]]["pm_label"], "centre": ref[a["stem"]]["shooting_centre"]} for a in accm]
    for a in accm:
        x, _ = CR.r1m(Image.open(ROOT / ref[a["stem"]]["image"]).convert("RGB")); t = Image.fromarray(np.asarray(x)).convert("RGB"); d = ImageDraw.Draw(t)
        c, r = a["R1m_tta"][m]["c"], a["R1m_tta"][m]["r"]
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=(0, 255, 0), width=3); d.rectangle([0, 0, 383, 18], fill=(0, 0, 0))
        d.text((4, 3), f"{tr} {a['stem']} {a['device']} pm{ref[a['stem']]['pm_label']}", fill=(255, 255, 255)); tiles.append(t)
if tiles:
    g = Image.new("RGB", (384 * 5, 384 * ((len(tiles) + 4) // 5)))
    for i, t in enumerate(tiles): g.paste(t, ((i % 5) * 384, (i // 5) * 384))
    g.save(OUT / "no_mask_accepted.jpg", quality=88)
json.dump(out, open(OUT / "audit.json", "w"), indent=1); print(json.dumps(out, indent=1))
