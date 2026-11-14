"""Paper 2 (Chaksu): reference disc in the FE input frame (square crop from fe_external_check.box, resized to 384).
No FE model. Adds gt centre/radius at 384 and whether the true disc lies inside the 15-46 px validity range. Run once."""
import csv, json, sys, hashlib, datetime
from pathlib import Path
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); sys.path.insert(0, str(ROOT / "fe"))
from fe_external_check import box
S, R_MIN, R_MAX = 384, 15, 46
D = ROOT / "results/fe/chaksu_ref_test"; IN, OUT = D / "reference_test.csv", D / "reference_scale_test.csv"
if OUT.exists(): sys.exit("scale file already exists")
rows, out = list(csv.DictReader(open(IN))), []
for i, r in enumerate(rows, 1):
    l, t, s = box(Image.open(ROOT / r["image"]).convert("RGB"))
    gr = float(r["ref_r"]) * S / s; gx, gy = (float(r["ref_cx"]) - l) * S / s, (float(r["ref_cy"]) - t) * S / s
    out.append({"device": r["device"], "stem": r["stem"], "box_l": l, "box_t": t, "box_s": s, "gt_cx384": gx, "gt_cy384": gy,
                "gt_r384": gr, "gt_in_rule": R_MIN <= gr <= R_MAX, "gt_centre_inside_crop": 0 <= gx < S and 0 <= gy < S})
    if i % 200 == 0: print(f"{i}/{len(rows)}", flush=True)
with open(OUT, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
summ = {}
for dev in ("Bosch", "Forus", "Remidio"):
    g = np.array([o["gt_r384"] for o in out if o["device"] == dev])
    summ[dev] = {"n": int(len(g)), "gt_r384_p5_50_95": [float(np.percentile(g, p)) for p in (5, 50, 95)],
                 "in_rule": int(((g >= R_MIN) & (g <= R_MAX)).sum()), "above_46": int((g > R_MAX).sum()), "below_15": int((g < R_MIN).sum()),
                 "centre_outside_crop": sum(not o["gt_centre_inside_crop"] for o in out if o["device"] == dev)}
json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"), "fe_outputs_read": False,
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "summary": summ},
          open(D / "scale_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1))
