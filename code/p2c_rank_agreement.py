"""Paper 2 (Chaksu) secondary analysis, after the single locked run.
Pre-registered: rank agreement across the nine models between RFMiD 1.0 camera A/C valid-frame rates (no ground truth)
and Chaksu failure proportions. Specified after the run: A/C combined as the unweighted mean; A, C and B also reported.
POST HOC: in-distribution ground-truth metrics (IDRiD centre error, PAPILA disc Dice) vs Chaksu failure.
Spearman correlation with an exact two-sided permutation p value over all 9! rank orders."""
import json, itertools, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT = Path("/home/test/OpticNeuroFusion_Starter")
SRC, RES = ROOT / "results/fe/review/summary_all_models.json", ROOT / "results/fe/chaksu_eval/locked_results.json"
pre = json.load(open(ROOT / "results/fe/chaksu_eval/preregistration.json"))
assert hashlib.sha256(SRC.read_bytes()).hexdigest() == pre["rfmid_validity_source_sha256"]["results/fe/review/summary_all_models.json"], "RFMiD source changed"
S, R = json.load(open(SRC)), json.load(open(RES)); models = list(S)
P = np.array(list(itertools.permutations(range(len(models)))))
def spearman_exact(x, y):
    rx, ry = rankdata(x), rankdata(y); rxc = rx - rx.mean(); Y = ry[P]; Yc = Y - Y.mean(1, keepdims=True)
    r = (Yc @ rxc) / (np.sqrt((Yc ** 2).sum(1)) * np.sqrt((rxc ** 2).sum()))
    obs = float(np.corrcoef(rx, ry)[0, 1]); return obs, float(np.mean(np.abs(r) >= abs(obs) - 1e-12))
X = {"rfmid_valid_meanAC": [np.mean([S[m]["rfmid"]["valid"]["A"], S[m]["rfmid"]["valid"]["C"]]) for m in models],
     "rfmid_valid_A": [S[m]["rfmid"]["valid"]["A"] for m in models], "rfmid_valid_C": [S[m]["rfmid"]["valid"]["C"] for m in models],
     "rfmid_valid_B": [S[m]["rfmid"]["valid"]["B"] for m in models],
     "POSTHOC_idrid_centre_err_median": [S[m]["idrid_loc"]["decode_err_median"] for m in models],
     "POSTHOC_papila_disc_dice": [S[m]["papila"]["dice_disc"] for m in models]}
Yv = {g: [R["descriptive"][m][g]["failure"]["p"] for m in models] for g in ("all", "Bosch", "Remidio")}
print(f"{'model':18s} " + " ".join(f"{k[:14]:>14s}" for k in X) + "  " + " ".join(f"fail_{g:7s}" for g in Yv))
for i, m in enumerate(models):
    print(f"{m:18s} " + " ".join(f"{X[k][i]:14.3f}" for k in X) + "  " + " ".join(f"{Yv[g][i]:12.3f}" for g in Yv))
out = {}
for k, x in X.items():
    out[k] = {}
    for g, y in Yv.items():
        rho, p = spearman_exact(x, y); out[k][g] = {"rho": rho, "p_exact": p}
print("\nSpearman rho (exact p) vs Chaksu failure; negative rho for validity/Dice = higher score, fewer failures")
for k, v in out.items(): print(f"{k:34s} " + "  ".join(f"{g}: {v[g]['rho']:+.2f} (p={v[g]['p_exact']:.3f})" for g in v))
json.dump({"note": __doc__.strip(), "models": models, "x": X, "chaksu_failure": Yv, "spearman": out},
          open(ROOT / "results/fe/chaksu_eval/secondary_rank_agreement.json", "w"), indent=1)
print("saved results/fe/chaksu_eval/secondary_rank_agreement.json")
