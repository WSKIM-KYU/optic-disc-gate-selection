"""Paper 2 figures, JBHI double-column width (7.16 in), 7 pt text: rendering of locked and exploratory results only (no new inference).
Writes results/fe/figures/Fig1-Fig5 (.png 600 dpi and .pdf). Fig 5 uses the single-pass R1m size check as the size component of the
gate for the threshold sweep; the deployed operating point (dots) uses the stored gate decision."""
import json, csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
ROOT = Path("/home/test/OpticNeuroFusion_Starter"); R = ROOT / "results/fe"; OUT = R / "figures"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "fe")); sys.path.insert(0, str(ROOT))
W = 7.16
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
NAMES = list(json.load(open(R / "review/summary_all_models.json")))
LAB = {"fe_v14_seed42": "R-s42", "fe_v14_seed43": "R-s43", "fe_v14_seed44": "R-s44", "fe_v14p_seed42": "P-s42", "fe_v14p_seed43": "P-s43", "fe_v14p_seed44": "P-s44",
       "fe_v14_f10_seed42": "R-f10", "fe_v14_f25_seed42": "R-f25", "fe_v14_f50_seed42": "R-f50"}
HL = {"fe_v14_seed42": "#2e7d4f", "fe_v14p_seed42": "#2c6fbb", "fe_v14p_seed44": "#c0392b"}
def save(fig, name): fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); plt.close(fig); print("saved", name)
T = json.load(open(R / "chaksu_test_eval/locked_results.json")); G = json.load(open(R / "riga_eval/locked_results.json")); P = json.load(open(R / "palm_eval/locked_results.json"))["primary_analysis"]

# Fig 1
fig, ax = plt.subplots(figsize=(W, 1.35)); ax.axis("off"); ax.set_xlim(0, 7); ax.set_ylim(0, 1.35)
steps = [("Training\nPAPILA, FIVES,\nIDRiD, RIM-ONE\nnine models", 0), ("Chákṣu Train\n1,009 images\nlocked\nevaluation", 1), ("Chákṣu Train\ndevelopment\ncrop fix,\nTTA selection", 0),
         ("Chákṣu Test\n336 images\nlocked\nH1-H3 met", 1), ("RIGA\n747 images\nlocked: G3, G4\nmet; G1 not met", 1), ("Exploratory\nconsistent-but-\nwrong models;\nfull gate", 0),
         ("PALM\n1,111 images\nlocked,\ntwo-stage\nP1-P3 met", 1)]
for i, (txt, locked) in enumerate(steps):
    x = 0.04 + i
    ax.add_patch(FancyBboxPatch((x, 0.28), 0.86, 0.95, boxstyle="round,pad=0.02", fc="#d9efe1" if locked else "#eeeeee", ec="#2e7d4f" if locked else "#888888", lw=0.8))
    ax.text(x + 0.43, 0.755, txt, ha="center", va="center", fontsize=6)
    if i < len(steps) - 1: ax.annotate("", xy=(x + 1.03, 0.755), xytext=(x + 0.91, 0.755), arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
ax.text(0.04, 0.05, "green: locked, pre-registered, single run    grey: development or exploratory", fontsize=6, color="#444444")
save(fig, "Fig1_study_flow")

# Fig 2
cols = [("Chákṣu Test", "Bosch"), ("Chákṣu Test", "Forus"), ("Chákṣu Test", "Remidio"), ("RIGA", "Messidor"), ("RIGA", "BinRushed"), ("RIGA", "Magrabi"), ("PALM", "Zeiss"), ("PALM", "Canon")]
def fval(ds, g, n):
    if ds == "Chákṣu Test": return T["secondary"]["tta_selection_summary"][g][n]["failure"]
    if ds == "RIGA": return G["secondary"]["tta_selection_summary"][g][n]["failure"]
    return P["secondary"]["failure_by_group"][g][n]
M = np.array([[fval(ds, g, n) * 100 for ds, g in cols] for n in NAMES])
fig, ax = plt.subplots(figsize=(W * 0.72, 2.6)); im = ax.imshow(M, cmap="Reds", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([g for _, g in cols]); ax.set_yticks(range(len(NAMES))); ax.set_yticklabels([LAB[n] for n in NAMES])
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="white" if M[i, j] > 60 else "black")
for ds, (a, b) in (("Chákṣu Test", (0, 2)), ("RIGA", (3, 5)), ("PALM", (6, 7))):
    ax.annotate(ds, xy=((a + b) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
for x in (2.5, 5.5): ax.axvline(x, color="white", lw=1.5)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("failure (%)")
save(fig, "Fig2_failure_heatmap")

# Fig 3
rc = list(csv.DictReader(open(R / "chaksu_posthoc/rootcause_train.csv")))
grp = {"Bosch": [float(r["fundus_area_frac"]) for r in rc if r["device"] == "Bosch"], "Forus": [float(r["fundus_area_frac"]) for r in rc if r["device"] == "Forus"],
       "Remidio\ncorrect": [float(r["fundus_area_frac"]) for r in rc if r["device"] == "Remidio" and r["fe_v14_seed42_fail"] == "0"],
       "Remidio\nfailed": [float(r["fundus_area_frac"]) for r in rc if r["device"] == "Remidio" and r["fe_v14_seed42_fail"] == "1"]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.3]})
a1.boxplot(list(grp.values()), showfliers=False, medianprops=dict(color="#c0392b")); a1.set_xticks(range(1, 5)); a1.set_xticklabels(list(grp))
a1.set_ylabel("fundus area / R0 crop area"); a1.set_title("(a) Fundus fraction of the original crop (Chákṣu Train)")
crop = {"Chákṣu Test": {tr: (T["H1_crop_remedy"][tr]["R0_failure"]["p"], T["H1_crop_remedy"][tr]["R1m_failure"]["p"]) for tr in ("research", "product")},
        "RIGA": {tr: (G["G4_crop"][tr]["R0_failure"]["p"], G["G4_crop"][tr]["R1m_failure"]["p"]) for tr in ("research", "product")},
        "PALM": {tr: (P["secondary"]["crop_R1m_vs_R0"][tr]["R0_failure"], P["secondary"]["crop_R1m_vs_R0"][tr]["R1m_failure"]) for tr in ("research", "product")}}
x, w, k, ticks, groups = 0.0, 0.18, 0, [], []
for ds, v in crop.items():
    start = x
    for tr in ("research", "product"):
        r0, r1 = v[tr]; a2.bar(x, r0 * 100, w, color="#bbbbbb", label="R0 crop" if k == 0 else None); a2.bar(x + w, r1 * 100, w, color="#2e7d4f", label="R1m crop" if k == 0 else None)
        ticks.append((x + w / 2, "R" if tr == "research" else "P")); x += 0.5; k += 1
    groups.append(((start + x - 0.5 + w) / 2, ds)); x += 0.25
a2.set_xticks([t for t, _ in ticks]); a2.set_xticklabels([l for _, l in ticks])
for gx, ds in groups: a2.annotate(ds, xy=(gx, 0), xycoords=("data", "axes fraction"), xytext=(0, -16), textcoords="offset points", ha="center", va="top", fontsize=6.5)
a2.set_ylabel("deployed pipeline failure (%)"); a2.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0)); a2.set_ylim(0, 68); a2.set_title("(b) Deployed pipelines, R0 vs R1m crop (R research, P product)")
save(fig, "Fig3_root_cause_crop")

# Fig 4
def crit_rows(D):
    s = D["secondary"]["tta_selection_summary"]["all"]; rows = D["rows"]
    return ({n: s[n]["flag_rate"] for n in NAMES}, {n: 1 - float(np.mean([r["R1m_tta"][n]["valid"] for r in rows])) for n in NAMES}, {n: s[n]["failure"] for n in NAMES})
sets = {"Chákṣu Test": crit_rows(T), "RIGA": crit_rows(G),
        "PALM": ({n: P["secondary"]["selection_rates"][n]["flag"] for n in NAMES}, {n: P["secondary"]["selection_rates"][n]["reject"] for n in NAMES}, P["secondary"]["failure_by_group"]["pooled"])}
fig, axs = plt.subplots(2, 3, figsize=(W, 4.0), sharey=True)
for j, (ds, (fl, rj, fa)) in enumerate(sets.items()):
    for i, (crit, name) in enumerate(((fl, "TTA flag rate"), (rj, "full-gate rejection rate"))):
        ax = axs[i, j]; xs = [crit[n] * 100 for n in NAMES]; ys = [fa[n] * 100 for n in NAMES]
        for n, xx, yy in zip(NAMES, xs, ys): ax.scatter(xx, yy, s=16, color=HL.get(n, "#9a9a9a"), zorder=3 if n in HL else 2, edgecolor="none")
        rho = spearmanr(xs, ys).correlation; ax.set_title(f"{ds}: {name} (rho = {rho:.2f})"); ax.set_xlabel(f"{name} (%)")
        if j == 0: ax.set_ylabel("failure (%)")
handles = [Line2D([], [], marker="o", ls="", color=c, label=LAB[n]) for n, c in HL.items()] + [Line2D([], [], marker="o", ls="", color="#9a9a9a", label="other six models")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02)); fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "Fig4_selection_criteria")

# Fig 5
import p2p_palm_eval as E
def rows_locked(D): return [{"fail": {n: r["R1m_tta"][n]["fail"] for n in NAMES}, "spread": {n: r["R1m_tta"][n]["spread"] for n in NAMES},
                             "valid": {n: r["R1m_tta"][n]["valid"] for n in NAMES}, "size": {n: r["R1m"][n]["reason"] == "ok" for n in NAMES}} for r in D["rows"]]
A = json.load(open(R / "palm_eval/stageA_outputs.json")); ref = {r["stem"]: r for r in csv.DictReader(open(R / "palm_ref/reference_palm.csv"))}; prow = []
for a in A["rows"]:
    r = ref[a["stem"]]
    if a["refuge_dup"] or r["has_mask"] != "True": continue
    sc = E.score_row(a, r)
    prow.append({"fail": {n: sc["R1m_tta"][n]["fail"] for n in NAMES}, "spread": {n: a["R1m_tta"][n]["spread"] for n in NAMES}, "valid": {n: a["R1m_tta"][n]["valid"] for n in NAMES},
                 "size": {n: a["R1m"][n]["status"] == "ok" and a["R1m"][n]["radius"] is not None and 15 <= a["R1m"][n]["radius"] <= 46 for n in NAMES}})
RC = {"Chákṣu Test": rows_locked(T), "RIGA": rows_locked(G), "PALM": prow}
fig, axs = plt.subplots(1, 3, figsize=(W, 2.1), sharey=True)
for ax, (ds, rows) in zip(axs, RC.items()):
    for n, col in HL.items():
        f = np.array([r["fail"][n] for r in rows]); s = np.array([r["spread"][n] if r["spread"][n] is not None else np.inf for r in rows]); z = np.array([r["size"][n] for r in rows])
        cov, risk = [], []
        for t in np.unique(np.concatenate([s[np.isfinite(s)], [np.inf]])):
            acc = z & (s <= t)
            if acc.sum() >= 5: cov.append(acc.mean() * 100); risk.append(f[acc].mean() * 100)
        ax.plot(cov, risk, color=col, lw=1.0, label=LAB[n])
        v = np.array([r["valid"][n] for r in rows]); ax.scatter([v.mean() * 100], [f[v].mean() * 100 if v.any() else np.nan], color=col, s=18, edgecolor="black", linewidth=0.5, zorder=3)
    ax.set_title(ds); ax.set_xlabel("coverage (%)")
axs[0].set_ylabel("silent failure among accepted (%)"); axs[0].legend(frameon=False)
save(fig, "Fig5_risk_coverage")
