"""FE v1: partial-label multi-task U-Net (disc, cup, vessels) on PAPILA + FIVES at 384."""
import argparse, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "results/fe"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def log(m): print(time.strftime("[%H:%M:%S] ") + m, flush=True)

class Up(nn.Module):
    def __init__(s, cin, cskip, cout):
        super().__init__(); s.c = nn.Sequential(nn.Conv2d(cin + cskip, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                                                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))
    def forward(s, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False); return s.c(torch.cat([x, skip], 1))

class UNet(nn.Module):
    def __init__(s, n_out=3):
        super().__init__(); r = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        s.stem = nn.Sequential(r.conv1, r.bn1, r.relu); s.pool = r.maxpool; s.l1, s.l2, s.l3, s.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        s.u4, s.u3, s.u2, s.u1 = Up(512, 256, 256), Up(256, 128, 128), Up(128, 64, 64), Up(64, 64, 32)
        s.head = nn.Conv2d(32, n_out, 1)
    def forward(s, x):
        x0 = s.stem(x); x1 = s.l1(s.pool(x0)); x2 = s.l2(x1); x3 = s.l3(x2); x4 = s.l4(x3)
        y = s.u1(s.u2(s.u3(s.u4(x4, x3), x2), x1), x0)
        return F.interpolate(s.head(y), scale_factor=2, mode="bilinear", align_corners=False)

def augment(x, y, g):
    b = x.shape[0]; dev = x.device
    ang = (torch.rand(b, device=dev, generator=g) - 0.5) * math.radians(30); sc = 1 + (torch.rand(b, device=dev, generator=g) - 0.5) * 0.2
    flip = (torch.rand(b, device=dev, generator=g) < 0.5).float() * 2 - 1
    th = torch.zeros(b, 2, 3, device=dev)
    th[:, 0, 0] = torch.cos(ang) / sc * flip; th[:, 0, 1] = -torch.sin(ang) / sc; th[:, 1, 0] = torch.sin(ang) / sc * flip; th[:, 1, 1] = torch.cos(ang) / sc
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False); y = F.grid_sample(y, grid, align_corners=False)
    c = 1 + (torch.rand(b, 1, 1, 1, device=dev, generator=g) - 0.5) * 0.2; sh = (torch.rand(b, 1, 1, 1, device=dev, generator=g) - 0.5) * 0.1
    return ((x - 0.5) * c + 0.5 + sh).clamp(0, 1), y

def loss_fn(logit, y, av):
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none").mean((2, 3))
    p = torch.sigmoid(logit); inter = (p * y).sum((2, 3)); dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)
    w = av.float(); return ((bce + dice) * w).sum() / w.sum().clamp_min(1)

def geom(m):
    ys, xs = np.nonzero(m > 0.5)
    if len(ys) < 10: return None
    return np.array([xs.mean(), ys.mean()]), math.sqrt(len(ys) / math.pi), ys.max() - ys.min() + 1

def evaluate(model, X, Y, A, dev, bs=16):
    model.eval(); P = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i+bs]).to(dev).permute(0, 3, 1, 2).float().div(255)
            P.append(torch.sigmoid(model((x - MEAN.to(dev)) / STD.to(dev)).float()).cpu().numpy())
    P = np.concatenate(P); Yf = Y.astype(np.float32) / 255; res = {}
    for c, name in enumerate(("disc", "cup", "vessel")):
        idx = np.where(A[:, c] == 1)[0]
        if len(idx) == 0: continue
        pb, gb = P[idx, c] > 0.5, Yf[idx, c] > 0.5
        res[f"dice_{name}"] = float(np.mean([(2 * (a & b).sum() + 1) / (a.sum() + b.sum() + 1) for a, b in zip(pb, gb)]))
    idx = np.where(A[:, 0] == 1)[0]; e, rr, cdr = [], [], []
    for i in idx:
        gp, gg = geom(P[i, 0]), geom(Yf[i, 0])
        if gp is None or gg is None: e.append(np.nan); continue
        e.append(np.linalg.norm(gp[0] - gg[0]) / gg[1]); rr.append(abs(gp[1] - gg[1]) / gg[1])
        cp, cg = geom(P[i, 1]), geom(Yf[i, 1])
        if cp is not None and cg is not None: cdr.append(abs(cp[2] / gp[2] - cg[2] / gg[2]))
    if len(idx):
        e = np.array(e); res.update({"disc_center_err_median": float(np.nanmedian(e)), "disc_center_err_p90": float(np.nanpercentile(e, 90)),
                                     "disc_detect_fail": int(np.isnan(e).sum()), "radius_rel_err_median": float(np.median(rr)) if rr else None,
                                     "vcdr_abs_err_median": float(np.median(cdr)) if cdr else None})
    return res

def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    d = np.load(OUT / "fe_cache.npz"); meta = json.loads((OUT / "fe_meta.json").read_text())
    X, Y, A = d["img"], d["mask"], d["avail"]; sp = np.array([m["split"] for m in meta])
    tr, va, te = (np.where(sp == s)[0] for s in ("train", "val", "test"))
    model = UNet().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = math.ceil(len(tr) / a.batch); total = a.epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / (2 * steps)) * 0.5 * (1 + math.cos(math.pi * min(t, total) / total)))
    g = torch.Generator(device="cuda"); g.manual_seed(a.seed); rng = np.random.default_rng(a.seed); best, hist = -1, []
    run = OUT / f"fe_v1_seed{a.seed}"; run.mkdir(parents=True, exist_ok=True)
    log(f"START FE v1: train={len(tr)} val={len(va)} test={len(te)}")
    for ep in range(1, a.epochs + 1):
        model.train(); perm = rng.permutation(tr); tl = 0; t0 = time.time()
        for k in range(steps):
            idx = np.sort(perm[k * a.batch:(k + 1) * a.batch])
            x = torch.from_numpy(X[idx]).to(dev).permute(0, 3, 1, 2).float().div(255); y = torch.from_numpy(Y[idx]).to(dev).float().div(255)
            x, y = augment(x, y, g); av = torch.from_numpy(A[idx]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model((x - MEAN.to(dev)) / STD.to(dev)).float(), y, av)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        if ep % 5 == 0 or ep == a.epochs:
            m = evaluate(model, X[va], Y[va], A[va], dev); score = np.mean([m["dice_disc"], m["dice_cup"], m["dice_vessel"]])
            hist.append({"epoch": ep, "loss": tl / steps, **m})
            log(f"ep{ep:03d} loss={tl/steps:.4f} val disc={m['dice_disc']:.3f} cup={m['dice_cup']:.3f} vessel={m['dice_vessel']:.3f} "
                f"center_err={m['disc_center_err_median']:.3f} vCDR_err={m['vcdr_abs_err_median']:.3f} ({time.time()-t0:.0f}s/ep)")
            if score > best: best = score; torch.save(model.state_dict(), run / "best.pt")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev, weights_only=True))
    test = {ds: evaluate(model, X[te][[meta[i]["dataset"] == ds for i in te]], Y[te][[meta[i]["dataset"] == ds for i in te]],
                         A[te][[meta[i]["dataset"] == ds for i in te]], dev) for ds in ("papila", "fives")}
    (run / "metrics.json").write_text(json.dumps({"config": vars(a), "history": hist, "test": test}, indent=2), encoding="utf-8")
    log("TEST " + json.dumps(test))
    log(f"DONE FE v1 seed{a.seed}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seed", type=int, default=42); main(ap.parse_args())
