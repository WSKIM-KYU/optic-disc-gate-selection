"""FE v1 external check on Akram (CC BY 4.0): vessel Dice + overlay sheets of disc/cup/center for visual review."""
import json, math, sys
from pathlib import Path
import numpy as np, torch
from scipy import ndimage
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fe_train import UNet, MEAN, STD
ROOT = Path(__file__).resolve().parent.parent; S = 384
OUT = ROOT / "results/fe/qc"; OUT.mkdir(parents=True, exist_ok=True)

def box(img):
    a = np.asarray(img.convert("L"), dtype=np.float32); ys, xs = np.where(a > 15)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * 1.02
    return cx - half, cy - half, 2 * half

def crop(img, b, resample):
    l, t, s = b
    return img.crop((int(round(l)), int(round(t)), int(round(l + s)), int(round(t + s)))).resize((S, S), resample)

def outline(mask):
    m = mask > 0.5; e = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)); return e

def _main():
    dev = torch.device("cuda"); model = UNet().to(dev).eval()
    model.load_state_dict(torch.load(ROOT / "results/fe/fe_v1_seed42/best.pt", map_location=dev, weights_only=True))
    items = sorted((ROOT / "data/fe/akram/AV").glob("*/*.JPG")); rows, tiles = [], []
    for f in items:
        img = Image.open(f).convert("RGB"); b = box(img); x = np.asarray(crop(img, b, Image.BILINEAR))
        gt_path = f.with_name(f.stem + "--vessels.jpg")
        gt = np.asarray(crop(Image.open(gt_path).convert("L"), b, Image.BILINEAR), dtype=np.float32) / 255 < 0.5 if gt_path.exists() else None  # Akram GT: dark vessels on white
        t = torch.from_numpy(x).to(dev).permute(2, 0, 1)[None].float().div(255)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model((t - MEAN.to(dev)) / STD.to(dev)).float())[0].cpu().numpy()
        raw = p[0] > 0.5; sm = ndimage.gaussian_filter(p[0], sigma=6); cy0, cx0 = np.unravel_index(np.argmax(sm), sm.shape)
        hm_lab, _ = ndimage.label(sm >= 0.5 * sm[cy0, cx0]); core = hm_lab == hm_lab[cy0, cx0]
        disc = core & raw
        if disc.sum() < 20: disc = core
        n = int(ndimage.label(raw)[1]); spur = float((raw & ~disc).sum() / max(raw.sum(), 1))
        p[0] = np.where(disc, p[0], 0); ys, xs = np.nonzero(disc)
        found = len(ys) >= 10; r = math.sqrt(len(ys) / math.pi) if found else 0
        row = {"id": f.stem, "disc_found": bool(found), "disc_radius_px384": round(r, 1),
               "disc_center": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)] if found else None,
               "disc_max_prob": round(float(p[0].max()), 3), "n_components": int(n), "spurious_frac": round(spur, 3)}
        if gt is not None:
            pv = p[2] > 0.5; row["vessel_dice"] = round(float((2 * (pv & gt).sum() + 1) / (pv.sum() + gt.sum() + 1)), 4)
        rows.append(row)
        v = x.copy(); v[outline(p[0])] = [255, 0, 0]; v[outline(p[1])] = [0, 128, 255]
        im = Image.fromarray(v); d = ImageDraw.Draw(im)
        if found:
            cx, cy = xs.mean(), ys.mean(); d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 255, 0))
        d.text((4, 4), f.stem + ("" if found else "  NO DISC"), fill=(255, 255, 255)); tiles.append(im.resize((256, 256)))
    for k in range(0, len(tiles), 16):
        sheet = Image.new("RGB", (256 * 4, 256 * 4))
        for j, im in enumerate(tiles[k:k + 16]): sheet.paste(im, ((j % 4) * 256, (j // 4) * 256))
        sheet.save(OUT / f"akram_cc3_sheet_{k // 16 + 1:02d}.jpg", quality=90)
    (OUT / "akram_fe_check_cc3.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    vd = [r["vessel_dice"] for r in rows if "vessel_dice" in r]; rad = [r["disc_radius_px384"] for r in rows if r["disc_found"]]
    print(f"images={len(rows)}  disc_found={sum(r['disc_found'] for r in rows)}  "
          f"vessel_dice median={np.median(vd):.3f} [p10 {np.percentile(vd, 10):.3f}, p90 {np.percentile(vd, 90):.3f}]  "
          f"disc_radius median={np.median(rad):.1f}px (range {min(rad):.1f}-{max(rad):.1f})")
    low = sorted(rows, key=lambda r: r["disc_max_prob"])[:5]
    print("lowest disc confidence:", [(r["id"], r["disc_max_prob"]) for r in low])
    print("SHEETS:", sorted(str(p.name) for p in OUT.glob("akram_cc3_sheet_*.jpg")))
    sf = [r["spurious_frac"] for r in rows]
    print(f"spurious fraction: median={np.median(sf):.3f}  images with >20% spurious={sum(x > 0.2 for x in sf)}/{len(sf)}")


if __name__ == "__main__":
    _main()
