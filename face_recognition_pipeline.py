"""
EECS 700 - Lightweight Face Recognition System (binary templates + Hamming distance)
====================================================================================

Pipeline
--------
  Task 1  Pretrained ArcFace (InsightFace "buffalo_l", ResNet-50, trained on WebFace600K)
          -> 512-D float embedding per image
          -> binarized with a per-dimension threshold (median of the gallery embeddings)
          -> 512-bit template.  Saved to GallerySet_binary.csv / ProbeSet_binary.csv
  Task 2  3976 x 994 normalized Hamming-distance score matrix
  Task 3  Genuine vs impostor distributions (probability density)
  Task 4  Decidability d'
  Task 5  CMC curve (rank-1 accuracy, lowest rank reaching max accuracy)
  Task 6  ROC (FAR vs FRR) + EER
  Q7      Per-condition (noisy / profile / blurred / occluded) 994 x 994 matrices, plots, d'
  Bonus   (--bonus) Improved templates for the weakest condition:
          flip test-time augmentation + padded re-detection + random-hyperplane
          (SimHash) expansion to 4096 bits.  Still binary, still Hamming distance.

Dependencies (Python 3.9 - 3.11)
--------------------------------
  pip install numpy pandas matplotlib opencv-python onnxruntime insightface
  (GPU: replace onnxruntime with onnxruntime-gpu and run with --ctx 0)
  Windows: insightface needs "Microsoft C++ Build Tools" (Desktop development with C++)
  installed before pip install.
  The buffalo_l model (~280 MB) downloads automatically to ~/.insightface/models on first run.

Dataset paths
-------------
  Extract GallerySet.rar and ProbeSet.rar.  Folders may contain sub-folders; the script
  walks them recursively and reads any .jpg/.jpeg/.png/.bmp/.tif/.tiff/.pgm/.ppm file whose
  name contains  subjectXXX_imgN  (e.g. subject001_img1.jpg).

Execution
---------
  python face_recognition_pipeline.py --gallery ./GallerySet --probe ./ProbeSet --out ./results
  python face_recognition_pipeline.py --gallery ./GallerySet --probe ./ProbeSet --out ./results --bonus

Outputs (in --out)
------------------
  GallerySet_features.csv, ProbeSet_features.csv     raw 512-D float features (Q1)
  GallerySet_binary.csv,   ProbeSet_binary.csv       512-bit binary templates (Task 1 / Q2)
  score_matrix_all.csv (+ .npy)                      3976 x 994 matrix A (Task 2)
  dist_all.png, cmc_all.png, roc_all.png             Tasks 3, 5, 6
  score_matrix_<cond>.csv, dist_<cond>.png           Q7 (noisy, profile, blurred, occluded)
  snippets_Q1_Q2.txt                                 feature / binary snippets for slides
  metrics.json, metrics.txt                          d', rank-1, max-rank, EER, Q7 table
  detection_log.csv                                  how each face was located (detector / fallback)
  bonus_* files                                      only with --bonus
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pgm", ".ppm"}
NAME_RE = re.compile(r"(subject\d+)_(img\d+)", re.IGNORECASE)
CONDITIONS = {"img2": "noisy", "img3": "profile", "img4": "blurred", "img5": "occluded"}
ARC_SIZE = 112


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def list_images(root):
    """Return a DataFrame [subject, sample, label, path] sorted by subject then sample."""
    rows = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            m = NAME_RE.search(p.stem)
            if m:
                subj, samp = m.group(1).lower(), m.group(2).lower()
                rows.append({"subject": subj, "sample": samp,
                             "label": f"{subj}_{samp}", "path": str(p)})
    if not rows:
        sys.exit(f"No images matching subjectXXX_imgN found under {root}")
    df = pd.DataFrame(rows).drop_duplicates("label")
    df["_sid"] = df["subject"].str.extract(r"(\d+)").astype(int)
    df["_smp"] = df["sample"].str.extract(r"(\d+)").astype(int)
    return df.sort_values(["_sid", "_smp"]).drop(columns=["_sid", "_smp"]).reset_index(drop=True)


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR, also converts grayscale to 3-channel
    if img is None:  # handles unicode paths on Windows
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read {path}")
    return img


# --------------------------------------------------------------------------------------
# Model (Task 1 - feature extraction)
# --------------------------------------------------------------------------------------
class ArcFaceExtractor:
    """InsightFace buffalo_l: RetinaFace detector + ArcFace ResNet-50 recognizer."""

    def __init__(self, ctx_id=-1, det_size=640):
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if ctx_id >= 0 else ["CPUExecutionProvider"])
        self.face_align = face_align
        self.app = FaceAnalysis(name="buffalo_l", providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size), det_thresh=0.5)
        self.det = self.app.models["detection"]
        self.rec = self.app.models["recognition"]

    def _detect(self, img, thresh):
        old = self.det.det_thresh
        self.det.det_thresh = thresh
        bboxes, kpss = self.det.detect(img, max_num=0, metric="default")
        self.det.det_thresh = old
        if bboxes is None or len(bboxes) == 0:
            return None
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return kpss[int(np.argmax(areas))]

    def aligned_crop(self, img, robust=False):
        """
        Same preprocessing for gallery and probe:
          1) RetinaFace detection (largest face) -> 5-point similarity alignment -> 112x112
          2) if no face: retry with lower threshold on a padded image (tight crops / profiles)
          3) if still none: resize whole image to 112x112 (assumes image is already a face crop)
        """
        kps = self._detect(img, 0.5)
        if kps is not None:
            return self.face_align.norm_crop(img, landmark=kps, image_size=ARC_SIZE), "detected"
        pad = int(0.4 * max(img.shape[:2]))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        thresholds = [0.3, 0.15] if robust else [0.3]
        for t in thresholds:
            kps = self._detect(padded, t)
            if kps is not None:
                return self.face_align.norm_crop(padded, landmark=kps, image_size=ARC_SIZE), f"padded@{t}"
        return cv2.resize(img, (ARC_SIZE, ARC_SIZE), interpolation=cv2.INTER_AREA), "fallback_resize"

    def embed(self, crops, flip_tta=False, batch=64):
        out = []
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            f = self.rec.get_feat(chunk)
            if flip_tta:
                f = f + self.rec.get_feat([cv2.flip(c, 1) for c in chunk])
            out.append(f)
        feats = np.vstack(out).astype(np.float32)
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)


def extract(extractor, df, robust=False, flip_tta=False, tag=""):
    crops, how = [], []
    n = len(df)
    for k, path in enumerate(df["path"]):
        c, h = extractor.aligned_crop(read_image(path), robust=robust)
        crops.append(c)
        how.append(h)
        if (k + 1) % 200 == 0 or k + 1 == n:
            print(f"  [{tag}] aligned {k + 1}/{n}", flush=True)
    feats = extractor.embed(crops, flip_tta=flip_tta)
    return feats, how


# --------------------------------------------------------------------------------------
# Binarization / matching / metrics
# --------------------------------------------------------------------------------------
def binarize(feats, thresholds):
    return (feats > thresholds).astype(np.uint8)


def hamming_matrix(P, G):
    """Normalized Hamming distance between every row of P and every row of G (fraction of bits)."""
    P = P.astype(np.float32)
    G = G.astype(np.float32)
    d = P @ (1.0 - G).T + (1.0 - P) @ G.T
    return d / P.shape[1]


def split_scores(A, probe_subj, gallery_subj):
    mask = np.asarray(probe_subj)[:, None] == np.asarray(gallery_subj)[None, :]
    return A[mask], A[~mask], mask


def decidability(gen, imp):
    mg, mi = gen.mean(), imp.mean()
    sg, si = gen.std(), imp.std()
    return abs(mg - mi) / np.sqrt((sg ** 2 + si ** 2) / 2.0), mg, sg, mi, si


def cmc_curve(A, mask):
    """Rank of the true gallery match for each probe (1 = best). Ties counted pessimistically."""
    gen_scores = np.where(mask, A, np.inf).min(axis=1)
    ranks = (A < gen_scores[:, None]).sum(axis=1) + (A == gen_scores[:, None]).sum(axis=1)
    n_gal = A.shape[1]
    cmc = np.array([(ranks <= r).mean() for r in range(1, n_gal + 1)])
    return cmc, ranks


def roc_far_frr(gen, imp, n_bits):
    """Distances are multiples of 1/n_bits, so every possible threshold is evaluated exactly."""
    thr = np.arange(n_bits + 1) / n_bits
    gs, is_ = np.sort(gen), np.sort(imp)
    far = np.searchsorted(is_, thr, side="right") / len(is_)      # impostor accepted: d <= t
    frr = 1.0 - np.searchsorted(gs, thr, side="right") / len(gs)  # genuine rejected:  d >  t
    k = int(np.argmin(np.abs(far - frr)))
    eer = (far[k] + frr[k]) / 2.0
    return thr, far, frr, eer, thr[k]


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
def plot_distributions(gen, imp, dprime, title, path):
    lo = max(0.0, min(gen.min(), imp.min()) - 0.02)
    hi = min(1.0, max(gen.max(), imp.max()) + 0.02)
    bins = np.linspace(lo, hi, 80)
    plt.figure(figsize=(8, 5))
    plt.hist(imp, bins=bins, density=True, alpha=0.55, color="tab:red",
             label=f"Impostor (n={len(imp):,})")
    plt.hist(gen, bins=bins, density=True, alpha=0.55, color="tab:green",
             label=f"Genuine (n={len(gen):,})")
    plt.xlabel("Normalized Hamming distance")
    plt.ylabel("Probability density")
    plt.title(f"{title}   d' = {dprime:.3f}")
    plt.xlim(lo, hi)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_cmc(cmc, rank1, max_rank, path, max_show=50):
    r = np.arange(1, len(cmc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lim in zip(axes, [len(cmc), min(max_show, len(cmc))]):
        ax.plot(r[:lim], cmc[:lim] * 100, lw=2, marker="." if lim <= 50 else None)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Identification rate (%)")
        ax.set_xlim(1, lim)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"CMC (all ranks)  rank-1 = {rank1 * 100:.2f}%")
    axes[1].set_title(f"CMC (ranks 1-{min(max_show, len(cmc))})  max {cmc.max() * 100:.2f}% at rank {max_rank}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc(far, frr, eer, path):
    plt.figure(figsize=(7, 6))
    plt.plot(far * 100, frr * 100, lw=2, label="ROC")
    plt.plot([0, 100], [0, 100], "k--", lw=1, label="FAR = FRR")
    plt.scatter([eer * 100], [eer * 100], c="red", zorder=5, label=f"EER = {eer * 100:.3f}%")
    plt.xlabel("False Accept Rate, FAR (%)")
    plt.ylabel("False Reject Rate, FRR (%)")
    plt.title("ROC (FAR vs FRR)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    ok = (far > 0) & (frr > 0)
    if ok.sum() < 2:  # perfect separation: nothing to show on a log scale
        return
    plt.figure(figsize=(7, 6))  # log-scale version, easier to read when EER is small
    plt.loglog(far[ok] * 100, frr[ok] * 100, lw=2)
    plt.scatter([eer * 100], [eer * 100], c="red", zorder=5, label=f"EER = {eer * 100:.3f}%")
    plt.xlabel("FAR (%) [log]")
    plt.ylabel("FRR (%) [log]")
    plt.title("ROC (FAR vs FRR), log scale")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_log.png"), dpi=200)
    plt.close()


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def save_matrix(df, cols, prefix, out):
    pd.DataFrame(df, columns=cols).to_csv(out / f"{prefix}.csv", index=False)


def template_frame(meta, arr, prefix):
    cols = [f"{prefix}{i}" for i in range(arr.shape[1])]
    data = pd.DataFrame(arr, columns=cols)
    return pd.concat([meta[["label", "subject", "sample"]].reset_index(drop=True), data], axis=1)


def save_score_matrix(A, rows, cols, out, name):
    np.save(out / f"{name}.npy", A.astype(np.float32))
    pd.DataFrame(np.round(A, 6), index=rows, columns=cols).to_csv(out / f"{name}.csv")


def write_snippets(gal, prb, gf, pf, gb, pb, path, n_subj=5, n_show=10):
    subs = gal["subject"].iloc[:n_subj].tolist()
    lines = [f"Feature dimensionality before binarization: {gf.shape[1]}",
             f"Binary template length: {gb.shape[1]} bits", ""]
    for title, G, P, fmt in [("Q1 - raw ArcFace features (L2-normalized), first "
                              f"{n_show} of {gf.shape[1]} dims", gf, pf, "{:+.4f}"),
                             (f"Q2 - binary templates, first {n_show * 4} of {gb.shape[1]} bits",
                              gb, pb, "{}")]:
        lines += ["=" * 80, title, "=" * 80]
        k = n_show if fmt != "{}" else n_show * 4
        for s in subs:
            gi = gal.index[gal["subject"] == s][0]
            lines.append(f"{gal.at[gi, 'label']:<20}" + " ".join(fmt.format(v) for v in G[gi, :k]))
            for pi in prb.index[prb["subject"] == s]:
                lines.append(f"{prb.at[pi, 'label']:<20}" + " ".join(fmt.format(v) for v in P[pi, :k]))
            lines.append("")
    Path(path).write_text("\n".join(lines))


def evaluate(A, probe_subj, gallery_subj, n_bits, out, tag, title, full=True):
    gen, imp, mask = split_scores(A, probe_subj, gallery_subj)
    d, mg, sg, mi, si = decidability(gen, imp)
    plot_distributions(gen, imp, d, title, str(out / f"dist_{tag}.png"))
    res = {"d_prime": d, "genuine_mean": mg, "genuine_std": sg, "impostor_mean": mi,
           "impostor_std": si, "n_genuine": int(len(gen)), "n_impostor": int(len(imp))}
    if full:
        cmc, _ = cmc_curve(A, mask)
        rank1 = float(cmc[0])
        max_rank = int(np.argmax(cmc >= cmc.max() - 1e-12) + 1)
        plot_cmc(cmc, rank1, max_rank, str(out / f"cmc_{tag}.png"))
        _, far, frr, eer, eer_thr = roc_far_frr(gen, imp, n_bits)
        plot_roc(far, frr, eer, str(out / f"roc_{tag}.png"))
        res.update({"rank1": rank1, "max_identification_rate": float(cmc.max()),
                    "lowest_rank_at_max": max_rank, "eer": float(eer),
                    "eer_threshold": float(eer_thr)})
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EECS 700 binary-template face recognition")
    ap.add_argument("--gallery", required=True, help="Path to extracted GallerySet folder")
    ap.add_argument("--probe", required=True, help="Path to extracted ProbeSet folder")
    ap.add_argument("--out", default="results", help="Output folder")
    ap.add_argument("--ctx", type=int, default=-1, help="-1 = CPU, 0 = first GPU")
    ap.add_argument("--bonus", action="store_true", help="Run bonus improvement on weakest condition")
    ap.add_argument("--bonus_bits", type=int, default=4096, help="SimHash length for bonus")
    ap.add_argument("--seed", type=int, default=700)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gal = list_images(args.gallery)
    prb = list_images(args.probe)
    gal = gal[gal["sample"] == "img1"].reset_index(drop=True)
    prb = prb[prb["sample"].isin(CONDITIONS)].reset_index(drop=True)
    print(f"Gallery images: {len(gal)}   Probe images: {len(prb)}")

    # ---------------- Task 1: features + binary templates ----------------
    ext = ArcFaceExtractor(ctx_id=args.ctx)
    print("Extracting gallery features ...")
    gf, g_how = extract(ext, gal, tag="gallery")
    print("Extracting probe features ...")
    pf, p_how = extract(ext, prb, tag="probe")

    pd.DataFrame({"label": list(gal["label"]) + list(prb["label"]),
                  "method": g_how + p_how}).to_csv(out / "detection_log.csv", index=False)

    thresholds = np.median(gf, axis=0)  # learned from gallery only, applied to both sets
    np.save(out / "binarization_thresholds.npy", thresholds)
    gb, pb = binarize(gf, thresholds), binarize(pf, thresholds)
    n_bits = gb.shape[1]

    template_frame(gal, gf, "f").to_csv(out / "GallerySet_features.csv", index=False, float_format="%.6f")
    template_frame(prb, pf, "f").to_csv(out / "ProbeSet_features.csv", index=False, float_format="%.6f")
    template_frame(gal, gb, "b").to_csv(out / "GallerySet_binary.csv", index=False)
    template_frame(prb, pb, "b").to_csv(out / "ProbeSet_binary.csv", index=False)
    write_snippets(gal, prb, gf, pf, gb, pb, out / "snippets_Q1_Q2.txt")

    # ---------------- Tasks 2-6: full probe set ----------------
    A = hamming_matrix(pb, gb)
    save_score_matrix(A, prb["label"], gal["label"], out, "score_matrix_all")
    metrics = {"model": "InsightFace buffalo_l ArcFace (ResNet-50, WebFace600K)",
               "feature_dim": int(gf.shape[1]), "template_bits": int(n_bits),
               "binarization": "bit = 1 if feature > per-dimension gallery median else 0",
               "all": evaluate(A, prb["subject"], gal["subject"], n_bits, out, "all",
                               "All probes vs gallery")}

    # ---------------- Q7: per condition ----------------
    metrics["conditions"] = {}
    for samp, cond in CONDITIONS.items():
        idx = np.where(prb["sample"] == samp)[0]
        Ac = hamming_matrix(pb[idx], gb)
        save_score_matrix(Ac, prb["label"].iloc[idx], gal["label"], out, f"score_matrix_{cond}")
        metrics["conditions"][cond] = evaluate(Ac, prb["subject"].iloc[idx], gal["subject"],
                                               n_bits, out, cond, f"{cond.capitalize()} ({samp}) probes",
                                               full=True)
        metrics["conditions"][cond]["n_probe"] = int(len(idx))
    weakest = min(metrics["conditions"], key=lambda c: metrics["conditions"][c]["d_prime"])
    metrics["weakest_condition"] = weakest

    # ---------------- Bonus ----------------
    if args.bonus:
        samp = {v: k for k, v in CONDITIONS.items()}[weakest]
        print(f"Bonus: improving weakest condition '{weakest}' ({samp}) ...")
        idx = np.where(prb["sample"] == samp)[0]
        sub = prb.iloc[idx].reset_index(drop=True)
        gf2, _ = extract(ext, gal, robust=True, flip_tta=True, tag="bonus-gallery")
        pf2, how2 = extract(ext, sub, robust=True, flip_tta=True, tag=f"bonus-{weakest}")
        mu = gf2.mean(axis=0)                                  # center on gallery mean
        rng = np.random.default_rng(args.seed)
        W = rng.standard_normal((gf2.shape[1], args.bonus_bits)).astype(np.float32)
        gb2 = ((gf2 - mu) @ W > 0).astype(np.uint8)            # SimHash: random hyperplanes
        pb2 = ((pf2 - mu) @ W > 0).astype(np.uint8)
        template_frame(gal, gb2, "b").to_csv(out / "bonus_GallerySet_binary.csv", index=False)
        template_frame(sub, pb2, "b").to_csv(out / f"bonus_ProbeSet_{weakest}_binary.csv", index=False)
        A2 = hamming_matrix(pb2, gb2)
        save_score_matrix(A2, sub["label"], gal["label"], out, f"bonus_score_matrix_{weakest}")
        r = evaluate(A2, sub["subject"], gal["subject"], args.bonus_bits, out, f"bonus_{weakest}",
                     f"BONUS {weakest} probes (flip-TTA + {args.bonus_bits}-bit SimHash)")
        r["delta_d_prime"] = r["d_prime"] - metrics["conditions"][weakest]["d_prime"]
        r["method"] = (f"padded low-threshold re-detection + horizontal-flip TTA + "
                       f"{args.bonus_bits}-bit random-hyperplane hashing (seed {args.seed})")
        metrics["bonus"] = r

    # ---------------- Report ----------------
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    a = metrics["all"]
    lines = [
        f"Model: {metrics['model']}",
        f"Feature dim: {metrics['feature_dim']}   Template bits: {n_bits}",
        f"Detection: {pd.Series(g_how + p_how).value_counts().to_dict()}",
        "",
        f"ALL PROBES ({A.shape[0]} x {A.shape[1]})",
        f"  Genuine  mean/std: {a['genuine_mean']:.4f} / {a['genuine_std']:.4f}",
        f"  Impostor mean/std: {a['impostor_mean']:.4f} / {a['impostor_std']:.4f}",
        f"  d'               : {a['d_prime']:.4f}",
        f"  Rank-1           : {a['rank1'] * 100:.2f}%",
        f"  Max rate {a['max_identification_rate'] * 100:.2f}% first reached at rank {a['lowest_rank_at_max']}",
        f"  EER              : {a['eer'] * 100:.3f}%  (threshold {a['eer_threshold']:.4f})",
        "",
        f"Q7 PER CONDITION ({len(gal)} x {len(gal)} each)",
        f"  {'condition':<10}{'d_prime':>9}{'rank1%':>9}{'EER%':>9}",
    ]
    for c, r in metrics["conditions"].items():
        lines.append(f"  {c:<10}{r['d_prime']:>9.4f}{r['rank1'] * 100:>9.2f}{r['eer'] * 100:>9.3f}")
    lines.append(f"  Weakest: {weakest}")
    if "bonus" in metrics:
        b = metrics["bonus"]
        lines += ["", f"BONUS ({weakest})", f"  revised d' = {b['d_prime']:.4f}",
                  f"  delta d'   = {b['delta_d_prime']:+.4f}  (needs > 1.5)"]
    txt = "\n".join(lines)
    (out / "metrics.txt").write_text(txt)
    print("\n" + txt)
    print(f"\nAll outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
