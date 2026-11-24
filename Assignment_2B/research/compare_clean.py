"""Score the final ensemble trained on raw data and the one trained with --clean on two label sets:
the original test labels (every observed value, faults included) and the cleaned labels (faults removed).

Usage (from Assignment_2B/): python research/compare_clean.py
"""
import json, sys, pathlib
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import train, ensemble

F = "staeformer_ext-lag-wk4-diff-roll-cyc-event"
SETS = {"raw-trained": [f"{F}_final_s{s}" for s in range(3)],
        "clean-trained": [f"{F}_clean_final_s{s}" for s in range(3)]}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(train.ROOT / "cache" / "traffic.npz", allow_pickle=True)
    fault = np.load(train.ROOT / "cache" / "fault_mask.npy")
    y = torch.as_tensor(d["flow"], device=dev)
    masks = {"original labels": torch.as_tensor(d["mask"], device=dev),
             "cleaned labels": torch.as_tensor(d["mask"] & ~fault, device=dev)}
    cache, out = {}, {}
    for name, runs in SETS.items():
        preds = []
        for r in runs:
            p, _, windows = ensemble.member_predictions(r, dev, cache)
            preds.append(p)
        p = torch.stack(preds).mean(0)
        for lab, m in masks.items():
            data = {"flow": y, "mask": m}           # score() reads only flow and mask
            s = ensemble.score(p, data, windows)
            out[f"{name} | {lab}"] = s
            print(f"{name:14s} {lab:16s} avg MAE {ensemble.avg_mae(s):.3f}  "
                  + "  ".join(f"{h} {v['MAE']:.2f}" for h, v in s.items()))
    (train.LOGS / f"{F}_clean_comparison.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
