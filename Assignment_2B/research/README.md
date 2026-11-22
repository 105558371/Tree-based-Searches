# Research models (2B)

Six spatio-temporal models trained on the same data, split and metrics, so they can be compared fairly.
Run commands from `Assignment_2B/`. Separate from `core/tune.py` (which tunes the basic LSTM/GRU/GNN).

| Step | Command |
|---|---|
| Build dataset (from `../data`) | `python research/prepare_data.py` → `research/cache/traffic.npz` (not committed, ~20 s to rebuild) |
| Train a model | `python research/train.py --model {gwn,stgcn,agcrn,mtgnn,stid,staeformer} --epochs 30` |

Outputs: `outputs/weights/research/<model>.pt` (best validation epoch), `outputs/logs/research/<model>_metrics.json` and `<model>_train_log.txt`.

- **Task:** last 12 steps (3 h) at all 40 sites → next 4 steps (15/30/45/60 min)
- **Inputs:** normalised flow + time of day
- **Split by time:** train 2014–2023, validate 2024, test 2025-01-01 → 2026-11-16
- **Gaps:** ≤1 h interpolated, longer gaps use the site's weekday/time-of-day average (training years only).
  Loss and metrics ignore filled values; MAPE skips flows under 10 vehicles.
- **Training:** 30 epochs × 60,000 random training windows, Adam lr 1e-3, batch 64,
  early stopping (patience 6), Apple M2 (MPS)

## Files

| File | Contents |
|---|---|
| `prepare_data.py` | Shared dataset + distance-based adjacency (Gaussian kernel on lat/long) |
| `train.py` | Training, evaluation and the model registry |
| `graph_wavenet.py` | Graph WaveNet |
| `gnn_models.py` | STGCN, AGCRN, MTGNN |
| `more_models.py` | STID, STAEformer |

## Results (test 2025–2026, averaged over the 4 horizons; full table in `outputs/logs/research/comparison.md`)

| Model | Type | MAE | RMSE | MAPE | Params | Train time |
|---|---|---|---|---|---|---|
| STAEformer | Transformer | **28.30** | **45.76** | **10.61%** | 842k | 138 min |
| Graph WaveNet | GNN | 28.70 | 46.28 | 10.91% | 294k | 56 min |
| AGCRN | GNN | 28.88 | 46.72 | 10.82% | 749k | 46 min |
| STGCN | GNN | 29.79 | 47.93 | 10.94% | 160k | 29 min |
| MTGNN | GNN | 29.86 | 47.87 | 11.24% | 163k | 15 min |
| STID | MLP | 31.05 | 50.93 | 11.62% | 61k | 2 min |
| Last value | Baseline | 58.74 | 91.46 | — | — | — |

## References

- Wu et al. (2019) *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*, IJCAI — github.com/nnzhan/Graph-WaveNet
- Yu et al. (2018) *Spatio-Temporal Graph Convolutional Networks* (STGCN), IJCAI
- Bai et al. (2020) *Adaptive Graph Convolutional Recurrent Network* (AGCRN), NeurIPS
- Wu et al. (2020) *Connecting the Dots: Multivariate Time Series Forecasting with GNNs* (MTGNN), KDD
- Shao et al. (2022) *Spatial-Temporal Identity: A Simple yet Effective Baseline* (STID), CIKM
- Liu et al. (2023) *STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA*, CIKM
