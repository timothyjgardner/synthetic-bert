# Synthetic BERT

BERT-style masked prediction on synthetic syllable time series — circles in high-dimensional space with Markov switching dynamics.

## Dataset Generation

The dataset is a synthetic time series that mimics syllable-structured sequential data (like birdsong). A point traverses one of 10 circles embedded in 20-dimensional space, switching between circles according to a sparse Markov transition matrix.

**Key properties:**

- **10 circles** in 20D ambient space, each in its own random 2D sub-plane, all sharing the same center (origin)
- **Fixed angular velocity per circle**, with periods from 40 steps (fastest) to 400 steps (slowest) — a 10x speed range
- **~400 step dwell time** per visit, achieved by varying the number of complete revolutions (quantised to whole laps so entry/exit angles are fixed)
- **Sparse off-diagonal transition matrix** with ring connectivity plus long-range shortcuts
- **Controllable geometric overlap** via `--subspace-dim` (see below)
- **SNR ≈ 2.5** — isotropic Gaussian noise added to every observation

```bash
python markov_circles_timeseries.py                    # generate with UMAP
python markov_circles_timeseries.py --no-umap          # skip UMAP (faster)
python markov_circles_timeseries.py --subspace-dim 4   # with geometric overlap
```

### Subspace overlap control

The `--subspace-dim` flag forces all circle planes into a shared subspace, controlling how much trajectories overlap:

| `--subspace-dim` | Behaviour |
|---|---|
| **20** (default) | Planes span full 20D — **minimal overlap** |
| **4–6** | Planes share directions — **significant overlap** |
| **2** | All circles coplanar — **maximum overlap** |

### Sample data windows

Raw 20-dimensional time series with state labels. Each row is one ambient dimension; the coloured strip at top shows which circle is active.

![Sample window 0](sample_window_0.png)

![Sample window 1](sample_window_1.png)

![Sample window 2](sample_window_2.png)

### UMAP of raw data at different subspace dimensions

#### subspace_dim = 20 (no overlap)

![UMAP subspace 20](umap_subspace_20.png)

With the full 20D ambient space, UMAP cleanly separates all 10 circles into distinct clusters. Levina-Bickel dimension in UMAP space drops to ~1.3 at k=100 — the estimator sees isolated 1D curves.

#### subspace_dim = 4 (significant overlap)

![UMAP subspace 4](umap_subspace_4.png)

With 10 circle planes crammed into a 4D subspace, UMAP can no longer fully separate them — clusters merge and trajectories intermingle. Levina-Bickel stays near ~1.9 at k=100 because the overlapping circles fill the 2D UMAP plane more uniformly.

| Metric | subspace_dim=20 | subspace_dim=4 |
|---|---|---|
| LB dim (k=10) | 2.25 | 2.27 |
| LB dim (k=30) | 1.79 | 2.07 |
| LB dim (k=100) | 1.31 | 1.94 |

## BERT Masked Prediction Model

A transformer encoder is trained to predict masked patches of the time series from surrounding context. This is a continuous analogue of BERT — instead of masking discrete tokens, we mask contiguous 16-step patches of the 20D signal and train the model to reconstruct them.

### Architecture

```
Input (batch, 512, 20)
  → replace masked positions with learnable [MASK] embedding
  → Linear(20 → 128)
  → sinusoidal positional encoding
  → 4 × TransformerEncoderLayer (4 heads, 512-dim FFN, GELU)
  → Linear(128 → 512 → 20)
  → MSE loss on masked positions only
```

872K parameters. 15% of time steps masked per window in contiguous patches of 16.

### Training

```bash
python masked_model.py --epochs 500          # train
python masked_model.py --eval bert_model.pt  # evaluate & visualize
```

The model uses AdamW with linear warmup (20 epochs) followed by cosine decay, trained on sliding windows with stride 128.

![Training loss](training_loss.png)

The training loss drops from ~27 (baseline: predicting zero) to ~8.8, which is close to the **noise floor of ~8.0** (noise_std² = 2.83² ≈ 8). This means the model has learned to reconstruct the noiseless circular signal nearly perfectly — the remaining error is irreducible observation noise.

### Masked predictions

The model fills in masked patches (grey regions in the state strip) using only the surrounding unmasked context:

![BERT prediction 0](bert_prediction_0.png)

![BERT prediction 1](bert_prediction_1.png)

![BERT prediction 2](bert_prediction_2.png)

Each figure shows four rows: (1) state labels with masked regions in grey, (2) ground truth heatmap, (3) model prediction, (4) prediction error on masked positions only. The model accurately reconstructs the oscillatory structure through the masked regions.

## Learned Representations

After training, we extract the intermediate representations from each transformer layer by running the full dataset through the model without masking. UMAP reveals how the model organises the data internally.

```bash
python evaluate_representations.py --checkpoint bert_model.pt
```

![Representation UMAP](representation_umap.png)

**Input (20D):** The raw data — circles overlap due to subspace_dim=4 and noise.

**Layer 1–2:** The early transformer layers begin to separate circles and denoise the signal.

**Layer 3–4:** The deeper layers learn increasingly structured representations. The model discovers a lower-dimensional organisation of the 10 circles, clustering points by their circle identity and phase — exactly the information needed to predict masked patches.

The progression from input to layer 4 shows the transformer learning to untangle the overlapping circles into a cleaner geometric structure, despite never being given explicit circle labels during training.

## Scripts

| Script | Description |
|---|---|
| `markov_circles_timeseries.py` | Generate the synthetic time series dataset |
| `dataset.py` | PyTorch Dataset with sliding windows and patch masking |
| `masked_model.py` | BERT-style masked prediction model (train & eval) |
| `evaluate_representations.py` | Extract and visualise intermediate representations |
| `estimate_dimension.py` | Levina-Bickel intrinsic dimension estimation |
| `levina_bickel_demo.py` | Single-circle dimension estimation demo |

## Requirements

```
numpy
scipy
matplotlib
umap-learn
torch
```

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib umap-learn torch
```
