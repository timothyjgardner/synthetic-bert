# Synthetic Song

Synthetic syllable-based time series on circles in high-dimensional space, with Markov switching dynamics and Levina-Bickel intrinsic dimension estimation.

## Overview

This project generates a time series that mimics syllable-structured sequential data (like birdsong). A point traverses one of 10 circles embedded in 20-dimensional space, switching between circles according to a sparse Markov transition matrix. Each circle has a fixed angular velocity, producing distinct oscillation frequencies that serve as a signature for each "syllable."

### Key properties

- **10 circles** in 20D ambient space, each in its own random 2D sub-plane, all sharing the same center (origin)
- **Statistically similar radii** (~20), large enough that the circular signal spans all dimensions
- **Fixed angular velocity per circle**, with periods ranging from 40 steps (fastest) to 400 steps (slowest) — a 10x speed range
- **~400 step dwell time** per visit, achieved by varying the number of complete revolutions (quantised to whole laps so entry/exit angles are fixed)
- **Sparse off-diagonal transition matrix** with ring connectivity plus long-range shortcuts

## Scripts

### `markov_circles_timeseries.py`

Generates the Markov-switching circle time series with optional UMAP visualisation.

```bash
python markov_circles_timeseries.py            # full run with UMAP
python markov_circles_timeseries.py --no-umap  # skip UMAP (much faster)
```

### `estimate_dimension.py`

Runs the Levina-Bickel MLE dimension estimator on the saved synthetic-song dataset at various k values, both globally and per-circle.

```bash
python estimate_dimension.py
```

### `levina_bickel_demo.py`

Demonstrates the Levina-Bickel MLE intrinsic dimension estimator on a single noisy circle, showing how the estimate depends on the neighbourhood scale `k` and noise level.

```bash
python levina_bickel_demo.py
```

## Results

### Markov-switching time series summary

![Markov circles time series](markov_circles_timeseries.png)

**Top row:** Markov state sequence over time, sparse transition matrix, and per-circle period (steps per revolution).
**Bottom row:** UMAP embedding coloured by circle index and by time step. Slow circles (long period) form coherent loops; fast circles appear as scattered points because consecutive time steps are far apart on the circle.

### Sample data windows

Raw 20-dimensional time series with state labels. Each column is one time step; each row is one ambient dimension. The coloured strip at top shows which circle is active.

![Sample window 0](sample_window_0.png)

![Sample window 1](sample_window_1.png)

![Sample window 2](sample_window_2.png)

### Dimension estimation on the synthetic-song dataset

![Dimension estimates](dimension_estimates.png)

Levina-Bickel MLE intrinsic dimension estimates computed on 2000 subsampled points from the synthetic-song dataset (SNR ≈ 2.5).

There are two ways to measure the dimension of this dataset: **globally** (all points pooled together) and **per-circle** (only points from one circle at a time). These give very different answers, and the difference is informative.

**Left — Global dimension vs k (all circles pooled):**

| k | Estimated dimension |
|---|---|
| 5 | 15.3 |
| 10 | 11.7 |
| 20 | 9.3 |
| 50 | 6.9 |
| 100 | 6.6 |
| 200 | 7.2 |

When all 10 circles are pooled, the estimator sees a **mixture of 10 different 2D sub-planes** in 20D space, plus noise. This is not a single 2-dimensional manifold — it is a union of manifolds that collectively span a higher-dimensional subspace. The global estimate bottoms out around ~6.6, reflecting this multi-manifold structure:

- **Small k (5–20):** Neighbourhoods are dominated by noise, which is isotropic in all 20 dimensions. The estimate inflates toward the ambient dimension (20).
- **Intermediate k (50–100):** Neighbourhoods are large enough to average out noise but still mostly sample from a single circle. The estimate reaches a minimum (~6.6), which exceeds 2 because some neighbourhoods straddle transitions between circles that lie in different 2D sub-planes.
- **Large k (150–200):** Neighbourhoods span points from multiple circles in different sub-planes. The union of several 2D sub-planes spans an increasingly high-dimensional subspace, pushing the estimate back up.

**Right — Per-circle dimension at selected k values:**

| Circle | Period | k=10 | k=30 | k=100 |
|--------|--------|------|------|-------|
| 0 | 40 | 11.3 | 7.2 | 2.8 |
| 1 | 80 | 11.3 | 7.3 | 3.0 |
| 5 | 240 | 12.3 | 8.3 | 3.4 |
| 9 | 400 | 10.7 | 6.4 | 2.7 |

When we restrict to points from a **single circle**, the estimator sees exactly one 2D sub-plane plus noise. At k=100, all circles converge to ~2.5–3.5, close to the true manifold dimension of 2. The residual above 2 comes from observation noise inflating the local dimension estimate.

**Why the global and per-circle estimates differ:** The global estimate is higher because the Levina-Bickel estimator assumes data lies on a single smooth manifold. When the data is actually a *mixture* of manifolds in different sub-planes, neighbourhoods that mix points from different circles see a higher effective dimension — the dimension of the union, not any individual component. Restricting to one circle at a time eliminates this mixing and recovers the true per-manifold dimension.

### Levina-Bickel demo (single noisy circle)

![Levina-Bickel results](levina_bickel_results.png)

Estimated intrinsic dimension vs neighbourhood size `k` for a single circle in 10D with varying noise levels. Without noise the estimator correctly finds dimension ~1. With noise, small `k` overestimates (noise dominates) and large `k` recovers the manifold.

## Requirements

```
numpy
scipy
matplotlib
umap-learn
```

Install into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy scipy matplotlib umap-learn
```
