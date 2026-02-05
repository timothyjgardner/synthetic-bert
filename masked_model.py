"""
BERT-style Masked Time Series Model

Trains a transformer encoder to predict masked patches of the
synthetic-song time series.  The model sees a window of 20D observations
with contiguous blocks masked out, and learns to reconstruct the missing
portions from the surrounding context.

Architecture
------------
  Input (batch, seq_len, 20)
    → replace masked positions with learnable [MASK] embedding
    → Linear projection to d_model
    → add sinusoidal positional encoding
    → N × TransformerEncoderLayer (self-attention + FFN)
    → Linear projection back to 20
    → MSE loss on masked positions only

Usage
-----
    python masked_model.py                        # train with defaults
    python masked_model.py --epochs 100           # custom epochs
    python masked_model.py --eval bert_model.pt   # evaluate & visualize
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import SyntheticSongDataset


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MaskedTimeSeriesBERT(nn.Module):
    """
    BERT-style masked prediction model for continuous time series.

    Parameters
    ----------
    feature_dim : int
        Dimension of each time step (20 for our synthetic data).
    d_model : int
        Internal transformer dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer encoder layers.
    d_ff : int
        Feed-forward hidden dimension.
    dropout : float
        Dropout rate.
    max_len : int
        Maximum sequence length.
    """

    def __init__(
        self,
        feature_dim=20,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        max_len=2048,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        # Learnable mask embedding (replaces masked positions before projection)
        self.mask_token = nn.Parameter(torch.randn(feature_dim))

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output projection back to feature space
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, feature_dim),
        )

    def forward(self, x, mask):
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim) – original observations
        mask : (batch, seq_len) – bool, True = masked (to predict)

        Returns
        -------
        pred : (batch, seq_len, feature_dim) – predicted values
        """
        # Replace masked positions with the learnable mask token
        x_masked = x.clone()
        x_masked[mask] = self.mask_token

        # Project to transformer dimension
        h = self.input_proj(x_masked)

        # Add positional encoding
        h = self.pos_enc(h)

        # Transformer encoder (self-attention over the full sequence)
        h = self.transformer(h)

        # Project back to feature space
        pred = self.output_proj(h)
        return pred


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse_loss(pred, target, mask):
    """MSE loss computed only on masked positions."""
    mask_expanded = mask.unsqueeze(-1).expand_as(pred)
    n_masked = mask_expanded.sum()
    if n_masked == 0:
        return torch.tensor(0.0, device=pred.device)
    loss = ((pred - target) ** 2 * mask_expanded).sum() / n_masked
    return loss


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, state, mask in loader:
        x = x.to(device)
        mask = mask.to(device)

        pred = model(x, mask)
        loss = masked_mse_loss(pred, x, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, state, mask in loader:
        x = x.to(device)
        mask = mask.to(device)

        pred = model(x, mask)
        loss = masked_mse_loss(pred, x, mask)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_predictions(model, dataset, device, n_samples=3, save_dir='.'):
    """Generate prediction-vs-ground-truth plots for sample windows."""
    model.eval()
    save_dir = Path(save_dir)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)),
                         replace=False)

    for i, idx in enumerate(indices):
        x, state, mask = dataset[idx]
        x_in = x.unsqueeze(0).to(device)
        mask_in = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x_in, mask_in)

        x_np = x.numpy()                  # (seq_len, 20)
        pred_np = pred[0].cpu().numpy()    # (seq_len, 20)
        mask_np = mask.numpy()             # (seq_len,)
        state_np = state.numpy()           # (seq_len,)
        seq_len = x_np.shape[0]

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            4, 2, height_ratios=[1, 4, 4, 4],
            width_ratios=[1, 0.02], hspace=0.15, wspace=0.03,
        )

        # -- State strip with mask overlay --
        ax_top = fig.add_subplot(gs[0, 0])
        for t in range(seq_len):
            ax_top.axvspan(t, t + 1,
                           color=plt.cm.tab10(state_np[t] / 10),
                           alpha=0.8, linewidth=0)
        for t in range(seq_len):
            if mask_np[t]:
                ax_top.axvspan(t, t + 1, color='grey', alpha=0.5,
                               linewidth=0)
        ax_top.set_xlim(0, seq_len)
        ax_top.set_yticks([])
        ax_top.set_ylabel('State', fontsize=9)
        ax_top.set_title(f'Prediction sample {i}  (grey = masked)',
                         fontsize=12)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # -- Ground truth heatmap --
        ax_gt = fig.add_subplot(gs[1, 0], sharex=ax_top)
        vmax = max(abs(x_np.min()), abs(x_np.max()))
        im = ax_gt.imshow(
            x_np.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[0, seq_len, x_np.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_gt.set_ylabel('Dimension', fontsize=9)
        ax_gt.set_title('Ground truth', fontsize=10)
        plt.setp(ax_gt.get_xticklabels(), visible=False)

        # -- Prediction heatmap --
        ax_pred = fig.add_subplot(gs[2, 0], sharex=ax_top)
        ax_pred.imshow(
            pred_np.T, aspect='auto', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[0, seq_len, pred_np.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_pred.set_ylabel('Dimension', fontsize=9)
        ax_pred.set_title('Model prediction', fontsize=10)
        plt.setp(ax_pred.get_xticklabels(), visible=False)

        # -- Error heatmap (masked positions only) --
        error = np.zeros_like(x_np)
        error[mask_np] = pred_np[mask_np] - x_np[mask_np]
        ax_err = fig.add_subplot(gs[3, 0], sharex=ax_top)
        err_max = max(abs(error.min()), abs(error.max()), 1e-6)
        ax_err.imshow(
            error.T, aspect='auto', cmap='RdBu_r',
            vmin=-err_max, vmax=err_max,
            extent=[0, seq_len, error.shape[1] - 0.5, -0.5],
            interpolation='none',
        )
        ax_err.set_xlabel('Time step', fontsize=9)
        ax_err.set_ylabel('Dimension', fontsize=9)
        ax_err.set_title('Prediction error (masked positions only)',
                         fontsize=10)

        # Colorbars / dummy axes
        ax_cb = fig.add_subplot(gs[1, 1])
        plt.colorbar(im, cax=ax_cb)
        for row in [0, 2, 3]:
            fig.add_subplot(gs[row, 1]).axis('off')

        fname = save_dir / f'bert_prediction_{i}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Training loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(train_losses, val_losses, save_path='training_loss.png'):
    """Save a training/validation loss curve plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train MSE', linewidth=2)
    ax.plot(epochs, val_losses, label='Val MSE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Masked MSE Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a BERT-style masked prediction model on '
                    'synthetic-song data.')
    # Data
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing data.npz and config.json')
    parser.add_argument('--seq-len', type=int, default=512,
                        help='Sequence length per window')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride between windows')
    parser.add_argument('--mask-ratio', type=float, default=0.15,
                        help='Fraction of time steps to mask')
    parser.add_argument('--mask-patch-size', type=int, default=16,
                        help='Contiguous patch size for masking')
    # Model
    parser.add_argument('--d-model', type=int, default=128,
                        help='Transformer model dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=512,
                        help='Feed-forward hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Fraction of data for validation')
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default='bert_model.pt',
                        help='Path to save model checkpoint')
    parser.add_argument('--eval', type=str, default=None, metavar='CKPT',
                        help='Evaluate and visualize from a checkpoint '
                             '(skip training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # ---- Device ----
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Dataset ----
    full_ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        mask_seed=None,  # random masking each epoch for augmentation
    )

    # Train / val split
    n_val = max(1, int(len(full_ds) * args.val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    print(f"Dataset: {len(full_ds)} windows  "
          f"(train={n_train}, val={n_val})")
    print(f"  seq_len={args.seq_len}, stride={args.stride}, "
          f"feature_dim={full_ds.feature_dim}")
    print(f"  mask_ratio={args.mask_ratio}, "
          f"patch_size={args.mask_patch_size}")

    # ---- Model ----
    model = MaskedTimeSeriesBERT(
        feature_dim=full_ds.feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.seq_len + 64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters  "
          f"(d_model={args.d_model}, layers={args.n_layers}, "
          f"heads={args.n_heads})")

    # ---- Eval-only mode ----
    if args.eval is not None:
        ckpt = torch.load(args.eval, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        val_loss = evaluate(model, val_loader, device)
        print(f"\nValidation MSE: {val_loss:.4f}")

        viz_ds = SyntheticSongDataset(
            data_dir=args.data_dir,
            seq_len=args.seq_len,
            stride=args.stride,
            mask_ratio=args.mask_ratio,
            mask_patch_size=args.mask_patch_size,
            mask_seed=123,
        )
        visualize_predictions(model, viz_ds, device, n_samples=3)
        return

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ---- Training loop ----
    best_val = float('inf')
    train_losses, val_losses = [], []

    print(f"\n{'Epoch':<7} {'Train MSE':<12} {'Val MSE':<12} "
          f"{'LR':<12} {'Best'}")
    print('-' * 55)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args),
            }, args.checkpoint)

        marker = ' *' if is_best else ''
        print(f"{epoch:<7} {train_loss:<12.4f} {val_loss:<12.4f} "
              f"{lr:<12.6f}{marker}")

    print(f"\nBest val MSE: {best_val:.4f}")
    print(f"Checkpoint saved to {args.checkpoint}")

    # ---- Loss curve ----
    plot_loss_curve(train_losses, val_losses)

    # ---- Visualize with best model ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])

    viz_ds = SyntheticSongDataset(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        mask_ratio=args.mask_ratio,
        mask_patch_size=args.mask_patch_size,
        mask_seed=123,  # deterministic for reproducible visualisation
    )
    visualize_predictions(model, viz_ds, device, n_samples=3)


if __name__ == '__main__':
    main()
