import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import DiffusionDenoiser
from dataset import StockDataset
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from plot import plot_training_curves
import config


SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)


def train_one_epoch(denoiser, loader, optimizer, device, scaler=None):
    denoiser.train()
    total_loss = 0
    n = 0
    for x_dyn, x_stat, y in loader:
        x_dyn = x_dyn.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            loss = denoiser.compute_loss(x_dyn)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(x_dyn)
        n += len(x_dyn)
    return total_loss / n


@torch.no_grad()
def evaluate(denoiser, loader, device):
    denoiser.eval()
    total_loss = 0
    n = 0
    for x_dyn, x_stat, y in loader:
        x_dyn = x_dyn.to(device)
        loss = denoiser.compute_loss(x_dyn)
        total_loss += loss.item() * len(x_dyn)
        n += len(x_dyn)
    return total_loss / n


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    df = build_merged_dataset(start_date=config.TRAIN_START,
                              end_date=config.VAL_END)
    print("Building features...")
    df, avail_features = build_features(df)

    train_df = df[df['trade_date'] <= config.TRAIN_END]
    val_df = df[df['trade_date'] >= config.VAL_START]
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    print("Building datasets...")
    train_ds = StockDataset(train_df, avail_features, STATIC_FEATURES,
                            cache_tag="train")
    val_ds = StockDataset(val_df, avail_features, STATIC_FEATURES,
                          cache_tag="val")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    denoiser = DiffusionDenoiser(
        feature_dim=len(avail_features),
        seq_len=config.SEQ_LEN,
        hidden_dim=config.DIFFUSION_HIDDEN_DIM,
        time_dim=config.DIFFUSION_TIME_DIM,
        n_timesteps=config.DIFFUSION_T,
        beta_start=config.DIFFUSION_BETA_START,
        beta_end=config.DIFFUSION_BETA_END,
    ).to(device)

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    save_path = os.path.join(config.CACHE_DIR, "denoiser_pretrained.pt")
    history = {'train_loss': [], 'val_loss': [], 'ic': [], 'icir': []}

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(denoiser, train_loader, optimizer,
                                     device, scaler)
        val_loss = evaluate(denoiser, val_loader, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['ic'].append(0.0)
        history['icir'].append(0.0)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'denoiser_state': denoiser.state_dict(),
                'config': {
                    'SEQ_LEN': config.SEQ_LEN,
                    'DIFFUSION_T': config.DIFFUSION_T,
                    'DIFFUSION_HIDDEN_DIM': config.DIFFUSION_HIDDEN_DIM,
                    'DIFFUSION_TIME_DIM': config.DIFFUSION_TIME_DIM,
                    'DIFFUSION_BETA_START': config.DIFFUSION_BETA_START,
                    'DIFFUSION_BETA_END': config.DIFFUSION_BETA_END,
                    'DYNAMIC_FEATURES': avail_features,
                },
            }, save_path)
            print(f"  -> Saved best denoiser (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping.")
                break

    plot_training_curves(history)
    print(f"\nTraining done. Best val_loss: {best_val_loss:.6f}")
    print(f"Denoiser saved to: {save_path}")


if __name__ == "__main__":
    main()
