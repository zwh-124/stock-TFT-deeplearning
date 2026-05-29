import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CompetitionTFT, DiffusionDenoiser
from dataset import StockDataset
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from plot import plot_training_curves, plot_ic_distribution, plot_feature_importance
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


def compute_daily_ic(predictions, targets, dates):
    daily_ics = []
    unique_dates = np.unique(dates)
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < 10:
            continue
        pred_d = predictions[mask]
        tgt_d = targets[mask]
        if np.std(pred_d) < 1e-8 or np.std(tgt_d) < 1e-8:
            continue
        ic = np.corrcoef(pred_d, tgt_d)[0, 1]
        if not np.isnan(ic):
            daily_ics.append(ic)
    if len(daily_ics) == 0:
        return 0.0, 0.0, []
    ic_mean = np.mean(daily_ics)
    ic_std = np.std(daily_ics) + 1e-8
    return ic_mean, ic_mean / ic_std, daily_ics


def compute_direction_accuracy(predictions, targets, dates):
    unique_dates = np.unique(dates)
    daily_accs = []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < 10:
            continue
        pred_d = predictions[mask]
        tgt_d = targets[mask]
        correct = ((pred_d > 0) == (tgt_d > 0)).mean()
        daily_accs.append(correct)
    if not daily_accs:
        return 0.0
    return np.mean(daily_accs)


def gated_feature_loss(pred, gate_weights, target):
    """Gated multi-feature loss with variance normalization."""
    per_feature_mse = (pred - target) ** 2  # (B, F)
    feat_scale = (target ** 2).mean(dim=0).clamp(min=1e-6)  # (F,)
    norm_mse = per_feature_mse / feat_scale

    weighted_loss = (gate_weights * norm_mse).sum(dim=-1)
    entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1)
    return weighted_loss.mean() - config.LAMBDA_ENTROPY * entropy.mean()


def train_one_epoch(model, loader, optimizer, device,
                    denoiser=None, denoiser_optimizer=None,
                    scaler=None):
    model.train()
    if denoiser:
        denoiser.train()
    total_loss = 0
    n = 0
    for x_dyn, x_stat, y in loader:
        x_dyn = x_dyn.to(device)
        x_stat = x_stat.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        if denoiser_optimizer:
            denoiser_optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred, gate_weights = model(x_dyn, x_stat)
            loss = gated_feature_loss(pred, gate_weights, y)
            if denoiser is not None:
                d_loss = denoiser.compute_loss(x_dyn)
                combined = loss + config.LAMBDA_DENOISE * d_loss
            else:
                combined = loss
        if scaler is not None:
            scaler.scale(combined).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            if denoiser_optimizer:
                scaler.step(denoiser_optimizer)
            scaler.update()
        else:
            combined.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if denoiser_optimizer:
                denoiser_optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, dataset, close_idx):
    """Evaluate model. Requires loader with shuffle=False for date tracking."""
    model.eval()
    total_loss = 0
    n = 0
    all_preds, all_targets, all_dates = [], [], []
    idx = 0
    for x_dyn, x_stat, y in loader:
        x_dyn = x_dyn.to(device)
        x_stat = x_stat.to(device)
        y = y.to(device)
        pred, gate_weights = model(x_dyn, x_stat)
        loss = gated_feature_loss(pred, gate_weights, y)
        total_loss += loss.item() * len(y)
        n += len(y)
        all_preds.append(pred[:, close_idx].cpu().numpy())
        all_targets.append(y[:, close_idx].cpu().numpy())
        batch_dates = [dataset.samples[idx + i]['date']
                       for i in range(len(y))]
        all_dates.extend(batch_dates)
        idx += len(y)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_dates = np.array(all_dates)
    ic, icir, daily_ics = compute_daily_ic(all_preds, all_targets, all_dates)
    dir_acc = compute_direction_accuracy(all_preds, all_targets, all_dates)
    return total_loss / n, ic, icir, dir_acc, daily_ics


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

    model = CompetitionTFT(
        dynamic_input_dim=len(avail_features),
        static_input_dim=len(STATIC_FEATURES),
        hidden_dim=config.HIDDEN_DIM,
        seq_len=config.SEQ_LEN,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
        static_categorical=STATIC_CATEGORICAL,
        static_n_continuous=len(STATIC_CONTINUOUS),
        avail_features=avail_features,
    ).to(device)

    denoiser = None
    denoiser_optimizer = None
    if config.USE_DIFFUSION_DENOISER:
        denoiser = DiffusionDenoiser(
            feature_dim=len(avail_features),
            seq_len=config.SEQ_LEN,
            hidden_dim=config.DIFFUSION_HIDDEN_DIM,
            time_dim=config.DIFFUSION_TIME_DIM,
            n_timesteps=config.DIFFUSION_T,
            beta_start=config.DIFFUSION_BETA_START,
            beta_end=config.DIFFUSION_BETA_END,
        ).to(device)
        denoiser_optimizer = torch.optim.Adam(
            denoiser.parameters(), lr=config.LR)
        model.encoder.denoiser = denoiser

    encoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.denoiser.")]
    optimizer = torch.optim.Adam(encoder_params, lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS)

    close_idx = avail_features.index('close')
    best_val_loss = np.inf
    patience_counter = 0
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    model_path = os.path.join(config.CACHE_DIR, "best_model.pt")
    history = {'train_loss': [], 'val_loss': [], 'ic': [], 'icir': [],
                'dir_acc': [], 'daily_ics': []}

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            denoiser, denoiser_optimizer, scaler)
        val_loss, val_ic, val_icir, val_dir_acc, val_daily_ics = evaluate(
            model, val_loader, device, val_ds, close_idx)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['ic'].append(val_ic)
        history['icir'].append(val_icir)
        history['dir_acc'].append(val_dir_acc)
        history['daily_ics'] = val_daily_ics

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"IC: {val_ic:.4f} | ICIR: {val_icir:.4f} | "
              f"DirAcc: {val_dir_acc:.4f} | "
              f"LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'state_dict': model.state_dict(),
                'denoiser_state': denoiser.state_dict() if denoiser else None,
                'config': {
                    'USE_CSI300': config.USE_CSI300,
                    'HIDDEN_DIM': config.HIDDEN_DIM,
                    'NUM_HEADS': config.NUM_HEADS,
                    'SEQ_LEN': config.SEQ_LEN,
                    'PRED_HORIZON': config.PRED_HORIZON,
                    'DROPOUT': config.DROPOUT,
                    'DYNAMIC_FEATURES': avail_features,
                    'STATIC_FEATURES': list(STATIC_FEATURES),
                    'STATIC_CATEGORICAL': STATIC_CATEGORICAL,
                    'STATIC_CONTINUOUS': STATIC_CONTINUOUS,
                },
            }, model_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping.")
                break

    plot_training_curves(history)

    if 'daily_ics' in history and history['daily_ics']:
        plot_ic_distribution(history['daily_ics'])

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    with torch.no_grad():
        sample_x_dyn, sample_x_stat, _ = next(iter(val_loader))
        sample_x_dyn = sample_x_dyn.to(device)
        sample_x_stat = sample_x_stat.to(device)
        _, gate_w = model(sample_x_dyn, sample_x_stat)
        avg_gate = gate_w.cpu().numpy().mean(axis=0)
    plot_feature_importance(avg_gate, avail_features)

    print(f"\nTraining done. Best val_loss: {best_val_loss:.6f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
