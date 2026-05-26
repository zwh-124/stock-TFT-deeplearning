import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CompetitionTFT
from dataset import StockDataset
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
import config


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
        return 0.0, 0.0
    ic_mean = np.mean(daily_ics)
    ic_std = np.std(daily_ics) + 1e-8
    return ic_mean, ic_mean / ic_std


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n = 0
    for x_dyn, x_stat, y in loader:
        x_dyn = x_dyn.to(device)
        x_stat = x_stat.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x_dyn, x_stat)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, dataset):
    model.eval()
    total_loss = 0
    n = 0
    all_preds, all_targets, all_dates = [], [], []
    idx = 0
    for x_dyn, x_stat, y in loader:
        x_dyn = x_dyn.to(device)
        x_stat = x_stat.to(device)
        y = y.to(device)
        pred = model(x_dyn, x_stat)
        loss = criterion(pred, y)
        total_loss += loss.item() * len(y)
        n += len(y)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        batch_dates = [dataset.samples[idx + i]['date']
                       for i in range(len(y))]
        all_dates.extend(batch_dates)
        idx += len(y)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_dates = np.array(all_dates)
    ic, icir = compute_daily_ic(all_preds, all_targets, all_dates)
    return total_loss / n, ic, icir


def main():
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
                              shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = CompetitionTFT(
        dynamic_input_dim=len(avail_features),
        static_input_dim=len(STATIC_FEATURES),
        hidden_dim=config.HIDDEN_DIM,
        seq_len=config.SEQ_LEN,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS)
    criterion = nn.MSELoss()

    best_ic = -np.inf
    patience_counter = 0
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    model_path = os.path.join(config.CACHE_DIR, "best_model.pt")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_ic, val_icir = evaluate(
            model, val_loader, criterion, device, val_ds)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"IC: {val_ic:.4f} | ICIR: {val_icir:.4f} | "
              f"LR: {lr:.2e}")

        if val_ic > best_ic:
            best_ic = val_ic
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  -> Saved best model (IC={best_ic:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print("Early stopping.")
                break

    print(f"\nTraining done. Best IC: {best_ic:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
