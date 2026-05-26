import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from model import CompetitionTFT
from dataset import StockDataset
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
import config


@torch.no_grad()
def predict_latest(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    all_preds = []
    for x_dyn, x_stat, _ in loader:
        x_dyn = x_dyn.to(device)
        x_stat = x_stat.to(device)
        pred = model(x_dyn, x_stat)
        all_preds.append(pred.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    records = []
    for i, s in enumerate(dataset.samples):
        records.append({
            'date': s['date'],
            'code': s['code'],
            'pred_score': all_preds[i],
        })
    return pd.DataFrame(records)


def get_signals(pred_df, n_hold=config.N_HOLD, k_swap=config.K_SWAP,
                current_portfolio=None):
    latest_date = pred_df['date'].max()
    day_pred = pred_df[pred_df['date'] == latest_date].sort_values(
        'pred_score', ascending=False)

    top_codes = day_pred['code'].head(n_hold).tolist()

    if current_portfolio is None:
        return {'date': latest_date, 'buy': top_codes, 'sell': [],
                'portfolio': top_codes}

    ranked = day_pred.set_index('code')['pred_score']
    port_scores = [(c, ranked.get(c, -999)) for c in current_portfolio]
    port_scores.sort(key=lambda x: x[1])
    sell_codes = [c for c, _ in port_scores[:k_swap]]
    buy_codes = [c for c in top_codes if c not in current_portfolio][:k_swap]
    new_portfolio = [c for c in current_portfolio
                     if c not in sell_codes] + buy_codes

    return {'date': latest_date, 'buy': buy_codes, 'sell': sell_codes,
            'portfolio': new_portfolio}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(config.CACHE_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("No trained model found. Run train.py first.")
        return

    print("Loading latest data...")
    df = build_merged_dataset(start_date="20260301", end_date="20261231")
    df, avail_features = build_features(df)

    ds = StockDataset(df, avail_features, STATIC_FEATURES)
    print(f"Samples: {len(ds)}")

    model = CompetitionTFT(
        dynamic_input_dim=len(avail_features),
        static_input_dim=len(STATIC_FEATURES),
        hidden_dim=config.HIDDEN_DIM,
        seq_len=config.SEQ_LEN,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    pred_df = predict_latest(model, ds, device)
    signals = get_signals(pred_df)

    print(f"\n=== Prediction for {signals['date']} ===")
    print(f"Buy ({len(signals['buy'])}): {signals['buy']}")
    print(f"Sell ({len(signals['sell'])}): {signals['sell']}")
    print(f"Portfolio ({len(signals['portfolio'])}): {signals['portfolio']}")

    out_path = os.path.join(config.CACHE_DIR, "latest_signals.csv")
    pd.DataFrame({'code': signals['portfolio']}).to_csv(out_path, index=False)
    print(f"\nSignals saved to {out_path}")


if __name__ == "__main__":
    main()
