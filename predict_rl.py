import os
import argparse
import numpy as np
import pandas as pd
import torch

import config
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from model import TFTEncoder, PortfolioPolicy
from env import AShareTradingEnv
from train_rl import get_obs_for_date, build_port_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True,
                        help="Target date YYYYMMDD")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = build_merged_dataset(config.TRAIN_START, config.TEST_END)
    df, avail_features = build_features(df)
    env = AShareTradingEnv(df)

    dynamic_dim = len(avail_features)
    static_dim = len(STATIC_FEATURES)

    encoder = TFTEncoder(dynamic_dim, static_dim,
                         hidden_dim=config.HIDDEN_DIM,
                         seq_len=config.SEQ_LEN,
                         num_heads=config.NUM_HEADS,
                         dropout=config.DROPOUT).to(device)
    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=4, dropout=config.DROPOUT).to(device)

    ckpt_path = os.path.join(config.CACHE_DIR, "best_rl_policy.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    policy.load_state_dict(ckpt["policy"])
    encoder.eval()
    policy.eval()

    target_date = args.date
    if target_date not in env.date_to_idx:
        print(f"Date {target_date} not in data range.")
        return
    date_idx = env.date_to_idx[target_date]

    orders = []
    bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)

    for phase in ["open", "close"]:
        env.reset(start_date_idx=date_idx)
        env.phase = phase

        dyn_t, stat_t, mask_t = get_obs_for_date(
            df, avail_features, date_idx, env, config.SEQ_LEN, device)
        port_state = build_port_state(env, device)

        with torch.no_grad():
            enc = encoder(dyn_t, stat_t)
            dist = policy(enc, port_state, mask_t, phase)
            action = dist.probs.argmax(dim=-1)

        weights = bins[action].cpu().numpy()
        top_k_idx = np.argsort(weights)[-config.N_HOLD:]

        prices = env.open_prices[date_idx] if phase == "open" \
            else env.close_prices[date_idx]
        limit_up, limit_down = env._limit_prices(date_idx)
        nav = env._compute_nav()

        for idx in top_k_idx:
            w = weights[idx]
            if w <= 0 or np.isnan(prices[idx]):
                continue
            shares = int(w * nav / prices[idx] / config.LOT) * config.LOT
            if shares <= 0:
                continue
            code = env.codes[idx]
            orders.append({
                "date": target_date,
                "phase": phase,
                "code": code,
                "side": "buy",
                "shares": shares,
                "target_price": prices[idx],
                "limit_up": limit_up[idx],
                "limit_down": limit_down[idx],
            })

    result_df = pd.DataFrame(orders)
    out_path = os.path.join(config.CACHE_DIR, "latest_rl_signals.csv")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(result_df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()