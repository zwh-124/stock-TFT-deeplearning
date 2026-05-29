import os
import argparse
import numpy as np
import pandas as pd
import torch

import config
from data_loader import build_merged_dataset
from feature_engine import build_features, DYNAMIC_FEATURES, STATIC_FEATURES
from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
from model import TFTEncoder, PortfolioPolicy, DiffusionDenoiser
from env import AShareTradingEnv
from rl_utils import get_obs_for_date, build_port_state, ObsCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True,
                        help="Target date YYYYMMDD")
    parser.add_argument("--episode-day", type=int, default=0,
                        help="Current day within the 10-day episode (0-9)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(config.CACHE_DIR, "best_rl_policy.pt")
    if not os.path.exists(ckpt_path):
        print(f"No RL model found at {ckpt_path}. Run train_rl.py first.")
        return

    end_date = max(config.TEST_END, args.date)
    df = build_merged_dataset(config.TRAIN_START, end_date)
    df, avail_features = build_features(df)
    env = AShareTradingEnv(df)

    dynamic_dim = len(avail_features)
    static_dim = len(STATIC_FEATURES)

    encoder = TFTEncoder(dynamic_dim, static_dim,
                         hidden_dim=config.HIDDEN_DIM,
                         seq_len=config.SEQ_LEN,
                         num_heads=config.NUM_HEADS,
                         dropout=config.DROPOUT,
                         static_categorical=STATIC_CATEGORICAL,
                         static_n_continuous=len(STATIC_CONTINUOUS)).to(device)
    policy = PortfolioPolicy(config.HIDDEN_DIM, n_bins=config.N_BINS,
                             n_extra_state=6, dropout=config.DROPOUT).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if 'config' in ckpt:
        saved_cfg = ckpt['config']
        for key, cur in {'HIDDEN_DIM': config.HIDDEN_DIM,
                         'SEQ_LEN': config.SEQ_LEN,
                         'NUM_HEADS': config.NUM_HEADS,
                         'N_BINS': config.N_BINS,
                         'N_HOLD': config.N_HOLD}.items():
            saved = saved_cfg.get(key)
            if saved is not None and saved != cur:
                print(f"ERROR: config mismatch — {key}: "
                      f"trained={saved}, current={cur}")
                return

        saved_bins = saved_cfg.get('BINS')
        if saved_bins is not None and saved_bins != config.BINS:
            print(f"ERROR: BINS mismatch — "
                  f"trained={saved_bins}, current={config.BINS}")
            return

        saved_features = saved_cfg.get('DYNAMIC_FEATURES')
        if saved_features is not None and saved_features != avail_features:
            missing = set(saved_features) - set(avail_features)
            extra = set(avail_features) - set(saved_features)
            print(f"ERROR: feature mismatch.\n"
                  f"  Missing from current: {missing}\n"
                  f"  Extra in current: {extra}")
            return

    if any(k.startswith('denoiser.') for k in ckpt["encoder"].keys()):
        denoiser = DiffusionDenoiser(
            feature_dim=dynamic_dim,
            seq_len=config.SEQ_LEN,
            hidden_dim=config.DIFFUSION_HIDDEN_DIM,
            time_dim=config.DIFFUSION_TIME_DIM,
            n_timesteps=config.DIFFUSION_T,
            beta_start=config.DIFFUSION_BETA_START,
            beta_end=config.DIFFUSION_BETA_END,
        ).to(device)
        encoder.denoiser = denoiser

    encoder.load_state_dict(ckpt["encoder"])
    policy.load_state_dict(ckpt["policy"])
    encoder.eval()
    policy.eval()

    target_date = args.date
    if target_date not in env.date_to_idx:
        print(f"Date {target_date} not in data range "
              f"[{env.dates[0]}, {env.dates[-1]}].")
        return
    date_idx = env.date_to_idx[target_date]

    if date_idx < config.SEQ_LEN:
        print(f"ERROR: date_idx={date_idx} < SEQ_LEN={config.SEQ_LEN}. "
              f"Not enough history for {target_date}. "
              f"Earliest valid date: {env.dates[config.SEQ_LEN]}")
        return

    grouped = df.sort_values('trade_date').groupby('ts_code')
    obs_cache = ObsCache(grouped, avail_features, env, config.SEQ_LEN)

    orders = []
    bins = torch.tensor(config.BINS, device=device, dtype=torch.float32)

    env.reset(start_date_idx=date_idx, episode_len=config.EPISODE_LEN)
    env.episode_day = args.episode_day

    is_last_day = args.episode_day >= config.EPISODE_LEN - 1

    for phase in ["open", "close"]:
        env.phase = phase

        if is_last_day and phase == "close":
            for i in range(env.n_stocks):
                if env.holdings[i] > 0:
                    prices_now = env.close_prices[date_idx]
                    if not np.isnan(prices_now[i]) and prices_now[i] > 0:
                        orders.append({
                            "date": target_date, "phase": phase,
                            "code": env.codes[i], "side": "sell",
                            "shares": int(env.holdings[i]),
                            "target_price": prices_now[i],
                            "limit_up": env._limit_prices(date_idx)[0][i],
                            "limit_down": env._limit_prices(date_idx)[1][i],
                        })
            break

        dyn_t, stat_t, mask_t = obs_cache.get_obs(date_idx, env, device)
        port_state = build_port_state(env, device)

        with torch.no_grad():
            enc = encoder(dyn_t, stat_t)
            dist = policy(enc, port_state, mask_t, phase)
            action = dist.probs.argmax(dim=-1)

        weights = bins[action].cpu().numpy()
        top_k_idx = np.argsort(weights)[-config.N_HOLD:]

        target_w = np.zeros(env.n_stocks)
        for idx in top_k_idx:
            target_w[idx] = weights[idx]
        w_sum = target_w.sum()
        if w_sum > 0:
            target_w = target_w / w_sum

        prices = env.open_prices[date_idx] if phase == "open" \
            else env.close_prices[date_idx]
        limit_up, limit_down = env._limit_prices(date_idx)
        val_prices = env.get_valuation_prices()
        nav = env._compute_nav(val_prices)

        current_holdings = env.holdings + env.locked
        for i in range(env.n_stocks):
            if np.isnan(prices[i]) or prices[i] <= 0:
                continue
            target_shares = int(target_w[i] * nav / prices[i]
                                / config.LOT) * config.LOT
            held = current_holdings[i]
            if target_shares > held:
                buy_shares = target_shares - held
                if buy_shares > 0:
                    orders.append({
                        "date": target_date, "phase": phase,
                        "code": env.codes[i], "side": "buy",
                        "shares": int(buy_shares),
                        "target_price": prices[i],
                        "limit_up": limit_up[i],
                        "limit_down": limit_down[i],
                    })
            elif target_shares < held and env.holdings[i] > 0:
                sell_shares = min(held - target_shares, env.holdings[i])
                if sell_shares > 0:
                    orders.append({
                        "date": target_date, "phase": phase,
                        "code": env.codes[i], "side": "sell",
                        "shares": int(sell_shares),
                        "target_price": prices[i],
                        "limit_up": limit_up[i],
                        "limit_down": limit_down[i],
                    })

        env.step(target_w)

        nav_after = env._compute_nav()
        if env.cash > config.MAX_CASH:
            remaining = env.cash - config.MAX_CASH
            ranked = sorted(
                [(target_w[i], i) for i in top_k_idx
                 if not np.isnan(prices[i]) and prices[i] > 0],
                key=lambda x: -x[0])
            for _, i in ranked:
                extra_shares = int(remaining / (prices[i] * (1 + config.COMMISSION))
                                   / config.LOT) * config.LOT
                if extra_shares > 0:
                    orders.append({
                        "date": target_date, "phase": phase,
                        "code": env.codes[i], "side": "buy",
                        "shares": int(extra_shares),
                        "target_price": prices[i],
                        "limit_up": limit_up[i],
                        "limit_down": limit_down[i],
                    })
                    cost = extra_shares * prices[i] * (1 + config.COMMISSION)
                    remaining -= cost
                    if remaining < prices[i] * config.LOT:
                        break

    result_df = pd.DataFrame(orders)
    out_path = os.path.join(config.CACHE_DIR, "latest_rl_signals.csv")
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(result_df.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()