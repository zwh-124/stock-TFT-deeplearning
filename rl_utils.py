"""Shared utilities for RL pipeline (train_rl, backtest_rl, predict_rl)."""
import numpy as np
import torch

import config
from feature_engine import STATIC_FEATURES
from dataset import rolling_normalize_window


class ObsCache:
    """预计算所有股票的特征数据，避免每次调用时重复做pandas过滤。"""

    def __init__(self, grouped, avail_features, env, seq_len):
        self.seq_len = seq_len
        self.avail_features = avail_features
        self.n_feat = len(avail_features)
        self.stock_data = {}

        for code in env.codes:
            if code not in grouped.groups:
                continue
            stock_df = grouped.get_group(code)
            dynamic_data = stock_df[avail_features].values.astype(np.float32)
            dates = stock_df['trade_date'].values
            static_data = stock_df[STATIC_FEATURES].iloc[0].values.astype(
                np.float32)
            self.stock_data[code] = (dynamic_data, dates, static_data)

    def get_obs(self, date_idx, env, device):
        date = env.dates[date_idx]
        n = len(env.codes)
        seq_len = self.seq_len

        dyn_arr = np.zeros((n, seq_len, self.n_feat), dtype=np.float32)
        stat_arr = np.zeros((n, len(STATIC_FEATURES)), dtype=np.float32)
        mask_arr = np.zeros(n, dtype=bool)

        for i, code in enumerate(env.codes):
            if code not in self.stock_data:
                continue
            dynamic_data, dates, static_data = self.stock_data[code]
            end_idx = np.searchsorted(dates, date, side='right')
            if end_idx < seq_len:
                continue
            dyn_arr[i] = rolling_normalize_window(
                dynamic_data, seq_len, end_idx)
            stat_arr[i] = static_data
            mask_arr[i] = not env.suspended[date_idx, i]

        dyn_t = torch.tensor(dyn_arr, device=device)
        stat_t = torch.tensor(stat_arr, device=device)
        mask_t = torch.tensor(mask_arr, device=device, dtype=torch.bool)
        return dyn_t, stat_t, mask_t


def get_obs_for_date(grouped, avail_features, date_idx, env, seq_len, device):
    """Build encoder input tensors for all stocks at a given date."""
    date = env.dates[date_idx]
    codes = env.codes
    n = len(codes)
    n_feat = len(avail_features)

    dyn_arr = np.zeros((n, seq_len, n_feat), dtype=np.float32)
    stat_arr = np.zeros((n, len(STATIC_FEATURES)), dtype=np.float32)
    mask_arr = np.zeros(n, dtype=bool)

    for i, code in enumerate(codes):
        if code not in grouped.groups:
            continue
        stock_df = grouped.get_group(code)
        date_pos = stock_df[stock_df['trade_date'] <= date]
        if len(date_pos) < seq_len:
            continue
        dynamic_data = date_pos[avail_features].values.astype(np.float32)
        dyn_arr[i] = rolling_normalize_window(
            dynamic_data, seq_len, len(dynamic_data))
        stat_arr[i] = date_pos[STATIC_FEATURES].iloc[0].values.astype(
            np.float32)
        mask_arr[i] = not env.suspended[date_idx, i]

    dyn_t = torch.tensor(dyn_arr, device=device)
    stat_t = torch.tensor(stat_arr, device=device)
    mask_t = torch.tensor(mask_arr, device=device, dtype=torch.bool)
    return dyn_t, stat_t, mask_t


def build_port_state(env, device):
    """Build portfolio state tensor [N_stocks, 4]."""
    prices = env.get_valuation_prices()
    nav = env._compute_nav(prices)
    n = env.n_stocks

    cash_frac = np.full(n, env.cash / (nav + 1e-8), dtype=np.float32)
    hold_val = env.holdings.astype(np.float64) * np.nan_to_num(prices, 0)
    hold_frac = (hold_val / (nav + 1e-8)).astype(np.float32)
    lock_val = env.locked.astype(np.float64) * np.nan_to_num(prices, 0)
    lock_frac = (lock_val / (nav + 1e-8)).astype(np.float32)
    prev_w = env.prev_weights.astype(np.float32)

    state = np.stack([cash_frac, hold_frac, lock_frac, prev_w], axis=-1)
    return torch.tensor(state, device=device)
