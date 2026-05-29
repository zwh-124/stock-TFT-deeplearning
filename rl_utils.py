"""Shared utilities for RL pipeline (train_rl, backtest_rl, predict_rl)."""
import numpy as np
import torch

import config
from feature_engine import STATIC_FEATURES, STATIC_CATEGORICAL, STATIC_CONTINUOUS
from dataset import rolling_normalize_window


class ObsCache:
    """预计算所有股票的特征数据和日期索引，批量标准化避免逐股票循环。"""

    def __init__(self, grouped, avail_features, env, seq_len):
        self.seq_len = seq_len
        self.avail_features = avail_features
        self.n_feat = len(avail_features)
        self.n_stocks = len(env.codes)
        self.n_dates = len(env.dates)

        cat_features = list(STATIC_CATEGORICAL.keys())

        self.dynamic_matrix = np.full(
            (self.n_stocks, self.n_dates, self.n_feat), np.nan,
            dtype=np.float32)
        self.static_cat_arr = np.zeros(
            (self.n_stocks, len(cat_features)), dtype=np.float32)
        self.stock_age_matrix = np.zeros(
            (self.n_stocks, self.n_dates), dtype=np.float32)
        self.valid_start = np.full(self.n_stocks, self.n_dates, dtype=np.int32)

        date_to_idx = env.date_to_idx
        for i, code in enumerate(env.codes):
            if code not in grouped.groups:
                continue
            stock_df = grouped.get_group(code)
            self.static_cat_arr[i] = stock_df[cat_features].iloc[0].values
            dates = stock_df['trade_date'].values
            data = stock_df[avail_features].values.astype(np.float32)
            ages = stock_df['stock_age'].values.astype(np.float32) \
                if 'stock_age' in stock_df.columns else np.zeros(len(stock_df))
            for row_idx, d in enumerate(dates):
                if d in date_to_idx:
                    di = date_to_idx[d]
                    self.dynamic_matrix[i, di] = data[row_idx]
                    self.stock_age_matrix[i, di] = ages[row_idx]
            first_valid = np.where(
                ~np.isnan(self.dynamic_matrix[i, :, 0]))[0]
            if len(first_valid) >= seq_len:
                self.valid_start[i] = first_valid[seq_len - 1]

    def get_obs(self, date_idx, env, device):
        seq_len = self.seq_len
        n = self.n_stocks

        valid = (date_idx >= self.valid_start) & (date_idx < self.n_dates)
        windows = self.dynamic_matrix[:, date_idx - seq_len + 1:date_idx + 1, :]

        mean = np.nanmean(windows, axis=1, keepdims=True)
        std = np.nanstd(windows, axis=1, keepdims=True, ddof=0) + 1e-8
        normalized = (windows - mean) / std
        normalized = np.nan_to_num(normalized, 0.0)

        normalized[~valid] = 0.0
        mask_arr = valid & ~env.suspended[date_idx]

        stock_ages = self.stock_age_matrix[:, date_idx:date_idx + 1]
        stat_full = np.concatenate(
            [self.static_cat_arr, stock_ages], axis=-1)

        dyn_t = torch.tensor(normalized, device=device)
        stat_t = torch.tensor(stat_full, device=device)
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
    """Build portfolio state tensor [N_stocks, 6]."""
    prices = env.get_valuation_prices()
    nav = env._compute_nav(prices)
    n = env.n_stocks

    cash_frac = np.full(n, env.cash / (nav + 1e-8), dtype=np.float32)
    hold_val = env.holdings.astype(np.float64) * np.nan_to_num(prices, 0)
    hold_frac = (hold_val / (nav + 1e-8)).astype(np.float32)
    lock_val = env.locked.astype(np.float64) * np.nan_to_num(prices, 0)
    lock_frac = (lock_val / (nav + 1e-8)).astype(np.float32)
    prev_w = env.prev_weights.astype(np.float32)

    ep_len = getattr(env, 'episode_len', config.EPISODE_LEN)
    ep_day = getattr(env, 'episode_day', 0)
    ep_progress = np.full(n, ep_day / max(ep_len, 1), dtype=np.float32)
    is_last = np.full(n, float(ep_day >= ep_len - 1), dtype=np.float32)

    state = np.stack([cash_frac, hold_frac, lock_frac, prev_w,
                      ep_progress, is_last], axis=-1)
    return torch.tensor(state, device=device)
