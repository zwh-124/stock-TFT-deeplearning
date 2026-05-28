import hashlib
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import config


def rolling_normalize_window(dynamic_data, seq_len, end_idx):
    """Extract a normalized window [end_idx-seq_len : end_idx] from dynamic_data.

    Uses the rolling mean/std computed over the window itself (same logic as
    _build_samples). This is the public helper for env.py to reuse.

    Args:
        dynamic_data: np.ndarray of shape (T, num_features), full history for one stock
        seq_len: window length
        end_idx: the index of the last row (exclusive) in the window

    Returns:
        np.ndarray of shape (seq_len, num_features), normalized
    """
    window = dynamic_data[end_idx - seq_len:end_idx]
    mean = window.mean(axis=0)
    std = window.std(axis=0, ddof=0) + 1e-8
    normalized = (window - mean) / std
    return np.nan_to_num(normalized, 0.0).astype(np.float32)


_DATASET_VERSION = "v3_delta_target"


def _cache_key(df, dynamic_features, static_features, seq_len, pred_horizon):
    """Generate a hash based on inputs to detect stale cache."""
    h = hashlib.md5()
    h.update(_DATASET_VERSION.encode())
    h.update(str(dynamic_features).encode())
    h.update(str(static_features).encode())
    h.update(f"{seq_len}_{pred_horizon}".encode())
    h.update(f"{len(df)}_{df['trade_date'].min()}_{df['trade_date'].max()}".encode())
    h.update(str(sorted(df['ts_code'].unique()[:10])).encode())
    h.update(str(df.iloc[:5].values.tobytes()[:256]).encode())
    return h.hexdigest()[:16]


class StockDataset(Dataset):
    def __init__(self, df, dynamic_features, static_features,
                 seq_len=config.SEQ_LEN, pred_horizon=config.PRED_HORIZON,
                 cache_tag="dataset"):
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.samples = []

        # --- cache logic ---
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        key = _cache_key(df, dynamic_features, static_features,
                         seq_len, pred_horizon)
        cache_path = os.path.join(config.CACHE_DIR,
                                  f"{cache_tag}_{key}.pkl")

        if os.path.exists(cache_path):
            print(f"Loading cached samples from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            return
        # --- end cache logic ---

        self._build_samples(df, dynamic_features, static_features,
                            seq_len, pred_horizon)

        # atomic write: temp file + rename to prevent incomplete cache
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(self.samples, f)
        os.replace(tmp_path, cache_path)
        print(f"Cached {len(self.samples)} samples to {cache_path}")

    def _build_samples(self, df, dynamic_features, static_features,
                       seq_len, pred_horizon):
        from feature_engine import STATIC_CATEGORICAL, STATIC_CONTINUOUS
        cat_features = list(STATIC_CATEGORICAL.keys())
        cont_features = STATIC_CONTINUOUS

        close_idx = dynamic_features.index('close')
        groups = list(df.groupby('ts_code'))

        for code, group in tqdm(groups, desc="Building dataset"):
            group = group.sort_values('trade_date').reset_index(drop=True)
            n = len(group)
            if n < seq_len + pred_horizon:
                continue

            static_cat = group[cat_features].iloc[0].values.astype(np.float32)
            cont_data = group[cont_features].values.astype(np.float32)
            dynamic_data = group[dynamic_features].values.astype(np.float32)
            dates = group['trade_date'].values

            dyn_df = pd.DataFrame(dynamic_data, columns=dynamic_features)
            roll_mean = dyn_df.rolling(seq_len, min_periods=1).mean().values
            roll_std = dyn_df.rolling(seq_len, min_periods=1).std(ddof=0).values

            roll_mean = roll_mean[seq_len - 1:]
            roll_std = roll_std[seq_len - 1:]

            num_samples = n - seq_len - pred_horizon + 1
            if num_samples <= 0:
                continue

            close_col = dynamic_data[:, close_idx]
            current_closes = close_col[seq_len - 1:seq_len - 1 + num_samples]
            future_closes = close_col[seq_len + pred_horizon - 1:
                                      seq_len + pred_horizon - 1 + num_samples]

            valid = ((current_closes != 0) &
                     ~np.isnan(current_closes) &
                     ~np.isnan(future_closes))

            for j in range(num_samples):
                if not valid[j]:
                    continue
                i = seq_len + j
                window = dynamic_data[i - seq_len:i]
                if np.all(np.isnan(window)):
                    continue

                mean = roll_mean[j]
                std = roll_std[j] + 1e-8
                x_dynamic = (window - mean) / std
                x_dynamic = np.nan_to_num(x_dynamic, 0.0).astype(np.float32)

                last_row = dynamic_data[i - 1]
                future_row = dynamic_data[i + pred_horizon - 1]
                y_features = ((future_row - last_row) / std)
                y_features = np.nan_to_num(y_features, 0.0).astype(np.float32)

                self.samples.append({
                    'x_dynamic': x_dynamic,
                    'x_static': np.concatenate([static_cat, cont_data[i]]),
                    'y': y_features,
                    'date': dates[i],
                    'code': code,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.from_numpy(s['x_dynamic']),
                torch.from_numpy(s['x_static']),
                torch.from_numpy(s['y']))
