import hashlib
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import config


def _cache_key(df, dynamic_features, static_features, seq_len, pred_horizon):
    """Generate a hash based on inputs to detect stale cache."""
    h = hashlib.md5()
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
        close_idx = dynamic_features.index('close')
        groups = list(df.groupby('ts_code'))

        for code, group in tqdm(groups, desc="Building dataset"):
            group = group.sort_values('trade_date').reset_index(drop=True)
            n = len(group)
            if n < seq_len + pred_horizon:
                continue

            static_data = group[static_features].iloc[0].values.astype(np.float32)
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

                y_return = np.float32(
                    (future_closes[j] - current_closes[j]) / current_closes[j])

                self.samples.append({
                    'x_dynamic': x_dynamic,
                    'x_static': static_data,
                    'y': y_return,
                    'date': dates[i],
                    'code': code,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.from_numpy(s['x_dynamic']),
                torch.from_numpy(s['x_static']),
                torch.tensor(s['y']))
