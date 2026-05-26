import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import config


def load_basic_info():
    df = pd.read_csv(config.BASIC_CSV)
    bj_codes = set(df[df['market'] == '北交所']['ts_code'].tolist())
    return df, bj_codes


def load_csi300_codes():
    iw_dir = config.INDEX_WEIGHT_DIR
    files = sorted([f for f in os.listdir(iw_dir) if '000300' in f])
    if not files:
        return None
    latest = files[-1]
    df = pd.read_csv(os.path.join(iw_dir, latest))
    return set(df['con_code'].tolist())


def load_st_codes():
    st_codes_by_date = {}
    st_dir = config.ST_DIR
    for f in sorted(os.listdir(st_dir)):
        if not f.endswith('.csv'):
            continue
        date = f.replace('.csv', '')
        df = pd.read_csv(os.path.join(st_dir, f))
        st_codes_by_date[date] = set(df['ts_code'].tolist())
    return st_codes_by_date


def load_daily_data(start_date=None, end_date=None, stock_pool=None):
    files = sorted(os.listdir(config.DAILY_DIR))
    dfs = []
    for f in tqdm(files, desc="Loading daily"):
        if not f.endswith('.csv'):
            continue
        date = f.replace('.csv', '')
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        df = pd.read_csv(os.path.join(config.DAILY_DIR, f))
        if stock_pool:
            df = df[df['ts_code'].isin(stock_pool)]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_metric_data(start_date=None, end_date=None, stock_pool=None):
    files = sorted(os.listdir(config.METRIC_DIR))
    dfs = []
    for f in tqdm(files, desc="Loading metric"):
        if not f.endswith('.csv'):
            continue
        date = f.replace('.csv', '')
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        df = pd.read_csv(os.path.join(config.METRIC_DIR, f))
        if stock_pool:
            df = df[df['ts_code'].isin(stock_pool)]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_moneyflow_data(start_date=None, end_date=None, stock_pool=None):
    files = sorted(os.listdir(config.MONEYFLOW_DIR))
    dfs = []
    for f in tqdm(files, desc="Loading moneyflow"):
        if not f.endswith('.csv'):
            continue
        date = f.replace('.csv', '')
        if start_date and date < start_date:
            continue
        if end_date and date > end_date:
            continue
        df = pd.read_csv(os.path.join(config.MONEYFLOW_DIR, f))
        if stock_pool:
            df = df[df['ts_code'].isin(stock_pool)]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def filter_stocks(df, bj_codes, st_codes_by_date):
    df = df[~df['ts_code'].isin(bj_codes)].copy()
    all_st = set()
    for st_set in st_codes_by_date.values():
        all_st.update(st_set)
    df = df[~df['ts_code'].isin(all_st)]
    return df.reset_index(drop=True)


def build_merged_dataset(start_date=None, end_date=None):
    date_tag = f"_{start_date}_{end_date}" if start_date and end_date else ""
    base_name = "merged_csi300" if config.USE_CSI300 else "merged_all"
    cache_name = f"{base_name}{date_tag}.pkl"
    cache_path = os.path.join(config.CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_pickle(cache_path)

    print("Building merged dataset from scratch...")
    basic_df, bj_codes = load_basic_info()
    st_codes_by_date = load_st_codes()

    stock_pool = None
    if config.USE_CSI300:
        stock_pool = load_csi300_codes()
        if stock_pool:
            stock_pool = stock_pool - bj_codes
            print(f"Using CSI300 stock pool: {len(stock_pool)} stocks")

    daily = load_daily_data(start_date, end_date, stock_pool)
    metric = load_metric_data(start_date, end_date, stock_pool)
    moneyflow = load_moneyflow_data(start_date, end_date, stock_pool)

    metric_cols = ['ts_code', 'trade_date', 'turnover_rate',
                   'turnover_rate_f', 'volume_ratio', 'pe_ttm',
                   'pb', 'ps_ttm', 'total_mv', 'circ_mv']
    metric = metric[[c for c in metric_cols if c in metric.columns]]

    mf_cols = ['ts_code', 'trade_date', 'buy_elg_amount',
               'sell_elg_amount', 'buy_lg_amount', 'sell_lg_amount',
               'net_mf_amount']
    moneyflow = moneyflow[[c for c in mf_cols if c in moneyflow.columns]]

    daily['trade_date'] = daily['trade_date'].astype(str)
    metric['trade_date'] = metric['trade_date'].astype(str)
    moneyflow['trade_date'] = moneyflow['trade_date'].astype(str)

    merged = daily.merge(metric, on=['ts_code', 'trade_date'], how='left')
    merged = merged.merge(moneyflow, on=['ts_code', 'trade_date'], how='left')

    merged = filter_stocks(merged, bj_codes, st_codes_by_date)

    industry_map = dict(zip(basic_df['ts_code'],
                            basic_df['industry'].astype('category').cat.codes))
    merged['industry_code'] = merged['ts_code'].map(industry_map).fillna(-1)

    merged = merged.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    merged.to_pickle(cache_path)
    print(f"Saved merged data to {cache_path}, shape: {merged.shape}")
    return merged


if __name__ == "__main__":
    df = build_merged_dataset(start_date="20200101", end_date="20251231")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"Stocks: {df['ts_code'].nunique()}")
    print(f"Missing ratio:\n{df.isnull().mean().sort_values(ascending=False).head(10)}")
