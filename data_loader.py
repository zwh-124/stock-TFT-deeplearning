import os
import hashlib
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
    for f in reversed(files):
        df = pd.read_csv(os.path.join(iw_dir, f))
        if len(df) > 0:
            return set(df['con_code'].tolist())
    return None


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
    if not dfs:
        raise FileNotFoundError(
            f"No daily data found in {config.DAILY_DIR} "
            f"for range [{start_date}, {end_date}]")
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
    if not dfs:
        raise FileNotFoundError(
            f"No metric data found in {config.METRIC_DIR} "
            f"for range [{start_date}, {end_date}]")
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
    if not dfs:
        raise FileNotFoundError(
            f"No moneyflow data found in {config.MONEYFLOW_DIR} "
            f"for range [{start_date}, {end_date}]")
    return pd.concat(dfs, ignore_index=True)


def filter_stocks(df, bj_codes, st_codes_by_date):
    df = df[~df['ts_code'].isin(bj_codes)].copy()
    all_st = set()
    for st_set in st_codes_by_date.values():
        all_st.update(st_set)
    df = df[~df['ts_code'].isin(all_st)]
    return df.reset_index(drop=True)


def load_market_features():
    """Load market index features for 3 indices.

    Per index: pct_chg, realized_vol (5d), vol_ratio (vs ma5)
    Cross-index: mkt_mean_ret_5d, large_small_spread, mkt_overnight_gap
    """
    indices = {
        'idx001': '000001.SH.csv',
        'idx300': '000300.SH.csv',
        'idx399': '399006.SZ.csv',
    }
    per_index_dfs = {}
    for prefix, filename in indices.items():
        path = os.path.join(config.MARKET_DIR, filename)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['trade_date'] = df['trade_date'].astype(str)
        df = df.sort_values('trade_date').reset_index(drop=True)

        df[f'{prefix}_pct_chg'] = df['pct_chg']
        ret = df['pct_chg'] / 100.0
        df[f'{prefix}_realized_vol'] = ret.rolling(5, min_periods=2).std()
        vol_ma5 = df['vol'].rolling(5, min_periods=1).mean()
        df[f'{prefix}_vol_ratio'] = df['vol'] / (vol_ma5 + 1e-8)

        df[f'__{prefix}_ret'] = ret
        df[f'__{prefix}_gap'] = (df['open'] - df['pre_close']) / (
            df['pre_close'] + 1e-8)
        df[f'__{prefix}_ret_5d'] = ret.rolling(5, min_periods=1).sum()

        per_index_dfs[prefix] = df

    if not per_index_dfs:
        return None

    base_prefix = list(per_index_dfs.keys())[0]
    result = per_index_dfs[base_prefix][['trade_date']].copy()

    for prefix, df in per_index_dfs.items():
        cols = [f'{prefix}_pct_chg', f'{prefix}_realized_vol',
                f'{prefix}_vol_ratio']
        result = result.merge(df[['trade_date'] + cols], on='trade_date',
                              how='outer')

    all_rets = []
    all_gaps = []
    all_ret_5d = []
    for prefix, df in per_index_dfs.items():
        tmp = df[['trade_date', f'__{prefix}_ret', f'__{prefix}_gap',
                  f'__{prefix}_ret_5d']]
        result = result.merge(tmp, on='trade_date', how='left')
        all_rets.append(f'__{prefix}_ret')
        all_gaps.append(f'__{prefix}_gap')
        all_ret_5d.append(f'__{prefix}_ret_5d')

    result['mkt_mean_ret_5d'] = result[all_ret_5d].mean(axis=1)

    if '__idx300_ret' in result.columns and '__idx399_ret' in result.columns:
        result['large_small_spread'] = (
            result['__idx300_ret'] - result['__idx399_ret'])
    else:
        result['large_small_spread'] = 0.0

    result['mkt_overnight_gap'] = result[all_gaps].mean(axis=1)

    drop_cols = [c for c in result.columns if c.startswith('__')]
    result = result.drop(columns=drop_cols)
    return result


def _data_loader_version():
    """Hash key source files to detect stale cache."""
    h = hashlib.md5()
    for fname in ['data_loader.py', 'feature_engine.py']:
        path = os.path.join(os.path.dirname(__file__), fname)
        if os.path.exists(path):
            h.update(open(path, 'rb').read())
    return h.hexdigest()[:8]


def build_merged_dataset(start_date=None, end_date=None):
    date_tag = f"_{start_date}_{end_date}" if start_date and end_date else ""
    base_name = "merged_csi300" if config.USE_CSI300 else "merged_all"
    version = _data_loader_version()
    cache_name = f"{base_name}{date_tag}_{version}.pkl"
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

    market_features = load_market_features()
    if market_features is not None:
        merged = merged.merge(market_features, on='trade_date', how='left')

    merged = filter_stocks(merged, bj_codes, st_codes_by_date)

    # 过滤新股上市初期数据（前 IPO_SKIP_DAYS 个交易日）
    list_date_map = dict(zip(basic_df['ts_code'],
                             basic_df['list_date'].astype(str)))
    merged = merged.sort_values(['ts_code', 'trade_date'])
    merged['_days_since_ipo'] = merged.groupby('ts_code').cumcount()
    before_len = len(merged)
    merged = merged[merged['_days_since_ipo'] >= config.IPO_SKIP_DAYS]
    merged = merged.drop(columns=['_days_since_ipo'])
    print(f"IPO filter: removed {before_len - len(merged)} rows "
          f"(first {config.IPO_SKIP_DAYS} trading days per stock)")

    categories = sorted(basic_df['industry'].dropna().unique())
    cat_map = {c: i + 1 for i, c in enumerate(categories)}
    industry_map = {code: cat_map.get(ind, 0)
                    for code, ind in zip(basic_df['ts_code'], basic_df['industry'])}
    merged['industry_code'] = merged['ts_code'].map(industry_map).fillna(0).astype(int)

    area_cats = sorted(basic_df['area'].dropna().unique())
    area_map = {a: i + 1 for i, a in enumerate(area_cats)}
    area_code_map = {code: area_map.get(area, 0)
                     for code, area in zip(basic_df['ts_code'], basic_df['area'])}
    merged['area_code'] = merged['ts_code'].map(area_code_map).fillna(0).astype(int)

    market_cats = ['主板', '创业板', '科创板', '北交所']
    market_map = {m: i + 1 for i, m in enumerate(market_cats)}
    market_code_map = {code: market_map.get(mkt, 0)
                       for code, mkt in zip(basic_df['ts_code'], basic_df['market'])}
    merged['market_code'] = merged['ts_code'].map(market_code_map).fillna(0).astype(int)

    ent_cats = sorted(basic_df['act_ent_type'].dropna().unique())
    ent_map = {e: i + 1 for i, e in enumerate(ent_cats)}
    ent_code_map = {code: ent_map.get(ent, 0)
                    for code, ent in zip(basic_df['ts_code'], basic_df['act_ent_type'])}
    merged['ent_type_code'] = merged['ts_code'].map(ent_code_map).fillna(0).astype(int)

    list_date_map = dict(zip(basic_df['ts_code'], basic_df['list_date'].astype(str)))
    merged['_list_date'] = merged['ts_code'].map(list_date_map)
    merged['stock_age'] = (
        pd.to_datetime(merged['trade_date'], format='%Y%m%d') -
        pd.to_datetime(merged['_list_date'], format='%Y%m%d')
    ).dt.days / 365.25
    merged['stock_age'] = merged['stock_age'].clip(0, 40).astype(np.float32)
    merged = merged.drop(columns=['_list_date'])

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
