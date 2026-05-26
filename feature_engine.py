import numpy as np
import pandas as pd


def compute_ma_deviation(series, window):
    ma = series.rolling(window, min_periods=window).mean()
    return (series - ma) / (ma + 1e-8)


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def compute_bollinger_width(series, window=20):
    ma = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return (2 * std) / (ma + 1e-8)


def add_technical_features(group):
    close = group['close']
    high = group['high']
    low = group['low']

    group['ma5_dev'] = compute_ma_deviation(close, 5)
    group['ma10_dev'] = compute_ma_deviation(close, 10)
    group['ma20_dev'] = compute_ma_deviation(close, 20)
    group['rsi_14'] = compute_rsi(close, 14)
    group['macd_hist'] = compute_macd(close)
    group['atr_14'] = compute_atr(high, low, close, 14)
    group['boll_width'] = compute_bollinger_width(close, 20)
    group['vol_ma5_ratio'] = group['vol'] / (
        group['vol'].rolling(5, min_periods=5).mean() + 1e-8)
    return group


def add_moneyflow_features(group):
    buy_main = group.get('buy_elg_amount', 0) + group.get('buy_lg_amount', 0)
    sell_main = group.get('sell_elg_amount', 0) + group.get('sell_lg_amount', 0)
    total_flow = buy_main + sell_main + 1e-8
    group['main_net_ratio'] = (buy_main - sell_main) / total_flow
    group['net_mf_ma5'] = group['net_mf_amount'].rolling(5, min_periods=1).mean()
    return group


def add_fundamental_features(group):
    if 'total_mv' in group.columns:
        group['log_mv'] = np.log1p(group['total_mv'])
    if 'pe_ttm' in group.columns:
        group['pe_ttm'] = group['pe_ttm'].clip(-500, 500)
    if 'pb' in group.columns:
        group['pb'] = group['pb'].clip(-50, 50)
    return group


DYNAMIC_FEATURES = [
    'open', 'high', 'low', 'close', 'vol', 'amount', 'pct_chg', 'vwap',
    'ma5_dev', 'ma10_dev', 'ma20_dev', 'rsi_14', 'macd_hist',
    'atr_14', 'boll_width', 'vol_ma5_ratio',
    'turnover_rate', 'pe_ttm', 'pb', 'log_mv',
    'net_mf_amount', 'main_net_ratio', 'net_mf_ma5',
]

STATIC_FEATURES = ['industry_code']


def build_features(df):
    print("Computing features per stock...")
    groups = []
    for code, group in df.groupby('ts_code'):
        group = group.sort_values('trade_date').copy()
        group = add_technical_features(group)
        group = add_moneyflow_features(group)
        group = add_fundamental_features(group)
        groups.append(group)
    result = pd.concat(groups, ignore_index=True)
    avail = [f for f in DYNAMIC_FEATURES if f in result.columns]
    print(f"Dynamic features available: {len(avail)}/{len(DYNAMIC_FEATURES)}")
    return result, avail
