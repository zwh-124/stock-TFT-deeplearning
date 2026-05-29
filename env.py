import numpy as np
import pandas as pd
import config


class AShareTradingEnv:
    def __init__(self, panel_df, init_capital=config.INIT_CAPITAL,
                 lot=config.LOT, commission=config.COMMISSION,
                 stamp=config.STAMP, min_commission=config.MIN_COMMISSION,
                 limit_pct=config.LIMIT_PCT):
        self.init_capital = init_capital
        self.lot = lot
        self.commission = commission
        self.stamp = stamp
        self.min_commission = min_commission
        self.limit_pct = limit_pct

        self._prepare_panel(panel_df)

    def _prepare_panel(self, panel_df):
        self.codes = sorted(panel_df['ts_code'].unique())
        self.code_to_idx = {c: i for i, c in enumerate(self.codes)}
        self.n_stocks = len(self.codes)

        self.dates = sorted(panel_df['trade_date'].unique())
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}

        n_dates = len(self.dates)
        self.open_prices = np.full((n_dates, self.n_stocks), np.nan)
        self.close_prices = np.full((n_dates, self.n_stocks), np.nan)
        self.pre_close_prices = np.full((n_dates, self.n_stocks), np.nan)
        self.suspended = np.ones((n_dates, self.n_stocks), dtype=bool)

        pdf = panel_df.copy()
        pdf['_di'] = pdf['trade_date'].map(self.date_to_idx)
        pdf['_si'] = pdf['ts_code'].map(self.code_to_idx)
        valid = pdf.dropna(subset=['_di', '_si'])
        di = valid['_di'].astype(int).values
        si = valid['_si'].astype(int).values

        self.open_prices[di, si] = valid['open'].values
        self.close_prices[di, si] = valid['close'].values
        self.suspended[di, si] = False

        if 'pre_close' in valid.columns:
            pre_vals = valid['pre_close'].values.astype(float)
            has_pre = ~np.isnan(pre_vals)
            self.pre_close_prices[di[has_pre], si[has_pre]] = pre_vals[has_pre]

        mask = np.isnan(self.pre_close_prices)
        mask[0, :] = False
        shifted_close = np.empty_like(self.pre_close_prices)
        shifted_close[0, :] = np.nan
        shifted_close[1:, :] = self.close_prices[:-1, :]
        self.pre_close_prices = np.where(mask, shifted_close, self.pre_close_prices)

    # === PLACEHOLDER_ENV_METHODS ===

    def reset(self, start_date_idx=0, episode_len=None):
        self.cash = self.init_capital
        self.holdings = np.zeros(self.n_stocks, dtype=np.int64)
        self.locked = np.zeros(self.n_stocks, dtype=np.int64)
        self.prev_weights = np.zeros(self.n_stocks, dtype=np.float64)
        self.current_idx = start_date_idx
        self.phase = "open"
        self.episode_start_idx = start_date_idx
        self.episode_len = episode_len if episode_len is not None else config.EPISODE_LEN
        self.episode_day = 0
        return self._get_state()

    def _get_state(self):
        nav = self._compute_nav()
        return {
            "date_idx": self.current_idx,
            "phase": self.phase,
            "cash": self.cash,
            "holdings": self.holdings.copy(),
            "locked": self.locked.copy(),
            "prev_weights": self.prev_weights.copy(),
            "nav": nav,
            "episode_day": self.episode_day,
            "is_last_day": self.episode_day >= self.episode_len - 1,
        }

    def _compute_nav(self, prices=None):
        if prices is None:
            prices = self.close_prices[self.current_idx]
        stock_value = np.nansum(
            (self.holdings + self.locked).astype(np.float64) * prices)
        return self.cash + stock_value

    def get_valuation_prices(self):
        """Return the appropriate prices for current phase valuation.

        open phase: use previous day's close (last known price)
        close phase: use current day's close
        """
        if self.phase == "open":
            if self.current_idx > 0:
                return self.close_prices[self.current_idx - 1]
            return self.open_prices[self.current_idx]
        return self.close_prices[self.current_idx]

    def _limit_prices(self, date_idx):
        pre = self.pre_close_prices[date_idx]
        limit_up = np.round(pre * (1 + self.limit_pct), 2)
        limit_down = np.round(pre * (1 - self.limit_pct), 2)
        return limit_up, limit_down

    def _can_buy(self, date_idx):
        limit_up, _ = self._limit_prices(date_idx)
        prices = self.open_prices[date_idx] if self.phase == "open" \
            else self.close_prices[date_idx]
        valid_price = ~np.isnan(prices)
        hit_limit_up = np.nan_to_num(prices, nan=np.inf) >= limit_up
        mask = ~self.suspended[date_idx] & ~hit_limit_up & valid_price
        return mask

    def _can_sell(self, date_idx):
        _, limit_down = self._limit_prices(date_idx)
        prices = self.open_prices[date_idx] if self.phase == "open" \
            else self.close_prices[date_idx]
        valid_price = ~np.isnan(prices)
        hit_limit_down = np.nan_to_num(prices, nan=-np.inf) <= limit_down
        mask = ~self.suspended[date_idx] & ~hit_limit_down & valid_price
        return mask

    def step(self, target_weights, force_liquidate=False):
        """Execute one decision step.

        Args:
            target_weights: np.ndarray [n_stocks], target portfolio weight.
            force_liquidate: if True, sell all holdings ignoring target_weights.

        Returns:
            (next_state, reward, done, info)
        """
        date_idx = self.current_idx
        nav_before = self._compute_nav()

        is_last_day = self.episode_day >= self.episode_len - 1
        if force_liquidate or (is_last_day and self.phase == "close"):
            target_weights = np.zeros(self.n_stocks)

        if self.phase == "open":
            prices = self.open_prices[date_idx]
        else:
            prices = self.close_prices[date_idx]

        can_buy = self._can_buy(date_idx)
        can_sell = self._can_sell(date_idx)

        target_values = target_weights * nav_before
        target_shares = np.zeros(self.n_stocks, dtype=np.int64)
        for i in range(self.n_stocks):
            if np.isnan(prices[i]) or prices[i] <= 0:
                continue
            target_shares[i] = int(target_values[i] / prices[i] / self.lot) * self.lot

        available = self.holdings.copy()
        sell_shares = np.maximum(available - target_shares, 0)
        sell_shares = np.where(can_sell, sell_shares, 0)

        sell_revenue = 0.0
        for i in range(self.n_stocks):
            if sell_shares[i] > 0:
                revenue = sell_shares[i] * prices[i]
                comm = max(revenue * self.commission, self.min_commission)
                stamp_tax = revenue * self.stamp
                net = revenue - comm - stamp_tax
                sell_revenue += net
                self.holdings[i] -= sell_shares[i]

        self.cash += sell_revenue

        buy_order = []
        for i in range(self.n_stocks):
            want = target_shares[i] - self.holdings[i]
            if want > 0 and can_buy[i]:
                buy_order.append((target_weights[i], i, want))
        buy_order.sort(key=lambda x: -x[0])

        for _, i, shares in buy_order:
            cost = shares * prices[i]
            comm = max(cost * self.commission, self.min_commission)
            total = cost + comm
            if total > self.cash:
                shares = int(self.cash / (prices[i] * (1 + self.commission))
                             / self.lot) * self.lot
                if shares <= 0:
                    continue
                cost = shares * prices[i]
                comm = max(cost * self.commission, self.min_commission)
                total = cost + comm
            self.cash -= total
            self.locked[i] += shares

        cash_penalty = 0.0
        if self.cash > config.MAX_CASH:
            cash_penalty = config.LAMBDA_CASH_PENALTY * (
                (self.cash - config.MAX_CASH) / self.init_capital)
            self._force_reduce_cash(prices, can_buy)

        turnover = (np.abs(target_weights - self.prev_weights)).sum()

        if self.phase == "open":
            nav_after = self._compute_nav()
            self.phase = "close"
        else:
            nav_after = self._compute_nav()
            self.current_idx += 1
            self.phase = "open"
            self.holdings += self.locked
            self.locked[:] = 0
            self.episode_day += 1

        episode_done = self.episode_day >= self.episode_len
        data_done = self.current_idx >= len(self.dates)
        done = episode_done or data_done

        self.prev_weights = target_weights.copy()
        reward = np.log(nav_after / nav_before + 1e-10) \
            - config.LAMBDA_TURNOVER * turnover - cash_penalty

        info = {"nav": nav_after, "turnover": turnover,
                "cash_penalty": cash_penalty}
        return self._get_state() if not done else None, reward, done, info

    def _force_reduce_cash(self, prices, can_buy):
        """Buy stocks to bring cash below MAX_CASH."""
        buyable = np.where(
            can_buy & ~np.isnan(prices) & (prices > 0))[0]
        if len(buyable) == 0:
            return
        current_val = (self.holdings + self.locked).astype(float) * \
            np.nan_to_num(prices, 0)
        weights = current_val[buyable]
        w_sum = weights.sum()
        if w_sum < 1e-8:
            weights = np.ones(len(buyable)) / len(buyable)
        else:
            weights = weights / w_sum

        excess = self.cash - config.MAX_CASH
        for j, idx in enumerate(buyable):
            alloc = excess * weights[j]
            shares = int(alloc / (prices[idx] * (1 + self.commission))
                         / self.lot) * self.lot
            if shares <= 0:
                continue
            cost = shares * prices[idx]
            comm = max(cost * self.commission, self.min_commission)
            total = cost + comm
            if total > self.cash:
                continue
            self.cash -= total
            self.locked[idx] += shares
            if self.cash <= config.MAX_CASH:
                break

    def clone(self):
        """Create a lightweight copy for GRPO group sampling."""
        import copy
        new_env = copy.copy(self)
        new_env.holdings = self.holdings.copy()
        new_env.locked = self.locked.copy()
        new_env.prev_weights = self.prev_weights.copy()
        return new_env