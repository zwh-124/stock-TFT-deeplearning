"""
DEPRECATED: Supervised backtest removed in v2 architecture.
The model now uses TFTEncoder -> PortfolioPolicy end-to-end with GRPO.
Use backtest_rl.py for RL-based backtesting.
"""


def main():
    print("DEPRECATED: Supervised backtest removed in v2 architecture.")
    print("Use 'python backtest_rl.py' for RL-based backtesting.")


if __name__ == "__main__":
    main()
