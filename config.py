import os

# ====== 修改这里为你本地的数据路径 ======
DATA_ROOT = os.environ.get("STOCK_DATA_ROOT",
                           "/data/sczli/Adata")
CACHE_DIR = os.environ.get("STOCK_CACHE_DIR",
                           "/data/sczli/Adata/cache")
# ==========================================

DAILY_DIR = os.path.join(DATA_ROOT, "daily")
METRIC_DIR = os.path.join(DATA_ROOT, "metric")
MONEYFLOW_DIR = os.path.join(DATA_ROOT, "moneyflow")
ST_DIR = os.path.join(DATA_ROOT, "stock_st")
BASIC_CSV = os.path.join(DATA_ROOT, "basic.csv")
MARKET_DIR = os.path.join(DATA_ROOT, "market")
INDEX_WEIGHT_DIR = os.path.join(DATA_ROOT, "index_weight")

USE_CSI300 = True

TRAIN_START = "20240701"
TRAIN_END = "20241231"
VAL_START = "20250101"
VAL_END = "20250301"
TEST_START = "20250301"
TEST_END = "20250601"

SEQ_LEN = 20
PRED_HORIZON = 1

HIDDEN_DIM = 32
NUM_HEADS = 4
DROPOUT = 0.1
LR = 1e-4
EPOCHS = 2
BATCH_SIZE = 32
PATIENCE = 5

N_HOLD = 20
K_SWAP = 3
INIT_CAPITAL = 1_000_000
