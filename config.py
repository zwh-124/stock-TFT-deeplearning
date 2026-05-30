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
IPO_SKIP_DAYS = 40
STATIC_EMBED_DIM = 16

TRAIN_START = "20220701"
TRAIN_END = "20240701"
VAL_START = "20240702"
VAL_END = "20250430"
TEST_START = "20250501"
TEST_END = "20260501"

SEQ_LEN = 30
PRED_HORIZON = 1

HIDDEN_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.4
LR = 5e-4
EPOCHS = 100
BATCH_SIZE = 2048
PATIENCE = 25

N_HOLD = 20
K_SWAP = 3
INIT_CAPITAL = 1_000_000

# ====== RL (GRPO) 超参 ======
N_BINS = 6
BINS = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20]
GRPO_G = 2
GRPO_BETA = 0.04
GRPO_REF_REFRESH = 150
LAMBDA_TURNOVER = 1e-4
COMMISSION = 3e-4
STAMP = 1e-3
MIN_COMMISSION = 5.0
LIMIT_PCT = 0.10
LOT = 100
RL_STEPS = 1500
LR_POLICY = 3e-4
LR_ENCODER = 1e-5
N_EXTRA_STATE = 6

# ====== Episode & Competition Constraints ======
EPISODE_LEN = 10
MAX_CASH = 150_000
LAMBDA_CASH_PENALTY = 0.01

# ====== Diffusion Denoiser 超参 ======
USE_DIFFUSION_DENOISER = True
DIFFUSION_T = 200
DIFFUSION_BETA_START = 1e-4
DIFFUSION_BETA_END = 0.02
DIFFUSION_HIDDEN_DIM = 256
DIFFUSION_TIME_DIM = 128
DENOISE_T_START = 30
DENOISE_STEPS = 3
LAMBDA_DENOISE = 0.1
