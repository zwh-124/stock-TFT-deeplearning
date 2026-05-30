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

# ====== 股票池选择 ======
# 旧开关，向后兼容：仅当 UNIVERSE 为 None 时生效
USE_CSI300 = True

# 新的可组合选股规格。非 None 时优先于 USE_CSI300 生效。
# 语义：include 各选择器结果取并集得到候选池，再减去 exclude 的并集。
# 北交所 / ST 始终被强制剔除（见 data_loader.filter_stocks），与此处无关。
#
# 选择器类型：
#   {"type": "index",    "code": "000300.SH"}         指数成分（可用 000300.SH / 399006.SZ）
#   {"type": "market",   "value": "主板"}             按板块（主板/创业板/科创板/北交所），value 可为 str 或 list
#   {"type": "industry", "value": ["银行", "半导体"]}  按行业，value 可为 str 或 list
#   {"type": "area",     "value": "深圳"}             按地域，value 可为 str 或 list
#   {"type": "codes",    "value": ["600519.SH"]}      按显式代码列表
#   {"type": "all"}                                    全市场（include 为空亦视为全市场）
#
# 示例与各设置下大致股票数见 docs/UNIVERSE.md。设为 None 则回退到 USE_CSI300。
UNIVERSE = {
    "name": "csi300",
    "include": [
        {"type": "index", "code": "000300.SH"},
    ],
    "exclude": [],
}

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

HIDDEN_DIM = 128
NUM_HEADS = 4
DROPOUT = 0.4
LR = 5e-4
EPOCHS = 300
BATCH_SIZE = 2048
PATIENCE = 25

N_HOLD = 20
K_SWAP = 3
INIT_CAPITAL = 1_000_000

# ====== RL (GRPO) 超参 ======
N_BINS = 6
BINS = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20]
GRPO_G = 8
GRPO_BETA = 0.04
GRPO_REF_REFRESH = 150
LAMBDA_TURNOVER = 5e-5
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

# ====== 改进方案超参 ======
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-3
LAMBDA_AUX = 0.05
LAMBDA_BENCHMARK = 0.5

# ====== 诊断开关（默认不影响原算法）======
DIAG_INTERVAL = 10       # 每多少个 update 打印一次诊断指标；设 0 关闭
DIAG_SMOKE_TEST = False  # True 时用合成奖励测试 RL 管线（需手动开启）

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
