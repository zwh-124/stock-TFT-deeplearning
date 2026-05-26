# 项目说明书：基于 TFT 的 A 股 Top-K 轮动选股系统

## 一、项目概述

本项目实现了一个基于 Temporal Fusion Transformer (TFT) 的 A 股量化选股系统，用于深度学习基础大作业。核心任务是：利用历史量价数据和基本面数据训练深度学习模型，预测个股未来收益率，并基于预测分数进行 Top-K 轮动策略回测。

评估指标：截面 Rank IC 和 ICIR（信息比率）。

## 二、文件结构与模块说明

| 文件 | 职责 |
|------|------|
| `config.py` | 全局配置：数据路径、超参数、回测参数 |
| `data_loader.py` | 数据加载与合并：读取 daily/metric/moneyflow CSV，过滤 ST 和北交所，合并为统一 DataFrame |
| `feature_engine.py` | 特征工程：技术指标（MA偏离、RSI、MACD、ATR、布林带宽）、资金流特征、基本面特征 |
| `dataset.py` | PyTorch Dataset：向量化滚动窗口标准化 + 滑动窗口采样 + 样本缓存（hash 校验 + 原子写入），严防未来函数 |
| `model.py` | TFT 模型定义：GRN、VSN、LSTM + Multi-Head Attention |
| `train.py` | 训练流程：MSE 损失、Early Stopping、IC/ICIR 评估 |
| `backtest.py` | 回测系统：基于样本外测试集的 Top-K 轮动策略、净值曲线、Sharpe/最大回撤 |
| `predict.py` | 推理模块：加载模型生成最新交易信号 |

## 三、数据流水线

```
/data/sczli/Adata/daily/*.csv  ─┐
/data/sczli/Adata/metric/*.csv ─┼─→ data_loader.build_merged_dataset()
/data/sczli/Adata/moneyflow/*.csv─┘         │
                                            ▼
                                   feature_engine.build_features()
                                            │
                                            ▼
                                   dataset.StockDataset (滚动标准化)
                                            │
                                            ▼
                                   train.py → model → backtest.py
```

## 四、模型架构 (CompetitionTFT)

```
输入: dynamic_x (B, 20, D_dyn), static_x (B, D_stat)
  │                                    │
  ▼                                    ▼
Linear(1, 32) per var → (B,20,D,32)  Linear(D_stat, 32)
  │                                    │
  ▼                                    ▼
VariableSelectionNetwork              GRN (static_encoder)
  │                                    │
  ▼                                    │
+ Positional Embedding                │
  │                                    │
  ▼                                    │
LSTM (2层, hidden=32)                  │
  │                                    │
  ▼                                    ▼
GRN (enrichment) ←── static_context broadcast
  │
  ▼
Multi-Head Attention (4 heads)
  │
  ▼
Gated fusion (attention_gate from static)
  │
  ▼
GRN (post_attn)
  │
  ▼
Linear(32, 1) → 预测收益率分数 (用于截面排序选股)
```

关键组件：
- **GRN (Gated Residual Network)**：带门控机制的残差网络，控制信息流通
- **VSN (Variable Selection Network)**：自动学习各特征的重要性权重
- **位置编码**：Learnable Embedding，为时序位置提供信息
- **2层 LSTM**：捕捉时序依赖关系
- **Multi-Head Self-Attention**：捕捉长程依赖

## 五、CACHE_DIR 的作用

`CACHE_DIR` 定义在 `config.py` 第 6-7 行：

```python
CACHE_DIR = os.environ.get("STOCK_CACHE_DIR",
                           "/data/sczli/Adata/cache")
```

默认值为 `/data/sczli/Adata/cache`。它的作用是：

| 缓存文件 | 生成者 | 用途 |
|----------|--------|------|
| `merged_csi300_<start>_<end>.pkl` | `data_loader.py` | 合并后的 DataFrame 缓存（按日期范围区分），避免每次重新读取上千个 CSV |
| `train_<hash>.pkl` / `val_<hash>.pkl` / `test_<hash>.pkl` | `dataset.py` | 构建好的样本缓存（hash 基于特征列表、配置参数、数据指纹），避免重复标准化计算 |
| `best_model.pt` | `train.py` | 最优模型权重检查点 |
| `backtest_nav.csv` | `backtest.py` | 回测净值曲线结果 |
| `latest_signals.csv` | `predict.py` | 最新交易信号 |

**缓存安全机制**：
- **hash 命名**：`dataset.py` 的样本缓存文件名包含输入数据的 MD5 hash，数据或配置变化时自动失效
- **原子写入**：先写 `.tmp` 临时文件再 `os.replace` 重命名，防止中断导致不完整缓存被读取
- **日期范围区分**：`data_loader.py` 的缓存文件名包含 start/end 日期，不同区间互不干扰

## 六、训练时 RAM 占用分析

### 6.1 模型参数量

以默认配置（`dynamic_input_dim=23`, `static_input_dim=1`, `hidden_dim=32`）估算：

| 组件 | 参数量（约） |
|------|-------------|
| dynamic_embedding (Linear 1→32) | 64 |
| static_embedding (Linear 1→32) | 64 |
| pos_embedding (Embedding 20×32) | 640 |
| static_encoder (GRN) | ~4K |
| enrichment_grn (GRN) | ~4K |
| attention_gate (Linear 32→32) | 1,056 |
| VSN joint_grn (input=23×32=736) | ~34K |
| VSN 23个 variable_grns | ~92K |
| LSTM (2层, 32→32) | ~17K |
| Multi-Head Attention (4 heads) | ~4K |
| post_attn_grn (GRN) | ~4K |
| fc_out (Linear 32→1) | 33 |
| **总计** | **~160K 参数 ≈ 0.6 MB (float32)** |

### 6.2 CPU RAM 占用

| 来源 | 估算 |
|------|------|
| 合并后 DataFrame（CSI300, 6个月） | ~300股 × 120天 × 30列 × 8B ≈ **8 MB** |
| 特征计算中间变量 | ~50-100 MB |
| `StockDataset.samples` 列表 | 471959样本 × (20×23×4B + overhead) ≈ **1.0 GB** |
| DataLoader workers (4个) | 每个 worker fork 后 COW，实际增量 ~200-500 MB |
| 模型 + 优化器状态 (CPU 副本) | ~5 MB |
| **CPU RAM 总计** | **约 2-4 GB** |

> `StockDataset` 是最大的内存消耗者。每个样本存储一个 (20, 23) 的 float32 数组作为输入特征，约 1.8 KB/样本。

### 6.3 GPU VRAM 占用

以 `batch_size=32` 估算单 batch 前向+反向传播：

| 来源 | 估算 |
|------|------|
| 模型参数 | ~0.6 MB |
| 优化器状态 (Adam: 2× params) | ~1.2 MB |
| 输入张量 `dynamic_x` (32×20×23) | ~58 KB |
| `embedded_dynamic` (32×20×23×32) | **1.3 MB** |
| VSN 中间计算 (flat_x + processed_vars) | ~5 MB |
| LSTM 隐状态 + 输出 | ~0.2 MB |
| Attention Q/K/V + scores | ~0.5 MB |
| 反向传播梯度 (约等于前向激活) | ~7 MB |
| PyTorch/CUDA 固定开销 | ~300 MB |
| **GPU VRAM 总计** | **约 300-400 MB** |

### 6.4 结论

- **CPU RAM**：需要 **2-4 GB**，主要瓶颈是 Dataset 中预计算的样本数组
- **GPU VRAM**：需要 **~350 MB**，模型较小，GPU 利用率偏低（瓶颈在 CPU 端数据搬运）
- 对于任意现代 GPU 和 8GB+ 内存的机器均可运行

## 七、能否不输出任何文件（零磁盘占用）？

**结论：理论可行但实际不推荐。**

当前代码会向 `CACHE_DIR` 写入以下文件：

| 文件 | 是否可省略 | 代价 |
|------|-----------|------|
| `merged_csi300_*.pkl` (~数百MB) | 可省略 | 每次运行需重新读取 1200+ 个 CSV 文件，耗时数分钟 |
| `train_*.pkl` / `val_*.pkl` / `test_*.pkl` (~数百MB-1GB) | 可省略 | 每次运行需重新构建样本（向量化后约 1-2 分钟） |
| `best_model.pt` (~0.6MB) | **不可省略** | 没有 checkpoint 则无法做推理和回测 |
| `backtest_nav.csv` | 可省略 | 改为仅打印到 stdout |
| `latest_signals.csv` | 可省略 | 改为仅打印到 stdout |

**如果你希望最小化磁盘写入**，可以做以下修改：

1. 删除 `data_loader.py` 中的 pickle 缓存逻辑，每次从源 CSV 加载
2. 删除 `dataset.py` 中的样本缓存逻辑，每次重新构建
3. 将 `backtest.py` 和 `predict.py` 中的 `to_csv` 调用替换为 `print()` 输出
4. 模型权重 `best_model.pt` 无法避免——它是训练产出的核心产物，推理和回测都依赖它

**最小不可避免的磁盘输出**：`best_model.pt` 约 0.6 MB。

如果连这个也不想保存（纯粹的训练+立即回测一体化），可以把 train → predict → backtest 整合到一个脚本中，模型权重全程驻留在内存中不落盘。但这意味着每次回测都必须重新训练，不切实际。

## 八、运行方式

```bash
# 1. 训练（生成 best_model.pt）
python train.py

# 2. 样本外回测（使用 TEST_START ~ TEST_END 区间，需要先训练）
python backtest.py

# 3. 生成最新信号（需要先训练）
python predict.py
```

**数据划分**：

| 用途 | 区间 | 配置项 |
|------|------|--------|
| 训练 | 2024-07-01 ~ 2024-12-31 | TRAIN_START / TRAIN_END |
| 验证（early stopping） | 2025-01-01 ~ 2025-03-01 | VAL_START / VAL_END |
| 测试（样本外回测） | 2025-03-01 ~ 2025-06-01 | TEST_START / TEST_END |

## 九、超参数配置 (config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| SEQ_LEN | 20 | 输入序列长度（交易日） |
| PRED_HORIZON | 1 | 预测未来第几天的收益 |
| HIDDEN_DIM | 32 | 模型隐藏层维度 |
| NUM_HEADS | 4 | 注意力头数 |
| DROPOUT | 0.1 | Dropout 率 |
| LR | 1e-4 | 学习率 |
| EPOCHS | 2 | 最大训练轮数 |
| BATCH_SIZE | 32 | 批大小 |
| PATIENCE | 5 | Early Stopping 耐心值 |
| N_HOLD | 20 | 持仓股票数量 |
| K_SWAP | 3 | 每日最大换仓数 |
| USE_CSI300 | True | 是否限制股票池为沪深300 |
