# `TFT_cuda.py` 改进说明

本文档说明相对于原始文件 `TFT.py`，拷贝文件 `TFT_cuda.py` 做了哪些改进、每处改进的原因，以及这些改动对程序行为的影响。

## 1. 改动目标

本次改动的目标有两个：

1. 在不修改原始 `TFT.py` 的前提下，拷贝出一份新文件 `TFT_cuda.py`。
2. 在新文件上以尽可能小的代价修复阻塞运行的 bug，并补充设备选择逻辑：
   - 有 CUDA 时默认优先使用 CUDA
   - 无 CUDA 时自动回退到 CPU

改动过程中遵循了“尽量尊重原有逻辑”的原则，没有重写模型整体结构，也没有改动数据集构造、标签定义、训练目标等核心设计。

## 2. 文件层面的处理

### 2.1 新增拷贝文件

- 原文件：`stock-TFT-deeplearning/TFT.py`
- 新文件：`stock-TFT-deeplearning/TFT_cuda.py`

这样做的原因是：

- 保留原始版本，便于对照和回退
- 将 CUDA 支持与 bug 修复集中在新文件中，降低对现有代码的侵入性

## 3. 功能性改进总览

`TFT_cuda.py` 中一共做了三处真正影响功能的改动：

1. 修复 `post_attn_grn` 初始化参数传递错误
2. 修复动态特征嵌入与变量选择网络之间的张量形状不匹配问题
3. 新增 device 自动选择与模型/输入迁移逻辑

其中前两项属于 bug 修复，第三项属于 CUDA 支持增强。

## 4. 详细改进说明

### 4.1 修复 `post_attn_grn` 的构造参数错误

#### 原始写法

原始文件 `TFT.py` 中，`CompetitionTFT.__init__` 里的代码如下：

```python
self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
```

对应位置在：

- 原文件 `TFT.py:128`
- 新文件修复后位置 `TFT_cuda.py:138`

#### 问题原因

`GatedResidualNetwork` 的构造函数定义为：

```python
def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
```

第三个位置参数是 `output_dim`，第四个参数才是 `dropout`。

因此原代码中的：

```python
GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
```

实际上把 `dropout=0.1` 误传给了 `output_dim`。这会导致内部 `nn.Linear(hidden_dim, output_dim)` 试图使用浮点数 `0.1` 作为输出维度，最终在模型初始化阶段直接报错。

也就是说，这个问题会让程序在模型还没真正开始前向传播之前就失败。

#### 修复后的写法

`TFT_cuda.py` 中改为：

```python
self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
```

对应位置：

- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:138)

#### 为什么这是“最小代价修改”

因为这里只修正了参数传递方式，没有改变：

- 网络层数
- 隐藏维度
- dropout 数值
- 前向传播路径

本质上只是把“错误的参数绑定”改成了“正确的参数绑定”。

### 4.2 修复动态特征嵌入与变量选择网络的形状不匹配

#### 原始写法

原始文件中动态特征的嵌入与输入变量选择网络的写法是：

```python
self.dynamic_embedding = nn.Linear(dynamic_input_dim, hidden_dim)
```

以及：

```python
embedded_dynamic = self.dynamic_embedding(dynamic_x)
expanded_dynamic = embedded_dynamic.unsqueeze(-2)
selected_features = self.vsn(expanded_dynamic)
```

对应位置：

- 原文件 `TFT.py:114`
- 原文件 `TFT.py:135-140`

#### 问题原因

这里的核心矛盾在于 `VariableSelectionNetwork` 的设计假设是：

```python
x.shape == [batch, seq_len, num_vars, hidden_dim]
```

也就是说，它希望“每个动态变量都有一份单独的 hidden embedding”，然后在变量维度 `num_vars` 上做选择。

但原始代码先做的是：

```python
dynamic_x: [batch, seq_len, 5]
self.dynamic_embedding(dynamic_x): [batch, seq_len, hidden_dim]
```

这一步其实已经把 5 个动态变量混合到了一起，随后再执行：

```python
unsqueeze(-2)
```

得到的是：

```python
[batch, seq_len, 1, hidden_dim]
```

这会让 `VariableSelectionNetwork` 误以为“只有 1 个变量”，但它内部又是按 `dynamic_input_dim=5` 构建的，因此在 `joint_grn` 和残差映射阶段会出现维度不匹配，最终导致矩阵乘法报错。

换句话说，原始代码并不是简单的“少了一个维度”，而是“变量选择模块期待 5 个变量嵌入，但上游实际只提供了 1 组混合后的嵌入”。

#### 修复思路

为了尽可能尊重原有逻辑，修复没有去删除 `VariableSelectionNetwork`，而是反过来让动态特征输入真正满足它的设计前提。

原始设计显然是希望：

- 每个时间步有 `dynamic_input_dim` 个动态变量
- 每个变量单独映射到 `hidden_dim`
- 然后在变量维度上做选择

因此修复方式是：

1. 把动态嵌入层从“5 维整体映射到 hidden”改成“单变量 1 维映射到 hidden”
2. 在前向传播里先给输入补一个末尾维度，再交给线性层

#### 修复后的写法

动态嵌入层由：

```python
self.dynamic_embedding = nn.Linear(dynamic_input_dim, hidden_dim)
```

改为：

```python
self.dynamic_embedding = nn.Linear(1, hidden_dim)
```

对应位置：

- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:124)

前向传播中由：

```python
embedded_dynamic = self.dynamic_embedding(dynamic_x)
expanded_dynamic = embedded_dynamic.unsqueeze(-2)
selected_features = self.vsn(expanded_dynamic)
```

改为：

```python
embedded_dynamic = self.dynamic_embedding(dynamic_x.unsqueeze(-1))
selected_features = self.vsn(embedded_dynamic)
```

对应位置：

- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:145)
- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:149)

#### 修改后的张量形状变化

现在各阶段的形状关系如下：

```text
dynamic_x                           -> [batch, seq_len, 5]
dynamic_x.unsqueeze(-1)             -> [batch, seq_len, 5, 1]
dynamic_embedding(...)              -> [batch, seq_len, 5, hidden_dim]
vsn(...)                            -> [batch, seq_len, hidden_dim]
```

这与 `VariableSelectionNetwork` 的实现完全对齐。

#### 为什么这是“最小代价修改”

因为这次修复没有改变模型的大方向，仍然保留了原本的：

- 动态特征嵌入
- 变量选择网络
- LSTM
- Multi-head Attention
- 输出层

只是把动态特征送入 `VariableSelectionNetwork` 的方式，修正为与该模块的输入假设一致。

### 4.3 新增 device 自动选择逻辑

#### 原始情况

原始 `TFT.py` 中没有任何显式设备选择逻辑，主要表现为：

- 没有 `torch.cuda.is_available()`
- 没有 `torch.device(...)`
- 没有 `model.to(device)`
- 没有把输入张量放到对应设备

因此即便运行环境本身支持 CUDA，原代码也不会自动使用 GPU。

#### 修复后的写法

在 `__main__` 中新增：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

对应位置：

- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:181)

并将模型迁移到该设备：

```python
model = CompetitionTFT(dynamic_input_dim=5, static_input_dim=2, hidden_dim=64).to(device)
```

对应位置：

- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:184)

同时将示例输入张量也创建在相同设备：

```python
dummy_dynamic = torch.randn(32, 60, 5, device=device)
dummy_static = torch.randn(32, 2, device=device)
```

对应位置：

- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:187)
- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:188)

#### 这样修改的原因

如果只把模型放到 GPU，而输入还留在 CPU，就会触发设备不一致错误。

因此完整的设备支持至少需要保证三点：

1. 先判断当前环境是否可用 CUDA
2. 模型放到选中的 device 上
3. 输入张量也放到相同 device 上

这样才能真正实现：

- 有 GPU 时自动走 GPU
- 无 GPU 时自动走 CPU

#### 输出信息增强

为便于运行时确认实际使用的设备，新增了一行输出：

```python
print(f"当前使用设备: {device}")
```

对应位置：

- [TFT_cuda.py](/home/sczli/Programs/DeepLearning/01_dl_assignment/stock-TFT-deeplearning/TFT_cuda.py:191)

这属于辅助性增强，不改变模型计算逻辑，但能直接帮助确认程序是否已经进入 CUDA 路径。

## 5. 未改动的部分

为了控制修改范围，以下内容保持原样，没有重写：

- `StockDataset` 的整体构造逻辑
- 滚动标准化方案
- 标签收益率的定义方式
- `GatedResidualNetwork` 的主体实现
- `VariableSelectionNetwork` 的主体实现
- LSTM 与多头注意力的整体连接方式
- 最终输出单值收益率分数的设计

这意味着本次工作重点是“修复现有代码使其可运行，并补齐设备选择能力”，而不是重新设计模型。

## 6. 改动后的运行结果

使用解释器：

```text
/data/sczli/conda_env/pytorch_env/bin/python
```

运行：

```text
stock-TFT-deeplearning/TFT_cuda.py
```

实际得到输出：

```text
当前使用设备: cuda
模型输出的选股打分形状: torch.Size([32])
模型已准备好进行基于得分的 Top-K 轮动策略回测！
```

这说明：

1. 程序已经不再卡在初始化 bug 上
2. 变量选择网络相关的张量形状已经对齐
3. 当前环境下已成功走到 CUDA 路径
4. 前向传播可以正常完成

## 7. 最终总结

`TFT_cuda.py` 相比原始 `TFT.py`，本次改进的核心价值是：

1. 修复了一个会直接导致模型初始化失败的参数传递错误
2. 修复了一个会导致变量选择网络输入维度不匹配的结构性问题
3. 补上了完整的 device 自动选择逻辑，使程序能在有 CUDA 时优先使用 GPU，无 CUDA 时自动回退 CPU

这些改动都尽量控制在最小范围内，目的不是改变原模型思路，而是让原有设计真正跑起来，并具备明确的 CUDA 使用能力。
