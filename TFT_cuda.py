import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ---------------------- 1. 严防未来函数的数据预处理与Dataset ----------------------
class StockDataset(Dataset):
    def __init__(self, df, seq_len=60, pred_horizon=1):
        """
        df: 必须包含 'code'(股票代码), 'date', 动态特征(如close, vol等), 静态特征(如industry_code)
        注意：df 必须是按日期排好序的原始数据，绝对不能提前做全局标准化！
        """
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.dynamic_features = ['open', 'high', 'low', 'close', 'volume']  # 示例动态特征
        self.static_features = ['industry_code', 'log_market_cap']          # 示例静态特征

        self.samples = []

        # 核心：按股票分组，并在组内进行严格的滚动窗口采样和标准化
        for code, group in df.groupby('code'):
            group = group.sort_values('date').reset_index(drop=True)
            if len(group) < seq_len + pred_horizon:
                continue

            # 提取静态特征（同一只股票取第一条即可）
            static_data = group[self.static_features].iloc[0].values.astype(np.float32)

            # 滚动标准化 (Rolling Standardization) - 严防未来函数
            # 对每一个时间步 t，只用 t 之前的历史数据计算均值和标准差
            dynamic_data = group[self.dynamic_features].values.astype(np.float32)
            norm_dynamic = np.zeros_like(dynamic_data)

            for i in range(seq_len, len(group)):
                hist_window = dynamic_data[i - seq_len:i]  # [i-seq_len, i-1] 共 seq_len 天
                mean = hist_window.mean(axis=0)
                std = hist_window.std(axis=0) + 1e-8
                norm_dynamic[i] = (dynamic_data[i] - mean) / std

            # 构造滑动窗口样本
            for i in range(seq_len, len(group) - pred_horizon + 1):
                # 输入是过去 seq_len 天的标准化后的动态特征
                x_dynamic = norm_dynamic[i - seq_len:i]
                # 标签是未来第 pred_horizon 天的收盘价相对于当天的收益率
                current_close = dynamic_data[i - 1, 3]  # 假设 close 在第4列
                future_close = dynamic_data[i + pred_horizon - 1, 3]
                y_return = (future_close - current_close) / current_close

                self.samples.append({
                    'x_dynamic': x_dynamic,
                    'x_static': static_data,
                    'y': np.array([y_return], dtype=np.float32),
                    'date': group.iloc[i]['date'],
                    'code': code
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['x_dynamic']),
            torch.tensor(sample['x_static']),
            torch.tensor(sample['y'])
        )


# ---------------------- 2. 适配比赛的 TFT 模型定义 ----------------------
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        output_dim = output_dim if output_dim is not None else input_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(output_dim, output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip_proj(x)
        x = F.elu(self.input_proj(x))
        x = self.fc1(x)
        x = self.dropout(F.elu(x))
        x = self.fc2(x)
        gate = torch.sigmoid(self.gate(x))
        x = gate * x + (1 - gate) * residual
        return self.layer_norm(x)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dim, dropout=0.1):
        super().__init__()
        self.joint_grn = GatedResidualNetwork(
            input_dims * hidden_dim,
            hidden_dim,
            output_dim=input_dims,
            dropout=dropout
        )
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout) for _ in range(input_dims)
        ])

    def forward(self, x):
        batch, seq_len, num_vars, hidden_dim = x.shape
        flat_x = x.reshape(batch, seq_len, -1)
        selection_weights = self.joint_grn(flat_x)
        selection_weights = F.softmax(selection_weights, dim=-1).unsqueeze(-1)
        processed_vars = torch.stack([grn(x[..., i, :]) for i, grn in enumerate(self.variable_grns)], dim=-2)
        selected_features = torch.sum(processed_vars * selection_weights, dim=-2)
        return selected_features


class CompetitionTFT(nn.Module):
    def __init__(self, dynamic_input_dim, static_input_dim, hidden_dim=64, seq_len=60, num_heads=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len

        # 1. 特征嵌入
        self.dynamic_embedding = nn.Linear(1, hidden_dim)
        self.static_embedding = nn.Linear(static_input_dim, hidden_dim)

        # 2. 静态特征处理分支
        self.static_encoder = GatedResidualNetwork(hidden_dim, hidden_dim)
        self.enrichment_grn = GatedResidualNetwork(hidden_dim, hidden_dim)
        self.attention_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        # 3. 变量选择网络
        self.vsn = VariableSelectionNetwork(dynamic_input_dim, hidden_dim, dropout)

        # 4. 时序建模 (LSTM + Attention)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.post_attn_grn = GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)

        # 5. 输出层：直接输出一个预测分数用于 Top-K 选股
        # 这里改回单一输出，直接预测未来收益率，方便排序
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, dynamic_x, static_x):
        embedded_dynamic = self.dynamic_embedding(dynamic_x.unsqueeze(-1))
        embedded_static = self.static_embedding(static_x)
        static_context = self.static_encoder(embedded_static)

        selected_features = self.vsn(embedded_dynamic)

        lstm_out, _ = self.lstm(selected_features)
        enrichment = self.enrichment_grn(lstm_out + static_context.unsqueeze(1))

        attn_out, _ = self.multihead_attn(enrichment, enrichment, enrichment)
        attn_gate = self.attention_gate(static_context).unsqueeze(1)
        gated_attn = attn_gate * attn_out + (1 - attn_gate) * enrichment
        final_features = self.post_attn_grn(gated_attn)

        # 输出最后一天的预测收益率分数
        out = self.fc_out(final_features[:, -1, :])
        return out.squeeze(-1)


# ---------------------- 3. 训练与评估指标 (IC & ICIR) ----------------------
def calculate_ic_ir(predictions, targets):
    """计算 Rank IC 和 ICIR"""
    ic_list = []
    # 假设 predictions 和 targets 形状为 [total_samples,]
    # 在实际训练中，需要按日期分组来计算每天截面的 IC
    # 这里简化演示整体相关性
    corr = np.corrcoef(predictions, targets)[0, 1]
    return corr  # 实际比赛中需按 date.groupby 计算每日 IC 后再求均值和波动


if __name__ == "__main__":
    # 模拟数据加载与训练流程
    # dummy_df = pd.read_csv('your_stock_data.csv')
    # dataset = StockDataset(dummy_df, seq_len=60)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False) # 严禁 shuffle=True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型
    model = CompetitionTFT(dynamic_input_dim=5, static_input_dim=2, hidden_dim=64).to(device)

    # 模拟一次前向传播
    dummy_dynamic = torch.randn(32, 60, 5, device=device)
    dummy_static = torch.randn(32, 2, device=device)
    pred_scores = model(dummy_dynamic, dummy_static)

    print(f"当前使用设备: {device}")
    print(f"模型输出的选股打分形状: {pred_scores.shape}")  # torch.Size([32])
    print("模型已准备好进行基于得分的 Top-K 轮动策略回测！")
