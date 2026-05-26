import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_vars, hidden_dim, dropout=0.1):
        super().__init__()
        self.joint_grn = GatedResidualNetwork(
            num_vars * hidden_dim, hidden_dim,
            output_dim=num_vars, dropout=dropout)
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_vars)
        ])

    def forward(self, x):
        batch, seq_len, num_vars, hidden_dim = x.shape
        flat_x = x.reshape(batch, seq_len, -1)
        weights = self.joint_grn(flat_x)
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        processed = torch.stack(
            [grn(x[..., i, :]) for i, grn in enumerate(self.variable_grns)],
            dim=-2)
        return torch.sum(processed * weights, dim=-2)


class CompetitionTFT(nn.Module):
    def __init__(self, dynamic_input_dim, static_input_dim, hidden_dim=64,
                 seq_len=60, num_heads=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.dynamic_embedding = nn.Linear(1, hidden_dim)
        self.static_embedding = nn.Linear(static_input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)

        self.static_encoder = GatedResidualNetwork(hidden_dim, hidden_dim)
        self.enrichment_grn = GatedResidualNetwork(hidden_dim, hidden_dim)
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

        self.vsn = VariableSelectionNetwork(dynamic_input_dim, hidden_dim, dropout)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.post_attn_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, dropout=dropout)

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, dynamic_x, static_x):
        batch, seq_len, num_vars = dynamic_x.shape

        embedded_dynamic = self.dynamic_embedding(dynamic_x.unsqueeze(-1))
        selected_features = self.vsn(embedded_dynamic)

        pos_ids = torch.arange(seq_len, device=dynamic_x.device)
        selected_features = selected_features + self.pos_embedding(pos_ids)

        embedded_static = self.static_embedding(static_x)
        static_context = self.static_encoder(embedded_static)

        lstm_out, _ = self.lstm(selected_features)
        enrichment = self.enrichment_grn(
            lstm_out + static_context.unsqueeze(1))

        attn_out, _ = self.multihead_attn(enrichment, enrichment, enrichment)
        attn_gate = self.attention_gate(static_context).unsqueeze(1)
        gated_attn = attn_gate * attn_out + (1 - attn_gate) * enrichment
        final_features = self.post_attn_grn(gated_attn)

        out = self.fc_out(final_features[:, -1, :])
        return out.squeeze(-1)
