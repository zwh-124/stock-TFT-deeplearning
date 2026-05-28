import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionDenoiser(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim=256, time_dim=128,
                 n_timesteps=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.flat_dim = feature_dim * seq_len
        self.n_timesteps = n_timesteps
        self.time_dim = time_dim

        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod',
                             torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod',
                             torch.sqrt(1.0 - alpha_cumprod))

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim // 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_dim, self.flat_dim)

    def _predict_noise(self, x_t_flat, t):
        t_emb = sinusoidal_embedding(t, self.time_dim // 2)
        t_emb = self.time_mlp(t_emb)
        h = self.net(x_t_flat)
        h = h + t_emb
        h = self.mid(h)
        return self.out(h)

    def compute_loss(self, x_0):
        B = x_0.shape[0]
        x_flat = x_0.reshape(B, -1)
        t = torch.randint(0, self.n_timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_flat)
        sqrt_ac = self.sqrt_alpha_cumprod[t].unsqueeze(-1)
        sqrt_omac = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
        x_t = sqrt_ac * x_flat + sqrt_omac * noise
        pred_noise = self._predict_noise(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def denoise(self, x_raw, t_start=50, n_steps=5):
        B = x_raw.shape[0]
        x_t = x_raw.reshape(B, -1)
        timesteps = torch.linspace(t_start, 0, n_steps + 1).long()
        for i in range(len(timesteps) - 1):
            t_cur = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            t_batch = torch.full((B,), t_cur, device=x_raw.device, dtype=torch.long)
            eps_pred = self._predict_noise(x_t, t_batch)
            ac_cur = self.alpha_cumprod[t_cur]
            sqrt_ac_cur = self.sqrt_alpha_cumprod[t_cur]
            sqrt_omac_cur = self.sqrt_one_minus_alpha_cumprod[t_cur]
            x0_pred = (x_t - sqrt_omac_cur * eps_pred) / sqrt_ac_cur
            if t_next > 0:
                sqrt_ac_next = self.sqrt_alpha_cumprod[t_next]
                sqrt_omac_next = self.sqrt_one_minus_alpha_cumprod[t_next]
                x_t = sqrt_ac_next * x0_pred + sqrt_omac_next * eps_pred
            else:
                x_t = x0_pred
        return x_t.reshape(B, self.seq_len, self.feature_dim)


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


class TFTEncoder(nn.Module):
    def __init__(self, dynamic_input_dim, static_input_dim, hidden_dim=64,
                 seq_len=60, num_heads=4, dropout=0.1, denoiser=None):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.denoiser = denoiser

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

    def forward(self, dynamic_x, static_x):
        seq_len = dynamic_x.shape[1]

        if self.denoiser is not None:
            with torch.no_grad():
                denoised = self.denoiser.denoise(
                    dynamic_x,
                    t_start=config.DENOISE_T_START,
                    n_steps=config.DENOISE_STEPS,
                )
            dynamic_x = denoised.detach()

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

        return final_features[:, -1, :]


class CompetitionTFT(nn.Module):
    def __init__(self, dynamic_input_dim, static_input_dim, hidden_dim=64,
                 seq_len=60, num_heads=4, dropout=0.1):
        super().__init__()
        self.dynamic_input_dim = dynamic_input_dim
        self.encoder = TFTEncoder(dynamic_input_dim, static_input_dim,
                                  hidden_dim, seq_len, num_heads, dropout)
        self.fc_out = nn.Linear(hidden_dim, dynamic_input_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dynamic_input_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, dynamic_x, static_x):
        features = self.encoder(dynamic_x, static_x)
        pred = self.fc_out(features)
        gate_weights = self.feature_gate(features)
        return pred, gate_weights


class PortfolioPolicy(nn.Module):
    def __init__(self, hidden_dim, n_bins=6, n_extra_state=4, dropout=0.1):
        super().__init__()
        input_dim = hidden_dim + n_extra_state
        self.head_open = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins),
        )
        self.head_close = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_bins),
        )

    def forward(self, enc_features, port_state, mask, head="open"):
        x = torch.cat([enc_features, port_state], dim=-1)
        if head == "open":
            logits = self.head_open(x)
        else:
            logits = self.head_close(x)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        return torch.distributions.Categorical(logits=logits)
