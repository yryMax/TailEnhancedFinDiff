import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torchview import draw_graph

# --- Adaptive Layer Norm ---
class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2)
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x, cond):
        gamma_beta = self.proj(cond)
        while gamma_beta.dim() < x.dim():
            gamma_beta = gamma_beta.unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma) + beta


# --- Dilated Temporal Convolution Block ---
class DilatedTemporalConv(nn.Module):
    def __init__(self, hidden_dim, dilations=(1, 2, 4)):
        super().__init__()
        layers = []
        for d in dilations:
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d),
                nn.GELU(),
            ])
        self.net = nn.Sequential(*layers[:-1])
        self.residual_proj = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        return self.net(x) + self.residual_proj(x)


# --- Factorized Block with AdaLN + Dilated Conv ---
class FactorizedBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, num_heads=4, dilations=(1, 2, 4)):
        super().__init__()
        self.time_conv = DilatedTemporalConv(hidden_dim, dilations=dilations)
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.adanorm1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.adanorm2 = AdaLayerNorm(hidden_dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x, cond, dropout_mask):
        B, N, T, C = x.shape

        # Pathway A: Dilated Temporal Conv
        x_time = x.view(B * N, T, C).transpose(1, 2)
        out_time = self.time_conv(x_time).transpose(1, 2).view(B, N, T, C)

        # Pathway B: Cross-asset Attention
        x_feature = x.transpose(1, 2).reshape(B * T, N, C)
        attn_mask = (dropout_mask == 0).repeat_interleave(T, dim=0)
        out_feature, _ = self.feature_attn(
            query=x_feature, key=x_feature, value=x_feature,
            key_padding_mask=attn_mask
        )
        out_feature = out_feature.view(B, T, N, C).transpose(1, 2)

        # Combine with AdaLN
        mixed = self.adanorm1(x + out_time + out_feature, cond)
        out = self.adanorm2(mixed + self.ffn(mixed), cond)
        return out


# --- Full DDPM using diffusers embeddings ---
class FactorizedDDPM(nn.Module):
    def __init__(
        self,
        num_stocks=81,
        time_steps=22,
        hidden_dim=64,
        cond_dim=128,
        num_layers=3,
        num_heads=4,
        dilations=(1, 2, 4),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stocks = num_stocks
        self.time_steps = time_steps

        # --- Input projections ---
        self.data_proj = nn.Linear(1, hidden_dim)
        self.stock_embedding = nn.Embedding(num_stocks, hidden_dim)
        self.temporal_pos_emb = nn.Embedding(time_steps, hidden_dim)

        # --- Diffusion timestep conditioning (from diffusers) ---
        # Timesteps: sinusoidal positional encoding
        # TimestepEmbedding: 2-layer MLP with SiLU activation
        self.t_sinusoidal = Timesteps(cond_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.t_embed = TimestepEmbedding(in_channels=cond_dim, time_embed_dim=cond_dim)

        # --- Core blocks ---
        self.blocks = nn.ModuleList([
            FactorizedBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                dilations=dilations,
            )
            for _ in range(num_layers)
        ])

        # --- Output ---
        self.final_norm = AdaLayerNorm(hidden_dim, cond_dim)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, noisy_data, t, stock_ids, dropout_mask):
        """
        noisy_data:   [B, 81, 22]
        t:            [B] or [B, 1]
        stock_ids:    [81]
        dropout_mask: [B, 81]
        """
        B, N, T = noisy_data.shape

        # --- Conditioning (diffusers handles sin/cos + MLP) ---
        t_sin = self.t_sinusoidal(t.view(-1))  # [B, cond_dim]
        cond = self.t_embed(t_sin)              # [B, cond_dim]

        # --- Input assembly ---
        x = self.data_proj(noisy_data.unsqueeze(-1))  # [B, N, T, C]
        s_emb = self.stock_embedding(stock_ids).view(1, N, 1, self.hidden_dim)
        t_pos = self.temporal_pos_emb(
            torch.arange(T, device=x.device)
        ).view(1, 1, T, self.hidden_dim)
        x = x + s_emb + t_pos

        # --- Process ---
        for block in self.blocks:
            x = block(x, cond, dropout_mask)

        # --- Output ---
        x = self.final_norm(x, cond)
        return self.out_proj(x).squeeze(-1)  # [B, N, T]


# --- Quick sanity check ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FactorizedDDPM(
        num_stocks=81,
        time_steps=22,
        hidden_dim=64,
        cond_dim=128,
        num_layers=3,
    ).to(device)

    B = 4
    noisy_data = torch.randn(B, 81, 22, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    stock_ids = torch.arange(81, device=device)
    dropout_mask = torch.ones(B, 81, device=device)
    dropout_mask[:, torch.randperm(81)[:16]] = 0

    out = model(noisy_data, t, stock_ids, dropout_mask)
    print(f"Output shape: {out.shape}")  # Expected: [4, 81, 22]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    graph = draw_graph(
        model,
        input_data=(noisy_data, t, stock_ids, dropout_mask),
        expand_nested=False,  # show internals of sub-modules
        depth=1,  # how deep to expand
        save_graph=True,
        filename="factorized_ddpm",
        directory=".",
    )


