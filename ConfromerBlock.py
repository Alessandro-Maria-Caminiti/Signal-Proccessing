import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout)
        )
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        x = x + self.conv(x.transpose(1, 2)).transpose(1, 2)  # Conv1D expects (B, C, T)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1, max_len=1000):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self._generate_positional_encoding(embed_dim, max_len=max_len)

        self.conformer_blocks = nn.Sequential(
            *[ConformerBlock(embed_dim, num_heads, ff_mult=ff_dim // embed_dim, dropout=dropout)
              for _ in range(num_layers)]
        )

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src):
        # Handle input shape (B, C, T, F) → (B, T, C*F)
        if src.dim() == 4:
            batch_size, channels, seq_len, features = src.shape
            src = src.view(batch_size, seq_len, channels * features)

        # (B, T, F) input
        src = self.input_projection(src)
        src = src + self.positional_encoding[:, :src.size(1), :].to(src.device)

        x = self.conformer_blocks(src)
        return self.fc_out(x)  # Output shape: (B, T, output_dim)

    def _generate_positional_encoding(self, embed_dim, max_len):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, embed_dim)
