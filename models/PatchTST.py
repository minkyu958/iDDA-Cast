"""
Time-LLM compatible PatchTST-style baseline.

This is a clean compatibility implementation, not a verbatim copy of the
official PatchTST source. It preserves the main ideas:
- instance normalization
- channel-independent patching
- shared Transformer encoder across channels
- flatten prediction head

Input:
    x_enc: [B, seq_len, C]
Output:
    y:     [B, pred_len, C]
"""

import math
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.patch_len = getattr(configs, "patch_len", 16)
        self.stride = getattr(configs, "stride", 8)

        d_model = getattr(configs, "d_model", 128)
        n_heads = getattr(configs, "n_heads", 8)
        e_layers = getattr(configs, "e_layers", 3)
        d_ff = getattr(configs, "d_ff", 256)
        dropout = getattr(configs, "dropout", 0.1)

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model({d_model}) must be divisible by n_heads({n_heads})."
            )

        # Replication padding by one stride, matching the commonly used
        # PatchTST long-term forecasting setup.
        self.pad = nn.ReplicationPad1d((0, self.stride))

        padded_len = self.seq_len + self.stride
        self.patch_num = (padded_len - self.patch_len) // self.stride + 1
        if self.patch_num <= 0:
            raise ValueError("patch_len is larger than the available sequence.")

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_num, d_model)
        )
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model),
        )

        head_dropout = getattr(configs, "head_dropout", 0.0)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(head_dropout),
            nn.Linear(self.patch_num * d_model, self.pred_len),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [B, L, C]
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = x / stdev

        bsz, _, n_vars = x.shape

        # [B, C, L]
        x = x.permute(0, 2, 1)
        x = self.pad(x)

        # [B, C, patch_num, patch_len]
        x = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )

        # channel independence: [B*C, patch_num, patch_len]
        x = x.contiguous().view(
            bsz * n_vars,
            self.patch_num,
            self.patch_len,
        )

        x = self.patch_proj(x)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.encoder(x)

        # [B*C, pred_len]
        y = self.head(x)

        # [B, pred_len, C]
        y = y.view(bsz, n_vars, self.pred_len).permute(0, 2, 1)

        y = y * stdev[:, :1, :] + means[:, :1, :]
        return y
