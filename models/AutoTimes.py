"""
Time-LLM compatible AutoTimes-style GPT-2 baseline.

IMPORTANT
---------
AutoTimes' official training/evaluation pipeline is autoregressive and differs
from the standard Time-LLM run_main.py loop. This file is a compatibility
implementation for unified experiments. For paper-faithful reproduction,
use the official AutoTimes repository.

Core ideas preserved here:
- non-overlapping time-series segments as tokens
- linear tokenizer: token_len -> GPT hidden dimension
- frozen pretrained GPT-2
- linear detokenizer: GPT hidden dimension -> token_len
- autoregressive segment forecasting

Input:
    x_enc: [B, seq_len, C]
Output:
    y:     [B, pred_len, C]
"""

import math
import torch
import torch.nn as nn

from transformers import GPT2Model


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.token_len = getattr(configs, "token_len", 96)
        self.gpt2_path = getattr(configs, "gpt2_path", "gpt2")

        self.gpt2 = GPT2Model.from_pretrained(
            self.gpt2_path,
            output_attentions=False,
            output_hidden_states=False,
        )

        # AutoTimes uses a frozen pretrained LLM.
        for p in self.gpt2.parameters():
            p.requires_grad = False

        hidden = self.gpt2.config.n_embd

        self.encoder = nn.Linear(
            self.token_len,
            hidden,
        )
        self.decoder = nn.Linear(
            hidden,
            self.token_len,
        )

    def _next_segment(self, x):
        """
        x: [B, L, C]
        returns one predicted segment: [B, token_len, C]
        """
        means = x.mean(dim=1, keepdim=True).detach()
        z = x - means
        stdev = torch.sqrt(
            torch.var(
                z,
                dim=1,
                keepdim=True,
                unbiased=False,
            ) + 1e-5
        ).detach()
        z = z / stdev

        bsz, length, n_vars = z.shape

        # Use the most recent complete non-overlapping segments.
        usable = (length // self.token_len) * self.token_len
        if usable < self.token_len:
            pad_len = self.token_len - length
            left = z[:, :1, :].repeat(1, pad_len, 1)
            z = torch.cat([left, z], dim=1)
            usable = self.token_len

        z = z[:, -usable:, :]

        # [B, C, L] -> [B*C, L]
        z = z.permute(0, 2, 1).contiguous()
        z = z.view(bsz * n_vars, usable)

        # [B*C, token_num, token_len]
        tokens = z.unfold(
            dimension=-1,
            size=self.token_len,
            step=self.token_len,
        )

        embeds = self.encoder(tokens)

        hidden = self.gpt2(
            inputs_embeds=embeds
        ).last_hidden_state

        # The last hidden token predicts the next numeric segment.
        next_norm = self.decoder(
            hidden[:, -1, :]
        )

        # [B*C, token_len] -> [B, token_len, C]
        next_norm = next_norm.view(
            bsz,
            n_vars,
            self.token_len,
        ).permute(0, 2, 1)

        next_seg = (
            next_norm * stdev[:, :1, :]
            + means[:, :1, :]
        )

        return next_seg

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        history = x_enc
        preds = []

        steps = math.ceil(
            self.pred_len / self.token_len
        )

        for _ in range(steps):
            next_seg = self._next_segment(history)
            preds.append(next_seg)

            # Sliding autoregressive context.
            if history.size(1) > self.token_len:
                history = torch.cat(
                    [
                        history[:, self.token_len:, :],
                        next_seg,
                    ],
                    dim=1,
                )
            else:
                history = next_seg

        y = torch.cat(preds, dim=1)
        return y[:, : self.pred_len, :]
