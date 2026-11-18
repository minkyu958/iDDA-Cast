"""
Time-LLM compatible FPT / GPT4TS-style baseline.

Clean compatibility implementation based on the architecture described in
"One Fits All: Power General Time Series Analysis by Pretrained LM".

Main ideas:
- normalize each sample/channel
- patch the time series
- project numeric patches into GPT-2 hidden space
- use a pretrained GPT-2 backbone
- freeze most GPT-2 parameters
- predict the future horizon from flattened hidden states

Input:
    x_enc: [B, seq_len, C]
Output:
    y:     [B, pred_len, C]
"""

import torch
import torch.nn as nn

from transformers import GPT2Model


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = getattr(
            configs,
            "patch_size",
            getattr(configs, "patch_len", 16),
        )
        self.stride = getattr(configs, "stride", 8)

        self.gpt_layers = getattr(
            configs,
            "gpt_layers",
            getattr(configs, "llm_layers", 6),
        )
        self.gpt2_path = getattr(configs, "gpt2_path", "gpt2")
        self.freeze_gpt = bool(getattr(configs, "freeze_gpt", 1))

        self.pad = nn.ReplicationPad1d((0, self.stride))
        padded_len = self.seq_len + self.stride
        self.patch_num = (
            padded_len - self.patch_len
        ) // self.stride + 1

        self.gpt2 = GPT2Model.from_pretrained(
            self.gpt2_path,
            output_attentions=False,
            output_hidden_states=False,
        )

        if self.gpt_layers < len(self.gpt2.h):
            self.gpt2.h = self.gpt2.h[: self.gpt_layers]

        hidden = self.gpt2.config.n_embd

        self.in_layer = nn.Linear(
            self.patch_len,
            hidden,
        )

        self.out_layer = nn.Linear(
            hidden * self.patch_num,
            self.pred_len,
        )

        if self.freeze_gpt:
            for name, param in self.gpt2.named_parameters():
                # Keep layer norms and positional embeddings trainable,
                # while freezing the main pretrained blocks.
                if ("ln" in name) or ("wpe" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        x = x_enc - means

        stdev = torch.sqrt(
            torch.var(
                x,
                dim=1,
                keepdim=True,
                unbiased=False,
            ) + 1e-5
        ).detach()

        x = x / stdev

        bsz, _, n_vars = x.shape

        # [B, C, L]
        x = x.permute(0, 2, 1)
        x = self.pad(x)

        # [B, C, N, P]
        x = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )

        # [B*C, N, P]
        x = x.contiguous().view(
            bsz * n_vars,
            self.patch_num,
            self.patch_len,
        )

        x = self.in_layer(x)

        x = self.gpt2(
            inputs_embeds=x
        ).last_hidden_state

        x = x.reshape(
            bsz * n_vars,
            -1,
        )

        y = self.out_layer(x)

        # [B, pred_len, C]
        y = y.view(
            bsz,
            n_vars,
            self.pred_len,
        ).permute(0, 2, 1)

        y = y * stdev[:, :1, :] + means[:, :1, :]
        return y
