"""
TimesNet adapted for the Time-LLM repository.

Place this file at:
    models/TimesNet.py

It intentionally reuses the layers that already exist in Time-LLM:
    - layers.Embed.DataEmbedding
    - layers.Conv_Blocks.Inception_Block_V1

No extra TimesNet layer files are required.

Compatibility goals:
    - run_main.py forward signature:
      model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    - long_term_forecast / short_term_forecast
    - accelerate + fp16 friendly FFT path
    - works even though the current Time-LLM parser does not define
      --top_k or --num_kernels (defaults: 5 and 6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def _dominant_periods(x: torch.Tensor, k: int):
    """Return dominant periods and sample-wise spectral weights.

    Args:
        x: [B, T, D]
        k: number of dominant frequencies

    FFT is computed in float32 for better compatibility with fp16 training,
    especially when T is not a power of two.
    """
    B, T, D = x.shape

    # [B, F, D] -- force float32 for robust CUDA FFT under mixed precision.
    spectrum = torch.fft.rfft(x.float(), dim=1)
    amplitude = spectrum.abs()

    # Global frequency importance: [F]
    global_amp = amplitude.mean(dim=0).mean(dim=-1)
    global_amp = global_amp.clone()
    global_amp[0] = 0.0  # ignore DC component

    # Excluding DC, there are F-1 usable frequencies.
    max_k = max(1, global_amp.numel() - 1)
    k = max(1, min(int(k), max_k))

    top_idx = torch.topk(global_amp, k=k, dim=0).indices  # [k]

    # Frequency-bin index f corresponds approximately to period T // f.
    periods = torch.div(T, top_idx, rounding_mode='floor').clamp_min(1)

    # Per-sample weights for the selected frequencies: [B, k]
    sample_weight = amplitude.mean(dim=-1).index_select(dim=1, index=top_idx)

    return periods, sample_weight


class TimesBlock(nn.Module):
    """TimesNet temporal 2D-variation block."""

    def __init__(self, configs):
        super().__init__()

        # The original Time-LLM parser does not have these two arguments.
        # Defaults follow the Time-Series-Library defaults.
        self.top_k = int(getattr(configs, 'top_k', 5))
        self.num_kernels = int(getattr(configs, 'num_kernels', 6))

        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model,
                configs.d_ff,
                num_kernels=self.num_kernels,
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff,
                configs.d_model,
                num_kernels=self.num_kernels,
            ),
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        periods, period_weight = _dominant_periods(x, self.top_k)

        outputs = []
        actual_k = periods.numel()

        for i in range(actual_k):
            period = int(periods[i].item())

            # Pad T to a multiple of the current period.
            if T % period != 0:
                padded_len = ((T // period) + 1) * period
                pad = x.new_zeros(B, padded_len - T, D)
                y = torch.cat([x, pad], dim=1)
            else:
                padded_len = T
                y = x

            # [B, padded_len, D]
            # -> [B, D, padded_len // period, period]
            y = y.reshape(B, padded_len // period, period, D)
            y = y.permute(0, 3, 1, 2).contiguous()

            # 2D variation modeling.
            y = self.conv(y)

            # Back to [B, padded_len, D], then remove padding.
            y = y.permute(0, 2, 3, 1).contiguous()
            y = y.reshape(B, padded_len, D)
            outputs.append(y[:, :T, :])

        # [B, T, D, k]
        stacked = torch.stack(outputs, dim=-1)

        # Softmax in float32 for numerical stability, then restore model dtype.
        weight = F.softmax(period_weight.float(), dim=1).to(dtype=stacked.dtype)
        weight = weight[:, None, None, :]  # [B, 1, 1, k]

        y = torch.sum(stacked * weight, dim=-1)

        # Residual connection.
        return y + x


class Model(nn.Module):
    """TimesNet model adapted to Time-LLM's model API."""

    def __init__(self, configs):
        super().__init__()

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        self.num_layers = configs.e_layers

        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        self.blocks = nn.ModuleList(
            [TimesBlock(configs) for _ in range(self.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            # Extend temporal length from seq_len to seq_len + pred_len.
            self.predict_linear = nn.Linear(
                self.seq_len,
                self.seq_len + self.pred_len,
            )
            self.projection = nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True,
            )
        else:
            # This adaptation is primarily intended for forecasting in Time-LLM.
            # Keeping a projection here makes anomaly/imputation paths possible.
            self.projection = nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True,
            )

    def _normalization_stats_for_output(self, means, stdev):
        """Match normalization statistics to c_out.

        For the user's M setting, enc_in == c_out and the full vectors are used.
        A c_out=1 fallback uses the last input variable, matching common MS usage.
        """
        if self.c_out == self.enc_in:
            return means, stdev
        if self.c_out == 1:
            return means[..., -1:], stdev[..., -1:]

        raise ValueError(
            f'TimesNet de-normalization expects c_out == enc_in or c_out == 1, '
            f'but got enc_in={self.enc_in}, c_out={self.c_out}.'
        )

    def forecast(self, x_enc, x_mark_enc=None):
        # x_enc: [B, seq_len, enc_in]

        # Instance-wise normalization used by TimesNet forecasting.
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = centered / stdev

        # Value + positional + temporal embedding: [B, seq_len, d_model]
        x = self.enc_embedding(x, x_mark_enc)

        # Extend the temporal axis for forecasting.
        x = self.predict_linear(x.transpose(1, 2)).transpose(1, 2)
        # [B, seq_len + pred_len, d_model]

        # Stacked TimesBlocks.
        for block in self.blocks:
            x = self.layer_norm(block(x))

        # Back to data space.
        out = self.projection(x)

        # De-normalize.
        means_out, stdev_out = self._normalization_stats_for_output(means, stdev)
        out = out * stdev_out + means_out

        # Time-LLM training loop needs only the prediction horizon.
        return out[:, -self.pred_len:, :]

    def imputation(self, x_enc, x_mark_enc=None):
        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = centered / stdev

        x = self.enc_embedding(x, x_mark_enc)
        for block in self.blocks:
            x = self.layer_norm(block(x))
        out = self.projection(x)

        means_out, stdev_out = self._normalization_stats_for_output(means, stdev)
        return out * stdev_out + means_out

    def anomaly_detection(self, x_enc):
        return self.imputation(x_enc, None)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_dec and x_mark_dec are accepted to match run_main.py,
        # but TimesNet forecasting does not require a decoder input.
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc)

        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc)

        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)

        raise NotImplementedError(
            f'This Time-LLM adaptation currently targets forecasting; '
            f'unsupported task_name={self.task_name!r}.'
        )

