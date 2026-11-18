# iDDA-Cast ECA V3 (Paper): fixed Fourier seasonal decomposition and full Flatten Head.
# Preserves the uploaded single-LLM ECA path and iterative ISF (raw feedback).
# ECA, PFD, SRE/SPE and ISF retained. Fourier fit and all-patch Flatten are fixed.
# Intended for research ablation; performance needs real-data validation.

from math import sqrt, pi
from contextlib import nullcontext
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    AutoConfig, AutoModel, AutoTokenizer
)

from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)         # (B, n_vars, nf)
        x = self.linear(x)          # (B, n_vars, pred_len)
        x = self.dropout(x)
        return x



class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        target_embedding: (B, L, d_model)
        source_embedding: (S, d_llm)
        value_embedding : (S, d_llm)
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        """
        target_embedding: (B, L, H, E)
        source_embedding: (S, H, E)
        value_embedding : (S, H, E)
        """
        B, L, H, E = target_embedding.shape
        scale = 1.0 / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


class Model(nn.Module):
    """
    Prophet-inspired Time-LLM

    기존:
        X = Trend + Seasonal + Residual

    변경:
        X = Trend + Seasonality + Holiday/Event + Residual

    구성:
    - RevIN normalization
    - Prophet-style decomposition
        1) Trend: changepoint 기반 piecewise linear trend
        2) Seasonality: Fourier series 기반 periodic pattern
        3) Holiday/Event: 이상치 기반 event effect
        4) Residual: 나머지 오차
    - 각 성분별 PatchEmbedding
    - ReprogrammingLayer를 통해 LLM token space로 매핑
    - LLM 입력:
        [prompt | trend_tokens | seasonal_tokens | holiday_tokens | residual_tokens]
    - LLM 출력의 마지막 patch tokens만 forecasting head에 사용
    - RevIN denormalization
    - Iterative autoregressive forecasting (default: 96 steps per iteration)
    - Denormalized prediction is fed back to the raw input history.
      Each iteration recalculates RevIN statistics and decomposition.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        self.d_ff = int(configs.d_ff)
        self.top_k = int(getattr(configs, "top_k", 5))

        # No new argparse flags required: settings are read from environment.
        # 'off' allows strict same-code baseline/ablation with original head/PFD.
        dname = (str(getattr(configs, 'data', '')) + ' ' +
                 str(getattr(configs, 'data_path', ''))).lower()
        self.dataset_kind = ('exchange' if 'exchange' in dname else
                             'wind' if 'wind' in dname else 'other')
        selected_profile = os.getenv('IDDA_V3_PROFILE', 'auto').strip().lower()
        if selected_profile == 'auto':
            selected_profile = self.dataset_kind
        if selected_profile in ('none', '0', 'disabled'):
            selected_profile = 'off'
        if selected_profile not in ('off', 'exchange', 'wind', 'other'):
            raise ValueError('Invalid IDDA_V3_PROFILE: ' + selected_profile)
        self.v3_profile = selected_profile
        self.v3_active = selected_profile in ('exchange', 'wind')
        self.v3_anchor = self.v3_active and os.getenv('IDDA_V3_ANCHOR', '1') != '0'
        self.v3_llm_chunk = int(os.getenv('IDDA_V3_LLM_CHUNK', '8'))
        if self.v3_llm_chunk < 1:
            raise ValueError('IDDA_V3_LLM_CHUNK must be >= 1')
        self.v3_trend_mix = float(os.getenv(
            'IDDA_V3_TREND_MIX', '0.85' if selected_profile == 'exchange' else '0.8'))
        self.v3_checkpoint = self.v3_active and os.getenv('IDDA_V3_CHECKPOINT', '1') != '0'

        # Iterative AR: each call to the existing LLM predicts this many steps.
        # pred_len is the final horizon, ar_len is the step size per iteration.
        self.ar_len = int(getattr(configs, "ar_len", 96))
        if self.ar_len <= 0:
            raise ValueError(f"ar_len must be positive, got {self.ar_len}")
        self.ar_detach_history = getattr(configs, "ar_detach_history", True)
        if isinstance(self.ar_detach_history, str):
            self.ar_detach_history = self.ar_detach_history.lower() in ("1", "true", "yes", "on")
        else:
            self.ar_detach_history = bool(self.ar_detach_history)

        # llm_dim이 문자열로 들어오는 경우를 대비
        self.d_llm = int(configs.llm_dim)

        # -------------------------
        # 0) Prophet-style decomposition 설정
        # -------------------------
        self.period = getattr(configs, "period", None)

        if self.period is None:
            self.period = getattr(configs, "default_period", 24)

        self.period = int(self.period)

        # Trend: piecewise linear trend
        self.n_changepoints = int(
            getattr(configs, "n_changepoints", getattr(configs, "prophet_n_changepoints", 10))
        )
        self.changepoint_range = float(
            getattr(configs, "changepoint_range", getattr(configs, "prophet_changepoint_range", 0.8))
        )
        self.changepoint_prior_scale = float(
            getattr(configs, "changepoint_prior_scale", getattr(configs, "prophet_changepoint_prior_scale", 0.05))
        )

        # Seasonality: Fourier series
        periods_cfg = getattr(configs, "prophet_seasonal_periods", None)

        if periods_cfg is None:
            self.seasonal_periods = [float(self.period)]
        elif isinstance(periods_cfg, str):
            self.seasonal_periods = [
                float(p.strip()) for p in periods_cfg.split(",") if p.strip()
            ]
        elif isinstance(periods_cfg, (list, tuple)):
            self.seasonal_periods = [float(p) for p in periods_cfg]
        else:
            self.seasonal_periods = [float(periods_cfg)]

        if len(self.seasonal_periods) == 0:
            self.seasonal_periods = [float(self.period)]

        self.fourier_order = int(
            getattr(configs, "fourier_order", getattr(configs, "prophet_fourier_order", 5))
        )
        # Paper-consistent: keep the configured Fourier order unchanged (default 5).
        self.seasonality_prior_scale = float(
            getattr(configs, "seasonality_prior_scale", getattr(configs, "prophet_seasonality_prior_scale", 10.0))
        )

        # 기본값은 additive 권장
        # RevIN 이후 multiplicative는 불안정할 수 있음
        self.seasonality_mode = str(getattr(configs, "seasonality_mode", "additive")).lower()

        # Holiday/Event
        # 실제 holiday flag가 없으면 이상치 기반 event component로 사용
        self.holiday_prior_scale = float(
            getattr(configs, "holiday_prior_scale", getattr(configs, "prophet_holiday_prior_scale", 1.0))
        )
        self.holiday_threshold = float(
            getattr(configs, "holiday_threshold", 3.0)
        )
        self.holiday_lower_window = int(
            getattr(configs, "holiday_lower_window", 0)
        )
        self.holiday_upper_window = int(
            getattr(configs, "holiday_upper_window", 0)
        )

        holiday_idx = int(getattr(configs, "holiday_feature_index", -1))
        self.holiday_feature_index = holiday_idx if holiday_idx >= 0 else None

        # -------------------------
        # 1) Patch 설정
        # -------------------------
        self.trend_patch_len = getattr(configs, "trend_patch_len", configs.patch_len)
        self.trend_stride = getattr(configs, "trend_stride", configs.stride)

        # seasonal patch는 대표 주기의 절반 정도로 설정
        main_period = int(round(self.seasonal_periods[0]))
        seasonal_half = max(2, (main_period + 1) // 2)
        seasonal_stride_default = max(1, seasonal_half // 2)

        self.seasonal_patch_len = getattr(configs, "seasonal_patch_len", seasonal_half)
        self.seasonal_stride = getattr(configs, "seasonal_stride", seasonal_stride_default)

        self.holiday_patch_len = getattr(configs, "holiday_patch_len", configs.patch_len)
        self.holiday_stride = getattr(configs, "holiday_stride", configs.stride)

        self.residual_patch_len = getattr(configs, "residual_patch_len", configs.patch_len)
        self.residual_stride = getattr(configs, "residual_stride", configs.stride)
        if self.v3_active:
            self._v3_configure_patches(configs)
        print('[iDDA ECA Paper] profile=%s head=Flatten(all patches) FourierOrder=%d anchor=%s chunk=%d' % (
            self.v3_profile, self.fourier_order, self.v3_anchor, self.v3_llm_chunk))
        print('[iDDA ECA V3] patches: T=%s/%s S=%s/%s E=%s/%s R=%s/%s' % (
            self.trend_patch_len, self.trend_stride,
            self.seasonal_patch_len, self.seasonal_stride,
            self.holiday_patch_len, self.holiday_stride,
            self.residual_patch_len, self.residual_stride))

        # -------------------------
        # 2) LLM 로드
        # -------------------------
        llm_model_name = str(configs.llm_model).upper()

        if llm_model_name == 'LLAMA':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = False
            self.llama_config.output_hidden_states = False

            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local LLaMA model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )

            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local LLaMA tokenizer files not found. Attempting to download...")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif llm_model_name == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')
            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = False
            self.gpt2_config.output_hidden_states = False

            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:
                print("Local GPT2 model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local GPT2 tokenizer files not found. Attempting to download...")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif llm_model_name == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = False
            self.bert_config.output_hidden_states = False

            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:
                print("Local BERT model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local BERT tokenizer files not found. Attempting to download...")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )

        elif llm_model_name == 'QWEN':
            qwen_name = getattr(configs, "qwen_model_name", "Qwen/Qwen2.5-0.5B")

            self.qwen_config = AutoConfig.from_pretrained(
                qwen_name,
                trust_remote_code=True
            )
            self.qwen_config.num_hidden_layers = configs.llm_layers
            self.qwen_config.output_attentions = False
            self.qwen_config.output_hidden_states = False

            try:
                self.llm_model = AutoModel.from_pretrained(
                    qwen_name,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.qwen_config,
                )
            except EnvironmentError:
                print("Local Qwen model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    qwen_name,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.qwen_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen_name,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local Qwen tokenizer files not found. Attempting to download...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    qwen_name,
                    trust_remote_code=True,
                    local_files_only=False
                )

        else:
            raise Exception(f'LLM model is not defined: {configs.llm_model}')

        # pad token 설정
        # GPT2, LLaMA, Qwen 계열은 pad_token이 없을 수 있음
        added_pad_token = False

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.tokenizer.pad_token = '[PAD]'
                added_pad_token = True

        if added_pad_token:
            self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # LLM freeze
        for p in self.llm_model.parameters():
            p.requires_grad = False
        if hasattr(self.llm_model.config, 'use_cache'):
            self.llm_model.config.use_cache = False
        self.v3_frozen_eval = self.v3_active and os.getenv('IDDA_V3_FROZEN_EVAL', '1') != '0'
        if self.v3_frozen_eval:
            # eval() disables dropout but does NOT block gradient flow to SRE.
            self.llm_model.eval()

        # -------------------------
        # 3) 데이터셋별 Prompt 설명문 자동 선택
        # -------------------------
        self.description = self._resolve_dataset_description(configs)
        print(f'[Dataset Prompt] {self.description}')

        self.dropout = nn.Dropout(configs.dropout)

        # -------------------------
        # 4) Component별 PatchEmbedding
        # -------------------------
        self.trend_patch_embedding = PatchEmbedding(
            configs.d_model,
            self.trend_patch_len,
            self.trend_stride,
            configs.dropout
        )

        self.seasonal_patch_embedding = PatchEmbedding(
            configs.d_model,
            self.seasonal_patch_len,
            self.seasonal_stride,
            configs.dropout
        )

        self.holiday_patch_embedding = PatchEmbedding(
            configs.d_model,
            self.holiday_patch_len,
            self.holiday_stride,
            configs.dropout
        )

        self.residual_patch_embedding = PatchEmbedding(
            configs.d_model,
            self.residual_patch_len,
            self.residual_stride,
            configs.dropout
        )

        # -------------------------
        # 5) LLM vocab embedding mapping
        # -------------------------
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]

        # 기존 Time-LLM 방식 유지
        # Qwen은 vocab_size가 커서 메모리가 크면 num_tokens를 줄이는 것도 가능
        self.num_tokens = int(getattr(configs, "num_tokens", 1000))
        # Run scripts explicitly use --num_tokens 128 for the compact profile.
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model,
            configs.n_heads,
            self.d_ff,
            self.d_llm
        )

        # -------------------------
        # 6) Patch 개수 계산
        # -------------------------
        trend_patch_nums = max(
            1,
            int((configs.seq_len - self.trend_patch_len) / self.trend_stride + 2)
        )
        seasonal_patch_nums = max(
            1,
            int((configs.seq_len - self.seasonal_patch_len) / self.seasonal_stride + 2)
        )
        holiday_patch_nums = max(
            1,
            int((configs.seq_len - self.holiday_patch_len) / self.holiday_stride + 2)
        )
        residual_patch_nums = max(
            1,
            int((configs.seq_len - self.residual_patch_len) / self.residual_stride + 2)
        )

        self.patch_nums = (
            trend_patch_nums
            + seasonal_patch_nums
            + holiday_patch_nums
            + residual_patch_nums
        )

        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            raise NotImplementedError

        # Paper-consistent full-token Flatten -> Linear head.
        # Input: (B, N, d_ff, all_component_patch_tokens)
        # Output: (B, N, ar_len). No pooling, no token dropping.
        self.output_projection = FlattenHead(
            configs.enc_in, self.head_nf, self.ar_len,
            head_dropout=configs.dropout)
        # Retain V3's trainable projection of the full frozen-LLM hidden state.
        # Identity-like initialization reproduces the original first d_ff channels
        # at initialization while allowing remaining features to be learned.
        self.llm_output_projection = nn.Linear(self.d_llm, self.d_ff)
        with torch.no_grad():
            self.llm_output_projection.weight.zero_()
            self.llm_output_projection.bias.zero_()
            self.llm_output_projection.weight[:, :self.d_ff].copy_(
                torch.eye(self.d_ff))
        print('[iDDA ECA V3] head trainable parameters: %s' %
              sum(p.numel() for p in self.output_projection.parameters() if p.requires_grad))

        # RevIN
        self.revin = Normalize(configs.enc_in, affine=False)

    @staticmethod
    def _cli_provided(flag):
        """Look up explicitly passed flags, not argparse defaults."""
        forms = ('--' + flag, '--' + flag.replace('_', '-'))
        return any(x == form or x.startswith(form + '=')
                   for form in forms for x in sys.argv[1:])

    def _v3_configure_patches(self, configs):
        """Profile presets, while preserving explicit existing CLI patch flags."""
        if self.v3_profile == 'exchange':
            ter = (20, 10)
            seasonal = (10, 5)
        else:
            ter = (36, 12)
            seasonal = (72, 24)

        def override(name, default, alias=None):
            if self._cli_provided(name):
                return int(getattr(configs, name))
            if alias and self._cli_provided(alias):
                return int(getattr(configs, alias))
            return int(default)

        ter_len = override('patch_len', ter[0])
        ter_stride = override('stride', ter[1])
        self.trend_patch_len = override('trend_patch_len', ter_len)
        self.trend_stride = override('trend_stride', ter_stride)
        self.holiday_patch_len = override('holiday_patch_len', ter_len, 'event_patch_len')
        self.holiday_stride = override('holiday_stride', ter_stride, 'event_stride')
        self.residual_patch_len = override('residual_patch_len', ter_len)
        self.residual_stride = override('residual_stride', ter_stride)
        self.seasonal_patch_len = override('seasonal_patch_len', seasonal[0])
        self.seasonal_stride = override('seasonal_stride', seasonal[1])
        for p, st in ((self.trend_patch_len, self.trend_stride),
                      (self.seasonal_patch_len, self.seasonal_stride),
                      (self.holiday_patch_len, self.holiday_stride),
                      (self.residual_patch_len, self.residual_stride)):
            if p < 1 or st < 1 or p > self.seq_len:
                raise ValueError('Invalid component patch profile: %s/%s' % (p, st))

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, 'v3_frozen_eval', False):
            self.llm_model.eval()
        return self

    @staticmethod
    def _moving_average(x, kernel_size):
        k = min(int(kernel_size), x.shape[1])
        if k % 2 == 0:
            k -= 1
        if k < 3:
            return x
        left = (k - 1) // 2
        y = F.avg_pool1d(
            F.pad(x.transpose(1, 2).float(), (left, left), mode='replicate'),
            kernel_size=k, stride=1).transpose(1, 2)
        return y.to(dtype=x.dtype)

    def _v3_anchor_forecast(self, x, trend, seasonal, period):
        """Data-scale-free initialization in CURRENT normalized (RevIN) space.

        Anchors only current observed input, never future target values.
        Residual forecast from SAME LLM head corrects this inductive prior.
        """
        B, T, N = x.shape
        horizon = self.ar_len
        t = torch.arange(1, horizon + 1, dtype=x.dtype, device=x.device).view(1, -1, 1)
        last = x[:, -1:, :]
        recent_trend = trend[:, -1:, :]
        if self.v3_profile == 'exchange':
            # Exchange is usually more reliably predicted by persistence than
            # extrapolating a weak weekly harmonic or a very old global trend.
            baseline = last.expand(-1, horizon, -1).clone()
            if period > 1 and T >= period:
                template = seasonal[:, -period:, :]
                sfuture = template[:, torch.arange(horizon, device=x.device) % period, :]
                baseline = baseline + 0.15 * (sfuture - seasonal[:, -1:, :])
            return baseline

        # Wind: daily (96-step) Fourier structure plus smoothed level.
        if period > 1 and T >= period:
            template = seasonal[:, -period:, :]
            sfuture = template[:, torch.arange(horizon, device=x.device) % period, :]
        else:
            sfuture = torch.zeros((B, horizon, N), device=x.device, dtype=x.dtype)
        seasonal_last = seasonal[:, -1:, :]
        # Past innovations dissipate rather than being extrapolated for 96 steps.
        innovation = last - recent_trend - seasonal_last
        decay = torch.exp(-t / 12.0)
        return recent_trend + sfuture + innovation * decay

    @staticmethod
    def _resolve_dataset_description(configs):
        """Build an in-code dataset prompt without requiring an external prompt file.

        Priority: an explicitly enabled custom content prompt, then a recognized
        dataset identifier (data, data_path, model_id, root_path), then a neutral
        fallback for unknown datasets.
        """
        custom_text = str(getattr(configs, 'content', '') or '').strip()
        if getattr(configs, 'prompt_domain', 0) and custom_text:
            return custom_text

        candidates = [
            str(getattr(configs, key, '') or '').strip().lower()
            for key in ('data', 'data_path', 'model_id', 'root_path')
        ]
        descriptions = (
            ('etth', 'The Electricity Transformer Temperature (ETT) dataset '
                     'contains hourly transformer temperature and load measurements.'),
            ('ettm', 'The Electricity Transformer Temperature (ETT) dataset '
                     'contains transformer temperature and load measurements '
                     'sampled every 15 minutes.'),
            ('exchange', 'The Exchange dataset contains daily exchange-rate '
                         'observations for multiple currencies, which can exhibit '
                         'trend shifts and irregular fluctuations.'),
            ('wind', 'The Wind dataset contains multivariate wind-related '
                     'observations sampled every 15 minutes, with time-varying '
                     'and irregular fluctuations.'),
            ('weather', 'The Weather dataset contains multivariate meteorological '
                        'observations, including atmospheric variables with '
                        'potentially different temporal and seasonal patterns.'),
            ('ili', 'The ILI dataset contains weekly observations of '
                    'influenza-like illness activity and related measures.'),
        )
        for candidate in candidates:
            for keyword, description in descriptions:
                if keyword in candidate:
                    return description

        return ('The input dataset contains multivariate time-series '
                'observations. Forecast future values from the observed '
                'historical temporal patterns.')

    # ============================================================
    # Prophet-style decomposition helpers
    # ============================================================
    def _fit_component_with_design(self, y, design, penalty):
        """
        Ridge least squares로 component를 추정한다.

        y      : (B, T, N)
        design : (T, D)
        penalty: (D,)

        return:
            fitted: (B, T, N)
        """
        B, T, N = y.shape
        device = y.device
        orig_dtype = y.dtype

        # mixed precision 상태에서도 decomposition fitting은 fp32로 수행
        if device.type == "cuda":
            amp_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            amp_ctx = nullcontext()

        with amp_ctx:
            y_work = y.float()
            X = design.to(device=device, dtype=torch.float32)
            penalty = penalty.to(device=device, dtype=torch.float32)

            # Y: (T, B*N)
            Y = y_work.permute(0, 2, 1).contiguous().reshape(B * N, T).transpose(0, 1)

            # beta = (X^T X + Lambda)^(-1) X^T Y
            XtX = X.transpose(0, 1).matmul(X)
            reg = torch.diag(penalty)
            Xty = X.transpose(0, 1).matmul(Y)

            beta = torch.linalg.solve(XtX + reg, Xty)  # (D, B*N)
            fitted = X.matmul(beta)                    # (T, B*N)

            fitted = fitted.transpose(0, 1).contiguous().reshape(B, N, T)
            fitted = fitted.permute(0, 2, 1).contiguous()

        return fitted.to(orig_dtype)

    def _piecewise_linear_trend(self, x):
        """
        Prophet의 g(t)에 해당하는 trend.

        g(t) = a + b*t + sum_j delta_j * max(0, t - changepoint_j)

        x: (B, T, N)
        """
        B, T, N = x.shape
        device = x.device

        t = torch.linspace(
            0.0,
            1.0,
            T,
            device=device,
            dtype=torch.float32
        )

        cols = [
            torch.ones_like(t),
            t
        ]

        n_cp = min(self.n_changepoints, max(0, T - 2))
        cp_range = min(max(self.changepoint_range, 0.0), 1.0)

        if n_cp > 0:
            changepoints = torch.linspace(
                0.0,
                cp_range,
                n_cp + 2,
                device=device,
                dtype=torch.float32
            )[1:-1]

            for cp in changepoints:
                cols.append(torch.relu(t - cp))

        design = torch.stack(cols, dim=1)  # (T, D)

        # intercept와 slope는 거의 규제하지 않음
        penalty = torch.full(
            (design.size(1),),
            1e-4,
            device=device,
            dtype=torch.float32
        )

        # changepoint_prior_scale이 클수록 trend 변화가 유연해짐
        if design.size(1) > 2:
            cp_penalty = 1.0 / (self.changepoint_prior_scale ** 2 + 1e-6)
            penalty[2:] = cp_penalty

        trend = self._fit_component_with_design(x, design, penalty)
        if self.v3_active:
            # Local trend adapts to shifts while global trend keeps continuity.
            window = 21 if self.v3_profile == 'exchange' else 97
            local = self._moving_average(x, window)
            mix = max(0.0, min(1.0, self.v3_trend_mix))
            trend = (1.0 - mix) * trend + mix * local
        return trend

    def _fourier_seasonality(self, x):
        """
        Prophet의 s(t)에 해당하는 seasonality.

        Fourier series:
            sin(2*pi*k*t/period), cos(2*pi*k*t/period)

        x: (B, T, N)
        """
        B, T, N = x.shape
        device = x.device

        t = torch.arange(T, device=device, dtype=torch.float32)
        cols = []

        for period in self.seasonal_periods:
            if period <= 1:
                continue

            for k in range(1, self.fourier_order + 1):
                angle = 2.0 * pi * k * t / float(period)
                cols.append(torch.sin(angle))
                cols.append(torch.cos(angle))

        if len(cols) == 0:
            return torch.zeros_like(x)

        design = torch.stack(cols, dim=1)

        # seasonality_prior_scale이 클수록 seasonality를 더 유연하게 반영
        season_penalty = 1.0 / (self.seasonality_prior_scale ** 2 + 1e-6)

        penalty = torch.full(
            (design.size(1),),
            season_penalty,
            device=device,
            dtype=torch.float32
        )

        # Fixed Fourier basis with ridge fitting; no data-dependent amplitude gate.
        seasonal = self._fit_component_with_design(x, design, penalty)
        return seasonal

    def _dilate_event_mask(self, mask, lower_window=0, upper_window=0):
        """
        Prophet의 lower_window / upper_window와 유사하게
        이벤트 영향 구간을 앞뒤로 확장한다.

        mask: (B, T, N)
        """
        lower_window = max(0, int(lower_window))
        upper_window = max(0, int(upper_window))

        if lower_window == 0 and upper_window == 0:
            return mask

        B, T, N = mask.shape
        base = mask
        out = torch.zeros_like(mask)

        for offset in range(-lower_window, upper_window + 1):
            if offset < 0:
                # 이벤트 이전 구간에 영향
                out[:, :T + offset, :] = torch.maximum(
                    out[:, :T + offset, :],
                    base[:, -offset:, :]
                )
            elif offset > 0:
                # 이벤트 이후 구간에 영향
                out[:, offset:, :] = torch.maximum(
                    out[:, offset:, :],
                    base[:, :T - offset, :]
                )
            else:
                out = torch.maximum(out, base)

        return out

    def _holiday_event_effect(self, remainder, x_mark_enc=None):
        """
        Prophet의 h(t)에 해당하는 holiday/event effect.

        실제 holiday flag가 있으면:
            x_mark_enc[:, :, holiday_feature_index] 사용

        실제 holiday flag가 없으면:
            remainder에서 큰 이상치를 event-like component로 추정

        remainder: (B, T, N)
        x_mark_enc: (B, T, D) or None
        """
        B, T, N = remainder.shape
        device = remainder.device
        dtype = remainder.dtype

        use_given_holiday_mark = False

        if (
            x_mark_enc is not None
            and self.holiday_feature_index is not None
            and self.holiday_feature_index < x_mark_enc.shape[-1]
        ):
            h_mark = x_mark_enc[:, :, self.holiday_feature_index].to(device=device)
            mask = (h_mark > 0.5).float().unsqueeze(-1).expand(-1, -1, N)
            use_given_holiday_mark = True

        else:
            # holiday 정보가 없으면 robust 이상치 기반으로 event 추정
            med = remainder.median(dim=1, keepdim=True).values
            mad = (remainder - med).abs().median(dim=1, keepdim=True).values
            robust_std = 1.4826 * mad + 1e-6

            mask = ((remainder - med).abs() > self.holiday_threshold * robust_std).float()

        mask = mask.to(device=device, dtype=dtype)

        mask = self._dilate_event_mask(
            mask,
            lower_window=self.holiday_lower_window,
            upper_window=self.holiday_upper_window
        )

        # holiday_prior_scale이 클수록 holiday/event component가 더 강하게 반영
        scale = self.holiday_prior_scale / (self.holiday_prior_scale + 1.0)

        if use_given_holiday_mark:
            # 명시적 holiday flag가 있는 경우: holiday 구간의 평균 효과 추정
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            effect = (remainder * mask).sum(dim=1, keepdim=True) / denom
            holiday = mask * effect * scale
        else:
            # 명시적 holiday flag가 없는 경우: 이상치성 remainder를 event component로 분리
            holiday = mask * remainder * scale

        return holiday

    def _decompose(self, x_enc, x_mark_enc=None):
        """
        Prophet-style additive decomposition.

        x_enc = trend + seasonality + holiday/event + residual

        return:
            trend    : (B, T, N)
            seasonal : (B, T, N)
            holiday  : (B, T, N)
            residual : (B, T, N)
            period   : int
        """
        # 1) Trend: piecewise linear changepoint trend
        trend = self._piecewise_linear_trend(x_enc)

        # 2) Seasonality: Fourier basis
        if self.seasonality_mode == "multiplicative":
            # 간단한 multiplicative 근사
            # RevIN 이후에는 additive가 더 안정적이므로 기본값은 additive 권장
            base = trend.abs().clamp_min(1e-3)
            seasonal_ratio = self._fourier_seasonality((x_enc - trend) / base)
            seasonal = seasonal_ratio * base
        else:
            seasonal = self._fourier_seasonality(x_enc - trend)

        # 3) Holiday/Event
        remainder = x_enc - trend - seasonal
        holiday = self._holiday_event_effect(remainder, x_mark_enc)

        # 4) Residual
        residual = x_enc - trend - seasonal - holiday

        main_period = int(round(self.seasonal_periods[0]))
        return trend, seasonal, holiday, residual, main_period

    # ============================================================
    # Forward / Forecast
    # ============================================================
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]

        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Iterative AR with denormalized feedback and a fixed seq_len history.

        Every iteration calls the unchanged *single-LLM early-concatenation*
        pathway in _forecast_segment():
            raw history -> RevIN -> PFD -> [prompt|T|S|E|R]
            -> one LLM -> one forecasting head -> RevIN denorm.
        The denormalized ar_len-step prediction updates the raw history.
        """
        if x_enc.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x_enc.size(1)}")

        history = x_enc
        mark_history = x_mark_enc
        future_marks = None
        if x_mark_dec is not None:
            if x_mark_dec.size(1) < self.pred_len:
                raise ValueError("x_mark_dec must contain at least pred_len future time marks")
            # Common dataset convention: [label_len past marks | pred_len future marks].
            future_marks = x_mark_dec[:, -self.pred_len:, :]

        if mark_history is not None and future_marks is not None:
            if mark_history.shape[0] != future_marks.shape[0] or mark_history.shape[-1] != future_marks.shape[-1]:
                raise ValueError("x_mark_enc and x_mark_dec must agree on batch and feature dimensions")

        forecasts = []
        for offset in range(0, self.pred_len, self.ar_len):
            # Each iteration recomputes its own RevIN statistics / PFD / prompt.
            # The original LLM token order and all of its modules are unchanged.
            segment = self._forecast_segment(history, mark_history, step_offset=offset)
            take = min(self.ar_len, self.pred_len - offset)
            current = segment[:, :take, :]
            forecasts.append(current)  # Keep each step's direct loss gradient.

            if offset + take >= self.pred_len:
                break

            # Feed back predictions *after* denorm. Detach only the feedback,
            # not the separately accumulated predictions / their direct loss.
            feedback = current.detach() if self.ar_detach_history else current
            history = torch.cat((history, feedback), dim=1)[:, -self.seq_len:, :]

            # Slide available time marks forward by the same number of steps.
            # Never use future observed target VALUES; only future calendar marks.
            if mark_history is not None:
                if future_marks is not None:
                    new_marks = future_marks[:, offset:offset + take, :]
                    mark_history = torch.cat((mark_history, new_marks), dim=1)[:, -self.seq_len:, :]
                else:
                    # No future marks: do not reuse old holidays/events as though
                    # they described predicted steps. PFD falls back to MAD events.
                    mark_history = None

        return torch.cat(forecasts, dim=1)  # (B, pred_len, N), raw scale

    def _forecast_segment(self, x_enc, x_mark_enc, step_offset=0):
        """Original early-concat architecture; now outputs ar_len raw-scale steps.

        [Prompt | Trend | Seasonal | Holiday/Event | Residual] -> ONE LLM
        """
        # 0) RevIN norm
        x_enc = self.revin(x_enc, 'norm')   # (B, T, N)

        # 1) Prophet-style decomposition
        trend, seasonal, holiday, residual, period = self._decompose(x_enc, x_mark_enc)

        B, T, N = x_enc.size()
        BN = B * N

        period_int = int(period)
        phase_now = (T - 1 + step_offset) % period_int if period_int > 0 else 0

        # (B, N, T) -> (B*N, T, 1)
        trend_flat = trend.permute(0, 2, 1).contiguous().reshape(BN, T, 1)
        season_flat = seasonal.permute(0, 2, 1).contiguous().reshape(BN, T, 1)
        holiday_flat = holiday.permute(0, 2, 1).contiguous().reshape(BN, T, 1)
        residual_flat = residual.permute(0, 2, 1).contiguous().reshape(BN, T, 1)

        # -------------------------
        # Trend statistics
        # -------------------------
        trend_last = trend_flat[:, -1, 0]

        if T >= 2:
            w = min(8, T - 1)
            d1 = trend_flat[:, 1:, 0] - trend_flat[:, :-1, 0]
            slope_recent = d1[:, -w:].mean(dim=1)

            if d1.size(1) >= 2 * w:
                slope_prev = d1[:, -(2 * w):-w].mean(dim=1)
            else:
                slope_prev = slope_recent

            slope_accel = slope_recent - slope_prev
        else:
            slope_recent = torch.zeros_like(trend_last)
            slope_accel = torch.zeros_like(trend_last)

        if T >= 3:
            w2 = min(8, T - 2)
            d2 = (
                trend_flat[:, 2:, 0]
                - 2 * trend_flat[:, 1:-1, 0]
                + trend_flat[:, :-2, 0]
            )
            curv_recent = d2[:, -w2:].mean(dim=1)
        else:
            curv_recent = torch.zeros_like(trend_last)

        # -------------------------
        # Seasonality statistics
        # -------------------------
        if period_int >= 2 and T >= period_int:
            seg = season_flat[:, -period_int:, 0]
        else:
            seg = season_flat[:, :, 0]

        amp_recent = 0.5 * (seg.max(dim=1).values - seg.min(dim=1).values)
        amp_long = 0.5 * (
            season_flat[:, :, 0].max(dim=1).values
            - season_flat[:, :, 0].min(dim=1).values
        )

        # -------------------------
        # Holiday/Event statistics
        # -------------------------
        h_abs = holiday_flat.abs()
        h_active_ratio = (h_abs > 1e-6).float().mean(dim=1)[:, 0]
        h_mean_effect = holiday_flat.mean(dim=1)[:, 0]
        h_max_abs_effect = h_abs.max(dim=1).values[:, 0]

        # -------------------------
        # Residual statistics
        # -------------------------
        r_min = residual_flat.min(dim=1).values[:, 0]
        r_max = residual_flat.max(dim=1).values[:, 0]
        r_med = residual_flat.median(dim=1).values[:, 0]

        r_mean = residual_flat.mean(dim=1, keepdim=True)
        r_std = residual_flat.std(dim=1, keepdim=True) + 1e-6

        spike_ratio = (
            torch.abs(residual_flat - r_mean) > 3.0 * r_std
        ).float().mean(dim=1)[:, 0]

        # residual lag
        lags = self.calcute_lags(residual_flat)  # (B*N, top_k)

        # -------------------------
        # Prompt 생성
        # -------------------------
        prompt = []

        for b in range(BN):
            trend_prefix = (
                f"[TREND] Focus on non-periodic baseline and changepoint-aware direction. "
                f"last={trend_last[b].item():.4f}, "
                f"slope={slope_recent[b].item():.6f}, "
                f"accel={slope_accel[b].item():.6f}, "
                f"curvature={curv_recent[b].item():.6f}. "
                f"Rule: continue the smooth baseline and avoid confusing trend with short-term oscillations.\n"
            )

            seasonal_prefix = (
                f"[SEASONALITY] Focus on Fourier-based periodic pattern. "
                f"main_period={period_int}, "
                f"phase_now={phase_now}, "
                f"amp_recent={amp_recent[b].item():.4f}, "
                f"amp_long={amp_long[b].item():.4f}, "
                f"fourier_order={self.fourier_order}. "
                f"Rule: preserve phase-aligned peaks and troughs.\n"
            )

            holiday_prefix = (
                f"[HOLIDAY_EVENT] Focus on irregular event or holiday-like effects. "
                f"active_ratio={h_active_ratio[b].item():.4f}, "
                f"mean_effect={h_mean_effect[b].item():.4f}, "
                f"max_abs_effect={h_max_abs_effect[b].item():.4f}. "
                f"Rule: capture abrupt event-driven deviations separately from trend and seasonality.\n"
            )

            residual_prefix = (
                f"[RESIDUAL] Focus on remaining noise and unexplained fluctuations. "
                f"min={r_min[b].item():.4f}, "
                f"max={r_max[b].item():.4f}, "
                f"median={r_med[b].item():.4f}, "
                f"spike_ratio={spike_ratio[b].item():.4f}, "
                f"top_lags={lags[b].tolist()}. "
                f"Rule: be robust and do not overfit isolated residual spikes.\n"
            )

            prompt_body = (
                f"Dataset: {self.description} "
                f"Task: forecast the next {self.ar_len} steps given the previous {self.seq_len} steps. "
                f"Total forecast horizon = {self.pred_len}; autoregressive offset = {step_offset}. "
                f"Decomposition: X = Trend + Seasonality + Holiday/Event + Residual. "
                f"Seasonality mode = {self.seasonality_mode}. "
                f"Seasonal periods = {self.seasonal_periods}. "
            )

            prompt_ = (
                f"<|start_prompt|>"
                f"{trend_prefix}"
                f"{seasonal_prefix}"
                f"{holiday_prefix}"
                f"{residual_prefix}"
                f"{prompt_body}"
                f"<|end_prompt|>"
            )

            prompt.append(prompt_)

        # prompt -> token embeddings
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 if self.v3_active else 2048,
        )
        prompt_ids = encoded.input_ids.to(x_enc.device)
        prompt_mask = encoded.attention_mask.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)

        # -------------------------
        # Source embeddings
        # -------------------------
        # word_embeddings: (vocab, d_llm)
        # mapping_layer input: (d_llm, vocab)
        # output: (d_llm, num_tokens)
        # final: (num_tokens, d_llm)
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)

        # -------------------------
        # Component별 PatchEmbedding
        # -------------------------
        trend_in = trend.permute(0, 2, 1).contiguous()
        seasonal_in = seasonal.permute(0, 2, 1).contiguous()
        holiday_in = holiday.permute(0, 2, 1).contiguous()
        residual_in = residual.permute(0, 2, 1).contiguous()

        trend_tokens, n_vars = self.trend_patch_embedding(trend_in.float())
        seasonal_tokens, _ = self.seasonal_patch_embedding(seasonal_in.float())
        holiday_tokens, _ = self.holiday_patch_embedding(holiday_in.float())
        residual_tokens, _ = self.residual_patch_embedding(residual_in.float())

        # patch token -> LLM token space
        trend_tokens = self.reprogramming_layer(
            trend_tokens,
            source_embeddings,
            source_embeddings
        )

        seasonal_tokens = self.reprogramming_layer(
            seasonal_tokens,
            source_embeddings,
            source_embeddings
        )

        holiday_tokens = self.reprogramming_layer(
            holiday_tokens,
            source_embeddings,
            source_embeddings
        )

        residual_tokens = self.reprogramming_layer(
            residual_tokens,
            source_embeddings,
            source_embeddings
        )

        # -------------------------
        # Token Fusion: exactly one ECA sequence per (sample, variable)
        # [SPE prompt | SRE Trend | SRE Seasonal | SRE Event | SRE Residual]
        # -------------------------
        llm_in = torch.cat(
            [prompt_embeddings, trend_tokens, seasonal_tokens,
             holiday_tokens, residual_tokens], dim=1)
        patch_mask = torch.ones(
            (llm_in.shape[0], self.patch_nums),
            dtype=prompt_mask.dtype, device=llm_in.device)
        attention_mask = torch.cat([prompt_mask, patch_mask], dim=1)
        max_pos = getattr(self.llm_model.config, 'max_position_embeddings', None)
        if max_pos is None:
            max_pos = getattr(self.llm_model.config, 'n_positions', None)
        if max_pos is not None and llm_in.shape[1] > int(max_pos):
            raise ValueError(
                'LLM tokens (%d) exceed context limit (%d). Reduce patch count '
                'or shorten prompts.' % (llm_in.shape[1], int(max_pos)))

        # One frozen LLM with chunked batch execution to reduce peak VRAM;
        # gradients still flow through the complete trainable SRE pathway.
        if self.v3_active:
            outs = []
            for lo in range(0, llm_in.shape[0], self.v3_llm_chunk):
                hi = lo + self.v3_llm_chunk
                block_inputs = llm_in[lo:hi]
                block_mask = attention_mask[lo:hi]
                # Non-reentrant checkpoint works even with a frozen, eval-mode
                # transformer: the trainable reprogrammed embeddings keep grads.
                if self.training and torch.is_grad_enabled() and self.v3_checkpoint:
                    def frozen_llm_forward(emb, mask):
                        return self.llm_model(
                            inputs_embeds=emb, attention_mask=mask).last_hidden_state
                    block_out = checkpoint(
                        frozen_llm_forward, block_inputs, block_mask,
                        use_reentrant=False)
                else:
                    block_out = self.llm_model(
                        inputs_embeds=block_inputs,
                        attention_mask=block_mask).last_hidden_state
                outs.append(block_out)
            llm_out = torch.cat(outs, dim=0)
        else:
            llm_out = self.llm_model(
                inputs_embeds=llm_in, attention_mask=attention_mask).last_hidden_state

        if self.llm_output_projection is None:
            llm_out = llm_out[:, :, :self.d_ff]
        else:
            llm_out = self.llm_output_projection(llm_out)

        # (B*N, tokens, d_ff) -> (B, N, tokens, d_ff)
        llm_out = torch.reshape(
            llm_out,
            (-1, n_vars, llm_out.shape[-2], llm_out.shape[-1])
        )

        # (B, N, tokens, d_ff) -> (B, N, d_ff, tokens)
        llm_out = llm_out.permute(0, 1, 3, 2).contiguous()

        # 마지막 patch tokens만 사용
        llm_out = llm_out[:, :, :, -self.patch_nums:]

        # head: (B, N, d_ff, patch_nums) -> (B, N, ar_len)
        dec_out = self.output_projection(llm_out)

        # The prediction head learns a *correction* to a per-domain structural
        # anchor. There is still a single forecasting head after ONE ECA LLM.
        if self.v3_anchor:
            anchor = self._v3_anchor_forecast(x_enc, trend, seasonal, period)
            dec_out = dec_out + anchor.permute(0, 2, 1).contiguous()

        # (B, N, ar_len) -> (B, ar_len, N)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # RevIN denorm
        dec_out = self.revin(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        """
        x_enc: (B*, T, 1)

        return:
            lags: (B*, top_k)
        """
        q_fft = torch.fft.rfft(
            x_enc.permute(0, 2, 1).contiguous(),
            dim=-1
        )
        k_fft = torch.fft.rfft(
            x_enc.permute(0, 2, 1).contiguous(),
            dim=-1
        )

        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        mean_value = torch.mean(corr, dim=1)  # (B*, T)

        k = min(self.top_k, mean_value.shape[-1])
        _, lags = torch.topk(mean_value, k, dim=-1)

        # T가 너무 짧아서 top_k보다 작은 경우 padding
        if k < self.top_k:
            pad = torch.zeros(
                lags.shape[0],
                self.top_k - k,
                device=lags.device,
                dtype=lags.dtype
            )
            lags = torch.cat([lags, pad], dim=-1)

        return lags
