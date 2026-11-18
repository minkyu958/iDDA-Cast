# Exchange-optimized iDDA-Cast patch variant.
# Based on the user-supplied Exchange patch sweep:
#   Trend / Event / Residual: 20/10
#   Seasonal: 8/4, 10/5 (selected default), 18/9
# All other architectural, optimization, PFD and ISF code is retained.
# Auto Exchange defaults can be overridden with existing CLI patch flags.

from math import sqrt, pi
from contextlib import nullcontext
import sys
import torch
import torch.nn as nn
from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    AutoConfig, AutoModel, AutoTokenizer,
)
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers
transformers.logging.set_verbosity_error()

# Forecasting head: patch features -> prediction length
class FlattenHead(nn.Module):
    """
    (B, N, d_ff, patch_num) -> (B, N, pred_len)
    """

    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# Patch-to-LLM embedding reprogramming
class ReprogrammingLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        target_embedding: (B*, L, d_model)
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
        target_embedding: (B*, L, H, E)
        source_embedding: (S, H, E)
        value_embedding : (S, H, E)
        """
        _, _, _, E = target_embedding.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum('blhe,she->bhls', target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum('bhls,she->blhe', A, value_embedding)
        return reprogramming_embedding

# DDAI-Cast: PFD + SRE/SPE + shared frozen LLM + raw-input AR feedback
class Model(nn.Module):
    """
    DDAI-Cast: Input-level iterative autoregression with per-step RevIN/PFD.

    Main idea
    ---------
    1) RevIN normalization

    2) Parameter-free / Prophet-inspired decomposition

           X_norm = Trend + Seasonality + Event + Residual

    3) SRE:
       each component is patched and
       reprogrammed independently.

    4) SPE:
       each component receives its own
       structural prompt.

    5) Structural Parallel Processing

       Trend
       -> [Trend Prompt | Trend Tokens]
       -> Frozen LLM
       -> Trend Output Projection

       Seasonality
       -> [Seasonality Prompt | Seasonal Tokens]
       -> Frozen LLM
       -> Seasonal Output Projection

       Event
       -> [Event Prompt | Event Tokens]
       -> Frozen LLM
       -> Event Output Projection

       Residual
       -> [Residual Prompt | Residual Tokens]
       -> Frozen LLM
       -> Residual Output Projection

       The four prompt-token pairs are
       processed as independent branches.

       For memory efficiency,
       the four branches share ONE frozen
       LLM weight set and are forwarded
       sequentially:

           Trend
           -> Seasonality
           -> Event
           -> Residual

       This preserves branch independence
       without packing all four branches
       into one large batch.

    6) Input-level iterative autoregression (ar_len=96 by default)

           raw history -> RevIN -> PFD -> SRE/SPE -> Frozen LLM
           -> component forecasts -> sum in normalized space
           -> RevIN denorm with CURRENT history statistics
           -> slide RAW input history -> repeat

       RevIN and PFD are recomputed at EVERY AR iteration.
       All four component branches use the same current-history
       RevIN statistics. No separate component histories are kept.

    7) Concatenate original-scale forecast segments to pred_len.

    Important
    ---------
    - Component-specific prompts are
      concatenated with only their own
      component tokens.

    - PatchEmbedding,
      ReprogrammingLayer,
      OutputProjection,
      Forecast Head
      are component-specific.

    - OutputProjection maps the full
      LLM hidden representation into
      the same d_ff feature space.

    OOM prevention
    --------------
    - num_tokens default:
          1000 -> 128

    - Frozen LLM gradient checkpointing:
          enabled by default

    - Component-wise sequential LLM
      processing is preserved.

    - AR history detach is preserved.
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = int(configs.pred_len)
        self.seq_len = int(configs.seq_len)
        self.d_ff = int(configs.d_ff)
        self.top_k = int(getattr(configs, 'top_k', 5))
        self.ar_len = int(getattr(configs, 'ar_len', 96))
        if self.ar_len <= 0:
            raise ValueError(f'ar_len must be positive, got {self.ar_len}')
        self.ar_detach_history = bool(getattr(configs, 'ar_detach_history', True))
        self.period = getattr(configs, 'period', None)
        if self.period is None:
            self.period = getattr(configs, 'default_period', 24)
        self.period = int(self.period)
        self.n_changepoints = int(getattr(configs, 'n_changepoints', getattr(configs, 'prophet_n_changepoints', 10)))
        self.changepoint_range = float(getattr(configs, 'changepoint_range', getattr(configs,
            'prophet_changepoint_range', 0.8)))
        self.changepoint_prior_scale = float(getattr(configs, 'changepoint_prior_scale', getattr(configs,
            'prophet_changepoint_prior_scale', 0.05)))
        periods_cfg = getattr(configs, 'prophet_seasonal_periods', None)
        if periods_cfg is None:
            self.seasonal_periods = [float(self.period)]
        elif isinstance(periods_cfg, str):
            self.seasonal_periods = [float(p.strip()) for p in periods_cfg.split(',') if p.strip()]
        elif isinstance(periods_cfg, (list, tuple)):
            self.seasonal_periods = [float(p) for p in periods_cfg]
        else:
            self.seasonal_periods = [float(periods_cfg)]
        if len(self.seasonal_periods) == 0:
            self.seasonal_periods = [float(self.period)]
        self.fourier_order = int(getattr(configs, 'fourier_order', getattr(configs, 'prophet_fourier_order', 5)))
        self.seasonality_prior_scale = float(getattr(configs, 'seasonality_prior_scale', getattr(configs,
            'prophet_seasonality_prior_scale', 10.0)))
        self.seasonality_mode = str(getattr(configs, 'seasonality_mode', 'additive')).lower()
        self.event_prior_scale = float(getattr(configs, 'event_prior_scale', getattr(configs, 'holiday_prior_scale',
            getattr(configs, 'prophet_holiday_prior_scale', 1.0))))
        self.event_threshold = float(getattr(configs, 'event_threshold', getattr(configs, 'holiday_threshold', 3.0)))
        self.event_lower_window = int(getattr(configs, 'event_lower_window', getattr(configs, 'holiday_lower_window', 0)))
        self.event_upper_window = int(getattr(configs, 'event_upper_window', getattr(configs, 'holiday_upper_window', 0)))
        event_idx = int(getattr(configs, 'event_feature_index', getattr(configs, 'holiday_feature_index', -1)))
        self.event_feature_index = event_idx if event_idx >= 0 else None
        # Exchange-specific patch defaults derived from the supplied experiments.
        # Trend/Event/Residual = 20/10, Seasonal = 10/5 by default.
        # Explicit command-line patch flags always take precedence. Other datasets
        # retain their existing behavior and their existing parser defaults.
        self._configure_component_patches(configs)
        self.llm_model, self.tokenizer = self._load_llm(configs)
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
        self.d_llm = int(self.llm_model.get_input_embeddings().weight.shape[1])
        for p in self.llm_model.parameters():
            p.requires_grad = False
        self.use_llm_checkpoint = bool(getattr(configs, 'use_llm_checkpoint', True))
        if self.use_llm_checkpoint:
            if hasattr(self.llm_model, 'gradient_checkpointing_enable'):
                self.llm_model.gradient_checkpointing_enable()
            if hasattr(self.llm_model.config, 'use_cache'):
                self.llm_model.config.use_cache = False
        # Automatically select the correct domain description for each dataset.
        # A non-empty --prompt_domain/--content can still override it deliberately.
        self.description = self._resolve_dataset_description(configs)
        print(f'[Dataset Prompt] {self.description}')
        self.dropout = nn.Dropout(configs.dropout)
        self.trend_patch_embedding = PatchEmbedding(configs.d_model, self.trend_patch_len, self.trend_stride,
            configs.dropout)
        self.seasonal_patch_embedding = PatchEmbedding(configs.d_model, self.seasonal_patch_len, self.seasonal_stride,
            configs.dropout)
        self.event_patch_embedding = PatchEmbedding(configs.d_model, self.event_patch_len, self.event_stride,
            configs.dropout)
        self.residual_patch_embedding = PatchEmbedding(configs.d_model, self.residual_patch_len, self.residual_stride,
            configs.dropout)
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = int(getattr(configs, 'num_tokens', 128))
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.trend_reprogramming = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm,
            configs.dropout)
        self.seasonal_reprogramming = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm,
            configs.dropout)
        self.event_reprogramming = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm,
            configs.dropout)
        self.residual_reprogramming = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm,
            configs.dropout)
        self.trend_patch_nums = self._calc_patch_num(self.seq_len, self.trend_patch_len, self.trend_stride)
        self.seasonal_patch_nums = self._calc_patch_num(self.seq_len, self.seasonal_patch_len, self.seasonal_stride)
        self.event_patch_nums = self._calc_patch_num(self.seq_len, self.event_patch_len, self.event_stride)
        self.residual_patch_nums = self._calc_patch_num(self.seq_len, self.residual_patch_len, self.residual_stride)
        if self.task_name not in ['long_term_forecast', 'short_term_forecast']:
            raise NotImplementedError
        self.trend_output_projection = self._make_output_projection()
        self.seasonal_output_projection = self._make_output_projection()
        self.event_output_projection = self._make_output_projection()
        self.residual_output_projection = self._make_output_projection()
        self.trend_head = FlattenHead(configs.enc_in, self.d_ff * self.trend_patch_nums, self.ar_len,
            head_dropout=configs.dropout)
        self.seasonal_head = FlattenHead(configs.enc_in, self.d_ff * self.seasonal_patch_nums, self.ar_len,
            head_dropout=configs.dropout)
        self.event_head = FlattenHead(configs.enc_in, self.d_ff * self.event_patch_nums, self.ar_len,
            head_dropout=configs.dropout)
        self.residual_head = FlattenHead(configs.enc_in, self.d_ff * self.residual_patch_nums, self.ar_len,
            head_dropout=configs.dropout)
        # Number of trainable parameters in component forecast heads; these
        # depend on the patch profile, so print them for reproducible ablations.
        print('[Patch Configuration] Forecast-head trainable parameters: '
              + ', '.join(
                  f'{name}={sum(p.numel() for p in module.parameters() if p.requires_grad):,}'
                  for name, module in (
                      ('Trend', self.trend_head), ('Seasonal', self.seasonal_head),
                      ('Event', self.event_head), ('Residual', self.residual_head)
                  )
              ))
        self.revin = Normalize(configs.enc_in, affine=False)

    @staticmethod
    def _cli_provided(name):
        """True if a CLI value was explicitly supplied (not merely an argparse default).

        Recognizes both --option value and --option=value. This lets Exchange
        use 20/10 + 10/5 when run_main.py declares older 16/8 defaults.
        """
        flag = '--' + name.replace('_', '-')
        flag_underscore = '--' + name
        return any(
            item == flag or item.startswith(flag + '=') or
            item == flag_underscore or item.startswith(flag_underscore + '=')
            for item in sys.argv[1:]
        )

    def _configure_component_patches(self, configs):
        """Resolve all four component patch/stride pairs *before* creating heads.

        Dataset Exchange (when corresponding flags are not explicitly set):
          Trend = Event = Residual: patch 20, stride 10
          Seasonal:                  patch 10, stride 5

        To reproduce the sweep, pass --seasonal_patch_len 8/10/18 and
        --seasonal_stride 4/5/9. All component-specific CLI overrides remain
        available; no new run_main.py argparse options are necessary.

        Non-Exchange datasets use exactly the original resolution policy.
        """
        dataset = str(getattr(configs, 'data', '') or '').strip().lower()
        data_path = str(getattr(configs, 'data_path', '') or '').lower()
        self.exchange_patch_profile = (dataset == 'exchange' or 'exchange_rate' in data_path)

        def get_num(name, default):
            value = getattr(configs, name, None)
            return int(default if value is None else value)

        if self.exchange_patch_profile:
            # The user's Exchange table holds T/E/R constant at 20/10.
            # General --patch_len and --stride can intentionally override
            # the common T/E/R pair; component flags can override individually.
            common_len = (get_num('patch_len', 20)
                          if self._cli_provided('patch_len') else 20)
            common_stride = (get_num('stride', 10)
                             if self._cli_provided('stride') else 10)

            def choose(name, fallback, alias=None):
                if self._cli_provided(name):
                    return get_num(name, fallback)
                if alias is not None and self._cli_provided(alias):
                    return get_num(alias, fallback)
                return int(fallback)

            self.trend_patch_len = choose('trend_patch_len', common_len)
            self.trend_stride = choose('trend_stride', common_stride)
            self.event_patch_len = choose('event_patch_len', common_len, 'holiday_patch_len')
            self.event_stride = choose('event_stride', common_stride, 'holiday_stride')
            self.residual_patch_len = choose('residual_patch_len', common_len)
            self.residual_stride = choose('residual_stride', common_stride)
            self.seasonal_patch_len = choose('seasonal_patch_len', 10)
            self.seasonal_stride = choose('seasonal_stride', 5)
        else:
            # Original behavior for ETT/Weather/Wind/ILI and any other dataset.
            default_len = get_num('patch_len', 16)
            default_stride = get_num('stride', 8)
            main_period = int(round(self.seasonal_periods[0]))
            seasonal_half = max(2, (main_period + 1) // 2)
            seasonal_stride_default = max(1, seasonal_half // 2)
            self.trend_patch_len = get_num('trend_patch_len', default_len)
            self.trend_stride = get_num('trend_stride', default_stride)
            self.seasonal_patch_len = get_num('seasonal_patch_len', seasonal_half)
            self.seasonal_stride = get_num('seasonal_stride', seasonal_stride_default)
            self.event_patch_len = get_num('event_patch_len',
                                           get_num('holiday_patch_len', default_len))
            self.event_stride = get_num('event_stride',
                                        get_num('holiday_stride', default_stride))
            self.residual_patch_len = get_num('residual_patch_len', default_len)
            self.residual_stride = get_num('residual_stride', default_stride)

        components = {
            'Trend': (self.trend_patch_len, self.trend_stride),
            'Seasonal': (self.seasonal_patch_len, self.seasonal_stride),
            'Event': (self.event_patch_len, self.event_stride),
            'Residual': (self.residual_patch_len, self.residual_stride),
        }
        for name, (patch, stride) in components.items():
            if patch < 1 or stride < 1 or patch > self.seq_len:
                raise ValueError(
                    f'{name}: invalid patch/stride ({patch}/{stride}) for '
                    f'seq_len={self.seq_len}; require 1 <= patch <= seq_len, stride >= 1.')
        preset = 'EXCHANGE 20/10 (TER), 10/5 (S)' if self.exchange_patch_profile else 'ORIGINAL'
        print(f'[Patch Configuration] Dataset={dataset or "unknown"}, profile={preset}, seq_len={self.seq_len}')
        for name, (patch, stride) in components.items():
            print(f'[Patch Configuration] {name:8s}: length={patch:3d}, stride={stride:3d}, '
                  f'patches={self._calc_patch_num(self.seq_len, patch, stride)}')

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

    def _make_output_projection(self):
        """
        Trainable d_llm -> d_ff projection.

        Initial behavior:

            output =
                input[..., :d_ff]

        Example:

            d_llm = 768
            d_ff  = 128

        Initially:
            hidden 1~128
            are passed unchanged.

            hidden 129~768
            have zero contribution.

        During training:
            all projection weights
            remain trainable.
        """
        if self.d_ff > self.d_llm:
            raise ValueError(f'd_ff ({self.d_ff}) must be <= d_llm ({self.d_llm}) for identity-like initialization.')
        layer = nn.Linear(self.d_llm, self.d_ff, bias=True)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
            eye = torch.eye(self.d_ff, dtype=layer.weight.dtype, device=layer.weight.device)
            layer.weight[:, :self.d_ff].copy_(eye)
        return layer

    def _load_llm(self, configs):
        llm_model_name = str(configs.llm_model).upper()
        if llm_model_name == 'LLAMA':
            config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            config.num_hidden_layers = configs.llm_layers
            config.output_attentions = False
            config.output_hidden_states = False
            if hasattr(config, 'use_cache'):
                config.use_cache = False
            try:
                model = LlamaModel.from_pretrained('huggyllama/llama-7b', trust_remote_code=True,
                    local_files_only=True, config=config)
            except EnvironmentError:
                print('Local LLaMA model files not found. Attempting to download...')
                model = LlamaModel.from_pretrained('huggyllama/llama-7b', trust_remote_code=True,
                    local_files_only=False, config=config)
            try:
                tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b', trust_remote_code=True,
                    local_files_only=True)
            except EnvironmentError:
                print('Local LLaMA tokenizer files not found. Attempting to download...')
                tokenizer = LlamaTokenizer.from_pretrained('huggyllama/llama-7b', trust_remote_code=True,
                    local_files_only=False)
        elif llm_model_name == 'GPT2':
            config = GPT2Config.from_pretrained('openai-community/gpt2')
            config.num_hidden_layers = configs.llm_layers
            config.output_attentions = False
            config.output_hidden_states = False
            if hasattr(config, 'use_cache'):
                config.use_cache = False
            try:
                model = GPT2Model.from_pretrained('openai-community/gpt2', trust_remote_code=True,
                    local_files_only=True, config=config)
            except EnvironmentError:
                print('Local GPT2 model files not found. Attempting to download...')
                model = GPT2Model.from_pretrained('openai-community/gpt2', trust_remote_code=True,
                    local_files_only=False, config=config)
            try:
                tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', trust_remote_code=True,
                    local_files_only=True)
            except EnvironmentError:
                print('Local GPT2 tokenizer files not found. Attempting to download...')
                tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', trust_remote_code=True,
                    local_files_only=False)
        elif llm_model_name == 'BERT':
            config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            config.num_hidden_layers = configs.llm_layers
            config.output_attentions = False
            config.output_hidden_states = False
            if hasattr(config, 'use_cache'):
                config.use_cache = False
            try:
                model = BertModel.from_pretrained('google-bert/bert-base-uncased', trust_remote_code=True,
                    local_files_only=True, config=config)
            except EnvironmentError:
                print('Local BERT model files not found. Attempting to download...')
                model = BertModel.from_pretrained('google-bert/bert-base-uncased', trust_remote_code=True,
                    local_files_only=False, config=config)
            try:
                tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased', trust_remote_code=True,
                    local_files_only=True)
            except EnvironmentError:
                print('Local BERT tokenizer files not found. Attempting to download...')
                tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased', trust_remote_code=True,
                    local_files_only=False)
        elif llm_model_name == 'QWEN':
            qwen_name = getattr(configs, 'qwen_model_name', 'Qwen/Qwen2.5-0.5B')
            config = AutoConfig.from_pretrained(qwen_name, trust_remote_code=True)
            config.num_hidden_layers = configs.llm_layers
            config.output_attentions = False
            config.output_hidden_states = False
            if hasattr(config, 'use_cache'):
                config.use_cache = False
            try:
                model = AutoModel.from_pretrained(qwen_name, trust_remote_code=True, local_files_only=True, config=config)
            except EnvironmentError:
                print('Local Qwen model files not found. Attempting to download...')
                model = AutoModel.from_pretrained(qwen_name, trust_remote_code=True, local_files_only=False, config=config)
            try:
                tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True, local_files_only=True)
            except EnvironmentError:
                print('Local Qwen tokenizer files not found. Attempting to download...')
                tokenizer = AutoTokenizer.from_pretrained(qwen_name, trust_remote_code=True, local_files_only=False)
        else:
            raise ValueError(f'LLM model is not defined: {configs.llm_model}')
        return (model, tokenizer)

    @staticmethod
    def _calc_patch_num(seq_len, patch_len, stride):
        return max(1, int((seq_len - patch_len) / stride + 2))

    def _source_embeddings(self):
        """
        Build shared reduced
        source-token basis.

        word_embeddings:
            (V, d_llm)

        transpose:
            (d_llm, V)

        mapping_layer:
            (d_llm, num_tokens)

        return:
            (num_tokens, d_llm)
        """
        return self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

    def _tokenize_prompts(self, prompts, device):
        """
        Tokenize component-specific
        prompts.

        Returns
        -------
        prompt_embeddings:
            (B*, Pmax, d_llm)

        attention_mask:
            (B*, Pmax)

        Final branch input:

            [
                valid component prompt
                |
                component reprogrammed tokens
            ]
        """
        encoded = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=2048)
        prompt_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids)
        return (prompt_embeddings, attention_mask)

    def _fit_component_with_design(self, y, design, penalty):
        """
        Ridge least-squares component fitting.

        y:
            (B, T, N)

        design:
            (T, D)

        penalty:
            (D,)

        return:
            fitted
            (B, T, N)
        """
        B, T, N = y.shape
        device = y.device
        orig_dtype = y.dtype
        if device.type == 'cuda':
            amp_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            amp_ctx = nullcontext()
        with amp_ctx:
            y_work = y.float()
            X = design.to(device=device, dtype=torch.float32)
            penalty = penalty.to(device=device, dtype=torch.float32)
            Y = y_work.permute(0, 2, 1).contiguous().reshape(B * N, T).transpose(0, 1)
            XtX = X.transpose(0, 1).matmul(X)
            reg = torch.diag(penalty)
            Xty = X.transpose(0, 1).matmul(Y)
            beta = torch.linalg.solve(XtX + reg, Xty)
            fitted = X.matmul(beta)
            fitted = fitted.transpose(0, 1).contiguous().reshape(B, N, T)
            fitted = fitted.permute(0, 2, 1).contiguous()
        return fitted.to(orig_dtype)

    def _piecewise_linear_trend(self, x):
        """
        Prophet-inspired piecewise
        linear trend:

            g(t) =
                a
                + b*t
                + sum_j delta_j
                  * max(0, t-cp_j)

        x:
            (B, T, N)
        """
        _, T, _ = x.shape
        device = x.device
        t = torch.linspace(0.0, 1.0, T, device=device, dtype=torch.float32)
        cols = [torch.ones_like(t), t]
        n_cp = min(self.n_changepoints, max(0, T - 2))
        cp_range = min(max(self.changepoint_range, 0.0), 1.0)
        if n_cp > 0:
            changepoints = torch.linspace(0.0, cp_range, n_cp + 2, device=device, dtype=torch.float32)[1:-1]
            for cp in changepoints:
                cols.append(torch.relu(t - cp))
        design = torch.stack(cols, dim=1)
        penalty = torch.full((design.size(1),), 0.0001, device=device, dtype=torch.float32)
        if design.size(1) > 2:
            cp_penalty = 1.0 / (self.changepoint_prior_scale ** 2 + 1e-06)
            penalty[2:] = cp_penalty
        return self._fit_component_with_design(x, design, penalty)

    def _fourier_seasonality(self, x):
        """
        Fourier-series seasonality.

        x:
            (B, T, N)
        """
        _, T, _ = x.shape
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
        season_penalty = 1.0 / (self.seasonality_prior_scale ** 2 + 1e-06)
        penalty = torch.full((design.size(1),), season_penalty, device=device, dtype=torch.float32)
        return self._fit_component_with_design(x, design, penalty)

    def _dilate_event_mask(self, mask, lower_window=0, upper_window=0):
        lower_window = max(0, int(lower_window))
        upper_window = max(0, int(upper_window))
        if lower_window == 0 and upper_window == 0:
            return mask
        _, T, _ = mask.shape
        base = mask
        out = torch.zeros_like(mask)
        for offset in range(-lower_window, upper_window + 1):
            if offset < 0:
                out[:, :T + offset, :] = torch.maximum(out[:, :T + offset, :], base[:, -offset:, :])
            elif offset > 0:
                out[:, offset:, :] = torch.maximum(out[:, offset:, :], base[:, :T - offset, :])
            else:
                out = torch.maximum(out, base)
        return out

    def _event_effect(self, remainder, x_mark_enc=None):
        """
        Event component.

        If event/holiday flag exists
        in x_mark_enc:
            use it.

        Otherwise:
            Median-MAD event detection.

        remainder:
            (B, T, N)

        x_mark_enc:
            (B, T, D)
            or None
        """
        _, _, N = remainder.shape
        device = remainder.device
        dtype = remainder.dtype
        use_given_event_mark = False
        if x_mark_enc is not None and self.event_feature_index is not None and (self.event_feature_index < x_mark_enc.shape[-1]):
            event_mark = x_mark_enc[:, :, self.event_feature_index].to(device=device)
            mask = (event_mark > 0.5).float().unsqueeze(-1).expand(-1, -1, N)
            use_given_event_mark = True
        else:
            med = remainder.median(dim=1, keepdim=True).values
            mad = (remainder - med).abs().median(dim=1, keepdim=True).values
            robust_std = 1.4826 * mad + 1e-06
            mask = ((remainder - med).abs() > self.event_threshold * robust_std).float()
        mask = mask.to(device=device, dtype=dtype)
        mask = self._dilate_event_mask(mask, lower_window=self.event_lower_window, upper_window=self.event_upper_window)
        scale = self.event_prior_scale / (self.event_prior_scale + 1.0)
        if use_given_event_mark:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            effect = (remainder * mask).sum(dim=1, keepdim=True) / denom
            event = mask * effect * scale
        else:
            event = mask * remainder * scale
        return event

    def _decompose(self, x_enc, x_mark_enc=None):
        """
        Additive decomposition:

            x_enc =
                trend
                + seasonal
                + event
                + residual
        """
        trend = self._piecewise_linear_trend(x_enc)
        if self.seasonality_mode == 'multiplicative':
            base = trend.abs().clamp_min(0.001)
            seasonal_ratio = self._fourier_seasonality((x_enc - trend) / base)
            seasonal = seasonal_ratio * base
        else:
            seasonal = self._fourier_seasonality(x_enc - trend)
        remainder = x_enc - trend - seasonal
        event = self._event_effect(remainder, x_mark_enc)
        residual = x_enc - trend - seasonal - event
        main_period = int(round(self.seasonal_periods[0]))
        return (trend, seasonal, event, residual, main_period)

    def _build_component_prompts(self, trend, seasonal, event, residual, period, forecast_horizon=None, step_offset=0):
        """
        Returns four prompt lists,
        each with B*N strings.
        """
        B, T, N = trend.shape
        BN = B * N
        if forecast_horizon is None:
            forecast_horizon = self.ar_len
        forecast_horizon = int(forecast_horizon)
        step_offset = int(step_offset)
        period_int = int(period)
        phase_now = (T - 1 + step_offset) % period_int if period_int > 0 else 0
        trend_flat = trend.permute(0, 2, 1).contiguous().reshape(BN, T, 1)
        seasonal_flat = seasonal.permute(0, 2, 1).contiguous().reshape(BN, T, 1)
        event_flat = event.permute(0, 2, 1).contiguous().reshape(BN, T, 1)
        residual_flat = residual.permute(0, 2, 1).contiguous().reshape(BN, T, 1)
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
            d2 = trend_flat[:, 2:, 0] - 2 * trend_flat[:, 1:-1, 0] + trend_flat[:, :-2, 0]
            curv_recent = d2[:, -w2:].mean(dim=1)
        else:
            curv_recent = torch.zeros_like(trend_last)
        if period_int >= 2 and T >= period_int:
            seasonal_seg = seasonal_flat[:, -period_int:, 0]
        else:
            seasonal_seg = seasonal_flat[:, :, 0]
        amp_recent = 0.5 * (seasonal_seg.max(dim=1).values - seasonal_seg.min(dim=1).values)
        amp_long = 0.5 * (seasonal_flat[:, :, 0].max(dim=1).values - seasonal_flat[:, :, 0].min(dim=1).values)
        event_abs = event_flat.abs()
        event_active_ratio = (event_abs > 1e-06).float().mean(dim=1)[:, 0]
        event_mean_effect = event_flat.mean(dim=1)[:, 0]
        event_max_abs_effect = event_abs.max(dim=1).values[:, 0]
        residual_min = residual_flat.min(dim=1).values[:, 0]
        residual_max = residual_flat.max(dim=1).values[:, 0]
        residual_med = residual_flat.median(dim=1).values[:, 0]
        residual_mean = residual_flat.mean(dim=1, keepdim=True)
        residual_std = residual_flat.std(dim=1, keepdim=True) + 1e-06
        spike_ratio = (torch.abs(residual_flat - residual_mean) > 3.0 * residual_std).float().mean(dim=1)[:, 0]
        lags = self.calcute_lags(residual_flat)
        trend_prompts = []
        seasonal_prompts = []
        event_prompts = []
        residual_prompts = []
        for b in range(BN):
            common = (
                f'Dataset: {self.description} Forecast segment horizon: {forecast_horizon}'
                f'. Total forecast horizon: {self.pred_len}. Autoregressive offset: {step_offset}. Input length: '
                f'{T}. '
            )
            trend_prompt = (
                f'<|start_prompt|>[TREND FORECASTING] {common}'
                f'You are forecasting only the trend component of a decomposed time series. last='
                f'{trend_last[b].item():.4f}, slope={slope_recent[b].item():.6f}, acceleration='
                f'{slope_accel[b].item():.6f}, curvature={curv_recent[b].item():.6f}'
                f'. Focus on the non-periodic baseline, long-term direction, and '
                f'changepoint-aware dynamics. Do not model seasonal oscillations, event effects, '
                f'or residual noise.<|end_prompt|>'
            )
            seasonal_prompt = (
                f'<|start_prompt|>[SEASONALITY FORECASTING] {common}'
                f'You are forecasting only the seasonality component of a decomposed time series. main_period='
                f'{period_int}, phase_now={phase_now}, amp_recent={amp_recent[b].item():.4f}, amp_long='
                f'{amp_long[b].item():.4f}, fourier_order={self.fourier_order}'
                f'. Preserve phase-aligned peaks and troughs and maintain the observed periodic '
                f'structure. Do not model long-term trend, irregular event effects, or residual '
                f'noise.<|end_prompt|>'
            )
            event_prompt = (
                f'<|start_prompt|>[EVENT FORECASTING] {common}'
                f'You are forecasting only the event component of a decomposed time series. active_ratio='
                f'{event_active_ratio[b].item():.4f}, mean_effect={event_mean_effect[b].item():.4f}'
                f', max_abs_effect={event_max_abs_effect[b].item():.4f}'
                f'. Focus on abrupt, irregular, event-driven deviations separated from trend and '
                f'seasonality. Do not treat normal seasonal peaks as event effects.<|end_prompt|>'
            )
            residual_prompt = (
                f'<|start_prompt|>[RESIDUAL FORECASTING] {common}'
                f'You are forecasting only the residual component of a decomposed time series. min='
                f'{residual_min[b].item():.4f}, max={residual_max[b].item():.4f}, median='
                f'{residual_med[b].item():.4f}, spike_ratio={spike_ratio[b].item():.4f}, top_lags='
                f'{lags[b].tolist()}'
                f'. Focus on remaining short-term unexplained dynamics and be robust to isolated '
                f'spikes. Do not reconstruct trend, seasonality, or event components.<|end_prompt|>'
            )
            trend_prompts.append(trend_prompt)
            seasonal_prompts.append(seasonal_prompt)
            event_prompts.append(event_prompt)
            residual_prompts.append(residual_prompt)
        return (trend_prompts, seasonal_prompts, event_prompts, residual_prompts)

    def _prepare_component_branch(self, component, prompts, patch_embedding, reprogramming_layer, source_embeddings):
        """
        Prepare ONE component-specific
        LLM input.

        Flow:

            component
               ↓
            PatchEmbedding
               ↓
            Reprogramming
               ↓
            PromptEmbedding
               ↓
            [
                valid prompt
                |
                component tokens
            ]

        The prepared branch is immediately
        sent through the shared frozen LLM
        before the next branch is prepared.
        """
        B, _, N = component.shape
        component_in = component.permute(0, 2, 1).contiguous()
        component_tokens, n_vars = patch_embedding(component_in.float())
        if N != n_vars:
            raise RuntimeError(f'PatchEmbedding returned n_vars={n_vars}, but input has N={N}.')
        component_tokens = reprogramming_layer(component_tokens, source_embeddings, source_embeddings)
        prompt_embeddings, prompt_mask = self._tokenize_prompts(prompts, component.device)
        if prompt_embeddings.size(0) != component_tokens.size(0):
            raise RuntimeError(
                f'Prompt batch and component-token batch do not match: '
                f'{prompt_embeddings.size(0)} vs {component_tokens.size(0)}'
            )
        BN = component_tokens.size(0)
        patch_num = component_tokens.size(1)
        prompt_lengths = prompt_mask.sum(dim=1).long()
        total_lengths = prompt_lengths + patch_num
        max_total_len = int(total_lengths.max().item())
        llm_in = component_tokens.new_zeros(BN, max_total_len, self.d_llm)
        llm_mask = torch.zeros(BN, max_total_len, device=component.device, dtype=prompt_mask.dtype)
        for i in range(BN):
            p_len = int(prompt_lengths[i].item())
            llm_in[i, :p_len, :] = prompt_embeddings[i, :p_len, :]
            llm_in[i, p_len:p_len + patch_num, :] = component_tokens[i]
            llm_mask[i, :p_len + patch_num] = 1
        return {'llm_in': llm_in, 'attention_mask': llm_mask, 'prompt_lengths': prompt_lengths,
            'patch_num': patch_num, 'n_vars': n_vars, 'batch_size': B}

    @staticmethod
    def _right_pad_branch(branch, target_len):
        """
        Right-pad prepared branch
        to shared sequence length.
        """
        x = branch['llm_in']
        mask = branch['attention_mask']
        if x.size(1) == target_len:
            return (x, mask)
        pad_len = target_len - x.size(1)
        x_pad = x.new_zeros(x.size(0), pad_len, x.size(2))
        mask_pad = mask.new_zeros(mask.size(0), pad_len)
        return (torch.cat([x, x_pad], dim=1), torch.cat([mask, mask_pad], dim=1))

    def _project_branch_output(self, llm_out, branch, output_projection, head):
        """
        Extract component patch-token
        hidden states.

        Full LLM hidden:
            d_llm

        Output Projection:
            d_llm -> d_ff

        Forecast Head:
            -> ar_len
        """
        BN, _, _ = llm_out.shape
        patch_num = int(branch['patch_num'])
        prompt_lengths = branch['prompt_lengths']
        n_vars = int(branch['n_vars'])
        B = int(branch['batch_size'])
        token_offsets = torch.arange(patch_num, device=llm_out.device, dtype=prompt_lengths.dtype).unsqueeze(0)
        token_indices = prompt_lengths.unsqueeze(1) + token_offsets
        gather_index = token_indices.unsqueeze(-1).expand(-1, -1, llm_out.size(-1))
        component_hidden = torch.gather(llm_out, dim=1, index=gather_index)
        component_hidden = output_projection(component_hidden)
        component_hidden = component_hidden.reshape(B, n_vars, patch_num, self.d_ff)
        component_hidden = component_hidden.permute(0, 1, 3, 2).contiguous()
        y_component = head(component_hidden)
        y_component = y_component.permute(0, 2, 1).contiguous()
        if BN != B * n_vars:
            raise RuntimeError(f'Unexpected branch batch: BN={BN}, B*n_vars={B * n_vars}.')
        return y_component

    def _run_component_branch(self, component, prompts, patch_embedding, reprogramming_layer, output_projection, head,
        source_embeddings):
        """
        Run ONE component branch
        end-to-end:

            Component
                ↓
            PatchEmbedding
                ↓
            Reprogramming
                ↓
            Prompt
                ↓
            Shared Frozen LLM
                ↓
            Output Projection
                ↓
            Forecast Head

        Components remain conceptually
        independent.

        Only one component enters the
        shared LLM at a time.
        """
        branch = self._prepare_component_branch(component=component, prompts=prompts, patch_embedding=patch_embedding,
            reprogramming_layer=reprogramming_layer, source_embeddings=source_embeddings)
        llm_out = self.llm_model(inputs_embeds=branch['llm_in'], attention_mask=branch['attention_mask']).last_hidden_state
        y_component = self._project_branch_output(llm_out=llm_out, branch=branch, output_projection=output_projection,
            head=head)
        del llm_out
        del branch
        return y_component

    def _parallel_component_forecasts(self, trend, seasonal, event, residual, trend_prompts, seasonal_prompts,
        event_prompts, residual_prompts, source_embeddings):
        """
        Structural component-wise
        processing with SEQUENTIAL
        LLM execution.

        Conceptually:

            Trend branch

            Seasonality branch

            Event branch

            Residual branch

        Each branch keeps:

            - Prompt
            - PatchEmbedding
            - ReprogrammingLayer
            - OutputProjection
            - Forecast Head

        Only frozen LLM weights
        are shared.

        Execution:

            Trend
              ↓
            Seasonality
              ↓
            Event
              ↓
            Residual
        """
        y_trend = self._run_component_branch(
            component=trend, prompts=trend_prompts, patch_embedding=self.trend_patch_embedding,
            reprogramming_layer=self.trend_reprogramming, output_projection=self.trend_output_projection,
            head=self.trend_head, source_embeddings=source_embeddings,
        )
        y_seasonal = self._run_component_branch(
            component=seasonal, prompts=seasonal_prompts, patch_embedding=self.seasonal_patch_embedding,
            reprogramming_layer=self.seasonal_reprogramming, output_projection=self.seasonal_output_projection,
            head=self.seasonal_head, source_embeddings=source_embeddings,
        )
        y_event = self._run_component_branch(
            component=event, prompts=event_prompts, patch_embedding=self.event_patch_embedding,
            reprogramming_layer=self.event_reprogramming, output_projection=self.event_output_projection,
            head=self.event_head, source_embeddings=source_embeddings,
        )
        y_residual = self._run_component_branch(
            component=residual, prompts=residual_prompts, patch_embedding=self.residual_patch_embedding,
            reprogramming_layer=self.residual_reprogramming, output_projection=self.residual_output_projection,
            head=self.residual_head, source_embeddings=source_embeddings,
        )
        return (y_trend, y_seasonal, y_event, y_residual)

    @staticmethod
    def _update_input_history(history, next_segment, detach_history=True):
        """Slide (B,seq_len,N) RAW history by the predicted segment.

        detach_history=True prevents later AR steps from backpropagating
        through earlier predictions. Direct gradients still flow from
        every predicted segment to the final supervised loss.
        """
        if history.ndim != 3 or next_segment.ndim != 3:
            raise ValueError('history and next_segment must have shape (B,T,N)')
        if history.size(0) != next_segment.size(0) or history.size(2) != next_segment.size(2):
            raise ValueError('history / next_segment batch or channel mismatch')
        segment = next_segment.detach() if detach_history else next_segment
        return torch.cat((history, segment), dim=1)[:, -history.size(1):, :].contiguous()

    def _autoregressive_forecast_raw_history(self, x_enc, x_mark_enc=None, x_mark_dec=None, return_components=False):
        """Forecast in ar_len chunks with denorm -> RAW Input feedback.

        For EACH step:
          1. RevIN(norm) on CURRENT raw History.
          2. PFD / prompts / per-component SRE-LLM-head predictions.
          3. Sum four components in CURRENT normalized space.
          4. RevIN(denorm) immediately, with CURRENT step's statistics.
          5. Update raw History using denormalized prediction.
          6. Re-run norm/PFD on updated History next iteration.

        x_mark_dec (if provided) supplies known FUTURE time features:
        standard layout (B,label_len+pred_len,D) or (B,pred_len,D).
        Decoder labels / targets are never fed into the history.
        If decoder time features are absent, event marks are not
        reused with stale timestamps in future iterations; subsequent
        decompositions fall back to median-MAD event extraction.

        Debug component keys ending in '_norm' refer to per-step LOCAL
        normalized spaces. They must not be added across different AR
        steps as though they shared a single RevIN mean / stdev.
        """
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len:
            raise ValueError(f'Expected x_enc (B,{self.seq_len},N); got {tuple(x_enc.shape)}')
        if self.pred_len <= 0:
            raise ValueError(f'pred_len must be positive: {self.pred_len}')
        steps = (self.pred_len + self.ar_len - 1) // self.ar_len
        history_raw = x_enc
        mark_history = x_mark_enc
        predictions_raw = []
        if return_components:
            component_predictions = {'trend_norm': [], 'seasonal_norm': [], 'event_norm': [], 'residual_norm': [],
                'sum_norm': []}
        source_embeddings = self._source_embeddings()
        if x_mark_dec is not None and mark_history is not None and (steps > 1):
            if (x_mark_dec.ndim != 3 or x_mark_dec.size(0) != history_raw.size(0)
                    or x_mark_dec.size(2) != mark_history.size(2)
                    or x_mark_dec.size(1) < self.pred_len):
                raise ValueError(
                    'x_mark_dec must have shape (B,label_len+pred_len,D) or (B,pred_len,D), '
                    'with the same B and D as x_mark_enc'
                )
            future_mark_start = x_mark_dec.size(1) - self.pred_len
        else:
            future_mark_start = None
        for step in range(steps):
            step_offset = step * self.ar_len
            take_len = min(self.ar_len, self.pred_len - step_offset)
            # Each iteration recalculates RevIN stats on updated RAW history.
            x_norm = self.revin(history_raw, 'norm')
            trend, seasonal, event, residual, period = self._decompose(x_norm, mark_history)
            trend_prompts, seasonal_prompts, event_prompts, residual_prompts = self._build_component_prompts(trend,
                seasonal, event, residual, period, forecast_horizon=self.ar_len, step_offset=step_offset)
            next_trend, next_seasonal, next_event, next_residual = self._parallel_component_forecasts(
                trend=trend, seasonal=seasonal, event=event, residual=residual,
                trend_prompts=trend_prompts, seasonal_prompts=seasonal_prompts,
                event_prompts=event_prompts, residual_prompts=residual_prompts,
                source_embeddings=source_embeddings,
            )
            segment_norm = next_trend + next_seasonal + next_event + next_residual
            # Sum components, then denormalize BEFORE feeding back predictions.
            segment_raw = self.revin(segment_norm, 'denorm')
            predictions_raw.append(segment_raw[:, :take_len, :])
            if return_components:
                for key, value in (('trend_norm', next_trend), ('seasonal_norm', next_seasonal), ('event_norm',
                    next_event), ('residual_norm', next_residual), ('sum_norm', segment_norm)):
                    component_predictions[key].append(value[:, :take_len, :])
            if step + 1 < steps:
                # Append the predicted segment; never append future ground truth.
                history_raw = self._update_input_history(history_raw, segment_raw, detach_history=self.ar_detach_history)
                if mark_history is not None:
                    if future_mark_start is None:
                        mark_history = None
                    else:
                        mark_start = future_mark_start + step_offset
                        next_marks = x_mark_dec[:, mark_start:mark_start + self.ar_len,
                            :].to(device=mark_history.device, dtype=mark_history.dtype)
                        if next_marks.size(1) != self.ar_len:
                            raise ValueError('Not enough future time markers for AR update')
                        mark_history = torch.cat((mark_history, next_marks), dim=1)[:, -self.seq_len:, :].contiguous()
        prediction = torch.cat(predictions_raw, dim=1)
        if return_components:
            components = {name: torch.cat(parts, dim=1) for name, parts in component_predictions.items()}
            return (prediction, components)
        return prediction

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Return (B,pred_len,N) predictions in ORIGINAL data units.

        The decoder target x_dec is deliberately unused: each AR step
        feeds back only its OWN denormalized prediction, never ground
        truth future targets (avoids teacher-forcing/data leakage).
        """
        return self._autoregressive_forecast_raw_history(x_enc=x_enc, x_mark_enc=x_mark_enc, x_mark_dec=x_mark_dec,
            return_components=False)

    def forecast_with_components(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """Return (final_original_scale_forecast, local_norm_components).

        Component tensors preserve the original debug API keys, but
        each AR segment is normalized according to that segment's
        CURRENT input History: summing sum_norm and denormalizing once
        with any single step's stats is NOT valid. Use the returned
        final prediction for raw-space metrics (MSE/MAE).
        """
        return self._autoregressive_forecast_raw_history(x_enc=x_enc, x_mark_enc=x_mark_enc, x_mark_dec=x_mark_dec,
            return_components=True)

    def calcute_lags(self, x_enc):
        """
        x_enc:
            (B*,T,1)

        return:
            lags
            (B*,top_k)
        """
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        k = min(self.top_k, mean_value.shape[-1])
        _, lags = torch.topk(mean_value, k, dim=-1)
        if k < self.top_k:
            pad = torch.zeros(lags.shape[0], self.top_k - k, device=lags.device, dtype=lags.dtype)
            lags = torch.cat([lags, pad], dim=-1)
        return lags
