import argparse
import os
import random
import time

import numpy as np
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM, TimesNet, PatchTST, FPT, AutoTimes
from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate, vali, load_content


os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_device_float(x, device):
    if x is None:
        return None
    return x.float().to(device)


def _slice_eval_features(x, features):
    if str(features).upper() == "MS":
        return x[:, :, -1:]
    return x


def _gather_tensor_for_save(tensor, accelerator):
    tensor = tensor.detach()
    if accelerator.num_processes > 1:
        tensor = accelerator.gather_for_metrics(tensor)
    return tensor.float().cpu().numpy()


@torch.no_grad()
def evaluate_components(args, accelerator, model, test_loader, setting):

    raw_model = accelerator.unwrap_model(model)
    if not hasattr(raw_model, "decompose_future_for_metrics"):
        accelerator.print(
            "Component evaluation skipped: models/TimeLLM.py must provide "
            "Model.decompose_future_for_metrics()."
        )
        return None

    names = ("trend", "seasonal", "event", "residual", "total")
    device = accelerator.device
    sq = {name: torch.zeros((), device=device, dtype=torch.float64) for name in names}
    ab = {name: torch.zeros((), device=device, dtype=torch.float64) for name in names}
    count = {name: torch.zeros((), device=device, dtype=torch.float64) for name in names}
    var_sq = {}
    var_ab = {}
    var_count = {}
    pred_recon_sq = torch.zeros((), device=device, dtype=torch.float64)
    true_recon_sq = torch.zeros((), device=device, dtype=torch.float64)
    recon_count = torch.zeros((), device=device, dtype=torch.float64)

    save_predictions = bool(int(args.save_component_predictions))
    pred_store = {name: [] for name in names}
    true_store = {name: [] for name in names}

    model.eval()
    progress = tqdm(
        enumerate(test_loader), total=len(test_loader),
        disable=not accelerator.is_local_main_process,
        desc="PFD component MSE/MAE evaluation",
    )

    for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress:
        batch_x = _to_device_float(batch_x, device)
        batch_y = _to_device_float(batch_y, device)
        batch_x_mark = _to_device_float(batch_x_mark, device)
        batch_y_mark = _to_device_float(batch_y_mark, device)

        # Same decoder input construction as the original run_main.py.
        dec_inp = torch.cat([
            batch_y[:, :args.label_len, :],
            torch.zeros_like(batch_y[:, -args.pred_len:, :]),
        ], dim=1)

        # No true future values are sent to the forecaster. x_dec is
        # unused by this DDAI-Cast model; decoder time marks are allowed.
        if args.use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                final_pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            final_pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if isinstance(final_pred, (tuple, list)):
            final_pred = final_pred[0]

        final_pred = final_pred[:, -args.pred_len:, :]
        final_true = batch_y[:, -args.pred_len:, :]
        final_pred = _slice_eval_features(final_pred, args.features).float()
        final_true = _slice_eval_features(final_true, args.features).float()
        if tuple(final_pred.shape) != tuple(final_true.shape):
            raise ValueError(
                f"Prediction/target mismatch: {final_pred.shape} vs {final_true.shape}"
            )

        # Use historical input and future markers for component evaluation.
        history_for_metrics = _slice_eval_features(batch_x, args.features).float()
        pred_comp = raw_model.decompose_future_for_metrics(
            history_for_metrics, final_pred, batch_x_mark, batch_y_mark
        )
        true_comp = raw_model.decompose_future_for_metrics(
            history_for_metrics, final_true, batch_x_mark, batch_y_mark
        )
        pred_comp['total'] = final_pred
        true_comp['total'] = final_true

        if not var_sq:
            n_vars = final_pred.shape[-1]
            var_sq = {name: torch.zeros(n_vars, device=device, dtype=torch.float64) for name in names}
            var_ab = {name: torch.zeros(n_vars, device=device, dtype=torch.float64) for name in names}
            var_count = {name: torch.zeros(n_vars, device=device, dtype=torch.float64) for name in names}

        for name in names:
            diff = (pred_comp[name] - true_comp[name]).double()
            if not torch.isfinite(diff).all():
                raise ValueError(f"Non-finite component evaluation values: {name}")
            sq[name] += diff.square().sum()
            ab[name] += diff.abs().sum()
            count[name] += float(diff.numel())
            var_sq[name] += diff.square().sum(dim=(0, 1))
            var_ab[name] += diff.abs().sum(dim=(0, 1))
            var_count[name] += float(diff.size(0) * diff.size(1))

        pred_sum = sum(pred_comp[name] for name in names[:4])
        true_sum = sum(true_comp[name] for name in names[:4])
        pred_recon_sq += (pred_sum.double() - final_pred.double()).square().sum()
        true_recon_sq += (true_sum.double() - final_true.double()).square().sum()
        recon_count += float(final_pred.numel())

        if save_predictions:
            for name in names:
                # Keep original accelerate gather behavior for array saving.
                pred_np = _gather_tensor_for_save(pred_comp[name], accelerator)
                true_np = _gather_tensor_for_save(true_comp[name], accelerator)
                if accelerator.is_local_main_process:
                    pred_store[name].append(pred_np)
                    true_store[name].append(true_np)

    if not var_sq:
        raise RuntimeError("test_loader did not yield any test batches")

    # Accumulate the complete test set (not the mean of batch means).
    for name in names:
        sq[name] = accelerator.reduce(sq[name], reduction='sum')
        ab[name] = accelerator.reduce(ab[name], reduction='sum')
        count[name] = accelerator.reduce(count[name], reduction='sum')
        var_sq[name] = accelerator.reduce(var_sq[name], reduction='sum')
        var_ab[name] = accelerator.reduce(var_ab[name], reduction='sum')
        var_count[name] = accelerator.reduce(var_count[name], reduction='sum')
    pred_recon_sq = accelerator.reduce(pred_recon_sq, reduction='sum')
    true_recon_sq = accelerator.reduce(true_recon_sq, reduction='sum')
    recon_count = accelerator.reduce(recon_count, reduction='sum')

    result = {
        name: {
            'mse': (sq[name] / count[name].clamp_min(1)).item(),
            'mae': (ab[name] / count[name].clamp_min(1)).item(),
            'mse_per_variable': (var_sq[name] / var_count[name].clamp_min(1)).cpu().numpy(),
            'mae_per_variable': (var_ab[name] / var_count[name].clamp_min(1)).cpu().numpy(),
        }
        for name in names
    }
    pred_recon_mse = (pred_recon_sq / recon_count.clamp_min(1)).item()
    true_recon_mse = (true_recon_sq / recon_count.clamp_min(1)).item()
    result_dir = os.path.join(args.component_result_dir, setting + '-' + args.model_comment)

    if accelerator.is_local_main_process:
        os.makedirs(result_dir, exist_ok=True)
        print('\n' + '=' * 63)
        print('PFD COMPONENT METRICS | Predicted future vs Ground truth')
        print('Full test dataset; same scale as model outputs / batch_y')
        print('-' * 63)
        print(f'{"Component":<20} {"MSE":>18} {"MAE":>18}')
        print('-' * 63)
        for name in names:
            print(f'{name:<20} {result[name]["mse"]:>18.8f} {result[name]["mae"]:>18.8f}')
        print('-' * 63)
        print(f'Pred reconstruction MSE: {pred_recon_mse:.12e}')
        print(f'True reconstruction MSE: {true_recon_mse:.12e}')
        print('=' * 63)

        import csv
        csv_path = os.path.join(result_dir, 'component_metrics.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            writer.writerow(['component', 'MSE', 'MAE'])
            for name in names:
                writer.writerow([name, f'{result[name]["mse"]:.12f}', f'{result[name]["mae"]:.12f}'])

        with open(os.path.join(result_dir, 'component_metrics.txt'), 'w', encoding='utf-8') as fp:
            fp.write('PFD COMPONENT METRICS | Forecast vs Ground truth\n')
            fp.write('Component, MSE, MAE\n')
            for name in names:
                fp.write(f'{name}, {result[name]["mse"]:.12f}, {result[name]["mae"]:.12f}\n')
            fp.write(f'Pred reconstruction MSE: {pred_recon_mse:.12e}\n')
            fp.write(f'True reconstruction MSE: {true_recon_mse:.12e}\n')
            fp.write('\nPer-variable (MSE, MAE)\n')
            for name in names:
                fp.write(f'{name} MSE: {result[name]["mse_per_variable"]}\n')
                fp.write(f'{name} MAE: {result[name]["mae_per_variable"]}\n')

        arrays = {
            f'{name}_{kind}': value
            for name in names
            for kind, value in (
                ('mse', result[name]['mse']),
                ('mae', result[name]['mae']),
                ('mse_per_variable', result[name]['mse_per_variable']),
                ('mae_per_variable', result[name]['mae_per_variable']),
            )
        }
        arrays.update(
            pred_reconstruction_mse=pred_recon_mse,
            true_reconstruction_mse=true_recon_mse,
        )
        np.savez_compressed(os.path.join(result_dir, 'component_metrics.npz'), **arrays)

        if save_predictions:
            tensors = {}
            for name in names:
                tensors[f'{name}_pred'] = np.concatenate(pred_store[name], axis=0)
                tensors[f'{name}_true'] = np.concatenate(true_store[name], axis=0)
            np.savez_compressed(
                os.path.join(result_dir, 'component_predictions.npz'), **tensors
            )
        print(f'Saved component results: {result_dir}')

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
    return result


# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description='Time-LLM: train + PFD component metrics without checkpoints')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# =========================
# basic config
# =========================
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, choices=[1], help='1: train from scratch then component-test; checkpoint-free mode')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument(
    '--model',
    type=str,
    required=True,
    default='Autoformer',
    help='model name, options: '
         '[Autoformer, DLinear, TimesNet, PatchTST, FPT, AutoTimes, TimeLLM]'
)
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# =========================
# data loader
# =========================
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M: multivariate predict multivariate, '
                         'S: univariate predict univariate, '
                         'MS: multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s: secondly, t: minutely, h: hourly, d: daily, '
                         'b: business days, w: weekly, m: monthly], '
                         'you can also use more detailed freq like 15min or 3h')

# =========================
# forecasting task
# =========================
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--ar_len', type=int, default=96, help='iterative forecast segment length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# =========================
# model define
# =========================
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='use domain prompt content')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model')  # LLAMA, GPT2, BERT, QWEN
parser.add_argument('--llm_dim', type=int, default=4096, help='LLM model dimension')
parser.add_argument('--llm_layers', type=int, default=6)

# =========================
# Baseline model options
# =========================

# PatchTST
parser.add_argument(
    '--head_dropout',
    type=float,
    default=0.0,
    help='PatchTST prediction head dropout'
)

# FPT
parser.add_argument(
    '--gpt_layers',
    type=int,
    default=6,
    help='number of GPT-2 layers for FPT'
)

parser.add_argument(
    '--gpt2_path',
    type=str,
    default='gpt2',
    help='Hugging Face GPT-2 name or local path'
)

parser.add_argument(
    '--freeze_gpt',
    type=int,
    default=1,
    help='1: freeze GPT-2 backbone'
)

# AutoTimes
parser.add_argument(
    '--token_len',
    type=int,
    default=96,
    help='AutoTimes time-series token length'
)

# Prophet-inspired decomposition options used by the supplied model.
parser.add_argument('--period', type=int, default=24, help='main seasonal period')
parser.add_argument('--default_period', type=int, default=24, help='fallback seasonal period')
parser.add_argument('--n_changepoints', type=int, default=10)
parser.add_argument('--changepoint_range', type=float, default=0.8)
parser.add_argument('--changepoint_prior_scale', type=float, default=0.05)
parser.add_argument('--fourier_order', type=int, default=5)
parser.add_argument('--seasonality_prior_scale', type=float, default=10.0)
parser.add_argument('--seasonality_mode', type=str, default='additive')
parser.add_argument('--event_prior_scale', type=float, default=1.0)
parser.add_argument('--event_threshold', type=float, default=3.0)
parser.add_argument('--event_lower_window', type=int, default=0)
parser.add_argument('--event_upper_window', type=int, default=0)
parser.add_argument('--event_feature_index', type=int, default=-1)
parser.add_argument('--prophet_seasonal_periods', type=str, default=None)
parser.add_argument('--num_tokens', type=int, default=1000)
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--num_kernels', type=int, default=6, help='number of kernels for TimesNet')

# Optional component-specific patch settings.
parser.add_argument('--trend_patch_len', type=int, default=None)
parser.add_argument('--trend_stride', type=int, default=None)
parser.add_argument('--seasonal_patch_len', type=int, default=None)
parser.add_argument('--seasonal_stride', type=int, default=None)
parser.add_argument('--event_patch_len', type=int, default=None)
parser.add_argument('--event_stride', type=int, default=None)
parser.add_argument('--residual_patch_len', type=int, default=None)
parser.add_argument('--residual_stride', type=int, default=None)

# =========================
# fair comparison options
# =========================
parser.add_argument('--repeat_raw_tokens', action='store_true', default=False,
                    help='repeat raw patch tokens to match longer token sequence length')
parser.add_argument('--target_token_factor', type=int, default=1,
                    help='multiply raw patch token length by this factor')

# =========================
# optimization
# =========================
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--percent', type=int, default=100)

# Hybrid Backbone: Qwen + LLaMA
parser.add_argument(
    '--hybrid_backbones',
    type=int,
    default=0,
    choices=[0, 1],
    help='Enable Qwen + LLaMA hybrid backbone'
)

parser.add_argument(
    '--llama_layers',
    type=int,
    default=4,
    help='Number of LLaMA layers'
)

parser.add_argument(
    '--llama_model_name',
    type=str,
    default='huggyllama/llama-7b',
    help='Hugging Face model ID or local LLaMA path'
)


# =========================
# component evaluation
# =========================
parser.add_argument(
    '--component_eval',
    type=int,
    default=1,
    help='1: evaluate PFD Trend/Seasonal/Event/Residual after the last training epoch',
)
parser.add_argument(
    '--save_component_predictions',
    type=int,
    default=0,
    help='1: save all component prediction/target tensors; 0: metrics only',
)
parser.add_argument(
    '--component_result_dir',
    type=str,
    default='./component_results',
    help='directory for component-wise metrics and predictions',
)
args = parser.parse_args()

# argparse gives None to explicitly optional component patch settings.  The
# supplied model uses getattr(configs, name, fallback), so remove None-valued
# attributes to preserve its internal fallback behavior.
for _name in [
    'trend_patch_len', 'trend_stride',
    'seasonal_patch_len', 'seasonal_stride',
    'event_patch_len', 'event_stride',
    'residual_patch_len', 'residual_stride',
]:
    if getattr(args, _name) is None:
        delattr(args, _name)

# reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Training-only execution. DeepSpeed ZeRO-2 gets the optimizer below.
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


# ============================================================
# Main experiment loop
# ============================================================
for ii in range(args.itr):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des,
        ii
    )

    # Prompt/domain content must exist before constructing TimeLLM.Model.
    args.content = load_content(args)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()

    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()

    elif args.model == 'TimesNet':
        model = TimesNet.Model(args).float()

    elif args.model == 'PatchTST':
        model = PatchTST.Model(args).float()

    elif args.model == 'FPT':
        model = FPT.Model(args).float()

    elif args.model == 'AutoTimes':
        model = AutoTimes.Model(args).float()

    else:
        model = TimeLLM.Model(args).float()

    if int(args.component_eval) == 1 and not hasattr(model, 'decompose_future_for_metrics'):
        raise AttributeError(
            'The selected models/TimeLLM.py must implement decompose_future_for_metrics().'
        )

    if accelerator.is_local_main_process:
        accelerator.print("=" * 70)
        accelerator.print(f"Experiment: {setting}")
        accelerator.print(f"Model: {args.model}")
        accelerator.print(f"is_training: {args.is_training}")
        accelerator.print("checkpoint_mode: disabled (last epoch weights)")
        accelerator.print(f"component_eval: {args.component_eval}")
        accelerator.print(f"save_component_predictions: {args.save_component_predictions}")
        accelerator.print("=" * 70)

    # Checkpoint-free mode: train all requested epochs and evaluate the
    # final in-memory model. Never save, load, delete or touch checkpoints.
    # DeepSpeed ZeRO-2 receives the optimizer through accelerator.prepare.
    train_steps = len(train_loader)
    if train_steps == 0:
        raise RuntimeError('Training dataset has zero batches')

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model_optim,
            T_max=20,
            eta_min=1e-8,
        )
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate,
        )

    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader,
        vali_loader,
        test_loader,
        model,
        model_optim,
        scheduler,
    )

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    time_now = time.time()

    # ====================================================
    # Training
    # ====================================================
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        raw_model = accelerator.unwrap_model(model)

        if hasattr(raw_model, "reset_epoch_time"):
            raw_model.reset_epoch_time()
        if hasattr(raw_model, "reset_block_time"):
            raw_model.reset_block_time()

        model.train()
        sync_if_cuda(accelerator.device)
        epoch_start_time = time.time()

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            disable=not accelerator.is_local_main_process,
        )

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            dec_inp = torch.zeros_like(
                batch_y[:, -args.pred_len:, :]
            ).float().to(accelerator.device)
            dec_inp = torch.cat(
                [batch_y[:, :args.label_len, :], dec_inp],
                dim=1,
            ).float().to(accelerator.device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                        )[0]
                    else:
                        outputs = model(
                            batch_x,
                            batch_x_mark,
                            dec_inp,
                            batch_y_mark,
                        )

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    target = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, target)
                    train_loss.append(loss.item())

                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()

            else:
                if args.output_attention:
                    outputs = model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                    )[0]
                else:
                    outputs = model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                    )

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                target = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, target)
                train_loss.append(loss.item())

                accelerator.backward(loss)
                model_optim.step()

            if (i + 1) % 100 == 20 and accelerator.is_local_main_process:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1,
                        epoch + 1,
                        loss.item(),
                    )
                )
                speed = (time.time() - time_now) / max(iter_count, 1)
                left_time = speed * (
                    (args.train_epochs - epoch) * train_steps - i
                )
                accelerator.print(
                    '\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(
                        speed,
                        left_time,
                    )
                )
                iter_count = 0
                time_now = time.time()

            if args.lradj == 'TST':
                adjust_learning_rate(
                    accelerator,
                    model_optim,
                    scheduler,
                    epoch + 1,
                    args,
                    printout=False,
                )
                scheduler.step()

        sync_if_cuda(accelerator.device)
        total_epoch_time = time.time() - epoch_start_time

        if hasattr(raw_model, "get_epoch_time"):
            total_forward_time = raw_model.get_epoch_time()
        else:
            total_forward_time = -1.0

        train_loss = np.average(train_loss)

        accelerator.print(
            f"Epoch {epoch + 1:03d} | "
            f"train_total: {total_epoch_time:.2f}s | "
            f"train_forward: {total_forward_time:.2f}s"
        )

        # Optional model-internal block timing.
        if accelerator.is_local_main_process:
            if hasattr(raw_model, "print_block_time_summary"):
                raw_model.print_block_time_summary(grouped=False)
                raw_model.print_block_time_summary(grouped=True)

            if hasattr(raw_model, "get_group_block_time_summary"):
                grouped_summary = raw_model.get_group_block_time_summary()
                accelerator.print(
                    "[Grouped Totals] "
                    f"normalize: {grouped_summary['normalize_block']['total']:.4f}s | "
                    f"stats: {grouped_summary['stats_block']['total']:.4f}s | "
                    f"prompt: {grouped_summary['prompt_block']['total']:.4f}s | "
                    f"patch: {grouped_summary['patch_block']['total']:.4f}s | "
                    f"reprog: {grouped_summary['reprogramming_block']['total']:.4f}s | "
                    f"llm: {grouped_summary['llm_block']['total']:.4f}s | "
                    f"output: {grouped_summary['output_block']['total']:.4f}s"
                )

        # Validation/test monitoring during training is kept exactly in the
        # existing style. Final component metrics are computed only after
        # the final training epoch is finished (no checkpoint).
        sync_if_cuda(accelerator.device)
        val_start_time = time.time()
        vali_loss, vali_mae_loss = vali(
            args,
            accelerator,
            model,
            vali_data,
            vali_loader,
            criterion,
            mae_metric,
        )
        sync_if_cuda(accelerator.device)
        total_val_time = time.time() - val_start_time

        sync_if_cuda(accelerator.device)
        test_start_time = time.time()
        test_loss, test_mae_loss = vali(
            args,
            accelerator,
            model,
            test_data,
            test_loader,
            criterion,
            mae_metric,
        )
        sync_if_cuda(accelerator.device)
        total_test_time = time.time() - test_start_time

        accelerator.print(
            f"Epoch {epoch + 1:03d} | "
            f"val_total: {total_val_time:.2f}s | "
            f"test_total: {total_test_time:.2f}s"
        )

        accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} | Vali Loss: {2:.7f} | "
            "Test Loss: {3:.7f} | Test MAE: {4:.7f}".format(
                epoch + 1,
                train_loss,
                vali_loss,
                test_loss,
                test_mae_loss,
            )
        )

        # Checkpoint-free: no EarlyStopping (it writes checkpoint files).
        # The last epoch's in-memory weights are used at test time.

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print(
                    "lr = {:.10f}".format(model_optim.param_groups[0]['lr'])
                )
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print(
                        "lr = {:.10f}".format(
                            model_optim.param_groups[0]['lr']
                        )
                    )
                adjust_learning_rate(
                    accelerator,
                    model_optim,
                    scheduler,
                    epoch + 1,
                    args,
                    printout=True,
                )
        else:
            accelerator.print(
                'Updating learning rate to {}'.format(
                    scheduler.get_last_lr()[0]
                )
            )

    # ========================================================
    # Final component-wise evaluation
    # ========================================================
    if int(args.component_eval) == 1:
        evaluate_components(
            args=args,
            accelerator=accelerator,
            model=model,
            test_loader=test_loader,
            setting=setting,
        )

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()


