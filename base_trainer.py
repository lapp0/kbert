# Based these projects, attribution for discovered improvements within
# https://github.com/KellerJordan/modded-nanogpt
# and https://github.com/Synthyra/SpeedRunningESM2

from typing import get_origin, get_args, Union, Optional

import argparse
import contextlib
import dataclasses
import math
import os
import pathlib
import subprocess
import sys
import time
import uuid

import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloading import DistributedPaddedDataLoader
from model import CastedLinear, ModelConfig
from optimizer import Muon


# setup dist
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = torch.device(f'cuda:{ddp_local_rank}')
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
dist.barrier()
master_process = (ddp_rank == 0)

print(f'using device: {device}')


# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    pathlib.Path('logs').mkdir(exist_ok=True)
    logfile = pathlib.Path('logs') / f'{run_id}.txt'


def print0(s, logonly=False):
    if master_process:
        with logfile.open('a') as f:
            if not logonly:
                print(s)
            print(s, file=f)


# log input code and environment
if master_process:
    print(logfile.stem)
print0('=' * 100, logonly=True)
print0("Command: " + " ".join(sys.argv) + "\n", logonly=True)
print0('=' * 100, logonly=True)
for filename in [sys.argv[0], __file__, "optimizer.py", "model.py", "dataloading.py"]:
    print0(open(filename, 'r', encoding='utf-8').read(), logonly=True)
    print0('=' * 100, logonly=True)
print0(f'Running python {sys.version}', logonly=True)
print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:', logonly=True)
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('=' * 100, logonly=True)


@dataclasses.dataclass
class TrainingArguments:
    # Data hyperparams
    input_bin: str
    input_valid_bin: str

    # Optimization hyperparams
    batch_size: int  # split into (num devices * accum steps) local mini batches
    grad_accum_per_device: int
    num_steps: int
    warmup_steps: int
    cooldown_steps: int
    max_length: int
    max_epochs: Optional[int]

    # Evaluation and logging hyperparams
    valid_loss_every: int
    hf_model_name: Optional[str]
    save_every: Optional[int] = None

    # adam
    lr_head: Optional[float] = None
    lr_embed: float = 0.01
    lr_scalar: float = 0.005
    # muon
    lr_hidden: float = 0.005
    muon_momentum_warmup_steps: int = 300  # steps for warmup momentum, 0.85 -> 0.95


class BaseTrainer:
    def __init__(self, args, model, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

        print0(f'Model config: {getattr(model, "config", None)}')
        print0(f'Args: {self.args.__dict__}')

        # prepare dataloaders in accordance with global batch & local mini-batch sizes
        self.local_mini_batch_size = args.batch_size // (ddp_world_size * args.grad_accum_per_device)
        self.train_loader, self.valid_loader = self._create_data_loaders()

        # Move model to GPU, compile, wrap in DDP
        self.model = self.prepare_model(model)
        self.raw_model = self.model.module

        # Setup Muon + Adam + LR schedulers (subclass can override `_create_param_groups` if needed)
        self.optimizers, self.schedulers, self.muon_optimizer = self.get_optimizers_and_schedulers(self.raw_model, args)

        # For tracking training time
        self.training_time_ms = 0
        self.t0 = None

    def _create_data_loaders(self):
        train_loader = DistributedPaddedDataLoader(
            self.args.input_bin,
            self.local_mini_batch_size,
            ddp_rank,
            ddp_world_size,
            bos_id=self.tokenizer.cls_token_id,
            pad_id=self.tokenizer.pad_token_id,
            allow_windowed=self.dataloader_allow_windowed,
            max_epochs=self.args.max_epochs
        )
        valid_loader = DistributedPaddedDataLoader(
            self.args.input_valid_bin,
            self.local_mini_batch_size,
            ddp_rank,
            ddp_world_size,
            bos_id=self.tokenizer.cls_token_id,
            pad_id=self.tokenizer.pad_token_id,
            allow_windowed=self.dataloader_allow_windowed,
            max_epochs=1
        )

        print0(f'Grad accumulation steps per device: {self.args.grad_accum_per_device}')
        print0(f'Across {ddp_world_size} GPUs')
        print0(f'Totaling {ddp_world_size * self.args.grad_accum_per_device} steps across devices per train step.')
        print0(f'Adjusted local mini batch size: {self.local_mini_batch_size} tokens')
        print0(f'Total batch size: {self.args.batch_size} tokens')

        print0(f'Training DataLoader: {len(train_loader.files)} files')
        print0(f'Validation DataLoader: {len(valid_loader.files)} files')
        print0('=' * 100, logonly=True)

        return train_loader, valid_loader

    def prepare_model(self, model):
        model = model.cuda().bfloat16()
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        config.coordinate_descent_tuning = True
        model = torch.compile(model)
        model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        return model

    def get_optimizers_and_schedulers(self, raw_model, args):
        """
        Initializes Muon + Adam optimizers and returns them along with their LR schedulers.
        """
        adam_params = [
            dict(params=[raw_model.lm_head.weight], lr=args.lr_embed),
            dict(params=[p for p in raw_model.encoder.parameters() if p.ndim < 2], lr=args.lr_scalar),
        ]
        hidden_matrix_params = [p for p in raw_model.encoder.blocks.parameters() if p.ndim == 2]
        adam_optimizer = torch.optim.Adam(adam_params, betas=(0.8, 0.95), fused=True)
        muon_optimizer = Muon(hidden_matrix_params, lr=args.lr_hidden, momentum=0.95)
        optimizers = [adam_optimizer, muon_optimizer]

        def get_lr(it):
            assert it <= args.num_steps
            if it < args.warmup_steps:
                return (it + 1) / args.warmup_steps
            elif it < args.num_steps - args.cooldown_steps:
                return 1.0
            else:
                decay_ratio = (args.num_steps - it) / args.cooldown_steps
                return decay_ratio

        schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

        return optimizers, schedulers, muon_optimizer

    def get_param_counts(self):
        encoder_params = 0
        head_params = 0
        for name, param in self.model.named_parameters():
            if name.endswith("_head.weight"):
                head_params += param.numel()
            else:
                encoder_params += param.numel()
        return dict(encoder=encoder_params, head=head_params)

    def train_step(self, step):
        self.model.train()
        for i in range(self.args.grad_accum_per_device):
            with self.model.no_sync() if i < (self.args.grad_accum_per_device - 1) else contextlib.nullcontext():
                self.model(**self.next_train_batch).backward()
                self.next_train_batch = self.next_batch(train=True)
        for p in self.model.parameters():
            p.grad /= self.args.grad_accum_per_device  # DDP averages, local accum steps SUM

        # Update Muon momentum
        frac = min(step / self.args.muon_momentum_warmup_steps, 1)
        for group in self.muon_optimizer.param_groups:
            group['momentum'] = (1 - frac) * 0.85 + frac * 0.95

        # Step both optimizers, then schedulers
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.step()
            sched.step()

        self.model.zero_grad(set_to_none=True)

    def validation_step(self, step, timed_steps):
        torch.cuda.synchronize()
        self.training_time_ms += 1000 * (time.perf_counter() - self.t0)
        self.model.eval()
        self.valid_loader.reset()
        val_loss = torch.tensor(0.0, device="cuda")
        valid_tokens = torch.tensor(0, device="cuda")

        pad_id = self.tokenizer.pad_token_id
        with torch.no_grad():
            val_batch = self.next_batch(train=False)
            while val_batch is not None:
                val_inputs = val_batch["input_ids"]
                num_val_tokens = (val_inputs != pad_id).sum()
                valid_tokens += num_val_tokens
                val_loss += self.model(**val_batch) * num_val_tokens
                val_batch = self.next_batch(train=False)

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        val_loss /= valid_tokens

        print0(
            f'step:{step}/{self.args.num_steps} val_loss:{val_loss:.4f} '
            f'train_time:{self.training_time_ms:.0f}ms '
            f'step_avg:{self.training_time_ms / (timed_steps - 1):.2f}ms '
            f'perplexity:{(math.e**val_loss):.4f} '
            f'param_count:{str(self.get_param_counts())} '
            f'tokens: {valid_tokens.item():,}'
        )
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def save_step(self, step, timed_steps):
        torch.cuda.synchronize()
        self.training_time_ms += 1000 * (time.perf_counter() - self.t0)

        # Save state
        log = dict(
            step=step,
            model=self.raw_model.state_dict(),
            optimizers=[opt.state_dict() for opt in self.optimizers]
        )
        save_path = f'logs/state_step{step:06d}.pt'
        torch.save(log, save_path)

        if self.args.hf_model_name:
            try:
                self.model.module.push_to_hub(self.args.hf_model_name, subfolder=f'step{step:06d}')
            except Exception as e:
                print0(e)

        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def next_batch(self, train):
        seq = self.train_loader.next_batch() if train else self.valid_loader.next_batch()
        if not seq.numel():
            return None
        with torch.no_grad():
            self.train_tokens += (seq != self.tokenizer.pad_token_id).sum()
        return seq

    def train(self):
        self.train_tokens = 0

        self.training_time_ms = 0
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

        self.step = 0
        self.next_train_batch = self.next_batch(train=True)

        for self.step in range(self.args.num_steps + 1):
            last_step = (self.step == self.args.num_steps)
            if self.step == 10:
                self.training_time_ms = 0
                torch.cuda.synchronize()
                self.t0 = time.perf_counter()
            timed_steps = float('nan') if self.step <= 11 else (self.step - 10) + 1  # <= 11 to avoid bug in val

            if (self.args.valid_loss_every and self.step % self.args.valid_loss_every == 0) or last_step:
                self.validation_step(self.step, timed_steps)
            if master_process and self.args.save_every and (last_step or (self.step % self.args.save_every == 0)):
                self.save_step(self.step, timed_steps)
            if last_step:
                break
            self.train_step(self.step)

            # occasional runtime log
            if self.step % 100 == 0:
                approx_time = self.training_time_ms + 1000 * (time.perf_counter() - self.t0)
                print0(f'step:{self.step + 1}/{self.args.num_steps} train_time:{approx_time:.0f}ms '
                       f'step_avg:{approx_time / timed_steps:.2f}ms')

        # final logging
        dist.all_reduce(self.train_tokens, op=dist.ReduceOp.SUM)
        print0(f'Total train tokens: {self.train_tokens:,}')  # includes the next batch which isn't actually trained on

        print0(f'peak memory consumption training:\n{torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB')
        print0(f'Train Time: {self.training_time_ms:.0f}ms '
               f'| Step Avg: {self.training_time_ms / (timed_steps - 1):.2f}ms '
               f'| Param Count: {str(self.get_param_counts())}')
        print0(f'Total train time (min): {self.training_time_ms / 60000:.2f}')
        print0(f'Total train time (hours): {self.training_time_ms / 3600000:.2f}')

        if self.args.hf_model_name is not None:
            self.model.module.push_to_hub(self.args.hf_model_name)


def parse_args(dataclass_map=None):
    parser = argparse.ArgumentParser()
    dataclass_map = dataclass_map or {"train": TrainingArguments, "model": ModelConfig}

    def resolve_type(field):
        origin = get_origin(field.type)
        if origin is Union:
            args = get_args(field.type)
            non_none_types = [arg for arg in args if arg is not type(None)]  # Exclude NoneType
            if len(non_none_types) == 1:
                return non_none_types[0]
        return field.type

    # Dynamically add arguments for each dataclass
    for prefix, dataclass_type in dataclass_map.items():
        for field in dataclasses.fields(dataclass_type):
            arg_name = f"--{prefix}.{field.name}"
            arg_type = resolve_type(field)
            if field.default != dataclasses.MISSING:
                default = field.default
            elif field.default_factory != dataclasses.MISSING:  # Handle default_factory
                default = field.default_factory()
            else:
                default = None
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=default,
                help=f"{field.name} for {prefix} (type: {arg_type.__name__})"
            )
    args = parser.parse_args()
    result = {}
    for prefix, dataclass_type in dataclass_map.items():
        kwargs = {
            field.name: getattr(args, f"{prefix}.{field.name}")
            for field in dataclasses.fields(dataclass_type)
        }
        result[prefix] = dataclass_type(**kwargs)
    return result
