import os
import sys

from dataclasses import dataclass, fields, MISSING
from typing import get_origin, get_args, Union, Optional

import argparse
import uuid
import time
import contextlib
from functools import partial
import math
import torch
import torch.distributed as dist
import torch._inductor.config as config
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

from optimizer import Muon
from model import ModelConfig, KBERT, CastedLinear
from dataloading import DistributedPaddedDataLoader


code = open(sys.argv[0]).read()
code += open('optimizer.py', 'r', encoding='utf-8').read()
code += open('model.py', 'r', encoding='utf-8').read()
code += open('dataloading.py', 'r', encoding='utf-8').read()


@dataclass
class TrainingArguments:
    # Data hyperparams
    input_bin: str = "data/fineweb-edu/fwedu_train_*.bin"
    input_valid_bin: str = "data/fineweb-edu/fwedu_valid_*.bin"
    input_test_bin: str = "data/fineweb-edu/fwedu_test_*.bin"
    # Optimization hyperparams
    batch_size: int = 8*64*1024
    grad_accum: int = 1
    num_steps: int = 20000
    warmup_steps: int = 200
    cooldown_steps: int = 2000
    max_length: int = 2**16
    # Evaluation and logging hyperparams
    valid_loss_every: int = 250
    hf_model_name: Optional[str] = None
    save_every: Optional[int] = None


def get_param_count(model):
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
    return total_params


def train(args, model_config):
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
        Path('logs').mkdir(exist_ok=True)
        # logdir = Path('logs') / f'{run_id}'
        # logdir.mkdir()
        logfile = Path('logs') / f'{run_id}.txt'
        print(logfile.stem)
        # create the log file
        with logfile.open('w') as f:
            # begin the log by printing this file (the Python code)
            print(code, file=f)
            print('=' * 100, file=f)

    def print0(s, logonly=False):
        if master_process:
            with logfile.open('a') as f:
                if not logonly:
                    print(s)
                print(s, file=f)

    # log information about the hardware/software environment this is running on
    # and print the full `nvidia-smi` to file
    print0(f'Running python {sys.version}')
    print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:')
    import subprocess
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print0(f'{result.stdout}', logonly=True)
    print0('='*100, logonly=True)

    print0(f'Model config: {model_config}')
    print0(f'Args: {args.__dict__}')

    # calculate the steps of gradient accumulation required to attain the desired global batch size
    # args.batch_size should refer to the total amount of tokens per backward pass
    train_accumulation_steps = 1
    batch_size = args.batch_size

    assert ddp_world_size == 1 or args.grad_accum == 1, 'Cannot currently use both DDP and gradient accumulation'
    if ddp_world_size > 1:
        train_accumulation_steps = ddp_world_size
        batch_size = args.batch_size // ddp_world_size
    elif args.grad_accum > 1:
        train_accumulation_steps *= args.grad_accum
        batch_size = args.batch_size // args.grad_accum

    print0(f'Train accumulation steps: {train_accumulation_steps}')
    print0(f'Adjusted local batch size: {batch_size} tokens')
    print0(f'Across {ddp_world_size} GPUs')
    print0(f'Total batch size: {args.batch_size} tokens')

    # load tokens
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    eos_id, pad_id = tokenizer.sep_token_id, tokenizer.pad_token_id
    train_loader = DistributedPaddedDataLoader(args.input_bin, batch_size, ddp_rank, ddp_world_size, eos_id=eos_id, pad_id=pad_id)
    valid_loader = DistributedPaddedDataLoader(args.input_valid_bin, batch_size, ddp_rank, ddp_world_size, eos_id=eos_id, pad_id=pad_id)
    test_loader = DistributedPaddedDataLoader(args.input_test_bin, batch_size // 8, ddp_rank, ddp_world_size, eos_id=eos_id, pad_id=pad_id)
    print0(f'Training DataLoader: {len(train_loader.files)} files')
    print0(f'Validation DataLoader: {len(valid_loader.files)} files')
    print0(f'Testing DataLoader: {len(test_loader.files)} files')
    print0('='*100, logonly=True)

    train_input_ids = train_loader.next_batch()

    model = KBERT(model_config)
    model = model.cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)

    # wrap model in DDP only if using distributed training
    if ddp_world_size > 1:
        model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        raw_model = model.module
    else:
        raw_model = model

    # init the optimizers
    embed_params = [*raw_model.embed.parameters(), *raw_model.value_embeds.parameters()]
    params = list(raw_model.blocks.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
    optimizer1 = torch.optim.Adam(embed_params, lr=0.3, betas=(0.8, 0.95), fused=True)
    optimizer2 = torch.optim.Adam([raw_model.lm_head.weight], lr=0.001, betas=(0.8, 0.95), fused=True)
    optimizer3 = Muon(matrix_params, lr=0.01, momentum=0.95)
    optimizer4 = torch.optim.Adam(scalar_params, lr=0.01, betas=(0.8, 0.95), fused=True)
    optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

    # learning rate decay scheduler (linear warmup and cooldown)
    def get_lr(it):
        assert it <= args.num_steps
        # 1) linear warmup for warmup_steps steps
        if it < args.warmup_steps:
            return (it+1) / args.warmup_steps
        # 2) constant lr for a while
        elif it < args.num_steps - args.cooldown_steps:
            return 1.0
        # 3) linear cooldown
        else:
            decay_ratio = (args.num_steps - it) / args.cooldown_steps
            return decay_ratio

    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

    # Linearly increase the sliding window size and MLM prob

    class LerpTensor:
        def __init__(self, start_val, end_val, precision):
            self.start, self.end, self.prec = start_val, end_val, precision
            self.prev_val = None
            dtype = torch.int32 if isinstance(precision, int) else torch.float
            self.gpu_val = torch.tensor(0, dtype=dtype, device="cuda")

        def __call__(self, frac_done):
            val = ((1 - frac_done) * self.start + frac_done * self.end) // self.prec * self.prec
            if val != self.prev_val:
                self.prev_val = val
                self.gpu_val.copy_(val, non_blocking=True)
            return self.gpu_val

    final_mask_prob = torch.tensor(0.12, device='cuda')
    final_keep_replace_prob = torch.tensor(0.015, device='cuda')

    lerp_mask_prob = LerpTensor(start_val=0.24, end_val=0.12, precision=0.01)
    lerp_keep_replace_prob = LerpTensor(start_val=0.24, end_val=0.015, precision=0.015)
    lerp_sw_size = LerpTensor(start_val=1024, end_val=args.max_length, precision=128)


    # Start training loop
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ### BEGIN TRAINING LOOP ###
    for step in range(args.num_steps + 1):
        last_step = (step == args.num_steps)
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
        # steps with dummy data first, and then re-initialize the model and reset the loader.
        # TODO
        # We should add this before the hackathon
        if step == 10:
            training_time_ms = 0
            t0 = time.perf_counter()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

        frac_done = step / args.num_steps  # training progress
        mlm_prob = lerp_mask_prob(frac_done)
        keep_replace_prob = lerp_keep_replace_prob(frac_done)
        sliding_window_size = lerp_sw_size(frac_done)

        # once in a while evaluate the validation dataset
        if args.valid_loss_every > 0 and step % args.valid_loss_every == 0 or last_step:
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            # run validation batches
            model.eval()
            valid_loader.reset()
            val_loss, valid_tokens = 0.0, 0
            with torch.no_grad():
                input_ids = valid_loader.next_batch()
                while input_ids.numel():
                    batch_valid_tokens = (input_ids != pad_id).sum()
                    valid_tokens += batch_valid_tokens
                    val_loss += model(input_ids, sliding_window_size, final_mask_prob, final_keep_replace_prob) * batch_valid_tokens
                    input_ids = valid_loader.next_batch()
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
            val_loss /= valid_tokens
            # log val loss to console and to logfile
            print0(f'step:{step}/{args.num_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms perplexity:{(math.e**val_loss):.4f} param_count:{get_param_count(model):,} tokens: {valid_tokens.item():,}')
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # save checkpoint every `save_every` steps
        if master_process and args.save_every:
            if last_step or (step % args.save_every == 0):
                # stop the clock
                torch.cuda.synchronize()
                training_time_ms += 1000 * (time.perf_counter() - t0)
                # save the state of the training process
                log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                torch.save(log, 'logs/state_step%06d.pt' % step)

                if args.hf_model_name:
                    try:
                        model.module.push_to_hub(args.hf_model_name, subfolder='step%06d' % step)
                    except Exception as e:
                        print0(e)

                torch.cuda.synchronize()
                t0 = time.perf_counter()

        if last_step:
            break

        # --------------- FORWARD AND BACKWARD PASS -----------------
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            with contextlib.ExitStack() as stack:
                if ddp_world_size > 1 and i < train_accumulation_steps: # there's no need to sync gradients every accumulation step
                    stack.enter_context(model.no_sync())
                #if step >= 5:
                #    stack.enter_context(torch.compiler.set_stance(skip_guard_eval_unsafe=True))
                model(train_input_ids, sliding_window_size, mlm_prob, keep_replace_prob).backward()
                train_input_ids = train_loader.next_batch()
        if train_accumulation_steps != 1:
            for p in model.parameters():
                p.grad /= train_accumulation_steps
        # momentum warmup for Muon
        frac = min(step/300, 1)
        for group in optimizer3.param_groups:
            group['momentum'] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- FORWARD AND BACKWARD PASS END -------------------
        # everything that follows now is just eval, diagnostics, prints, logging, etc.
        if step % 100 == 0:
            approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
            print0(f'step:{step+1}/{args.num_steps} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms')

    print0(f'peak memory consumption training: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB')

    # save the model to huggingface
    try:
        if ddp_world_size > 1:
            model.module.push_to_hub(args.hf_model_name)
        else:
            model.push_to_hub(args.hf_model_name)
    except Exception as e:
        print(e)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.manual_seed(42)
    model.eval()
    test_loader.reset()

    test_loss, test_tokens = 0.0, 0
    with torch.no_grad():
        input_ids = test_loader.next_batch()
        while input_ids.numel():
            batch_test_tokens = (input_ids != pad_id).sum()
            test_tokens += batch_test_tokens
            test_loss += model(input_ids, sliding_window_size, final_mask_prob, final_keep_replace_prob) * batch_test_tokens
            input_ids = test_loader.next_batch()
    if ddp_world_size > 1:
        dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(test_tokens, op=dist.ReduceOp.SUM)
    test_loss /= test_tokens

    original_test_loss = test_loss
    print0(f"Original test loss (regular forward pass): {original_test_loss:.4f}")

    test_loss, test_tokens = 0.0, 0
    all_logits, all_labels = [], []
    with torch.no_grad():
        input_ids = test_loader.next_batch()
        while input_ids.numel():
            batch_test_tokens = (input_ids != pad_id).sum()
            test_tokens += batch_test_tokens
            logits, loss, labels = model.inference(input_ids, sliding_window_size, final_mask_prob, final_keep_replace_prob)
            test_loss += loss * batch_test_tokens
            all_logits.extend(logits.detach().cpu().flatten().tolist())
            all_labels.extend(labels.detach().cpu().flatten().tolist())
            input_ids = test_loader.next_batch()
            if ddp_world_size > 1:
                dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(test_tokens, op=dist.ReduceOp.SUM)
    test_loss /= test_tokens

    import numpy as np
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    mask = (all_labels != -100)
    all_labels = all_labels[mask]
    all_logits = all_logits[mask]

    test_precision = precision_score(all_labels, all_logits, average='weighted')
    test_recall = recall_score(all_labels, all_logits, average='weighted')
    test_f1 = f1_score(all_labels, all_logits, average='weighted')
    test_accuracy = accuracy_score(all_labels, all_logits)
    test_mcc = matthews_corrcoef(all_labels, all_logits)

    print0(f"Test results (inference pass): {test_tokens.item()}")
    print0(f'Test tokens: {test_tokens.item()}')
    print0(f'Loss: {test_loss:.4f} | Perplexity: {math.e**test_loss:.4f}')
    print0(f'Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f} | Accuracy: {test_accuracy:.4f} | MCC: {test_mcc:.4f}')
    print0(f'Train Time: {training_time_ms:.0f}ms | Step Avg: {training_time_ms/(timed_steps-1):.2f}ms | Param Count: {get_param_count(model):,}')
    print0(f'Total train time (min): {training_time_ms / 60000:.2f}')
    print0(f'Total train time (hours): {training_time_ms / 3600000:.2f}')
    print0(f"peak memory consumption testing: {torch.cuda.max_memory_allocated() // 1024 // 1024 // 1024} GiB")
    # -------------------------------------------------------------------------
    # clean up nice
    if ddp_world_size > 1:
        dist.destroy_process_group()


default_dataclass_map = {"train": TrainingArguments, "model": ModelConfig}
def parse_args(dataclass_map=None):
    parser = argparse.ArgumentParser()
    dataclass_map = dataclass_map or default_dataclass_map

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
        for field in fields(dataclass_type):
            arg_name = f"--{prefix}.{field.name}"
            arg_type = resolve_type(field)
            if field.default != MISSING:
                default = field.default
            elif field.default_factory != MISSING:  # Handle default_factory
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
            for field in fields(dataclass_type)
        }
        result[prefix] = dataclass_type(**kwargs)
    return result


if __name__ == "__main__":
    cl_args = parse_args()
    train(args=cl_args["train"], model_config=cl_args["model"])
