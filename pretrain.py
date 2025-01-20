from typing import Optional
from dataclasses import dataclass

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from base_trainer import BaseTrainer, TrainingArguments, parse_args
from model import KBERTForMaskedLM, ModelConfig


class LerpTensor:
    def __init__(self, start_val, end_val, precision):
        self.start, self.end, self.prec = start_val, end_val, precision
        self.prev_val = None
        dtype = torch.int32 if isinstance(precision, int) else torch.float
        self.gpu_val = torch.tensor(0, dtype=dtype, device="cuda")

    def __call__(self, frac_done):
        val = ((1 - frac_done) * self.start + frac_done * self.end) // self.prec * self.prec
        if val != self.prev_val:
            self.gpu_val.copy_(val, non_blocking=True)
            self.prev_val = val
        return self.gpu_val


class MLMTrainer(BaseTrainer):
    dataloader_allow_windowed = True

    def __init__(self, args, model, tokenizer):
        super().__init__(args, model, tokenizer)

        # Only for MLM
        self.lerp_mask_prob = LerpTensor(0.2, 0.12, 0.01)
        self.lerp_keep_replace_prob = LerpTensor(0.09, 0.015, 0.0075)
        self.lerp_sw_size = LerpTensor(1024, self.args.max_length, 128)

        self.final_mask_prob = torch.tensor(0.12, device='cuda')
        self.final_keep_replace_prob = torch.tensor(0.015, device='cuda')

    def next_batch(self, train):
        seq = super().next_batch(train)
        if seq is None:
            return None

        frac_done = self.step / self.args.num_steps
        sw_size = self.lerp_sw_size(frac_done)
        mask_prob = self.lerp_mask_prob(frac_done) if train else self.final_mask_prob
        keep_replace_prob = self.lerp_keep_replace_prob(frac_done) if train else self.final_keep_replace_prob

        return {
            "input_ids": seq,
            "labels": seq.clone(),
            "sliding_window_size": sw_size,
            "mask_prob": mask_prob,
            "keep_replace_prob": keep_replace_prob
        }

@dataclass
class PretrainingArguments(TrainingArguments):
    # Data hyperparams
    input_bin: str = "data/fineweb-edu/fwedu_train_*.bin"
    input_valid_bin: str = "data/fineweb-edu/fwedu_valid_*.bin"

    # Optimization hyperparams
    batch_size: int = 4 * 64 * 1024 * 3 // 2
    grad_accum_per_device: int = 4
    num_steps: int = 50_000
    warmup_steps: int = 2_000
    cooldown_steps: int = 40_000
    max_length: int = 8_192
    max_epochs: int = None
    valid_loss_every: int = 1000

    # adam
    lr_head: Optional[float] = None
    lr_embed: float = 0.006
    lr_scalar: float = 0.003
    # muon
    lr_hidden: float = 0.003
    muon_momentum_warmup_steps: int = 300  # steps for warmup momentum, 0.85 -> 0.95

    hf_model_name: Optional[str] = "lapp0/kbert_base"


if __name__ == "__main__":
    cl_args = parse_args({"train": PretrainingArguments, "model": ModelConfig})
    model_config = cl_args["model"]

    try:
        trainer = MLMTrainer(
            args=cl_args["train"],
            model=KBERTForMaskedLM(model_config),
            tokenizer=AutoTokenizer.from_pretrained(model_config.tokenizer_uri)
        )
        trainer.train()
    finally:
        dist.destroy_process_group()
