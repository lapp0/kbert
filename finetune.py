from dataclasses import dataclass
import math
import time
import typing

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from base_trainer import BaseTrainer, TrainingArguments, parse_args, print0
from model import KBERTForSequenceClassification, SequenceClassificationModelConfig


class SeqClassificationTrainer(BaseTrainer):
    dataloader_allow_windowed = False

    def next_batch(self, train):
        seq = super().next_batch(train)
        if seq is None:
            return None

        # Convert to sequence classification format by extracting class (pos 1) and removing format code (pos 2)
        bos_idxs = (seq == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
        labels = seq[bos_idxs + 1].unsqueeze(0)

        assert seq[0].item() == self.tokenizer.cls_token_id, seq
        assert torch.all(seq[bos_idxs + 2] == torch.iinfo(torch.uint16).max)

        # Given sequences of <bos><label><uint16 max><tok><tok>..., remove <label><uint16>
        mask = torch.ones_like(seq, dtype=torch.bool)
        mask[bos_idxs + 1] = False
        mask[bos_idxs + 2] = False
        removed_count = (~mask).sum().item()
        seq = torch.cat([seq[mask], torch.full_like(seq[:removed_count], self.tokenizer.pad_token_id)])

        return {
            "input_ids": seq,
            "labels": labels.long()
        }

    @torch.no_grad()
    def validation_step(self, step, timed_steps):
        torch.cuda.synchronize()
        self.training_time_ms += 1000 * (time.perf_counter() - self.t0)
        self.model.eval()
        self.valid_loader.reset()
        val_loss = torch.tensor(0.0, device="cuda")
        valid_tokens = torch.tensor(0, device="cuda")
        num_seqs = torch.tensor(0, device="cuda")
        accuracy = torch.tensor(0.0, device="cuda")

        val_batch = self.next_batch(train=False)
        while val_batch is not None:
            val_inputs = val_batch["input_ids"]
            num_val_tokens = (val_inputs != self.tokenizer.pad_token_id).sum()
            valid_tokens += num_val_tokens
            batch_loss, logits = self.model(**val_batch, return_logits=True)
            val_loss += batch_loss * num_val_tokens
            accuracy += (logits.argmax(dim=-1) == val_batch["labels"]).float().sum()
            num_seqs += val_batch["labels"].numel()
            val_batch = self.next_batch(train=False)

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        val_loss /= valid_tokens

        dist.all_reduce(num_seqs, op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
        accuracy /= num_seqs

        print0(
            f'step:{step}/{self.args.num_steps} val_loss:{val_loss:.4f} '
            f'train_time:{self.training_time_ms:.0f}ms '
            f'step_avg:{self.training_time_ms / (timed_steps - 1):.2f}ms '
            f'perplexity:{(math.e**val_loss):.4f} '
            f'param_count:{str(self.get_param_counts())} '
            f'tokens: {valid_tokens.item():,} '
            f"validation accuracy: {accuracy * 100:.2f}%"
        )
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()


@dataclass
class FinetuneMNLIArguments(TrainingArguments):
    base_model: str = "lapp0/kbert_base"

    objective: str = "seq_classification"
    input_bin: str = "data/mnli/mnli_train_*.bin"
    input_valid_bin: str = "data/mnli/mnli_validation_*.bin"

    lr_head: float = 2e-2
    lr_embed: float = 1e-4
    lr_scalar: float = 3e-4
    lr_hidden: float = 3e-4

    batch_size: int = 4 * 1024
    grad_accum_per_device: int = 1

    num_steps: int = 2000
    cooldown_steps: int = 1200
    warmup_steps: int = 100

    valid_loss_every: int = 100
    hf_model_name: str = "lapp0/kbert_finetuned_mlni"



@dataclass
class MNLISequenceClassificationModelConfig(SequenceClassificationModelConfig):
    num_labels: int = 3
    softcap: float = None
    head_dropout: float = 0.5
    label_smoothing: float = 0.00

    # dataset is balanced, no class weights applied here
    # >>> ds = datasets.load_dataset("nyu-mll/glue", "mnli")
    # >>> collections.Counter(ds["train"]["label"])
    # Counter({2: 130903, 1: 130900, 0: 130899})
    class_weights: typing.Optional[typing.List[float]] = None


if __name__ == "__main__":
    cl_args = parse_args({"train": FinetuneMNLIArguments, "model": MNLISequenceClassificationModelConfig})
    training_args = cl_args["train"]
    model_config = cl_args["model"]
    try:
        trainer = SeqClassificationTrainer(
            args=training_args,
            model=KBERTForSequenceClassification.from_pretrained(training_args.base_model, config=model_config),
            tokenizer=AutoTokenizer.from_pretrained(model_config.tokenizer_uri)
        )
        trainer.train()
    finally:
        dist.destroy_process_group()
