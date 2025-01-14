from dataclasses import dataclass
import math
import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from trainer_base import BaseTrainer, TrainingArguments, parse_args, print0
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
            "labels": labels
        }

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
                val_inputs, val_labels = self.next_batch(train=False)

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
        val_loss /= valid_tokens

        print0(
            f'step:{step}/{self.args.num_steps} val_loss:{val_loss:.4f} '
            f'train_time:{self.training_time_ms:.0f}ms '
            f'step_avg:{self.training_time_ms/(timed_steps-1):.2f}ms '
            f'perplexity:{(math.e**val_loss):.4f} '
            f'param_count:{str(self.get_param_counts())} '
            f'tokens: {valid_tokens.item():,}'
        )
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()


@dataclass
class FinetuneMNLIArguments(TrainingArguments):
    base_model: str = "lapp0/kbert_trial0"

    objective: str = "seq_classification"
    input_bin: str = "data/mnli/mnli_train_*.bin"
    input_valid_bin: str = "data/mnli/mnli_validation_*.bin"

    lr_head: float = 0.01
    lr_embed: float = 0.01 / 4
    lr_scalar: float = 0.005 / 4
    lr_hidden: float = 0.005 / 4

    batch_size: int = 8 * 1024
    num_steps: int = 2_500
    cooldown_steps: int = 2_000
    warmup_steps: int = 100

    valid_loss_every: int = 10
    hf_model_name: str = "lapp0/kbert_finetuned_mlni"



@dataclass
class MNLISequenceClassificationModelConfig(SequenceClassificationModelConfig):
    num_labels: int = 3


if __name__ == "__main__":
    cl_args = parse_args({"train": FinetuneMNLIArguments, "model": MNLISequenceClassificationModelConfig})
    training_args = cl_args["train"]
    model_config = cl_args["model"]

    #trainer = ...
    #try:
        #trainer.train(...)
    #finally:
        #dist.destroy_process_group()
    #train(
    #    args=training_args,
    #    model=KBERTForSequenceClassification.from_pretrained(
    #        training_args.base_model, config=model_config
    #    ),
    #    tokenizer=AutoTokenizer.from_pretrained(model_config.tokenizer_uri)
    #)
