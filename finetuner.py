from dataclasses import dataclass

from trainer import parse_args, train, TrainingArguments
from transformers import AutoTokenizer
from model import KBERTForSequenceClassification, SequenceClassificationModelConfig


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


# TODO: Implement from_pretrained for base_model


if __name__ == "__main__":
    cl_args = parse_args({"train": FinetuneMNLIArguments, "model": MNLISequenceClassificationModelConfig})
    training_args = cl_args["train"]
    model_config = cl_args["model"]

    train(
        args=training_args,
        model=KBERTForSequenceClassification.from_pretrained(
            training_args.base_model, config=model_config
        ),
        tokenizer=AutoTokenizer.from_pretrained(model_config.tokenizer_uri)
    )