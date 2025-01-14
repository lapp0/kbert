**KBERT** (Keller BERT) is an open-source project for training the pareto frontier of transformer encoder models.

Specifically, the goal is to produce a model which beats ModernBERT and DeBERTaV3 on both SQuAD v2.0 and MNLI with minimal compute.

#### Related Works

- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt): Introduces substantial architectural and performance improvements to gpt2 training.
- [SpeedRunningESM2](https://github.com/Synthyra/SpeedRunningESM2): Adapts modded-nanogpt to encoder model for protein structure prediction.

## Quick Start

Setup environment, dependencies, and data
```
git clone https://github.com/lapp0/kbert && cd kbert
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20250103+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade

python data/download_fineweb_edu.py --num_chunks 120  # ~100M tokens / chunk
python data/download_mnli.py

export N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
```

Save HF credentials (only need to run once):
```
huggingface-cli login
```

#### Pretrain KBERT on Fineweb EDU with MLM objective
```
torchrun --standalone --nproc_per_node=$N_GPU pretrain.py \
  --train.hf_model_name PRETRAINED_HF_URI
```

#### Finetune KBERT on MNLI with sequence classification objective
```
torchrun --standalone --nproc_per_node=$N_GPU finetune.py \
  --train.base_model PRETRAINED_HF_URI
  --train.hf_model_name FINETUNED_HF_URI
```

## Benchmarks
|                      | KBERT-base | [DeBERTa-v3-base](https://arxiv.org/abs/2111.09543) | [ModernBERT-base](https://arxiv.org/abs/2412.13663) |
|----------------------|------------|-----------------------------------------------------|-----------------------------------------------------|
| Training Tokens      | ?          | 400 billion*                                        | 1.7 trillion                                        |
| **Metrics**          |            |                                                     |                                                     |
| MNLI                 | ?          | ?                                                   | ?                                                   |
| SQuAD v2.0           | ?          | ?                                                   | ?                                                   |
| **Parameters**       | 150M       | 185M                                                | **150M**                                            |
| Encoder Parameters   | 106M       | 87M                                                 | 111M                                                |
| Embedding Parameters | 39M        | 98M                                                 | 39M                                                 |

 *Estimate is based on papers stated "160GB of data" @ 10 epochs, (estimated 4 bytes per token)

