KBERT (Keller BERT, based on Keller Jordans modded-nanogpt) is an open-source project seeking to produce the pareto frontier of BERT models. The objective is to produce a BERT variant which surpasses ModernBERT-base's GLUE score in as few FLOPs as possible.

Based on
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [SpeedRunningESM2](https://github.com/Synthyra/SpeedRunningESM2)

## Quick Start

Setup environment, dependencies, and data
```
git clone https://github.com/lapp0/kbert && cd kbert
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20250103+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade

python data/download_fineweb_edu.py --num_chunks 120  # ~100M tokens / chunk
python data/download_mnli.py

export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
```

Save HF credentials (only need to run once):
```
huggingface-cli login
```

#### Pretrain KBERT on Fineweb EDU with MLM objective
```
torchrun --standalone --nproc_per_node=$NUM_GPUS trainer.py --train.hf_model_name HUB_MODEL_URI_HERE
```

#### Finetune KBERT on MNLI with sequence classification objective
```
torchrun --standalone --nproc_per_node=$NUM_GPUS finetuner.py --train.hf_model_name HUB_MODEL_URI_HERE
```


### Push to Huggingface Hub While Training

1)
2) Specify your own HF model URI for training:
```
torchrun --standalone --nproc_per_node=$NUM_GPUS trainer.py
```


## Benchmarks to match
|                      | KBERT | [DeBERTa-v3-base](https://arxiv.org/abs/2111.09543) | [ModernBERT-base](https://arxiv.org/abs/2412.13663) |
|----------------------|-------|-----------------------------------------------------|-----------------------------------------------------|
| Training Tokens      | ?     | 800 billion*                                        | 1.7 trillion                                        |
| **Metrics**          |       |                                                     |                                                     |
| MNLI                 | ?     | ?                                                   | ?                                                   |
| SQuAD v2.0           | ?     | ?                                                   | ?                                                   |
| **Parameters**       | ?     | 185M                                                | 150M                                                |
| Encoder Parameters   | ?     | 87M                                                 | 111M                                                |
| Embedding Parameters | ?     | 98M                                                 | 39M                                                 |

 *Estimate is based on papers stated 160GB of data @ 10 epochs

|~Matches |Parameters|Time      |Hardware |Log | Val loss | Test loss |
|--------|----------|----------|---------|----|-----------|-----------|
TODO
