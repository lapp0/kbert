KBERT (Keller BERT, based on Keller Jordans modded-nanogpt) is an open-source project seeking to produce the pareto frontier of BERT models. The objective is to produce a BERT variant which surpasses ModernBERT-base's GLUE score in as few FLOPs as possible.

Based on
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [SpeedRunningESM2](https://github.com/Synthyra/SpeedRunningESM2)

## Quick Start

Setup environment

```
git clone https://github.com/lapp0/kbert
cd kbert
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20250103+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade

python data/download_fineweb_edu.py --num_chunks 30
python data/download_mnli.py
```

```
export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
```

#### Pretrain KBERT on Fineweb EDU with MLM objective
```
torchrun --standalone --nproc_per_node=$NUM_GPUS trainer.py
```

#### Finetune KBERT on MNLI with sequence classification objective
```
torchrun --standalone --nproc_per_node=$NUM_GPUS finetuner.py
```


### Push to Huggingface Hub While Training

1) Login to save credentials via `huggingface-cli login` (only need to run once)
2) Specify your own HF model URI for training:
```
torchrun --standalone --nproc_per_node=$NUM_GPUS trainer.py --train.hf_model_name lapp0/kbert_trial
```


## Benchmarks to match
TODO

## Successful runs showcase

|~Matches |Parameters|Time      |Hardware |Log | Val loss | Test loss |
|--------|----------|----------|---------|----|-----------|-----------|
TODO
