KBERT (Keller BERT, based on Keller Jordans modded-nanogpt) is an open-source project seeking to produce the pareto frontier of BERT models. The objective is to produce a BERT variant which surpasses ModernBERT-base's GLUE score in as few FLOPs as possible.

Based on
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- [SpeedRunningESM2](https://github.com/Synthyra/SpeedRunningESM2)

## Quick Start

Setup environment and train KBERT

```
git clone https://github.com/lapp0/kbert
cd kbert
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20250103+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python data/download_fineweb_edu.py --num_chunks 30
torchrun --standalone --nproc_per_node=8 trainer.py
```

##### Push to Huggingaface Hub While Training

1) Login once to save credentials via `huggingface-cli login` (only run once)
2) Specify your own HF model name for training:
```
torchrun --standalone --nproc_per_node=8 trainer.py --train.hf_model_name lapp0/kbert_trial
```


## Benchmarks to match
TODO

## Successful runs showcase

|~Matches |Parameters|Time      |Hardware |Log | Val loss | Test loss |
|--------|----------|----------|---------|----|-----------|-----------|
TODO
