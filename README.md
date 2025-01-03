KBERT (Keller Bert) is an open-source collaboration to produce the pareto frontier of BERT models, based on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and [SpeedRunningESM2](https://github.com/Synthyra/SpeedRunningESM2).

## Quick Start

Setup environment and train ESM2

```
git clone https://github.com/lapp0/kbert
cd kbert
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20241204+cu124 torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python data/download_fineweb_edu.py --num_chunks 10
torchrun --standalone --nproc_per_node=8 trainer.py
```
torchvision is needed to fix an import error with transformers.

## Benchmarks to match
TODO

## Successful runs showcase

|~Matches |Parameters|Time      |Hardware |Log | Val loss | Test loss |
|--------|----------|----------|---------|----|-----------|-----------|
TODO
