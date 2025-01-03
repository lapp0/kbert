KBERT (Keller Bert) is an open-source collaboration to produce the pareto frontier of BERT models, based on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and [SpeedRunningESM2](https://github.com/Synthyra/SpeedRunningESM2).

## Quick Start

Setup environment and train KBERT

```
git clone https://github.com/lapp0/kbert
cd kbert
pip install -r requirements.txt
pip install --pre torch==2.6.0.dev20250103+cu124 torchvision torchaudio--index-url https://download.pytorch.org/whl/nightly/cu124 --upgrade
python data/download_fineweb_edu.py --num_chunks 10
torchrun --standalone --nproc_per_node=8 trainer.py
```

##### Push to Huggingaface Hub While Training
Only need to save your hub token once:
```
HF_TOKEN=<YOUR HUGGINGFACE TOKEN> python3 -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('${HF_TOKEN}')"
```
Specify your own HF model name
```
torchrun --standalone --nproc_per_node=8 trainer.py --train.hf_model_name lapp0/kbert_trial
```


## Benchmarks to match
TODO

## Successful runs showcase

|~Matches |Parameters|Time      |Hardware |Log | Val loss | Test loss |
|--------|----------|----------|---------|----|-----------|-----------|
TODO
