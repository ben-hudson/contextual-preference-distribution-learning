# contextual-preference-distribution-learning
Code to accompany the paper [Contextual Preference Preference Distribution Learning](https://arxiv.org/abs/2603.17139), published in CPAIOR 2026 proceedings.

To use the code as close as possible to how it was in the camera-ready submission, use tag `cpaior` for all models except MaxEnt IRL. Use tag `cpaior-maxent-irl` for MaxEnt IRL.

## Setup
Create a virtual environment (we used Python 3.10.11 and venv) and run
```
pip install -e .
```

## Example usage
Run CPDL with the parameters in the paper:
```
python scripts/cpaior_experiment.py --model=ours --model_samples=300 --lr_start=0.0025 --lr_sched_patience=15 --lr_sched_rel_thresh=0.0331 --max_epochs=200
```

Run the risk-neutral oracle:
```
python scripts/cpaior_experiment.py --model oracle --policy risk-neutral
```
