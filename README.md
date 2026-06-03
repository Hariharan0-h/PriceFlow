# PriceFlow — RL Dynamic Pricing

> Infotact Technical Internship — Advanced DS/ML | Project 2

## Setup

```bash
pip install -r requirements.txt
nbstripout --install          # pre-commit hook: strips notebook outputs
```

## Run training

```bash
python train.py
```

Checkpoint saved to `checkpoints/dqn_best.pt` (gitignored).

## Run tests

```bash
pytest tests/ -v
```

## Project structure

```
env/            # Custom Gymnasium environment + demand function
agents/         # Baselines, Q-learning, DQN agent + replay buffer
models/         # PyTorch network definitions
utils/          # Evaluation runner + plotting helpers
tests/          # pytest unit tests
notebooks/      # Analysis notebooks (outputs stripped by nbstripout)
train.py        # Re-train DQN from scratch
```

## Commit convention

```
feat:      new feature
model:     model/architecture change
analysis:  evaluation / experiment
viz:       plotting / dashboard
test:      unit tests
chore:     tooling, gitignore, README
```

Every commit must reference its issue: `(fixes #N)`
