# Agro Pest Forecasting

Notebook-driven experiments for pest/disease forecasting.

## Structure
- notebooks/: experiment notebooks
- src/: reusable code (TBD)
- scripts/: run scripts (TBD)
- configs/: configs (TBD)

## Data
Large data/artifacts are kept outside the repo.

## Examples
python -m scripts.run_train --run 5 --out_root outputs | tee outputs/logs/train_run5.log

python -m scripts.run_eval --run 5 --out_root outputs

python -m scripts.run_pi --run 5 --out_root outputs

python -m scripts.run_sfs --run 5 --out_root outputs
