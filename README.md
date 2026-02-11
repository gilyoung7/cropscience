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

python -m rice.scripts.run_split_seed --pest sbw --run 0 --auto_split_seed --auto_split_topk 3
python -m rice.scripts.run_train --pest sbw --run 0 --split_seeds_json outputs/sbw/splits/selected_split_seeds.json --split_seed_from_topk_idx 0
python -m rice.scripts.run_eval --pest sbw --run 0 --split_seeds_json outputs/sbw/splits/selected_split_seeds.json --split_seed_from_topk_idx 0
