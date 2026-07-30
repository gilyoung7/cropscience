# All-pest E5d run

Trains and evaluates the WBPH-best **E5d** recipe on every pest the code supports, across two
GPU servers, producing both the **dev** and the **clean** protocol numbers.

E5d = shared band-causal time encoder + per-offset independent mu heads (12) + onset (`l_offset`)
target + 25x late penalty + **very-early penalty removed** (`asym_weight_early: 5 -> 0`), sigma 8,
coverage-aware LightGBM selector, plus a fold-wise post-hoc mu shift.

## Pests

8 trainable pests (`pests.tsv`). **BPH2 is excluded**: the slug loads, but it has no Stage-2
batch data in any year (no dispatch CSV, no climatology, no `lead_v3` ckpt) and only 26 test
events on a 2020-2024 window.

| server | pests | cost* |
|---|---|---|
| 1 | WBPH, rice_stem_borer_2, rice_stem_borer_1, brown_spot | ~500 |
| 2 | sheath_blight, BPH, blast, bacterial_blight | ~448 |

\* `events / batch_train`. Batch 16 (WBPH, BPH, rsb1, rsb2) costs ~4x the steps of batch 64 for
the same data, so balancing on event count alone would be badly skewed.

**WBPH is the positive control.** It runs through the identical new code path, so its dev and
clean numbers must reproduce the known 0.3956 / 0.309865. If they do not, the port is wrong and
no other pest's number should be trusted.

## Order of operations

```bash
# 0. pre-flight (do this first; it is the cheap thing that prevents 48 wasted GPU-hours)
cd <cropscience>
.venv/bin/python rice/experiments/allpests_e5d/vendor_check.py
.venv/bin/python rice/experiments/allpests_e5d/dry_run.py --level full

# 1. server 1
bash rice/experiments/allpests_e5d/run_server.sh 1
# 1'. server 2 (other machine, same repo state)
bash rice/experiments/allpests_e5d/run_server.sh 2
```

`run_server.sh` re-runs the fast dry run itself and refuses to start on any FAIL.

## Roots and environment

Nothing here hard-codes a machine. `repo_paths.py` (Python) and `common.sh` (bash) both derive
the repo root from their own file location — this directory is always
`<root>/rice/experiments/allpests_e5d` — so a checkout under any prefix runs unmodified.

| variable | meaning | default |
|---|---|---|
| `CROPSCIENCE_ROOT` | repo root override | derived from the script's own path |
| `WBPH_WS_ROOT` | external `wbph_interval_perf_202607` workspace | sibling of the repo root |
| `ALLPESTS_OUT_ROOT` | results root | `<root>/rice/outputs_allpests_e5d` |
| `RICE_DAILY_CSV` | daily union table (`rice/configs/base.py`) | `/home/gpu4080/ygdata/rice/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv` |
| `RICE_LONG_BY_PEST_DIR` | per-pest observation CSV directory | `/home/gpu4080/ygdata/rice/LONG_by_pest` |

The last two point at data that lives outside the repo, so they cannot be derived from the repo
root. Only the *directory* is relocatable for `LONG_by_pest` — each pest config still owns its
own `RICE_LONG_<korean name>.csv` filename. `BPH2` is the one pest whose observation file sits
outside `LONG_by_pest`; it is excluded from this run and was left hard-coded.

A host that does not mirror the first server's `/home/gpu4080/ygdata` layout sets both to
wherever its own copy of the data lives — no code change and no edit to this file:

```bash
export RICE_LONG_BY_PEST_DIR=<DATA_ROOT>/LONG_by_pest
export RICE_DAILY_CSV=<DATA_ROOT>/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv
```

The external workspace is **optional** and never imported; only `capacity_report.py` and
`smoke_wandb_wbph2024.py` read WBPH's published result CSVs from it, and both guard on
existence. `repo_paths.py` refuses to guess: if the derived root does not contain `rice/configs`,
`rice/pests` and `rice/src/pest_resolver.py` it exits with instructions rather than proceeding
against the wrong tree.

Interpreter: `.venv` at the repo root, built from `requirements.txt` in this directory. The
Torch build is pinned to **cu118** because a 525.60.x driver does not meet the 525.60.13 floor
of the cu121 wheels; check `nvidia-smi` on both servers before changing it.

```bash
python3 -m venv .venv
.venv/bin/pip install -r rice/experiments/allpests_e5d/requirements.txt
```

## Per-pest chain (`run_pest.sh`)

| # | step | writes | marker |
|---|---|---|---|
| 1 | `train_dev` | `dev/ckpt/<year>/` | `.done_train_dev` |
| 2 | `grid_dev` | `dev/grid/<pest>_E5d_grid_1to75.csv` | `.done_grid_dev` |
| 3 | `splits` | `clean/split_assignment.json` | `.done_splits` |
| 4 | `train_clean` | `clean/ckpt/<year>/` | `.done_train_clean` |
| 5 | `grid_clean` | `clean/grid/<pest>_E5d_clean_grid_1to75.csv` | `.done_grid_clean` |
| 6 | `eval` | `eval/{dev,clean}_*` | `.done_eval` |

Dev must finish before clean: step 3 enumerates the selection cohort from the dev grid.

**Resume** is automatic — every step is skipped when its marker exists, and training
additionally skips any year whose `checkpoint_run4.pt` is already on disk. Kill and relaunch
freely. `FORCE=1` ignores all markers.

## Reproducibility across the two servers

Nothing is exchanged between servers, and nothing needs to be:

- **Hyperparameters** are not written by us. `common.sh:extract_args` greps each cell's own
  production `stage2_lead_v3_train.log` and sed-patches only the two E5d mu-loss knobs. They
  genuinely differ per (pest, year) — the Stage-1 gate CSV is one of three variants chosen per
  cell, WBPH/BPH/rsb1/rsb2 carry `--batch_train_override 16` and the diseases do not, and 2024
  dropped `--stage2_gaussian_loss_mode`. Substituting a sibling's args would silently change the
  model.
- **Seed** `--split_seed 42` and `--seeds 0` come from those same commands; the 5 evaluation
  seeds are fixed in `pest_paths.SEEDS`.
- **Selection splits** are a pure hash of `sample_id` (`md5("42:"+sid)` for clean 40/30/30,
  `md5("dev50:"+sid)` for the dev half) — both servers derive identical partitions with no
  shared file. The two salts differ deliberately so dev and clean do not share a fit set.
- **Data version** is the shared read-only cache under `rice/outputs/cache/`; the cache key
  contains no pest term, so all pests read the same daily preprocessing.

## Uniformity — what is and is not held constant

The pests do **not** all run on one common DOY window. Each cell inherits its own production
geometry and we deliberately do not pass `--doy_start_override` / `--doy_end_override`.
`dry_run.py` check [7] audits this across all 24 cells every run.

**Identical across all 24 cells** (a FAIL if any drifts):

| knob | value |
|---|---|
| `d_model_override` | 48 |
| `stage2_pmf_sigma` | 5.0 (training; evaluation uses sigma 8) |
| `stage2_pmf_mu_mode` | `lead_from_alert` |
| `stage2_pmf_target_mode` | `l_offset` (onset) |
| `stage2_pmf_asym_weight` / `_early` | 25.0 / 0.0 — the E5d cell |
| `stage2_nowcast_window` | 28 |
| `split_mode` / `split_seed` | `year` / 42 |

Also identical: architecture env (shared encoder + offset-specific heads), the 12 candidate
offsets, the 5 evaluation seeds, the `train <= y-2` chronology, and both selection-split hashes.

**Varies per cell, by design** (reported as WARN, never overridden):

- **DOY geometry** — BPH is 140-270 (T=131); the other seven are 60-300 (T=241).
- **Feature width** — BPH 27 -> 33 channels; the others 45 -> 51.
- **Stage-1 gate CSV** — one of `gate_{A_baseline,D_history,dispatch_group_tau}_R088_*`,
  chosen per (pest, year) by the original gate selection.
- **`batch_train_override`** — 16 for WBPH in all three years and for BPH / rsb1 / rsb2 in 2024
  only; absent (trainer default 64) elsewhere.
- **2024** dropped `--stage2_gaussian_loss_mode`, `--stage2_gaussian_interval_lambda` and
  `--stage2_pmf_delta_max` relative to 2022/2023.

**Why not force a common window.** `alert_tstar` in the gate CSV is an absolute DOY produced by
a Stage-1 model that was itself fit in that pest's own season frame, and every candidate offset
is `issue - alert`. Overriding Stage-2's geometry while Stage-1 stays frozen would score the
model in a frame its alerts were never built for. It would also stop the run being "the WBPH E5d
recipe applied unchanged", which is the entire point of the sweep. There is precedent for a
common window — `run_stage2_phase5d_7pests.sh:32` forces 1-300 — but that is a different
lineage (`split_mode=site_year`, per-pest seeds), not the `lead_v3` production line E5d inherits.

**Consequence:** BPH's IoU is not directly comparable to the other seven. Report it in its own
row, do not rank it against them, and do not pool it into a cross-pest mean.

## Raw artifacts (both protocols computable later)

`dev/grid/*.csv` and `clean/grid/*.csv` are the offsets-1..75 candidate grids — every row the
selector could ever choose, with `mu`, `L`, `R`, `alert_tstar`, `iou80`. Both evaluations are
computed from these, so any re-scoring (different sigma, different shift grid, a third protocol)
needs no retraining. Do not delete them.

## Known issues

- **blast/2023** has a 0-byte training log (the run happened — the ckpt exists — the log was
  never captured). `extract_args` falls back to blast/2022's command and re-points the three
  year fields plus the dispatch CSV. This is valid only because blast's 2023 `hparams.json`
  matches its 2022 one on all 97 non-year keys; `dry_run.py` check [4] re-asserts that equality
  every run, so the fallback cannot rot silently.
- **BPH geometry** is doy 140-270, T=131, against 60-300 / T=241 for the other seven. Anchors
  1..75 and candidate offsets up to 60 are therefore far more sparsely feasible; expect fewer
  grid rows and treat BPH's IoU as not directly comparable to the rest. `dry_run.py` check [5]
  emits this as a WARN, not a FAIL.
- **rice_stem_borer_1/2** have no `direct_neighbor` production track, so the
  `--stage2_add_neighbor_history` channels have never been validated for them. They are also the
  sparsest pests (698 / 1029 event rows against sheath_blight's 15446). Low-confidence results
  are expected, not a bug.
- The dev half-split is **not** the same partition the WBPH 2x2 used; WBPH's published dev
  number came from `scripts/95`. Small deviation from 0.3956 is expected on that basis alone —
  the clean number is the strict reproduction target.

## Vendored dependencies (`vendor/`)

Both servers execute **only** the code under `rice/experiments/allpests_e5d/vendor/`. There is
no fallback to the external `wbph_interval_perf_202607` workspace — a fallback is precisely how
two machines end up silently running different code. Server 2 does not need that workspace at
all.

`VENDOR_MANIFEST.csv` pins all 10 files by sha256, size, original absolute path and vendoring
timestamp. `vendor_check.py` re-hashes them; `dry_run.py` check [12] turns any missing file or
hash mismatch into a FAIL, which stops training.

| file | purpose |
|---|---|
| `scripts/87_make_shared_offset_grid.py` | `collect()` — the offsets-1..75 grid forward |
| `src/{io_utils,selector_utils,diagnostics,eval_metrics}.py` | candidate build, coverage-aware selector, metrics |
| `src/vendor/{model,run_train}.py` | E5d model + Stage-2 trainer |
| `_patched_train.py` | clean-fold training (val restricted to val_ckpt) |

**`87_make_shared_offset_grid.py` is deliberately pinned to the 2026-07-29 revision**, which
adds the module-level `PEST` global so one `collect()` can serve all 8 pests. Its default is
`"WBPH"`, so every pre-existing WBPH result stays bit-identical. This is knowingly *not* the
revision that produced the original WBPH numbers — it is behaviourally identical for
`pest=WBPH`, and the manifest's `note` column records the same caveat.

If the external workspace happens to exist, `vendor_check.py` reports whether it has drifted
from the pin. That is **informational only**: the pinned copy is authoritative and the external
copy is never imported.

Results go to `$ALLPESTS_OUT_ROOT` (default `rice/outputs_allpests_e5d/`), also independent of
the external workspace.
