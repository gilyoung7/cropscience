# Multi-pest E5d pretraining → pest-specific fine-tuning

Experiment **B**. Pretrain one E5d backbone on several pests' pooled data, then use it as the
initialization for each pest's own E5d model. Compared against experiment **A**, the existing
per-pest scratch run in `rice/outputs_allpests_e5d/`.

**The only intended difference between A and B is which tensors are non-random at step 0.**
Everything else — architecture, loss, `asym_weight=25 / _early=0`, `pmf_mode=gaussian`,
`mu_mode=lead_from_alert`, optimizer, lr, patience, max_epochs, seed, offset grid, selector,
evaluation code — is held identical. Any second difference makes the comparison uninterpretable,
so this directory is written to avoid introducing one.

This is **not** the pilot→final curriculum. That varies loss difficulty within one pest. This
varies *what the weights saw first*, across pests.

## Status

Design + skeleton + a structural smoke test. **No real-data training has run**, because the
`LONG_by_pest` observation CSVs are absent on this host — see [Blockers](#blockers).

## 1. Existing E5d backbone / head structure

`vendor/src/vendor/model.py`, class `HierarchicalCausalHazardTransformer`. With the E5d config
(`use_shared_multi_offset=True`, `mu_head_mode="offset_specific"`, `pmf_mode="gaussian"`) the
148,910 parameters split cleanly:

| group | keys | params | where | shape depends on |
|---|---|---|---|---|
| `in_proj.*` | 2 | 2,208 | `model.py:139` | **`d_in`** and `d_model` |
| `time_encoder.*` | 36 | 84,816 | `model.py:149` | `d_model` only |
| `offset_mu_heads.*` | 48 | 28,812 | `model.py:230` (ModuleList of 12) | `d_model` only |
| `head.*`, `head_mu.*`, `tstar_encoder.*`, `tstar_pos.pe` | 21 | 57,650 | `169`, `205`, `160`, `151` | `d_model` only |

`in_proj` is the **only** parameter that touches `d_in`. The last group is **dead in E5d**:
`head` is unreachable (the gaussian branch returns at `model.py:768`), `head_mu` is bypassed at
`548-550`, `tstar_encoder` is bypassed at `495`. They still occupy 22% of the checkpoint.

There is **no pest embedding, no pest ID input, and no per-pest parameter anywhere** — `grep -i
pest model.py` returns nothing. The model is pest-agnostic by construction. Pest identity enters
only through (a) which checkpoint is loaded, (b) the feature composition behind `d_in`, and (c)
two **non-parameter** runtime attributes set from outside: `model.alert_tstar_feat_idx` and
`model.doy_start` (`run_train.py:1621,1623`). Those are plain Python attributes, absent from
`state_dict`, so **they must be re-set after every load** — a forgotten
`alert_tstar_feat_idx` silently makes the offset router read `X[..., -1]` (`model.py:440-442`
falls back to `-1` with no validation, unlike the validated path at `335-339`).

## 2. Recommended structure — 안 2 (no pest embedding), with a caveat

**Chosen: mix the pests through one unmodified E5d model. No pest embedding, no pest-specific
heads during pretraining.**

This works only because the participating pests share one input contract, which
`contract_check.py` verifies against the 24 production checkpoints rather than assuming:

```
DOY 60-300   T=241   d_in=45   feature_hash=10fc6a3840338df5
d_model=48   n_head=4   n_layers=3   stage2_nowcast_window=28
```

All 7 non-BPH pests match on **all** of these, including the *ordered* 45-name feature list, and
`alert_tstar` sits at index 30 for every one of them. So a mixed batch is genuinely the same
tensor layout, and every parameter — `in_proj` included — is both shape-compatible and
semantically comparable across pests.

Why not the alternatives:

- **안 1 (pest ID embedding)** adds parameters that have no counterpart in the pest-specific
  model. They must be dropped at fine-tune time, so the pretrained function is not the function
  being fine-tuned, and the embedding absorbs exactly the pest-discriminative signal we want
  pushed into the shared trunk. It also breaks `state_dict` interchange with baseline A.
- **안 3 (shared trunk + 7 pest-specific heads)** is defensible and is the natural fallback if
  mixing turns out to degrade the trunk (head-level interference). It costs 7× the head
  parameters, needs a routing layer that the deployed per-pest model will not have, and its
  extra complexity buys nothing unless interference is actually observed. Recommended as an
  **ablation**, not the first run.

안 2 keeps the fine-tuned artifact byte-layout-identical to baseline A's, which is what makes the
comparison fair and the deployment story trivial (8 independent per-pest models, as today).

## 3. Pest embedding / pest-specific head: not needed

Not for the first run. Justification above. The one thing that **is** required is a pest key in
every sample id — see §5.

## 4. Checkpoint transfer

**Default: transfer `in_proj.*` + `time_encoder.*` (38 keys, 87,024 params). Re-initialize
`offset_mu_heads.*`.** The mu heads encode pest-specific lead-time calibration, which is
precisely what per-pest fine-tuning should learn; the trunk encodes shared weather/season
structure. `--include-heads` exists for the ablation.

**This needs no change to the vendored trainer.** `--stage2_warm_start_ckpt` already loads a
bundle with `load_state_dict(strict=False)` (`run_train.py:1668-1743`), transferring whatever keys
the bundle contains. So "backbone only" is achieved by **writing a bundle that contains only the
backbone keys** — the filtering happens at export time, not load time. `backbone_transfer.py
export` does that, emitting the trainer's own bundle format.

Buffers (`pos.pe`, `candidate_offsets_buf`) are deliberately **not** exported. `pos.pe` is sized
from `T` and `candidate_offsets_buf` is the offset-order guard (`model.py:264-271`); both are
rebuilt correctly by the target constructor, and shipping them would convert a season-length
difference into a silent shape clash.

Fine-tuning therefore runs the **unmodified vendored trainer**, identical to baseline A except
for one added flag:

```
--stage2_warm_start_ckpt <pretrain>/backbone_only.pt --stage2_warm_start_seed 0
```

Note `common.sh:88` strips exactly these two flags from the extracted production args, so the
fine-tune launcher must re-add them *after* `extract_args`.

## 5. Year-wise leakage design

`split_by_year` (`rice/src/dataset.py:209-225`) gives `train = {year < val_year}`. The baseline
always passes `val_year = y-1`, `test = [y,y]` (`common.sh:80-82`), so **train = year ≤ y-2**.
The pretraining shard takes exactly that set and nothing else.

Applied uniformly across pests, that yields the required property: the backbone for eval year `y`
sees **no pest's** year `y` and **no pest's** year `y-1`.

This matters more than it appears. The daily weather cache is **shared across all pests by
design** (`rice/configs/config.py:48-50`) — the `X` features of a given station-year are the same
numbers for every pest; only labels differ. Admitting pest A's `y-1` rows would put pest B's
`val_ckpt` / `val_fit` / `val_cal` inputs in front of the trunk even though pest B was never
named. The uniform `≤ y-2` gate is what closes that hole. The 40/30/30 bucket hash is **not**
pest-salted (`pest_splits.py:30-37` hashes only `"42:<site>-<year>"`), so a station-year lands in
the same bucket for every pest — which is why the year gate, not the bucket, has to do the work.

**dev vs clean need separate backbones.** The two phases differ *only* in the validation set —
`_patched_train.py:38-45` restricts val to the `val_ckpt` ids and passes `train` through
untouched. So their **train** sets are identical, and a single backbone per year would be
data-legal for both. They are still kept separate because checkpoint selection differs: a
backbone early-stopped against the full `y-1` cohort has been *selected* using `val_fit` and
`val_cal` site-years that the clean protocol treats as untouchable. Sharing it would leak
selection signal into the clean fold.

→ **6 pretrain checkpoints: {dev, clean} × {2022, 2023, 2024}.**

Every shard writes `leak_manifest.json` with the pests, included years, excluded years,
row/group/site counts, source file sha256s, and five evaluated assertions (not prose):
`max_train_year_le_eval_minus_2`, `val_year_absent_from_train`, `test_year_absent_from_train`,
`group_ids_pest_prefixed`, `feature_contract_matches_reference`.

**Two landmines this design defuses:**

1. `group_stage2_samples_by_site_year` groups on `(site_id, year)` only
   (`rice/src/dataset.py:740-757`). The same station-year exists in every pest's corpus, so a
   naive concat would **merge two pests' rows into one causal group** — same group, different
   labels. Every id here is `(pest, site_id, year)`.
2. `resolve_pest()` mutates module-level config globals (`rice/configs/config.py:42-50`), so a
   second pest in the same process overwrites the first. Multi-pest training in one process is
   structurally impossible without refactoring the config layer. Hence the two-phase split:
   **one process per pest** for sample export, then a mixing process that never touches a pest
   config.

## 6. Data volume and sampling

Train-row counts from the production checkpoints' own `split_counts` (2024 fold):

| pest | train rows | balanced weight |
|---|---|---|
| brown_spot | 883,314 | 1.00 |
| sheath_blight | 632,729 | 1.00 |
| rice_stem_borer_1 | 609,667 | 1.00 |
| rice_stem_borer_2 | 598,002 | 1.00 |
| blast | 596,161 | 1.00 |
| bacterial_blight | 556,668 | 1.00 |
| WBPH | 278,653 | 1.00 |

Max/min ≈ **3.2×** — real but not extreme. Under natural sampling WBPH contributes ~6.7% of
steps against brown_spot's 21%, and WBPH is the positive control whose number must reproduce.

**Recommended: `balanced`** for the first run (each pest equal expected mass), with `sqrt` as the
ablation. At 3.2× spread, `sqrt` and `balanced` differ modestly, and `balanced` has the clearer
interpretation — "the trunk saw every pest equally" — which is what a general backbone claim
needs. `natural` is recorded for completeness.

**One batch mixes several pests** (`WeightedRandomSampler` over a flat index, not
pest-homogeneous batches taken in turn). With LayerNorm-only normalization, alternating
homogeneous batches would still make every gradient step single-pest, letting the trunk oscillate
between pests instead of averaging toward shared structure. `multipest_sampler.make_sampler`
returns the realized per-pest probabilities so the log records what actually happened.

**Normalization stays per pest**, computed from that shard's `≤ y-2` rows exactly as the trainer
does (`run_train.py:1374` → `dataset.py:643-692`). A pooled normalizer would change every
fine-tuned model's input contract relative to baseline A — a second difference on top of
initialization. Per-pest normalizers mean pest *p* sees the same input space in both phases.

## 7. Including BPH: prerequisite work

**BPH must be excluded from the shared backbone.** It fails the contract on five counts, not
just DOY:

| | 7 pests | BPH |
|---|---|---|
| DOY / T | 60-300 / 241 | 140-270 / **131** |
| `d_in` | 45 | **27** |
| `feature_cols` | 15 engineered | **6 raw daily** |
| feature overlap | — | **15 of 45 names** |

BPH's Stage-2 was built from a different feature generation: raw Korean-named dailies
(`일강수량(mm)`, `평균기온(°C)`, `GDD10_since_gs`) where the 7 use engineered 7-day aggregates
(`rain_7d_sum`, `tmean_7d_mean`, …), and BPH has **none** of the phenology or dispatch channels.
Confirmed at `rice/pests/BPH/features.py:2-14,25-26` — BPH's `run=4` is a different branch
entirely.

So a DOY re-window alone is insufficient; BPH needs a full Stage-1 → dispatch → Stage-2
regeneration on the 45-feature pipeline. **Nothing was regenerated here.** The existing script is
`rice/experiments/allpests_e5d/bph_regen_prepare.sh`, which patches a copy of BPH's Stage-1
template (`phase_t_lead_aware_train.py` reads DOY from the template ckpt, not a CLI flag) and
prints the plan:

```bash
bash rice/experiments/allpests_e5d/bph_regen_prepare.sh          # patch template + print plan
bash rice/experiments/allpests_e5d/bph_regen_prepare.sh --emit   # + write bph_regen_run.sh
```

After that lands, re-run `contract_check.py`; BPH joins the eligible set only if its
`feature_hash` matches.

## 8. Files

New, all under this directory — nothing outside it is modified:

| file | purpose | runnable now |
|---|---|---|
| `contract_check.py` | gate: prove the pests share one input contract | **yes** |
| `e5d_common.py` | paths, E5d model factory, JSONL logging, GPU state | yes |
| `backbone_transfer.py` | key grouping; export a backbone-only warm-start bundle | **yes** |
| `multipest_sampler.py` | mixed-pest dataset, pest-aware ids, sampling policies | yes |
| `export_pest_samples.py` | one pest → one shard + leakage manifest | blocked (needs `LONG_by_pest`) |
| `smoke_test.py` | `--mode real` \| `--mode structural` | structural: **yes** |

Outputs go to `rice/outputs_allpests_e5d_multipest_pretrain/` only.
`rice/outputs_allpests_e5d/`, `rice/outputs_allpests_e5d_curriculum/` and the 10 pinned
`vendor/` files are never written.

**Still to write: the pretraining trainer.** The production loss is **not importable** — it is
inline inside `run_train.py:main()` (a ~2000-line function); the module exposes no loss function
(`grep "^def .*loss"` returns nothing). Reimplementing it would fork the objective under
comparison, which is the one thing that must not differ. The sanctioned route is therefore a
**copy of `run_train.py` into this directory** (`multipest_run_train.py`), with the loss and the
training loop untouched and only the data-source section replaced: `resolve_pest` + `load_obs` +
`build_samples_season` + `split_samples` → `MixedPestDataset` over the exported shards. That fork
is deliberately not written blind, with no data to test it against.

## Blockers

| blocker | evidence | consequence |
|---|---|---|
| `LONG_by_pest` CSVs absent | `export_pest_samples.py` reports `No such file: /home/gpu4080/ygdata/rice/LONG_by_pest/RICE_LONG_흰등멸구.csv`; the daily cache *hits*, so this is the only data gap | no shard can be built → no real-data smoke test, no pretraining |
| sheath_blight Stage-1: 9 broken symlinks | `find rice/outputs -xtype l` → 9, all pointing into server 1's `outputs_stage1/seed_stability` | sheath_blight must be dropped from the backbone until fixed |
| baseline A absent | `rice/outputs_allpests_e5d/` does not exist on this host | experiment B has nothing to be compared against yet |
| BPH contract | above | BPH excluded; 7 pests eligible |
