# Dataset Notes

These notes summarize how the public dataset release connects to the reproducibility repository.

## Released dataset

The public benchmark release lives outside this repository and currently uses Kaggle as the primary host:

- `https://doi.org/10.34740/kaggle/ds/10271409`

This repository should link to the release rather than duplicate all benchmark artifacts.

## Authoritative released files

The current paper is based on the finalized files in the dataset release, especially:

- `full80_prompt_multi_annotation/Full80_valid_correct_strategies_annotated.csv`
- `full80_prompt_multi_annotation/human_ai_reliability_sampled_rows.csv`
- `full80_prompt_multi_annotation/human_ai_reliability_summary.csv`
- `repeated_run_subset20/subset20_multi_run1_annotated.csv`
- `repeated_run_subset20/subset20_multi_run2_annotated.csv`
- `strategy_inventory/complete_strategy_families_AoPS_LLM.csv`
- `full80_prompt_single_accuracy_labels/single_run_correctness_by_problem_model.csv`

## Important interpretation notes

- Strategy families are problem-specific.
- Benchmark-novel labels are corpus-relative.
- The repeated-run subset is intended for robustness and saturation analysis.
- Intermediate run folders in the broader project tree are not the public release. Use the finalized files in `dataset_release/` when reproducing the paper.

## Repository role

This repository contains:

- prompt files
- n-shot coding examples
- preprocessing and runner scripts
- the main analysis notebook
- paper-facing figures and summary tables

It is the reproducibility package, not the benchmark host.
