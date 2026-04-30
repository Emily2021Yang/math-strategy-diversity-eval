# Strategy Diversity Benchmark Dataset

This dataset release accompanies the paper on strategy diversity in mathematical reasoning by large language models.

## Overview

The benchmark contains 80 competition-style mathematics problems drawn from AMC 10/12 and AIME, together with:

- benchmark metadata
- a finalized human-coded strategy inventory
- finalized coding of model outputs
- repeated-run subset files for robustness analysis
- prompt-single final-answer correctness labels

The benchmark is designed to evaluate not only whether models solve problems correctly, but also how many **distinct valid strategies** they recover relative to a human reference set derived from Art of Problem Solving (AoPS) solutions.

## Key ideas

### AoPS reference strategies

For each problem, AoPS human solutions were grouped into problem-specific strategy families:

- `s1`, `s2`, `s3`, ...

These labels are **problem-specific**, not global. For example, `s1` on one problem is unrelated to `s1` on another problem.

### Benchmark-novel strategies

If a model produced a valid correct strategy that did not match any AoPS reference family for that problem, it was assigned a benchmark-novel label:

- `n1`, `n2`, `n3`, ...

Important: **"novel" means novel relative to the collected AoPS benchmark corpus**, not universally novel in mathematics.

### Countable strategies

A model output counts as a distinct strategy only if:

1. the strategy is valid, and
2. the final result is correct

Rows that are invalid or have an incorrect result are retained in coding files for transparency, but they do not count toward strategy-diversity totals.

## Dataset structure

```text
dataset_release/
├── benchmark_metadata/
├── strategy_inventory/
├── full_benchmark_coding/
├── repeated_run_subset/
└── accuracy_labels/
```

## File descriptions

### `benchmark_metadata/`

- `Problems80.csv`
  Main benchmark metadata file, including problem text and benchmark labels used in the paper release.

### `strategy_inventory/`

- `complete_strategy_after_coding.xlsx`
  Finalized strategy inventory containing AoPS reference strategies and benchmark-novel strategies.

- `per_problem_aops_novel_total_strategy_counts.csv`
  Per-problem inventory reporting the number of AoPS strategies, benchmark-novel strategies, and total strategies.

### `full_benchmark_coding/`

- `finalized_valid_correct_strategies_all.csv`
  Finalized coding for the full 80-problem benchmark across all models and multiple-strategy outputs.

- `gpt_gemini_agreement_all.csv`
  Rows where the two AI coders agreed.

- `gpt_gemini_discrepancies_all.csv`
  Rows where the two AI coders disagreed and were later reviewed by humans.

### `repeated_run_subset/`

- `stability_subset20_multi_run1_finalized_complete.csv`
  Finalized coding for repeated run 1 on the 20-problem subset.

- `stability_subset20_multi_run2_finalized_complete.csv`
  Finalized coding for repeated run 2 on the 20-problem subset.

- `four_models_responses_long_format_run1.csv`
  Structured long-format model outputs for run 1.

- `four_models_responses_long_format_run2.csv`
  Structured long-format model outputs for run 2.

### `accuracy_labels/`

- `single_run_correctness_by_problem_model.csv`
  Final-answer correctness labels for the baseline single-solve prompt.

## Important columns

Across coding files, the most important finalized columns are:

- `final_assigned_strategy_id`
  Final strategy label after AI coding and human adjudication.

- `final_strategy_valid`
  Whether the strategy is mathematically valid.

- `final_result_correct`
  Whether the final result is correct.

- `count_as_distinct_strategy`
  Equals `1` only when the strategy is both valid and result-correct.

- `final_distinct_strategy_id`
  The strategy ID used for diversity counting. This is blank when a row does not count as a strategy.

## Suggested citation

If you use this dataset, please cite the associated paper and link to the dataset release page.

## Notes

- Strategy families are problem-specific.
- Benchmark-novel labels are corpus-relative.
- The repeated-run subset is intended for robustness and saturation analysis, not as a separate benchmark.
