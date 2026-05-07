# Strategy Diversity Evaluation Framework for Mathematical Reasoning

This repository is the **reproducibility package** for the strategy-diversity benchmark paper. It contains the code, analysis notebooks, prompts, and paper-facing assets used to reproduce the released results.

The benchmark evaluates mathematical reasoning along three linked dimensions:

1. final-answer correctness
2. recovery of AoPS reference strategies
3. benchmark-novel valid strategies

The released benchmark data are hosted separately. This repository is the companion code-and-analysis package.

## Public links

- GitHub repository: `https://github.com/Emily2021Yang/math-strategy-diversity-eval.git`
- Kaggle dataset: `https://doi.org/10.34740/kaggle/ds/10271409`
- Paper link: `TBD`
- Hugging Face mirror: `TBD`

## Repository layout

```text
github_repo/
├── analysis.ipynb
├── analysis_outputs/
├── assets/
│   ├── main_text_figures/
│   ├── appendix_figures/
│   ├── main_text_tables/
│   ├── appendix_tables/
│   └── manuscript/
├── docs/
├── data_links/
└── src/
    ├── prompts/
    ├── in-context example/
    ├── preprocessing/
    └── runners/
```

## What is in the repository

### `src/`

Core workflow assets:

- benchmark prompts
- AI-coder prompt
- n-shot coding examples
- preprocessing scripts for long-format conversion
- run scripts for benchmark querying and strategy coding

### `analysis.ipynb`

Primary notebook for generating paper-facing summaries and figures from the released benchmark files.

### `analysis_outputs/`

CSV and image outputs used to support the current paper analyses, including:

- overall and domain-level paired-gap summaries
- domain coverage and novelty summaries
- repeated-run summary files for the 20-problem subset

### `assets/`

Paper-facing figures, tables, and manuscript drafts. These are organized by role rather than by the full paper build system.

## Reproducing the current paper workflow

At a high level:

1. download the released dataset from Kaggle
2. use `src/preprocessing/` and `src/runners/` for query/coding workflows if needed
3. reproduce analysis summaries and figures from `analysis.ipynb`

The released paper uses the finalized files in the public dataset release, especially:

- `full80_prompt_multi_annotation/Full80_valid_correct_strategies_annotated.csv`
- `repeated_run_subset20/subset20_multi_run1_annotated.csv`
- `repeated_run_subset20/subset20_multi_run2_annotated.csv`

## Definitions

- `s#`: AoPS reference strategy for a specific problem
- `n#`: benchmark-novel valid strategy for a specific problem
- `novel`: novel relative to the collected AoPS corpus, not universally novel in mathematics

## Notes

- Strategy identifiers are problem-specific rather than global.
- The repeated-run subset is intended for robustness and saturation analysis, not as a separate benchmark.
- Intermediate run folders elsewhere in the project tree are not the public release; the dataset release contains the finalized benchmark artifacts used in the current paper.

## Citation

If you use this repository, please cite the associated paper and link both the repository and the dataset release.
