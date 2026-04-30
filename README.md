# math-strategy-diversity-eval
A strategy diversity evaluation framework for mathematical reasoning with released data, coding tools, and reproducible analysis.

The framework evaluates model reasoning along three dimensions:

1. final-answer accuracy
2. coverage of human reference strategies
3. benchmark-novel valid strategies

The released dataset is hosted separately. This repository is the reproducibility package for the paper and analysis.

## Contents

- `src/`: parsing and coding scripts
- `analysis.ipynb`: analysis notebook
- `analysis_outputs/`: paper-facing summary files
- `assets/`: manuscript drafts, tables, and figures
- `docs/`: supporting notes
- `data_links/`: links to the public dataset release

## Dataset

Public dataset links at Kaggle:
https://doi.org/10.34740/kaggle/ds/10271409

## Definitions

- `s#`: AoPS reference strategy for a specific problem
- `n#`: benchmark-novel valid strategy for a specific problem
- `novel`: novel relative to the collected AoPS corpus, not universally novel in mathematics

## Reproducing the analysis

To reproduce the paper workflow:

1. download the released dataset
2. run the parsing and coding scripts in `src/`
3. reproduce tables and figures from `analysis.ipynb`

## License

Code in this repository may be released under a separate license from the dataset. See the dataset release for data-specific usage notes.

## Citation

If you use this project, please cite the associated paper and link both the repository and the dataset release.
