# `src/`

This folder contains the main workflow assets used in the evaluation framework.

## Structure

- `prompts/`: benchmark prompting templates and AI-coder prompt text
- `in-context example/`: n-shot coding examples and annotation-support materials
- `preprocessing/`: scripts for converting raw outputs into structured formats
- `runners/`: scripts for running benchmark generation and strategy coding

## Files

### `prompts/`

- `prompt_single.txt`: baseline single-solution prompt
- `prompt_multi.txt`: multiple-strategy prompt
- `strategy_coding_prompt.txt`: prompt used for AI strategy coding

### `in-context example/`

- `coding_examples_nshot.csv`: four-shot examples used to guide AI coders

### `preprocessing/`

- `convert_multi_responses_to_long_format.py`: converts structured model outputs into long-format strategy rows

### `runners/`

- `run_math_benchmark.py`: runs benchmark inference over selected models and prompts
- `run_strategy_coding.py`: runs the AI-coder strategy-labeling pipeline

## Workflow note

The released paper results are based on the finalized public benchmark files, not on intermediate raw run folders alone. These scripts are provided so the end-to-end workflow is inspectable and reproducible.
