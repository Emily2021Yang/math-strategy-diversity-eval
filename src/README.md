# src

This folder contains the main reproducibility assets used in the evaluation framework.

## Structure

- `prompts/`: benchmark prompts used to collect model outputs
- `coding/`: AI-coder prompt and n-shot coding examples
- `preprocessing/`: scripts for converting raw outputs into structured formats
- `runners/`: scripts for running benchmark generation and strategy coding

## Files

### `prompts/`
- `prompt_single.txt`: baseline single-solution prompt
- `prompt_multi.txt`: multi-strategy prompt
- `strategy_coding_prompt.txt`: prompt used for AI strategy coding

### `coding/`
- `coding_examples_nshot.csv`: n-shot examples used to guide AI coders

### `preprocessing/`
- `convert_multi_responses_to_long_format.py`: converts model outputs into long-format strategy rows

### `runners/`
- `run_math_benchmark.py`: runs benchmark inference over selected models/prompts
- `run_strategy_coding.py`: runs the AI-coder strategy-labeling pipeline
