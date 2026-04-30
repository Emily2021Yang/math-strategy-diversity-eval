#!/usr/bin/env python3
"""Code generated math strategies against a human reference strategy list.

The script uses:
- a long-format generated strategy CSV, one generated strategy per row;
- strategy_reference_stage1.csv, one human reference strategy per row;
- examples of coding.csv, a few hand-coded examples used in the prompt.

It writes separate coder outputs for GPT and Gemini by default. Each coder sees
only the generated strategy being coded and the reference strategies for the
same problem_id.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TIMEOUT_SECONDS = 180
OUTPUT_COLUMNS = [
    "problem_id",
    "problem",
    "generated_source",
    "generated_model_id",
    "strategy_index",
    "generated_strategy_name",
    "generated_method_summary",
    "generated_key_steps",
    "generated_final_answer",
    "coder_model",
    "assigned_strategy_id",
    "strategy_match_confidence",
    "strategy_valid",
    "validity_confidence",
    "result_correct",
    "raw_response",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-input",
        nargs="+",
        default=["Benchmark/runs/20260407_155325_Claude_p2/responses_long_format.csv"],
        help="One or more long-format CSVs with one generated strategy per row.",
    )
    parser.add_argument(
        "--reference",
        default="Benchmark/strategy_reference_stage1.csv",
        help="Human reference strategy CSV.",
    )
    parser.add_argument(
        "--examples",
        default="Benchmark/examples of coding.csv",
        help="Hand-coded example CSV.",
    )
    parser.add_argument(
        "--prompt-file",
        default="Benchmark/strategy_coding_prompt.txt",
        help="Prompt template with {{examples_json}} and {{coding_item_json}} placeholders.",
    )
    parser.add_argument(
        "--problems",
        default="Benchmark/Problems81.csv",
        help="Problem CSV used to fill problem text when generated input lacks it.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to Benchmark/strategy_coding_runs/<timestamp>.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of generated strategies to code when not sampling/choosing problem IDs.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based starting row after filtering/sorting.",
    )
    parser.add_argument(
        "--problem-ids",
        nargs="+",
        default=None,
        help="Specific problem IDs to code. Codes all generated strategies for these problems.",
    )
    parser.add_argument(
        "--all-reference-problems",
        action="store_true",
        help="Code all generated strategies whose problem_id appears in the reference list.",
    )
    parser.add_argument(
        "--sample-problem-count",
        type=int,
        default=None,
        help="Randomly sample this many problem IDs and code all generated strategies for them.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260416,
        help="Random seed for --sample-problem-count.",
    )
    parser.add_argument(
        "--exclude-example-problems",
        action="store_true",
        help="Exclude problem IDs used in the hand-coded examples from random sampling.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt", "gemini"],
        choices=["gpt", "gemini"],
        help="Coder models to run.",
    )
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature. Omit by default because some reasoning models do not support it.",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file with API keys. Existing environment variables take precedence.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write prompt_units.jsonl but do not call model APIs.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    return df


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return normalize_columns(pd.read_csv(path, low_memory=False))


def load_problem_texts(path: Path) -> dict[str, str]:
    df = read_csv(path)
    if "Problem_ID" not in df.columns or "Problem" not in df.columns:
        return {}
    return {
        clean_value(row["Problem_ID"]): clean_value(row["Problem"])
        for _, row in df.iterrows()
        if clean_value(row["Problem_ID"])
    }


def load_reference(path: Path) -> dict[str, list[dict[str, str]]]:
    df = read_csv(path)
    required = ["Problem_ID", "Strategy_ID", "strategy_name", "method_summary", "key_steps", "final_answer"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Reference file is missing required columns: {missing}")

    df = df[df["Problem_ID"].notna() & df["Strategy_ID"].notna()].copy()
    df["Problem_ID"] = df["Problem_ID"].map(clean_value)
    df["Strategy_ID"] = df["Strategy_ID"].map(clean_value)
    df = df[(df["Problem_ID"] != "") & (df["Strategy_ID"] != "")]

    reference: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        problem_id = clean_value(row["Problem_ID"])
        item = {
            "strategy_id": clean_value(row["Strategy_ID"]),
            "strategy_name": clean_value(row["strategy_name"]),
            "method_summary": clean_value(row["method_summary"]),
            "key_steps": clean_value(row["key_steps"]),
            "final_answer": clean_value(row["final_answer"]),
        }
        equivalent_notes = clean_value(row.get("equivalent_notes"))
        if equivalent_notes:
            item["equivalent_notes"] = equivalent_notes
        coding_scope_note = clean_value(row.get("coding_scope_note"))
        if coding_scope_note:
            item["coding_scope_note"] = coding_scope_note
        reference.setdefault(problem_id, []).append(item)
    return reference


def load_generated(path: Path, problem_texts: dict[str, str]) -> pd.DataFrame:
    df = read_csv(path)
    aliases = {
        "Problem_ID": "problem_id",
        "Problem": "problem",
        "Strategy_Index": "strategy_index",
    }
    df = df.rename(columns={old: new for old, new in aliases.items() if old in df.columns})
    required = [
        "problem_id",
        "strategy_index",
        "generated_strategy_name",
        "generated_method_summary",
        "generated_key_steps",
        "generated_final_answer",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Generated input is missing required columns: {missing}")

    if "problem" not in df.columns:
        df["problem"] = df["problem_id"].map(lambda value: problem_texts.get(clean_value(value), ""))
    if "generated_source" not in df.columns:
        df["generated_source"] = path.parent.name
    if "generated_model_id" not in df.columns:
        if "model_id" in df.columns:
            df["generated_model_id"] = df["model_id"]
        else:
            df["generated_model_id"] = ""
    for col in required + ["problem"]:
        df[col] = df[col].map(clean_value)
    df["generated_source"] = df["generated_source"].map(clean_value)
    df["generated_model_id"] = df["generated_model_id"].map(clean_value)
    return df[df["problem_id"] != ""].reset_index(drop=True)


def load_generated_many(paths: list[str], problem_texts: dict[str, str]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    frames = []
    ids_by_input: dict[str, set[str]] = {}
    for path_text in paths:
        path = Path(path_text)
        df = load_generated(path, problem_texts)
        df["generated_input"] = str(path)
        frames.append(df)
        ids_by_input[str(path)] = set(df["problem_id"].astype(str))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["problem_id", "generated_source", "strategy_index"]).reset_index(drop=True)
    return combined, ids_by_input


def parse_possible_json_list(value: str) -> Any:
    value = clean_value(value)
    if not value:
        return []
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def generated_strategy_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "strategy_index": clean_value(row.get("strategy_index")),
        "generated_strategy_name": clean_value(row.get("generated_strategy_name")),
        "generated_method_summary": clean_value(row.get("generated_method_summary")),
        "generated_key_steps": parse_possible_json_list(clean_value(row.get("generated_key_steps"))),
        "generated_final_answer": clean_value(row.get("generated_final_answer")),
    }


def build_examples(path: Path, reference: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    df = read_csv(path)
    if "Problem_ID" in df.columns:
        df = df.rename(columns={"Problem_ID": "problem_id", "Problem": "problem"})

    required = [
        "example_id",
        "problem_id",
        "problem",
        "generated_strategy_name",
        "generated_method_summary",
        "generated_key_steps",
        "generated_final_answer",
        "assigned_strategy_id",
        "strategy_match_confidence",
        "strategy_valid",
        "validity_confidence",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Examples file is missing required columns: {missing}")

    examples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        problem_id = clean_value(row["problem_id"])
        refs = reference.get(problem_id, [])
        correct_coding = {
            "assigned_strategy_id": clean_value(row["assigned_strategy_id"]),
            "strategy_match_confidence": coerce_number_or_text(row["strategy_match_confidence"]),
            "strategy_valid": coerce_number_or_text(row["strategy_valid"]),
            "validity_confidence": coerce_number_or_text(row["validity_confidence"]),
        }
        if "result_correct" in df.columns and clean_value(row.get("result_correct")):
            correct_coding["result_correct"] = coerce_number_or_text(row.get("result_correct"))
        else:
            inferred = infer_result_correct(row, refs)
            if inferred != "":
                correct_coding["result_correct"] = inferred
        examples.append(
            {
                "example_id": clean_value(row["example_id"]),
                "problem_id": problem_id,
                "problem": clean_value(row["problem"]),
                "generated_strategy": generated_strategy_from_row(row),
                "reference_strategies_for_this_problem": refs,
                "correct_coding": correct_coding,
            }
        )
    return examples


def coerce_number_or_text(value: Any) -> Any:
    text = clean_value(value)
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def normalize_answer(value: Any) -> str:
    text = clean_value(value).lower()
    text = text.replace("$", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", "", text)
    return text


def infer_result_correct(row: pd.Series, reference_items: list[dict[str, str]]) -> Any:
    generated_answer = normalize_answer(row.get("generated_final_answer"))
    if not generated_answer:
        return ""
    reference_answers = {
        normalize_answer(item.get("final_answer"))
        for item in reference_items
        if normalize_answer(item.get("final_answer"))
    }
    if not reference_answers:
        return ""
    return 1 if generated_answer in reference_answers else 0


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    template = path.read_text(encoding="utf-8")
    required_placeholders = ["{{examples_json}}", "{{coding_item_json}}"]
    missing = [placeholder for placeholder in required_placeholders if placeholder not in template]
    if missing:
        raise ValueError(f"Prompt template is missing placeholders: {missing}")
    return template


def build_prompt(unit: dict[str, Any], examples: list[dict[str, Any]], template: str) -> str:
    return (
        template.replace("{{examples_json}}", json.dumps(examples, ensure_ascii=False, indent=2))
        .replace("{{coding_item_json}}", json.dumps(unit, ensure_ascii=False, indent=2))
        .strip()
        + "\n"
    )


def build_units(
    generated: pd.DataFrame,
    reference: dict[str, list[dict[str, str]]],
    *,
    start: int,
    limit: int | None,
    problem_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    units = []
    skipped_without_reference = 0
    rows = generated
    if problem_ids is not None:
        problem_id_set = set(problem_ids)
        rows = rows[rows["problem_id"].isin(problem_id_set)].copy()
    else:
        rows = rows.iloc[start:].copy()

    for _, row in rows.iterrows():
        problem_id = clean_value(row["problem_id"])
        refs = reference.get(problem_id, [])
        if not refs:
            skipped_without_reference += 1
            continue
        units.append(
            {
                "problem_id": problem_id,
                "problem": clean_value(row["problem"]),
                "generated_source": clean_value(row.get("generated_source")),
                "generated_model_id": clean_value(row.get("generated_model_id")),
                "generated_strategy": generated_strategy_from_row(row),
                "reference_strategies_for_this_problem": refs,
            }
        )
        if limit is not None and len(units) >= limit:
            break
    if not units:
        raise ValueError("No generated strategies had same-problem reference strategies.")
    if skipped_without_reference:
        print(f"Skipped {skipped_without_reference} generated rows without same-problem references.")
    return units


def choose_problem_ids(
    *,
    reference: dict[str, list[dict[str, str]]],
    ids_by_input: dict[str, set[str]],
    examples_path: Path,
    explicit_problem_ids: list[str] | None,
    all_reference_problems: bool,
    sample_problem_count: int | None,
    exclude_example_problems: bool,
    random_seed: int,
) -> list[str] | None:
    if explicit_problem_ids:
        return [clean_value(problem_id) for problem_id in explicit_problem_ids if clean_value(problem_id)]
    if all_reference_problems:
        return sorted(reference)
    if sample_problem_count is None:
        return None

    candidate_ids = set(reference)
    for ids in ids_by_input.values():
        candidate_ids &= ids

    if exclude_example_problems:
        examples = read_csv(examples_path)
        example_col = "Problem_ID" if "Problem_ID" in examples.columns else "problem_id"
        if example_col in examples.columns:
            candidate_ids -= set(examples[example_col].map(clean_value))

    candidates = sorted(candidate_ids)
    if sample_problem_count > len(candidates):
        raise ValueError(
            f"Cannot sample {sample_problem_count} problem IDs; only {len(candidates)} candidates are available."
        )
    rng = random.Random(random_seed)
    return sorted(rng.sample(candidates, sample_problem_count))


def extract_json_object(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON response: {text[:300]}")


def normalize_coding_response(parsed: Any) -> dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and "assigned_strategy_id" in item:
                return item
    raise ValueError(f"Expected JSON object coding response, got {type(parsed).__name__}")


def sanitize_error(message: str) -> str:
    sanitized = message
    for key_name in ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]:
        value = os.getenv(key_name)
        if value:
            sanitized = sanitized.replace(value, "[REDACTED]")
    sanitized = re.sub(r"key=[^)&\\s]+", "key=[REDACTED]", sanitized)
    return sanitized


def is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return "quota exceeded" in lowered or "resource_exhausted" in lowered


def call_openai(prompt: str, model: str, temperature: float, timeout: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "text": {"format": {"type": "json_object"}},
        "reasoning": {"effort": "medium"},
    }
    if temperature is not None:
        payload["temperature"] = temperature

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    data = response.json()
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI HTTP {response.status_code}: {json.dumps(data, ensure_ascii=False)}")
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and part.get("text"):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


def call_gemini(prompt: str, model: str, temperature: float, timeout: int) -> str:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set.")

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    if temperature is not None:
        payload["generationConfig"]["temperature"] = temperature
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    data = response.json()
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {json.dumps(data, ensure_ascii=False)}")

    chunks: list[str] = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if part.get("text"):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


def output_row(unit: dict[str, Any], coder_model: str, parsed: dict[str, Any], raw_response: str) -> dict[str, Any]:
    generated = unit["generated_strategy"]
    return {
        "problem_id": unit["problem_id"],
        "problem": unit["problem"],
        "generated_source": unit.get("generated_source", ""),
        "generated_model_id": unit.get("generated_model_id", ""),
        "strategy_index": generated["strategy_index"],
        "generated_strategy_name": generated["generated_strategy_name"],
        "generated_method_summary": generated["generated_method_summary"],
        "generated_key_steps": json.dumps(generated["generated_key_steps"], ensure_ascii=False)
        if not isinstance(generated["generated_key_steps"], str)
        else generated["generated_key_steps"],
        "generated_final_answer": generated["generated_final_answer"],
        "coder_model": coder_model,
        "assigned_strategy_id": clean_value(parsed.get("assigned_strategy_id")),
        "strategy_match_confidence": clean_value(parsed.get("strategy_match_confidence")),
        "strategy_valid": clean_value(parsed.get("strategy_valid")),
        "validity_confidence": clean_value(parsed.get("validity_confidence")),
        "result_correct": clean_value(parsed.get("result_correct")),
        "raw_response": raw_response,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))
    output_dir = Path(args.output_dir) if args.output_dir else Path("Benchmark/strategy_coding_runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_texts = load_problem_texts(Path(args.problems))
    reference = load_reference(Path(args.reference))
    generated, ids_by_input = load_generated_many(args.generated_input, problem_texts)
    examples = build_examples(Path(args.examples), reference)
    prompt_template = load_prompt_template(Path(args.prompt_file))
    selected_problem_ids = choose_problem_ids(
        reference=reference,
        ids_by_input=ids_by_input,
        examples_path=Path(args.examples),
        explicit_problem_ids=args.problem_ids,
        all_reference_problems=args.all_reference_problems,
        sample_problem_count=args.sample_problem_count,
        exclude_example_problems=args.exclude_example_problems,
        random_seed=args.random_seed,
    )
    units = build_units(
        generated,
        reference,
        start=args.start,
        limit=None if selected_problem_ids is not None else args.limit,
        problem_ids=selected_problem_ids,
    )

    prompt_path = output_dir / "prompt_units.jsonl"
    with prompt_path.open("w", encoding="utf-8") as handle:
        for unit in units:
            prompt = build_prompt(unit, examples, prompt_template)
            handle.write(json.dumps({"unit": unit, "prompt": prompt}, ensure_ascii=False) + "\n")

    metadata = {
        "generated_input": args.generated_input,
        "reference": args.reference,
        "examples": args.examples,
        "prompt_file": args.prompt_file,
        "problems": args.problems,
        "limit": args.limit,
        "start": args.start,
        "problem_ids": selected_problem_ids,
        "all_reference_problems": args.all_reference_problems,
        "sample_problem_count": args.sample_problem_count,
        "random_seed": args.random_seed,
        "exclude_example_problems": args.exclude_example_problems,
        "models": args.models,
        "openai_model": args.openai_model,
        "gemini_model": args.gemini_model,
        "prompt_units": str(prompt_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if selected_problem_ids is not None:
        (output_dir / "selected_problem_ids.txt").write_text("\n".join(selected_problem_ids) + "\n", encoding="utf-8")

    if args.prepare_only:
        print(f"Prepared {len(units)} prompt units at {prompt_path}")
        return 0

    for model_name in args.models:
        rows: list[dict[str, Any]] = []
        output_path = output_dir / f"{model_name}_strategy_coding_output.csv"
        for index, unit in enumerate(units, start=1):
            prompt = build_prompt(unit, examples, prompt_template)
            try:
                if model_name == "gpt":
                    raw = call_openai(prompt, args.openai_model, args.temperature, args.timeout)
                    coder_label = f"gpt:{args.openai_model}"
                else:
                    raw = call_gemini(prompt, args.gemini_model, args.temperature, args.timeout)
                    coder_label = f"gemini:{args.gemini_model}"
                parsed = normalize_coding_response(extract_json_object(raw))
                rows.append(output_row(unit, coder_label, parsed, raw))
                print(f"[{model_name}] coded {index}/{len(units)} {unit['problem_id']} strategy {unit['generated_strategy']['strategy_index']}")
            except Exception as exc:
                sanitized_error = sanitize_error(str(exc))
                if model_name == "gemini" and is_quota_error(sanitized_error):
                    print(
                        f"[{model_name}] quota reached at {index}/{len(units)}: {sanitized_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                error_response = json.dumps({"error": sanitize_error(str(exc))}, ensure_ascii=False)
                rows.append(output_row(unit, f"{model_name}:ERROR", {}, error_response))
                print(f"[{model_name}] ERROR {index}/{len(units)}: {sanitize_error(str(exc))}", file=sys.stderr, flush=True)
            write_rows(output_path, rows)
            time.sleep(args.sleep_seconds)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
