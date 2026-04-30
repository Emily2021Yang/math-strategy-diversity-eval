#!/usr/bin/env python3
"""Convert multi-strategy benchmark responses into long-format strategy CSVs.

This script reads one or more `responses.csv` files produced by
`Benchmark/run_math_benchmark.py` for the `multi_strategy` task and writes a
long-format CSV with one generated strategy per row.

If multiple input files are provided for the same provider/model, later files
override earlier ones for the same problem_id. This is useful for merging retry
results into a final provider-level output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


STRATEGY_LIST_KEYS = ("strategies", "distinct_strategies", "solutions", "approaches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more responses.csv files to merge and convert.",
    )
    parser.add_argument(
        "--problems",
        default="Benchmark/Problems20_stability_subset.csv",
        help="Problem CSV used to fill the problem text.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output long-format CSV path.",
    )
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def stringify_json(value: Any) -> str:
    if value in (None, "", []):
        return ""
    return json.dumps(value, ensure_ascii=False)


def load_problem_texts(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        clean(row.get("Problem_ID")): clean(row.get("Problem"))
        for row in rows
        if clean(row.get("Problem_ID"))
    }


def load_and_merge_rows(paths: list[Path]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            key = (
                clean(row.get("problem_id")),
                clean(row.get("provider")),
                clean(row.get("model_id")),
            )
            merged[key] = row
    return list(merged.values())


def extract_json_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    stripped = text.strip()
    if not stripped:
        return candidates

    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.DOTALL)
    for block in fenced_blocks:
        parsed = try_load_json_relaxed(block.strip())
        if isinstance(parsed, dict):
            candidates.append(parsed)

    parsed = try_load_json_relaxed(stripped)
    if isinstance(parsed, dict):
        candidates.append(parsed)

    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[start_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)

    return candidates


def try_load_json_relaxed(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Some model outputs label a block as JSON but leave LaTeX backslashes
    # unescaped, e.g. \pmod or \phi. Escape only backslashes that are not part
    # of the standard JSON escape set.
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def score_candidate(candidate: dict[str, Any]) -> tuple[int, int]:
    for key in STRATEGY_LIST_KEYS:
        value = candidate.get(key)
        if isinstance(value, list):
            return (3, len(value))
    if any(clean(candidate.get(key)) for key in ("strategy_name", "name", "title", "approach")):
        return (2, 1)
    if any(clean(candidate.get(key)) for key in ("final_answer", "method_summary", "solution")):
        return (1, 0)
    return (0, 0)


def best_json_candidate(text: str) -> dict[str, Any] | None:
    candidates = extract_json_candidates(text)
    if not candidates:
        return None
    return max(candidates, key=score_candidate)


def strategies_from_candidate(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    for key in STRATEGY_LIST_KEYS:
        value = candidate.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def strip_markdown(value: str) -> str:
    text = clean(value)
    text = re.sub(r"^\*+|\*+$", "", text)
    text = re.sub(r"^`+|`+$", "", text)
    return text.strip()


def parse_markdown_strategies(text: str) -> list[dict[str, Any]]:
    heading_pattern = re.compile(
        r"(?m)^(?:##+\s*)?Strategy\s*(\d+)\s*:\s*(.+?)\s*$",
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return []

    strategies: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end].strip()

        method_match = re.search(
            r"\*\*Method(?:\s+summary)?\s*:\*\*\s*(.+?)(?=\n\s*\*\*Key\s+steps\s*:\*\*|\n\s*\*\*(?:Final\s+)?[Aa]nswer:\*\*|\n\s*\*\*[^*\n]+:\*\*|\Z)",
            section,
            flags=re.DOTALL | re.IGNORECASE,
        )
        key_steps_match = re.search(
            r"\*\*Key\s+steps\s*:\*\*\s*(.+?)(?=\n\s*\*\*(?:Final\s+)?[Aa]nswer:\*\*|\Z)",
            section,
            flags=re.DOTALL | re.IGNORECASE,
        )
        final_match = re.search(
            r"\*\*(?:Final\s+)?[Aa]nswer:\*\*\s*(.+?)(?=\n|$)",
            section,
            flags=re.DOTALL,
        )
        if not final_match:
            final_match = re.search(
                r"\\boxed\{(.+?)\}",
                section,
                flags=re.DOTALL,
            )
        if not final_match:
            final_match = re.search(
                r"(?:The\s+answer\s+is|Answer\s*:)\s*\**\s*(.+?)(?=\n|$)",
                section,
                flags=re.DOTALL | re.IGNORECASE,
            )

        key_steps: list[str] = []
        if key_steps_match:
            block = key_steps_match.group(1).strip()
            bullets = re.findall(r"(?m)^\s*-\s+(.*)$", block)
            if bullets:
                key_steps = [strip_markdown(item) for item in bullets if strip_markdown(item)]
            else:
                numbered = re.findall(r"(?m)^\s*\d+\.\s+(.*)$", block)
                if numbered:
                    key_steps = [strip_markdown(item) for item in numbered if strip_markdown(item)]
            if not key_steps and block:
                key_steps = [strip_markdown(block)]
        elif section:
            fallback_section = section
            if method_match:
                fallback_section = fallback_section.replace(method_match.group(0), "").strip()
            if final_match:
                try:
                    fallback_section = fallback_section.replace(final_match.group(0), "").strip()
                except IndexError:
                    pass
            paragraphs = [strip_markdown(part) for part in re.split(r"\n\s*\n", fallback_section) if strip_markdown(part)]
            if paragraphs:
                key_steps = paragraphs

        strategies.append(
            {
                "strategy_name": strip_markdown(match.group(2)),
                "method_summary": strip_markdown(method_match.group(1)) if method_match else "",
                "key_steps": key_steps,
                "final_answer": strip_markdown(final_match.group(1)) if final_match else "",
            }
        )
    return strategies


def coerce_key_steps(strategy: dict[str, Any]) -> str:
    key_steps = strategy.get("key_steps")
    if isinstance(key_steps, list):
        cleaned = [clean(step) for step in key_steps if clean(step)]
        return stringify_json(cleaned)
    if isinstance(key_steps, dict):
        return stringify_json(key_steps)

    text_value = clean(key_steps)
    if text_value:
        return text_value

    fallback = clean(strategy.get("solution"))
    if fallback:
        return fallback
    fallback = clean(strategy.get("why_distinct"))
    return fallback


def normalize_strategy(strategy: dict[str, Any]) -> dict[str, str]:
    strategy_name = ""
    for key in ("strategy_name", "name", "title", "approach"):
        if clean(strategy.get(key)):
            strategy_name = clean(strategy.get(key))
            break

    method_summary = clean(strategy.get("method_summary"))
    if not method_summary:
        method_summary = clean(strategy.get("solution"))

    final_answer = clean(strategy.get("final_answer"))

    return {
        "generated_strategy_name": strategy_name,
        "generated_method_summary": method_summary,
        "generated_key_steps": coerce_key_steps(strategy),
        "generated_final_answer": final_answer,
    }


def extract_strategies(row: dict[str, str]) -> list[dict[str, str]]:
    strategies_json = clean(row.get("strategies_json"))
    strategies: list[dict[str, Any]] = []
    if strategies_json:
        try:
            parsed = json.loads(strategies_json)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            strategies = [item for item in parsed if isinstance(item, dict)]

    if not strategies:
        candidate = best_json_candidate(clean(row.get("response_text")))
        if candidate is not None:
            strategies = strategies_from_candidate(candidate)

    if not strategies:
        strategies = parse_markdown_strategies(clean(row.get("response_text")))

    normalized = [normalize_strategy(item) for item in strategies if isinstance(item, dict)]
    if normalized:
        return normalized

    response_text = clean(row.get("response_text"))
    if not response_text:
        return []
    return [
        {
            "generated_strategy_name": "Unstructured response",
            "generated_method_summary": clean(response_text.split("\n\n", 1)[0]),
            "generated_key_steps": response_text,
            "generated_final_answer": clean(row.get("final_answer")),
        }
    ]


def build_long_rows(rows: list[dict[str, str]], problem_texts: dict[str, str]) -> list[dict[str, str]]:
    long_rows: list[dict[str, str]] = []
    for row in rows:
        if clean(row.get("status")) != "ok":
            continue

        problem_id = clean(row.get("problem_id"))
        strategies = extract_strategies(row)
        if not strategies:
            continue

        for index, strategy in enumerate(strategies, start=1):
            long_rows.append(
                {
                    "problem_id": problem_id,
                    "problem": problem_texts.get(problem_id, ""),
                    "domain": clean(row.get("domain")),
                    "provider": clean(row.get("provider")),
                    "model_label": clean(row.get("model_label")),
                    "model_id": clean(row.get("model_id")),
                    "generated_source": clean(Path(clean(row.get("raw_path"))).parts[2])
                    if clean(row.get("raw_path"))
                    else "",
                    "reponse_text": clean(row.get("response_text")),
                    "strategy_index": str(index),
                    "generated_strategy_name": strategy["generated_strategy_name"],
                    "generated_method_summary": strategy["generated_method_summary"],
                    "generated_key_steps": strategy["generated_key_steps"],
                    "generated_final_answer": strategy["generated_final_answer"],
                }
            )
    return long_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "problem_id",
        "problem",
        "domain",
        "provider",
        "model_label",
        "model_id",
        "generated_source",
        "reponse_text",
        "strategy_index",
        "generated_strategy_name",
        "generated_method_summary",
        "generated_key_steps",
        "generated_final_answer",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_paths = [Path(value) for value in args.input]
    problem_texts = load_problem_texts(Path(args.problems))
    merged_rows = load_and_merge_rows(input_paths)
    long_rows = build_long_rows(merged_rows, problem_texts)
    long_rows.sort(key=lambda row: (row["problem_id"], row["strategy_index"]))
    write_csv(Path(args.output), long_rows)
    print(f"Wrote {len(long_rows)} long-format rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
