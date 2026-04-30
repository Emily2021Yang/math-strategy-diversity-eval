#!/usr/bin/env python3
"""Run a cross-provider math benchmark from CSV/XLSX input.

This script is designed for:
1. A small pilot with one prompt per problem across four models.
2. A later full benchmark with symbolic-only and visualization-heavy tracks.
3. Both single-solution and multi-strategy prompting modes.

Expected input columns:
- Problem_ID (required)
- Problem (required)
- Domain (optional)
- R-level (optional)
- Track (optional; e.g. symbolic / visualization)

Environment variables:
- OPENAI_API_KEY
- GEMINI_API_KEY or GOOGLE_API_KEY
- ANTHROPIC_API_KEY
- DEEPSEEK_API_KEY
- QWEN_API_KEY or DASHSCOPE_API_KEY

Install dependencies if needed:
    pip install pandas openpyxl requests
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# The image prompt attached in chat was not machine-readable here, so keep your
# exact benchmark prompt in one place and preserve the {problem} placeholder.
# Use a plain string replacement later so JSON examples with braces do not need
# escaping like they would with str.format().
SINGLE_SOLUTION_PROMPT = """\
Solve the following math problem.

Provide your answer in the structured format below. Keep explanations concise.

Format:
{
  "final_answer": "...",
  "method_summary": "...",
  "key_steps": ["...", "...", "..."]
}

Problem:
{problem}
"""

MULTI_STRATEGY_PROMPT = """\
Solve the following math problem using multiple distinct strategies.

I will provide the definition of distinct strategies separately. Follow that definition carefully.

Return your response in valid JSON. Use this format:
{
  "final_answer": "...",
  "strategy_count": 0,
  "strategies": [
    {
      "strategy_name": "...",
      "solution": "...",
      "why_distinct": "..."
    }
  ]
}

Problem:
{problem}
"""


DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEEPSEEK_DEFAULT_MAX_OUTPUT_TOKENS = 32768
ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS = 4096

@dataclass(frozen=True)
class ModelSpec:
    label: str
    provider: str
    model_id: str
    default_max_output_tokens: int | None = None


# Recommended study lineup:
# - Use the strongest comparable option from each family for the pilot and the
#   full benchmark, so your design stays consistent.
RECOMMENDED_MODELS = [
    ModelSpec("OpenAI", "openai", "gpt-5.4"),
    ModelSpec("Google", "gemini", "gemini-3.1-pro-preview"),
    ModelSpec(
        "Anthropic",
        "anthropic",
        "claude-opus-4-6",
        ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    ),
    ModelSpec(
        "DeepSeek",
        "deepseek",
        "deepseek-reasoner",
        DEEPSEEK_DEFAULT_MAX_OUTPUT_TOKENS,
    ),
    ModelSpec("Qwen", "qwen", "qwen3-max"),
]


# Lower-cost fallback lineup if you later want a cheaper sensitivity run.
BUDGET_MODELS = [
    ModelSpec("OpenAI", "openai", "gpt-5.4-mini"),
    ModelSpec("Google", "gemini", "gemini-3-flash-preview"),
    ModelSpec(
        "Anthropic",
        "anthropic",
        "claude-sonnet-4-6",
        ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    ),
    ModelSpec(
        "DeepSeek",
        "deepseek",
        "deepseek-reasoner",
        DEEPSEEK_DEFAULT_MAX_OUTPUT_TOKENS,
    ),
    ModelSpec("Qwen", "qwen", "qwen-flash"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="Benchmark/pilot.csv",
        help="Path to a CSV or XLSX file with benchmark problems.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for outputs. Defaults to Benchmark/runs/<timestamp>.",
    )
    parser.add_argument(
        "--profile",
        choices=("recommended", "budget"),
        default="recommended",
        help="Model lineup to use.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of problems, useful for smoke tests.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of model labels/providers to run, e.g. --models Claude DeepSeek.",
    )
    parser.add_argument(
        "--task",
        choices=("single_solution", "multi_strategy"),
        default="single_solution",
        help="Benchmark task mode.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional text file containing the full prompt template. Keep the {problem} placeholder.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional global override for max generated tokens. If omitted, provider defaults are used.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional override for temperature. If omitted, each provider uses its default reasoning setting.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=("low", "medium", "high"),
        default="high",
        help="Reasoning effort for OpenAI responses API calls.",
    )
    parser.add_argument(
        "--anthropic-thinking",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Whether to enable Anthropic extended thinking. Disable it to reduce truncation on concise-output tasks.",
    )
    return parser.parse_args()


def load_problems(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input type: {path.suffix}")

    required_columns = {"Problem_ID", "Problem"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in ["Domain", "R-level", "Track"]:
        if column not in df.columns:
            df[column] = ""

    return df


def build_output_dir(user_output_dir: str | None) -> Path:
    if user_output_dir:
        output_dir = Path(user_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("Benchmark") / "runs" / stamp

    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_models(profile: str) -> list[ModelSpec]:
    return RECOMMENDED_MODELS if profile == "recommended" else BUDGET_MODELS


def filter_models(models: list[ModelSpec], requested: list[str] | None) -> list[ModelSpec]:
    if not requested:
        return models

    alias_map = {
        "claude": "anthropic",
        "anthropic": "anthropic",
        "gemini": "google",
        "google": "google",
        "gpt": "openai",
        "openai": "openai",
        "deepseek": "deepseek",
        "qwen": "qwen",
    }
    requested_normalized = set()
    for item in requested:
        normalized = item.strip().lower().rstrip(",")
        requested_normalized.add(normalized)
        if normalized in alias_map:
            requested_normalized.add(alias_map[normalized])

    filtered = [
        model
        for model in models
        if model.label.lower() in requested_normalized
        or model.provider.lower() in requested_normalized
    ]

    if not filtered:
        raise ValueError(
            "No models matched --models. Available labels/providers: "
            + ", ".join(f"{model.label}/{model.provider}" for model in models)
        )

    return filtered


def api_key_for(provider: str) -> str:
    env_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "qwen": "QWEN_API_KEY",
    }
    key_name = env_map[provider]

    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if provider == "qwen":
        return os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    return os.getenv(key_name, "")


def get_prompt_template(task: str, prompt_file: str | None) -> tuple[str, str]:
    if prompt_file:
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        template = prompt_path.read_text(encoding="utf-8")
        return template, str(prompt_path)

    templates = {
        "single_solution": SINGLE_SOLUTION_PROMPT,
        "multi_strategy": MULTI_STRATEGY_PROMPT,
    }
    return templates[task], f"builtin:{task}"


def build_prompt(problem: str, prompt_template: str) -> str:
    return prompt_template.replace("{problem}", problem.strip())


def post_json(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    try:
        data = response.json()
    except json.JSONDecodeError:
        response.raise_for_status()
        raise RuntimeError(f"Non-JSON response from {url}: {response.text[:500]}")

    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code} from {url}: {json.dumps(data, ensure_ascii=False)}"
        )
    return data


def extract_openai_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]

    chunks: list[str] = []
    for item in data.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and part.get("text"):
                chunks.append(part["text"])
    return "\n".join(chunks).strip()


def extract_openai_finish_reason(data: dict[str, Any]) -> str:
    for item in data.get("output", []):
        finish_reason = item.get("finish_reason")
        if finish_reason:
            return str(finish_reason)
    if data.get("status"):
        return str(data["status"])
    return ""


def call_openai(
    prompt: str,
    model: ModelSpec,
    timeout: int,
    max_output_tokens: int | None,
    temperature: float | None,
    openai_reasoning_effort: str,
) -> dict[str, Any]:
    payload = {
        "model": model.model_id,
        "input": prompt,
        "reasoning": {"effort": openai_reasoning_effort},
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {
        "Authorization": f"Bearer {api_key_for('openai')}",
        "Content-Type": "application/json",
    }
    data = post_json(
        url="https://api.openai.com/v1/responses",
        headers=headers,
        payload=payload,
        timeout=timeout,
    )
    return {
        "text": extract_openai_text(data),
        "finish_reason": extract_openai_finish_reason(data),
        "raw": data,
    }


def extract_gemini_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def extract_gemini_finish_reason(data: dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    return str(candidates[0].get("finishReason", ""))


def call_gemini(
    prompt: str,
    model: ModelSpec,
    timeout: int,
    max_output_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    api_key = api_key_for("gemini")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.model_id}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {},
    }
    if max_output_tokens is not None:
        payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
    if temperature is not None:
        payload["generationConfig"]["temperature"] = temperature
    if not payload["generationConfig"]:
        payload.pop("generationConfig")
    data = post_json(
        url=url,
        headers={"Content-Type": "application/json"},
        payload=payload,
        timeout=timeout,
    )
    return {
        "text": extract_gemini_text(data),
        "finish_reason": extract_gemini_finish_reason(data),
        "raw": data,
    }


def extract_anthropic_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for block in data.get("content", []):
        if block.get("type") == "text" and block.get("text"):
            chunks.append(block["text"])
    return "\n".join(chunks).strip()


def extract_anthropic_finish_reason(data: dict[str, Any]) -> str:
    stop_reason = data.get("stop_reason")
    if stop_reason:
        return str(stop_reason)
    return ""


def call_anthropic(
    prompt: str,
    model: ModelSpec,
    timeout: int,
    max_output_tokens: int | None,
    temperature: float | None,
    anthropic_thinking: str,
) -> dict[str, Any]:
    payload = {
        "model": model.model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }
    if anthropic_thinking == "enabled":
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": min(2048, (max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS) // 2),
        }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    # Anthropic currently requires temperature=1 when extended thinking is enabled.
    if anthropic_thinking == "enabled" and temperature == 1:
        payload["temperature"] = 1
    elif anthropic_thinking == "disabled" and temperature is not None:
        payload["temperature"] = temperature
    headers = {
        "x-api-key": api_key_for("anthropic"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = post_json(
        url="https://api.anthropic.com/v1/messages",
        headers=headers,
        payload=payload,
        timeout=timeout,
    )
    return {
        "text": extract_anthropic_text(data),
        "finish_reason": extract_anthropic_finish_reason(data),
        "raw": data,
    }


def extract_deepseek_text(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning_content = message.get("reasoning_content", "")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                chunks.append(str(item["text"]))
            elif isinstance(item, str):
                chunks.append(item)
        if chunks:
            return "\n".join(chunks).strip()

    return ""


def extract_deepseek_finish_reason(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("finish_reason", ""))


def call_deepseek(
    prompt: str,
    model: ModelSpec,
    timeout: int,
    max_output_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    payload = {
        "model": model.model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {
        "Authorization": f"Bearer {api_key_for('deepseek')}",
        "Content-Type": "application/json",
    }
    data = post_json(
        url="https://api.deepseek.com/chat/completions",
        headers=headers,
        payload=payload,
        timeout=timeout,
    )
    return {
        "text": extract_deepseek_text(data),
        "finish_reason": extract_deepseek_finish_reason(data),
        "raw": data,
    }


def extract_qwen_text(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                chunks.append(str(item["text"]))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks).strip()
    return ""


def extract_qwen_finish_reason(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("finish_reason", ""))


def call_qwen(
    prompt: str,
    model: ModelSpec,
    timeout: int,
    max_output_tokens: int | None,
    temperature: float | None,
) -> dict[str, Any]:
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
    payload = {
        "model": model.model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {
        "Authorization": f"Bearer {api_key_for('qwen')}",
        "Content-Type": "application/json",
    }
    data = post_json(
        url=f"{base_url}/chat/completions",
        headers=headers,
        payload=payload,
        timeout=timeout,
    )
    return {
        "text": extract_qwen_text(data),
        "finish_reason": extract_qwen_finish_reason(data),
        "raw": data,
    }


def call_model(
    prompt: str,
    model: ModelSpec,
    timeout: int,
    max_output_tokens: int | None,
    temperature: float | None,
    openai_reasoning_effort: str,
    anthropic_thinking: str,
) -> dict[str, Any]:
    if model.provider == "openai":
        return call_openai(
            prompt,
            model,
            timeout,
            max_output_tokens,
            temperature,
            openai_reasoning_effort,
        )
    if model.provider == "gemini":
        return call_gemini(prompt, model, timeout, max_output_tokens, temperature)
    if model.provider == "anthropic":
        return call_anthropic(
            prompt,
            model,
            timeout,
            max_output_tokens,
            temperature,
            anthropic_thinking,
        )
    if model.provider == "deepseek":
        return call_deepseek(prompt, model, timeout, max_output_tokens, temperature)
    if model.provider == "qwen":
        return call_qwen(prompt, model, timeout, max_output_tokens, temperature)
    raise ValueError(f"Unsupported provider: {model.provider}")


def effective_max_output_tokens(model: ModelSpec, override: int | None) -> int | None:
    if override is not None:
        return override
    return model.default_max_output_tokens


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_truncated(finish_reason: str) -> bool:
    normalized = finish_reason.strip().lower()
    return normalized in {"length", "max_tokens", "max_output_tokens"}


def parse_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[start_idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def stringify_json(value: Any) -> str:
    if value in (None, "", []):
        return ""
    return json.dumps(value, ensure_ascii=False)


def normalize_structured_output(task: str, response_text: str) -> dict[str, Any]:
    parsed = parse_first_json_object(response_text)
    result = {
        "parsed_json_valid": parsed is not None,
        "parsed_json": stringify_json(parsed) if parsed is not None else "",
        "final_answer": "",
        "method_summary": "",
        "key_steps_json": "",
        "strategy_count_declared": "",
        "strategy_count_parsed": "",
        "strategy_names_json": "",
        "strategies_json": "",
    }
    if parsed is None:
        return result

    result["final_answer"] = str(parsed.get("final_answer", "")).strip()
    result["method_summary"] = str(parsed.get("method_summary", "")).strip()
    result["key_steps_json"] = stringify_json(parsed.get("key_steps", []))

    strategy_count_declared = parsed.get("strategy_count", "")
    if strategy_count_declared != "":
        result["strategy_count_declared"] = strategy_count_declared

    if task == "multi_strategy":
        strategies = parsed.get("strategies")
        if not isinstance(strategies, list):
            for key in ("distinct_strategies", "solutions", "approaches"):
                if isinstance(parsed.get(key), list):
                    strategies = parsed.get(key)
                    break
        if isinstance(strategies, list):
            result["strategy_count_parsed"] = len(strategies)
            result["strategies_json"] = stringify_json(strategies)
            names: list[str] = []
            for strategy in strategies:
                if not isinstance(strategy, dict):
                    continue
                for key in ("strategy_name", "name", "title", "approach"):
                    value = strategy.get(key)
                    if value:
                        names.append(str(value))
                        break
            result["strategy_names_json"] = stringify_json(names)

    return result


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = build_output_dir(args.output_dir)
    raw_dir = output_dir / "raw"
    prompt_template, prompt_source = get_prompt_template(args.task, args.prompt_file)

    problems = load_problems(input_path)
    if args.limit is not None:
        problems = problems.head(args.limit)

    models = filter_models(get_models(args.profile), args.models)

    missing_keys = [provider for provider in {m.provider for m in models} if not api_key_for(provider)]
    if missing_keys:
        print(
            "Missing API keys for providers: "
            + ", ".join(sorted(missing_keys))
            + ". Set the matching environment variables first.",
            file=sys.stderr,
        )
        return 1

    result_rows: list[dict[str, Any]] = []
    jsonl_path = output_dir / "responses.jsonl"

    for _, row in problems.iterrows():
        problem_id = str(row["Problem_ID"])
        problem_text = str(row["Problem"])
        prompt = build_prompt(problem_text, prompt_template)

        for model in models:
            started = time.perf_counter()
            timestamp = datetime.now().isoformat(timespec="seconds")
            raw_name = f"{safe_slug(problem_id)}__{safe_slug(model.label)}.json"
            raw_path = raw_dir / raw_name

            print(f"[{timestamp}] {problem_id} -> {model.label} ({model.model_id})", flush=True)

            record = {
                "timestamp": timestamp,
                "problem_id": problem_id,
                "domain": row.get("Domain", ""),
                "r_level": row.get("R-level", ""),
                "track": row.get("Track", ""),
                "provider": model.provider,
                "model_label": model.label,
                "model_id": model.model_id,
                "task": args.task,
                "prompt_source": prompt_source,
                "temperature": "" if args.temperature is None else args.temperature,
                "max_output_tokens": effective_max_output_tokens(model, args.max_output_tokens),
                "prompt": prompt,
                "response_text": "",
                "parsed_json_valid": False,
                "parsed_json": "",
                "final_answer": "",
                "method_summary": "",
                "key_steps_json": "",
                "strategy_count_declared": "",
                "strategy_count_parsed": "",
                "strategy_names_json": "",
                "strategies_json": "",
                "finish_reason": "",
                "truncated": False,
                "latency_seconds": None,
                "status": "ok",
                "error": "",
                "raw_path": str(raw_path),
            }

            try:
                result = call_model(
                    prompt=prompt,
                    model=model,
                    timeout=args.timeout,
                    max_output_tokens=effective_max_output_tokens(model, args.max_output_tokens),
                    temperature=args.temperature,
                    openai_reasoning_effort=args.openai_reasoning_effort,
                    anthropic_thinking=args.anthropic_thinking,
                )
                latency = round(time.perf_counter() - started, 2)
                record["latency_seconds"] = latency
                record["response_text"] = result["text"]
                record.update(normalize_structured_output(args.task, result["text"]))
                record["finish_reason"] = result.get("finish_reason", "")
                record["truncated"] = is_truncated(record["finish_reason"])
                write_json(raw_path, result["raw"])
            except Exception as exc:  # noqa: BLE001
                latency = round(time.perf_counter() - started, 2)
                record["latency_seconds"] = latency
                record["status"] = "error"
                record["error"] = str(exc)
                write_json(raw_path, {"error": str(exc)})

            result_rows.append(record)

            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            time.sleep(args.sleep_seconds)

    output_csv = output_dir / "responses.csv"
    pd.DataFrame(result_rows).to_csv(output_csv, index=False)

    summary_rows = (
        pd.DataFrame(result_rows)
        .groupby(["model_label", "model_id", "status"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    summary_rows.to_csv(output_dir / "summary.csv", index=False)

    print(f"Saved outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
