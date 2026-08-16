from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_generation import make_evidence, score_case, summarize, validate_prompt_contract
from app.services.llm import REFUSAL_TEXT, build_messages

DEFAULT_DATASET = Path("evals/generation_reliability_v1.json")
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def generate_answer(
    tokenizer,
    model,
    question: str,
    evidence,
    max_new_tokens: int,
) -> tuple[str, float]:
    if not evidence:
        return REFUSAL_TEXT, 0.0

    messages = build_messages(question, evidence)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default="generation_benchmark_local.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = payload["cases"]
    contract = validate_prompt_contract(cases)
    if not contract["pass"]:
        raise SystemExit("Prompt contract validation failed")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    scored = []
    latencies = []
    for case in cases:
        evidence = make_evidence(case.get("evidence", []))
        answer, latency_ms = generate_answer(
            tokenizer,
            model,
            case["question"],
            evidence,
            args.max_new_tokens,
        )
        row = score_case(case, answer, evidence)
        row["generation_ms"] = round(latency_ms, 2)
        scored.append(row)
        latencies.append(latency_ms)
        print(f"{case['id']} | pass={row['pass']} | {answer}")

    summary = summarize(scored)
    summary["mean_generation_ms"] = sum(latencies) / len(latencies)
    summary["model"] = args.model

    output = {
        "benchmark": payload["name"],
        "dataset": args.dataset,
        "model": args.model,
        "prompt_contract": contract,
        "summary": summary,
        "cases": scored,
    }
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
