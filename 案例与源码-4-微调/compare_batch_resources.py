"""Compare two single-GPU LoRA batch arrangements without changing dependencies.

Run from the LLaMA-Factory root with its existing environment activated:
  python compare_batch_resources.py --config keywords_clean_train.yaml \
      --output-root /root/autodl-tmp/ch33-batch-comparison --steps 50

The output root must not exist. This is a resource trial, not a quality benchmark.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml


def read_command(*args):
    return subprocess.check_output(args, text=True).strip()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stop_process_group(process):
    """Stop the Linux training launcher and any torchrun workers it started."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The launcher may exit before its workers; clean up the group either way.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_training(config_path, log, env, timeout=600):
    process = subprocess.Popen(
        ["llamafactory-cli", "train", str(config_path)],
        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124
    finally:
        # Also runs on Ctrl+C, before the GPU monitor or next trial can finish.
        stop_process_group(process)


def make_config(base, batch, accumulation, steps, output):
    config = dict(base)
    config.update(
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accumulation,
        max_steps=steps,
        save_steps=steps,
        eval_steps=steps,
        skip_memory_metrics=False,
        output_dir=str(output),
    )
    for field in ("adapter_name_or_path", "resume_from_checkpoint"):
        config.pop(field, None)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    root = args.output_root.resolve()
    if root.exists():
        parser.error("output root already exists; choose a new directory")
    base = yaml.safe_load(args.config.read_text())
    if base.get("model_name_or_path") != "Qwen/Qwen3-0.6B":
        parser.error("this course trial is restricted to Qwen/Qwen3-0.6B")
    if base.get("finetuning_type") != "lora" or not base.get("fp16") or base.get("bf16"):
        parser.error("expected ordinary LoRA with fp16=true and bf16=false")
    if base.get("quantization_bit") or base.get("deepspeed"):
        parser.error("quantization and DeepSpeed are outside this batch comparison")
    gpu = read_command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")
    if len(gpu.splitlines()) != 1 or "V100" not in gpu:
        parser.error("expected exactly one V100 GPU")
    processes = read_command("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader")
    if processes:
        parser.error("GPU already has a compute process; inspect it before training")
    expected = {
        "keywords_train.jsonl": "f1f6077f0cdd781871f088ba51f44f30aec004e98a949fb2701ec61b719ca942",
        "keywords_validation.jsonl": "017372b0de86473c9267a3268c19b33a67009c94bfc3bc93529f97ae01cdf9ad",
    }
    data_dir = Path(base["dataset_dir"])
    hashes = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in expected}
    if hashes != expected:
        parser.error("course data fingerprints differ; inspect data before running")
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="4", USE_MODELSCOPE_HUB="1", PYTHONUNBUFFERED="1")
    report = {
        "gpu": gpu,
        "llamafactory_commit": read_command("git", "rev-parse", "HEAD"),
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "llamafactory", "peft")},
        "data_sha256": hashes,
        "effective_batch": 32,
        "update_steps": args.steps,
        "sampled_memory_scope": "whole CLI process: loading, training, validation and saving; nvidia-smi sampled every 200 ms",
        "sampled_memory_limit": "highest observed sample, not the true instantaneous peak",
        "quality_limit": "short resource trial; validation loss is not a keyword quality evaluation",
        "runs": [],
    }
    save_json(root / "comparison.json", report)
    for name, batch, accumulation in (("batch4-acc8", 4, 8), ("batch2-acc16", 2, 16)):
        run_dir = root / name
        run_dir.mkdir()
        config = make_config(base, batch, accumulation, args.steps, run_dir / "model")
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        print(f"START {name}: batch={batch}, accumulation={accumulation}, effective_batch=32, steps={args.steps}", flush=True)
        started = time.monotonic()
        with (run_dir / "gpu-samples.csv").open("w") as samples, (run_dir / "train.log").open("w") as log:
            monitor = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits", "-lms", "200",
            ], stdout=samples, stderr=subprocess.DEVNULL)
            try:
                code = run_training(config_path, log, env)
            finally:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
        elapsed = time.monotonic() - started
        rows = list(csv.reader((run_dir / "gpu-samples.csv").read_text().splitlines()))
        memory = [float(row[1]) for row in rows if len(row) == 4]
        results_path = run_dir / "model" / "all_results.json"
        trainer_results = json.loads(results_path.read_text()) if results_path.exists() else {}
        result = {
            "name": name, "batch": batch, "accumulation": accumulation,
            "return_code": code, "cli_wall_seconds": elapsed,
            "gpu_sample_count": len(memory),
            "highest_sampled_gpu_memory_mib": max(memory) if memory else None,
            "trainer_results": trainer_results,
            "checkpoint_saved": (run_dir / "model" / f"checkpoint-{args.steps}" / "trainer_state.json").is_file(),
        }
        report["runs"].append(result)
        save_json(root / "comparison.json", report)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if code:
            print("STOP: inspect train.log; no automatic dependency changes or retries.", flush=True)
            return code
    print("DONE: both trials finished; inspect comparison.json and the original logs.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
