"""第30章预处理核验：调用已安装的 LLaMA-Factory，不加载模型权重、不训练。

在 LLaMA-Factory 的 Python 环境运行；输入文件只读，输出目录必须不存在。
仅适用于本课程单轮、纯文本、非 packing 的关键词数据。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def inspect_sequence(full, cut, cutoff):
    for row in (full, cut):
        if len(row["input_ids"]) != len(row["labels"]):
            raise ValueError("input_ids/labels length mismatch")
        if any(label != -100 and label != token for token, label in zip(row["input_ids"], row["labels"])):
            raise ValueError("label differs from its input token")
    if len(cut["input_ids"]) > cutoff:
        raise ValueError("sequence exceeds cutoff")
    target = [token for token in cut["labels"] if token != -100]
    full_target = [token for token in full["labels"] if token != -100]
    return {
        "full_length": len(full["input_ids"]),
        "length": len(cut["input_ids"]),
        "truncated": full["input_ids"] != cut["input_ids"],
        "ignored_tokens": cut["labels"].count(-100),
        "supervised_tokens": len(target),
        "target_preserved": target == full_target,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("输出目录已存在，请使用新目录；不会覆盖已有结果。")
    if not (args.model_dir / "tokenizer.json").is_file():
        parser.error("请提供已经下载好的模型目录；本脚本不会下载模型。")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import dataclasses
    import importlib.metadata
    import inspect
    import platform
    import subprocess
    from datetime import datetime, timezone

    import yaml
    from transformers import Seq2SeqTrainingArguments
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    required = {
        "model_name_or_path": "Qwen/Qwen3-0.6B", "template": "qwen3_nothink",
        "enable_thinking": False, "packing": False, "train_on_prompt": False,
        "dataset": "keywords_train", "eval_dataset": "keywords_validation", "val_size": 0,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"本课程核验条件不符: {key}={config.get(key)!r}")
    raw = {}
    fingerprints = {}
    for name in ("train", "validation"):
        path = args.data_dir / f"keywords_{name}.jsonl"
        fingerprints[path.name] = sha256(path)
        raw[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in raw[name]:
            messages = row["conversations"]
            if [message["role"] for message in messages] != ["user", "assistant"]:
                raise ValueError("本脚本仅核验课程的单轮 user/assistant 数据。")
    fingerprints["dataset_info.json"] = sha256(args.data_dir / "dataset_info.json")
    before_config = sha256(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_fields = {field.name for field in dataclasses.fields(ModelArguments)}
    model_config = {key: value for key, value in config.items() if key in model_fields}
    model_config.update(model_name_or_path=str(args.model_dir.resolve()), cache_dir=str(args.output_dir / "cache"))
    model_args = ModelArguments(**model_config)
    module = load_tokenizer(model_args)
    tokenizer = module["tokenizer"]
    data_fields = {field.name for field in dataclasses.fields(DataArguments)}
    data_config = {key: value for key, value in config.items() if key in data_fields}
    data_config.update(dataset_dir=str(args.data_dir.resolve()), preprocessing_num_workers=1,
                       preprocessing_batch_size=64, overwrite_cache=True, tokenized_path=None)
    data_args = DataArguments(**data_config)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir / "unused-training-output"), use_cpu=True,
        do_train=False, do_eval=False, do_predict=False, predict_with_generate=False,
        fp16=False, bf16=False, report_to=[], seed=config["seed"],
    )
    cutoff = data_args.cutoff_len
    uncut_limit = 1_000_000  # 仅用于预处理对照，不会将该长度用于模型训练。
    data_args.cutoff_len = uncut_limit
    full = get_dataset(template, model_args, data_args, training_args, "sft", **module)
    data_args.cutoff_len = cutoff
    cut = get_dataset(template, model_args, data_args, training_args, "sft", **module)
    rows = []
    stats = {}
    for name, key in (("train", "train_dataset"), ("validation", "eval_dataset")):
        if len(full[key]) != len(raw[name]) or len(cut[key]) != len(raw[name]):
            raise ValueError(f"{name}: 原始/完整/截断后条数不同，需要单独检查被丢弃的样本。")
        group = []
        for index in range(len(raw[name])):
            result = inspect_sequence(full[key][index], cut[key][index], cutoff)
            if result["full_length"] >= uncut_limit:
                raise ValueError("完整序列仍触及上限，不能当作未截断对照。")
            result.update(split=name, line=index + 1)
            group.append(result)
        stats[name] = {
            "raw_count": len(raw[name]), "processed_count": len(cut[key]), "dropped_count": 0,
            "min_length": min(row["length"] for row in group),
            "max_length": max(row["length"] for row in group),
            "max_full_length": max(row["full_length"] for row in group),
            "truncated_count": sum(row["truncated"] for row in group),
            "target_changed_count": sum(not row["target_preserved"] for row in group),
            "all_ignored_count": sum(row["supervised_tokens"] == 0 for row in group),
        }
        rows.extend(group)
    ordered = sorted(rows, key=lambda row: (row["full_length"], row["split"], row["line"]))
    examples = []
    for size, item in zip(("short", "medium", "long"), (ordered[0], ordered[len(ordered) // 2], ordered[-1])):
        key = "train_dataset" if item["split"] == "train" else "eval_dataset"
        sequence = cut[key][item["line"] - 1]
        target_ids = [token for token in sequence["labels"] if token != -100]
        valid = [i for i, value in enumerate(sequence["labels"]) if value != -100]
        boundary = valid[0] if valid else len(sequence["labels"])
        positions = sorted(set(range(max(0, boundary - 4), min(len(sequence["labels"]), boundary + 5)))
                           | set(range(max(0, len(sequence["labels"]) - 3), len(sequence["labels"]))))
        examples.append({
            **item, "size": size, "original": raw[item["split"]][item["line"] - 1],
            "input_ids": sequence["input_ids"], "labels": sequence["labels"],
            "decoded_input": tokenizer.decode(sequence["input_ids"], skip_special_tokens=False),
            "decoded_target": tokenizer.decode(target_ids, skip_special_tokens=False),
            "boundary": [{"position": i, "token_id": sequence["input_ids"][i],
                          "token": tokenizer.convert_ids_to_tokens(sequence["input_ids"][i]),
                          "label": sequence["labels"][i]} for i in positions],
        })
    for filename, fingerprint in fingerprints.items():
        if sha256(args.data_dir / filename) != fingerprint:
            raise ValueError("核验过程中输入文件发生变化，请重新核验。")
    if sha256(args.config) != before_config:
        raise ValueError("核验过程中配置发生变化。")
    source_root = Path(inspect.getfile(get_dataset)).resolve().parents[3]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "model_id": config["model_name_or_path"],
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("llamafactory", "transformers", "datasets", "torch")},
        "llamafactory_commit": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "source_diffs": subprocess.check_output(["git", "-C", str(source_root), "diff", "HEAD", "--stat", "--", "src/llamafactory/data", "src/llamafactory/model"], text=True),
        "data_sha256": fingerprints, "config_sha256": before_config,
        "tokenizer_sha256": {name: sha256(args.model_dir / name) for name in ("tokenizer.json", "tokenizer_config.json", "config.json")},
        "settings": {**required, "cutoff_len": cutoff, "max_samples": config["max_samples"]},
        "audit_only_overrides": {"device": "cpu", "preprocessing_num_workers": 1, "preprocessing_batch_size": 64,
                                 "cache": "new isolated directory", "model_weights_loaded": False,
                                 "training_started": False, "test_set_read": False},
        "stats": stats, "examples": examples,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "lengths.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print("\nPREPROCESSING AUDIT COMPLETE — CPU only; no model weights; no training")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for sample in examples:
        print(f"{sample['size']}: {sample['split']} line {sample['line']}, length={sample['length']}, ignored={sample['ignored_tokens']}, supervised={sample['supervised_tokens']}")
        print("target:", repr(sample["decoded_target"]))


if __name__ == "__main__":
    main()
