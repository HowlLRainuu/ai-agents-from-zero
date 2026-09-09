"""按课程 YAML 执行 LLaMA-Factory 预测，同时记录实际精度与数据指纹。

在 LLaMA-Factory 根目录、已安装的训练环境中运行。
不修改模型加载器，不清洗预测文字，不覆盖已有输出目录。
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    # 只影响本进程；不修改系统或既有环境文件。
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import importlib.metadata
    import platform
    import torch
    import yaml
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("do_train") or not config.get("do_predict") or not config.get("predict_with_generate"):
        parser.error("只允许纯生成预测，不执行训练。")
    if not config.get("fp16_full_eval") or config.get("bf16") or config.get("bf16_full_eval"):
        parser.error("本课程核验要求 fp16_full_eval: true，且不启用 BF16。")
    output = Path(config["output_dir"])
    if output.exists():
        parser.error("输出目录已存在；请使用新目录，不覆盖已有结果。")
    registry = json.loads((Path(config["dataset_dir"]) / "dataset_info.json").read_text())
    data = Path(config["dataset_dir"]) / registry[config["eval_dataset"]]["file_name"]
    before_data = fingerprint(data)
    expected = min(sum(bool(line.strip()) for line in data.read_text().splitlines()), config["max_samples"])
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_sha256": fingerprint(args.config),
        "data_sha256": before_data,
        "expected_samples": expected,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("llamafactory", "transformers", "torch")},
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "prediction_steps": 0,
    }
    adapter = config.get("adapter_name_or_path")
    if adapter:
        record["adapter_sha256"] = fingerprint(Path(adapter) / "adapter_model.safetensors")

    class DtypeEvidence(TrainerCallback):
        def on_prediction_step(self, args, state, control, model=None, **kwargs):
            record["prediction_steps"] += 1
            if record["prediction_steps"] == 1:
                counts = Counter()
                for parameter in model.parameters():
                    if parameter.is_floating_point():
                        counts[str(parameter.dtype)] += parameter.numel()
                record["actual_parameter_dtypes"] = dict(counts)
                if set(counts) != {"torch.float16"}:
                    raise RuntimeError(f"实际预测精度不符合要求：{dict(counts)}")
                record["model_class"] = type(model).__name__
                record["model_path"] = getattr(model.config, "_name_or_path", None)
                print("ACTUAL PREDICTION DTYPES:", dict(counts), flush=True)

    run_exp(args=config, callbacks=[DtypeEvidence()])
    prediction_file = output / "generated_predictions.jsonl"
    actual = sum(bool(line.strip()) for line in prediction_file.read_text().splitlines())
    if actual != expected or fingerprint(data) != before_data:
        raise RuntimeError("预测条数不完整，或执行期间数据发生变化。")
    if not record.get("actual_parameter_dtypes"):
        raise RuntimeError("未取得实际推理精度记录。")
    record.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  actual_samples=actual, predictions_sha256=fingerprint(prediction_file))
    with (output / "prediction-evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"PREDICTION VERIFIED: {actual}/{expected}; FP16; {output}", flush=True)


if __name__ == "__main__":
    main()
