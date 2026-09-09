"""本次诊断专用：核对两种已验证的输入包装，原样保留模型回答。"""
import argparse
import importlib.util
from pathlib import Path


EMPTY_THOUGHT = "<think>\n\n</think>\n\n"
_spec = importlib.util.spec_from_file_location(
    "course_legacy_converter", Path(__file__).resolve().parents[2] / "convert_keywords_predictions.py"
)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)


def convert_records(references, predictions, template):
    if template not in ("qwen3_nothink", "qwen3"):
        raise ValueError("只接受本实验核验过的 qwen3_nothink 和 qwen3 包装")
    wrapped = []
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError("预测记录必须包含字符串 prompt")
        prompt = row["prompt"]
        if template == "qwen3":
            if not prompt.endswith(EMPTY_THOUGHT):
                raise ValueError("qwen3 输入缺少本次核验的完整空思考区块")
            # 仅移除输入包装的已知后缀，以复用原转换器的严格配对；predict 不变。
            prompt = prompt[:-len(EMPTY_THOUGHT)]
        wrapped.append({**row, "prompt": prompt})
    return _legacy.convert_records(references, wrapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--template", choices=["qwen3_nothink", "qwen3"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = convert_records(_legacy.read_jsonl(args.reference),
                           _legacy.read_jsonl(args.predictions), args.template)
    import json
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已严格匹配 {len(rows)} 条；原始回答未改写：{args.output}")


if __name__ == "__main__":
    main()
