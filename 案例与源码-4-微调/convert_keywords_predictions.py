#!/usr/bin/env python3
"""转换本次单轮 Qwen3 WebUI 预测；只核对结构，不清洗或改写回答。

仅接受已归档的 user→assistant 包装。其他模板、system 消息或多轮对话
应先核对实际文件，再编写对应规则；本脚本不猜测或按行号盲目配对。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_records(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> list[dict[str, str]]:
    expected: dict[str, tuple[str, str]] = {}
    for record in references:
        conversations = record.get("conversations") if isinstance(record, dict) else None
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError("参考数据必须是单轮 user、assistant 对话")
        user, assistant = conversations
        if (not isinstance(user, dict) or not isinstance(assistant, dict)
                or user.get("role") != "user" or assistant.get("role") != "assistant"):
            raise ValueError("参考数据角色不匹配")
        text, answer = user.get("content"), assistant.get("content")
        if not isinstance(text, str) or not text.strip() or not isinstance(answer, str):
            raise ValueError("参考数据 content 必须是字符串，输入不能为空")
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        if prompt in expected:
            raise ValueError("参考数据存在重复输入，无法唯一匹配")
        expected[prompt] = (text, answer)
    if not expected:
        raise ValueError("参考数据为空")

    converted: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(predictions, start=1):
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) for key in ("prompt", "predict", "label")
        ):
            raise ValueError(f"预测第 {index} 条必须包含字符串 prompt、predict、label")
        prompt = record["prompt"]
        if prompt not in expected:
            raise ValueError(f"预测第 {index} 条的输入或模板无法精确匹配")
        if prompt in seen:
            raise ValueError(f"预测第 {index} 条重复")
        text, answer = expected[prompt]
        # 本次 WebUI 的 label 末尾附有一个换行，只允许这一已知差异。
        if record["label"] not in (answer, answer + "\n"):
            raise ValueError(f"预测第 {index} 条的 label 与参考答案不一致")
        seen.add(prompt)
        converted.append({"input": text, "prediction": record["predict"]})
    if len(seen) != len(expected):
        raise ValueError(f"缺少预测：{len(expected) - len(seen)} 条")
    return converted


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converted = convert_records(read_jsonl(args.reference), read_jsonl(args.predictions))
    # 不覆盖原始预测或任何已有文件；全部核验成功后才创建结果。
    with args.output.open("x", encoding="utf-8") as output:
        for record in converted:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已匹配并转换 {len(converted)} 条；predict 原文未改写：{args.output}")


if __name__ == "__main__":
    main()
