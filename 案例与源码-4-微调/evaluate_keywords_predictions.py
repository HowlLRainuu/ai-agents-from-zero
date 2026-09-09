#!/usr/bin/env python3
"""对关键词抽取模型的预测结果进行可复现评估。

参考数据使用 LLaMA Factory 的 ShareGPT JSONL 格式；预测文件使用每行一个
{"input": "...", "prediction": "..."} 的 JSONL 格式。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _normalise_input(value: str) -> str:
    """让换行和多余空白不影响输入样本的匹配。"""

    return " ".join(value.split())


def _normalise_keyword(value: str) -> str:
    return " ".join(value.strip().split())


def parse_keywords(text: str) -> list[str]:
    """把英文或中文分号分隔的关键词转为去重后的列表。

    此处只用于内容评分，所以允许中文分号；格式合规性由
    is_valid_prediction_format 函数单独判断。
    """

    if not isinstance(text, str):
        return []

    keywords: list[str] = []
    seen: set[str] = set()
    for part in text.replace("；", ";").split(";"):
        keyword = _normalise_keyword(part)
        if keyword and keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)
    return keywords


def is_valid_prediction_format(text: str) -> bool:
    """检查关键词输出中可确定的形式规则。

    拒绝空项、中文分号、重复词、换行、常见答案前缀和列表编号。
    通过不代表每一项都是关键词：普通解释句和语义正确性仍须人工检查。
    """

    if not isinstance(text, str) or not text or text != text.strip():
        return False
    if "\n" in text or "\r" in text or "；" in text:
        return False
    if re.match(r"^(?:关键词(?:识别)?(?:如下)?|下面是答案|答案)\s*[:：]", text):
        return False

    parts = text.split(";")
    if any(not part.strip() for part in parts):
        return False
    # 不把 3.5B 这类数值关键词的句点误认成列表编号。
    list_marker = r"^(?:\d+[.)、](?!\d)|[（(]\d+[）)]|[-*•]\s+)"
    for part in parts:
        keyword = part.strip()
        # (110)取向、(111)定向金刚石薄膜中的数字是晶面/晶向标记，
        # 不是列表编号。只放行这一明确语境，孤立的 (10) 仍需检查。
        if re.match(r"^[（(]\d{3}[）)](?:取向|定向|晶面|晶向)", keyword):
            continue
        if re.match(list_marker, keyword):
            return False

    keywords = parse_keywords(text)
    return bool(keywords) and len(keywords) == len(parts)


def _as_conversations(record: dict[str, Any]) -> tuple[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("参考数据缺少完整 conversations 字段")

    user, assistant = conversations[0], conversations[1]
    if not isinstance(user, dict) or not isinstance(assistant, dict):
        raise ValueError("参考数据中的 conversations 格式不正确")
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise ValueError("参考数据应以 user 和 assistant 对话组成")

    input_text = user.get("content")
    reference = assistant.get("content")
    if not isinstance(input_text, str) or not isinstance(reference, str):
        raise ValueError("参考数据中的 content 必须是字符串")
    return input_text, reference


def _reference_index(records: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for record in records:
        input_text, reference = _as_conversations(record)
        key = _normalise_input(input_text)
        if not key:
            raise ValueError("参考数据中存在空输入")
        if key in index:
            raise ValueError("参考数据中存在重复输入，无法可靠匹配预测结果")
        if not is_valid_prediction_format(reference):
            raise ValueError("参考答案未通过格式检查，请先核对参考数据，不要清洗模型预测来代替")
        index[key] = (input_text, reference)
    return index


def _score(expected: set[str], predicted: set[str]) -> tuple[float, float, float]:
    correct = len(expected & predicted)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_records(
    references: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """根据参考答案和预测结果生成一份评估报告。

    require_all=True 适合正式测试：只要缺少一个测试输入的预测，就直接报错，
    避免把“只跑了一部分样本”误当成完整评估。
    """

    reference_by_input = _reference_index(references)
    evaluated_keys: set[str] = set()
    items: list[dict[str, Any]] = []

    for record in predictions:
        input_text = record.get("input")
        prediction = record.get("prediction")
        if not isinstance(input_text, str) or not isinstance(prediction, str):
            raise ValueError("预测文件的每一行都必须包含字符串 input 和 prediction")

        key = _normalise_input(input_text)
        if key not in reference_by_input:
            raise ValueError("预测文件包含测试集之外的输入")
        if key in evaluated_keys:
            raise ValueError("预测文件包含重复输入")
        evaluated_keys.add(key)

        original_input, reference = reference_by_input[key]
        expected = set(parse_keywords(reference))
        actual = set(parse_keywords(prediction))
        precision, recall, f1 = _score(expected, actual)
        format_valid = is_valid_prediction_format(prediction)
        items.append(
            {
                "input": original_input,
                "reference": reference,
                "prediction": prediction,
                "format_valid": format_valid,
                "exact_match": expected == actual,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "missing_keywords": sorted(expected - actual),
                "extra_keywords": sorted(actual - expected),
            }
        )

    missing_count = len(reference_by_input) - len(evaluated_keys)
    if require_all and missing_count:
        raise ValueError(f"缺少预测: 还有 {missing_count} 条测试输入未被评估")

    evaluated_samples = len(items)
    if not evaluated_samples:
        raise ValueError("预测文件为空，没有可评估的样本")

    def average(field: str) -> float:
        return sum(item[field] for item in items) / evaluated_samples

    format_failures = sum(not item["format_valid"] for item in items)
    exact_matches = sum(item["exact_match"] for item in items)
    return {
        "evaluated_samples": evaluated_samples,
        "reference_samples": len(reference_by_input),
        "missing_predictions": missing_count,
        "format_failures": format_failures,
        "format_pass_rate": (evaluated_samples - format_failures) / evaluated_samples,
        "exact_match_rate": exact_matches / evaluated_samples,
        "macro_precision": average("precision"),
        "macro_recall": average("recall"),
        "macro_f1": average("f1"),
        "items": items,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
            records.append(record)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估关键词抽取模型的预测结果")
    parser.add_argument("--reference", type=Path, required=True, help="keywords_test.jsonl 路径")
    parser.add_argument("--predictions", type=Path, required=True, help="模型预测 JSONL 路径")
    parser.add_argument("--output", type=Path, required=True, help="评估报告 JSON 输出路径")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="要求预测覆盖参考数据中的全部测试输入",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = evaluate_records(
        read_jsonl(args.reference),
        read_jsonl(args.predictions),
        require_all=args.require_all,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"评估样本：{report['evaluated_samples']} / {report['reference_samples']}")
    print(f"格式通过率：{report['format_pass_rate']:.2%}")
    print(f"完全匹配率：{report['exact_match_rate']:.2%}")
    print(f"Macro Precision：{report['macro_precision']:.4f}")
    print(f"Macro Recall：{report['macro_recall']:.4f}")
    print(f"Macro F1：{report['macro_f1']:.4f}")
    print(f"报告已写入：{args.output}")


if __name__ == "__main__":
    main()
