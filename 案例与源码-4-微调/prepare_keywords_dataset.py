#!/usr/bin/env python3
"""清洗课程关键词数据，并稳定划分训练、验证和测试集。

与关键词 JSONL 数据放在同一目录，供第 29 章直接运行和复现。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from evaluate_keywords_predictions import is_valid_prediction_format


MATERIALS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = MATERIALS_DIR / "keywords_data_sharegpt_small.jsonl"
DEFAULT_OUTPUT_DIR = MATERIALS_DIR / "processed" / "keywords-clean"
DEFAULT_SEED = "chapter-29-v1"


def normalize_text(text: str) -> str:
    """合并连续空白，供输入去重使用。"""
    return re.sub(r"\s+", " ", text).strip()


def normalize_keywords(answer: str) -> str:
    """去掉答案前缀，统一分号与空白，并按原顺序删除重复关键词。"""
    answer = re.sub(r"^\s*关键词\s*[:：]\s*", "", answer)
    keywords: list[str] = []
    seen: set[str] = set()

    for item in answer.replace("；", ";").split(";"):
        keyword = item.strip()
        if keyword and keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)

    if not keywords:
        raise ValueError("关键词答案为空")

    return ";".join(keywords)


def clean_record(record: object, line_number: int) -> tuple[str, str, dict, bool]:
    """检查一条 user -> assistant 样本，并返回清洗后的记录。"""
    if not isinstance(record, dict):
        raise ValueError(f"第 {line_number} 行不是 JSON 对象")

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"第 {line_number} 行必须有两条 conversations 消息")

    expected_roles = ("user", "assistant")
    for index, (message, expected_role) in enumerate(
        zip(conversations, expected_roles), start=1
    ):
        if not isinstance(message, dict):
            raise ValueError(f"第 {line_number} 行第 {index} 条消息不是对象")
        if message.get("role") != expected_role:
            raise ValueError(
                f"第 {line_number} 行角色顺序错误，应为 user -> assistant"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"第 {line_number} 行第 {index} 条消息内容为空")

    cleaned = copy.deepcopy(record)
    original_input = conversations[0]["content"]
    original_answer = conversations[1]["content"]
    cleaned_input = original_input.strip()
    cleaned_answer = normalize_keywords(original_answer)

    cleaned["conversations"][0]["content"] = cleaned_input
    cleaned["conversations"][1]["content"] = cleaned_answer

    return (
        normalize_text(cleaned_input),
        normalize_text(cleaned_answer),
        cleaned,
        cleaned_answer != original_answer,
    )


def load_and_clean(path: Path, report: dict | None = None) -> tuple[list[dict], int, int]:
    """读取 JSONL；相同输入同答案去重，冲突答案直接停止。"""
    unique_records: dict[str, tuple[str, dict]] = {}
    duplicate_count = 0
    normalized_count = 0
    audit = report if report is not None else {}
    audit.update(raw_records=0, changes=[], duplicates=[], review_required=[], _source_lines=[])
    first_lines: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                raise ValueError(f"第 {line_number} 行为空行")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} 第 {line_number} 行不是合法 JSON") from exc
            input_key, answer_key, cleaned, changed = clean_record(
                record, line_number
            )
            normalized_count += int(changed)
            audit["raw_records"] += 1
            input_id = hashlib.sha256(input_key.encode("utf-8")).hexdigest()
            if cleaned != record:
                audit["changes"].append({
                    "source_line": line_number,
                    "input_sha256": input_id,
                    "before": record["conversations"][1]["content"],
                    "after": cleaned["conversations"][1]["content"],
                    "input_trimmed": record["conversations"][0]["content"] != cleaned["conversations"][0]["content"],
                })
            if not is_valid_prediction_format(cleaned["conversations"][1]["content"]):
                audit["review_required"].append({
                    "source_line": line_number,
                    "input_sha256": input_id,
                    "answer": cleaned["conversations"][1]["content"],
                    "reason": "自动规范化后仍未通过格式检查；需结合原文人工复核，也可能是规则误判",
                })

            if input_key in unique_records:
                previous_answer, _ = unique_records[input_key]
                if answer_key != previous_answer:
                    raise ValueError(
                        f"第 {line_number} 行与已有输入重复，但答案冲突"
                    )
                duplicate_count += 1
                audit["duplicates"].append({"source_line": line_number, "kept_source_line": first_lines[input_key]})
                continue

            unique_records[input_key] = (answer_key, cleaned)
            first_lines[input_key] = line_number
            audit["_source_lines"].append(line_number)

    audit.update(clean_records=len(unique_records), normalized_answers=normalized_count, removed_duplicates=duplicate_count)
    return [record for _, record in unique_records.values()], duplicate_count, normalized_count


def stable_order_key(record: dict, seed: str) -> str:
    """根据固定种子和标准化输入生成稳定排序键。"""
    input_text = normalize_text(record["conversations"][0]["content"])
    payload = f"{seed}\0{input_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as target_file:
        for record in records:
            target_file.write(json.dumps(record, ensure_ascii=False))
            target_file.write("\n")


def input_keys(records: list[dict]) -> set[str]:
    return {
        normalize_text(record["conversations"][0]["content"])
        for record in records
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError(f"输出目录已存在：{args.output_dir}。保留冻结版本，另选新目录；复现时也不要覆盖原件。")
    report: dict = {}
    records, duplicate_count, normalized_count = load_and_clean(args.source, report)
    source_lines = dict(zip((normalize_text(row["conversations"][0]["content"]) for row in records), report.pop("_source_lines")))
    source_info = {"file": args.source.name, "records": report["raw_records"], "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest()}
    report["source"] = source_info
    report["semantic_review"] = "未逐条审核答案语义；格式检查不能替代标注审核"
    report["dedup_scope"] = "完整 user.content 合并连续空白后的精确匹配；不包含语义近重复检查"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "cleaning_report.json", report)
    if report["review_required"]:
        raise ValueError(f"有 {len(report['review_required'])} 条答案需人工复核。仅保存 cleaning_report.json，未生成训练文件。")
    if len(records) < 10:
        raise ValueError("不足 10 条，不生成可能为空的三份划分。请先补足样本。")
    records.sort(key=lambda record: stable_order_key(record, args.seed))

    train_end = int(len(records) * 0.8)
    validation_end = train_end + int(len(records) * 0.1)
    splits = {
        "keywords_train.jsonl": records[:train_end],
        "keywords_validation.jsonl": records[train_end:validation_end],
        "keywords_test.jsonl": records[validation_end:],
    }

    train_keys = input_keys(splits["keywords_train.jsonl"])
    validation_keys = input_keys(splits["keywords_validation.jsonl"])
    test_keys = input_keys(splits["keywords_test.jsonl"])
    assert train_keys.isdisjoint(validation_keys)
    assert train_keys.isdisjoint(test_keys)
    assert validation_keys.isdisjoint(test_keys)

    manifest = {
        "schema_version": 1,
        "data_version": args.output_dir.name,
        "source": source_info,
        "split_seed": args.seed,
        "split_method": "SHA-256(seed + NUL + normalized user input) 排序，按 80/10/10 划分",
        "script_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            Path(__file__), MATERIALS_DIR / "evaluate_keywords_predictions.py"
        )},
        "splits": {},
        "cross_split_duplicate_inputs": 0,
        "review_scope": report["semantic_review"] + "；" + report["dedup_scope"],
    }
    registry = {}
    for file_name, split_records in splits.items():
        target = args.output_dir / file_name
        write_jsonl(target, split_records)
        split_name = file_name.removeprefix("keywords_").removesuffix(".jsonl")
        manifest["splits"][split_name] = {
            "file": file_name, "records": len(split_records),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_lines": [source_lines[normalize_text(row["conversations"][0]["content"])] for row in split_records],
        }
        registry[target.stem] = {
            "file_name": file_name, "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
        print(f"{file_name}: {len(split_records)} 条")

    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "dataset_info.json", registry)

    print(f"规范化答案: {normalized_count} 条")
    print(f"移除重复输入: {duplicate_count} 条")
    print(f"划分种子: {args.seed}")
    print(f"输出目录: {args.output_dir}")
    print("已保存清洗明细、文件 SHA-256、原始行号和数据集登记。仍需抽查答案与近重复样本。")


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
