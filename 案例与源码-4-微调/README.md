# 关键词微调：课程数据与配置

第 29～33 章使用这里的 `keywords-clean` 数据。先按章节完成数据准备、训练、评估和导出；需要对照回答或曲线时，再打开 `examples/` 和 `results/`。

## 从哪里开始

| 学习步骤 | 使用什么 |
| --- | --- |
| 第 29 章：清洗与划分数据 | `keywords_data_sharegpt_small.jsonl`、`prepare_keywords_dataset.py` |
| 第 29 章：文档问答练习 | `document-qa-demo/` 中的 `product-manual.md` 与 `product_qa_reviewed.jsonl` |
| 第 30 章：理解模型输入与标签 | `audit_keywords_preprocessing.py`、`examples/preprocessing/` |
| 第 31 章：训练 | `processed/keywords-clean/`、`configs/keywords_clean_train.yaml` |
| 第 32 章：预测与评分 | 两份预测 YAML、预测核对脚本、转换脚本与评分脚本 |
| 第 32 章：导出模型 | `configs/keywords_clean_export.yaml` |
| 第 33 章选做：比较 batch 与显存 | `compare_batch_resources.py`、`results/batch-comparison/` |

## 数据文件

| 文件 | 用途 |
| --- | --- |
| `keywords_data_sharegpt_small.jsonl` | 2,000 条关键词练习样本 |
| `keywords_data_sharegpt.jsonl` | 49,499 条完整数据，供扩展练习 |
| `processed/keywords-clean/keywords_train.jsonl` | 1,600 条训练数据 |
| `processed/keywords-clean/keywords_validation.jsonl` | 200 条验证数据，用于调整方案与选择检查点 |
| `processed/keywords-clean/keywords_test.jsonl` | 200 条测试数据，方案确定后再使用 |
| `processed/keywords-clean/dataset_info.json` | LLaMA-Factory 读取三份数据所需的登记表 |
| `processed/keywords-clean/cleaning_report.json`、`manifest.json` | 处理明细、来源与划分说明 |

在本目录复现处理过程，结果另存到尚不存在的目录：

```bash
python3 prepare_keywords_dataset.py \
  --source keywords_data_sharegpt_small.jsonl \
  --seed chapter-29-v1 \
  --output-dir processed/keywords-clean-repro
```

种子 `chapter-29-v1` 用于固定分组，跟做时保持该值即可。脚本检查单轮 ShareGPT 结构、统一关键词格式并去重；遇到答案冲突或剩余格式问题时停止。它不能替代内容审核，也不能识别所有近似输入。参考答案仍可能存在内容疑点，准备自己的数据时应继续对照原文审核选词。

完整数据与 small 文件都在本目录中，主线跟做使用 small 文件即可。完整数据包含全部 small 样本，不能把两者分别当成独立的训练集和测试集。

## 脚本与配置

| 文件 | 作用 |
| --- | --- |
| `prepare_keywords_dataset.py` | 检查、清洗并划分关键词样本；复用评分脚本的格式检查 |
| `audit_keywords_preprocessing.py` | 在 LLaMA-Factory 环境中核对 token 长度、标签和截断情况 |
| `audit_keywords_prediction.py` | 按 YAML 运行预测，核对实际 FP16、数据指纹与完整条数 |
| `convert_keywords_predictions.py` | 将预测文件转换为评分格式，保留模型原始回答 |
| `evaluate_keywords_predictions.py` | 对照参考答案检查格式并计算关键词指标 |
| `compare_batch_resources.py` | 对比单卡 batch 4／累积 8 与 batch 2／累积 16，记录耗时和显存采样 |
| `configs/keywords_clean_train.yaml` | 第 31 章 LoRA 训练配置 |
| `configs/keywords_clean_predict_base.yaml` | 第 32 章原始模型测试预测配置 |
| `configs/keywords_clean_predict_lora.yaml` | 第 32 章 Adapter 测试预测配置，须先完成训练 |
| `configs/keywords_clean_export.yaml` | 第 32 章合并导出配置 |

按[第 31 章](/31-LLaMA-Factory环境搭建与微调实战.md)把 `processed/keywords-clean/` 中的六个文件上传到 AutoDL 的 `LLaMA-Factory/data/keywords-clean/`。上传 YAML 前，在 AutoDL 终端进入 `LLaMA-Factory/`，执行 `mkdir -p configs`，再上传到该目录。按自己的模型和输出位置调整配置，使用新的输出目录。

## 配套示例与结果

| 想检查什么 | 查看哪里 |
| --- | --- |
| 提示词、思考模式、参考标注 | [示例阅读与复算入口](/案例与源码-4-微调/examples/README.md) |
| 训练日志、Loss 曲线与统计 | `results/keywords-clean/training/`；对应训练配置为 `configs/keywords_clean_train.yaml` |
| 三条验证输入的模型回答 | [验证与合并模型对照](/案例与源码-4-微调/results/keywords-clean/验证与合并模型对照.json ':ignore')：共同设置、三种模型状态和原始回答 |
| 各 200 条测试预测与分数 | `results/keywords-clean/base/`、`lora/` 中的 `generated_predictions.jsonl` 与 `report.json` |
| 实际预测配置与精度 | 上述 `base/`、`lora/` 中的 `actual-config.yaml` 与 `prediction-evidence.json` |
| 导出文件与独立加载结果 | [导出文件清单](/案例与源码-4-微调/results/keywords-clean/导出文件清单.json ':ignore')与上述验证对照；导出配置为 `configs/keywords_clean_export.yaml` |
| batch 与显存对照 | `results/batch-comparison/` 中的配置、日志、GPU 采样、曲线及 `comparison.json` |
| WebUI 页面与训练状态 | `results/keywords-clean-webui-20260908/` 中的页面配置、训练参数、状态和曲线 |

**注意示例的对应关系：** 第 30 章日志分析、第 32 章预测、评分与导出对应 CLI 示例 `results/keywords-clean/`；第 31 章 WebUI 截图对应另一份训练结果。两组使用相同数据与核心参数，但 Adapter 和输出目录分别保存。自己跟做时，使用自己的那一份产物。

这些资料包含格式、漏词和重复等问题，可用于练习分析，不代表模型已经达到业务要求。目录中不含模型权重及完整续训检查点，不能把示例路径填入 WebUI 作为 Adapter 路径。

使用 Python 3.10 或以上版本，按[第 32 章第 4.4 节](/32-微调效果评估与模型部署.md?id=_44-运行转换与评分脚本)可在本机复算评分。转换后的 JSONL 和新评分报告由脚本生成，另存到 `results/keywords-clean-rescore/` 等未使用的目录。
