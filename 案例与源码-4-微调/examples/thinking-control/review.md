# 思考模式诊断：12 条预测逐条核对

下面对照验证集第 1、6、8 条的 12 份回答，使用相同的 B 提示词。以下回答直接读取原始预测文件，未删除分析、英文片段或重复词。

| 模型与模板 | 含分析段 | 形成完整列表 | 格式合格 |
| --- | --- | --- | --- |
| 原始模型 / qwen3_nothink | 3/3 | 0/3 | 0/3 |
| 原始模型 / qwen3 | 0/3 | 2/3 | 2/3 |
| Adapter / qwen3_nothink | 0/3 | 2/3 | 2/3 |
| Adapter / qwen3 | 0/3 | 2/3 | 2/3 |

“形成完整列表”只描述回答是否结束为列表，不代表主题信息齐全；“格式合格”使用课程原有评分脚本。所有回答内容仍需逐条阅读。

**机制已确认：** 在锁定的 LLaMA-Factory 版本中，`qwen3_nothink` 不加入空思考区块；`qwen3` 配合 `enable_thinking=False` 会在助手起始位置补入该区块。实际输入的差异只有末尾 4 个 token，用户消息相同，参考答案没有进入生成输入。

**回答现象已观察：** 原始模型的分析段从 3/3 变为 0/3，格式合格数从 0/3 变为 2/3；仍有英文改写、漏词和重复。Adapter 两组都是 2/3 格式合格，F68050 和第 8 条重复输出未消失。两组 qwen3_nothink 的 6 份回答与此前同配置样本逐字一致。

**结论范围：** 这支持把原始模型在本次条件下仍输出分析的原因定位到实际模板处理差异。已经排除本次参考答案泄漏、模型/Adapter 指纹改变和精度不一致；不能据此证明新模板适合所有输入、Adapter 优于充分优化的原始模型，或模型可上线。Adapter 的训练模板为 qwen3_nothink；这里比较的是推理模板，不能把结果理解为两种训练模板的效果比较。

文件用途和本地复算方法见[示例说明](/案例与源码-4-微调/examples/README.md)。10 条标注的草案见[参考标注审核](/案例与源码-4-微调/examples/thinking-control/annotation-review.md)，未用于重新评分。

## 验证集第 1 条

**冻结参考答案：** `独头掘进巷道;降温试验系统;制冷降温;焓差;制冷系数`

<details>
<summary>查看固定 B 提示词和完整输入</summary>

```text
请从下面的输入中抽取能够概括文章主题的关键词。
只输出一行关键词，关键词之间使用英文分号 ; 分隔。
不要输出“关键词：”、序号、解释或分析过程，不要重复关键词。
关键词应有原文依据，保留影响含义的完整名称，不为凑数量拆开短语。
原文同时给出全称和简称时优先使用全称，只有简称时保留简称。

待处理输入：
关键词抽取：
金属矿山独头掘进巷道制冷系数是计算独头掘进巷道降温需冷量非常重要的参数之一,关系到能否达到预期降温效果及影响着降温工程的的经济投入。利用“金属矿山深部开采降温试验系统”分别模拟通风降温、加大风量降温、制冷降温、加大制冷降温4种条件下独头掘进巷道的降温过程,并分析独头掘进巷道降温效果,在通风及加大风量降温后,独头掘进巷道内温度先下降然后趋于稳定,再也无法下降,只有通过制冷措施,对独头掘进工作面进行降温。通过监测数据的焓差值计算出降温制冷系数,同时利用多次试验得出制冷系数范围在1．38~1．68,进而为独头掘进巷道降温设计提供设计依据。
```

</details>

### 原始模型 / qwen3_nothink

含分析段：是；形成完整列表：否；格式合格：否。

含 <think> 分析段和多行；未形成最终列表。 分析提及制冷系数和试验系统等原文内容，但回答末尾仍在讨论选词，尚未交付可使用的关键词。

原始文件：[原始模型 / qwen3_nothink 第 1 行](/案例与源码-4-微调/examples/thinking-control/原始模型-qwen3_nothink-关闭思考.jsonl ":ignore")。

```text
<think>
好的，我需要从用户提供的输入中提取能够概括文章主题的关键词。首先，我需要仔细阅读并理解原文内容。

原文主要讲的是金属矿山独头掘进巷道的降温系统，特别是制冷系数的重要性。提到了几个关键点：制冷系数、降温试验系统、四种降温条件（通风、加大风量、制冷、加大制冷）以及制冷系数的范围。用户要求只输出一行关键词，用英文分号分隔，不能有序号、解释或分析过程，还要保留原文的完整名称，优先使用全称。

首先，我需要确定哪些是核心概念。原文中的关键词包括“金属矿山独头掘进巷道制冷系数”、“降温试验系统”、“通风降温”、“加大风量降温”、“制冷降温”、“制冷系数范围”。这些都直接对应原文的主题。

接下来，检查是否有其他可能的关键词，但用户要求只输出一行，所以需要确保没有重复或冗余的。同时，要确保使用全称，如果存在全称和简称的话，优先使用全称。例如，“金属矿山深部开采降温试验系统”是全称，而“降温试验系统”是简称，但用户可能希望保留全称
```

### 原始模型 / qwen3

含分析段：否；形成完整列表：是；格式合格：是。

通过现有形式检查；形成一行列表。 返回 metal-mining-drainage-coefficient;cooling-coefficient，未保留原文中文完整名称；drainage 没有对应原文术语，且遗漏独头掘进巷道、试验系统等主要信息。

原始文件：[原始模型 / qwen3 第 1 行](/案例与源码-4-微调/examples/thinking-control/原始模型-qwen3-关闭思考.jsonl ":ignore")。

```text
metal-mining-drainage-coefficient;cooling-coefficient
```

### Adapter / qwen3_nothink

含分析段：否；形成完整列表：是；格式合格：是。

通过现有形式检查；形成一行列表。 金属矿山、独头掘进巷道、制冷系数都有原文依据；漏掉降温试验系统、制冷降温、焓差等参考词。金属矿山未被参考答案收录，不因此判为无依据词。

原始文件：[Adapter / qwen3_nothink 第 1 行](/案例与源码-4-微调/examples/thinking-control/Adapter-qwen3_nothink-关闭思考.jsonl ":ignore")。

```text
金属矿山;独头掘进巷道;制冷系数
```

### Adapter / qwen3

含分析段：否；形成完整列表：是；格式合格：是。

通过现有形式检查；形成一行列表。 改为英文词组 metal mining;cooling coefficient;metal mining deep mining;cooling system，未保留原文完整名称，漏掉独头掘进巷道和试验系统的具体名称。不能仅凭没有分析段认定改善。

原始文件：[Adapter / qwen3 第 1 行](/案例与源码-4-微调/examples/thinking-control/Adapter-qwen3-关闭思考.jsonl ":ignore")。

```text
metal mining;cooling coefficient;metal mining deep mining;cooling system
```

## 验证集第 6 条

**冻结参考答案：** `夏波蒂;马铃薯;榆林市;种植表现;炸条;杂交育成;推广栽培;试种;食用品质;技术简介;试验站;农科所;加拿大;产量高;有性;审定;沙滩;品种;农业;母本`

<details>
<summary>查看固定 B 提示词和完整输入</summary>

```text
请从下面的输入中抽取能够概括文章主题的关键词。
只输出一行关键词，关键词之间使用英文分号 ; 分隔。
不要输出“关键词：”、序号、解释或分析过程，不要重复关键词。
关键词应有原文依据，保留影响含义的完整名称，不为凑数量拆开短语。
原文同时给出全称和简称时优先使用全称，只有简称时保留简称。

待处理输入：
识别出下文中的关键词：
夏波蒂是加拿大福瑞克通农业试验站用"F58050"为 母本,"Bakeking"为父本经有性杂交育成,1987年引入我国试种,未经过审定或认定,但被辛普劳公司作为炸条品 种在各地种植.2003年由榆林市农科所弓1人我市试种,通过5年观察,该品种产量高,炸条品质和食用品质优良,适 宜在我市北部风沙滩水地区推广栽培,现将种植表现及栽·培技术简介如下:
```

</details>

### 原始模型 / qwen3_nothink

含分析段：是；形成完整列表：否；格式合格：否。

含 <think> 英文分析段和多行；未形成最终列表。 分析擅自写成 hybrid corn variety、Xinping Company，原文没有这些表达；不能从分析文字中替模型整理答案。

原始文件：[原始模型 / qwen3_nothink 第 2 行](/案例与源码-4-微调/examples/thinking-control/原始模型-qwen3_nothink-关闭思考.jsonl ":ignore")。

```text
<think>
Okay, let's tackle this. The user wants me to extract keywords from the provided text that summarize the article's main topic. The output should be one line with the keywords separated by semicolons, no markdown, and no extra explanations or numbers.

First, I need to read through the input carefully. The text talks about a hybrid corn variety developed through crossbreeding between two parents: F58050 as the mother and Bakeking as the father. The variety was introduced in 1987 by the Canadian agricultural experiment station, but it wasn't approved or tested. It was later used by Xinping Company as a炸条 (which I think translates to "fry strip" or "fry product") crop in various regions. Then there's a mention of trials in 2003 by the local agricultural research institute in榆林, leading to high yield and good quality, suitable for cultivation in the northern wind-swept areas of the city.

Now, the main elements here are the hybrid variety, the parents (F58050 and Bakeking), the introduction date (1987), the approval status (not approved or tested), the use by Xinping Company, and the trials in 2
```

### 原始模型 / qwen3

含分析段：否；形成完整列表：是；格式合格：是。

通过现有形式检查；形成一行列表。 F58050 与 Bakeking 均有原文依据，但只返回亲本编号/名称，漏掉主题品种夏波蒂和种植表现等信息。

原始文件：[原始模型 / qwen3 第 2 行](/案例与源码-4-微调/examples/thinking-control/原始模型-qwen3-关闭思考.jsonl ":ignore")。

```text
F58050;Bakeking
```

### Adapter / qwen3_nothink

含分析段：否；形成完整列表：是；格式合格：是。

通过现有形式检查；形成一行列表。 F58050、Bakeking 有原文依据，F68050 没有原文依据；仍漏掉夏波蒂等主题信息。

原始文件：[Adapter / qwen3_nothink 第 2 行](/案例与源码-4-微调/examples/thinking-control/Adapter-qwen3_nothink-关闭思考.jsonl ":ignore")。

```text
F58050;Bakeking;F68050
```

### Adapter / qwen3

含分析段：否；形成完整列表：是；格式合格：是。

通过现有形式检查；形成一行列表。 与 qwen3_nothink 的回答逐字相同：仍包含无依据词 F68050，且漏掉夏波蒂等主题信息。

原始文件：[Adapter / qwen3 第 2 行](/案例与源码-4-微调/examples/thinking-control/Adapter-qwen3-关闭思考.jsonl ":ignore")。

```text
F58050;Bakeking;F68050
```

## 验证集第 8 条

**冻结参考答案：** `恩诺沙星;日本鳗鲡;残留;高效液相色谱法`

<details>
<summary>查看固定 B 提示词和完整输入</summary>

```text
请从下面的输入中抽取能够概括文章主题的关键词。
只输出一行关键词，关键词之间使用英文分号 ; 分隔。
不要输出“关键词：”、序号、解释或分析过程，不要重复关键词。
关键词应有原文依据，保留影响含义的完整名称，不为凑数量拆开短语。
原文同时给出全称和简称时优先使用全称，只有简称时保留简称。

待处理输入：
抽取出文本中的关键词：
采用高效液相色谱法检测日本鳗鲡肌肉、血清、肠、鳃和肝脏组织中恩诺沙星及其代谢物环丙沙星的残留.方法的日内和日间变异系数分别为1.86%和2.53%,标准添加回收率为(95±6)%;最低测量限为1.0 μg/kg.用现场试验方法研究恩诺沙星在鳗鱼体内的代谢残留规律.对约50 g鳗鱼按9 mg/kg鱼体重每天给药2次,连续投喂7 d.给药期间鳗鱼体内的药物含量呈锯齿状上升,停药60 d后鳗鱼肠、鳃和肝脏组织中药物即下降至1～5 μg/kg.肌肉和血清中药物残留到90 d,分别消除至3 μg/kg和4 μg/kg.所测组织的药物残留至停药120 d后降到检测限以下.故鳗鱼的停药期不应低于120 d.
```

</details>

### 原始模型 / qwen3_nothink

含分析段：是；形成完整列表：否；格式合格：否。

含 <think> 分析段和多行；未形成最终列表。 回答在讨论组织、药物和检测数据如何选取，末尾仍在分析。

原始文件：[原始模型 / qwen3_nothink 第 3 行](/案例与源码-4-微调/examples/thinking-control/原始模型-qwen3_nothink-关闭思考.jsonl ":ignore")。

```text
<think>
好的，我需要从用户提供的文本中提取关键词，然后用英文分号分隔。首先，我得仔细阅读原文，找出所有重要的信息点。

原文提到检测的是日本鳗鲡的肌肉、血清、肠、鳃和肝脏组织中的恩诺沙星和环丙沙星的残留。方法的日内和日间变异系数、标准添加回收率、最低测量限等数据。然后是用现场试验研究代谢残留规律，给药量和时间，以及停药后的残留情况，最后得出停药期不应低于120天。

接下来，我需要确定哪些是核心关键词。首先，检测的组织是肌肉、血清、肠、鳃、肝脏，所以这些应该作为关键词。然后是药物名称恩诺沙星和环丙沙星，以及检测方法的名称高效液相色谱法。还有数据部分，比如变异系数、回收率、最低限等，但用户要求只输出关键词，所以可能不需要这些数据，除非它们有影响含义。不过用户说要保留影响含义的完整名称，所以可能需要保留这些数据，但可能不需要单独列出，因为用户只要求关键词，而数据可能属于方法细节。

然后，检查
```

### 原始模型 / qwen3

含分析段：否；形成完整列表：否；格式合格：否。

重复 enoxacin，末尾有分号空项；未形成完整可用列表。 输出 enolamide、enoxacin 等原文未给出的英文词，未保留原文中的恩诺沙星、环丙沙星、日本鳗鲡等名称。此处不把英文词自行映射成正确药物名。

原始文件：[原始模型 / qwen3 第 3 行](/案例与源码-4-微调/examples/thinking-control/原始模型-qwen3-关闭思考.jsonl ":ignore")。

```text
enolamide; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin;
```

### Adapter / qwen3_nothink

含分析段：否；形成完整列表：否；格式合格：否。

重复 enoxacin，末尾有分号空项；未形成完整可用列表。 没有保留原文中文药物、研究对象和方法名称；不要把重复英文词归一化后当作正确回答。

原始文件：[Adapter / qwen3_nothink 第 3 行](/案例与源码-4-微调/examples/thinking-control/Adapter-qwen3_nothink-关闭思考.jsonl ":ignore")。

```text
enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin;
```

### Adapter / qwen3

含分析段：否；形成完整列表：否；格式合格：否。

重复 enoxacin，末尾有分号空项；未形成完整可用列表。 与 qwen3_nothink 的回答逐字相同，空思考区块未解决本条的重复输出。

原始文件：[Adapter / qwen3 第 3 行](/案例与源码-4-微调/examples/thinking-control/Adapter-qwen3-关闭思考.jsonl ":ignore")。

```text
enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin; enoxacin;
```
