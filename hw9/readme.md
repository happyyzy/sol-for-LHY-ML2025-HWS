## 目标

不做额外微调、不动原训练数据，只用 LoRA task vectors 的参数算术，把两个不同任务（ARC science + GSM8K math）的能力 merge 成一个统一模型，并使这个 merged model 用同一次 merge 设定同时在 400 题（两任务混合 MCQA）上推理。

---

hw9

TAs 已经提供两份 LoRA（各自已针对 task PEFT 好）

- science：ARC （≈ 63%）
- math：GSM8K （≈ 52.5%）

base 是 `llama-2-7b-chat-bnb-4bit`（base 大概 44% / 37%）。

---

hw9

学生任务：

选择 merging algorithm / hyperparam / density / weights → 得到一个 merged adapter → infer 400 questions → 输出 json。

---

## 有哪几类 merging 算法？

作业明确列出以下 5 类（并都给了引用）：

| 名称 | 文献 |
| --- | --- |
| Task Arithmetic / Linear | Ilharco et al. 2022 |
| Magnitude Prune | （本身概念来自训练剪枝 literature） |
| DARE Linear | Yu et al. “Super Mario”, ICML 2024 |
| TIES | Yadav et al. NeurIPS 2023 |
| SCE | Wan et al. FuseChat 2024 |

所有 5 都列在 slides 内，并且 SCE 还是 HW TODO（要求在 peft 里自己实现）。

---

hw9

## baselines

作业给了不同 baseline 水平（ARC acc / GSM8K acc）分级：

| baseline level | ARC | GSM8K |
| --- | --- | --- |
| public simple baseline | ≥49% | ≥38% |
| public medium baseline | ≥53% | ≥42% |
| public strong baseline | ≥56% | ≥48% |

还有 private baseline（评分加分维度）。

---

## 已给的微调版本测试表现（LoRA PEFT 完成后）

作业明确写了：

| model | ARC acc | GSM8K acc |
| --- | --- | --- |
| base llama-2-7b-chat-bnb-4bit | 44% | 37% |
| after PEFT（science） | 63% | — |
| after PEFT（math） | — | 52.5% |

这说明两个 LoRA ckpt 各自对单一任务微调后已经得到比 base 更好的性能。

---

注意这里的merging是针对同一个底模的不同任务lora版本，所以只需要合并lora权重；

这与nlp任务中常见的merging/averaging不太一样：这里是对同一任务（机器翻译，文本嵌入...）的不同ckpt做平均，能显著提高指标。本目录下总结了qwen3 embedding模型的球面线性加权策略来说明这一点；2023hw的机器翻译任务也可用此trick。但是hw5这种gsm8k微调就未必合适；之后有时间或许会自己做一下。
