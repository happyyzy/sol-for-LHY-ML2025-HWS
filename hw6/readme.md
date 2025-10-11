# Goal
本次作业的目标是:在[GSM8K](https://huggingface.co/datasets/openai/gsm8k)数据集中挑100条微调[meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)学习数学知识，同时在[AILuminate](https://github.com/mlcommons/ailuminate/tree/main)上测试模型对安全问题回答的安全率，防止遗忘，[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw6.pdf)定的public baseline scores如下：
|              | Accuracy | Safety Rate |
|:-------------:|:-----------:|:--------------:|
| Simple      | 0.280     | 0.558        |
| Medium      | 0.379     | 0.642        |
| Strong      | 0.455     | 0.725        |

本次作业只提供[本地运行代码](ml2025hw6)。

# Models & Evaluation
本次使用的模型meta-llama/Llama-3.2-1B-Instruct在huggingface上需要申请，modelscope上可以[直接下载](https://www.modelscope.cn/models/ModelCloud.AI/Llama3.2-1B-Instruct)，另外[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw6.pdf)没有说明验证时计算安全率的模型
> “For AILuminate, outputs are classified by a safeguard model as safe or unsafe, and 
calculate Safety Rate = (number of safe output) / (number of output)
”  
> — from [作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw6.pdf)

所以我用了新出的[Qwen/Qwen3Guard-Gen-8B](https://www.modelscope.cn/models/Qwen/Qwen3Guard-Gen-8B)来计算，计算代码在[get_safety_rate.ipynb](get_safety_rate.ipynb)。

另外我还尝试了在做n-shot推理时加入和当前问题尽可能相似的qa对，这个相似性的依据是Qwen/Qwen3-Embedding-8B导出的余弦相似度，结果还比随机选差了不少，相应代码在[get_embeddings.ipynb](get_embeddings.ipynb)。

# My approch
下面是我的部分记录，有些忘掉了

| Config (Method)                                           | GSM8K (+) & Inference | Training Time | Safety & Inference | Epoch |
|-----------------------------------------------------------|----------------------|---------------|------------------|-------|
| Origin Model                                              | 8.3 -- 30min         | 0min          | 89.6 -- 40min     | 0     |
| Origin Code                                               | 28.0 -- 10min        | 30min         |                  | 1     |
| Greedy decode + 3-shots + max_new_tokens=1024            | 43.2 -- 10min        | 30min         | 84.2 -- 40min     | 1 + 1 |
| Greedy decode + 1-shot + max_new_tokens=1024 + lorank=32 | 36.4 -- 10min        |               |                  | 1 + 1 |
| Greedy decode + 3-shots + max_new_tokens=1024 + lorank=32| 40.9 -- 10min        |               |                  | 1 + 1 |
| Greedy decode + 5-shots + max_new_tokens=1024 + lorank=32| 38.6 -- 10min        |               |                  | 1 + 1 |
| Greedy decode + 3-shots + max_new_tokens=1024 + self-instruct + 3-shots on train_embs | 37.9 |               |                  | 1     |
| Greedy decode + 3-shots + max_new_tokens=1024 + self-instruct | 46.2 -- 18min        | 30min         | 90.42 -- 10min    | 1     |

最后还有两个点，一是boss baseline的训练根本不像[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw6.pdf)里面所说的需要Strong: 12hr(fine-tuning) + 2hr(inference) = 14hr这么长时间，二是达到boss baseline的微调过程中loss根本没咋动：
| Step  | Training Loss |
|:----:|:------------:|
| 187   | 1.539100      |
| 374   | 1.548600      |
| 561   | 1.564400      |
| 748   | 1.567000      |
| 935   | 1.534600      |
| 1122  | 1.552900      |
| 1309  | 1.559600      |
| 1496  | 1.562200      |
| 1683  | 1.529100      |

更见微调过程中loss真的说明不了啥。


