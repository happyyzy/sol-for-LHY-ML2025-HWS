# Goal
本次作业的目标是：从Alpaca 52k数据集里面挑100条微调LLaMA2-7B，在Evol_Instruct数据集上测试，[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw5.pdf)提供的baselines如下：
### 📊 Baselines GPT-4o-mini Scores

| Model Type | Score |
|:----------:|:-----:|
| Simple Public | 2.4  |
| Medium Public | 4.15 |
| Strong Public | 4.4  |

本次作业只提供[本地运行代码](Homework5_Finetuning_is_Powerful.ipynb)。

# Models & Evaluation
基座模型是llama2-7b-bnb-4bit，unsloth改造的加速版本，微调时候用unsloth的框架，各种野鸡感。。比如导入好了tokenizer后还得用unsloth的方法再包一层:
```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama",  ### Use llama-3.1 template for better performance here
)
```
这里官方代码注释要用llama-3.1 template，但会引起后面报错，我用"llama"也没觉得影响表现。诸如此类的恶心地方挺多的，所以还是有米好啊hh。
本次的数据集Evol_Instruct都是情景问答类没有精确评分指标，所以要用其他模型评分，官方使用gpt4o-mini，并公开了[评分代码](https://drive.google.com/file/d/12WFH1mCBG2zTM29olVy_itq7-pPWmRtX/view?usp=sharing),我没有openai的api，看到ollama api上有gpt-oss-120b-cloud，于是用它来评分（希望gpt好兄弟们口味一致吧），
评分程序在[eval.ipynb](eval.ipynb)。

另外此作业的测试集没给答案，本来是要学生提交到网站上评分，我用ollama的deepseek-v3.1:671b-cloud的api做了一份答案在[evol_instruct_gt.json](evol_instruct_gt.json)里面，这样才能得到最终评分。

# My approch
本次作业挺简单的，首先还是max_new_tokens=1024设大一点，然后用advaned_dataset直接训练15轮就好了，最后评分4.75，作业指导里面的trick基本没用，特别是Curriculum Learning，我费了老大劲分了三层难度一层层训练，最后评分4.59还没直接训练好。
### 🧩 Training Progress Summary

<div>
  <progress value="195" max="195" style="width:300px; height:20px; vertical-align: middle;"></progress>
  [195/195 &nbsp;&nbsp;19:56, Epoch 15/15]
</div>

| **Step** | **Training Loss** |
|:--------:|:-----------------:|
| 1   | 1.1057 |
| 20  | 1.0866 |
| 40  | 0.9298 |
| 60  | 0.6943 |
| 80  | 0.3451 |
| 100 | 0.1399 |
| 120 | 0.0363 |
| 140 | 0.0412 |
| 160 | 0.0167 |
| 180 | 0.0096 |
| 195 | 0.0035 |
