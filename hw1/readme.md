# Goal
本次作业要完成一个会检索信息的agent，让它能根据搜索到的信息回答问题。
| Baseline         | Public (30 questions) | Private (60 questions) |
|------------------|-----------------------|------------------------|
| Simple baseline  | 4                     | 6                      |
| Medium baseline  | 10                    | 15                     |
| Strong baseline  | 18                    | 22                     |

本次作业只提供[本地代码](qwen3_rag.py)。

# Models & Evaluation
[官方代码](https://www.kaggle.com/code/u0ulin/ml2025-homework-1)用的模型是Meta-Llama-3.1-8B-Instruct-Q8_0.gguf,但是这个模型能力挺拉的，[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data//hw1.pdf)里面也说了可以随便选模型改pipline，并且
支持用更强大的模型。结果我先在这个模型上测试公开数据集，30道只答对7道。之后我又测试qwen3-4b直接就能答对12道，模型大小还减半，从此被qwen3圈粉（

私有数据集没答案，有不少稀奇古怪的问题，我本来想找个大模型来生成私有数据集答案来测试的，但是因为很多问题太偏大模型也不确定就放弃了。作业指导里说私有数据集正确率是调用gpt-4o结合正确答案判评的，但是没给脚本也就不知道具体prompt难以复现。遂只看公开数据集正确率。

# My approch
长话短说，我按照作业指导无论怎么改pipeline/prompt也就答对十一二道，用qwen3-4b也就十四五道，而且参考代码的web search工具依赖于google search，还经常超频限用，推理两三个问题就中断了，超级烦人。于是彻底转向qwen框架，
用qwen3_agent的Assistant直接答对24道，这个框架完全开源，大不了把所有代码都抄下来也算自己写的了。。

