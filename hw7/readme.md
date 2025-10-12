# Goal
用unsloth对unsloth/llama-3-8b-Instruct做DPO，50条训练集，10条测试集，作业任务是回答问题，整体如下。



## 📊 实验观察题 (3%)

### **问题 1** (0.5%)
**控制变量**: `num_epoch=3`, `data_size=50`

| 实验编号 | 参数设置 | 观察目标 |
|----------|----------|----------|
| **1a** | `support_ratio = 0` | 观察完全反对立场的效果 |
| **1b** | `support_ratio = 1` | 观察完全支持立场的效果 |

### **问题 2** (0.5%)
**控制变量**: `data_size=50`, `support_ratio=0`

| 实验编号 | 参数设置 | 观察目标 |
|----------|----------|----------|
| **2a** | `num_epoch = 1` | 观察单轮训练的效果 |
| **2b** | `num_epoch = 3` | 观察三轮训练的效果 |

### **问题 3** (0.5%)
**控制变量**: `num_epoch=3`, `support_ratio=0`

| 实验编号 | 参数设置 | 观察目标 |
|----------|----------|----------|
| **3a** | `data_size = 10` | 观察小数据集的效果 |
| **3b** | `data_size = 50` | 观察完整数据集的效果 |

---
这部分自己观察效果就行，没啥可说的。

## 🔍 模型测试题 (2%)

### **问题 4** - 使用训练好的模型  
**模型配置**: `data_size=50`, `support_ratio=0`, `num_epoch=3`

| 子问题 | 测试场景 | 考察重点 |
|--------|----------|----------|
| **4a** (0.5%) | System Prompt改为生成长回复 | 长度控制 vs 立场一致性 |
| **4b** (0.5%) | 询问其他图像风格<br>(One Piece、迪士尼风格) | 风格泛化能力 |
| **4c** (0.5%) | 询问其他艺术形式<br>(巴赫风格音乐) | 领域泛化能力 |
| **4d** (0.5%) | 使用中文Prompt提问 | 语言鲁棒性 |

---
这部分的测试结果在这里，模型对这四类问题的泛化效果都很好，50条训练数据就能取得这么好的结果，出乎我的意料。

## 📚 论文阅读题 (5.5%)

### 问题5 - InstructGPT 论文 (1.5%)

| 题号 | 问题内容 | 考察重点 |
|------|----------|----------|
| **5.1** | 方法介绍中哪个步骤是不正确的？ | 方法步骤的正确性识别 |
| **5.2** | 哪些步骤可以持续迭代？ | 迭代流程的理解 |
| **5.3** | 奖励建模中如何解决过拟合问题？ | 过拟合处理技术 |
| **5.4** | 给定奖励分数，计算核心损失项<br>奖励：$r_θ(x, y_w)=3.0$, $r_θ(x, y_l)=1.3$<br>公式：$loss(θ) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l) \sim D} [\log(\sigma(r_θ(x, y_w) - r_θ(x, y_l)))]$ | 损失函数计算能力 |
| **5.5** | 当时ChatGPT中过优化的症状有哪些？ | 过优化现象识别 |

### 2. DPO 框架中主要使用的损失函数

- **主要损失**：

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = - \mathbb{E}_{(x, y^+, y^-)} 
\Big[ \log \sigma \Big( \beta \big( \log \frac{\pi_\theta(y^+|x)}{\pi_{\mathrm{ref}}(y^+|x)}} 
- \log \frac{\pi_\theta(y^-|x)}{\pi_{\mathrm{ref}}(y^-|x)} \big) \Big) \Big]
$$


### 问题6 - DPO 论文 (2%)

| 题号 | 问题内容 | 考察重点 |
|------|----------|----------|
| **6.1** | 与之前RLHF方法的不同之处？ | 方法创新点 |
| **6.2** | DPO框架中使用的主要损失函数类型？ | 损失函数理解 |
| **6.3** | 参考策略($π_{ref}$)在DPO训练中的作用？ | 参考策略功能 |
| **6.4** | 使用GPT-4作为评估器的主要发现？ | 评估方法效果 |
| **6.5** | 哪个GPT-4提示能提供更代表人类的胜率？ | 提示工程效果 |

---

### 1. What makes this work different from prior RLHF methods?

- **核心区别**：
  - 传统 RLHF（如 PPO-based）需要三步：训练 reward model → 用 RL 优化 LM → 迭代。
  - **DPO 直接优化语言模型的概率分布**，无需 RL 训练：
\[
\mathcal{L}_{\text{DPO}} = - \log \sigma \Big( \beta [ (\log p_\theta(y^+|x) - \log p_{\text{ref}}(y^+|x)) - (\log p_\theta(y^-|x) - \log p_{\text{ref}}(y^-|x)) ] \Big)
\]
  - 通过对比 chosen / rejected 样本的相对概率直接优化模型。
- **优势**：
  - 更稳定、训练更快。
  - 避免 RL 中容易出现的奖励崩塌或训练不稳定。

---

### 2. What type of loss function is primarily used to train the language model in DPO?

- **主要损失函数**：
- $$\mathcal{L}_{\text{DPO}} = - \log \sigma \Big( \beta [ (\log p_\theta(y^+|x) - \log p_{\text{ref}}(y^+|x)) - (\log p_\theta(y^-|x) - \log p_{\text{ref}}(y^-|x)) ] \Big)$$

- **解释**：
  - \(y^+\) = chosen, \(y^-\) = rejected
  - \(\beta\) 控制 reward margin 强度
  - 通过 sigmoid + log，使模型学习 **相对偏好** 而不是绝对概率。

---

### 3. What is the role of the reference policy \(p_{\text{ref}}\) in DPO training?

- **作用**：
  1. 提供稳定的基线，防止模型生成非自然文本。
  2. DPO 优化的是**相对概率差**：
\[
(\log p_\theta(y^+|x) - \log p_{\text{ref}}(y^+|x)) - (\log p_\theta(y^-|x) - \log p_{\text{ref}}(y^-|x))
\]
  3. 保证语言模型既学偏好，又不丢失原本语言能力。

---

### 4. How can we tell during training whether the model is leaning toward chosen or rejected outputs?

- **关键指标**：
  - **reward margin**:  
\[
\text{rewards/margin} = \text{reward(chosen)} - \text{reward(rejected)}
\]  
> 正值 → 模型偏向 chosen  
> 负值 → 模型偏向 rejected
  - **accuracy**: fraction of samples where reward(chosen) > reward(rejected)  
> 越接近 1 → 模型更偏向 chosen  
> 越接近 0 → 模型偏向 rejected
  - **logps/chosen vs logps/rejected**:  
> chosen log-prob > rejected log-prob → 模型生成倾向 chosen

---

### 5. How does DPO improve model alignment with human preference?

- **机制**：
  1. 收集 human preference 数据：对每个 prompt 给出 chosen / rejected。
  2. 训练 DPO 直接优化 LM 使其**相对概率匹配人类偏好**。
  3. 引入 reference policy 保持语言质量。
- **结果**：
  - 模型生成内容更符合人类偏好。
  - 相比 PPO RLHF，训练更快、更稳定。
  - reward margin 和 logps 曲线可以实时监控模型偏好变化。

---



## 问题7 - DeepSeek 论文 (2%)

| 题号 | 问题内容 | 考察重点 |
|------|----------|----------|
| **7.1** | 关于PPO和GRPO的正确陈述？ | 方法对比理解 |
| **7.2** | PPO和GRPO的结构和方法哪些正确？ | 结构方法分析 |
| **7.3** | GRPO训练涉及多少模型，其中多少被主动训练？ | 训练架构理解 |
| **7.4** | GRPO算法如何计算优势来更新策略模型？ | 优势计算机制 |
| **7.5** | 在RL之前收集"cold-start"数据的主要好处？ | 数据策略价值 |

---

### **相关论文**
1. [**InstructGPT**](https://arxiv.org/abs/2203.02155) - Ouyang et al., 2022
2. [**DPO**](https://arxiv.org/abs/2305.18290) - Rafailov et al., 2023  
3. [**DeepSeekMath](https://arxiv.org/abs/2402.03300) [**DeepSeek-R1**](https://arxiv.org/abs/2402.03300) - Shao et al., 2024 & DeepSeek-AI, 2025
