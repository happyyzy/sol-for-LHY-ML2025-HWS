# Goal
本次作业的目标是：训练一个encoder-only的transformers来做一个自回归的宝可梦生成器,baselines如下
### 🧮 Baseline Evaluation Summary

| **Category** | **FID** | **PDR Estimate** | **Training Time** | **Score** |
|:------------:|:-------:|:----------------:|:-----------------:|:---------:|
| Public Simple Baseline  | ≤ 84.50 | ≥ 0.1 | 10 mins | +1pt |
| Private Simple Baseline | ≤ 84.50 | ≥ 0.1 | — | +1pt |
| Public Medium Baseline  | ≤ 81.00 | ≥ 0.5 | 20 mins | +1pt |
| Private Medium Baseline | ≤ 81.00 | ≥ 0.5 | — | +1pt |
| Public Strong Baseline  | ≤ 73.00 | ≥ 0.85 | 30 mins | +1pt |
| Private Strong Baseline | ≤ 73.00 | ≥ 0.85 | — | +1pt |

本次作业只提供本地代码。

# Evalution
fid和pdr，fid有现成的包，直接自己下载算分数就好了，pdr是宝可梦分类正确率，作业指导没指明是啥模型，我在网上也没找到合适的，感兴趣的可以自己下载个VL模型来分类。
# My approch
这次作业超简单，就把模型搞大一点就好了，把原配置的两层换六层，十分钟就训练完。fid立马暴跌至四十多，肉眼观察图片觉得也都是宝可梦，料想pdr也不会低。
```python
gpt2_config = {
    "activation_function": "gelu_new",    # Activation function used in the model
    "architectures": ["GPT2LMHeadModel"],  # Specifies the model type
    "attn_pdrop": 0.1,            # Dropout rate for attention layers
    "embd_pdrop": 0.1,            # Dropout rate for embeddings
    "initializer_range": 0.02,        # Standard deviation for weight initialization
    "layer_norm_epsilon": 1e-05,       # Small constant to improve numerical stability in layer norm
    "model_type": "gpt2",           # Type of model
    "n_ctx": 128,               # Context size (maximum sequence length)
    "n_embd": 64,              # Embedding size
    "n_head": 4,               # Number of attention heads
    "n_layer": 6,              # Number of transformer layers
    "n_positions": 400,           # Maximum number of token positions
    "resid_pdrop": 0.1,           # Dropout rate for residual connections
    "vocab_size": num_classes,       # Number of unique tokens in vocabulary
    "pad_token_id": None,          # Padding token ID (None means no padding token)
    "eos_token_id": None,          # End-of-sequence token ID (None means not explicitly defined)
}
```
