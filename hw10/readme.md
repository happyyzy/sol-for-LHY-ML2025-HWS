# Goal
本次作业的目标是: Customization of text-to-image (T2I) diffusion model，或者简单来讲就是换头，给了六种对象，希望在用prompt生成图像时对应的主体来自相应的类别。本次提供kaggle notebook[代码](https://www.kaggle.com/code/kodaria/ml2025-homework-10-diffusion/notebook?scriptVersionId=267248560)，无需本地环境。

# Hints
[官方作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw10.pdf)给了两种途径，一种是[blip diffusion](https://huggingface.co/salesforce/blipdiffusion)，简而言之就是通常的diffusion只能用clip接受文本提示，而blip diffusion把clip换成一个叫blip的组件，能同时接受文本和参考图像；然而blip diffusion之能接受一张参考图像并且生成可控性较差，所以还有一条路叫custom diffusion(https://arxiv.org/abs/2212.04488)，即在 Stable Diffusion 的文本编码空间中加入一个新 token（比如 “sks turtle”），并通过微调让模型学会把这个 token 对应到你提供的乌龟玩具样子，这种途径在参考图像是多张的时候往往效果更好。

# My approch
我尝试了blip diffusion后发现效果较差，无论怎么按照[Hints](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw10.pdf)调参数也离boss baseline差得远，后来发现blip diffusion是个很冷门的东西，自从当年的几篇论文后再也没动静，并且自从diffusers0.33版本后已停止对该模型的更新支持，目前在这类任务上比较主流的是controlnet的相关变体，我尝试了其中的[ip-adpter](https://huggingface.co/h94/IP-Adapter),在多次调参后成功通过[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw10.pdf)中的boss baseline。

另外ip-adapter其实就是搞个厉害点的clip把参考图像喂给它，更加说明blip完全是多此一举效果还差。。

## 📊 Performance Comparison

| Object   | DINO Score (Ref) | CLIP-T (Ref) | DINO Score (my) | CLIP-T (my) |
|-----------|------------------|---------------|------------------|---------------|
| Object 1  | 68               | 18            | 78               | 26            |
| Object 2  | 60               | 17            | 79               | 24            |
| Object 3  | 61               | 18            | 63               | 28            |
| Object 4  | 68               | 19            | 74               | 28            |
| Object 5  | 60               | 19            | 68               | 21            |
| Object 6  | 57               | 17            | 62               | 29            |

# Code
本次提供的代码一次性跑通是得不到这样的结果的，每个对象都要逐个调参，kaggle P100上一张图片一分钟左右，15×6张图片，验证过程相对快很多。另外我用的sdxl支持jax/flax，可以用kaggle的tup v5e-8推理，生成一张图片时间能暴降到2s。

主要参数有三个，[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw10.pdf)里面提到两个：num_inference_steps和guidance_scale，但是这两个经笔者尝试没啥用，去噪步数25左右就好，调得再大也没啥作用，guidance_scale7左右就好，也不太影响生成结果，过大/过小反而使生成结果很糟糕；

最关键的参数是ip-adpter的控制强度ip_adapter_scale,该参数因任务而易，比如对object-4我最开始设为ip_adapter_scale=0.8导致模型对prompt关注太少根本不给小狗戴上墨镜，clip score很低，最后设为0.6才达到满意效果；而对object-5，ip_adapter_scale=0.6又太关注prompt，生成的玩具机器人和参考图像差别太大导致DINO不及格，调到0.8才通过baseline。


# Evaluation
[作业指导](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2025-course-data/hw10.pdf)中讲了验证指标是[facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large)计算的相似度，用来验证生成图像和参考图像的相似程度，同时为了兼顾生成图像对prompt的贴合程度，还要用计算[openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14)score，注意加载这个clip的时候需要较低版本的transforemsers:
```python
pip install -q transformers==4.42.4
```
这部分验证代码也包含在代码文件里了。

# Reference
本次作业还参考了李宏毅老师2024年ml课的[hw10](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php),该作业和本次作业任务类似，是给了演员Brad的100个图片-文本对，直接微调diffusion来对给定的25个测试propmt生成人像。下面我来讲讲这份作业的坑。

首先我尝试了该作业的[官方代码](https://colab.research.google.com/drive/1dI_-HVggxyIwDVoreymviwg6ZOvEHiLS?usp=sharing#scrollTo=CnJtiRaRuTFX)结果各种报错，其中一部分报错是因为改作业测试要用tensorflow框架的ghostnet，而目前稳定版tf都不支持50系显卡，后又各种拉镜像才解决tf的问题，但是之后还是各种报错，遂发现24年作业目前也有整理好的[中文版本](https://blog.csdn.net/weixin_42426841/article/details/142362711)，于是转战。

中文版本分[精简版](https://www.kaggle.com/code/aidemos/14b-lora-stable-diffusion)和[完整版](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Demos/14a.%20%E5%B0%9D%E8%AF%95%E4%BD%BF%E7%94%A8%20LoRA%20%E5%BE%AE%E8%B0%83%20Stable%20Diffusion%20%E6%A8%A1%E5%9E%8B.ipynb),我先尝试精简版，发现是能跑通，但是无论怎么调参ghostnet distance离1.20的boss baseline还是差得远，最好大概也就1.26的样子，进而发现精简版和完整版的差别就在于精简版图省事把完整版的交叉验证给删了，导致微调过程就是按diffusion原本的mse loss蒙头乱撞，ghostnet distance根本下不去，也是被精简版代码说明给蒙骗了。
> “也因为精简版修改的东西比较多，所以不建议同时代入两个版本。最终训练效果一致。”  
> — from [精简版](https://www.kaggle.com/code/aidemos/14b-lora-stable-diffusion)

遂又又转战[完整版](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Demos/14a.%20%E5%B0%9D%E8%AF%95%E4%BD%BF%E7%94%A8%20LoRA%20%E5%BE%AE%E8%B0%83%20Stable%20Diffusion%20%E6%A8%A1%E5%9E%8B.ipynb)。

首先发现这个完整版在colab/kaggle里面根本跑不了，各种环境冲突报错，又被坑了一下
> “对于可以访问 Colab 的同学来说，一样建议使用下面这份代码进行学习。”  
> — from [完整版](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN/blob/master/Demos/14a.%20%E5%B0%9D%E8%AF%95%E4%BD%BF%E7%94%A8%20LoRA%20%E5%BE%AE%E8%B0%83%20Stable%20Diffusion%20%E6%A8%A1%E5%9E%8B.ipynb)。

之后在本地环境跑发现此代码有显存泄露问题，每轮交叉验证完显存还被模型占着导致第二轮以后的训练巨慢，把此问题解决后训练速度正常了，但是第二轮以后的交叉验证速度又急转直下，最后也没完全解决就慢着跑了很多轮（显存20g以上应该不用担心这个问题），过程中按[作业指导](https://docs.google.com/presentation/d/1kIe0UnPT_TV3Dw2TMzL4Uui78UJjyL8ikcUPWxep3YU/edit?pli=1#slide=id.g2dc8860317c_0_0)各种调参始终无法达到ghostnet distance=1.20的boss baseline，最好也就1.208的样子，后来觉得是因为直接微调还是缺乏导向型，应该转用DreamBooth这种更精细的微调方法。

然而我还是先试了下ip-adpter，发现只要在训练集里随便挑二十张照片喂给ip-adapter，连prompt都不要，就能达到ghostnet distance=1.22，然后又换了一个专门用于人脸的[ip-adapter-plus-face_sdxl_vit-h.safetensors](https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors)
```python
ip_adapter_scale=0.8
pipeline.load_ip_adapter("h94/IP-Adapter",
                         subfolder="sdxl_models",
                         weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"] ,
                         image_encoder_folder="models/image_encoder")
```
ghostnet distance直接暴降到1.15，clip score和faceless faces都达到boss baseline，但是老实说这版图片看起来换头感特严重,如下
![boss baseline](image_13.jpg)
还不如没达到boss baseline生成的自然，可能这就是ip-adapter-plus-face_sdxl_vit-h和ghostnet共同的人机审美吧。。

另外这里ip_adapter_scale=0.8也能让模型很听prompt的话，不像之前生成猫狗得调到0.6左右，可能模型训练数据里面人像比较多吧。

最后提一下作业指导的方法都很老，2025年的今天还推荐blip2，以及custom diffusion[微调方法](https://arxiv.org/abs/2212.04488)其实就是当年追text inversion和dreambooth热度的水文，现在也没人用了。感兴趣的的可以试试正统的text inversion/dreambooth,这方面非常成熟，有diffusers的官方脚本，甚至不少ui界面都兼容，一行代码都不用写。

