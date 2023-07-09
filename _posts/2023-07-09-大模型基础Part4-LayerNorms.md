---
layout: post
title:  "大模型基础Part4 - LayerNorms"
tags: 算法
excerpt_separator: <!--more-->
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_SVG" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            jax: ["input/TeX","output/SVG"],
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            
            }
        });
    </script>
</head>
<!--more-->
# Layer Norm 

Layer Norm在 [[2]](#2) 中被提出，作为RNN中的normalization方法，对标CNN中的BatchNorm。
Layer Norm的基本形式是：

$$
\mathbf{y} = \frac{\mathbf{x} - \mu}{ \sigma} \cdot \gamma + \beta
$$

其中$\mathbf{x}$作为Layer Norm的输入，通常是某个样本在某个layer上的hidden states。$\mu$和$\sigma$ 分别是其向量和方差，显然每个样本会取不同值。$\gamma$和$\beta$可以被训练。
原文中给出的Layer Norm带来提升的原因是：RNN类的模型在前向过程中容易梯度消失或爆炸，通过Layer Norm在每一层把输入拉回到合理尺度，从而使模型训练更稳定。

文献[[1]](#1) 中更近一步从反向传播角度分析了Layer Norm起作用的原因： 
- $\mu$ 的加入使得 $\frac{\partial l}{\partial \mathbf{x}}$ 的均值趋向于0；
- $\sigma$ 的加入使得 $\frac{\partial l}{\partial \mathbf{x}}$ 的方差减小。

同时也分析了原始Layer Norm中归一化之后的线性变换参数$\gamma$和$\beta$，认为他们没用，加入只会带来过拟合倾向。

# RMS Norm
RMS Norm[[3]](#3)的基本形式是：

$$
\mathbf{y} = \frac{\mathbf{x}}{RMS(\mathbf{x})} \cdot \gamma \qquad where\, RMS(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}
$$

对比Layer Norm主要区别就是把分子的偏置、原本方差计算中的偏置、以及线性变换的偏置都去掉了，这么做对效果几乎没影响，且节省计算资源，可以视为简化的Layer Norm。

# Pre/Post Layer/RMS Norm 
对Pre和Post两种模式的对比主要在论文[[4]](#4)中完成，区别是Layer Norm放在残差操作之后（Post）还是在Multi-Head Attention之前（Pre）。两种Norm方式的对比图（左Post右Pre）

![图](/_posts/pre_post_norm.png)
<p align="center" style="color:gray"> 图1 左侧为Post Norm，LN分别出现在attention和FFN的残差计算之后；右侧为Pre Norm，LN出现在attention和FFN之前。</p>

主要发现：
- 关注不同层数网络的最后一层FFN的第二层参数的梯度，随着层数$L$不同，Pre LN的梯度与$1/\sqrt{L}$ 相关，Post LN不相关。即网络层数越多，Pre LN的最后一层梯度尺度越小，Post LN则保持一致；
- 关注同一网络不同层FFN第二层参数的梯度，Pre LN各层梯度尺度稳定（实际上是越往后越小），Post LN越往后变越大。

通过试验发现： Pre LN参数梯度更稳定，不需要warm up，收敛速度快，但是Pre LN效果不如Post LN好，会掉点；Post LN则比较依赖warm up。


# DeepNorm 
DeepNorm [[4]](#4)想要Post LN的性能，同时想要Pre LN的稳定性。给出的方案是：
- 对残差链接放大一定倍数
- 在初始化时对FFN、V矩阵、输出映射矩阵缩小一定倍数
- 放大和缩小的倍数随网络结构调整，如下图

![图](/_posts/deepnorm.png)
<p align="center" style="color:gray"> 图1 左：实现代码，右：不同网络结构的参数配置</p>

如此操作后，达到的效果是每层的梯度尺度比Post LN更稳定，训练收敛性更好，同时在效果上与Post LN相当。


<div id="1">[1] Xu, J., Sun, X., Zhang, Z., Zhao, G., &#38; Lin, J. (2019). Understanding and Improving Layer Normalization. <i>Advances in Neural Information Processing Systems</i>, <i>32</i>. https://arxiv.org/abs/1911.07013v1</div>

<div id="2">[2] Ba, J. L., Kiros, J. R., &#38; Hinton, G. E. (2016). <i>Layer Normalization</i>. https://arxiv.org/abs/1607.06450v1</div>

<div id="3">[3] Zhang, B., &#38; Sennrich, R. (2019). Root Mean Square Layer Normalization. <i>Advances in Neural Information Processing Systems</i>, <i>32</i>. https://arxiv.org/abs/1910.07467v1</div>

<div id="4"> [4] Wang, H., Ma, S., Dong, L., Huang, S., Zhang, D., &#38; Wei, F. (2022). <i>DeepNet: Scaling Transformers to 1,000 Layers</i>. https://arxiv.org/abs/2203.00555v1</div>