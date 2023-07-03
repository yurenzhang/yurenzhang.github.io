---
layout: post
title:  "Gradient Checkpoint"
tags: 算法
excerpt_separator: <!--more-->
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            
            }
        });
    </script>
</head>


记录下Gradient Checkpoint节省内存的思路。<!--more-->
# BP过程和内存占用

首先回顾下BP过程中需要用到哪些数据。

![BP](/_posts/BP.png)
<p style="color:gray;text-align: center"> 图1 BP过程。 $z^l$为第$l$层的隐藏变量， $f()$为激活函数，$x^{l} = f(z^l)$为本层的输出，下一层的输入。</p>

<!-- $$
x^l = f(z^l) \\
z^{l+1} \\
W^{l+1}
$$ -->

在BP中比较核心的概念是“残差”$\delta$。
对应的残差传播的核心公式：

$$
\delta_i^l=  \frac{\partial{J}}{\partial{z_i^l}} = (\sum W^{l+1}_{ij}\delta_j^{l+1})*f'(z_i^l)
$$

它展示了在两个相邻层之间 $\delta_i^l$和$\delta_i^{l+1}$ 是怎么向后传播的，有了它就知道$W$怎样更新了：

$$
\frac{\partial J}{\partial W^{l+1}} = \delta^{l+1} (x^l)^T
$$





可以看到如果不考虑任何优化，对于每次feed forward，这个计算过程中需要计算和记录：

- $z^l$  ：每一层都需要记录，以做传播
- $\delta_i^l$ ：保留当前层，每层计算完后可以抛弃
- $f'(z^l)$可以基于$z^l$ 计算得到，不需要额外的存储负担
- $x^l$同理，每次计算后可以丢弃

随着层数增长，要缓存的$z^l$就越来越多，造成内存负担。此外，如果使用Adam优化，则需要额外储存$m$和$v$的信息。

<!-- $f(z^l)$

$z^l$

$z^{l+1}$

$\delta_j^{l+1} = \frac{\partial{J}}{\partial{z_j^{l+1}}}$ -->

---

# Gradient Checkpoint
不做优化的BP内存会随层数增长线性增长，参照[Github链接](#1)中的介绍，如果使用最少的内存，可以在每层计算BP时都从头开始跑一遍feed forward。这样内存省下了，但时间消耗过大。Gradient Checkpoint就是一种折中方案：在整个链路中选取若干层做checkpoint，其他层的$z^l$基于这些checkpoint做计算。
![checkpint](/_posts/checkpoint.png)
如此一来，就只有checkpoint层的中间结果持续存在于内存，其他层的$z^l$则在算到的时候再临时推算，从而减少内存。

在一个toy模型上做简单的测试，定义一个100层的CNN，用来做MNIST分类：
```python
self.conv_layers = nn.ModuleList()
self.conv_layers.append(nn.Conv2d(1, 320, 3, padding=2))
for i in range(100):
    self.conv_layers.append(nn.Conv2d(320, 320, 3,padding=2))
    self.conv_layers.append(nn.ReLU())
    self.conv_layers.append(nn.MaxPool2d(2, 2))
self.fc1 = nn.Linear(1280, 10)
self.fc2 = nn.Linear(10, 10)
```
在pytorch中可以直接调用checkpoint_sequential函数做checkpoint
```python
segments = 10
# checkpoint_sequential只对序列堆叠的模型有用，所以先对前100层作用
mid_result = checkpoint_sequential(modules[0], segments, inputs)
# checkpoint不支持view函数，所以单独拉出来
mid_result = mid_result.view(-1, 1280)
outputs = checkpoint_sequential(modules[1:], 1, mid_result)
```
对比直接运行和checkpoint方式运行占用的内存：
- 直接运行：～1139M
- checkpoint segment = 1：～1069M
- checkpoint segment = 2：～854M
- checkpoint segment = 4：～686M
- checkpoint segment = 8：～609M
- checkpoint segment = 16：～590M
- checkpoint segment = 32：～580M


<div id="1">[1] https://github.com/cybertronai/gradient-checkpointing </div>
<div id="2">[2] https://doi.org/10.1109/ACCESS.2019.2931579 </div>


