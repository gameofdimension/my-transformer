# my-transformer
---

### 动机

该 repo 用来探索 transformer 架构（主要是 decoder）的各种技术要点

### 方法

1. 参照官方实现，手工编码实现模型
2. 加载官方模型权重，测试若干个 case，确保二者输出一致
3. 一些模型如果比较大，可能需要裁剪，比如只使用前几层进行计算。因为各层架构一样，这样做是合理的

### 各模型技术要点一览

| |位置信息融入|normalization|激活函数|注意力机制|
|---|---|---|---|---|
|[gpt2](model/gpt2.py)|可学习绝对位置编码|LayerNorm|gelu|单向自注意力|
|[llama](model/llama.py)|RoPE|RMSNorm|silu|单向自注意力|
|[baichuan13b](model/baichuan13b.py)|Alibi|RMSNorm|silu|单向自注意力|
|[chatglm](model/chatglm.py)|RoPE, 2d position|post LayerNorm|gelu|前缀部分双向注意力，生成部分单向注意力|
|[chatglm2](model/chatglm2.py)|RoPE|RMSNorm|swiglu|单向，分组注意力|
|[mistral](model/mistral.py)|RoPE|RMSNorm|silu|单向，分组注意力，滑动窗口注意力|
|[mixtral](model/mixtral.py)|RoPE|RMSNorm|silu|单向，分组注意力|
|[phi2](model/phi2.py)|RoPE|LayerNorm|gelu|单向注意力|


- gpt2
  1. 标准 transformer 的 decoder 部分，只是去掉了来自 encoder 的 cross attention 输入。或者从另一个角度，它跟作为 encoder 的 bert 架构一样，但是 attention mask 是下三角的，以达到单向注意力的目的
  2. 不同于原始 transformer，没有用正余弦位置编码，而是可学习的绝对位置编码
  3. hugging face 实现中线性层用一维卷积实现，没有用 `nn.Linear` ，因此加载权重的时候需要转置一下
- llama
  1. 没有使用标准 LayerNorm，而是 RMSNorm。二者区别是后者假定数据均值为 0，因此无需平移
  2. 弃用绝对位置编码，使用旋转位置编码，作用在 query 和 key 上
- baichuan13b
  1. baichuan7b 官方称架构跟 llama 一致。采用 RMSNorm 和旋转位置编码
  2. baichuan13b 的一个不同是使用 Alibi 编码位置信息，在 attention score 上加了跟距离成正比的偏移量
- chatglm
  1. 来自清华的 glm 架构，宣称通吃自然语言理解和自然语言生成
  2. 不同于其他的 transformer 变种，它的一个 token 的位置信息用两个数字表示：所在的块编号和块内位置
  3. 前缀输入部分使用双向注意力，生成部分是单向注意力，这些需在 attention mask 上表达出来
  4. 使用了旋转位置编码，但是旋转向量时前半部分应用块编号，后半部分应用块内位置
- chatglm2
  1. 名字上看可能以为跟 chatglm 差不多，但是变化很大，更像一般的生成式 decoder 了。也说得过去，发论文和商业毕竟是不同的面向
  2. 使用了 RMSNorm ，而 chatglm 使用的是 LayerNorm
  3. 2d 位置编码去掉了。跟大多数流行架构一样，用唯一的数字表示一个 token 的位置，不同点在于只用该信息旋转输入向量的前半部分，后半部分不变
  4. 使用标准单向注意力，attention mask 也没那么复杂了，估计对自然语言理解的关注降低了
  5. 使用了 GQA ，所谓的分组注意力。标准注意力每个头对应一组 qkv ，为了减少显存占用，可以缩减 key 和 value 的数量，如果减到 1 则是 MQA，标准注意力是 h（头数），二者之间则是 GQA
- mistral
  1. 分组注意力。32个 query head，8 个 kv head
  2. 滑动窗口。4096 的滑动窗口，因此 prompt 较短的一般应用下跟其他的非滑动窗口架构没有差异
- mixtal
  1. 跟 mistral 一样的分组注意力
  2. 取消了滑动窗口，huggingface 上的早期版本的 `config.json` 写着有 4096 的滑动窗口。后面官方证明是乌龙，Reddit 看到有人被坑到，发帖吐槽了。该模型支持最长 32k 上下文，没有滑动窗口情况下注意力部分的计算量很大，如何高效训练的值得探究
  3. MoE 取代 decoder transformer 中的 FFN，各个 expert 本身是一个 FFN，其实就是用一组 FFN 取代原来的一个 FFN。训练上可能挺麻烦的，要保证各个 expert 被差不多同样程度的训练到，同时还要做到高效。推理从概念上极其简单：仅根据当前 token 的 hidden state 从 8 个experts 中选择 top2，hidden state 再经过这两个 FFN 计算，加权求和即是 MoE 的最终结果
- phi2
  1. 微软 phi 系列的最新版本
  2. RoPE 旋转的维数是 32，而 head 的维数 80。这种设计还是相对少见，按理说 RoPE 计算开销也不大，为什么要这样省搞不懂。chatglm2 也是只旋转部分维度，但是那里我宁愿相信是因为 chatglm 的历史负担
  3. FFN 和 Attention 并行而不是串行，扫了一眼论文也不是它的首创，貌似也是出于高效利用某类硬件的目的
