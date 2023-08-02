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
|gpt2|可学习绝对位置编码|LayerNorm|gelu_new|单向自注意力|
|llama|RoPE|RMSNorm|silu|单向自注意力|
|baichuan13b|Alibi|RMSNorm|silu|单向自注意力|
|chatglm|RoPE, 2d position|post LayerNorm|gelu|前缀部分双向注意力，生成部分单向注意力|
|chatglm2|RoPE|RMSNorm|swiglu|单向，分组注意力|

- gpt2
  1. 标准 transformer 的 decoder 部分，只是去掉了来自 encoder 的 cross attention 输入。或者从另一个角度，它跟作为 encoder 的 bert 架构一样，但是 attention mask 是下三角的，以达到单向注意力的目的
  2. 不同于原始 transformer，没有用正余弦位置编码，而是可学习的绝对位置编码
  3. hugging face 实现中线性层用一阶卷积实现，没有用 `nn.Linear` ，因此加载权重的时候需要转置一下
- llama
  1. 没有使用标准 layernorm，而是 rmsnorm。二者区别是后者假定数据均值为 0，因此无需平移
  2. 弃用绝对位置编码，使用旋转位置编码，作用在 query 和 key 上
- baichuan13b
  1. baichuan7b 官方称架构跟 llama 一致。采用 rmsnorm 和旋转位置编码
  2. baichuan13b 的一个不同是使用 alibi 编码位置信息，在 attention score 上加了跟距离成正比的偏移量
- chatglm
  1. 来自清华的 glm 架构，宣称通吃自然语言理解和自然语言生成
  2. 不同于其他的 transformer 变种，它的一个 token 的位置信息用两个数字表示：所在的块编号和块内位置
  3. 前缀输入部分使用双向注意力，生成部分是单向注意力，这些需在 attention mask 上表达出来
  4. 使用了旋转位置编码，但是旋转向量时前半部分应用块编号，后半部分应用块内位置
- chatglm2
  1. 名字上看可能以为跟 chatglm 差不多，但是变化很大，更像一般的生成式 decoder 了。也说得过去，发论文和商业毕竟是不同的面向
  2. 使用了 rmsnorm ，而 chatglm 使用的是 layernorm
  3. 2d 位置编码去掉了。跟大多数流行架构一样，用唯一的数字表示一个 token 的位置，不同点在于只用该信息旋转输入向量的前半部分，后半部分不变
  4. 使用标准单向注意力，attention mask 也没那么复杂了，估计对自然语言理解的关注降低了
  5. 使用了 GQA ，所谓的分组注意力。标准注意力每个头对应一组 qkv ，为了减少显存占用，可以缩减 key 和 value 的数量，如果减到 1 则是 MQA，标准注意力是 h（头数），二者之间则是 GQA