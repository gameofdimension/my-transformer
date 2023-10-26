### 问题

已知 $Q\in \mathbb{R}^{m\times d},K \in \mathbb{R}^{n\times d},V \in \mathbb{R}^{n\times e}$，自注意力的计算过程可以表示为：

$S=QK^T$

$P=softmax(s)$

$O=PV$

其中的 softmax 操作是在行内进行，并且 $S$ 的计算略去了 $\sqrt d$ 的缩放因子。在 transformer 架构下，一般有 $m=n,d=e$，从以上公式我们可以看到 $S,P$ 的形状为 $m\times n$，在上下文序列长度很大（训练时一般在数千）的情况下，它们对显卡内存的消耗是很大的，由此带来的对显存的 io 也非常可观。[flash attention](https://arxiv.org/abs/2205.14135) 即是期望在不显式生成 $S,P$ 的前提下高效准确地完成自注意力的前向和反向运算。

### 数值稳定的 softmax

softmax 的数学表达式是这样的

$[p_1,p_2,\cdots,p_n]=$

$softmax([s_1,s_2,\cdots,s_n])=$

$[\frac{e^{s_1}}{\sum_i{e^{s_i}}},\frac{e^{s_2}}{\sum_i{e^{s_i}}},\cdots,\frac{e^{s_n}}{\sum_i{e^{s_i}}}]$

这里存在着指数运算，如果 $s_i$ 是很大的正数，则很容易造成溢出（超出浮点数范围），导致结果出现 nan；如果 $s_i$ 是绝对值很大的负数，也有下溢丢失精度的问题。

注意到 $softmax(s_1,s_2,\cdots,s_n)=softmax(s_1-c,s_2-c,\cdots,s_n-c)$ 其中 $c$ 为任意常数，那么我们将 $s$ 向量减去一个适当的标量常数则可以避免溢出，实务上一般取 $c=\max_i(s_i)$，以尽可能保证幂的指数部分在一个合适的范围。

### 前向分块计算
给定 $K,V$ 的前提下， $Q$ 中不同行 $Q_i$ 对应的计算互不依赖，所以不妨从 $Q\in\mathbb{R}^{1\times d}$ 进行思考， $Q$ 和 $K$ 点乘得到分数向量 $\{s_i\}$ 然后 softmax 得到一个离散概率分布 $p_i$，以其为 weight 对 $V_i$ 做加权求和

\$O=\sum\_{i=1}^n{p_i\cdot V_i}=$

$\sum_{i=1}^n\frac{e^{Q_1K_i^T}}{\sum_{j=1}^n e^{Q_1K_j^T}}\cdot V_i=$

$\frac{\sum_{i=1}^ne^{Q_1K_i^T}\cdot V_i}{\sum_{i=1}^n e^{Q_1K_i^T}}=$

$\frac{\sum_{i=1}^ne^{s_i}\cdot V_i}{\sum_{i=1}^ne^{s_i}}=$

$\frac{\sum_{i=1}^ne^{s_i-m}\cdot V_i}{\sum_{i=1}^ne^{s_i-m}}=$

$\frac{numerator}{denominator}$

其中 $s_i=Q_1 K_i^T,m=\max_i{s_i}$。有了这个公式，迭代式计算的代码就很好写了，分别累加分子和分母，最后做一个除法。临时的累加结果内存占用都很小，达到了节省内存的目的。flash attention 在分块内采用了高效的向量和矩阵计算，分块之间则用累加的思想进行合并，唯一要注意的是上面提到的数值稳定性，需正确地维护当前见过的最大的 $s_i$。

以下展示下两个分块计算 $a..b$ 和 $c..d$ 的结果如何合并：

$\frac{\sum_{i=a}^be^{s_i}\cdot V_i+\sum_{i=c}^de^{s_i}\cdot V_i}{\sum_{i=a}^be^{s_i}+\sum_{i=c}^de^{s_i}}=$

$\frac{e^{m_1}\sum_{i=a}^be^{s_i-m_1}\cdot V_i+e^{m_2}\sum_{i=c}^de^{s_i-m_2}\cdot V_i}{e^{m_1}\sum_{i=a}^be^{s_i-m_1}+e^{m_2}\sum_{i=c}^de^{s_i-m_2}}=$

$\frac{e^{m_1-m}\sum_{i=a}^be^{s_i-m_1}\cdot V_i+e^{m_2-m}\sum_{i=c}^de^{s_i-m_2}\cdot V_i}{e^{m_1-m}\sum_{i=a}^be^{s_i-m_1}+e^{m_2-m}\sum_{i=c}^de^{s_i-m_2}}=$

$\frac{e^{m_1-m}\cdot numerator_1+e^{m_2-m}\cdot numerator_2}{e^{m_1-m}\cdot denominator_1+e^{m_2-m}\cdot denominator_2}$

其中 $m_1=\max_{i=a..b}(s_i),m_2=\max_{i=c..d}(s_i),m=\max(m_1,m_2)$，易知以上合并公式可以保持计算的数值稳定性。
有了上面的铺垫，flash attention 团队最新的 flash decoding 工作就很好理解了：在 $K/V\$ 序列的维度分块并且并行计算，最后根据以上所述思路合并多块计算结果，实现难点也是落在了为了数值稳定性的最大指数的维护而已。

### 反向分块计算
反向计算的公式如下：

$dV=P^T\cdot{dO}$

${dP}={dO}\cdot V^T$

$dS=dsoftmax(dP)$

$dQ=dS\cdot K$

$dK=dS^T\cdot Q$

这里单独出现的 $dX$，代表真实含义其实都是 $\frac{df}{dX}$，$f$ 为某个损失函数，此种记法以下不再特别说明。

同样的我们从 $m=1$ 进行思考。全量的 $S_1,P_1$ 不可用，计算必然也是分块的。

从第一个公式可知计算局部的 $dV_{i..j}$，只需局部的 $P_{1,i..j}$ 即可，在保存了前向计算的分母 $denominator$ 和最大指数 $m$ 的前提下，我们只需 $S_{1,i..j}$ 即可得出 $P_{1,i..j}$。因为 $m=1$，为记号简单，以下将 $S_1,P_1$ 简记为 $S,P$。

第二个公式，局部的 $dP_{i..j}$ 只需局部的 $V_{i..j}$。

第四个公式，从局部的 $dS_{i..j}$ 和局部的 $K_{i..j}$ 得到 $dQ$ 的一部分增量，所有分块对应的这种增量的总和即是最终的 $dQ$。

第五个公式，从局部的 $dS_{i..j}$ 和局部的 $Q_{i..j}$ 得到 $dK_{i..j}$。

第三个公式是块硬骨头，如何从局部的 $S_{i..j},P_{i..j},dP_{i..j}$ 得到局部的 $dS_{i..j}$。直觉上任意分量 $S_{k}$ 参与 $P_{1..n}$ 所有分量的计算，那么按照求导的链式法则，要求其导数，需要计算来自 $P_{1..n}$ 所有分量的梯度，但是我们目前还没有下标 $j$ 之后的 $P_{j+1..n}$ 的梯度。公式表达则是如下，令

$(s_1,s_2,\cdots,s_n)=S$

$(p_1,p_2,\cdots,p_n)=P=softmax(s_1,s_2,\cdots,s_n)$

我们有

```math
(ds_1,ds_2,\cdots,ds_n)=dS=dP\cdot\frac{dP}{dS}=(dp_1,dp_2,\cdots,dp_n)\cdot\begin{bmatrix}
    \frac{dp_1}{ds_1} & \frac{dp_1}{ds_2}  & \dots  & \frac{dp_1}{ds_n} \\
    \frac{dp_2}{ds_1} & \frac{dp_2}{ds_2}  & \dots  & \frac{dp_2}{ds_n} \\
    \vdots & \vdots  & \ddots & \vdots \\
    \frac{dp_n}{ds_1} & \frac{dp_n}{ds_2}  & \dots  & \frac{dp_n}{ds_n}
\end{bmatrix}
```

初看起来，要分块式的计算 $dS$ 是不可能了。
但是论文中有个很精彩的变换，突破了以上困境，简述如下：计算出每个 $\frac{dp_i}{ds_j}$，然后我们有

```math
dS=dP\cdot
\begin{bmatrix}
    p_1-p_1 p_1 & -p_1 p_2  & \dots  & -p_1 p_n \\
    -p_2 p_1 & p_2-p_2 p_2  & \dots  & -p_2 p_n \\
    \vdots & \vdots  & \ddots & \vdots \\
    -p_n p_1 & -p_n p_2  & \dots  & p_n-p_n p_n
\end{bmatrix}=dP \cdot \{diag(P)-P^TP\}=dP\circ P-(dP\cdot P^T)P
```

其中 $\circ$ 表示逐点相乘。同时可以知道 $dP\cdot P^T$ 为一个标量，因而是可以事先计算（分块增量式）出来的，这样有了分块的 $P_{i..j},dP_{i..j}$，就可以计算出分块的 $dS_{i..j}$ 了。
其实 $dP\cdot P^T$ 还可以进一步推导，因为 $O=PV,dP=dO\cdot V^T$ 我们有 $dP\cdot P^T=dO\cdot V^T\cdot P^T=dO\cdot (PV)^T=dO\cdot O^T$。显然 $dO\cdot O^T$ 的计算量更小，这也正是论文中所采用的方法。

### 概念层面复现

之所以叫概念层面复现，是因为主要是为了展示笔者所理解的算法思想，而并非追求百分百复刻

- [1d 版增量计算](https://github.com/gameofdimension/my-transformer/blob/f21cfe4e2b732dad57bf59c0009e27595203bbd0/flash/incremental.py#L4)。从 $S,V$ 计算 $O$
- [2d 版增量计算](https://github.com/gameofdimension/my-transformer/blob/f21cfe4e2b732dad57bf59c0009e27595203bbd0/flash/incremental.py#L73)。从 $S,V$ 计算 $O$
- flash attention 分块式[前向计算](https://github.com/gameofdimension/my-transformer/blob/f21cfe4e2b732dad57bf59c0009e27595203bbd0/flash/v2.py#L11)
- flash attention 分块式[后向计算](https://github.com/gameofdimension/my-transformer/blob/f21cfe4e2b732dad57bf59c0009e27595203bbd0/flash/v2.py#L84)
- 模仿 flash decoding 的[多块结果合并](https://github.com/gameofdimension/my-transformer/blob/f21cfe4e2b732dad57bf59c0009e27595203bbd0/flash/incremental.py#L89)
