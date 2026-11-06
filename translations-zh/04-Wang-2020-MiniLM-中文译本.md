> 出处说明：本文是论文《MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers》（Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, Ming Zhou, 2020）的中文译本，原文链接：https://arxiv.org/abs/2002.10957 （译自 arXiv v2，2020 年 4 月 6 日版）

# MiniLM：用于预训练 Transformer 任务无关压缩的深度自注意力蒸馏

**原题：** MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, Ming Zhou
微软研究院（Microsoft Research）
{wenwan,fuwei,lidong1,t-habao,nanya,mingzhou}@microsoft.com

通讯作者：Furu Wei <fuwei@microsoft.com>

> 译者注：文中「译者注」均为译者为初学者添加，原文没有。公式用 LaTeX 写出；参考文献保留英文原文。原文中 BERT_BASE 等下标在下文写作 BERT<sub>BASE</sub>。「任务无关」（task-agnostic）指蒸馏发生在预训练阶段，得到的小模型不针对某个具体任务，之后可以像 BERT 一样拿去微调到任意下游任务。

---

## 摘要（Abstract）

预训练语言模型（例如 BERT（Devlin et al., 2018）及其变体）在各种 NLP 任务上取得了显著成功。然而，这些模型通常包含数亿个参数，受延迟和容量的限制，给实际应用中的微调（fine-tuning）和在线服务（online serving）带来了挑战。在本工作中，我们提出一种简单而有效的方法来压缩基于 Transformer（Vaswani et al., 2017）的大型预训练模型，称为深度自注意力蒸馏（deep self-attention distillation）。小模型（学生，student）通过深度模仿大模型（教师，teacher）的自注意力模块（self-attention module）来训练，该模块在 Transformer 网络中起着至关重要的作用。具体来说，我们提出蒸馏教师最后一个 Transformer 层的自注意力模块，这对学生而言既有效又灵活。此外，除了已有工作使用的注意力分布（attention distributions，即查询和键的缩放点积（scaled dot-product）），我们还引入自注意力模块中值（values）之间的缩放点积，作为新的深度自注意力知识。另外，我们表明，引入一个助教（teacher assistant）（Mirzadeh et al., 2019）也有助于大型预训练 Transformer 模型的蒸馏。实验结果表明，我们的单语言模型¹ 在不同参数规模的学生模型上都优于最先进的基线。特别是，它只用教师模型 50% 的 Transformer 参数和计算量，就在 SQuAD 2.0 和若干 GLUE 基准任务上保留了 99% 以上的准确率。将深度自注意力蒸馏应用于多语言预训练模型，我们也取得了有竞争力的结果。

¹ 代码和模型已公开：https://aka.ms/minilm

---

## 1. 引言（Introduction）

语言模型（language model，LM）预训练已经在各种自然语言处理任务上取得了显著成功（Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018; Dong et al., 2019; Yang et al., 2019; Joshi et al., 2019; Liu et al., 2019）。BERT（Devlin et al., 2018）及其变体等预训练语言模型，利用大规模文本语料、通过根据上下文预测词语来学习上下文相关的文本表示（contextualized text representations），并可以加上额外的特定任务层进行微调，以适配下游任务。然而，这些模型通常包含数亿个参数，受延迟和容量的限制，给实际应用中的微调和在线服务带来了挑战。

知识蒸馏（knowledge distillation，KD）（Hinton et al., 2015; Romero et al., 2015）已被证明是一种很有前景的方法，它可以把一个大模型（称为教师模型）压缩成一个小模型（称为学生模型），小模型使用少得多的参数和计算，同时在下游任务上取得有竞争力的结果。已有一些工作以特定任务（task-specific）的方式把预训练大型 LM 蒸馏到小模型中（Tang et al., 2019; Turc et al., 2019b; Sun et al., 2019a; Aguilar et al., 2019）。它们先在特定任务上微调预训练 LM，然后再进行蒸馏。特定任务蒸馏是有效的，但微调大型预训练模型的成本仍然很高，对于大型数据集尤其如此。与特定任务蒸馏不同，任务无关（task-agnostic）的 LM 蒸馏模仿原始预训练 LM 的行为，学生模型可以直接在下游任务上微调（Tsai et al., 2019; Sanh et al., 2019; Jiao et al., 2019; Sun et al., 2019b）。

以往的工作使用教师 LM 在掩码语言建模（masked language modeling）预测上的软目标概率（soft target probabilities），或教师的中间表示，来指导任务无关学生的训练。DistilBERT（Sanh et al., 2019）使用软标签（soft-label）蒸馏损失和余弦嵌入损失（cosine embedding loss），并从教师中每两层取一层来初始化学生。但这要求学生的每个 Transformer 层都与其教师具有相同的架构。TinyBERT（Jiao et al., 2019）和 MobileBERT（Sun et al., 2019b）利用了更细粒度的知识，包括 Transformer 网络的隐藏状态（hidden states）和自注意力分布，并逐层（layer-to-layer）把这些知识迁移给学生模型。为了执行逐层蒸馏，TinyBERT 采用一个均匀函数来确定教师层与学生层之间的映射，并用一个参数矩阵对学生的隐藏状态做线性变换。MobileBERT 假设教师和学生层数相同，并引入瓶颈模块（bottleneck module）使二者的隐藏层大小保持一致。

在本工作中，我们为基于 Transformer 的任务无关 LM 蒸馏提出深度自注意力蒸馏框架。核心思想是深度模仿自注意力模块，它是基于 Transformer 的教师模型和学生模型中最根本、最重要的组件。具体来说，我们提出蒸馏教师模型最后一个 Transformer 层的自注意力模块。与以往方法相比，使用最后一个 Transformer 层的知识而不是进行逐层知识蒸馏，缓解了教师模型与学生模型之间层映射的困难，我们学生模型的层数也可以更灵活。此外，除了已有工作使用的注意力分布（即查询和键的缩放点积），我们还引入自注意力模块中值之间的缩放点积，作为新的深度自注意力知识。使用自注意力值之间的缩放点积，还能把不同维度的表示转换成相同维度的关系矩阵（relation matrices），无需引入额外参数来变换学生的表示，从而允许学生模型采用任意的隐藏维度。最后，我们表明，引入助教（Mirzadeh et al., 2019）有助于基于 Transformer 的大型预训练模型的蒸馏，而且所提出的深度自注意力蒸馏还能进一步提升性能。

我们在下游 NLP 任务上进行了大量实验。实验结果表明，我们的单语言模型在不同参数规模的学生模型上都优于最先进的基线。具体来说，从 BERT<sub>BASE</sub> 蒸馏得到的 6 层、768 隐藏维度模型速度快 2.0 倍，同时在 SQuAD 2.0 和若干 GLUE 基准任务上保留了 99% 以上的准确率。此外，我们从 XLM-R<sub>Base</sub> 蒸馏得到的多语言模型，也以少得多的 Transformer 参数取得了有竞争力的表现。

---

## 2. 预备知识（Preliminary）

多层 Transformer（Vaswani et al., 2017）一直是最先进的预训练模型中使用最广泛的网络结构。本节我们简要介绍 Transformer 网络及其核心组件自注意力机制（self-attention mechanism）。我们还会介绍现有的 Transformer 网络知识蒸馏方法，尤其是在把基于 Transformer 的大型预训练模型蒸馏成小型 Transformer 模型这一场景下的方法。

### 2.1. 输入表示（Input Representation）

在 BERT（Devlin et al., 2018）中，文本由 WordPiece（Wu et al., 2016）切分成子词（subword）单元。例如，单词 "forecasted" 被切分为 "forecast" 和 "##ed"，其中 "##" 表示这些片段属于同一个词。如果输入文本包含不止一个片段（segment），则用一个特殊边界词元 [SEP] 将它们分隔开。序列开头会添加一个特殊词元 [CLS]，用来获得整个输入的表示。输入词元的向量表示（$\{x_i\}_{i=1}^{|x|}$）由对应的词元嵌入（token embedding）、绝对位置嵌入（absolute position embedding）和片段嵌入（segment embedding）相加得到。

### 2.2. 骨干网络：Transformer（Backbone Network: Transformer）

Transformer（Vaswani et al., 2017）用于为输入词元编码上下文信息。输入向量 $\{x_i\}_{i=1}^{|x|}$ 被打包成 $H^0 = [x_1, \cdots, x_{|x|}]$。然后堆叠的 Transformer 块按如下方式计算编码向量：

$$
H^l = \mathrm{Transformer}_l(H^{l-1}),\quad l \in [1, L] \tag{1}
$$

其中 $L$ 是 Transformer 层数，最终输出为 $H^L = [h_1^L, \cdots, h_{|x|}^L]$。隐藏向量 $h_i^L$ 用作 $x_i$ 的上下文表示。每个 Transformer 层由一个自注意力子层和一个全连接前馈网络（feed-forward network）组成。两个子层都使用残差连接（residual connection）（He et al., 2016），之后接层归一化（layer normalization）（Ba et al., 2016）。

**自注意力（Self-Attention）** 在每一层中，Transformer 使用多个自注意力头（self-attention heads）来聚合上一层的输出向量。对于第 $l$ 个 Transformer 层，一个自注意力头的输出 $AO_{l,a}$（$a \in [1, A_h]$）按如下方式计算：

$$
Q_{l,a} = H^{l-1} W_{l,a}^{Q},\quad K_{l,a} = H^{l-1} W_{l,a}^{K},\quad V_{l,a} = H^{l-1} W_{l,a}^{V} \tag{2}
$$

$$
A_{l,a} = \mathrm{softmax}\!\left(\frac{Q_{l,a} K_{l,a}^{\top}}{\sqrt{d_k}}\right) \tag{3}
$$

$$
AO_{l,a} = A_{l,a} V_{l,a} \tag{4}
$$

其中上一层的输出 $H^{l-1} \in \mathbb{R}^{|x| \times d_h}$ 分别通过参数矩阵 $W_{l,a}^{Q}, W_{l,a}^{K}, W_{l,a}^{V} \in \mathbb{R}^{d_h \times d_k}$ 线性投影为查询、键和值三元组。$A_{l,a} \in \mathbb{R}^{|x| \times |x|}$ 表示注意力分布，由查询和键的缩放点积计算得到。$A_h$ 表示自注意力头的数量。在 BERT 中，$d_k \times A_h$ 等于隐藏维度 $d_h$。

> **图 1**（未重绘）：深度自注意力蒸馏概览（Overview of Deep Self-Attention Distillation）。学生通过深度模仿教师最后一个 Transformer 层的自注意力行为来训练。除了自注意力分布之外，我们引入自注意力值关系迁移（value-relation transfer），帮助学生实现更深层的模仿。我们的学生模型命名为 MiniLM。图中左侧为教师（$L$ 层）、右侧为学生（$M$ 层），在两者的最后一层上分别对「查询–键缩放点积」做注意力迁移（Attention Transfer，KL 散度），对「值–值缩放点积」做值关系迁移（Value-Relation Transfer，KL 散度）。

### 2.3. Transformer 蒸馏（Transformer Distillation）

知识蒸馏（Hinton et al., 2015; Romero et al., 2015）是指在一个迁移特征集（transfer feature set）上，用大型教师模型 $T$ 提供的软标签和中间表示来训练小型学生模型 $S$。知识蒸馏被建模为最小化教师特征与学生特征之间的差异：

$$
\mathcal{L}_{KD} = \sum_{e \in \mathcal{D}} L\big(f^S(e),\, f^T(e)\big) \tag{5}
$$

其中 $\mathcal{D}$ 表示训练数据，$f^S(\cdot)$ 和 $f^T(\cdot)$ 分别表示学生模型和教师模型的特征，$L(\cdot)$ 表示损失函数。均方误差（mean squared error，MSE）和 KL 散度（KL-divergence）常被用作损失函数。

对于基于 Transformer 的 LM 蒸馏，教师模型在掩码语言建模预测上的软目标概率、嵌入层输出、自注意力分布以及每个 Transformer 层的输出（隐藏状态）都被用作特征，帮助学生训练。DistilBERT 使用了软标签和嵌入层输出。TinyBERT 和 MobileBERT 进一步利用了每个 Transformer 层的自注意力分布和输出。对于 MobileBERT，为了进行逐层蒸馏，要求学生与教师层数相同。此外，它还引入瓶颈和倒瓶颈（inverted bottleneck）模块，使教师和学生的隐藏层大小也保持一致。为了逐层迁移知识，TinyBERT 采用一个均匀函数来映射教师层和学生层。由于学生的隐藏层大小可以比教师小，因此引入了一个参数矩阵来变换学生的特征。

> 译者注：原文中 DistilBERT 多处拼作「DistillBERT」，译文统一写作 DistilBERT；表格中保留原文拼写。

---

## 3. 深度自注意力蒸馏（Deep Self-Attention Distillation）

图 1 给出了深度自注意力蒸馏的概览。核心思想有三点。第一，我们提出通过深度模仿教师最后一层的自注意力模块（Transformer 中的关键组件）来训练学生。第二，除了在自注意力模块中进行注意力分布（即查询和键的缩放点积）的迁移之外，我们还引入值之间关系（即值之间的缩放点积）的迁移，以实现更深层的模仿。此外，我们表明，当教师模型与学生模型的规模差距很大时，引入助教（Mirzadeh et al., 2019）也有助于大型预训练 Transformer 模型的蒸馏。

### 3.1. 自注意力分布迁移（Self-Attention Distribution Transfer）

注意力机制（Bahdanau et al., 2015）一直是 NLP 任务中非常成功的神经网络组件，对预训练 LM 也至关重要。一些工作表明，预训练 LM 的自注意力分布捕获了丰富的层次化语言学信息（Jawahar et al., 2019; Clark et al., 2019）。以往的 Transformer 蒸馏工作已经使用过自注意力分布的迁移（Jiao et al., 2019; Sun et al., 2019b; Aguilar et al., 2019）。我们同样利用自注意力分布来帮助学生训练。具体来说，我们最小化教师与学生自注意力分布之间的 KL 散度：

$$
\mathcal{L}_{AT} = \frac{1}{A_h |x|} \sum_{a=1}^{A_h} \sum_{t=1}^{|x|} D_{KL}\big(\mathbf{A}_{L,a,t}^{T} \,\|\, \mathbf{A}_{M,a,t}^{S}\big) \tag{6}
$$

其中 $|x|$ 和 $A_h$ 分别表示序列长度和注意力头的数量。$L$ 和 $M$ 分别表示教师和学生的层数。$\mathbf{A}_L^T$ 和 $\mathbf{A}_M^S$ 分别是教师和学生最后一个 Transformer 层的注意力分布，由查询和键的缩放点积计算得到。

与以往逐层迁移教师知识的工作不同，我们只使用教师最后一个 Transformer 层的注意力图（attention maps）。蒸馏最后一个 Transformer 层的注意力知识，使我们的学生模型在层数上更加灵活，也省去了寻找最佳层映射的工作。

> 译者注：式 (6) 中的下标 $t$ 表示第 $t$ 个词元。注意力矩阵的每一行都是该词元对所有词元的一个概率分布，所以可以逐行计算 KL 散度，再对所有头和所有词元取平均。

### 3.2. 自注意力值关系迁移（Self-Attention Value-Relation Transfer）

除了注意力分布之外，我们提出使用自注意力模块中值之间的关系来指导学生训练。值关系（value relation）通过值之间的多头缩放点积来计算。教师与学生值关系之间的 KL 散度被用作训练目标：

$$
\mathbf{VR}_{L,a}^{T} = \mathrm{softmax}\!\left(\frac{\mathbf{V}_{L,a}^{T} {\mathbf{V}_{L,a}^{T}}^{\top}}{\sqrt{d_k}}\right) \tag{7}
$$

$$
\mathbf{VR}_{M,a}^{S} = \mathrm{softmax}\!\left(\frac{\mathbf{V}_{M,a}^{S} {\mathbf{V}_{M,a}^{S}}^{\top}}{\sqrt{d'_k}}\right) \tag{8}
$$

$$
\mathcal{L}_{VR} = \frac{1}{A_h |x|} \sum_{a=1}^{A_h} \sum_{t=1}^{|x|} D_{KL}\big(\mathbf{VR}_{L,a,t}^{T} \,\|\, \mathbf{VR}_{M,a,t}^{S}\big) \tag{9}
$$

其中 $\mathbf{V}_{L,a}^{T} \in \mathbb{R}^{|x| \times d_k}$ 和 $\mathbf{V}_{M,a}^{S} \in \mathbb{R}^{|x| \times d'_k}$ 分别是教师和学生最后一个 Transformer 层自注意力模块中一个注意力头的值。$\mathbf{VR}_L^T \in \mathbb{R}^{A_h \times |x| \times |x|}$ 和 $\mathbf{VR}_M^S \in \mathbb{R}^{A_h \times |x| \times |x|}$ 分别是教师和学生最后一个 Transformer 层的值关系。

训练损失由注意力分布迁移损失与值关系迁移损失相加得到：

$$
\mathcal{L} = \mathcal{L}_{AT} + \mathcal{L}_{VR} \tag{10}
$$

引入值之间的关系，使学生能够深度模仿教师的自注意力行为。此外，使用缩放点积可以把不同隐藏维度的向量转换成相同大小的关系矩阵，这使我们的学生可以使用更灵活的隐藏维度，并避免引入额外参数来变换学生的表示。

> 译者注：关键在于形状。教师的值矩阵是 $|x| \times d_k$，学生是 $|x| \times d'_k$，维度不同无法直接比较；但 $VV^\top$ 的形状都是 $|x| \times |x|$，只和句子长度有关，所以不需要额外的投影矩阵就能对齐。

### 3.3. 助教（Teacher Assistant）

沿用 Mirzadeh et al.（2019）的做法，我们引入一个助教（即中等规模的学生模型），进一步提升更小学生模型的性能。

假设教师模型由隐藏层大小为 $d_h$ 的 $L$ 层 Transformer 组成，学生模型是隐藏层大小为 $d'_h$ 的 $M$ 层 Transformer。对于更小的学生（$M \le \frac{1}{2}L$，$d'_h \le \frac{1}{2}d_h$），我们首先把教师蒸馏成一个 $L$ 层 Transformer、隐藏层大小为 $d'_h$ 的助教。然后用这个助教模型作为教师来指导最终学生的训练。引入助教弥合了教师模型与更小学生模型之间的规模差距，有助于基于 Transformer 的预训练 LM 的蒸馏。此外，把深度自注意力蒸馏与助教结合，能为更小的学生模型带来进一步的提升。

### 3.4. 与以往工作的比较（Comparison with Previous Work）

表 1 给出了与以往方法（Sanh et al., 2019; Jiao et al., 2019; Sun et al., 2019b）的比较。MobileBERT 提出使用一个专门设计的倒瓶颈模型作为教师，其模型规模与 BERT<sub>LARGE</sub> 相同。其他方法都使用 BERT<sub>BASE</sub> 进行实验。在用于蒸馏的知识方面，我们的方法引入自注意力模块中值之间的缩放点积作为新知识，以深度模仿教师的自注意力行为。TinyBERT 和 MobileBERT 逐层地把教师的知识迁移给学生。MobileBERT 假设学生与教师层数相同。TinyBERT 采用均匀策略来确定其层映射。DistilBERT 用教师的参数初始化学生，因此仍然需要选择教师模型的哪些层。MiniLM 蒸馏教师最后一个 Transformer 层的自注意力知识，允许学生的层数灵活变化，也省去了寻找最佳层映射的工作。DistilBERT 和 MobileBERT 要求学生的隐藏层大小与教师相同。TinyBERT 使用一个参数矩阵来变换学生的隐藏状态。使用值关系使我们的学生可以使用任意的隐藏层大小，而无需引入额外参数。

**表 1：** 与以往基于 Transformer 的任务无关 LM 蒸馏方法的比较。（✓ 对应原文中的「X」标记，表示「是 / 有此要求」。）

| 方法 | 教师模型 | 蒸馏的知识 | 逐层蒸馏 | 对学生层数有要求 | 对学生隐藏层大小有要求 |
|---|---|---|---|---|---|
| DistillBERT | BERT<sub>BASE</sub> | 软目标概率；嵌入输出 | | | ✓ |
| TinyBERT | BERT<sub>BASE</sub> | 嵌入输出；隐藏状态；自注意力分布 | ✓ | | |
| MobileBERT | IB-BERT<sub>LARGE</sub> | 软目标概率；隐藏状态；自注意力分布 | ✓ | ✓ | ✓ |
| MiniLM | BERT<sub>BASE</sub> | 自注意力分布；自注意力值关系 | | | |

> 译者注：表 1 中各「X」标记所在的列，是译者依据 `pdftotext -layout` 的版面位置判断的，并与正文 3.4 节的描述一致。

---

## 4. 实验（Experiments）

我们对不同参数规模的学生模型进行蒸馏实验，并在下游任务上评估蒸馏后的模型，包括抽取式问答（extractive question answering）和 GLUE 基准。

### 4.1. 蒸馏设置（Distillation Setup）

我们使用不区分大小写（uncased）版本的 BERT<sub>BASE</sub> 作为教师。BERT<sub>BASE</sub>（Devlin et al., 2018）是一个 12 层的 Transformer，隐藏层大小为 768，有 12 个注意力头，包含约 109M 参数。学生模型中注意力分布和值关系的头数都设为 12。我们使用英文维基百科²和 BookCorpus（Zhu et al., 2015）中的文档作为预训练数据，遵循 Devlin et al.（2018）的预处理和 WordPiece 分词方式。词表大小为 30,522。最大序列长度为 512。我们使用 Adam（Kingma & Ba, 2015），$\beta_1 = 0.9$，$\beta_2 = 0.999$。我们以 1024 的批大小和 5e-4 的峰值学习率，把 768 隐藏维度的 6 层学生模型训练 400,000 步。对于其他架构的学生模型，批大小和峰值学习率分别设为 256 和 3e-4。我们在前 4,000 步使用线性预热（linear warmup），之后线性衰减。dropout 率为 0.1，权重衰减（weight decay）为 0.01。

² 维基百科版本：enwiki-20181101。

我们还使用一个内部（in-house）预训练的、BERT<sub>BASE</sub> 规模的 Transformer 模型作为教师模型，把它蒸馏成隐藏层大小为 384 的 12 层和 6 层学生模型。对于 12 层模型，我们使用 Adam（Kingma & Ba, 2015），$\beta_1 = 0.9$，$\beta_2 = 0.98$。该模型以 2048 的批大小和 6e-4 的峰值学习率训练 400,000 步。6 层模型的批大小和峰值学习率设为 512 和 4e-4。其余超参数与上面基于 BERT 的蒸馏模型相同。

在训练多语言 MiniLM 模型时，我们使用 Adam（Kingma & Ba, 2015），$\beta_1 = 0.9$，$\beta_2 = 0.999$。12 层学生模型以 256 的批大小和 3e-4 的峰值学习率训练 1,000,000 步。6 层学生模型以 512 的批大小和 6e-4 的峰值学习率训练 400,000 步。

我们使用 8 块 V100 GPU，以混合精度训练（mixed precision training）的方式蒸馏学生模型。沿用 Sun et al.（2019a）和 Jiao et al.（2019）的做法，推理时间在 QNLI 训练集上、使用相同超参数进行评估。我们报告在单块 P100 GPU 上运行 100 个批次的平均时间。

### 4.2. 下游任务（Downstream Tasks）

沿用以往的语言模型预训练工作（Devlin et al., 2018; Liu et al., 2019）以及任务无关的预训练语言模型蒸馏工作（Sanh et al., 2019; Jiao et al., 2019; Sun et al., 2019b），我们在抽取式问答和 GLUE 基准上评估蒸馏模型。

**表 2：** 从 BERT<sub>BASE</sub> 蒸馏得到的、公开发布的 768 隐藏维度 6 层模型之间的比较。我们比较的是不经过特定任务蒸馏和数据增强的任务无关蒸馏模型。SQuAD 2.0 报告 F1，其他数据集报告准确率。DistilBERT 的 GLUE 结果取自 Sanh et al.（2019）。SQuAD 2.0 结果是我们微调其公开模型³得到的。对于 TinyBERT，为了公平比较，我们微调其公开模型的最新版本⁴。我们的微调实验结果是每个任务运行 4 次的平均值。

| 模型 | 参数量 | SQuAD2 | MNLI-m | SST-2 | QNLI | CoLA | RTE | MRPC | QQP | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT<sub>BASE</sub> | 109M | 76.8 | 84.5 | 93.2 | 91.7 | 58.9 | 68.6 | 87.3 | 91.3 | 81.5 |
| DistillBERT | 66M | 70.7 | 79.0 | 90.7 | 85.3 | 43.6 | 59.9 | 87.5 | 84.9 | 75.2 |
| TinyBERT | 66M | 73.1 | 83.5 | 91.6 | 90.5 | 42.8 | 72.2 | 88.4 | 90.6 | 79.1 |
| MiniLM | 66M | 76.4 | 84.0 | 92.0 | 91.0 | 49.2 | 71.5 | 88.4 | 91.0 | 80.4 |

³ DistilBERT 的公开模型取自 https://github.com/huggingface/transformers/tree/master/examples/distillation
⁴ 我们使用的是第 2 版 TinyBERT，取自 https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT

**抽取式问答（Extractive Question Answering）** 给定一段文章 $P$，任务是通过预测起止位置，从文章中选出一段连续文本来回答问题 $Q$。我们在 SQuAD 2.0（Rajpurkar et al., 2018）上评估，它一直是主要的问答基准之一。沿用 BERT（Devlin et al., 2018），我们把问题和文章的词元与特殊词元打包在一起，构成输入："[CLS] Q [SEP] P [SEP]"。我们引入两个线性输出层，分别预测每个词元作为答案片段起始位置和结束位置的概率。没有答案的问题被视为答案片段的起止位置都在 [CLS] 词元上。

**GLUE** 通用语言理解评测（General Language Understanding Evaluation，GLUE）基准（Wang et al., 2019）由九个句子级分类任务组成，包括语言可接受性语料库（Corpus of Linguistic Acceptability，CoLA）（Warstadt et al., 2018）、斯坦福情感树库（Stanford Sentiment Treebank，SST）（Socher et al., 2013）、微软研究院释义语料库（Microsoft Research Paraphrase Corpus，MRPC）（Dolan & Brockett, 2005）、语义文本相似度基准（Semantic Textual Similarity Benchmark，STS）（Cer et al., 2017）、Quora 问题对（Quora Question Pairs，QQP）（Chen et al., 2018）、多体裁自然语言推理（Multi-Genre Natural Language Inference，MNLI）（Williams et al., 2018）、问题自然语言推理（Question Natural Language Inference，QNLI）（Rajpurkar et al., 2016）、识别文本蕴含（Recognizing Textual Entailment，RTE）（Dagan et al., 2006; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009）以及 Winograd 自然语言推理（Winograd Natural Language Inference，WNLI）（Levesque et al., 2012）。我们在 [CLS] 词元之上加一个线性分类器来预测标签概率。

### 4.3. 主要结果（Main Results）

以往的工作（Sanh et al., 2019; Sun et al., 2019a; Jiao et al., 2019）通常把 BERT<sub>BASE</sub> 蒸馏成一个隐藏层大小为 768 的 6 层学生模型。我们首先使用相同的学生架构进行蒸馏实验。SQuAD 2.0 和 GLUE 开发集上的结果见表 2。由于 MobileBERT 是把一个专门设计的、带倒瓶颈模块且规模与 BERT<sub>LARGE</sub> 相同的教师，蒸馏成一个使用瓶颈模块的 24 层学生，我们没有把自己的模型与 MobileBERT 比较。MiniLM 在大多数任务上都优于 DistilBERT³ 和 TinyBERT⁴。我们的模型在 SQuAD 2.0 上比这两个最先进模型高出 3.0% 以上的 F1，在 CoLA 上高出 5.0% 以上的准确率。表 4 给出了不同参数规模模型的推理时间。我们的 6 层 768 维学生模型比原始 BERT<sub>BASE</sub> 快 2.0 倍，同时在 SQuAD 2.0 和 MNLI 等多种任务上保留了 99% 以上的性能。

**表 3：** 从 BERT<sub>BASE</sub> 蒸馏得到的不同架构学生模型之间的比较。$M$ 和 $d'_h$ 表示学生模型的层数和隐藏维度。TA 表示助教⁵。微调结果为 4 次运行的平均值。

| 架构 | 参数量 | 模型 | SQuAD 2.0 | MNLI-m | SST-2 | 平均 |
|---|---|---|---|---|---|---|
| $M=6;d'_h=384$ | 22M | MLM-KD（软标签蒸馏） | 67.9 | 79.6 | 89.8 | 79.1 |
| | | TinyBERT | 71.6 | 81.4 | 90.2 | 81.1 |
| | | MiniLM | 72.4 | 82.2 | 91.0 | 81.9 |
| | | MiniLM（w/ TA） | 72.7 | 82.4 | 91.2 | 82.1 |
| $M=4;d'_h=384$ | 19M | MLM-KD（软标签蒸馏） | 65.3 | 77.7 | 88.8 | 77.3 |
| | | TinyBERT | 66.7 | 79.2 | 88.5 | 78.1 |
| | | MiniLM | 69.4 | 80.3 | 90.2 | 80.0 |
| | | MiniLM（w/ TA） | 69.7 | 80.6 | 90.6 | 80.3 |
| $M=3;d'_h=384$ | 17M | MLM-KD（软标签蒸馏） | 59.9 | 75.2 | 88.0 | 74.4 |
| | | TinyBERT | 63.6 | 77.4 | 88.4 | 76.5 |
| | | MiniLM | 66.2 | 78.8 | 89.3 | 78.1 |
| | | MiniLM（w/ TA） | 66.9 | 79.1 | 89.7 | 78.6 |

⁵ 助教只在 MiniLM（w/ TA）模型中引入。各表中的 MiniLM 模型都是直接从其教师模型蒸馏得到的。

**表 4：** 不同模型的嵌入层（Emd）和 Transformer 层（Trm）参数量，以及推理时间。

| 层数 | 隐藏层大小 | 参数量（Emd） | 参数量（Trm） | 推理时间 |
|---|---|---|---|---|
| 12 | 768 | 23.4M | 85.1M | 93.1s（1.0×） |
| 6 | 768 | 23.4M | 42.5M | 46.9s（2.0×） |
| 12 | 384 | 11.7M | 21.3M | 34.8s（2.7×） |
| 6 | 384 | 11.7M | 10.6M | 17.7s（5.3×） |
| 4 | 384 | 11.7M | 7.1M | 12.0s（7.8×） |
| 3 | 384 | 11.7M | 5.3M | 9.2s（10.1×） |

我们还针对更小的学生模型进行了实验。我们把 MiniLM 与我们自己实现的 MLM-KD（使用掩码语言建模预测的软目标概率进行知识蒸馏）和 TinyBERT 进行比较，它们使用相同的数据和超参数训练。SQuAD 2.0、MNLI 和 SST-2 开发集上的结果见表 3。MiniLM 在这三个任务上都优于软标签蒸馏和我们实现的 TinyBERT。深度自注意力蒸馏对更小的模型同样有效。此外，我们表明，引入助教⁵ 对基于 Transformer 的预训练 LM 蒸馏也有帮助，对更小的模型尤其如此。把深度自注意力蒸馏与助教结合，能为更小的学生模型带来进一步的提升。

### 4.4. 消融研究（Ablation Studies）

我们在若干任务上进行消融测试，分析自注意力值关系迁移的贡献。SQuAD 2.0、MNLI 和 SST-2 的开发集结果见表 5：对于不同参数规模的学生模型，使用自注意力值关系迁移都对最终结果有正面贡献。蒸馏值关系这一细粒度知识，帮助学生模型深度模仿教师的自注意力行为，从而进一步提升模型性能。

**表 5：** 自注意力值关系（Value-Rel）迁移的有效性。微调结果为 4 次运行的平均值。

| 架构 | 模型 | SQuAD2 | MNLI-m | SST-2 |
|---|---|---|---|---|
| $M=6;d'_h=384$ | MiniLM | 72.4 | 82.2 | 91.0 |
| | -Value-Rel（去掉值关系） | 71.0 | 80.9 | 89.9 |
| $M=4;d'_h=384$ | MiniLM | 69.4 | 80.3 | 90.2 |
| | -Value-Rel | 67.5 | 79.0 | 89.2 |
| $M=3;d'_h=384$ | MiniLM | 66.2 | 78.8 | 89.3 |
| | -Value-Rel | 64.2 | 77.8 | 88.3 |

我们还比较了作用于自注意力模块中值上的不同损失函数。我们把所提出的值关系，与直接在教师和学生的值上计算均方误差（MSE）进行比较。如果学生的隐藏维度小于教师，就引入一个额外的参数矩阵来变换学生的值。三个任务上的开发集结果见表 6。使用值关系取得了更好的表现。具体来说，我们的方法在 SQuAD 基准上带来约 1.0% 的 F1 提升。而且，我们的方法不需要引入额外参数。我们也尝试过迁移隐藏状态之间的关系，但发现对于不同的教师模型，学生模型的表现不稳定。

**表 6：** 不同损失函数的比较：在值关系（值之间的缩放点积）上计算 KL 散度，与在值上计算均方误差（MSE）。后者引入一个参数矩阵，把学生的值变换为与教师的值相同的维度（Jiao et al., 2019）。微调结果为每个任务运行 4 次的平均值。

| 架构 | 模型 | SQuAD2 | MNLI-m | SST-2 |
|---|---|---|---|---|
| $M=6;d'_h=384$ | MiniLM | 72.4 | 82.2 | 91.0 |
| | Value-MSE | 71.4 | 82.0 | 90.8 |
| $M=4;d'_h=384$ | MiniLM | 69.4 | 80.3 | 90.2 |
| | Value-MSE | 68.3 | 80.1 | 89.9 |
| $M=3;d'_h=384$ | MiniLM | 66.2 | 78.8 | 89.3 |
| | Value-MSE | 65.5 | 78.4 | 89.3 |

为了证明蒸馏教师最后一个 Transformer 层的自注意力知识是有效的，我们把我们的方法与逐层蒸馏进行比较。我们迁移相同的知识，并采用 Jiao et al.（2019）中的均匀策略来映射教师层与学生层，进行逐层蒸馏。三个任务上的开发集结果见表 7。MiniLM 取得了更好的结果。它还缓解了教师与学生之间层映射的困难。此外，蒸馏教师最后一个 Transformer 层比逐层蒸馏需要的计算更少，训练速度更快。

**表 7：** 蒸馏教师最后一个 Transformer 层的知识与逐层蒸馏之间的比较。我们采用 Jiao et al.（2019）中的均匀策略来确定教师层与学生层之间的映射。微调结果为每个任务运行 4 次的平均值。

| 架构 | 模型 | SQuAD 2.0 | MNLI-m | SST-2 | 平均 |
|---|---|---|---|---|---|
| $M=6;d'_h=384$ | MiniLM | 72.4 | 82.2 | 91.0 | 81.9 |
| | +逐层蒸馏（Layer-to-Layer Distillation） | 71.6 | 81.8 | 90.6 | 81.3 |
| $M=4;d'_h=384$ | MiniLM | 69.4 | 80.3 | 90.2 | 80.0 |
| | +逐层蒸馏 | 67.6 | 79.9 | 89.6 | 79.0 |
| $M=3;d'_h=384$ | MiniLM | 66.2 | 78.8 | 89.3 | 78.1 |
| | +逐层蒸馏 | 64.8 | 77.7 | 88.6 | 77.0 |

---

## 5. 讨论（Discussion）

### 5.1. 更好的教师，更好的学生（Better Teacher Better Student）

我们报告了从一个内部预训练 Transformer 模型蒸馏得到的 MiniLM 的结果，这个教师模型遵循 UniLM（Dong et al., 2019; Bao et al., 2020），规模与 BERT<sub>BASE</sub> 相同。该教师模型使用与 RoBERTa<sub>BASE</sub>（Liu et al., 2019）类似的预训练数据集训练，包括来自英文维基百科、BookCorpus（Zhu et al., 2015）、OpenWebText⁶、CC-News（Liu et al., 2019）和 Stories（Trinh & Le, 2018）的 160GB 文本语料。我们使用相同的语料，把教师模型蒸馏成隐藏层大小为 384 的 12 层和 6 层模型。12x384 模型被用作助教来训练 6x384 模型。SQuAD 2.0 和 GLUE 基准的开发集结果见表 8，MiniLM 的结果显著提升。12x384 的 MiniLM 实现了 2.7 倍加速，同时在 SQuAD 2.0 和 GLUE 基准数据集上的表现比 BERT<sub>BASE</sub> 更有竞争力。

⁶ skylion007.github.io/OpenWebTextCorpus

**表 8：** 从内部预训练 Transformer 模型（BERT<sub>BASE</sub> 规模，12 层 Transformer，768 隐藏层大小，12 个自注意力头）蒸馏得到的 MiniLM 在 SQuAD 2.0 和 GLUE 基准上的结果。我们报告隐藏层大小为 384 的 12 层ᵃ和 6 层ᵇ模型。微调结果为 4 次运行的平均值。

| 模型 | 参数量 | SQuAD2 | MNLI-m | SST-2 | QNLI | CoLA | RTE | MRPC | QQP | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT<sub>BASE</sub> | 109M | 76.8 | 84.5 | 93.2 | 91.7 | 58.9 | 68.6 | 87.3 | 91.3 | 81.5 |
| MiniLMᵃ | 33M | 81.7 | 85.7 | 93.0 | 91.5 | 58.5 | 73.3 | 89.5 | 91.3 | 83.1 |
| MiniLMᵇ（w/ TA） | 22M | 75.6 | 83.3 | 91.5 | 90.5 | 47.5 | 68.8 | 88.9 | 90.6 | 79.6 |

### 5.2. 用于自然语言生成任务的 MiniLM（MiniLM for NLG Tasks）

我们还在自然语言生成（natural language generation，NLG）任务上评估 MiniLM，例如问题生成（question generation）和生成式摘要（abstractive summarization）。沿用 Dong et al.（2019），我们通过使用特定的自注意力掩码（self-attention mask），把 MiniLM 作为序列到序列（sequence-to-sequence）模型进行微调。

**问题生成（Question Generation）** 我们在答案感知的问题生成（answer-aware question generation）任务（Du & Cardie, 2018）上进行实验。给定一段输入文章和一个答案，任务是生成一个以该答案为答案的问题。评估使用 SQuAD 1.1 数据集（Rajpurkar et al., 2016）。MiniLM、UniLM<sub>LARGE</sub> 以及若干最先进模型的结果见表 9，我们蒸馏得到的 12x384 和 6x384 模型在问题生成任务上取得了有竞争力的表现。

**表 9：** 隐藏层大小为 384 的 12 层ᵃ和 6 层ᵇ模型在 SQuAD 1.1 上的问题生成结果。第一组采用 Du & Cardie（2018）中的数据划分，第二组与 Zhao et al.（2018）相同。MTR 是 METEOR 的缩写，RG 是 ROUGE 的缩写，B 是 BLEU 的缩写。

| 模型 | 参数量 | B-4 | MTR | RG-L |
|---|---|---|---|---|
| *数据划分同 Du & Cardie（2018）* | | | | |
| (Du & Cardie, 2018) | | 15.16 | 19.12 | - |
| (Zhang & Bansal, 2019) | | 18.37 | 22.65 | 46.68 |
| UniLM<sub>LARGE</sub> | 340M | 22.78 | 25.49 | 51.57 |
| MiniLMᵃ | 33M | 21.07 | 24.09 | 49.14 |
| MiniLMᵇ（w/ TA） | 22M | 20.31 | 23.43 | 48.21 |
| *数据划分同 Zhao et al.（2018）* | | | | |
| (Zhao et al., 2018) | | 16.38 | 20.25 | 44.48 |
| (Zhang & Bansal, 2019) | | 20.76 | 24.20 | 48.91 |
| UniLM<sub>LARGE</sub> | 340M | 24.32 | 26.10 | 52.69 |
| MiniLMᵃ | 33M | 23.27 | 25.15 | 50.60 |
| MiniLMᵇ（w/ TA） | 22M | 22.01 | 24.24 | 49.51 |

**生成式摘要（Abstractive Summarization）** 我们在两个生成式摘要数据集上评估 MiniLM，即 XSum（Narayan et al., 2018）和非匿名化版本的 CNN/DailyMail（See et al., 2017）。该生成任务要把一篇文档浓缩成一段简洁流畅的摘要，同时传达其关键信息。我们报告这些数据集上的 ROUGE 分数（Lin, 2004）。表 10 给出了 MiniLM、基线、若干最先进模型以及预训练 Transformer 模型的结果。我们的 12x384 模型以少得多的参数，超过了基于 BERT 的方法 BERTSumAbs（Liu & Lapata, 2019）和预训练序列到序列模型 MASS<sub>BASE</sub>（Song et al., 2019）。此外，我们的 6x384 MiniLM 也取得了有竞争力的表现。

**表 10：** 隐藏层大小为 384 的 12 层ᵃ和 6 层ᵇ模型在 CNN/DailyMail 和 XSum 上的生成式摘要结果。评价指标为 ROUGE（RG）分数的 F1 版本。

| 模型 | 参数量 | CNN/DM RG-1 | CNN/DM RG-2 | CNN/DM RG-L | XSum RG-1 | XSum RG-2 | XSum RG-L |
|---|---|---|---|---|---|---|---|
| Lead-3 | | 40.42 | 17.62 | 36.67 | 16.30 | 1.60 | 11.95 |
| PtrNet (See et al., 2017) | | 39.53 | 17.28 | 36.38 | 28.10 | 8.02 | 21.72 |
| Bottom-Up (Gehrmann et al., 2018) | | 41.22 | 18.68 | 38.34 | - | - | - |
| UniLM<sub>LARGE</sub> (Dong et al., 2019) | 340M | 43.08 | 20.43 | 40.34 | - | - | - |
| BART<sub>LARGE</sub> (Lewis et al., 2019a) | 400M | 44.16 | 21.28 | 40.90 | 45.14 | 22.27 | 37.25 |
| T5<sub>11B</sub> (Raffel et al., 2019) | 11B | 43.52 | 21.55 | 40.69 | - | - | - |
| MASS<sub>BASE</sub> (Song et al., 2019) | 123M | 42.12 | 19.50 | 39.01 | 39.75 | 17.24 | 31.95 |
| BERTSumAbs (Liu & Lapata, 2019) | 156M | 41.72 | 19.39 | 38.76 | 38.76 | 16.33 | 31.15 |
| T5<sub>BASE</sub> (Raffel et al., 2019) | 220M | 42.05 | 20.34 | 39.40 | - | - | - |
| MiniLMᵃ | 33M | 42.66 | 19.91 | 39.73 | 40.43 | 17.72 | 32.60 |
| MiniLMᵇ（w/ TA） | 22M | 41.57 | 19.21 | 38.64 | 38.79 | 16.39 | 31.10 |

> 译者注：Lead-3 是摘要任务的常用简单基线，直接取文章前三句作为摘要。

### 5.3. 多语言 MiniLM（Multilingual MiniLM）

我们在多语言预训练模型上进行任务无关知识蒸馏实验。我们使用 XLM-R<sub>Base</sub>⁷（Conneau et al., 2019）作为教师，并用相同的语料把它蒸馏成隐藏层大小为 384 的 12 层和 6 层模型。6x384 模型使用 12x384 模型作为助教来训练。由于多语言预训练模型的词表比单语言模型大得多（单语言 BERT 为 30k，XLM-R 为 250k），对多语言预训练模型做软标签蒸馏需要更多计算。MiniLM 只使用教师最后一个 Transformer 层的深度自注意力知识。对于多语言预训练模型，MiniLM 的训练速度比软标签蒸馏快得多。

⁷ 我们在蒸馏和微调实验中使用的是 v0 版本的 XLM-R<sub>Base</sub>。

> 译者注：软标签蒸馏需要在每个被遮盖的位置上计算整个词表的概率分布，词表越大计算越多；而 MiniLM 只比较 $|x| \times |x|$ 的注意力/值关系矩阵，与词表大小无关。

我们在跨语言自然语言推理（XNLI）基准（Conneau et al., 2018）和跨语言问答（MLQA）基准（Lewis et al., 2019b）上评估学生模型。

**表 11：** 隐藏层大小为 384 的 12 层ᵃ和 6 层ᵇ多语言模型在 XNLI 上的跨语言分类结果。我们报告 15 种 XNLI 语言各自的准确率以及平均准确率。mBERT、XLM-100 和 XLM-R<sub>Base</sub> 的结果来自 Conneau et al.（2019）。

| 模型 | 层数 | 隐藏维度 | en | fr | es | de | el | bg | ru | tr | ar | vi | th | zh | hi | sw | ur | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mBERT | 12 | 768 | 82.1 | 73.8 | 74.3 | 71.1 | 66.4 | 68.9 | 69.0 | 61.6 | 64.9 | 69.5 | 55.8 | 69.3 | 60.0 | 50.4 | 58.0 | 66.3 |
| XLM-100 | 16 | 1280 | 83.2 | 76.7 | 77.7 | 74.0 | 72.7 | 74.1 | 72.7 | 68.7 | 68.6 | 72.9 | 68.9 | 72.5 | 65.6 | 58.2 | 62.4 | 70.7 |
| XLM-R<sub>Base</sub> | 12 | 768 | 84.6 | 78.4 | 78.9 | 76.8 | 75.9 | 77.3 | 75.4 | 73.2 | 71.5 | 75.4 | 72.5 | 74.9 | 71.1 | 65.2 | 66.5 | 74.5 |
| MiniLMᵃ | 12 | 384 | 81.5 | 74.8 | 75.7 | 72.9 | 73.0 | 74.5 | 71.3 | 69.7 | 68.8 | 72.1 | 67.8 | 70.0 | 66.2 | 63.3 | 64.2 | 71.1 |
| MiniLMᵇ（w/ TA） | 6 | 384 | 79.2 | 72.3 | 73.1 | 70.3 | 69.1 | 72.0 | 69.1 | 64.5 | 64.9 | 69.0 | 66.0 | 67.8 | 62.9 | 59.0 | 60.6 | 68.0 |

**表 12：** 不同多语言预训练模型及我们的蒸馏模型的 Transformer 层（Trm）和嵌入层（Emd）参数量。

| 模型 | 层数 | 隐藏层大小 | 词表大小 | 参数量（Trm） | 参数量（Emd） |
|---|---|---|---|---|---|
| mBERT | 12 | 768 | 110k | 85M | 85M |
| XLM-15 | 12 | 1024 | 95k | 151M | 97M |
| XLM-100 | 16 | 1280 | 200k | 315M | 256M |
| XLM-R<sub>Base</sub> | 12 | 768 | 250k | 85M | 192M |
| MiniLMᵃ | 12 | 384 | 250k | 21M | 96M |
| MiniLMᵇ | 6 | 384 | 250k | 11M | 96M |

**XNLI** 表 11 给出了我们的蒸馏学生模型以及若干预训练 LM 在 XNLI 上的结果。沿用 Conneau et al.（2019），我们在所有语言的联合开发集上选出最佳单模型。表 12 列出了不同多语言预训练模型及我们的蒸馏模型的 Transformer 参数量和嵌入参数量。MiniLM 以少得多的 Transformer 参数在 XNLI 上取得了有竞争力的表现。此外，12x384 的 MiniLM 与使用 MLM 目标训练的 mBERT（Devlin et al., 2018）和 XLM（Lample & Conneau, 2019）相比表现更好。

**MLQA** 表 13 给出了跨语言问答结果。沿用 Lewis et al.（2019b），我们采用 SQuAD 1.1 作为训练数据，并使用 MLQA 英文开发数据进行早停（early stopping）。12x384 的 MiniLM 表现比 mBERT 和 XLM 更有竞争力。我们的 6 层 MiniLM 也取得了有竞争力的表现。

**表 13：** 隐藏层大小为 384 的 12 层ᵃ和 6 层ᵇ多语言模型在 MLQA 上的跨语言问答结果。我们报告 7 种 MLQA 语言各自的 F1 和 EM（精确匹配，exact match）分数（格式为 F1 / EM）。mBERT 和 XLM-15 的结果取自 Lewis et al.（2019b）。† 表示 XLM-R<sub>Base</sub> 的结果取自 Conneau et al.（2019）。我们还报告了自己微调 XLM-R<sub>Base</sub> 得到的结果（‡）。

| 模型 | 层数 | 隐藏维度 | en | es | de | ar | hi | vi | zh | 平均 |
|---|---|---|---|---|---|---|---|---|---|---|
| mBERT | 12 | 768 | 77.7 / 65.2 | 64.3 / 46.6 | 57.9 / 44.3 | 45.7 / 29.8 | 43.8 / 29.7 | 57.1 / 38.6 | 57.5 / 37.3 | 57.7 / 41.6 |
| XLM-15 | 12 | 1024 | 74.9 / 62.4 | 68.0 / 49.8 | 62.2 / 47.6 | 54.8 / 36.3 | 48.8 / 27.3 | 61.4 / 41.8 | 61.1 / 39.6 | 61.6 / 43.5 |
| XLM-R<sub>Base</sub>† | 12 | 768 | 77.8 / 65.3 | 67.2 / 49.7 | 60.8 / 47.1 | 53.0 / 34.7 | 57.9 / 41.7 | 63.1 / 43.1 | 60.2 / 38.0 | 62.9 / 45.7 |
| XLM-R<sub>Base</sub>‡ | 12 | 768 | 80.3 / 67.4 | 67.0 / 49.2 | 62.7 / 48.3 | 55.0 / 35.6 | 60.4 / 43.7 | 66.5 / 45.9 | 62.3 / 38.3 | 64.9 / 46.9 |
| MiniLMᵃ | 12 | 384 | 79.4 / 66.5 | 66.1 / 47.5 | 61.2 / 46.5 | 54.9 / 34.9 | 58.5 / 41.3 | 63.1 / 42.1 | 59.0 / 33.8 | 63.2 / 44.7 |
| MiniLMᵇ（w/ TA） | 6 | 384 | 75.5 / 61.9 | 55.6 / 38.2 | 53.3 / 37.7 | 43.5 / 26.2 | 46.9 / 31.5 | 52.0 / 33.1 | 48.8 / 27.3 | 53.7 / 36.6 |

---

## 6. 相关工作（Related Work）

### 6.1. 预训练语言模型（Pre-trained Language Models）

语言模型的无监督预训练（Peters et al., 2018; Howard & Ruder, 2018; Radford et al., 2018; Devlin et al., 2018; Baevski et al., 2019; Song et al., 2019; Dong et al., 2019; Yang et al., 2019; Joshi et al., 2019; Liu et al., 2019; Lewis et al., 2019a; Raffel et al., 2019）已在大量 NLP 任务上取得显著提升。早期的预训练方法（Peters et al., 2018; Radford et al., 2018）基于标准语言模型。最近，BERT（Devlin et al., 2018）提出用掩码语言建模目标训练一个深度双向 Transformer 编码器，学习左右上下文之间的交互。Liu et al.（2019）表明，在更多数据上训练更长时间可以取得非常强的表现。Joshi et al.（2019）通过遮盖连续的随机片段（span）来扩展 BERT。Yang et al.（2019）以打乱的顺序自回归地（auto-regressively）预测被遮盖的词元。

为了扩展预训练 Transformer 在 NLG 任务上的适用性，Dong et al.（2019）利用特定的自注意力掩码扩展 BERT，联合优化双向、单向和序列到序列的掩码语言建模目标。Raffel et al.（2019）采用编码器–解码器（encoder-decoder）Transformer，通过在编码器和解码器中预测被遮盖的词元来进行序列到序列预训练。与 Raffel et al.（2019）不同，Lewis et al.（2019a）在解码器中自回归地预测词元。

### 6.2. 知识蒸馏（Knowledge Distillation）

知识蒸馏已被证明是在保持准确率的同时压缩大模型的一种有前景的方法。它把一个大模型或一个神经网络集成（教师）的知识迁移到单个轻量模型（学生）中。Hinton et al.（2015）首先提出，用教师的软目标分布来训练蒸馏模型，从而把教师的知识迁移给学生。Romero et al.（2015）引入教师隐藏层的中间表示来指导学生训练。注意力图的知识（Zagoruyko & Komodakis, 2017; Hu et al., 2018）也被引入来帮助训练。

在本工作中，我们聚焦于基于 Transformer 的大型预训练语言模型的任务无关知识蒸馏。已有一些工作在下游任务上对微调后的语言模型进行特定任务蒸馏。Tang et al.（2019）把微调后的 BERT 蒸馏成一个极小的双向 LSTM。Turc et al.（2019a）在特定任务蒸馏中用一个小型预训练 LM 初始化学生。Sun et al.（2019a）引入教师每隔 $k$ 层的隐藏状态，进行逐层知识蒸馏。Aguilar et al.（2019）进一步引入自注意力分布的知识，并提出渐进式（progressive）和堆叠式（stacked）蒸馏方法。特定任务蒸馏需要先在下游任务上微调大型预训练 LM，再进行知识迁移。微调大型预训练 LM 的过程成本高、耗时长，对于大型数据集尤其如此。

对于任务无关蒸馏，蒸馏模型模仿原始的大型预训练 LM，可以直接在下游任务上微调。在实践中，预训练 LM 的任务无关压缩更受欢迎。MiniBERT（Tsai et al., 2019）使用掩码语言建模预测的软目标分布来指导多语言学生模型的训练，并证明了其在序列标注任务上的有效性。DistilBERT（Sanh et al., 2019）使用教师的软标签和嵌入输出来训练学生。TinyBERT（Jiao et al., 2019）和 MobileBERT（Sun et al., 2019b）进一步引入自注意力分布和隐藏状态来训练学生。MobileBERT 为教师和学生分别采用倒瓶颈和瓶颈模块，使二者的隐藏维度相同。为了进行逐层蒸馏，MobileBERT 的学生模型必须与其教师层数相同。此外，MobileBERT 提出一种自底向上的渐进方案来迁移教师的知识。当教师和学生层数不同时，TinyBERT 使用均匀策略来映射二者的层，并引入一个线性矩阵，把学生的隐藏状态变换成与教师相同的维度。TinyBERT 还为下游任务引入了特定任务蒸馏和数据增强，带来了进一步的提升。

与以往工作不同，我们的方法利用教师最后一个 Transformer 层的自注意力分布和值关系，帮助学生深度模仿教师的自注意力行为。使用最后一个 Transformer 层的知识而不是逐层蒸馏，避免了对学生层数的限制，也省去了寻找最佳层映射的工作。蒸馏自注意力值之间的关系，使学生的隐藏层大小更加灵活，并避免引入线性矩阵来变换学生的表示。

---

## 7. 结论（Conclusion）

在本工作中，我们提出一种简单而有效的知识蒸馏方法，用于压缩基于 Transformer 的大型预训练语言模型。学生通过深度模仿教师的自注意力模块来训练，而自注意力模块是 Transformer 网络的关键组件。我们提出使用教师最后一个 Transformer 层的自注意力分布和值关系来指导学生训练，这对学生模型而言既有效又灵活。此外，我们表明，引入助教也有助于基于 Transformer 的预训练 LM 蒸馏，而所提出的深度自注意力蒸馏能进一步提升性能。我们从 BERT<sub>BASE</sub> 蒸馏得到的学生模型在 SQuAD 2.0 和 GLUE 基准任务上保持了高准确率，并优于最先进的基线。深度自注意力蒸馏也可以用来压缩更大规模的预训练模型，我们将此留作未来工作。

> 译者注：原文没有单独的致谢（Acknowledgments）部分。

---

## 参考文献（References）

Aguilar, G., Ling, Y., Zhang, Y., Yao, B., Fan, X., and Guo, E. Knowledge distillation from internal representations. CoRR, abs/1910.03723, 2019. URL http://arxiv.org/abs/1910.03723.

Ba, L. J., Kiros, J. R., and Hinton, G. E. Layer normalization. CoRR, abs/1607.06450, 2016. URL http://arxiv.org/abs/1607.06450.

Baevski, A., Edunov, S., Liu, Y., Zettlemoyer, L., and Auli, M. Cloze-driven pretraining of self-attention networks. arXiv preprint arXiv:1903.07785, 2019.

Bahdanau, D., Cho, K., and Bengio, Y. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.

Bao, H., Dong, L., Wei, F., Wang, W., Yang, N., Liu, X., Wang, Y., Piao, S., Gao, J., Zhou, M., and Hon, H.-W. Unilmv2: Pseudo-masked language models for unified language model pre-training. arXiv preprint arXiv:2002.12804, 2020.

Bar-Haim, R., Dagan, I., Dolan, B., Ferro, L., and Giampiccolo, D. The second PASCAL recognising textual entailment challenge. In Proceedings of the Second PASCAL Challenges Workshop on Recognising Textual Entailment, 01 2006.

Bentivogli, L., Dagan, I., Dang, H. T., Giampiccolo, D., and Magnini, B. The fifth pascal recognizing textual entailment challenge. In In Proc Text Analysis Conference (TAC'09, 2009.

Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., and Specia, L. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055, 2017.

Chen, Z., Zhang, H., Zhang, X., and Zhao, L. Quora question pairs. 2018.

Clark, K., Khandelwal, U., Levy, O., and Manning, C. D. What does BERT look at? an analysis of bert's attention. CoRR, abs/1906.04341, 2019. URL http://arxiv.org/abs/1906.04341.

Conneau, A., Lample, G., Rinott, R., Williams, A., Bowman, S. R., Schwenk, H., and Stoyanov, V. Xnli: Evaluating cross-lingual sentence representations. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., and Stoyanov, V. Unsupervised cross-lingual representation learning at scale. CoRR, abs/1911.02116, 2019. URL http://arxiv.org/abs/1911.02116.

Dagan, I., Glickman, O., and Magnini, B. The pascal recognising textual entailment challenge. In Proceedings of the First International Conference on Machine Learning Challenges: Evaluating Predictive Uncertainty Visual Object Classification, and Recognizing Textual Entailment, MLCW'05, pp. 177–190, Berlin, Heidelberg, 2006. Springer-Verlag. ISBN 3-540-33427-0, 978-3-540-33427-9. doi: 10.1007/11736790_9. URL http://dx.doi.org/10.1007/11736790_9.

Devlin, J., Chang, M., Lee, K., and Toutanova, K. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805, 2018.

Dolan, W. B. and Brockett, C. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005), 2005.

Dong, L., Yang, N., Wang, W., Wei, F., Liu, X., Wang, Y., Gao, J., Zhou, M., and Hon, H.-W. Unified language model pre-training for natural language understanding and generation. In 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), 2019.

Du, X. and Cardie, C. Harvesting paragraph-level question-answer pairs from wikipedia. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pp. 1907–1917, 2018.

Gehrmann, S., Deng, Y., and Rush, A. Bottom-up abstractive summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 4098–4109, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/D18-1443.

Giampiccolo, D., Magnini, B., Dagan, I., and Dolan, B. The third PASCAL recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing, pp. 1–9, Prague, June 2007. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/W07-1401.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pp. 770–778, 2016.

Hinton, G. E., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. CoRR, abs/1503.02531, 2015. URL http://arxiv.org/abs/1503.02531.

Howard, J. and Ruder, S. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 328–339, Melbourne, Australia, July 2018. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/P18-1031.

Hu, M., Peng, Y., Wei, F., Huang, Z., Li, D., Yang, N., and Zhou, M. Attention-guided answer distillation for machine reading comprehension. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pp. 2077–2086, 2018. URL https://www.aclweb.org/anthology/D18-1232/.

Jawahar, G., Sagot, B., and Seddah, D. What does BERT learn about the structure of language? In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 3651–3657, 2019. URL https://www.aclweb.org/anthology/P19-1356/.

Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., and Liu, Q. Tinybert: Distilling BERT for natural language understanding. CoRR, abs/1909.10351, 2019. URL http://arxiv.org/abs/1909.10351.

Joshi, M., Chen, D., Liu, Y., Weld, D. S., Zettlemoyer, L., and Levy, O. Spanbert: Improving pre-training by representing and predicting spans. arXiv preprint arXiv:1907.10529, 2019.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, San Diego, CA, 2015. URL http://arxiv.org/abs/1412.6980.

Lample, G. and Conneau, A. Cross-lingual language model pretraining. Advances in Neural Information Processing Systems (NeurIPS), 2019.

Levesque, H., Davis, E., and Morgenstern, L. The winograd schema challenge. In Thirteenth International Conference on the Principles of Knowledge Representation and Reasoning, 2012.

Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., and Zettlemoyer, L. BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019a.

Lewis, P. S. H., Oguz, B., Rinott, R., Riedel, S., and Schwenk, H. MLQA: evaluating cross-lingual extractive question answering. CoRR, abs/1910.07475, 2019b. URL http://arxiv.org/abs/1910.07475.

Lin, C.-Y. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out: Proceedings of the ACL-04 Workshop, pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/W04-1013.

Liu, Y. and Lapata, M. Text summarization with pretrained encoders. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pp. 3730–3740, Hong Kong, China, 2019.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Mirzadeh, S., Farajtabar, M., Li, A., and Ghasemzadeh, H. Improved knowledge distillation via teacher assistant: Bridging the gap between student and teacher. CoRR, abs/1902.03393, 2019. URL http://arxiv.org/abs/1902.03393.

Narayan, S., Cohen, S. B., and Lapata, M. Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1797–1807, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1206. URL https://www.aclweb.org/anthology/D18-1206.

Peters, M., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., and Zettlemoyer, L. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 2227–2237, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/N18-1202.

Radford, A., Narasimhan, K., Salimans, T., and Sutskever, I. Improving language understanding by generative pre-training. 2018. URL https://s3-us-west-2.amazonaws.com/openaiassets/research-covers/language-unsupervised/languageunderstandingpaper.pdf.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683, 2019.

Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1264. URL https://www.aclweb.org/anthology/D16-1264.

Rajpurkar, P., Jia, R., and Liang, P. Know what you don't know: Unanswerable questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 2: Short Papers, pp. 784–789, 2018.

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. Fitnets: Hints for thin deep nets. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015. URL http://arxiv.org/abs/1412.6550.

Sanh, V., Debut, L., Chaumond, J., and Wolf, T. Distilbert, a distilled version of BERT: smaller, faster, cheaper and lighter. CoRR, abs/1910.01108, 2019. URL http://arxiv.org/abs/1910.01108.

See, A., Liu, P. J., and Manning, C. D. Get to the point: Summarization with pointer-generator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1073–1083, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1099. URL https://www.aclweb.org/anthology/P17-1099.

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pp. 1631–1642, 2013.

Song, K., Tan, X., Qin, T., Lu, J., and Liu, T.-Y. Mass: Masked sequence to sequence pre-training for language generation. arXiv preprint arXiv:1905.02450, 2019.

Sun, S., Cheng, Y., Gan, Z., and Liu, J. Patient knowledge distillation for BERT model compression. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pp. 4322–4331, 2019a.

Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y., and Zhou, D. Mobilebert: Task-agnostic compression of bert by progressive knowledge transfer, 2019b. URL https://openreview.net/pdf?id=SJxjVaNKwB.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2818–2826, 2016. doi: 10.1109/cvpr.2016.308.

Tang, R., Lu, Y., Liu, L., Mou, L., Vechtomova, O., and Lin, J. Distilling task-specific knowledge from BERT into simple neural networks. CoRR, abs/1903.12136, 2019. URL http://arxiv.org/abs/1903.12136.

Trinh, T. H. and Le, Q. V. A simple method for commonsense reasoning. ArXiv, abs/1806.02847, 2018.

Tsai, H., Riesa, J., Johnson, M., Arivazhagan, N., Li, X., and Archer, A. Small and practical BERT models for sequence labeling. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pp. 3630–3634, 2019.

Turc, I., Chang, M., Lee, K., and Toutanova, K. Well-read students learn better: The impact of student initialization on knowledge distillation. CoRR, abs/1908.08962, 2019a. URL http://arxiv.org/abs/1908.08962.

Turc, I., Chang, M., Lee, K., and Toutanova, K. Well-read students learn better: The impact of student initialization on knowledge distillation. CoRR, abs/1908.08962, 2019b. URL http://arxiv.org/abs/1908.08962.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems 30, pp. 5998–6008. Curran Associates, Inc., 2017. URL http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf.

Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJ4km2R5t7.

Warstadt, A., Singh, A., and Bowman, S. R. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471, 2018.

Williams, A., Nangia, N., and Bowman, S. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 1112–1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1101. URL https://www.aclweb.org/anthology/N18-1101.

Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., Krikun, M., Cao, Y., Gao, Q., Macherey, K., Klingner, J., Shah, A., Johnson, M., Liu, X., Kaiser, L., Gouws, S., Kato, Y., Kudo, T., Kazawa, H., Stevens, K., Kurian, G., Patil, N., Wang, W., Young, C., Smith, J., Riesa, J., Rudnick, A., Vinyals, O., Corrado, G., Hughes, M., and Dean, J. Google's neural machine translation system: Bridging the gap between human and machine translation. CoRR, abs/1609.08144, 2016. URL http://arxiv.org/abs/1609.08144.

Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., and Le, Q. V. XLNet: Generalized autoregressive pretraining for language understanding. In 33rd Conference on Neural Information Processing Systems (NeurIPS 2019), 2019.

Zagoruyko, S. and Komodakis, N. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings, 2017. URL https://openreview.net/forum?id=Sks9_ajex.

Zhang, S. and Bansal, M. Addressing semantic drift in question generation for semi-supervised question answering. CoRR, abs/1909.06356, 2019.

Zhao, Y., Ni, X., Ding, Y., and Ke, Q. Paragraph-level neural question generation with maxout pointer and gated self-attention networks. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 3901–3910, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/D18-1424.

Zhu, Y., Kiros, R., Zemel, R., Salakhutdinov, R., Urtasun, R., Torralba, A., and Fidler, S. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pp. 19–27, 2015.

---

## 附录（Appendix）

### A. GLUE 基准（GLUE Benchmark）

通用语言理解评测（GLUE）基准⁸（Wang et al., 2019）所用数据集的概况见表 14。

SQuAD 2.0⁹（Rajpurkar et al., 2018）的数据集统计和评价指标见表 15。

⁸ https://gluebenchmark.com/
⁹ http://stanford-qa.com

**表 14：** GLUE 基准概况。

| 语料 | 训练集 | 开发集 | 测试集 | 评价指标 |
|---|---|---|---|---|
| *单句任务（Single-Sentence Tasks）* | | | | |
| CoLA | 8.5k | 1k | 1k | 马修斯相关系数（Matthews Corr） |
| SST-2 | 67k | 872 | 1.8k | 准确率（Accuracy） |
| *相似度与释义任务（Similarity and Paraphrase Tasks）* | | | | |
| QQP | 364k | 40k | 391k | 准确率/F1 |
| MRPC | 3.7k | 408 | 1.7k | 准确率/F1 |
| STS-B | 7k | 1.5k | 1.4k | 皮尔逊/斯皮尔曼相关系数（Pearson/Spearman Corr） |
| *推理任务（Inference Tasks）* | | | | |
| MNLI | 393k | 20k | 20k | 准确率 |
| RTE | 2.5k | 276 | 3k | 准确率 |
| QNLI | 105k | 5.5k | 5.5k | 准确率 |
| WNLI | 634 | 71 | 146 | 准确率 |

**表 15：** SQuAD 2.0 的数据集统计和评价指标。

| 训练集 | 开发集 | 测试集 | 评价指标 |
|---|---|---|---|
| 130,319 | 11,873 | 8,862 | 精确匹配/F1（Exact Match/F1） |

### B. 微调超参数（Fine-tuning Hyper-parameters）

**抽取式问答（Extractive Question Answering）** 对于 SQuAD 2.0，最大序列长度为 384；如果长度超过 384，则使用大小为 128 的滑动窗口（sliding window）。对于从我们内部预训练模型蒸馏得到的 12 层模型，我们以 48 的批大小和 4e-5 的峰值学习率微调 3 个 epoch。其余蒸馏模型以 32 的批大小和 6e-5 的峰值学习率训练 3 个 epoch。

**GLUE** GLUE 基准的最大序列长度为 128。对于从 BERT<sub>BASE</sub> 蒸馏得到的学生模型，我们把批大小设为 32，学习率从 {2e-5, 3e-5, 4e-5, 5e-5} 中选取，epoch 数从 {3, 4, 5} 中选取。对于从我们内部预训练模型蒸馏得到的学生模型，批大小从 {32, 48} 中选取。我们对若干任务（CoLA、RTE 和 MRPC）用更多 epoch 微调（最多 10 个 epoch），这带来了小幅提升。对于 12 层模型，CoLA、RTE 和 MRPC 任务使用的学习率为 1.5e-5。

### C. SQuAD 2.0（原文标题如此）

> 译者注：提取的原文中，附录 C 的标题为「C. SQuAD 2.0」，但其内容是问题生成、生成式摘要和跨语言任务的微调超参数，与标题不符。译者用 PyMuPDF 核对过，PDF 中该标题确实为粗体「C. SQuAD 2.0」，并非提取错误，应是原文排版疏漏；此处照原文保留。

**问题生成（Question Generation）** 对于问题生成任务，我们把批大小设为 32，总长度设为 512。最大输出长度为 48。12 层和 6 层模型的学习率分别为 3e-5 和 8e-5。两者都微调 25 个 epoch。我们还使用了比率为 0.1 的标签平滑（label smoothing）（Szegedy et al., 2016）。解码时，我们使用束大小（beam size）为 5 的束搜索（beam search）。长度惩罚（length penalty）（Wu et al., 2016）为 1.3。

**生成式摘要（Abstractive Summarization）** 对于生成式摘要任务，我们把批大小设为 64，标签平滑比率设为 0.1。对于 CNN/DailyMail 数据集，总长度为 768，最大输出长度为 160。12 层和 6 层模型的学习率分别为 1e-4 和 1.5e-4。两者都微调 25 个 epoch。解码时，我们把束大小设为 5，长度惩罚设为 0.7。对于 XSum 数据集，总长度为 512，最大输出长度为 48。12 层和 6 层模型的学习率分别为 1e-4 和 1.5e-4。12 层模型微调 30 个 epoch，6 层模型微调 50 个 epoch。解码时，我们使用束大小为 5 的束搜索。长度惩罚设为 0.9。

**跨语言自然语言推理（Cross-lingual Natural Language Inference）** XNLI 的最大序列长度为 128。我们以 128 的批大小微调 5 个 epoch，学习率从 {3e-5, 4e-5, 5e-5, 6e-5} 中选取。

**跨语言问答（Cross-lingual Question Answering）** 对于 MLQA，最大序列长度为 512；如果长度超过 512，则使用大小为 128 的滑动窗口。我们以 32 的批大小微调 3 个 epoch。学习率从 {3e-5, 4e-5, 5e-5, 6e-5} 中选取。
