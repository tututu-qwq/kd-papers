> 出处说明：本文是论文《DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter》（Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf, 2019）的中文译本，原文链接：https://arxiv.org/abs/1910.01108 （译自 arXiv v4，2020 年 3 月 1 日版）

# DistilBERT：BERT 的蒸馏版本——更小、更快、更便宜、更轻量

**原题：** DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

Victor SANH, Lysandre DEBUT, Julien CHAUMOND, Thomas WOLF
Hugging Face
{victor,lysandre,julien,thomas}@huggingface.co

发表于 EMC²：第 5 届研讨会，与 NeurIPS'19 同期举办（EMC^2: 5th Edition Co-located with NeurIPS'19）

> 译者注：文中「译者注」均为译者为初学者添加，原文没有。公式用 LaTeX 写出；参考文献保留英文原文。

---

## 摘要（Abstract）

随着基于大规模预训练模型的迁移学习（Transfer Learning）在自然语言处理（Natural Language Processing，NLP）中越来越普遍，在边缘设备（on-the-edge）上运行这些大模型，或在训练、推理（inference）计算预算受限的条件下使用它们，仍然很有挑战。在这项工作中，我们提出一种方法，预训练一个更小的通用语言表示模型，称为 DistilBERT；它可以像更大的同类模型一样，经过微调（fine-tune）后在大量任务上取得良好表现。以往大多数工作研究的是用蒸馏来构建特定任务（task-specific）的模型，而我们在预训练（pre-training）阶段就利用知识蒸馏（knowledge distillation），并证明可以把 BERT 模型的规模缩小 40%，同时保留其 97% 的语言理解能力，速度还快 60%。为了利用大模型在预训练中学到的归纳偏置（inductive biases），我们引入一个三重损失（triple loss），它结合了语言建模（language modeling）损失、蒸馏损失和余弦距离（cosine-distance）损失。我们这个更小、更快、更轻的模型预训练成本更低；我们还通过一个概念验证实验和一项设备端对比研究，展示了它在设备端（on-device）计算中的能力。

---

## 1 引言（Introduction）

过去两年里，迁移学习方法在 NLP 中兴起，大规模预训练语言模型已经成为许多 NLP 任务的基本工具 [Devlin et al., 2018, Radford et al., 2019, Liu et al., 2019]。这些模型带来了显著提升，但它们往往有数亿个参数，而且目前关于预训练模型的研究¹ 表明，训练更大的模型仍然能在下游任务（downstream tasks）上取得更好的表现。

> **图 1**（未重绘）：最近发布的若干预训练语言模型的参数量（Parameter counts of several recently released pretrained language models）。

模型越来越大的趋势引发了几点担忧。第一是环境成本：正如 Schwartz et al. [2019] 和 Strubell et al. [2019] 所指出的，这些模型的计算需求呈指数级增长。第二，虽然在设备端实时运行这些模型有可能催生新颖有趣的语言处理应用，但它们日益增长的计算和内存需求可能会阻碍其广泛应用。

¹ 例如最近发布的 MegatronLM（https://nv-adlr.github.io/MegatronLM）。

在本文中，我们证明：用知识蒸馏预训练的小得多的语言模型，可以在许多下游任务上达到相近的表现。这样得到的模型在推理时更轻、更快，训练所需的计算预算也更小。我们的通用预训练模型经过微调后可以在多个下游任务上表现良好，保留了大模型的灵活性。我们还表明，压缩后的模型足够小，可以在边缘设备上运行，例如移动设备。

我们利用一个三重损失证明：一个小 40% 的 Transformer（Vaswani et al. [2017]），在一个更大的 Transformer 语言模型的监督下通过蒸馏进行预训练，可以在多种下游任务上取得相近的表现，同时推理速度快 60%。进一步的消融研究（ablation studies）表明，三重损失中的每个组成部分对取得最佳表现都很重要。我们已在 HuggingFace 的 Transformers² 库 [Wolf et al., 2019] 中公开了训练好的权重和训练代码。

² https://github.com/huggingface/transformers

---

## 2 知识蒸馏（Knowledge distillation）

知识蒸馏 [Bucila et al., 2006, Hinton et al., 2015] 是一种压缩技术：训练一个紧凑的模型（学生，student）去复现一个更大的模型（教师，teacher）或一个模型集成（ensemble）的行为。

在监督学习中，分类模型一般通过最大化金标准标签（gold labels）的估计概率来学习预测样本的类别。因此，标准的训练目标是最小化模型预测分布与训练标签的独热（one-hot）经验分布之间的交叉熵（cross-entropy）。一个在训练集上表现良好的模型，其输出分布会在正确类别上给出高概率，在其他类别上给出接近零的概率。但这些「接近零」的概率中，有些比另一些大，它们在一定程度上反映了模型的泛化能力，以及它在测试集上会表现得多好³。

³ 例如，对于句子 "I think this is the beginning of a beautiful [MASK]" 中被遮盖的词，BERT-base 的预测包含两个高概率的词（day 和 life），以及一条由合理预测组成的长尾（future、story、world……）。

**训练损失（Training loss）** 学生模型使用基于教师软目标概率（soft target probabilities）的蒸馏损失进行训练：

$$
L_{ce} = \sum_i t_i \ast \log(s_i)
$$

其中 $t_i$（以及 $s_i$）分别是教师（以及学生）估计的概率。这个目标利用了教师的完整分布，因而提供了丰富的训练信号。沿用 Hinton et al. [2015] 的做法，我们使用了带温度的 softmax（softmax-temperature）：

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $T$ 控制输出分布的平滑程度，$z_i$ 是模型对类别 $i$ 的打分。训练时学生和教师使用相同的温度 $T$，推理时把 $T$ 设为 1，恢复成标准 softmax。

> 译者注：原文的 $L_{ce}$ 公式没有写负号。作为要最小化的交叉熵，通常写作 $L_{ce} = -\sum_i t_i \log(s_i)$。这里按原文照录。

最终的训练目标是蒸馏损失 $L_{ce}$ 与监督训练损失的线性组合，在我们这里，监督训练损失就是掩码语言建模损失（masked language modeling loss）$L_{mlm}$ [Devlin et al., 2018]。我们发现，再加上一个余弦嵌入损失（cosine embedding loss，$L_{cos}$）是有益的，它会使学生和教师隐藏状态（hidden states）向量的方向趋于一致。

> 译者注：因此三重损失大致是 $L = \alpha L_{ce} + \beta L_{mlm} + \gamma L_{cos}$。原文正文没有给出各项权重。

---

## 3 DistilBERT：BERT 的蒸馏版本（DistilBERT: a distilled version of BERT）

**学生架构（Student architecture）** 在本工作中，学生模型 DistilBERT 的总体架构与 BERT 相同。我们去掉了词元类型嵌入（token-type embeddings）和池化层（pooler），并把层数减半。Transformer 架构中用到的大多数运算（线性层和层归一化，layer normalisation）在现代线性代数框架中都已高度优化。我们的研究表明，在参数预算固定的情况下，改变张量最后一维（隐藏层维度，hidden size）对计算效率的影响，比改变层数等其他因素要小。因此我们集中于减少层数。

**学生初始化（Student initialization）** 除了上述优化和架构上的选择之外，我们训练流程中的一个重要环节，是为子网络找到合适的初始化，使其能够收敛。利用教师网络和学生网络维度相同这一点，我们从教师中每两层取一层，用来初始化学生。

**蒸馏（Distillation）** 我们采用了 Liu et al. [2019] 最近提出的 BERT 训练最佳实践。具体来说，DistilBERT 借助梯度累积（gradient accumulation），在非常大的批次（每批最多 4K 个样本）上进行蒸馏，使用动态掩码（dynamic masking），并且不使用下一句预测（next sentence prediction）目标。

**数据和算力（Data and compute power）** 我们在与原始 BERT 模型相同的语料上训练 DistilBERT：英文维基百科（English Wikipedia）与多伦多图书语料（Toronto Book Corpus）[Zhu et al., 2015] 的拼接。DistilBERT 在 8 块 16GB 的 V100 GPU 上训练了大约 90 小时。作为对比，RoBERTa 模型 [Liu et al., 2019] 需要在 1024 块 32GB V100 上训练 1 天。

**表 1：DistilBERT 保留了 BERT 97% 的性能。** 在 GLUE 基准开发集上的比较。ELMo 的结果引自其作者的报告。BERT 和 DistilBERT 的结果是用不同随机种子运行 5 次的中位数。

| 模型 | 总分（Score） | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | STS-B | WNLI |
|---|---|---|---|---|---|---|---|---|---|---|
| ELMo | 68.7 | 44.1 | 68.6 | 76.6 | 71.1 | 86.2 | 53.4 | 91.5 | 70.4 | 56.3 |
| BERT-base | 79.5 | 56.3 | 86.7 | 88.6 | 91.8 | 89.6 | 69.3 | 92.7 | 89.0 | 53.5 |
| DistilBERT | 77.0 | 51.3 | 82.2 | 87.5 | 89.2 | 88.5 | 59.9 | 91.3 | 86.9 | 56.3 |

**表 2：DistilBERT 在下游任务上取得相当的表现。** 下游任务上的比较：IMDb（测试准确率）和 SQuAD 1.1（开发集上的 EM/F1）。D：在微调阶段再进行一步蒸馏。

| 模型 | IMDb（准确率） | SQuAD（EM/F1） |
|---|---|---|
| BERT-base | 93.46 | 81.2/88.5 |
| DistilBERT | 92.82 | 77.7/85.8 |
| DistilBERT (D) | - | 79.1/86.9 |

**表 3：DistilBERT 明显更小，并且始终更快。** 在 CPU 上、批大小为 1 时，完整跑一遍 GLUE 任务 STS-B（原文括注为情感分析）所需的推理时间。

| 模型 | 参数量（百万） | 推理时间（秒） |
|---|---|---|
| ELMo | 180 | 895 |
| BERT-base | 110 | 668 |
| DistilBERT | 66 | 410 |

> 译者注：STS-B 实际上是语义文本相似度（Semantic Textual Similarity）任务，并非情感分析；表 3 标题中的「sentiment analysis」是原文的说法，这里照译。

---

## 4 实验（Experiments）

**通用语言理解（General Language Understanding）** 我们在通用语言理解评测（General Language Understanding Evaluation，GLUE）基准 [Wang et al., 2018] 上评估 DistilBERT 的语言理解和泛化能力。GLUE 由 9 个用于评估自然语言理解系统的数据集组成。我们报告每个任务开发集（development set）上的分数，微调 DistilBERT 时不使用集成或多任务微调方案（这些与本工作基本正交）。我们把结果与 GLUE 作者提供的基线（baseline）比较：一个 ELMo（Peters et al. [2018]）编码器后接两个 BiLSTM⁴。

⁴ 我们使用 jiant [Wang et al., 2019] 计算基线。

9 个任务各自的结果以及宏平均分（macro-score，即各项分数的平均）列在表 1 中。在这 9 个任务中，DistilBERT 始终与 ELMo 基线持平或更好（在 STS-B 上准确率最多高出 19 个点）。DistilBERT 与 BERT 相比也表现得出奇地好，在参数少 40% 的情况下保留了 97% 的性能。

### 4.1 下游任务基准（Downstream task benchmark）

**下游任务（Downstream tasks）** 我们进一步研究 DistilBERT 在高效推理约束下、在若干下游任务上的表现：一个分类任务（IMDb 情感分类，Maas et al. [2011]）和一个问答任务（SQuAD v1.1，Rajpurkar et al. [2016]）。

如表 2 所示，DistilBERT 在 IMDb 基准上的测试准确率只比 BERT 低 0.6 个百分点，而模型小 40%。在 SQuAD 上，DistilBERT 与完整 BERT 的差距在 3.9 个点以内。

我们还研究了能否在适配阶段（adaptation phase）再增加一步蒸馏：在 SQuAD 上微调 DistilBERT 时，以一个已在 SQuAD 上微调过的 BERT 模型作为教师，为损失增加一项（知识蒸馏）。在这种设置下，就有两个连续的蒸馏步骤：一个在预训练阶段，一个在适配阶段。在这种情况下，考虑到模型的规模，我们取得了不错的表现：79.8 F1 和 70.4 EM，即与完整模型相差在 3 个点以内。

> 译者注：这里正文给出的 79.8 F1 / 70.4 EM，与表 2 中 DistilBERT (D) 的 79.1/86.9（EM/F1）不一致，原文如此，译者无法判断哪一处为准。

**模型大小与推理速度（Size and inference speed）**

为了进一步考察 DistilBERT 在加速与模型大小之间的权衡，我们在表 3 中比较了每个模型的参数量，以及在 CPU（Intel Xeon E5-2690 v3 Haswell @2.9GHz）上、批大小为 1 时完整跑一遍 STS-B 开发集所需的推理时间。DistilBERT 的参数比 BERT 少 40%，速度比 BERT 快 60%。

**设备端计算（On device computation）** 我们通过构建一个问答移动应用，研究 DistilBERT 能否用于边缘应用。我们在一部较新的智能手机（iPhone 7 Plus）上，比较它与我们之前训练的基于 BERT-base 的问答模型的平均推理时间。不计分词（tokenization）步骤，DistilBERT 比 BERT 快 71%，整个模型大小为 207 MB（还可以通过量化（quantization）进一步减小）。我们的代码已公开⁵。

⁵ https://github.com/huggingface/swift-coreml-transformers

### 4.2 消融研究（Ablation study）

本节我们研究三重损失的各个组成部分以及学生初始化对蒸馏模型性能的影响。我们报告 GLUE 上的宏平均分。表 4 给出了相对完整三重损失的变化量：去掉掩码语言建模损失影响很小，而两个蒸馏损失贡献了性能的很大一部分。

**表 4：消融研究。** 变化量相对于使用三重损失和教师权重初始化训练的模型。

| 消融设置 | GLUE 宏平均分的变化 |
|---|---|
| ∅ - $L_{cos}$ - $L_{mlm}$（去掉 $L_{ce}$） | -2.96 |
| $L_{ce}$ - ∅ - $L_{mlm}$（去掉 $L_{cos}$） | -1.46 |
| $L_{ce}$ - $L_{cos}$ - ∅（去掉 $L_{mlm}$） | -0.31 |
| 三重损失 + 随机权重初始化 | -3.69 |

> 译者注：表中「∅」表示该位置的损失被去掉，括号里的说明是译者加的。

---

## 5 相关工作（Related work）

**特定任务蒸馏（Task-specific distillation）** 以往的大多数工作集中于构建特定任务的蒸馏设置。Tang et al. [2019] 把微调过的分类模型 BERT 迁移到一个基于 LSTM 的分类器上。Chatterjee [2019] 把在 SQuAD 上微调过的 BERT 模型蒸馏到一个先前从 BERT 初始化的更小的 Transformer 模型中。在本工作中，我们发现使用通用的预训练蒸馏比特定任务的蒸馏更有益。Turc et al. [2019] 使用原始的预训练目标训练较小的学生模型，再通过蒸馏进行微调。如消融研究所示，我们发现利用教师的知识、以额外的蒸馏信号进行预训练是有益的。

**多重蒸馏（Multi-distillation）** Yang et al. [2019] 使用多任务学习（multi-task learning）来组合一个教师集成的知识，以此对蒸馏进行正则化。作者用多任务知识蒸馏（Multi-Task Knowledge Distillation），从一组大型问答模型中学出一个紧凑的问答模型。多重蒸馏的一个应用是多语言：Tsai et al. [2019] 采用了与我们类似的方法，完全通过蒸馏从零开始预训练一个多语言模型。然而，正如消融研究所示，通过初始化和额外损失来利用教师的知识会带来显著提升。

**其他压缩技术（Other compression techniques）** 人们也研究了其他压缩大模型的技术。权重剪枝（weights pruning）方面的最新进展表明，可以在测试时去掉自注意力（self-attention）中的一些头（head），而性能不会显著下降 Michel et al. [2019]。有些层甚至可以缩减到只有一个头。另一条研究路线利用量化来得到更小的模型（Gupta et al. [2015]）。剪枝和量化与本工作是正交的。

---

## 6 结论与未来工作（Conclusion and future work）

我们提出了 DistilBERT，一个通用的 BERT 预训练版本，体积小 40%，速度快 60%，保留了 97% 的语言理解能力。我们证明了通用语言模型可以成功地通过蒸馏训练，并用消融研究分析了各个组成部分。我们还进一步证明，DistilBERT 是边缘应用中一个很有吸引力的选择。

---

## 参考文献（References）

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2018.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar S. Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke S. Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692, 2019.

Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green ai. ArXiv, abs/1907.10597, 2019.

Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in nlp. In ACL, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, 2017.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. Transformers: State-of-the-art natural language processing, 2019.

Cristian Bucila, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In KDD, 2006.

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. ArXiv, abs/1503.02531, 2015.

Yukun Zhu, Ryan Kiros, Richard S. Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. 2015 IEEE International Conference on Computer Vision (ICCV), pages 19–27, 2015.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. In ICLR, 2018.

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. In NAACL, 2018.

Alex Wang, Ian F. Tenney, Yada Pruksachatkun, Katherin Yu, Jan Hula, Patrick Xia, Raghu Pappagari, Shuning Jin, R. Thomas McCoy, Roma Patel, Yinghui Huang, Jason Phang, Edouard Grave, Najoung Kim, Phu Mon Htut, Thibault F'evry, Berlin Chen, Nikita Nangia, Haokun Liu, Anhad Mohananey, Shikha Bordia, Nicolas Patry, Ellie Pavlick, and Samuel R. Bowman. jiant 1.1: A software toolkit for research on general-purpose text understanding models. http://jiant.info/, 2019.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In ACL, 2011.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100, 000+ questions for machine comprehension of text. In EMNLP, 2016.

Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, and Jimmy Lin. Distilling task-specific knowledge from bert into simple neural networks. ArXiv, abs/1903.12136, 2019.

Debajyoti Chatterjee. Making neural machine reading comprehension faster. ArXiv, abs/1904.00796, 2019.

Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: The impact of student initialization on knowledge distillation. ArXiv, abs/1908.08962, 2019.

Ze Yang, Linjun Shou, Ming Gong, Wutao Lin, and Daxin Jiang. Model compression with multi-task knowledge distillation for web-scale question answering system. ArXiv, abs/1904.09636, 2019.

Henry Tsai, Jason Riesa, Melvin Johnson, Naveen Arivazhagan, Xin Li, and Amelia Archer. Small and practical bert models for sequence labeling. In EMNLP-IJCNLP, 2019.

Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In NeurIPS, 2019.

Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. In ICML, 2015.
