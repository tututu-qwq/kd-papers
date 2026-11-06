> 出处说明：本文是论文《Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes》（Cheng-Yu Hsieh 等，2023）的中文译本，原文链接：https://arxiv.org/abs/2305.02301

# 逐步蒸馏！用更少训练数据和更小模型规模超越更大的语言模型（Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes）

**Cheng-Yu Hsieh**¹\*　**Chun-Liang Li**²　**Chih-Kuan Yeh**³　**Hootan Nakhost**²  
**Yasuhisa Fujii**³　**Alexander Ratner**¹　**Ranjay Krishna**¹　**Chen-Yu Lee**²　**Tomas Pfister**²  
¹华盛顿大学（University of Washington）　²Google Cloud AI Research　³Google Research  
cydhsieh@cs.washington.edu

arXiv:2305.02301v2 [cs.CL]，2023 年 7 月 5 日

\* 作者在 Google Cloud AI Research 担任学生研究员期间完成此项工作。  
¹ 源代码见：https://github.com/google-research/distilling-step-by-step

> 译者注：图不翻译、不重绘，下文用一句话说明图意。「译者注」均为帮助初学者理解而加，原文没有。

---

## 摘要（Abstract）

部署大语言模型（large language models, LLMs）很困难，因为它们内存效率低、计算开销大，难以用于实际应用。对此，研究者会训练更小的任务专用模型：要么用人工标签做微调（finetuning），要么用 LLM 生成的标签做蒸馏（distillation）。然而，微调与蒸馏都需要大量训练数据，才能达到与 LLM 相当的性能。我们提出「逐步蒸馏」（Distilling step-by-step），这是一种新机制，能够（a）训练出比 LLM 表现更好的更小模型，并且（b）所用训练数据少于微调或蒸馏所需。我们的方法在多任务（multi-task）框架中，把 LLM 的推理过程（rationales）提取出来，作为训练小模型的额外监督信号。我们在 4 个 NLP 基准上得到三点发现：第一，与微调和蒸馏相比，我们的机制用少得多的有标签/无标签训练样本就能取得更好表现；第二，与少样本提示（few-shot prompted）的 LLM 相比，我们用显著更小的模型规模就能取得更好表现；第三，我们同时减小了模型规模和所需数据量以超越 LLM——我们微调后的 770M T5 模型，仅用某基准上 80% 的可用数据，就超过了少样本提示的 540B PaLM 模型；而同一 T5 模型用标准微调，即便用上 100% 数据也难以追上。

---

## 1 引言（Introduction）

尽管大语言模型（LLMs）（Brown et al., 2020; Chowdhery et al., 2022; Thoppilan et al., 2022; Hoffmann et al., 2022; Smith et al., 2022b; Zhang et al., 2022）展现出令人印象深刻的少样本能力，但由于体量巨大，它们很难在真实应用中部署。服务一个 1750 亿参数的 LLM，使用专用基础设施至少需要 350GB GPU 显存（Zheng et al., 2022）。更糟的是，当今最先进的 LLM 已超过 5000 亿参数（Chowdhery et al., 2022），需要更多内存与算力。这类算力要求对大多数产品团队远不可承受，尤其是对低延迟应用。

> **图 1**：大语言模型（LLMs）虽有很强的零样本/少样本表现，但实际服务困难；传统训练小任务模型又需要大量训练数据。本文提出「逐步蒸馏」：从 LLM 中提取推理过程（rationales），作为有信息量的任务知识来训练小模型，从而同时减小部署模型规模和训练所需数据量。

为绕开大模型的部署难题，实践者往往改为部署更小的专用模型。这些小模型通常用两种范式之一训练：**微调**或**蒸馏**。微调用下游人工标注数据，更新一个预训练小模型（如 BERT（Devlin et al., 2018）或 T5（Raffel et al., 2020））（Howard and Ruder, 2018）。蒸馏则用更大 LLM 生成的标签训练同样的小模型（Tang et al., 2019; Wang et al., 2021; Smith et al., 2022a; Arora et al., 2022）。可惜，这些范式在缩小模型时付出了代价：要达到与 LLM 相当的性能，微调需要昂贵的人工标签，蒸馏则需要大量往往难以获得的无标签数据（Tang et al., 2019; Liang et al., 2020）。

本文提出「逐步蒸馏」（Distilling step-by-step），一种用更少训练数据训练更小模型的简单新机制。该机制减少了把 LLM 微调或蒸馏到更小模型规模时所需的训练数据。核心在于视角转变：不再把 LLM 当作噪声标签源，而把它们当作能推理的主体——LLM 可以用自然语言推理过程（rationales）为其预测标签辩护（Wei et al., 2022; Kojima et al., 2022）。例如，当问「Jesse 的房间长 11 英尺、宽 15 英尺。若她已有 16 平方英尺地毯，还需要多少地毯才能铺满整个地面？」时，可用思维链（chain-of-thought, CoT）技术（Wei et al., 2022）提示 LLM 给出中间推理「面积 = 长 × 宽。Jesse 的房间有 $11 \times 15$ 平方英尺。」，从而把输入与最终答案「$(11 \times 15) - 16$」更好地连起来。这些推理可包含相关任务知识，例如「面积 = 长 × 宽」，而小任务模型原本可能需要很多数据才能学到。因此我们把提取出的推理作为额外、更丰富的信息，通过多任务训练设置（同时做标签预测与推理预测任务）来训练小模型（Raffel et al., 2020; Narang et al., 2020）。

逐步蒸馏让我们能学到任务专用小模型：用超过 500× 更少的模型参数就能超过 LLM，并且所需训练样本远少于传统微调或蒸馏（图 1）。结果在 4 个 NLP 基准上给出三点有前景的经验结论。第一，与微调和蒸馏相比，我们的模型平均在各数据集上用少 50% 以上的训练样本就能取得更好表现（最高可减少超过 85%）。第二，我们的模型以小得多的规模（最高可达 2000× 更小）超过 LLM，大幅降低部署所需算力。第三，我们同时减小模型规模与所需数据量以超越 LLM。我们用 770M 的 T5 模型超过 5400 亿参数 LLM 的表现；该小模型仅使用现有微调方法原本需要的有标签数据集的 80%。当仅有无标签数据时，我们的小模型仍能与 LLM 持平或更好。我们仅用 11B 的 T5 模型就超过 540B PaLM 的表现。我们进一步表明：当小模型弱于 LLM 时，与标准蒸馏相比，逐步蒸馏能更高效地利用额外无标签数据来追上 LLM 表现。

---

## 2 相关工作（Related work）

我们的工作利用当今 LLM 涌现的推理能力，把 LLM 的任务专用知识蒸馏到更小的专家模型中。我们借鉴知识蒸馏研究，以及从人工生成推理与 LLM 生成推理中学习的方法。

**来自大模型的知识蒸馏。** 知识蒸馏已成功用于把更大、更强的教师模型中的知识迁移到适合实际应用的更小学生模型中（Buciluǎ et al., 2006; Hinton et al., 2015; Beyer et al., 2022; West et al., 2021; Fu et al., 2023）。它支持在有限有标签数据上学习，因为更大的教师模型常被用来生成带噪声伪标签的训练数据集（Chen et al., 2020; Iliopoulos et al., 2022; Wang et al., 2021; Smith et al., 2022a; Arora et al., 2022; Agrawal et al., 2022）。知识蒸馏常面临的一个局限，是依赖大量无标签数据才能造出有用的噪声训练集。尽管已有工作用数据增强减少对数据的饥渴（Tang et al., 2019; Liang et al., 2020; Srinivas and Fleuret, 2018; Milli et al., 2019），我们提出另一条路：不仅蒸馏标签，还蒸馏教师的推理过程，从而减少对大量无标签数据的需求。

**利用人工推理进行学习。** 利用 LLM 生成的推理是令人兴奋的新方向，而使用人工生成的推理则有丰富历史（Hase and Bansal, 2021）。例如，人工推理可用于正则化模型行为（Ross et al., 2017）；可作为额外输入引导模型预测（Rajani et al., 2019）；可提升整体模型表现（Zaidan et al., 2007; Zhang et al., 2016; Camburu et al., 2018; Hancock et al., 2019; Pruthi et al., 2022）；也可作为金标准标签，让模型通过生成类似推理而更可解释（Wiegreffe et al., 2021; Narang et al., 2020; Eisenstein et al., 2022）。可惜，人工推理成本高昂。

**利用 LLM 生成的推理进行学习。** 当今 LLM 能通过生成高质量推理步骤来解释其预测（Wei et al., 2022; Kojima et al., 2022）。这些推理步骤被用作增强 LLM 输入提示，以提升少样本或零样本表现（Wei et al., 2022; Kojima et al., 2022; Wang et al., 2022b）；也被用作额外微调数据，让 LLM「自我改进」（Zelikman et al., 2022; Huang et al., 2022）。可惜，无论怎样改进 LLM，其巨大规模都会限制在多数测试时应用中的效用。

相比之下，我们把生成的推理作为有信息量的监督，来训练更小的任务专用模型，即无需承担大计算或内存成本即可部署的模型。若干同期工作也提出了与我们类似的想法——用提取的推理作监督（Wang et al., 2022a; Ho et al., 2022; Magister et al., 2022; Li et al., 2023）。其中，PINTO（Wang et al., 2022a）在测试时仍依赖 LLM 生成推理，因而并未完全解决部署难题。与 Ho et al. (2022) 和 Magister et al. (2022) 相比，我们在他们的实验之外做了更细致的研究：变化训练数据集规模、探索下游模型规模，并证明我们的方法在完全无标签数据集上也有效。

---

## 3 逐步蒸馏（Distilling step-by-step）

我们提出新范式「逐步蒸馏」：利用 LLM 为其预测做推理的能力，以数据高效的方式训练更小模型。整体框架见图 2。该范式有两个简单步骤：第一，给定一个 LLM 与一个无标签数据集，我们提示 LLM 生成输出标签，以及为标签辩护的推理。推理是自然语言解释，为模型预测的标签提供支持（见图 2）。第二，除任务标签外，我们还用这些推理来训练更小的下游模型。直观上，推理提供了更丰富、更细致的信息，说明为何输入被映射到特定输出标签，且常包含仅从原始输入很难推断出的相关任务知识。

> **图 2**：逐步蒸馏总览。我们先用 CoT 提示从 LLM 提取推理（第 3.1 节）；再用生成的推理，在多任务学习框架中训练小任务模型——在输入样例前加上任务前缀，并根据给定任务前缀训练模型输出不同内容（第 3.2 节）。

### 3.1 从 LLM 中提取推理（Extracting rationales from LLMs）

近期研究观察到 LLM 一个有趣的涌现特性：它们能为预测生成支持性推理（Wei et al., 2022; Kojima et al., 2022）。这些研究多关注如何从 LLM 中引出推理能力（Nye et al., 2021; Wei et al., 2022; Kojima et al., 2022），而我们则把它们用于训练更小的下游模型。

具体地，我们用思维链（Chain-of-Thought, CoT）提示（Wei et al., 2022）从 LLM 引出并提取推理。如图 3 所示，给定无标签数据集 $x_i \in D$，我们先编写提示模板 $p$，说明任务应如何求解。每个提示是一个三元组 $(x_p, r_p, y_p)$，其中 $x_p$ 是示例输入，$y_p$ 是对应标签，$r_p$ 是用户提供的、解释为何 $x_p$ 可归为 $y_p$ 的推理。我们把每个输入 $x_i$ 接到 $p$ 后面，作为输入去提示 LLM，为每个 $x_i \in D$ 生成推理与标签。借助在 $p$ 中见到的示范，LLM 能模仿三元组示范，为 $x_i$ 生成推理 $\hat{r}_i$ 与输出 $\hat{y}_i$。

> **图 3**：我们使用少样本 CoT 提示，其中包含示例推理（原文中以绿色高亮）与标签（原文中以蓝色高亮），以在新输入样例上从 LLM 引出推理。

### 3.2 用推理训练更小模型（Training smaller models with rationales）

我们先描述学习任务专用模型的现有框架，再在此基础上把推理纳入训练。形式化地，把数据集记为 $D = \{(x_i, y_i)\}_{i=1}^{N}$，其中每个 $x_i$ 是输入，$y_i$ 是对应的期望输出标签。尽管我们的框架支持任意模态的输入与输出，实验中把 $x$ 与 $y$ 限制为自然语言。这种文本到文本框架（Raffel et al., 2020）涵盖多种 NLP 任务：分类、自然语言推理、问答等。

**标准微调与任务蒸馏。** 训练任务专用模型最常见的做法，是用监督数据微调预训练模型（Howard and Ruder, 2018）。在没有人工标注标签时，任务专用蒸馏（Hinton et al., 2015; Tang et al., 2019）用 LLM 教师生成伪噪声训练标签 $\hat{y}_i$ 来代替 $y_i$（Wang et al., 2021; Smith et al., 2022a; Arora et al., 2022）。

对这两种情形，小模型 $f$ 都被训练以最小化标签预测损失：

$$
\mathcal{L}_{\mathrm{label}} = \frac{1}{N}\sum_{i=1}^{N}\ell\big(f(x_i), \hat{y}_i\big), \tag{1}
$$

其中 $\ell$ 是预测词元与目标词元之间的交叉熵损失。为便于表述，式 (1) 中的 $\hat{y}_i$ 既可表示标准微调情形下的人工标注标签 $y_i$，也可表示模型蒸馏情形下的 LLM 预测标签 $\hat{y}_i$。

**带推理的多任务学习。** 为在 $x_i$ 与 $\hat{y}_i$ 之间建立更显式的联系，我们把提取出的推理 $\hat{r}_i$ 作为额外监督。把推理纳入下游模型训练有多种方式。一种直接做法是把 $\hat{r}_i$ 作为额外输入——如其他同期研究所提议（Rajani et al., 2019; Wang et al., 2022a）。换句话说，训练 $f(x_i, \hat{r}_i) \rightarrow \hat{y}_i$，以文本与推理 $[x, r]$ 同时为输入：

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\ell\big(f(x_i, \hat{r}_i), \hat{y}_i\big). \tag{2}
$$

可惜，这种设计在 $f$ 做预测前仍需 LLM 先生成推理，部署时仍离不开 LLM，限制了可部署性。

本文中，我们不把推理当作额外模型输入，而把「用推理学习」表述为多任务问题。具体地，我们训练模型 $f(x_i) \rightarrow (\hat{y}_i, \hat{r}_i)$，不仅预测任务标签，还根据文本输入生成对应推理：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{label}} + \lambda \mathcal{L}_{\mathrm{rationale}}, \tag{3}
$$

其中 $\mathcal{L}_{\mathrm{label}}$ 是式 (1) 中的标签预测损失，$\mathcal{L}_{\mathrm{rationale}}$ 是推理生成损失：

$$
\mathcal{L}_{\mathrm{rationale}} = \frac{1}{N}\sum_{i=1}^{N}\ell\big(f(x_i), \hat{r}_i\big). \tag{4}
$$

推理生成损失让模型学会为预测生成中间推理步骤，从而可能引导模型更好地预测最终标签。这就是我们提出的逐步蒸馏。与式 (2) 相比，测试时不需要推理 $\hat{r}_i$，因而测试时不再需要 LLM。

我们在输入样例前加上「任务前缀」`[label]`、`[rationale]`，并训练小模型：当提供 `[label]` 时输出 $\hat{y}_i$，当提供 `[rationale]` 时产生 $\hat{r}_i$（Raffel et al., 2020）。

> 译者注：多任务写法的关键是「小模型自己既能出答案也能出推理」；测试时只让它出答案即可，不必再调用大模型。这与把推理塞进输入端的做法不同。

---

## 4 实验（Experiments）

我们经验验证逐步蒸馏的有效性。第一，与标准微调与任务蒸馏相比，逐步蒸馏用少得多的训练样本就能取得更好表现，大幅提升学习小任务模型的数据效率（第 4.1 节）。第二，逐步蒸馏以小得多的模型规模超越 LLM 表现，大幅降低相对 LLM 的部署成本（第 4.2 节）。第三，我们考察逐步蒸馏超越 LLM 所需的最少资源——就训练样本数与模型规模两方面而言。结果表明逐步蒸馏用更少数据与更小模型即可超越 LLM，同时提升数据效率与可部署性（第 4.3 节）。最后，我们做消融研究，理解逐步蒸馏框架中不同组件与设计选择的影响（第 4.4 节）。

**设置。** 实验中，我们以 540B 的 PaLM 模型（Chowdhery et al., 2022）作为 LLM。任务专用下游模型使用 T5（Raffel et al., 2020），权重从公开来源² 的预训练权重初始化。CoT 提示在可用时遵循 Wei et al. (2022)，对新数据集则自行编写示例。更多实现细节见附录 A.1。

²https://huggingface.co/

**数据集。** 我们在涵盖 3 类 NLP 任务的 4 个流行基准上做实验：自然语言推理用 e-SNLI（Camburu et al., 2018）与 ANLI（Nie et al., 2020）；常识问答用 CQA（Talmor et al., 2019; Rajani et al., 2019）；算术应用题用 SVAMP（Patel et al., 2021）。更多数据集细节见附录 A.2。

### 4.1 减少训练数据（Reducing training data）

我们把逐步蒸馏与学习任务专用模型的两种最常见方法比较：（1）有人工标注样例时的**标准微调**（STANDARD FINETUNING），以及（2）仅有无标签样例时的**标准任务蒸馏**（STANDARD TASK DISTILLATION）。具体地，标准微调指用真实标签、经标准标签监督微调模型的主流「预训练—再微调」范式（Howard and Ruder, 2018）。另一方面，当仅有无标签样例时，标准任务蒸馏把教师 LLM 的预测标签当作真实标签来学习任务专用模型（Hinton et al., 2015; Chen et al., 2020; Wang et al., 2021; Smith et al., 2022a; Arora et al., 2022）。

在以下实验中，我们把任务专用模型固定为 220M 的 T5-Base，并比较不同方法在不同可用训练样本数下的任务表现。

**逐步蒸馏以少得多的有标签样例超过标准微调。** 用人工标注样例微调时，图 4 显示：在所用有标签样例数变化时，逐步蒸馏 优于标准微调。进一步地，逐步蒸馏能用少得多的有标签样例达到与标准微调相同的表现。尤其是，仅用完整 e-SNLI 数据集的 12.5%，逐步蒸馏就能超过用 100% 完整数据集训练的标准微调。类似地，在 ANLI、CQA 与 SVAMP 上，我们分别实现了 75%、25% 与 20% 的训练样本减少，以超过标准微调。

**逐步蒸馏以少得多的无标签样例超过标准蒸馏。** 当仅有无标签数据时，我们把逐步蒸馏与标准任务蒸馏比较。在图 5 中，我们观察到与微调设置总体相似的趋势。具体地，在所有 4 个数据集、不同无标签数据量下，逐步蒸馏都优于标准任务蒸馏。我们也看到逐步蒸馏需要少得多的无标签数据才能超过标准任务蒸馏。例如，在 e-SNLI 上，我们仅需完整无标签数据集的 12.5%，就能超过用 100% 训练样例做标准任务蒸馏所达到的表现。

> **图 4**：用 220M T5 模型，在不同规模的人工标注数据集上比较逐步蒸馏与标准微调。在所有数据集上，逐步蒸馏都能用少得多的训练样例（例如完整 e-SNLI 的 12.5%）超过用完整数据集训练的标准微调。

> **图 5**：与上图类似，用 220M T5 模型在不同规模的无标签数据集上比较逐步蒸馏与标准任务蒸馏。逐步蒸馏仅用完整无标签数据集的一小部分（例如 ANLI 上的 12.5%）即可超过标准任务蒸馏。

### 4.2 减小模型规模（Reducing model size）

在以下实验中，我们固定训练集规模（使用 100% 数据集），并把用逐步蒸馏与标准方法训练的不同规模小 T5 模型与 LLM 比较。具体地，我们考虑 3 种规模的 T5：220M T5-Base、770M T5-Large 与 11B T5-XXL。对 LLM，我们包含两种基线：（1）**少样本 CoT**（FEW-SHOT COT）（Wei et al., 2022），以及（2）**PINTO 调优**（PINTO TUNING）（Wang et al., 2022a）。少样本 CoT 直接用 CoT 示范提示 540B PaLM，在预测最终标签前生成中间步骤，不对 LLM 做进一步微调。PINTO 调优是我们对 Wang et al. (2022a) 的扩展，以处理该工作未研究的、超出问答的任务。这里我们在 PaLM 生成输出之上微调一个 220M T5-Base，可视为用额外参数对 LLM 做微调的方法（Zhang et al., 2020; Lester et al., 2021）。

我们分别在有标签数据集与无标签数据集两种宽泛场景下，把实验结果呈现在图 6 与图 7 中。我们按各方法用于预测的部署模型规模（横轴）与对应任务表现（纵轴）作图。

**在所用模型规模变化时，逐步蒸馏相对标准基线均有改进。** 在图 6 与图 7 中，我们看到逐步蒸馏在所有规模的 T5 模型上 改进标准微调与标准蒸馏。改进在 ANLI 上最明显：逐步蒸馏在任务准确率上平均分别超过标准微调与蒸馏 8% 与 13%。

**逐步蒸馏用小得多的任务专用模型超过 LLM。** 在图 6 中，当有人工标注数据集时，逐步蒸馏在所考虑的全部 4 个数据集上，总能用小得多的 T5 模型超过少样本 CoT 与 PINTO 调优。例如，相对 540B PaLM 的少样本 CoT，我们可在 e-SNLI 上用 220M（超过 2000× 更小）T5、在 ANLI 与 SVAMP 上用 770M（超过 700× 更小）T5、在 CQA 上用 11B（超过 45× 更小）T5 取得更好表现。即便用 PINTO 调优在可用有标签数据上进一步微调 540B PaLM，这些结果仍成立。³

在图 7 中，仅利用无标签样例，逐步蒸馏也在 4 个数据集中的 3 个上超过教师 LLM。具体地，逐步蒸馏用不到 PaLM 规模 3% 的 11B T5，就超过 540B PaLM 的少样本 CoT 表现。在蒸馏模型表现较弱的 SVAMP 上，我们假设性能差距来自数据集相对较少的数据点（即 800 个）。对此，我们提议用额外无标签样例增强数据集以缩小差距，如下所示。

³ 我们注意到 PETuning 方法可能优于 PINTO 调优。但它们在训练与部署上都需要海量资源，不是本文重点。

**无标签数据增强进一步改进逐步蒸馏。** 我们用 ASDiv 数据集（Miao et al., 2020）中的无标签样例增强 SVAMP 训练集。ASDiv 共有 2305 个样例，每个都是与 SVAMP 类似的数学应用题。在图 7 的 SVAMP 上，我们展示了用 ASDiv 增强训练集后、11B T5 上逐步蒸馏与标准任务蒸馏的表现。我们看到数据增强对两者都有明显改进。但即便加上无标签样例，标准任务蒸馏仍弱于少样本 CoT。另一方面，逐步蒸馏能更高效地利用新增样例的价值，达到与少样本 CoT 相同的表现水平——同样使用规模不到 540B PaLM 的 3% 的 T5 模型。

> **图 6**：在完整人工标注数据集上，对多种规模的 T5 做逐步蒸馏与标准微调，并与 LLM 基线（少样本 CoT 与 PINTO 调优）比较。逐步蒸馏能用小得多的模型超过 LLM 基线，例如在 ANLI 上超过 700× 更小的模型；标准微调用同一模型规模则无法匹配 LLM 表现。

> **图 7**：在无标签数据集上，对多种规模的 T5 做逐步蒸馏与标准任务蒸馏，并与少样本 CoT 比较。逐步蒸馏在 e-SNLI 上用 2000× 更小模型、在 ANLI 与 CQA 上用 45× 更小模型超过少样本 CoT。在 SVAMP 上，加入来自 ASDiv 的无标签样例后，我们缩小了与少样本 CoT 的差距，而标准蒸馏仍难以追上。

### 4.3 用最小模型规模与最少训练数据超越 LLM（Outperforming LLMs using minimum model size and least training data）

这里，以 LLM 表现为锚点，我们探索逐步蒸馏与标准微调/蒸馏要超过 LLM，在训练样例数与部署模型规模两方面最节省的资源需求。结果分别在有人工标注与无标签设置下，示于图 8 与图 9。我们按（1）所用训练样例数（横轴）、（2）最终任务表现（纵轴）、（3）模型规模（用阴影区域大小可视化）绘制不同结果模型。

**逐步蒸馏用更少数据、以小得多的模型超过 LLM。** 在图 8 的所有数据集上，我们看到逐步蒸馏仅用可用训练样例的一个子集，就能以小得多的 T5 模型超过 PaLM 的少样本 CoT。具体地，在 e-SNLI 上，逐步蒸馏可用超过 2000× 更小的模型（220M T5）以及仅 0.1% 的完整数据集，取得好于少样本 CoT 的表现。在图 9 中，当仅有无标签数据集时，我们观察到同样趋势：逐步蒸馏多数时候能以更小模型与更少数据超过少样本 CoT。例如在 ANLI 上，逐步蒸馏用 45× 更小模型与 50% 的完整无标签集超过 LLM。

**标准微调与蒸馏需要更多数据或更大模型。** 最后，在图 8 与图 9 中，我们看到标准微调与蒸馏往往需要更多数据或更大模型才能匹配 LLM 表现。例如在图 8 的 e-SNLI 上，逐步蒸馏仅用数据集的 0.1% 就超过 LLM，而标准微调需要更多数据才能追上。此外，在图 8 的 ANLI 上，逐步蒸馏用 770M 模型、仅 80% 训练集即可超过 PaLM，而标准微调即便用完整数据集也难以匹配 LLM，因而需要更大模型来弥合差距。

> **图 8**：通过粗粒度搜索，展示逐步蒸馏要超过 LLM 少样本 CoT 所需的最小 T5 规模与最少人工标注样例量。逐步蒸馏不仅用小得多的模型超过少样本 CoT，而且相对标准微调用少得多的训练样例即可做到。在 ANLI 上，我们用 770M 模型、仅 80% 数据集超过 LLM CoT，而标准微调即便用 100% 数据集也难以追上。

> **图 9**：与图 8 类似，但仅用无标签样例。相对标准任务蒸馏，逐步蒸馏能用小得多的模型与少得多的样例超过少样本 CoT。在 SVAMP 上，横轴对应用于增强原始 SVAMP 的 ASDiv 数据集规模，即 $x=0$ 表示无增强，$x=100$ 表示加入完整 ASDiv。

### 4.4 进一步消融研究（Further ablation studies）

到目前为止，我们聚焦于展示逐步蒸馏在减少微调或蒸馏小任务模型所需训练数据方面的有效性。本节做进一步分析，理解逐步蒸馏框架中不同组件的影响。具体地，我们研究（1）从不同 LLM 提取推理如何影响逐步蒸馏的有效性，以及（2）多任务训练方法相对其他可能设计选择（用 LLM 推理训练小任务模型）的比较。这里把小任务模型固定为 220M T5，并在所有数据集上使用 100% 数据。

**逐步蒸馏适用于不同规模、训练尚可的 LLM。** 除用 540B PaLM 作 LLM 外，这里还考虑相对较小的 20B GPT-NeoX 模型（Black et al., 2022），从中提取推理做逐步蒸馏。表 1 显示：与不同规模 LLM 耦合时，逐步蒸馏仍能相对标准微调带来性能提升。但从 20B GPT-NeoX 而非 540B PaLM 提取推理时，提升幅度更小。这可能是因为更大的 PaLM 提供了更高质量、对任务学习更有益的推理。

**表 1**：逐步蒸馏适用于不同规模的 LLM。当从 20B GPT-NeoX 提取推理时，逐步蒸馏相对标准微调仍能为 220M T5 带来性能提升。

| 数据集 | 方法 | LLM | e-SNLI | ANLI | CQA | SVAMP |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| STANDARD FINETUNING | N/A | 88.38 | 43.58 | 62.19 | 62.63 |
| DISTILLING STEP-BY-STEP | 20B | 89.12 | 48.15 | 63.25 | 63.00 |
| DISTILLING STEP-BY-STEP | 540B | 89.51 | 49.58 | 63.29 | 65.50 |

**多任务训练远比单任务「推理+标签」联合预测更有效。** 把 LLM 推理作为输出监督来训练任务专用模型，可以有不同做法。一种直接做法是把推理 $\hat{r}_i$ 与标签 $\hat{y}_i$ 拼成单一序列 $[\hat{r}_i, \hat{y}_i]$，并在训练小模型时把整个序列当作目标输出，如 Magister et al. (2022); Ho et al. (2022) 所考虑：

$$
\mathcal{L}_{\mathrm{single}} = \frac{1}{N}\sum_{i=1}^{N}\ell\big(f(x_i), [\hat{r}_i, \hat{y}_i]\big). \tag{5}
$$

在表 2 中，我们把这种单任务训练与我们提出的多任务训练比较，以利用 LLM 推理。我们看到：多任务训练不仅 带来更好表现，用 LLM 推理做单任务训练有时甚至会比标准微调更差，例如在 ANLI 与 CQA 上。事实上，Wiegreffe et al. (2021); Magister et al. (2022); Ho et al. (2022) 也观察到类似结果：简单把推理与标签预测当作单一联合任务，可能损害模型在标签预测上的表现。这验证了我们采用多任务训练的做法，并凸显需要谨慎对待推理，才能释放其真正益处。

**表 2**：我们提出的多任务训练框架 优于把推理与标签预测当作单一任务。单任务训练有时甚至比标准微调更差。

| 数据集 | 方法 | e-SNLI | ANLI | CQA | SVAMP |
| --- | --- | ---: | ---: | ---: | ---: |
| STANDARD FINETUNING | 88.38 | 43.58 | 62.19 | 62.63 |
| SINGLE-TASK TRAINING | 88.88 | 43.50 | 61.37 | 63.00 |
| MULTI-TASK TRAINING | 89.51 | 49.58 | 63.29 | 65.50 |

---

## 5 讨论（Discussion）

我们提出逐步蒸馏，从 LLM 中提取推理，作为训练小任务专用模型的有信息量监督。我们表明：逐步蒸馏减少了打造任务专用更小模型所需的训练数据集；也减少了达到乃至超过原 LLM 表现所需的模型规模。相对现有方法，逐步蒸馏提出了一种资源更高效的「训练—部署」范式。进一步研究展示了逐步蒸馏的泛化能力与其中的设计选择。最后，我们在下文讨论局限、未来方向与伦理声明。

### 局限性（Limitations）

我们的方法有若干局限。第一，用户需要制作少量示例示范（所有任务约 10-shot），才能使用少样本 CoT（Wei et al., 2022）提示机制。这一局限可借助近期进展改进：有研究表明无需任何用户标注示范也能引出推理（Kojima et al., 2022）。第二，用推理训练任务专用模型会带来轻微的训练时计算开销。但在测试时，我们的多任务设计自然避免该开销，因为它允许只预测标签而不生成推理。最后，尽管我们观察到用 LLM 推理的成功，也有证据表明 LLM 在更复杂的推理与规划任务上推理能力有限（Valmeekam et al., 2022）。未来工作应刻画推理质量如何影响逐步蒸馏。

### 伦理声明（Ethics statement）

值得注意的是，我们下游更小模型的行为会受到更大教师 LLM 所继承偏见的影响。我们设想：减少 LLM 反社会行为的同一研究进展，也可应用于改进更小的语言模型。

---

## 参考文献（References）

> 译者注：参考文献按原文保留英文，不翻译。

Priyanka Agrawal, Chris Alberti, Fantine Huot, Joshua Maynez, Ji Ma, Sebastian Ruder, Kuzman Ganchev, Dipanjan Das, and Mirella Lapata. 2022. Qameleon: Multilingual qa with only 5 examples. arXiv preprint arXiv:2211.08264.

Simran Arora, Avanika Narayan, Mayee F Chen, Laurel J Orr, Neel Guha, Kush Bhatia, Ines Chami, Frederic Sala, and Christopher Ré. 2022. Ask me anything: A simple strategy for prompting language models. arXiv preprint arXiv:2210.02441.

Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov. 2022. Knowledge distillation: A good teacher is patient and consistent. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10925–10934.

Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. 2022. GPT-NeoX-20B: An open-source autoregressive language model. In Proceedings of the ACL Workshop on Challenges & Perspectives in Creating Large Language Models.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Cristian Buciluǎ, Rich Caruana, and Alexandru Niculescu-Mizil. 2006. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 535–541.

Oana-Maria Camburu, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom. 2018. e-snli: Natural language inference with natural language explanations. Advances in Neural Information Processing Systems, 31.

Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. 2020. Big self-supervised models are strong semi-supervised learners. Advances in neural information processing systems, 33:22243–22255.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Jacob Eisenstein, Daniel Andor, Bernd Bohnet, Michael Collins, and David Mimno. 2022. Honest students from untrusted teachers: Learning an interpretable question-answering pipeline from a pretrained language model. arXiv preprint arXiv:2210.02498.

Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. 2023. Specializing smaller language models towards multi-step reasoning. arXiv preprint arXiv:2301.12726.

Braden Hancock, Antoine Bordes, Pierre-Emmanuel Mazare, and Jason Weston. 2019. Learning from dialogue after deployment: Feed yourself, chatbot! arXiv preprint arXiv:1901.05415.

Peter Hase and Mohit Bansal. 2021. When can models learn from explanations? a formal framework for understanding the roles of explanation data. arXiv preprint arXiv:2102.02201.

Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2(7).

Namgyu Ho, Laura Schmid, and Se-Young Yun. 2022. Large language models are reasoning teachers. arXiv preprint arXiv:2212.10071.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 328–339, Melbourne, Australia. Association for Computational Linguistics.

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. 2022. Large language models can self-improve. arXiv preprint arXiv:2210.11610.

Fotis Iliopoulos, Vasilis Kontonis, Cenk Baykal, Gaurav Menghani, Khoa Trinh, and Erik Vee. 2022. Weighted distillation with unlabeled examples. In Advances in Neural Information Processing Systems.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. arXiv preprint arXiv:2205.11916.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.

Liunian Harold Li, Jack Hessel, Youngjae Yu, Xiang Ren, Kai-Wei Chang, and Yejin Choi. 2023. Symbolic chain-of-thought distillation: Small models can also "think" step-by-step. arXiv preprint arXiv:2306.14050.

Kevin J Liang, Weituo Hao, Dinghan Shen, Yufan Zhou, Weizhu Chen, Changyou Chen, and Lawrence Carin. 2020. Mixkd: Towards efficient distillation of large-scale language models. arXiv preprint arXiv:2011.00593.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. 2022. Teaching small language models to reason. arXiv preprint arXiv:2212.08410.

Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. 2020. A diverse corpus for evaluating and developing english math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 975–984.

Smitha Milli, Ludwig Schmidt, Anca D Dragan, and Moritz Hardt. 2019. Model reconstruction from model explanations. In Proceedings of the Conference on Fairness, Accountability, and Transparency, pages 1–9.

Sharan Narang, Colin Raffel, Katherine Lee, Adam Roberts, Noah Fiedel, and Karishma Malkan. 2020. Wt5?! training text-to-text models to explain their predictions. arXiv preprint arXiv:2004.14546.

Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela. 2020. Adversarial NLI: A new benchmark for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. 2021. Show your work: Scratchpads for intermediate computation with language models. arXiv preprint arXiv:2112.00114.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2080–2094, Online. Association for Computational Linguistics.

Danish Pruthi, Rachit Bansal, Bhuwan Dhingra, Livio Baldini Soares, Michael Collins, Zachary C Lipton, Graham Neubig, and William W Cohen. 2022. Evaluating explanations: How much do explanations from the teacher aid students? Transactions of the Association for Computational Linguistics, 10:359–375.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67.

Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. 2019. Explain yourself! leveraging language models for commonsense reasoning. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4932–4942, Florence, Italy. Association for Computational Linguistics.

Andrew Slavin Ross, Michael C Hughes, and Finale Doshi-Velez. 2017. Right for the right reasons: Training differentiable models by constraining their explanations. arXiv preprint arXiv:1703.03717.

Ryan Smith, Jason A Fries, Braden Hancock, and Stephen H Bach. 2022a. Language models in the loop: Incorporating prompting into weak supervision. arXiv preprint arXiv:2205.02318.

Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. 2022b. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990.

Suraj Srinivas and François Fleuret. 2018. Knowledge transfer with jacobian matching. In International Conference on Machine Learning, pages 4723–4731. PMLR.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.

Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, and Jimmy Lin. 2019. Distilling task-specific knowledge from bert into simple neural networks. arXiv preprint arXiv:1903.12136.

Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kulshreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. 2022. Lamda: Language models for dialog applications. arXiv preprint arXiv:2201.08239.

Karthik Valmeekam, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. 2022. Large language models still can’t plan (a benchmark for llms on planning and reasoning about change). arXiv preprint arXiv:2206.10498.

Peifeng Wang, Aaron Chan, Filip Ilievski, Muhao Chen, and Xiang Ren. 2022a. Pinto: Faithful language reasoning using prompt-generated rationales. arXiv preprint arXiv:2211.01562.

Shuohang Wang, Yang Liu, Yichong Xu, Chenguang Zhu, and Michael Zeng. 2021. Want to reduce labeling cost? gpt-3 can help. arXiv preprint arXiv:2108.13487.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, and Denny Zhou. 2022b. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903.

Peter West, Chandra Bhagavatula, Jack Hessel, Jena D Hwang, Liwei Jiang, Ronan Le Bras, Ximing Lu, Sean Welleck, and Yejin Choi. 2021. Symbolic knowledge distillation: from general language models to commonsense models. arXiv preprint arXiv:2110.07178.

Sarah Wiegreffe, Ana Marasović, and Noah A. Smith. 2021. Measuring association between labels and free-text rationales. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10266–10284, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Omar Zaidan, Jason Eisner, and Christine Piatko. 2007. Using “annotator rationales” to improve machine learning for text categorization. In Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Proceedings of the Main Conference, pages 260–267, Rochester, New York. Association for Computational Linguistics.

Eric Zelikman, Yuhuai Wu, and Noah D Goodman. 2022. Star: Bootstrapping reasoning with reasoning. arXiv preprint arXiv:2203.14465.

Jeffrey O Zhang, Alexander Sax, Amir Zamir, Leonidas Guibas, and Jitendra Malik. 2020. Side-tuning: a baseline for network adaptation via additive side networks. In European Conference on Computer Vision, pages 698–714. Springer.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Ye Zhang, Iain Marshall, and Byron C. Wallace. 2016. Rationale-augmented convolutional neural networks for text classification. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 795–804, Austin, Texas. Association for Computational Linguistics.

Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Joseph E Gonzalez, et al. 2022. Alpa: Automating inter-and intra-operator parallelism for distributed deep learning. arXiv preprint arXiv:2201.12023.

---

## 附录 A 实验细节（Experiment detail）

### A.1 实现（Implementation）

我们在云端 A100×16 GPU 实例上做实验。使用 https://github.com/huggingface/transformers 上公开可用的包，按以下超参数训练 T5 模型：

- **T5-Base (220M) 与 T5-Large (770M)**：学习率 $= 5\times 10^{-5}$，batch size $= 64$，最大输入长度 $= 1024$，最多训练 10000 步。
- **T5-XXL (11B)**：学习率 $= 5\times 10^{-5}$，batch size $= 32$，最大输入长度 $= 1024$，最多训练 4000 步。

我们在 4 次随机运行上报告全部结果，并在图中给出标准误。

### A.2 数据集（Datasets）

我们提供实验所用数据集的更详细描述，包括获取来源与作者最初发布来源。请读者参考这些来源了解许可或使用/分发条款。据我们所知，所用数据集不包含可指名或唯一识别个人的信息，也不含冒犯性内容。

- **e-SNLI**：最初发布于 Camburu et al. (2018)，公开于 https://github.com/OanaMariaCamburu/e-SNLI。我们从 https://huggingface.co/datasets/esnli 获取。
- **ANLI**：最初发布于 Nie et al. (2020)，公开于 https://github.com/facebookresearch/anli。我们从 https://huggingface.co/datasets/anli 获取，实验使用 R1 划分。
- **CQA**：最初发布于 Talmor et al. (2019)，公开于 https://www.tau-nlp.sites.tau.ac.il/commonsenseqa。随后由 Rajani et al. (2019) 用人工标注解释增强，见 https://github.com/salesforce/cos-e。实验所用数据来自 https://huggingface.co/datasets/cos_e。
- **SVAMP**：最初发布于 Patel et al. (2021)。我们从 https://github.com/arkilpatel/SVAMP 获取。
- **ASDiv**：最初发布于 Miao et al. (2020)。我们从 https://github.com/chaochun/nlu-asdiv-dataset 获取。

对每个数据集，若原本没有验证集，我们从原始训练集中随机抽取 10% 作为验证集。对 CQA，由于原始测试集没有真实标签，我们用原始验证集作为测试集。数据集统计见表 3。

**表 3**：实验所用数据集统计。

| 数据集 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: |
| e-SNLI | 549,367 | 9,842 | 9,824 |
| ANLI | 16,946 | 1,000 | 1,000 |
| CQA | 8,766 | 975 | 1,221 |
| SVAMP | 720 | 80 | 200 |
