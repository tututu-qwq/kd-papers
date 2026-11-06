> 出处说明：本文是论文《MiniLLM: On-Policy Distillation of Large Language Models》（Yuxian Gu, Li Dong, Furu Wei, Minlie Huang；arXiv 版本标题与摘要一致，亦作 Knowledge Distillation of Large Language Models）的中文译本，原文链接：https://arxiv.org/abs/2306.08543

# MiniLLM：大语言模型的同策略蒸馏（MiniLLM: On-Policy Distillation of Large Language Models）

**Yuxian Gu**¹\*　**Li Dong**²　**Furu Wei**²　**Minlie Huang**¹†  
¹清华大学 CoAI 组（The CoAI Group, Tsinghua University）  
²微软研究院（Microsoft Research）  
guyx21@mails.tsinghua.edu.cn　{lidong1,fuwei}@microsoft.com　aihuang@tsinghua.edu.cn

arXiv:2306.08543v6 [cs.CL]，2026 年 1 月 31 日

\* 在微软研究院实习期间完成的贡献。  
† 通讯作者。

> 译者注：原文图不翻译、不重绘，下文用一句话说明图意。「译者注」为帮助初学者理解而加。

---

## 摘要（Abstract）

知识蒸馏（Knowledge Distillation, KD）是降低大语言模型（large language models, LLMs）高计算需求的有前景技术。然而，以往 KD 方法主要应用于白盒分类模型，或训练小模型去模仿 ChatGPT 这类黑盒模型 API。如何把白盒 LLM 的知识有效蒸馏到小模型中仍探索不足；随着开源 LLM 繁荣，这一点变得更重要。本文提出一种把 LLM 蒸馏到更小语言模型的 KD 方法。我们首先用反向 KL 散度（reverse Kullback-Leibler divergence, reverse KLD）替换标准 KD 中的前向 KL 散度（forward KLD）目标——它对生成式语言模型上的 KD 更合适，可防止学生模型高估教师分布的低概率区域。然后，我们推导一种有效的同策略（on-policy）优化方法来学习该目标。学生模型命名为 **MiniLLM**。在指令跟随（instruction-following）设定下的大量实验表明：相对基线，MiniLLM 生成更精确、整体质量更高的回复，曝光偏差（exposure bias）更低、校准更好、长文本生成表现更高。我们的方法可扩展到参数量从 120M 到 13B 的不同模型族。代码、数据与模型检查点见：https://github.com/microsoft/LMOps/tree/main/minillm。

> **图 1**：MiniLLM 与序列级 KD（SeqKD）在评估集上平均 GPT-4 反馈分数的比较。左：教师为 GPT-2-1.5B，学生为 GPT-2 125M/340M/760M；中：教师为 GPT-J 6B，学生为 GPT-2 760M/1.5B、GPT-Neo 2.7B；右：教师为 OPT 13B，学生为 OPT 1.3B/2.7B/6.7B。

---

## 1 引言（Introduction）

随着大语言模型（LLMs；BMR+20, HZD+21, BHA+21, CND+22, Ope23）快速发展，降低其高计算资源需求的常用技术是知识蒸馏（KD；HVD15）：用大教师模型的监督训练小学生学习模型。两类 KD 常用：**黑盒 KD**（仅可访问教师生成的文本）与**白盒 KD**（还可获得教师模型的输出分布或中间隐状态）[JBMD21]。近来，黑盒 KD 在用 LLM API 生成的提示—回复对上微调小模型方面展现出有前景的结果 [TGZ+23, CLL+23, WWZ+23, PLH+23]。随着更多开源 LLM 出现 [ZRG+22, TLI+23]，白盒 KD 对研究界与产业界更有价值，因为学生模型能从教师的输出分布与隐状态获得更好信号，从而可能得到更高表现。然而，白盒 KD 方法大多针对小规模（<1B 参数）语言理解模型研究 [SDCW19, WWD+20]，面向 LLM 的白盒 KD 仍待探索。

本文研究教师输出分布可用时的 LLM 白盒 KD。我们认为，标准 KD 目标 [KR16, SST+20, CLL+23, TGZ+23] 对以生成方式执行任务的 LLM 并非最优。给定教师分布 $p(y|x)$ 与由 $\theta$ 参数化的学生分布 $q_\theta(y|x)$，标准 KD 目标（包括若干序列级模型变体）本质上最小化教师与学生分布之间的近似前向 KL 散度 $\mathrm{KL}[p\|q_\theta]$，这迫使学生 $q_\theta$ 覆盖 $p$ 的所有模态（modes）。对文本分类任务，$\mathrm{KL}[p\|q_\theta]$ 效果较好，因为输出空间通常只有有限类别，$p$ 与 $q_\theta$ 的模态都很少。但对开放式文本生成——这也通常是 LLM 应用情形——输出空间复杂得多，且由于学生容量有限，$p(y|x)$ 可包含远多于 $q_\theta(y|x)$ 能表达的模态。最小化前向 KLD 会使 $q_\theta$ 给 $p$ 的空洞区域分配不合理的高概率 [MG19]，并在自由运行生成时产生在 $p$ 下极不可能的样本 [Hus15]。

> **图 2**：玩具实验：用前向 KLD 与反向 KLD，分别用单高斯拟合高斯混合目标分布。

为缓解该问题，我们提议最小化反向 KLD $\mathrm{KL}[q_\theta\|p]$，该方法在计算机视觉 [LPSK23] 与强化学习 [CPO+19] 中被广泛使用。相对 $\mathrm{KL}[p\|q_\theta]$，最小化 $\mathrm{KL}[q_\theta\|p]$ 会使 $q_\theta$ 寻找 $p$ 的主要模态，并对 $p$ 的空洞区域分配低概率 [M+05]，如图 2 所示并在第 2.1 节讨论。在文本生成中，这意味着学生避免学习教师分布中过多的长尾变体 [HBD+20]，而专注于生成正确性——这在要求真实性与可靠性的实际场景中至关重要 [JLF+23]。为优化 $\min_\theta \mathrm{KL}[q_\theta\|p]$，如第 2.2 节所示，我们用策略梯度（Policy Gradient）[SMSM99] 推导其梯度，并采用同策略训练方法。为进一步稳定并加速训练，我们提出：（1）单步分解以降低方差；（2）教师混合采样以缓解奖励黑客（reward hacking）；（3）长度归一化以消除长度偏置。最后，第 2.3 节给出整体 KD 算法。学生模型命名为 **MiniLLM**，表明该方法适合压缩大型（生成式）语言模型。

我们在涵盖大量 NLP 任务的指令跟随设定 [SWR+22, WBZ+22] 下，把方法应用到参数量从 120M 到 13B 的多种生成式语言模型 [RWC+19, ZRG+22, TLI+23]。我们用 5 个数据集，以 Rouge-L [Lin04]、GPT-4 反馈与人工评判做评估。实验表明：MiniLLM 在所有数据集上 consistently 优于标准 KD 基线，并可从 120M 良好扩展到 13B 模型（见图 1）。更多分析表明：MiniLLM 曝光偏差更低、校准更好、长回复生成表现更高，且多样性损失可忽略。

> 译者注：**前向 KL** $\mathrm{KL}[p\|q]$ 倾向于「覆盖所有模态」（mode-covering），学生容易把概率撒到教师几乎不会生成的区域；**反向 KL** $\mathrm{KL}[q\|p]$ 倾向于「寻找主要模态」（mode-seeking），更适合容量有限的学生。

---

## 2 方法（Method）

我们考虑条件文本生成：模型根据从分布 $p_x$ 采样的提示 $x$，产生回复 $y=\{y_t\}_{t=1}^{T}$，这通常也是 LLM 执行任务的方式。我们把 KD 表述为优化问题：最小化固定教师模型分布 $p(y|x)$ 与由 $\theta$ 参数化的学生模型分布 $q_\theta(y|x)$ 之间的差异。标准 KD 方法近似³最小化前向 KLD：

$$
\mathrm{KL}[p\|q_\theta]=\mathbb{E}_{x\sim p_x,\,y\sim p'}\log\frac{p(y|x)}{q_\theta(y|x)},
$$

其中 $p'$ 可以是真实数据分布（词级 KD）或教师分布 $p$（序列级 KD）。尽管广泛使用，当 $q_\theta$ 表达能力不足时，$\mathrm{KL}[p\|q_\theta]$ 在文本生成任务中倾向于高估 $p$ 的空洞区域 [JKH+23]。面向 LLM 的 KD 正属于这种情况：LLM 以生成方式执行任务，低容量学生无法完美模仿教师或人类的复杂文本生成分布。

> **图 3**：序列级 KD（左）与 MiniLLM（右）的比较。序列级 KD 迫使学生记忆教师生成的全部样本；MiniLLM 则用教师反馈改进自身生成文本。

³ 说「近似」是因为词级 KD 中 $y$ 从真实分布而非教师分布采样。对足够强的教师，可近似认为两分布相同。

### 2.1 MiniLLM：用反向 KLD 做知识蒸馏（Knowledge Distillation with Reverse KLD）

我们把学生与教师模型分布之间的反向 KLD 最小化，作为 MiniLLM 的学习目标：

$$
\theta=\arg\min_\theta \mathcal{L}(\theta)=\arg\min_\theta \mathrm{KL}[q_\theta\|p]
=\arg\min_\theta\left[-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta}\log\frac{p(y|x)}{q_\theta(y|x)}\right]. \tag{1}
$$

已有工作表明，最小化反向 KLD 会在生成建模中引发「寻模态」行为 [Hus15, NCT16, CDP+18, LPSK23]：$q_\theta$ 给 $p$ 的大模态分配高概率，并忽略小模态（图 2 的玩具实验示意）。本文首次在文本生成的 LLM KD 中研究该性质。最小化前向 KLD 会使 $q_\theta$ 在 $p$ 的零概率区域放置大概率质量，对应实践中生成低质量文本；而反向 KLD 关注 $p$ 的主要模态，对保证文本生成的正确性与忠实性至关重要。如图 3 所示，与最小化前向 KLD 的序列级 KD [KR16, TGZ+23] 不同，最小化反向 KLD 的 MiniLLM 并不强迫 $q_\theta$ 拟合从教师分布 $p$ 采样的所有 $y$。相反，它鼓励学生在自身容量内生成教师偏好的样本，这更可能实现。有趣的是，我们也发现可从逆强化学习（Inverse Reinforcement Learning）[ZMB+08] 角度理解 MiniLLM，相关推导见附录 A.1。

### 2.2 同策略蒸馏（On-Policy Distillation）

**梯度推导。** 注意到式 (1) 中目标函数 $\mathcal{L}(\theta)$ 的梯度可用策略梯度定理（Policy Gradient Theorem）[Wil92, HTAL17] 推导，用于同策略优化：

$$
\nabla\mathcal{L}(\theta)=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}(R_t-1)\nabla\log q_\theta(y_t|y_{<t},x), \tag{2}
$$

其中 $T=|y|$，$R_t=\sum_{t'=t}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}$ 是逐步质量度量 $r_{t'}=\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}$ 的累积。直观上，生成文本应通过增大 $p(y_{t'}|y_{<t'},x)$ 在教师分布下具有高概率，同时通过降低 $q_\theta(y_{t'}|y_{<t'},x)$ 保持多样。式 (2) 中的期望由蒙特卡洛采样计算。完整推导见附录 A.2。然而，策略梯度存在高方差与奖励黑客问题 [SHKK22]，尽管已有后续方案 [SWD+17]。此外，我们注意到 $R_t$ 偏好短句，会使学生输出空回复。因此我们提出三种策略缓解这些问题。

**单步分解（Single-Step Decomposition）。** [CPO+19] 发现单步生成质量 $r_t$ 对训练方差很关键，因为前端词元的误差会沿整句累积。为更关注 $r_t$，我们改写 $\nabla\mathcal{L}(\theta)$，从 $R_t$ 中分解出 $r_t$，并直接计算 $\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]$ 的梯度（完整推导见附录 A.3）：

$$
\nabla\mathcal{L}(\theta)
=\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\left[-\sum_{t=1}^{T}\nabla\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]\right]
+\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\left[-\sum_{t=1}^{T}R_{t+1}\nabla\log q_\theta(y_t|y_{<t},x)\right]
=(\nabla\mathcal{L})_{\mathrm{Single}}+(\nabla\mathcal{L})_{\mathrm{Long}}, \tag{3}
$$

其中 $q_\theta(t)=q_\theta(\cdot|y_{<t},x)$。注意 $\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]$ 可通过在词表上求和直接计算，无需蒙特卡洛，且对 $\theta$ 可导。该分解对单步生成质量给出更精确、高效的估计，降低训练方差并加速收敛。

**教师混合采样（Teacher-Mixed Sampling）。** 用式 (2) 训练时我们观察到奖励黑客 [SHKK22]：采样时 $q_\theta$ 有时产生退化句子 $y$（例如重复短语）却从教师那里拿到高分，对小学生模型尤其如此。为得到更好采样分布，我们在每一步混合教师与学生分布：

$$
\tilde{p}(y_t|y_{<t},x)=\alpha\cdot p(y_t|y_{<t},x)+(1-\alpha)\cdot q_\theta(y_t|y_{<t},x), \tag{4}
$$

其中 $\alpha$ 控制教师混入强度。从 $\tilde{p}$ 采样借助教师抑制低质量生成，缓解奖励黑客。我们用重要性采样改写 $(\nabla\mathcal{L})_{\mathrm{Single}}$ 与 $(\nabla\mathcal{L})_{\mathrm{Long}}$，得到梯度的无偏估计 [PSS00]：

$$
(\nabla\mathcal{L})_{\mathrm{Single}}=-\mathbb{E}_{x\sim p_x,\,y\sim\tilde{p}(\cdot|x)}\left[\sum_{t=1}^{T}w_t\nabla\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]\right],
\quad
(\nabla\mathcal{L})_{\mathrm{Long}}=-\mathbb{E}_{x\sim p_x,\,y\sim\tilde{p}(\cdot|x)}\left[\sum_{t=1}^{T}w_t R_{t+1}\nabla\log q_\theta(y_t|y_{<t},x)\right], \tag{5}
$$

其中 $w_t=\prod_{t'=1}^{t}\frac{q_\theta(y_{t'}|y_{<t'},x)}{\tilde{p}(y_{t'}|y_{<t'},x)}$ 是重要性权重。但 $w_t$ 在实践中方差很大，因为它需要在多时间步上乘逐词元重要性权重，每步方差会累积。因此我们近似取 $w_t\approx\frac{q_\theta(y_t|y_{<t},x)}{\tilde{p}(y_t|y_{<t},x)}$，以降低式 (5) 估计器的方差 [SSG+17, LKTF20]。

**长度归一化（Length Normalization）。** 我们发现长序列往往有较小的 $R_{t+1}$，这会鼓励模型产生短回复。因此对式 (3) 中的 $R_{t+1}$ 加入长度归一化：

$$
R^{\mathrm{Norm}}_{t+1}=\frac{1}{T-t-1}\sum_{t'=t+1}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}. \tag{6}
$$

**小结。** 综合以上策略，最终优化梯度为：

$$
\nabla\mathcal{L}(\theta)=-\mathbb{E}_{x\sim p_x,\,y\sim\tilde{p}(\cdot|x)}
\left[\sum_{t=1}^{T}w_t\left(
\underbrace{\nabla\sum_{y'\in\mathcal{V}}q_\theta(y'|y_{<t},x)\log\frac{p(y'|y_{<t},x)}{q_\theta(y'|y_{<t},x)}}_{(\nabla\mathcal{L})_{\mathrm{Single}}\text{ 部分}}
+\underbrace{R^{\mathrm{Norm}}_{t+1}\frac{\nabla q_\theta(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}}_{(\nabla\mathcal{L})^{\mathrm{Norm}}_{\mathrm{Long}}\text{ 部分}}
\right)\right], \tag{7}
$$

其中 $\mathcal{V}$ 是语言模型词表，$(\nabla\mathcal{L})^{\mathrm{Norm}}_{\mathrm{Long}}$ 是带 $R^{\mathrm{Norm}}_{t+1}$ 的 $(\nabla\mathcal{L})_{\mathrm{Long}}$。

### 2.3 训练算法（Training Algorithm）

我们从在大规模长文档语料 $\mathcal{D}_{\mathrm{PT}}$ 上预训练的学生模型出发。算法 1 通过在数据集 $\mathcal{D}$ 上、在教师监督下把学生适配到文本生成任务来训练 MiniLLM；教师可以是在 $\mathcal{D}$ 上微调过的 LLM [TGZ+23, CLL+23]，或具有良好任务泛化的模型 [CHL+22, Ope23]。算法中，我们先在 $\mathcal{D}$ 上微调学生，并选取损失最低的检查点作为后续训练初始化。然后基于式 (5) 与式 (6) 计算梯度 $(\nabla\mathcal{L})_{\mathrm{Single}}$ 与 $(\nabla\mathcal{L})^{\mathrm{Norm}}_{\mathrm{Long}}$，并加入裁剪策略 [SWD+17] 以进一步提高稳定性。与 [OWJ+22] 相同，我们加入语言建模损失 $\mathcal{L}_{\mathrm{PT}}=-\mathbb{E}_{d\sim\mathcal{D}_{\mathrm{PT}}}\log q_\theta(d)$，以保持模型在经典 NLP 基准上的表现。学生最终用组合梯度 $(\nabla\mathcal{L})_{\mathrm{Single}}+(\nabla\mathcal{L})^{\mathrm{Norm}}_{\mathrm{Long}}+\nabla\mathcal{L}_{\mathrm{PT}}$ 更新。整个同策略训练流程类似来自人类反馈的强化学习（RLHF；OWJ+22）。

**算法 1 MiniLLM：LLM 的知识蒸馏**

**输入：** 由提示与真实回复组成的条件生成数据集 $\mathcal{D}$；由长文档纯文本组成的预训练语料 $\mathcal{D}_{\mathrm{PT}}$；输出分布为 $p$ 的教师模型；在 $\mathcal{D}_{\mathrm{PT}}$ 上预训练、输出分布为 $q_{\theta_0}$ 的初始学生模型；学习率 $\eta$；批大小 $M$；裁剪阈值 $\epsilon$。  
**输出：** 输出分布为 $q_\theta$ 的学生模型。

1. 从 $\theta_0$ 出发，在 $\mathcal{D}$ 上用真实回复监督微调学生，选取验证损失最低的 $\theta$。  
2. **重复直至收敛：**  
   - 从 $\mathcal{D}$ 采样一小批提示，并从 $\tilde{p}$ 收集回复，得到 $S=\{(x_m,y_m)\}_{m=1}^{M}$；  
   - 从 $\mathcal{D}_{\mathrm{PT}}$ 采样一小批 $\mathcal{D}'_{\mathrm{PT}}=\{d_m\}_{m=1}^{M}$；  
   - 按式 (5) 计算 $(\nabla\mathcal{L})_{\mathrm{Single}}$；  
   - 按式 (5)(6) 计算 $(\nabla\mathcal{L})^{\mathrm{Norm}}_{\mathrm{Long}}$（对 $\rho_t(\theta)=\frac{q_\theta(y_t|y_{<t},x)}{\tilde{p}(y_t|y_{<t},x)}$ 使用 $\mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)$）；  
   - 计算语言建模损失梯度 $\nabla\mathcal{L}_{\mathrm{PT}}$；  
   - 更新：$\theta\leftarrow\theta-\eta\big[(\nabla\mathcal{L})_{\mathrm{Single}}+(\nabla\mathcal{L})^{\mathrm{Norm}}_{\mathrm{Long}}+\nabla\mathcal{L}_{\mathrm{PT}}\big]$。  
3. 返回 $q_\theta$。

---

## 3 实验（Experiments）

### 3.1 实验设置（Experimental Setup）

我们以指令跟随 [OWJ+22] 作为条件文本生成任务：模型按指令生成回复。我们先在由指令—回复对组成的数据集 $\mathcal{D}$ 上微调大模型作为教师，再在 $\mathcal{D}$ 上比较不同 KD 方法，评估学生的指令跟随表现。

**基座模型。** 学生来自三个模型族、多种规模：GPT-2 [RWC+19]（120M、340M、760M）、OPT [ZRG+22]（1.3B、2.7B、6.7B）与 LLaMA [TLI+23]（7B）。各模型族的教师分别使用 GPT-2-1.5B、OPT-13B 与 LLaMA-13B，这些模型事先在 $\mathcal{D}$ 上微调。用 GPT-J [WK21] 作教师的结果见附录 C.1。

**训练。** 训练数据来自 databricks-dolly-15K⁴，含 15K 人工撰写的指令—回复对。我们过滤超过模型上下文长度的样本，再随机划分 1K 与 0.5K 分别作验证与测试，剩余约 12.5K 用于训练。对 $\mathcal{D}_{\mathrm{PT}}$，GPT-2 族用 OpenWebText [GCPT19]，其他模型用 RoBERTa 训练语料 [LOG+19]。式 (4) 中教师混入强度全程取 $\alpha=0.2$。我们在验证集上用 Rouge-L [Lin04] 搜索超参数，因为它比验证损失更对齐人类偏好 [WMA+22]。更多细节见附录 B.1。

⁴https://github.com/databrickslabs/dolly/tree/master

**评估。** 我们在五个指令跟随数据集上评估训练后的模型：

- **DollyEval**：从 databricks-dolly-15k 划出的 500 样本测试集。  
- **SelfInst** [WKM+23]：面向用户的指令跟随集，252 样本。  
- **VicunaEval** [CLL+23]：Vicuna 评估所用的 80 道挑战性问题。  
- **S-NI**：SUPER-NATURALINSTRUCTIONS [WMA+22] 测试集，约 9K 样本、119 个任务。按 [PLH+23]，按真实回复长度划为 $[0,5]$、$[6,10]$、$[11,+\infty]$ 三个子集。第 3.2 节使用 $[11,+\infty]$ 子集，第 3.3 节分析全部子集。  
- **UnNI**：从 UNNATURALINSTRUCTIONS [HSLS23] 核心集随机采样 10K 样本评估；类似 S-NI，先在 $[11,+\infty]$ 子集评估，全部子集分析见附录 C.3。

我们采用三种指标评估模型生成的回复：

- **R-L**：Rouge-L [Lin04] 分数，度量生成精度。[WMA+22] 表明 Rouge-L 适合大规模指令跟随评估。  
- **GPT4**：向 GPT-4 询问，让它比较模型生成回复与真实答案⁵，并给双方打 1–10 分（提示见附录 B.2）。我们报告模型回复总分与真实答案总分之比。该指标仅用于 DollyEval、SelfInst 与 VicunaEval。  
- **人工评估**：按 [PLH+23] 在 SelfInst 上做人工评估：志愿者比较不同模型产生的两条回复，标注「胜」「平」或「负」。细节见附录 B.3。

对所有测试集，我们以温度 $=1$ 采样回复，并对每个提示用不同随机种子报告 5 次生成的平均分。

⁵ VicunaEval 的真实回复使用 ChatGPT 的生成 [Ope22]。

**基线。** 主实验考虑三种基线：

- **SFT w/o KD**：直接用黄金回复在 $\mathcal{D}$ 上微调学生。  
- **KD** [SDCW19, SST+20]：在 $\mathcal{D}$ 上以教师分布作为每步词元监督微调学生，亦称词级 KD。  
- **SeqKD** [KR16, CLL+23, TGZ+23, PLH+23, ZLX+23]：在教师生成的数据上微调学生。

### 3.2 结果（Results）

表 1 给出 R-L 与 GPT4 评估结果，我们有三点观察。

第一，把 MiniLLM 的整体表现与基线比较，可见在几乎所有情形下——不同基座、不同评估集、Rouge-L 与 GPT-4 两种打分——我们的 KD 方法蒸馏出的模型都优于基线，验证了该方法良好的泛化与高整体表现。我们也发现相对基线，MiniLLM 在 Dolly 以外的数据集上通常好得多，表明其分布外泛化较好。

第二，Rouge-L 分数显示 MiniLLM 产生与真实回复重叠最高、最精确的回复。在某些情形，尤其在 Vicuna、S-NI 与 UnNI 上，学生甚至达到比教师更高的 Rouge-L，与 [FLT+18] 的观察一致。我们推测：在 $\mathcal{D}$ 上标准教师强制（teacher-forcing）微调会给教师带来训练—推理差异，即曝光偏差 [BVJS15]。相反，MiniLLM 用策略优化方法训练，训练时从学生采样回复，从而缓解曝光偏差 [PH21]。第 3.3 节进一步分析曝光偏差。

第三，跨模型规模与模型族比较可见：当基座从 120M 变化到 13B、跨越三个模型族时，MiniLLM 的改进保持一致。该趋势亦见图 1，表明我们的方法在 LLM 时代具有出色的可扩展性与泛化性。基于 LLaMA 族在 SelfInst 上的人工评估结果见图 4：MiniLLM 获得优于所有基线的人类偏好，表现可与教师模型相当。

> **图 4**：人工评估结果。学生为 LLaMA-7B，教师为 LLaMA-13B。图中展示 MiniLLM 相对各基线的胜/平/负比例。

**表 1：评估结果。** GPT4 与 R-L 分别表示 5 个随机种子上的平均 GPT-4 反馈分数与 Rouge-L 分数。各模型规模下最佳分数加粗；学生超过教师的分数标以 *。

| 模型 | #参数 | 方法 | DollyEval GPT4 | DollyEval R-L | SelfInst GPT4 | SelfInst R-L | VicunaEval GPT4 | VicunaEval R-L | S-NI R-L | UnNI R-L |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | 1.5B | Teacher | 58.4 | 27.6 | 42.9 | 14.3 | 48.6 | 16.3 | 27.6 | 31.8 |
| | 120M | SFT w/o KD | 38.6 | 23.3 | 26.3 | 10.0 | 32.8 | 14.7 | 16.3 | 18.5 |
| | | KD | 40.3 | 22.8 | 27.8 | 10.8 | 31.9 | 13.4 | 19.7 | 22.0 |
| | | SeqKD | 41.2 | 22.7 | 26.2 | 10.1 | 31.0 | 14.3 | 16.4 | 18.8 |
| | | MiniLLM | **44.7** | **24.6** | **29.2** | **13.2** | **34.1** | **16.9\*** | **25.3** | **26.6** |
| | 340M | SFT w/o KD | 51.9 | 25.5 | 39.6 | 13.0 | 42.3 | 16.0 | 25.1 | 32.0 |
| | | KD | 51.6 | 25.0 | 39.2 | 12.0 | 42.8 | 15.4 | 23.7 | 31.0 |
| | | SeqKD | 50.5 | 25.3 | 39.0 | 12.6 | 43.0 | 16.9\* | 22.9 | 30.2 |
| | | MiniLLM | **52.2** | **25.4** | **40.5** | **15.6** | 42.6 | **17.7\*** | **27.4** | **34.5** |
| | 760M | SFT w/o KD | 50.7 | 25.4 | 38.3 | 12.4 | 43.1 | 16.1 | 21.5 | 27.1 |
| | | KD | 53.4 | 25.9 | 40.4 | 13.4 | 43.4 | 16.9\* | 25.3 | 31.7 |
| | | SeqKD | 52.0 | 25.6 | 38.9 | 14.0 | 42.4 | 15.9 | 26.1 | 32.9 |
| | | MiniLLM | **54.7** | **26.4** | **44.6\*** | **15.9** | **45.7** | **18.3\*** | **29.3\*** | **37.7\*** |
| OPT | 13B | Teacher | 70.3 | 29.2 | 56.1 | 18.4 | 58.0 | 17.8 | 30.4 | 36.1 |
| | 1.3B | SFT w/o KD | 52.6 | 26.0 | 37.7 | 11.4 | 40.5 | 15.6 | 23.1 | 28.4 |
| | | KD | 52.7 | 25.4 | 36.0 | 12.2 | 40.8 | 14.9 | 21.9 | 27.0 |
| | | SeqKD | 51.0 | 26.1 | 36.6 | 12.7 | 42.6 | 16.6 | 21.4 | 28.2 |
| | | MiniLLM | **60.7** | **26.7** | **47.0** | **14.8** | **50.6** | **17.9\*** | **28.6** | **33.4** |
| | 2.7B | SFT w/o KD | 55.4 | 27.1 | 38.9 | 13.9 | 44.8 | 16.6 | 24.9 | 32.3 |
| | | KD | 60.5 | 25.9 | 48.6 | 13.8 | 51.3 | 16.7 | 26.3 | 30.2 |
| | | SeqKD | 57.6 | 27.5 | 40.5 | 13.3 | 44.5 | 16.5 | 25.3 | 32.3 |
| | | MiniLLM | **63.2** | **27.4** | **52.7** | **17.2** | **55.9** | **19.1\*** | **30.7\*** | **35.1** |
| | 6.7B | SFT w/o KD | 67.9 | 27.6 | 56.4 | 16.4 | 57.3 | 17.8 | 30.3 | 28.6 |
| | | KD | 68.6 | 28.3 | 58.0 | 17.0 | 57.0 | 17.5 | 30.7\* | 26.7 |
| | | SeqKD | 69.6 | 28.5 | 54.0 | 17.0 | 57.6 | 17.9\* | 30.4 | 28.2 |
| | | MiniLLM | **70.8\*** | **29.0** | **58.5\*** | **17.5** | **60.1\*** | **18.7\*** | **32.5\*** | **36.7\*** |
| LLaMA | 13B | Teacher | 79.0 | 29.7 | 75.5 | 23.4 | 65.1 | 19.4 | 35.8 | 38.5 |
| | 7B | SFT w/o KD | 73.0 | 26.3 | 69.2 | 20.8 | 61.6 | 17.5 | 32.4 | 35.8 |
| | | KD | 73.7 | 27.4 | 70.5 | 20.2 | 62.7 | 18.4 | 33.7 | 37.9 |
| | | SeqKD | 73.6 | 27.5 | 71.5 | 20.8 | 62.6 | 18.1 | 33.7 | 37.6 |
| | | MiniLLM | **76.4** | **29.0** | **73.1** | **23.2** | **64.1** | **20.7\*** | **35.5** | **40.2\*** |

### 3.3 分析（Analysis）

> **图 5**：基于 GPT-2 族的教师规模定律。学生固定为 GPT-2-125M，教师分别为 GPT-2 340M、760M、1.5B，比较 MiniLLM 与 SeqKD。

**教师的规模定律。** 尽管直觉上更大教师可蒸馏出更好学生，[MFL+20] 表明增大教师规模并不保证学生改进，有时甚至损害蒸馏表现。尚不清楚扩大教师规模时 MiniLLM 如何表现。因此我们固定学生规模、用不同规模教师比较 MiniLLM 与 SeqKD。基于 GPT-2 族的结果见图 5，基于 OPT 族的结果见附录 C.2。可见 MiniLLM 始终优于 SeqKD，且学生表现与教师规模正相关，显示该方法压缩超大参数模型的潜力。

> **图 6**：由训练—解码差异导致的超额误差（ExAccErr）随生成长度累积。ExAccErr 越低，表示方法引入的曝光偏差越少。

**表 2**：SST2 与 BoolQ 上的 ECE 与准确率。学生模型中最佳分数加粗。

| | SST2 ECE | SST2 Acc. | BoolQ ECE | BoolQ Acc. |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 0.025 | 93.0 | 0.356 | 74.5 |
| KD | 0.191 | 84.7 | 0.682 | 63.5 |
| SeqKD | 0.243 | 66.5 | 0.681 | 62.8 |
| MiniLLM | **0.099** | **89.7** | **0.502** | **67.8** |

> **图 7**：按真实回复长度划分的 S-NI 不同子集上，各蒸馏模型相对 SFT 的 Rouge-L。

**表 3**：基于 LLaMA 族，测试集上 distinct 4-grams（Dist-4）与语言建模损失（Loss）。MiniLLM 保持生成多样性。

| | DollyEval Dist-4 | DollyEval Loss | SelfInst Dist-4 | SelfInst Loss |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 99.3 | 3.55 | 99.1 | 4.44 |
| SFT | 99.5 | 3.89 | 99.0 | 5.28 |
| MiniLLM | 99.0 | 3.95 | 98.6 | 5.33 |

**曝光偏差。** 被训练为最小化前向 KLD 的语言生成模型会遭受曝光偏差 [BVJS15]，源于教师强制训练与自由运行生成之间的差异。训练 MiniLLM 时，学生见到自己生成的样本，缓解训练—推理不匹配 [PH21]。图 6 用附录 B.5 定义的 ExAccErr 度量因曝光偏差导致的超额累积误差。实验基于 GPT-2-125M，教师为 GPT-2-1.5B，测试集为 Dolly。对每个提示采样 10 条回复以降低方差。可见基线的 ExAccErr 在生成过程中持续增长，而 MiniLLM 的 ExAccErr 低得多，且在长文本生成（>150 token）中误差停止累积。

**校准。** [Ope23] 表明用策略优化训练的模型可能校准较差。我们在两个广泛使用的文本分类数据集 SST2 [SPW+13] 与 BoolQ [CLC+19] 上，基于 LLaMA-7B 测试 MiniLLM 与 KD 基线的校准。我们设计零样本分类指令（见附录 B.2），并用标签词概率计算 ECE 分数 [NDZ+19]。由表 2，KD 与 SeqKD 模型校准差于教师，这可能解释它们在经典基准上表现偏低 [GWS+23]。我们怀疑最小化前向 KLD 会把高概率推到目标分布的空洞区域，导致学生与教师分布差异显著（见图 2 示例）。相反，MiniLLM 专注于准确学习目标分布的主要部分，缩小学生与教师之间的 ECE 差距。

**不同回复长度上的表现。** 我们研究真实回复长度属于不同范围时模型的表现。图 7 展示三种 S-NI 子集（按真实回复长度划分）上，不同 KD 模型相对 SFT 的 Rouge-L。可见所有方法在期望短回复（≤5 token）的提示上分数都低，可能因为训练集多数回复是长句，引入训练—评估分布偏移 [PLH+23]。此外，这些提示的输出空间相对较小，学生可覆盖教师的多数模态，因而反向 KLD 与前向 KLD 表现相近。对更长回复（≥6 token）的提示，由于输出空间复杂，教师分布比学生含有更多模态，从而显现 MiniLLM 相对标准 KD 的优势。UnNI 上的类似结果见附录 C.3。

**生成多样性。** [CCF+20] 发现最小化反向 KLD 优化的模型可能丢失模态，影响生成多样性。我们按 [PH21] 从三方面讨论多样性：（i）给定提示生成多条不同回复；（ii）生成语言上复杂的回复；（iii）生成能高覆盖真实数据分布的内容。对（i），我们认为对许多 NLP 应用，生成一条正确回复已足够，尤其在高真实性与可靠性场景 [JLF+23]。对（ii）（iii），表 3 报告回复的 distinct 4-gram 比例与测试集语言建模损失（LLaMA 族基座；细节见附录 B.4）。可见 MiniLLM 保持生成回复中的 distinct 4-gram 比例与测试集语言建模损失。

### 3.4 优化策略的消融研究（Ablation Studies on Optimization Strategies）

我们通过把 GPT-2-125M 从 GPT-2-1.5B 蒸馏出来，评估第 2.2 节提出的三种稳定并加速优化的策略的有效性。更多消融见附录 C.4。表 4 报告每次运行在验证集上的最佳 Rouge-L 及对应检查点的评估结果。我们也在图 8 中绘制训练过程中学生与教师之间的反向 KLD，曲线经 32 步平滑。可见教师混合采样与长度归一化对稳定训练有效。尽管没有这些策略时反向 KLD 也会下降，但我们发现模型很快学会生成在教师分布中概率很高的重复、短或无意义字符串（示例见附录 D），即奖励黑客 [SHKK22]。这也导致表 4 中生成表现偏低。从图 8 还可观察到单步分解有效降低训练过程方差，并带来更高的验证与测试分数。

**表 4**：应用不同 MiniLLM 优化策略组合时，验证集与测试集上的表现。

| | Valid. R-L | Dolly R-L |
| --- | ---: | ---: |
| MiniLLM | 27.4 | 24.6 |
| w/o Length Norm. | 17.4 | 14.7 |
| w/o Teacher-Mixed | 22.3 | 20.4 |
| w/o Single-Step | 27.0 | 23.7 |

> **图 8**：应用不同优化策略时，MiniLLM 训练过程中教师与学生之间的反向 KLD。

---

## 4 相关工作（Related Work）

**大语言模型。** 大语言模型（LLMs；BMR+20, TDFH+22, CND+22, Ope23, ADF+23）通过以生成方式解决多种 NLP 任务展现出优越表现。近期工作用指令调优 [WBZ+22, SWR+22, CHL+22] 或从人类反馈学习 [OWJ+22, BJN+22] 进一步改进 LLM 与人类的对齐，并打造通用 AI 助手 [Ope22, Goo23]。也有工作构建开源 LLM [ZRG+22, TLI+23, BSA+23] 以促进研究与产业发展。尽管吸引人，LLM 的广泛能力通常随模型规模涌现 [KMH+20, WTB+22]，需要海量算力。因此模型压缩对 LLM 的实际部署与进一步研究至关重要。

**知识蒸馏。** 知识蒸馏（KD；HVD15）作为广泛使用的模型压缩技术，旨在用教师指导训练学生 [RCG+15, SDCW19, JBMD21]。在 NLP 界，许多工作通过模仿教师输出分布 [SST+20, LHS+21, ZSL+23]、隐状态 [JYS+20, SCGL19] 或注意力分数 [WWD+20, WBH+21] 把 KD 用于文本分类。对文本生成，标准 KD 方法是近似最小化学生与教师生成分布之间的前向 KLD：以教师每步输出作监督 [SDCW19]，或直接在教师生成文本上训练 [KR16, TGZ+23, CLL+23, PLH+23]。本文在教师分布可用时最小化反向 KLD，这对 LLM 更合适。同期工作 [AVS+23, WLDM23] 也进一步探索 KD 中的分布差异度量。

**文本生成中的分布差异度量。** 分布差异度量在训练文本生成模型中起重要作用。前向 KLD 因可简单推导为最大似然估计（MLE）目标而被广泛使用 [ZZ19]。然而已有工作表明，最小化前向 KLD 会导致「零强制」行为：模型试图覆盖目标分布所有模态，牺牲主要模态的准确性 [Hus15]。一些工作改用其他度量补救，如反向 KLD [JHC+20]、全变差距离（Total Variation Distance）[JKH+23] 与最优传输（Optimal Transport）[LLW+20]。本文在面向 LLM 的知识蒸馏场景下处理该问题。

---

## 5 结论（Conclusion）

本文研究把 LLM 知识蒸馏到小语言模型的问题。我们发现：最小化前向 KLD 的标准蒸馏方法在语言生成场景中并非最优，因为教师输出分布比学生含有更多模态，前向 KLD 迫使学生高估教师分布的低概率区域。因此我们提出 **MiniLLM**：最小化教师与学生分布之间的反向 KLD，并设计算法优化该目标。大量实验表明：相对标准 KD 模型，MiniLLM 产生更精确、整体质量更高的回复；曝光偏差更低、校准更好、长文本生成表现更高，并保持良好生成多样性。

---

## 致谢（Acknowledgements）

本工作得到中国国家重点研发计划（No. 2021ZD0113304）、国家杰出青年科学基金（No. 62125604）以及国家自然科学基金重点项目（No. 61936010）资助。

---

## 参考文献（References）

> 译者注：参考文献按原文保留英文（含引用缩写形式）。

References
[ADF+23] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexan-
dre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm
2 technical report. arXiv preprint arXiv:2305.10403, 2023.
[AEABC22] Kushal Arora, Layla El Asri, Hareesh Bahuleyan, and Jackie Cheung. Why expo-
sure bias matters: An imitation learning perspective of error accumulation in language
generation. In Findings of ACL, 2022.
[AVS+23] Rishabh Agarwal, Nino Vieillard, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and
Olivier Bachem.
GKD: Generalized knowledge distillation for auto-regressive se-
quence models. arXiv preprint arXiv:2306.13649, 2023.
[BHA+21] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Syd-
ney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brun-
skill, et al.
On the opportunities and risks of foundation models.
arXiv preprint
arXiv:2108.07258, 2021.
[BJN+22] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova Das-
Sarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training
a helpful and harmless assistant with reinforcement learning from human feedback.
arXiv preprint arXiv:2204.05862, 2022.
10



[BLW+21] Sid Black, Gao Leo, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large
Scale Autoregressive Language Modeling with Mesh-Tensorflow, March 2021.
[BMR+20] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et al. Language models
are few-shot learners. In Proceedings of NeurIPS, 2020.
[BSA+23] Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O’Brien,
Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth,
Edward Raff, et al. Pythia: A suite for analyzing large language models across training
and scaling. arXiv preprint arXiv:2304.01373, 2023.
[BVJS15] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling
for sequence prediction with recurrent neural networks. In Proceedings of NeurIPS,
2015.
[CCF+20] Massimo Caccia, Lucas Caccia, William Fedus, Hugo Larochelle, Joelle Pineau, and
Laurent Charlin. Language gans falling short. In ICLR, 2020.
[CDP+18] Liqun Chen, Shuyang Dai, Yunchen Pu, Erjin Zhou, Chunyuan Li, Qinliang Su,
Changyou Chen, and Lawrence Carin. Symmetric variational autoencoder and con-
nections to adversarial learning. In Proceedings of AISTATS, 2018.
[CHL+22] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fe-
dus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling
instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.
[Cio21] Kamil Ciosek. Imitation learning by reinforcement learning. In ICLR, 2021.
[CLC+19] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael
Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of nat-
ural yes/no questions. In Proceedings of NAACL-HLT, 2019.
[CLL+23] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin
Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P.
Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,
March 2023.
[CND+22] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav
Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian
Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint
arXiv:2204.02311, 2022.
[CPO+19] Wojciech M Czarnecki, Razvan Pascanu, Simon Osindero, Siddhant Jayakumar, Grze-
gorz Swirszcz, and Max Jaderberg. Distilling policy distillation. In Proceedings of
AISTATS, 2019.
[CvdS21] Alex James Chan and Mihaela van der Schaar. Scalable bayesian inverse reinforcement
learning. In ICLR, 2021.
[FLT+18] Tommaso Furlanello, Zachary Lipton, Michael Tschannen, Laurent Itti, and Anima
Anandkumar. Born again neural networks. In Proceedings of ICML, 2018.
[GCPT19] Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext cor-
pus, 2019.
[Goo23] Google. Bard, 2023.
[GWS+23] Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel,
Sergey Levine, and Dawn Song. The false promise of imitating proprietary llms. arXiv
preprint arXiv:2305.15717, 2023.
[HBD+20] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of
neural text degeneration. In Proceedings of ICLR, 2020.
11



[HLM22] Yongchang Hao, Yuxin Liu, and Lili Mou. Teacher forcing recovers reward functions
for text generation. In Proceeings of NeurIPS, 2022.
[HSLS23] Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural instructions:
Tuning language models with (almost) no human labor. In Proceedings of ACL, 2023.
[HTAL17] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement
learning with deep energy-based policies. In Proceedings of ICML, 2017.
[Hus15] Ferenc Huszár. How (not) to train your generative model: Scheduled sampling, likeli-
hood, adversary? arXiv preprint arXiv:1511.05101, 2015.
[HVD15] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural
network. arXiv preprint arXiv:1503.02531, 2015.
[HZD+21] Xu Han, Zhengyan Zhang, Ning Ding, Yuxian Gu, et al. Pre-trained models: Past,
present and future. AI Open, 2021.
[JBMD21] Gou Jianping, Yu Baosheng, Stephen J Maybank, and Tao Dacheng. Knowledge dis-
tillation: A survey. IJCV, 2021.
[JHC+20] Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Tuo
Zhao. Smart: Robust and efficient fine-tuning for pre-trained natural language models
through principled regularized optimization. In Proceedings ACL, 2020.
[JKH+23] Haozhe Ji, Pei Ke, Zhipeng Hu, Rongsheng Zhang, and Minlie Huang.
Tailoring
language generation models under total variation distance. In Proceedings of ICLR,
2023.
[JLF+23] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin
Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language
generation. ACM Computing Surveys, 2023.
[JYS+20] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang,
and Qun Liu. Tinybert: Distilling bert for natural language understanding. In Findings
of EMNLP, 2020.
[KMH+20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Re-
won Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws
for neural language models. arXiv preprint arXiv:2001.08361, 2020.
[KR16] Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Pro-
ceedings of EMNLP, 2016.
[LGB+16] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-
promoting objective function for neural conversation models. In Kevin Knight, Ani
Nenkova, and Owen Rambow, editors, Proceedings of NAACL, 2016.
[LHS+21] Kevin J Liang, Weituo Hao, Dinghan Shen, Yufan Zhou, Weizhu Chen, Changyou
Chen, and Lawrence Carin.
Mix{kd}: Towards efficient distillation of large-scale
language models. In Proceedings of ICLR, 2021.
[Lin04] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Pro-
ceedings of Text Summarization Branches Out (ACL 2004), 2004.
[LKTF20] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu.
Offline reinforce-
ment learning: Tutorial, review, and perspectives on open problems. arXiv preprint
arXiv:2005.01643, 2020.
[LLW+20] Jianqiao Li, Chunyuan Li, Guoyin Wang, Hao Fu, Yuhchen Lin, Liqun Chen, Yizhe
Zhang, Chenyang Tao, Ruiyi Zhang, Wenlin Wang, et al. Improving text generation
with student-forcing optimal transport. In Proceedings of EMNLP, 2020.
12



[LOG+19] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly
optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692, 2019.
[LPSK23] Hyoje Lee, Yeachan Park, Hyun Seo, and Myungjoo Kang. Self-knowledge distillation
via dropout. Computer Vision and Image Understanding, 2023.
[M+05] Tom Minka et al. Divergence measures and message passing. Technical report, Cite-
seer, 2005.
[MFL+20] Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa,
and Hassan Ghasemzadeh. Improved knowledge distillation via teacher assistant. In
Proceedings of AAAI, 2020.
[MG19] Andrey Malinin and Mark Gales. Reverse KL-divergence training of prior networks:
Improved uncertainty and adversarial robustness. In Proceedings of NeurIPS, 2019.
[NCT16] Sebastian Nowozin, Botond Cseke, and Ryota Tomioka.
f-gan: Training genera-
tive neural samplers using variational divergence minimization. In Proceedings of
NeurIPS, 2016.
[NDZ+19] Jeremy Nixon, Michael W Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin
Tran. Measuring calibration in deep learning. In CVPR workshops, 2019.
[Ope22] OpenAI. OpenAI: Introducing ChatGPT, 2022.
[Ope23] OpenAI. GPT-4 technical report, 2023.
[OWJ+22] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Train-
ing language models to follow instructions with human feedback. In Proceedings of
NeurIPS, 2022.
[PH21] Richard Yuanzhe Pang and He He. Text generation by learning from demonstrations.
In Proceedings of ICLR, 2021.
[PLH+23] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruc-
tion tuning with GPT-4. arXiv preprint arXiv:2304.03277, 2023.
[PSS00] Doina Precup, Richard S Sutton, and Satinder P Singh. Eligibility traces for off-policy
policy evaluation. In Proceedings of ICML, 2000.
[RCG+15] Andrei A Rusu, Sergio Gomez Colmenarejo, Caglar Gulcehre, Guillaume Desjardins,
James Kirkpatrick, Razvan Pascanu, Volodymyr Mnih, Koray Kavukcuoglu, and Raia
Hadsell. Policy distillation. arXiv preprint arXiv:1511.06295, 2015.
[RWC+19] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
Sutskever. Language models are unsupervised multitask learners. OpenAI Technical
report, 2019.
[SCGL19] Siqi Sun, Yu Cheng, Zhe Gan, and Jingjing Liu. Patient knowledge distillation for
BERT model compression. In Proceedings EMNLP, 2019.
[SDCW19] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf.
DistilBERT,
a distilled version of bert:
smaller, faster, cheaper and lighter.
arXiv preprint
arXiv:1910.01108, 2019.
[SHKK22] Joar Max Viktor Skalse, Nikolaus HR Howe, Dmitrii Krasheninnikov, and David
Krueger. Defining and characterizing reward gaming. In Proceedings of NeurIPS,
2022.
[SMSM99] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gra-
dient methods for reinforcement learning with function approximation. Proceedings
of NeurIPS, 1999.
13



[SPW+13] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning,
Andrew Ng, and Christopher Potts. Recursive deep models for semantic composition-
ality over a sentiment treebank. In Proceedings of EMNLP, October 2013.
[SSG+17] Iulian V Serban, Chinnadhurai Sankar, Mathieu Germain, Saizheng Zhang, Zhouhan
Lin, Sandeep Subramanian, Taesup Kim, Michael Pieper, Sarath Chandar, Nan Rose-
mary Ke, et al.
A deep reinforcement learning chatbot.
arXiv preprint
arXiv:1709.02349, 2017.
[SST+20] Kaitao Song, Hao Sun, Xu Tan, Tao Qin, Jianfeng Lu, Hongzhi Liu, and Tie-Yan Liu.
LightPAFF: A two-stage distillation framework for pre-training and fine-tuning. arXiv
preprint arXiv:2004.12817, 2020.
[SWD+17] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
[SWR+22] Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid
Alyafeai, et al. Multitask prompted training enables zero-shot task generalization. In
Proceedings of ICLR, 2022.
[TDFH+22] Romal Thoppilan, Daniel De Freitas, Jamie Hall, Noam Shazeer, Apoorv Kul-
shreshtha, Heng-Tze Cheng, Alicia Jin, Taylor Bos, Leslie Baker, Yu Du, et al. Lamda:
Language models for dialog applications. arXiv preprint arXiv:2201.08239, 2022.
[TGZ+23] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos
Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An instruction-
following LLaMA model. https://github.com/tatsu-lab/stanford_alpaca,
2023.
[TLI+23] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lam-
ple.
LLaMA: Open and efficient foundation language models.
arXiv preprint
arXiv:2302.13971, 2023.
[TWS18] Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral cloning from observation.
In Proceedings of IJCAI, 2018.
[WBH+21] Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong, and Furu Wei. MiniLMv2:
Multi-head self-attention relation distillation for compressing pretrained transformers.
In Findings of ACL, 2021.
[WBZ+22] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian
Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are
zero-shot learners. In Proceedings of ICLR, 2022.
[Wil92] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist
reinforcement learning. Machine learning, 1992.
[WK21] Ben Wang and Aran Komatsuzaki. GPT-J-6B: A 6 Billion Parameter Autoregressive
Language Model, 2021.
[WKM+23] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel
Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with
self-generated instructions. In Proceedings of ACL, 2023.
[WLDM23] Yuqiao Wen, Zichao Li, Wenyu Du, and Lili Mou. f-divergence minimization for
sequence-level knowledge distillation. In Proceedings of ACL, 2023.
[WMA+22] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza
Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik,
David Stap, et al. Benchmarking generalization via in-context instructions on 1,600+
language tasks. In Proceedings of EMNLP, 2022.
14



[WTB+22] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud,
Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abili-
ties of large language models. Transactions on Machine Learning Research, 2022.
[WWD+20] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. MiniLM:
Deep self-attention distillation for task-agnostic compression of pre-trained transform-
ers. In Proceedings of NeurIPS, 2020.
[WWZ+23] Minghao Wu, Abdul Waheed, Chiyu Zhang, Muhammad Abdul-Mageed, and Al-
ham Fikri Aji. Lamini-lm: A diverse herd of distilled models from large-scale in-
structions. arXiv preprint arXiv:2304.14402, 2023.
[ZCS+23] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yong-
hao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge
with mt-bench and chatbot arena. In Proceedings of NeurIPS, 2023.
[ZLX+23] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao,
Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. LIMA: Less is more for alignment. In
Proceedings of NeurIPS, 2023.
[ZMB+08] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum
entropy inverse reinforcement learning. In Proceedings of AAAI, 2008.
[ZRG+22] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui
Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. OPT: Open
pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
[ZSL+23] Rongzhi Zhang, Jiaming Shen, Tianqi Liu, Jialu Liu, Michael Bendersky, Marc Na-
jork, and Chao Zhang. Do not blindly imitate the teacher: Using perturbed loss for
knowledge distillation. arXiv preprint arXiv:2305.05010, 2023.
[ZZ19] Huan Zhang and Hai Zhao. Minimum divergence vs. maximum margin: an empirical
comparison on seq2seq models. In International Conference on Learning Representa-
tions, 2019.
15



A
Derivations
A.1


---

## 附录 A 推导（Derivations）

### A.1 从逆强化学习视角看 MiniLLM（A Perspective of MiniLLM from Inverse Reinforcement Learning）

在第 2.1 节中，我们把 KD 表述为最小化教师分布与学生分布之间差异的优化问题，最终得到最小化反向 KLD 的目标。另一种看法是：把 KD 视为在教师模型指导下训练学生模型，这类似于智能体从环境反馈中学习。遵循 [PH21]，我们把词元生成当作马尔可夫决策过程（Markov Decision Process）。在每个时间步 $t$，学生模型根据策略（生成概率）$q_\theta(y_t|y_{<t}, x)$，在给定状态（前缀）$(y_{<t}, x)$ 的条件下，从动作空间（词表）$\mathcal{V}$ 中选择一个动作（词元）$y_t$。

从这个视角看，标准 KD 对应于模仿学习（imitation learning）[Cio21] 中的行为克隆（behavior cloning, BC；TWS18）。然而，已知 BC 通常弱于逆强化学习（Inverse Reinforcement Learning, IRL；ZMB+08）——另一种模仿学习方法：先从环境示范中恢复奖励模型，再用策略优化算法 [SMSM99, SWD+17] 以该奖励训练策略。因此，在 KD 场景中，我们希望先从环境（教师模型）诱导出奖励 $r(y_t,(y_{<t},x))$，再训练学生模型以最大化该奖励为目标。我们采用最大熵逆强化学习框架 [ZMB+08, CvdS21]，于是环境中的 Q 函数 $Q(y_t,(y_{<t},x))$ 满足软贝尔曼方程（soft Bellman Equation）：

$$
Q(y_t,(y_{<t},x))=r(y_t,(y_{<t},x))+\gamma\log\sum_{y'\in\mathcal{V}}\exp\big[Q(y',(y_{\le t},x))\big]. \tag{8}
$$

我们遵循 [HLM22]，把 Q 函数参数化为 $Q(y_t,(y_{<t},x))=f(y_t,(y_{<t},x))$，并假设 $\gamma=1$，其中 $f(y_t,(y_{<t},x))$ 是教师模型的输出 logits。⁶ 于是奖励为：

$$
r(y_t,(y_{<t},x))=f(y_t,(y_{<t},x))-\log\sum_{y'\in\mathcal{V}}\exp\big[f(y',(y_{\le t},x))\big]. \tag{9}
$$

为最大化奖励，我们应用最大熵强化学习 [HTAL17]，其学习目标为：

$$
\max_\theta \mathcal{J}(\theta)=\max_\theta \mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{|y|}\Big[r(y_t,(y_{<t},x))+\mathcal{H}\big[q_\theta(\cdot|y_{<t},x)\big]\Big], \tag{10}
$$

其中 $\mathcal{H}\big[q_\theta(\cdot|y_{<t},x)\big]=-\mathbb{E}_{y_t\sim q_\theta(\cdot|y_{<t},x)}\log q_\theta(\cdot|y_{<t},x)$ 是学生模型在时间步 $t$ 的熵。

⁶ 教师模型分布满足 $p(y_t|y_{<t},x)=\dfrac{\exp(f(y_t,(y_{<t},x)))}{\sum_{y'\in\mathcal{V}}\exp(f(y',(y_{<t},x)))}$。

**目标之间的等价性（Equivalence Between Objectives）。** 我们证明式 (10) 与式 (1) 近似等价。首先用结合律改写奖励之和 $\sum_{t=1}^{|y|}r(y_t,(y_{<t},x))$：

$$
\sum_{t=1}^{|y|}r(y_t,(y_{<t},x))
=\sum_{t=1}^{|y|}\left[f(y_t,(y_{<t},x))-\log\sum_{y'\in\mathcal{V}}\exp\big[f(y',(y_{\le t},x))\big]\right] \tag{11}
$$

$$
=f(y_1,(y_{<1},x))+\sum_{t=2}^{|y|}\left[f(y_t,(y_{<t},x))-\log\sum_{y'\in\mathcal{V}}\exp\big[f(y',(y_{<t},x))\big]\right] \tag{12}
$$

$$
-\log\sum_{y'\in\mathcal{V}}\exp\big[f(y',(y_{\le|y|},x))\big] \tag{13}
$$

$$
\approx\sum_{t=1}^{|y|}\left[f(y_t,(y_{<t},x))-\log\sum_{y'\in\mathcal{V}}\exp\big[f(y',(y_{<t},x))\big]\right] \tag{14}
$$

$$
=\sum_{t=1}^{|y|}\log\frac{\exp(f(y_t,(y_{<t},x)))}{\sum_{y'\in\mathcal{V}}\exp(f(y',(y_{<t},x)))} \tag{15}
$$

$$
=\sum_{t=1}^{|y|}\log p(y_t|y_{<t},x). \tag{16}
$$

于是 $\mathcal{J}(\theta)$ 可近似改写为：

$$
\mathcal{J}(\theta)\approx\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{|y|}\Big[\log p(y_t|y_{<t},x)+\mathcal{H}\big[q_\theta(\cdot|y_{<t},x)\big]\Big] \tag{17}
$$

$$
=\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{|y|}\Big[\log p(y_t|y_{<t},x)-\log\big[q_\theta(\cdot|y_{<t},x)\big]\Big] \tag{18}
$$

$$
=-\mathrm{KL}(q_\theta\|p) \tag{19}
$$

$$
=-\mathcal{L}(\theta). \tag{20}
$$

因此，最大化 $\mathcal{J}(\theta)$ 近似等价于最小化 $\mathcal{L}(\theta)$。

### A.2 式 (2) 的推导（Derivation of Equation 2）

我们用策略梯度定理（Policy Gradient Theorem）[SMSM99] 计算 $\mathcal{L}(\theta)=\mathrm{KL}[q_\theta\|p]$ 对 $\theta$ 的梯度：

$$
\begin{aligned}
\nabla\mathcal{L}(\theta)
&=-\nabla\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\log\frac{p(y|x)}{q_\theta(y|x)}\\
&=-\int\nabla\left[q_\theta(y|x)\log\frac{p(y|x)}{q_\theta(y|x)}\right]\,dy\,dx\\
&=-\int q_\theta(y|x)\nabla\log\frac{p(y|x)}{q_\theta(y|x)}\,dy\,dx
-\int\log\frac{p(y|x)}{q_\theta(y|x)}\nabla q_\theta(y|x)\,dy\,dx\\
&=\int q_\theta(y|x)\nabla\log q_\theta(y|x)\,dy\,dx
-\int q_\theta(y|x)\log\frac{p(y|x)}{q_\theta(y|x)}\nabla\log q_\theta(y|x)\,dy\,dx\\
&=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\left(\log\frac{p(y|x)}{q_\theta(y|x)}-1\right)\nabla\log q_\theta(y|x)\\
&=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}\left(\sum_{t'=1}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}-1\right)\nabla\log q_\theta(y_t|y_{<t},x)\\
&=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}\left(\sum_{t'=t}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}-1\right)\nabla\log q_\theta(y_t|y_{<t},x), \tag{21}
\end{aligned}
$$

其中式 (21) 基于如下事实：$\log q_\theta(y_t|y_{<t},x)$ 只能影响 $y$ 中位置 $\ge t$ 的词元。令 $R_t=\sum_{t'=t}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}$，即得式 (2)。

### A.3 式 (3) 的推导（Derivation of Equation 3）

为推导式 (3)，先记：

$$
(\nabla\mathcal{L})_{\mathrm{Single}}=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\left[\sum_{t=1}^{T}\nabla\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]\right],
\qquad
(\nabla\mathcal{L})_{\mathrm{Long}}=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}R_{t+1}\nabla\log q_\theta(y_t|y_{<t},x). \tag{22}
$$

然后把 $\nabla\mathcal{L}(\theta)$ 改写为：

$$
\nabla\mathcal{L}(\theta)=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}(R_t-1)\nabla\log q_\theta(y_t|y_{<t},x) \tag{23}
$$

$$
=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}R_{t+1}\nabla\log q_\theta(y_t|y_{<t},x) \tag{24}
$$

$$
-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}\left(\log\frac{p(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}-1\right)\nabla\log q_\theta(y_t|y_{<t},x) \tag{25}
$$

$$
=(\nabla\mathcal{L})_{\mathrm{Long}}-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}\mathbb{E}_{y_t\sim q_\theta(\cdot|y_{<t},x)}\left(\log\frac{p(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}-1\right)\nabla\log q_\theta(y_t|y_{<t},x) \tag{26}
$$

$$
=(\nabla\mathcal{L})_{\mathrm{Long}}-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\sum_{t=1}^{T}\nabla\mathbb{E}_{y_t\sim q_\theta(\cdot|y_{<t},x)}\left[-\log\frac{q_\theta(y_t|y_{<t},x)}{p(y_t|y_{<t},x)}\right] \tag{27}
$$

$$
=(\nabla\mathcal{L})_{\mathrm{Long}}-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta(\cdot|x)}\left[\sum_{t=1}^{T}\nabla\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]\right] \tag{28}
$$

$$
=(\nabla\mathcal{L})_{\mathrm{Long}}+(\nabla\mathcal{L})_{\mathrm{Single}}, \tag{29}
$$

其中式 (27) 使用了梯度的乘积法则，且 $r_t=\log\frac{p(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}$。

---

## 附录 B 实验细节（Experimental Details）

### B.1 训练细节（Training Details）

**基线。** 我们的基线包括 SFT w/o KD、KD 与 SeqKD。对参数量小于 1.3B 的模型，我们在学习率 $[5\times10^{-4},\,1\times10^{-4},\,5\times10^{-5}]$、批大小 $[32,\,64]$ 中搜索，并训练 20 个 epoch。对其余模型，在学习率 $[5\times10^{-5},\,1\times10^{-5},\,5\times10^{-6}]$、批大小 $[32,\,64]$ 中搜索，并训练 10 个 epoch。对 KD，我们遵循 [SST+20]，把蒸馏损失与真实回复上的语言建模损失按混合比例 $0.5$ 混合。各基线的检查点按验证集上的 Rouge-L [Lin04] 分数选取，因为如先前工作 [WMA+22, OWJ+22] 所述，我们也发现 Rouge-L 与人类评判更相关。

**MiniLLM。** 如第 2.3 节所述，MiniLLM 的训练有两个阶段，类似于来自人类反馈的强化学习（RLHF；OWJ+22）。

- **阶段 1：** 我们在指令—回复训练集 $\mathcal{D}$ 上微调学生模型，作为后续 MiniLLM 训练的起点。使用对应 SFT w/o KD 基线的最佳学习率与批大小，微调 3 个 epoch。注意与 SFT w/o KD 基线不同，本阶段选取验证损失最低的检查点，而非 Rouge-L 分数。
- **阶段 2：** 我们从阶段 1 的模型出发，按算法 1（正文中的 Algorithm 2.3）继续训练，在所有情形下使用学习率 $5\times10^{-6}$、小批大小 64。训练与验证集与阶段 1 相同。与 [OWJ+22] 类似，我们一次收集 256 条句子，并在做策略优化时采用 4 个内循环 epoch。裁剪率 $\epsilon$ 设为 $0.2$，模型最大长度设为 $512$。从 $q_\theta$ 采样时使用温度 $=1$。我们训练 5000 步，并用验证集上的 Rouge-L 分数选取最终检查点。实验基于 NVIDIA V100 32G GPU。把 LLaMA-7B 从 LLaMA-13B 蒸馏出来，在 16 块 GPU 上耗时不到 10 小时。

### B.2 自动评估细节（Automatic Evaluation Details）

评估时，我们从各模型采样回复：温度 $=1$，最大长度限制 $512$，随机种子 $[10,20,30,40,50]$。与 [TGZ+23] 类似，我们采用图 9 所示的提示包装，把每条指令—回复对转换为一个句子。对 GPT-4 反馈，我们使用图 10 的提示，并设温度 $=0.7$。对第 3.3 节「校准」段落中的分类任务，我们用图 11 与图 12 的模板提示模型做零样本文本分类。

> **图 9**：训练与评估用的提示包装（prompt wrapper）。

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:
```

> **图 10**：GPT-4 评估提示。

```
We would like to request your feedback on the performance of two AI assistants in response
to the user instruction and input displayed above.
Please rate the helpfulness, relevance, accuracy, and level of detail of their responses. Each
assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better
overall performance.
Please first output a single line containing only two values indicating the scores for Assistant
1 and 2, respectively. The two scores are separated by a space.
In the subsequent line, please provide a comprehensive explanation of your evaluation,
avoiding any potential bias and ensuring that the order in which the responses were pre-
sented does not affect your judgment.
```

> **图 11**：SST2 的零样本文本分类提示。

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
Determine the sentiment of the input sentence. Please respond as positive or negative.
### Input:
{sentence}
### Response:
```

> **图 12**：BoolQ 的零样本文本分类提示。

```
Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
Read the input passage and answer the question: {question}? Your answer should be “Yes”
or “No”.
### Input:
{passage}
### Response:
```

### B.3 人工评估细节（Human Evaluation Details）

遵循 [PLH+23]，我们用 SelfInst [WKM+23] 做人工评估。我们随机采样 50 条提示，因为发现更多提示对结果影响不大。我们请标注者比较基线模型与 MiniLLM 生成的回复，并决定哪条回复更好，或两者都没有明显更好。注意标注者看不到回复来自哪个模型。呈现给标注者的界面见图 13。

> **图 13**：人工评估界面示例（含指令、输入与两条待比较答案，标注者选择 1/2/平局）。原文也用作训练/评估提示包装的展示。

```
Below is an instruction that describes a task, paired with an input that provides further
context. Write a response that appropriately completes the request.
### Instruction:
Desk jobs require writing a lot of emails, so it isn’t surprising we get tired of repeating
ourselves. Come up with several synonyms for the given word.
### Input:
Sincerely
### Response:
##### Answer #1 #####
Fondly, affectionately, lovingly, tenderly, honestly, truly, faithfully, devotedly, passionately
##### Answer #2 #####
Faithfully, Gullibly, Humbly, Piously, Strangely, Weirdly, Yours truly
1: Answer #1 is better
2: Answer #2 is better
3: Tie
Your choice:
```

### B.4 生成多样性度量细节（Details About Generation Diversity Metrics）

在表 3 中，我们报告测试集上的 distinct 4-grams（Dist-4）与语言建模损失（Loss）。这两个指标的更多细节如下：

- **「Dist-4」** 是一个分数：$N/C$，其中 $N$ 是生成回复中 distinct 4-grams 的数量，$C$ 是 4-grams 总数。它是度量语言模型生成多样性的广泛使用指标 [LGB+16]。在 Dolly 测试集上、5 个随机种子下的 $(N/C)$ 见表 5。表 3 报告 5 个随机种子上的平均值。
- **「Loss」** 是测试集 $\mathcal{D}_{\mathrm{Test}}$ 上的负对数似然损失：$-\sum_{x,y\sim\mathcal{D}_{\mathrm{Test}}}\log q_\theta(y|x)$。它度量对真实数据分布的模态覆盖，因为本质上是真实数据分布与模型输出分布之间的前向 KLD。这与多样性相关——即给定同一上下文、用不同随机种子生成不同回复的能力。

**表 5**：$(N/C)$，其中 $N$ 是生成回复中 distinct 4-grams 数量，$C$ 是 4-grams 总数。报告在 Dolly 测试集上用 5 个随机种子 $[10,20,30,40,50]$ 评估时的数值。

| 模型 | Seed 10 | Seed 20 | Seed 30 | Seed 40 | Seed 50 |
| --- | --- | --- | --- | --- | --- |
| Teacher | 23562 / 23696 | 23653 / 23834 | 24306 / 24488 | 24207 / 24381 | 23803 / 23967 |
| KD | 25889 / 26064 | 24024 / 24197 | 25663 / 25843 | 25611 / 25763 | 26178 / 26339 |
| SeqKD | 25358 / 25519 | 25631 / 25822 | 26190 / 26370 | 25574 / 25748 | 26295 / 26522 |
| MiniLLM | 24187 / 24458 | 25011 / 25272 | 25100 / 25436 | 24067 / 24312 | 25205 / 25519 |

### B.5 曝光偏差分析（Exposure Bias Analysis）

遵循 [AEABC22]，我们用下式计算 ExAccErr：

$$
\mathrm{ExAccErr}(l)=\frac{R(l)-l\,\epsilon(l)}{l\,\epsilon(l)}\times 100\%, \tag{30}
$$

其中 $R(l)$ 是自由运行生成时、在时间步 $l$ 模仿教师分布 $p$ 的累积后悔（accumulated regret）：

$$
R(l)=\sum_{t=1}^{T}\mathbb{E}_{y_{<t}\sim q_\theta(\cdot|x),\,y_t\sim p(\cdot|y_{<t},x)}\log\frac{p(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}, \tag{31}
$$

而 $\epsilon(l)$ 是以从 $p$ 采样的 oracle 上下文为前缀时，$q_\theta$ 与 $p$ 之间的平均逐步误差：

$$
\epsilon(l)=\frac{1}{l}\sum_{t=1}^{T}\mathbb{E}_{y_{<t}\sim p(\cdot|x),\,y_t\sim p(\cdot|y_{<t},x)}\log\frac{p(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}. \tag{32}
$$

直观上，$q_\theta$ 在生成时的后悔由两部分组成：给定 oracle 上下文估计 $p$ 的误差，以及由低质量模型生成前缀造成的误差。前者由 $l\,\epsilon(l)$ 计算，后者反映曝光偏差。因此，ExAccErr 度量仅由曝光偏差造成的相对误差。

---

## 附录 C 补充结果（Additional Results）

### C.1 以 GPT-J 为教师模型（GPT-J as the Teacher Model）

表 6 给出以 GPT-J 为教师、以 GPT-2-760M、GPT-2-1.5B 与 GPT-Neo-2.7B [BLW+21] 为学生时的评估结果。MiniLLM 在多数情形下优于基线。

**表 6：以 GPT-J 为教师时的评估结果。** GPT4 与 R-L 分别表示 5 个随机种子上的平均 GPT-4 反馈分数与 Rouge-L 分数。各模型规模下最佳分数加粗；学生超过教师的分数标以 *。

| 模型 | 方法 | DollyEval GPT4 | DollyEval R-L | SelfInst GPT4 | SelfInst R-L | VicunaEval GPT4 | VicunaEval R-L | S-NI R-L | UnNI R-L |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-J-6B | Teacher | 65.8 | 27.3 | 57.4 | 17.3 | 55.8 | 17.4 | 28.0 | 33.6 |
| GPT-2-760M | SFT w/o KD | 50.7 | 25.4 | 38.3 | 12.4 | 43.1 | 16.1 | 21.5 | 27.1 |
| | KD | 51.6 | 26.7 | 38.9 | 13.4 | 43.4 | 16.4 | 25.9 | 33.2 |
| | SeqKD | 51.4 | 26.0 | 39.2 | 14.0 | 42.0 | 15.3 | 25.5 | 32.5 |
| | MiniLLM | **54.0** | 25.8 | **43.7** | **16.3** | **44.3** | **19.1\*** | **27.1** | **35.5\*** |
| GPT-2-1.5B | SFT w/o KD | 58.4 | **27.6\*** | 42.9 | 14.3 | 48.6 | 16.3 | 27.6 | 34.6\* |
| | KD | 56.5 | 26.6 | 46.0 | 14.5 | 47.2 | 16.5 | 27.6 | 34.9\* |
| | SeqKD | 58.5 | 27.0 | 43.2 | 13.6 | 46.6 | 16.9 | 28.0 | 34.2\* |
| | MiniLLM | **59.6** | 25.9 | **48.5** | **16.6** | **48.9** | **19.4\*** | **28.5\*** | **35.9\*** |
| GPT-Neo-2.7B | SFT w/o KD | 60.7 | 26.8 | 45.4 | 15.8 | 51.5 | 17.0 | 26.5 | 31.6 |
| | KD | 61.5 | 26.7 | 47.0 | 16.0 | 52.1 | 16.9 | 27.2 | 32.7 |
| | SeqKD | 60.8 | 25.6 | 47.2 | 16.2 | 53.0 | 16.9 | 26.1 | 32.9 |
| | MiniLLM | **63.4** | **28.5\*** | **52.5** | **17.1** | **54.1** | **18.6\*** | **29.8\*** | **35.4\*** |

### C.2 基于 OPT 族的教师规模定律（Scaling Law of Teacher based on the OPT family）

我们在图 14 中展示扩大教师模型规模时 MiniLLM 与 SeqKD 的表现。与第 3.3 节观察类似，MiniLLM 始终更好，并能从更大教师蒸馏出更好的学生模型。

> **图 14**：基于 OPT 族的教师规模定律。学生为 OPT-1.3B，教师分别为 OPT 2.7B、6.7B、13B，比较 MiniLLM 与 SeqKD。

### C.3 U-NI 上不同回复长度的表现（Performance of Response Length on U-NI）

按真实回复长度划分的不同 U-NI 子集上的表现见图 15。我们与第 3.3 节有相同观察：在短回复上，所有 KD 方法表现相近；在长回复上，MiniLLM 优于其他方法。

> **图 15**：按真实回复长度划分的 UnNI 不同评估子集上，蒸馏模型相对 SFT 模型的 Rouge-L。

### C.4 更多消融研究（More Ablation Studies）

**教师混入强度 $\alpha$ 的影响。** 在图 16 中，我们绘制 GPT-2-125M、OPT-1.3B 与 LLaMA-7B（教师分别为 GPT-2-1.5B、OPT-13B、LLaMA-13B）在不同教师混入强度 $\alpha$ 下、验证集上的最佳 Rouge-L。$\alpha=0.0$ 表示仅从学生分布采样；$\alpha=1.0$ 表示完全从教师分布采样。我们发现 $\alpha=0.2$ 在不同模型族与规模上通常合适，且更大模型对 $\alpha$ 的选择更稳健。

> **图 16**：教师混入探索中 $\alpha$ 取值对验证 Rouge-L 的影响。更大模型对 $\alpha$ 更稳健。

**加入预训练损失的影响。** 在表 7 中，我们通过比较 MiniLLM 与去掉预训练语料上语言建模损失的变体（w/o $\mathcal{L}_{\mathrm{PT}}$），研究算法中加入预训练损失的影响。我们与 [OWJ+22] 有类似观察：加入预训练损失有助于保持经典 NLP 任务上的能力，同时指令跟随任务表现几乎不变。

**表 7：加入预训练损失的效果。** 「CLS」是 SST2 与 BoolQ 上的平均准确率；「Inst.」是 Dolly、SelfInst 与 Vicuna 上的平均 Rouge-L。

| | | CLS | Inst. |
| --- | --- | ---: | ---: |
| 1.3B | MiniLLM | 70.2 | 52.8 |
| | w/o $\mathcal{L}_{\mathrm{PT}}$ | 65.7 | 53.2 |
| 7B | MiniLLM | 78.8 | 71.2 |
| | w/o $\mathcal{L}_{\mathrm{PT}}$ | 74.3 | 71.1 |

---

## 附录 D 案例（Cases）

我们基于 LLaMA 模型族，在表 8 中给出不同方法蒸馏出的模型所生成的一些案例。提示从 SelfInst 数据集采样。我们发现相对基线，MiniLLM 生成更详细、更准确的回复。

**表 8：SelfInst 数据集上的指令跟随案例。** MiniLLM 更能遵循指令，并提供更详细、准确的回复。去掉第 2.2 节的教师混合采样策略后，蒸馏模型会输出短回复（案例 #1）或简单重复输入（案例 #2）。

| | 内容 |
| --- | --- |
| **案例 #1 指令** | During writing, we added an asterisk for the word that did not come to mind. You will need to provide several examples to demonstrate all the words that can be used in the sentence instead of the asterisk. |
| **输入** | we * to know |
| **SFT w/o KD** | We can use “know” in the sentence instead of “we * to know” |
| **KD** | we went to know / we will go to know / we can go to know / … |
| **SeqKD** | we do to know / we can do to know / … |
| **MiniLLM** | We can use the following words in place of the asterisk: want, have, do, used, ought, desire, need, must |
| **MiniLLM w/o teacher-mixed** | We to know |
| **案例 #2 指令** | Take the title of the product and extract its attributes… Your answer must a list following the format of “attribute: value”. |
| **输入** | SAMSUNG 980 PRO SSD 2TB PCIe NVMe Gen 4 Gaming M.2 Internal Solid State Hard Drive Memory Card, Maximum Speed, Thermal Control, MZ-V8P2T0B |
| **SFT w/o KD** | Attribute: Maximum Speed / Value: 3500 MB/s |
| **KD** | attribute: brand, attribute: color, … |
| **SeqKD** | brand: Samsung, color: black, size: 2TB, … |
| **MiniLLM** | brand: SAMSUNG / color: 980 PRO / capacity: 2TB / interface: PCIe NVME Gen 4 / … |
| **MiniLLM w/o teacher-mixed** | SAMSUNGC 980 Pro SSD 2TB …（近似重复输入） |

> 译者注：表 8 中指令、输入与模型输出按原文保留英文，以便对照案例；说明文字已译为中文。

---

> 译者说明：附录 A–D 已按 arXiv PDF 全文译出（含式 (8)–(32)、表 5–8、图 9–16 说明与提示模板原文）。正文部分未改动。
