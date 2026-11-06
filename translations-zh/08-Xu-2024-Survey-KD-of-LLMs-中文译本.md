> 出处：*A Survey on Knowledge Distillation of Large Language Models*（Xiaohan Xu 等，2024），arXiv:2402.13116，https://arxiv.org/abs/2402.13116 。本译本依据 arXiv v4（2024 年 10 月 21 日）版本译出；图不翻译，仅以一句话说明图的内容；参考文献保留英文原文。标有「译者注」的内容为译者所加，非原文。

# 大语言模型知识蒸馏综述

**Xiaohan Xu¹, Ming Li², Chongyang Tao³, Tao Shen⁴, Reynold Cheng¹, Jinyang Li¹, Can Xu⁵, Dacheng Tao⁶, Tianyi Zhou²**

¹ 香港大学（The University of Hong Kong） ² 马里兰大学（University of Maryland） ³ 微软（Microsoft） ⁴ 悉尼科技大学（University of Technology Sydney） ⁵ 北京大学（Peking University） ⁶ 悉尼大学（The University of Sydney）

{shawnxxh,chongyangtao,hishentao}@gmail.com {minglii,tianyi}@umd.edu ckcheng@cs.hku.hk jl0725@connect.hku.hk

**摘要**——在大语言模型（Large Language Models, LLMs）时代，知识蒸馏（Knowledge Distillation, KD）已成为一种关键方法，用于把领先的专有 LLM（如 GPT-4）的先进能力迁移到 LLaMA、Mistral 等开源模型上。此外，随着开源 LLM 的蓬勃发展，KD 在两方面都发挥着重要作用：一是压缩这些模型，二是让模型以自身为教师、实现自我改进（self-improvement）。本文对 KD 在 LLM 领域中的作用进行了全面综述，着重阐述它在向较小模型传授高级知识方面的关键作用，以及它在模型压缩和自我改进中的用途。本综述围绕三大支柱精心组织：算法（algorithm）、技能（skill）与垂直化（verticalization）——全面考察 KD 的机制、特定认知能力的增强，以及它们在不同领域中的实际意义。尤为重要的是，本综述探讨了数据增强（Data Augmentation, DA）与 KD 之间的相互作用，说明 DA 如何在 KD 框架中成为提升 LLM 性能的强大范式。借助 DA 生成上下文丰富、针对特定技能的训练数据，KD 突破了传统边界，使开源模型能够逼近专有模型所特有的上下文驾驭能力、伦理对齐以及深层语义洞察。本文旨在为研究者和从业者提供一份有洞见的指南，详细概述当前知识蒸馏的方法并提出未来的研究方向。通过弥合专有 LLM 与开源 LLM 之间的差距，本综述凸显了构建更易获取、更高效、更强大的 AI 解决方案的潜力。最重要的是，我们坚定主张遵守规范 LLM 使用的法律条款，确保 LLM 的 KD 以合乎伦理、合法的方式应用。相关 GitHub 仓库见 https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs 。

**关键词（Index Terms）**——大语言模型，知识蒸馏，数据增强，技能蒸馏，监督微调

---

## 1 引言（Introduction）

在不断演进的人工智能（AI）版图中，GPT-3.5（Ouyang et al., 2022）、GPT-4（OpenAI et al., 2023）、Gemini（Team et al., 2023）和 Claude² 等专有¹（proprietary）大语言模型（LLM）已成为开创性的技术，重塑了我们对自然语言处理（NLP）的理解。这些模型以庞大的规模和复杂性为特征，开启了从生成类人文本到提供复杂问题求解能力的全新可能。这些 LLM 的核心意义在于其涌现能力（emergent abilities）（Wei et al., 2022a,b; Xu et al., 2024a）——即模型表现出超越其显式训练目标的能力，使其能以惊人的熟练度应对各种各样的任务。这些模型在理解与生成方面表现出色，推动了从创意生成到复杂问题求解的各类应用（OpenAI et al., 2023; Liang et al., 2022）。这些模型的潜力远不止当前的应用，它们有望彻底变革各行各业、增强人类创造力，并重新定义我们与技术的交互方式。

> 脚注 1：为简单起见，我们用“专有”同时指代 GPT-4 这类功能全面但闭源的 LLM，以及 LLaMA-2-70B 这类开源但规模巨大的 LLM——它们都以大量参数承载着丰富的知识。
>
> 脚注 2：https://www.anthropic.com/claude-in-slack

尽管 GPT-4 和 Gemini 等专有 LLM 能力卓越，它们也并非没有缺点，尤其是与开源模型的优势相比时。一个显著的缺点是其可获取性有限且成本较高（OpenAI et al., 2023）。这些专有模型通常伴随着高昂的使用费用和受限的访问权限，使个人和较小的组织难以使用。在数据隐私与安全方面（Wu et al., 2023a），使用这些专有 LLM 往往需要把敏感数据发送到外部服务器，从而引发对数据隐私与安全的担忧。这一点对于处理机密信息的用户尤为关键。此外，专有 LLM 的通用设计虽然强大，却未必总能契合小众应用的特定需求。因此，可获取性、成本和适应性方面的限制，给充分发挥专有 LLM 的潜力带来了重大挑战。

与专有 LLM 相比，LLaMA（Touvron et al., 2023）和 Mistral（Jiang et al., 2023a）等开源模型具有若干显著优势。开源模型的首要好处之一是其可获取性和适应性。没有许可费用或限制性使用政策的约束，这些模型更容易被从个人研究者到小型组织的更广泛用户所使用。这种开放性营造了更具协作性和包容性的 AI 研究环境，鼓励创新和多样化的应用。此外，开源 LLM 的可定制性使其能提供更有针对性的解决方案，满足通用大规模模型可能无法满足的特定需求。

然而，开源 LLM 也有其自身的缺点，主要源于其规模和资源相较专有模型相对有限。最显著的限制之一是模型规模较小，这往往导致其在包含大量指令的真实世界任务上表现较差（Zheng et al., 2023a）。参数较少的模型可能难以捕获 GPT-4 等更大模型所蕴含的知识深度与广度。此外，这些开源模型的预训练投入通常较少。投入减少可能导致预训练数据范围更窄，从而可能限制模型对多样化或专业化主题的理解和处理（Liang et al., 2022; Sun et al., 2024a）。再者，由于资源限制，开源模型通常经历的微调步骤也更少。微调对于针对特定任务或行业优化模型性能至关重要，缺乏微调会削弱模型在专业应用中的效果。当这些模型与经过高度微调、往往被定制为能在众多复杂场景中表现出色的专有 LLM 相比时，这一限制尤为明显（OpenAI et al., 2023）。

首先，正是认识到专有 LLM 与开源 LLM 之间的差距，KD 技术作为弥合这些模型间性能鸿沟的手段迅速兴起（Gou et al., 2021; Gupta and Agrawal, 2022）。在这一语境下，知识蒸馏是指利用 GPT-4 或 Gemini 等领先专有模型的更先进能力作为指导框架，来增强开源 LLM 的能力。这个过程类似于把一位技艺高超的教师（teacher）的“知识”传授给学生（student），学生（例如开源 LLM）学习模仿教师（例如专有 LLM）的表现特征。与传统的知识蒸馏算法（Gou et al., 2021）相比，数据增强（DA）（Feng et al., 2021）已成为实现 LLM 知识蒸馏的一种流行范式：用少量种子知识（seed knowledge）提示 LLM，针对特定技能或领域生成更多数据（Taori et al., 2023）。其次，KD 仍保留着压缩 LLM 的基本作用，使模型在不显著损失性能的前提下更加高效（Gu et al., 2024; Agarwal et al., 2024）。最近，以开源 LLM 作为自身的教师以实现自我改进的策略已成为一种很有前景的方法，能显著增强其能力（Yuan et al., 2024a; Chen et al., 2024a）。图 1 展示了 KD 在 LLM 语境中扮演的这三种关键角色。

> **图 1**（图不翻译）：示意 KD 在 LLM 中的三种方向——① 从闭源 LLM 向开源 LLM“提升能力”（Advance），② 从开源 LLM 向更小的语言模型“压缩”（Compress），③ 开源 LLM 的“自我改进”（Self-Improvement）。
> 原图注：KD 在 LLM 中扮演三种关键角色：1）首要的是增强能力；2）提供传统的压缩以提升效率；3）通过自生成知识实现自我改进这一新兴趋势。

知识蒸馏的一个关键方面是增强各种技能，例如高级的上下文遵循能力（如上下文学习（in-context learning）（Huang et al., 2022a）和指令遵循（instruction following）（Taori et al., 2023））、与用户意图更好地对齐（如人类价值观/原则（Cui et al., 2023a），以及思维链（chain-of-thought, CoT）之类的思维模式（Mukherjee et al., 2023）），以及 NLP 任务专精（如语义理解（Ding et al., 2023a）和代码生成（Chaudhary, 2023））。这些技能对于 LLM 被期望完成的各类应用至关重要，从日常对话到专业领域中的复杂问题求解。例如，在医疗健康（Wang et al., 2023a）、法律（LAW, 2023）或科学（Zhang et al., 2024）等对准确性和特定领域知识要求极高的垂直领域，知识蒸馏使开源模型能够通过向在这些领域经过大量训练和微调的专有模型学习，显著提升自身表现。

在 LLM 时代，知识蒸馏带来的好处是多方面的、变革性的（Gu et al., 2024）。借助一系列蒸馏技术，专有模型与开源模型之间的差距被显著缩小（Chiang et al., 2023; Xu et al., 2023a），甚至被填平（Zhao et al., 2023a）。这一过程不仅精简了计算需求，还提升了 AI 运行的环境可持续性，因为开源模型能以更少的计算开销变得更加强大。此外，知识蒸馏促进了更易获取、更公平的 AI 格局：较小的机构和个人研究者也能获得最先进的能力，从而鼓励更广泛的参与和 AI 进步的多样性。这种技术的民主化带来了更稳健、更通用、更易获取的 AI 解决方案，推动了各行业和研究领域的创新与发展。

之所以迫切需要一份关于 LLM 知识蒸馏的全面综述，是因为 AI 格局的快速演变（OpenAI et al., 2023; Team et al., 2023）以及这些模型日益增长的复杂性。随着 AI 不断渗透到各个领域，能够高效、有效地把知识从专有 LLM 蒸馏到开源 LLM，已不仅是一种技术追求，更成为一种现实必需。这一需求源于对更易获取、更具成本效益、更具适应性、能服务多样应用和用户的 AI 解决方案日益增长的需求。该领域的一份综述对于整合知识蒸馏当前的方法、挑战与突破至关重要。它可以成为研究者和从业者的指路明灯，引导他们把复杂的 AI 能力蒸馏为更易管理、更易获取的形式。此外，这样的综述还能照亮前进的道路，找出当前技术中的空白并提出未来研究方向。

> **图 2**（图不翻译）：本综述的总体框架图——左侧是教师 LLM（GPT-4、Claude、Gemini 等）在种子知识与技能/领域引导下进行知识引出（3.1 节：标注、扩展、数据整理、特征、反馈、自我知识），中间是蒸馏算法（3.2 节：监督微调、散度与相似度、强化学习、排序优化）训练学生模型（Llama、Vicuna 等），右侧列出技能（第 4 节）与垂直领域（第 5 节）。
> 原图注：本综述关于大语言模型知识蒸馏的概览。注意图中把“Section”缩写为“Sec.”。RM_S(·) 表示学生奖励模型。①②③④ 表示 LLM 知识蒸馏的各个步骤。

**综述结构。** 本综述的其余部分由若干全面的章节组成，每一节都深入探讨 LLM 领域内知识蒸馏的多个方面。在本引言之后，§2 给出知识蒸馏的基础概述，比较传统技术与 LLM 时代新兴的技术，并强调数据增强（DA）在其中的作用。§3 深入探讨从教师 LLM 中引出知识的方法以及核心蒸馏算法，考察从监督微调到涉及散度与相似度、强化学习和排序优化等更复杂策略的方法。接着，§4 聚焦技能蒸馏，探讨如何增强学生模型以改善上下文理解、与用户意图的对齐，以及在各类 NLP 任务上的表现，包括对自然语言理解（NLU）、自然语言生成（NLG）、信息检索、推荐系统和文本生成评估的讨论。在 §5 中，我们进入特定领域的垂直蒸馏，展示知识蒸馏技术如何应用于法律、医疗健康、金融和科学等专业领域，说明这些方法的实际意义和变革性影响。本综述在 §6 中提出开放问题，指出知识蒸馏研究中当前存在的挑战与空白，它们为未来工作提供了机会。最后，§7 的结论与讨论综合所得洞见，反思其对更广泛的 AI 与 NLP 研究社区的意义，并提出未来研究方向。图 2 展示了本综述的概览。

## 2 概述（Overview）

### 2.1 与传统方法的比较（Comparing Traditional Recipe）

在 AI 和深度学习（DL）领域，知识蒸馏的概念是指把知识从一个大型、复杂的模型（教师）迁移到一个更小、更高效的模型（学生）的过程（Gou et al., 2021）。这一技术对于缓解在实际应用中部署大规模模型时面临的计算需求和资源限制挑战至关重要。

从历史上看，在 LLM 时代之前，知识蒸馏技术主要集中于把知识从复杂、往往笨重的神经网络迁移到更紧凑、更高效的架构上（Sanh et al., 2019; Kim and Rush, 2016）。这一过程很大程度上是由在资源受限环境（如移动设备或边缘计算平台，其算力和内存有限）中部署机器学习模型的需求所驱动的。其重点主要放在针对单一任务的特定（ad-hoc）神经架构选择和训练目标上。这些早期方法通常是训练一个较小的学生网络去模仿较大的教师网络的输出，常用技术如软目标训练（soft target training），即学生从教师经过“软化”的 softmax 输出中学习。关于 AI 和 DL 中通用知识蒸馏技术的更多细节，请参阅综述（Gou et al., 2021）。

> **图 3**（图不翻译）：LLM 知识蒸馏的分类树，从“知识（Knowledge）”“蒸馏（Distillation）”两类 KD 算法，到“技能蒸馏（Skill Distillation）”与“垂直化蒸馏（Verticalization Distillation）”，每个叶节点列出代表性工作（如 Alpaca、Vicuna、WizardLM、MiniLLM、GKD、Orca、Toolformer、LLaVA 等）。
> 原图注：大语言模型知识蒸馏的分类体系。垂直化蒸馏的详细分类见图 7。

> 译者注：Hinton 2015 那篇论文中的“软目标”指教师模型在较高温度下输出的概率分布；它比只有一个正确答案的“硬标签”包含更多类别间相似性的信息。本综述说的“传统 KD”主要就是这种做法。

相比之下，LLM 的出现彻底改变了知识蒸馏的格局。当前 LLM 时代的知识蒸馏，已把重点从单纯的架构压缩转向知识的引出与迁移（knowledge elicitation and transfer）（Taori et al., 2023; Chaudhary, 2023; Tunstall et al., 2023）。这种范式转变很大程度上源于 GPT-4、Gemini 等 LLM 所拥有的广博而深厚的知识。而且 LLM 的参数无法获取，这使得很难用剪枝（pruning）（Han et al., 2016）或量化（quantization）（Liu et al., 2023a）技术来压缩它们。与早期以复制教师模型输出行为或缩小模型规模为目标不同，当前基于 LLM 的知识蒸馏关注的是引出这些模型所拥有的特定知识。

这种现代方法的关键在于启发式的、精心设计的提示（prompt），用于从 LLM 中引出特定知识（Ding et al., 2023b）或能力（Chaudhary, 2023）。这些提示经过精心构造，以挖掘 LLM 在各个领域的理解和能力，从自然语言理解（He et al., 2023a）到推理（Hsieh et al., 2023）和问题求解（Qiao et al., 2024）等更复杂的认知任务。把提示作为知识引出的手段，为蒸馏提供了一种更灵活、更动态的方式。它允许更有针对性地提取知识，聚焦于感兴趣的特定技能或领域。这种方法在利用 LLM 的涌现能力（即模型展现出超越其显式训练目标的能力）方面尤其有效。

此外，这一时期的知识蒸馏还强调迁移更抽象的品质，例如推理模式（Mitra et al., 2023）、偏好对齐（Cui et al., 2023a）和价值对齐（Sun et al., 2024b）。这与早期聚焦于输出复制（Taori et al., 2023）形成鲜明对比，表明正在转向更整体、更全面的认知能力迁移。当前的技术不仅涉及复制输出，还包括模仿教师模型的思维过程（Mitra et al., 2023）和决策（Asai et al., 2023）模式。这涉及思维链提示等复杂策略——训练学生模型学习教师的推理过程，从而增强其问题求解和决策能力。

### 2.2 与数据增强（DA）的关系（Relation to Data Augmentation）

在 LLM 时代，数据增强（DA）（Wang et al., 2022a; Ye et al., 2022）成为知识蒸馏过程中不可或缺的关键范式。与复述（paraphrasing）（Gangal et al., 2022）或回译（back-translation）（Longpre et al., 2019）等主要以某种机械方式扩充训练数据集的传统 DA 技术不同，LLM 语境下的 DA 着重于生成针对特定领域和技能定制的、新颖且上下文丰富的训练数据。

在 LLM 中，DA 与 KD 之间的关系既是共生的，也是基础性的。借助一组种子知识，KD 利用 DA 提示 LLM 产生显式的数据，这些数据封装了特定技能或领域专长（Chaudhary, 2023; West et al., 2022）。这种方法作为弥合专有模型与开源模型之间知识和能力差距的有力机制而脱颖而出。通过 DA，LLM 被提示去创建有针对性的高质量数据集，这些数据集不仅在数量上更大，而且在多样性和针对性上也很丰富。这种方法使蒸馏过程更加有效，确保被蒸馏的模型不仅复制教师模型的输出行为，还体现其深层的理解和认知策略。

DA 起到“力量倍增器”的作用，使被蒸馏的模型能够获得并完善那些原本需要指数级更大的数据集和计算资源才能获得的能力。它促进了更有效的知识迁移，关注学习的质的方面，而非量的扩张。在 KD 过程中对 DA 的这种策略性运用，标志着在利用 LLM 力量方面正向更高效、更可持续、更易获取的方式进行关键转变。它使开源模型能够逼近专有模型所特有的上下文驾驭能力、伦理对齐和深层语义洞察，从而让先进 AI 能力的获取更加民主化，并在更广泛的应用和用户中促进创新。

### 2.3 综述范围（Survey Scope）

在前文讨论的基础上，本综述旨在按照图 3 所示精心构建的分类体系，全面探索 LLM 语境下的知识蒸馏版图。本综述的范围通过三个主要方面来划定：KD 算法、技能蒸馏和垂直化蒸馏。每个方面都涵盖一系列子主题和方法。需要指出的是，KD 算法为技能蒸馏和垂直化蒸馏提供了技术基础。

**KD 算法。** 这一部分聚焦知识蒸馏的技术基础和方法。它深入探讨了从教师模型（例如专有 LLM）构建知识、并将这些知识整合进学生模型（例如开源 LLM）所涉及的过程。在“知识”这一主题下，我们深入研究标注（labeling）（Hsieh et al., 2023）、扩展（expansion）（Taori et al., 2023）、整理（curation）（Gunasekar et al., 2023）、特征理解（feature understanding）（Agarwal et al., 2024）、反馈机制（feedback）（Tunstall et al., 2023）以及自我知识生成（self-knowledge）（Wang et al., 2022a）等策略。这一探索旨在揭示识别、扩展和整理知识以实现有效蒸馏的各种方式。“蒸馏”小节考察了多种学习方法，如监督微调（supervised fine-tuning, SFT）（Wang et al., 2022a）、散度最小化（divergence minimization）（Agarwal et al., 2024）、强化学习技术（Cui et al., 2023a）以及排序优化策略（rank optimization）（Tunstall et al., 2023）。这些技术共同展示了 KD 如何使开源模型从专有模型中获取知识。

**技能蒸馏。** 这一方面考察通过 KD 得到增强的具体能力。它包括对上下文遵循（context following）（Taori et al., 2023; Luo et al., 2023c）的详细讨论，子主题包括指令遵循和检索增强生成（retrieval-augmented generation, RAG）能力。在对齐（alignment）方面（Mitra et al., 2023; Tunstall et al., 2023），本综述研究思维模式、人设/偏好建模（persona/preference modeling）以及价值对齐。“智能体（agent）”类别深入探讨工具使用（Tool Using）和规划（Planning）等技能。NLP 任务专精（Dai et al., 2023a; Jung et al., 2023; Chaudhary, 2023）则从自然语言理解（NLU）、自然语言生成（NLG）、信息检索、推荐系统、文本生成评估和代码生成等角度加以审视。最后，本综述讨论多模态（multi-modality）（Liu et al., 2023e; Zhao et al., 2023b），探讨 KD 如何增强 LLM 整合多种输入形式的能力。

**垂直化蒸馏。** 这一部分评估 KD 在不同垂直领域中的应用，深入说明被蒸馏的 LLM 如何针对法律（LAW, 2023）、医疗健康（Wang et al., 2023a）、金融（Zhang and Yang, 2023）、科学（Zhang et al., 2024）等专业领域进行定制。这一探索不仅展示了 KD 技术的实际意义，也凸显了它们对特定领域 AI 解决方案的变革性影响。

通过这些方面，本综述对 LLM 中的 KD 进行了全面分析，引导研究者和从业者了解这一快速发展领域中的方法、挑战与机遇。

**声明。** 本综述是我们为全面、深入地概述应用于 LLM 的知识蒸馏技术所做的认真努力，重点关注算法、技能增强和特定领域应用。鉴于该领域广阔且发展迅速，尤其是学术界普遍采用从训练数据中引出知识的做法，我们承认本文可能未能涵盖每一项相关研究或进展。尽管如此，本文努力介绍知识蒸馏的基础范式，重点阐述关键方法及其在一系列应用中的影响。

### 2.4 LLM 时代的蒸馏流程（Distillation Pipeline in LLM Era）

> **图 4**（图不翻译）：通用蒸馏流程示意——“技能/领域”引导（steer）教师 LLM，“种子知识”驱动（drive）教师生成“生成知识”，再按“学习目标”训练（train）学生模型；左半部分对应“知识引出”，右半部分对应“蒸馏算法”。
> 原图注：从大语言模型向学生模型蒸馏知识的通用流程示意。

LLM 的通用蒸馏流程是一个结构化、有条理的过程，旨在把知识从复杂的教师模型迁移到较简单的学生模型。这一流程对于在更易获取、更高效的开源模型中利用 GPT-4 或 Gemini 等模型的先进能力不可或缺。该流程大体可分为四个不同阶段，每个阶段都对知识蒸馏的成功起着关键作用。示意见图 4，更详细的流程也可见图 2。

**I. 目标技能或领域引导教师 LLM。** 第一阶段是把教师 LLM 引向某个特定的目标技能或领域。这是通过精心设计的指令或模板来实现的，用以引导 LLM 的关注点。这些指令旨在引出能体现 LLM 在特定方面（无论是医疗健康或法律之类的专业领域，还是推理或语言理解之类的技能）熟练程度的回答。

**II. 以种子知识作为输入。** 一旦确定了目标方向，下一步就是向教师 LLM 输入种子知识。这些种子知识通常由一个小数据集或与要从教师 LLM 中引出的技能或领域知识相关的特定数据线索组成。它起到催化剂的作用，促使教师 LLM 基于这些初始信息生成更丰富、更详细的输出。种子知识至关重要，因为它为教师模型提供了一个基础，教师模型可以在此之上构建和扩展，从而产生更全面、更深入的知识样例。

**III. 生成蒸馏知识。** 针对种子知识和引导指令，教师 LLM 生成知识样例。这些样例主要以问答（QA）对话或叙述性解释的形式出现，与 LLM 的自然语言处理/理解能力相契合。在某些特殊情况下，输出也可能包括 logits 或隐藏特征（hidden features），不过由于这类数据形式的复杂性和特殊要求，这种情况较少见。生成的知识样例构成了蒸馏知识的核心，封装了教师 LLM 的高级理解与技能。

**IV. 用特定学习目标训练学生模型。** 最后阶段是利用生成的知识样例训练学生模型。训练由一个与学习目标相一致的损失函数引导。损失函数量化学生模型在复制或适应教师模型知识方面的表现。通过最小化该损失，学生模型学会模仿教师的目标技能或领域知识，从而获得相似的能力。这一过程包括迭代地调整学生模型的参数，以减小其输出与教师模型输出之间的差异，确保知识的有效迁移。

本质上，上述四个阶段可以抽象为两个公式。第一个公式表示引出知识的过程：

$$
\mathcal{D}_I^{(\mathrm{kd})}=\{\mathrm{Parse}(o,s)\mid o\sim p_T(o\mid I\oplus s),\ \forall s\sim\mathcal{S}\}, \tag{1}
$$

其中 $\oplus$ 表示把两段文本融合（拼接）在一起，$I$ 表示用于引导 LLM 并引出知识的、针对某任务、技能或领域的指令或模板，$s\sim\mathcal{S}$ 表示种子知识中的一个样例，LLM 可以在其基础上探索并生成新的知识，$\mathrm{Parse}(o,s)$ 表示从教师 LLM 的输出 $o$（某些情况下还要加上输入 $s$）中解析出蒸馏样例（例如 $(x,y)$），$p_T$ 表示参数为 $\theta_T$ 的教师 LLM。给定为蒸馏构建的数据集 $\mathcal{D}_I^{(\mathrm{kd})}$，我们随后将学习目标定义为

$$
\mathcal{L}=\sum_I \mathcal{L}_I\big(\mathcal{D}_I^{(\mathrm{kd})};\theta_S\big), \tag{2}
$$

其中 $\sum_I$ 表示可能有多个任务或技能被蒸馏进同一个学生模型，$\mathcal{L}_I(\cdot;\cdot)$ 表示某个特定的学习目标，$\theta_S$ 是学生模型的参数。

> 译者注：可以把式（1）理解为“造数据”：用指令 $I$ 加上种子样例 $s$ 去问教师，把教师的回答整理成训练样本；式（2）理解为“用数据训练学生”：对每类任务/技能的数据各算一个损失，加起来一起优化学生参数 $\theta_S$。

在探讨了蒸馏流程以及 LLM 时代知识蒸馏的基础概念之后，我们现在把注意力转向这一时期中受到关注的具体算法。

## 3 知识蒸馏算法（Knowledge Distillation Algorithms）

本节梳理知识蒸馏的过程。根据第 2.4 节，它分为两个主要步骤：“知识（Knowledge）”，关注从教师 LLM 中引出知识（式 1）；以及“蒸馏（Distillation）”，关注把这些知识注入学生模型（式 2）。我们将在后续各小节中详细阐述这两个过程。

### 3.1 知识（Knowledge）

本小节关注从教师 LLM 中引出知识的方法。根据获取知识的方式，我们把它们分为标注（Labeling）、扩展（Expansion）、数据整理（Data Curation）、特征（Feature）、反馈（Feedback）和自我知识（Self-Knowledge）。图 5 展示了这些知识引出方法的示意。

#### 3.1.1 标注（Labeling）

标注知识是指以给定输入 $x$ 作为种子知识，按照指令 $I$ 或示例（demonstrations）$c$，用教师 LLM 为其标注输出 $y$，其中 $c=(x_1,y_1),\dots,(x_n,y_n)$。这种从教师 LLM 中引出知识的方法简单而有效，已被广泛应用于各种任务和应用。它只需要收集一个输入数据集并输入 LLM，即可得到所需的生成结果。此外，$y$ 的生成可以通过预先定义的 $I$ 和 $c$ 来控制。这一过程可以表述为：

$$
\mathcal{D}^{(\mathrm{lab})}=\{x,y\mid x\sim\mathcal{X},\ y\sim p_T(y\mid I\oplus c\oplus x)\}. \tag{3}
$$

输入 $x$ 可以来自现有的 NLP 任务数据集，它们是蒸馏工作的典型数据来源。许多工作都试图把强大的 LLM 作为教师，为一系列任务中的数据集样本进行标注。例如，在自然语言理解方面，有工作用 LLM 对文本进行分类（Gilardi et al., 2023; Ding et al., 2023a; He et al., 2023a）；在自然语言生成方面，LLM 帮助生成输出序列（Hsieh et al., 2023; Jung et al., 2023; Wang et al., 2021b）。文本生成评估任务利用 LLM 标注评估结果（Li et al., 2024b; Wang et al., 2023b），推理任务则利用 LLM 标注思维链（CoT）解释（Hsieh et al., 2023; Li et al., 2022; Ho et al., 2023; Magister et al., 2023; Fu et al., 2023; Ramnath et al., 2023; Li et al., 2023d; Liu et al., 2023g），等等。当前许多工作并不专注于特定任务，而是基于指令来标注输出，从而教会学生模型以遵循指令的更灵活方式解决任务。由各种 NLP 任务组成、并配以指令模板的集合，是 $x$ 的宝贵输入来源。例如，FLAN-v2 集合（Longpre et al., 2023）提供了大量公开可用、带指令的任务集，在 Orca 中这些任务被教师 LLM 生成的回复所标注（Mukherjee et al., 2023; Mitra et al., 2023）。这些 NLP 任务的指令是由预定义模板构建的，缺乏多样性，并且可能与人类自然提出的查询存在差距。人类与聊天模型之间的真实对话（如 ShareGPT）则提供了大规模数据，其中包含真实查询以及由强大 LLM 标注的生成结果。此外，Xu et al.（2023b）和 Anand et al.（2023）对从 Quora 和 Stack Overflow 等论坛中采样的真实问题进行了标注。

此外，标注过程可以由指令 $I$ 或示例 $c$ 来引导。用于引导标注的一种常用指令类型是思维链（CoT）提示（Hsieh et al., 2023; Fu et al., 2023; Magister et al., 2023）。Mukherjee et al.（2023）添加了多种系统消息（例如“You must generate a detailed and long answer.”（你必须生成一个详细而冗长的回答）或“explain like I’m five, think step-by-step”（像给五岁孩子讲解那样解释，一步一步思考））来引出丰富的信号。Yue et al.（2023a）和 Chenglin et al.（2023）标注了思维链（CoT）与思维程序（program-of-thought, PoT）推理依据（rationale）的混合知识。Xu et al.（2023b）提出了一种自我对话（self-chat）技术：由两个教师 LLM 模拟真实对话，为来自 Quora 和 Stack Overflow 的问题生成多轮对话。

#### 3.1.2 扩展（Expansion）

虽然标注方法简单有效，但它也面临一定的局限。首要的是，它受限于输入数据的规模和多样性。在真实世界的应用中，特别是涉及用户对话的应用，还存在对所涉数据隐私的担忧。为了解决这些局限，人们提出了各种扩展方法（Wang et al., 2022a; Taori et al., 2023; Chaudhary, 2023; Si et al., 2023; Ji et al., 2023a; Luo et al., 2023b,a; Wu et al., 2023c; Sun et al., 2024b; Xu et al., 2023a; Guo et al., 2023c; Rozière et al., 2023; West et al., 2022）。这些方法以示例作为种子知识，旨在通过上下文学习扩展出大规模、多样化的数据。

这些扩展方法的一个关键特点是利用 LLM 的上下文学习能力，生成与所提供的示例 $c$ 相似的数据。在标注方法中，输入 $x$ 是从现有数据集中采样的；与之不同，在扩展方法中，$x$ 和 $y$ 都由教师 LLM 生成。这一过程可以表述为：

$$
\mathcal{D}^{(\mathrm{exp})}=\{(x,y)\mid x\sim p_T(x\mid I\oplus c),\ y\sim p_T(y\mid I\oplus x)\}. \tag{4}
$$

> **图 5**（图不翻译）：用示意图分别画出六种从教师 LLM 引出知识的方式（标注、扩展、数据整理、特征、反馈、自我知识）中，输入、示例、元信息、教师与学生之间的数据流向。
> 原图注：从教师 LLM 引出知识的不同方法示意。标注：教师根据输入生成输出；扩展：教师通过上下文学习生成与给定示例相似的样本；数据整理：教师根据元信息（如某个主题或实体）合成数据；特征：把数据输入教师并提取其内部知识，如 logits 和特征；反馈：教师对学生的生成结果给出反馈，如偏好、纠正、对困难样本的扩展等；自我知识：学生先生成输出，然后对其进行高质量过滤，或由学生自己进行评估。

在这一公式中，$x$ 和 $y$ 表示由教师 LLM 生成的新的输入-输出对。输入 $x$ 基于一组输入-输出示例 $c$ 生成。随后，在指令 $I$ 的引导下，针对新输入 $x$ 生成输出 $y$。注意，示例既可以是预先定义的，也可以通过加入新生成的样本而动态更新。

扩展技术已被广泛用于从教师 LLM 中提取大量的指令遵循知识。Wang et al.（2022a）首先提出了一种迭代自举（bootstrapping）方法 Self-Instruct，利用 LLM 根据从 175 条人工编写指令中采样的若干示例，生成大量多样的指令。新生成的指令随后被加回初始指令池，使后续的扩展迭代受益。随后，Taori et al.（2023）把这种扩展方法应用到更强大的教师 LLM text-davinci-003 上，蒸馏出 52K 条高质量数据。为了提高扩展过程中的多样性和覆盖面，Wu et al.（2023c）和（Sun et al., 2024b）提示教师 LLM 生成与某些特定主题对应的指令。Xu et al.（2023a）提出了 Evol-Instruct 方法，从两个维度扩展指令：难度（例如把问题改写得更复杂）和多样性（例如生成更多长尾指令）。这种 Evol-Instruct 方法与领域无关，已被用于扩展编程（Luo et al., 2023a）和数学（Luo et al., 2023b）方面的蒸馏。此外，扩展方法可以用相似样本显著扩充 NLP 任务数据集，从而提升任务性能。例如，AugGPT（Dai et al., 2023a）利用教师 LLM 把训练样本中的每个句子改写成多个概念相似但语义不同的样本，以提升分类性能。类似地，TDG（He et al., 2023b）提出了目标数据生成（Targeted Data Generation, TDG）框架，该框架自动识别数据中具有挑战性的子群，并通过上下文学习用 LLM 为这些子群生成新样本。

总之，扩展方法利用 LLM 在上下文学习方面的优势，生成同时包含输入和输出、更加多样和广泛的数据集。然而，生成数据的质量和多样性在很大程度上依赖于教师 LLM 和初始种子示例。这种依赖可能导致数据集带有来自 LLM 的固有偏差（Yu et al., 2023a; Wei et al., 2023），以及同质化问题——生成结果最终可能趋于相似，从而限制了该方法所追求的多样性（Ding et al., 2023b）。此外，扩展过程可能会不经意间放大种子数据中存在的任何偏差。

> 译者注：Alpaca 就是“扩展”方法的典型例子：用 175 条人工写的种子指令，让 text-davinci-003 仿写出 5.2 万条指令和回答，再用它们微调 LLaMA-7B。

#### 3.1.3 数据整理（Data Curation）

在从 LLM 进行知识蒸馏时，对高质量、可扩展数据生成的追求催生了数据整理方法。这种方法是针对标注和扩展两种方法中观察到的局限而出现的。这些方法产生的数据往往质量参差不齐，数量上也受到限制。在标注方法中，种子知识来自任务数据集，可能带来噪声和脏数据。而在扩展方法中，输入 $x$ 源自种子示例，大量生成时可能导致数据同质化。为了克服这些挑战，数据整理方法以大量元信息（meta-information）作为种子知识，整理出高质量或大规模的数据（Ding et al., 2023b; Gunasekar et al., 2023; Li et al., 2023a; Mar, 2023; Liu et al., 2023d; Wei et al., 2023; Yu et al., 2024; Ye et al., 2022; Gao et al., 2023a; Yang and Nicolai, 2023）。

数据整理的一个显著特点是从零开始合成数据。大量多样的元信息（如主题或知识点）可以融入这一过程，以生成可控的 $x$ 和 $y$。因此，这一过程可以被精细地控制，产生不仅规模大而且质量高的数据集。数据整理的公式可以表示为：

$$
\mathcal{D}^{(\mathrm{cur})}=\{(x,y)\mid x\sim p_T(x\mid I\oplus m),\ y\sim p_T(y\mid I\oplus x)\}. \tag{5}
$$

在这一公式中，$m$ 表示用来引导合成 $x$ 的多样化元信息，$I$ 是引导教师 LLM 生成 $x$ 或 $y$ 的指令。

不同研究的区别主要在于元信息的来源和利用方式。UltraChat（Ding et al., 2023b）有效地展示了通过蒸馏知识整理出既高质量又多样化数据的过程。他们在三个领域收集了大量元信息：关于世界的问题（Questions about the World）、创作与生成（Creation and Generation）以及对现有材料的协助（Assistance on Existing Materials）。例如，在“关于世界的问题”之下，他们探索了“科技（Technology）”和“饮食（Food and Drink）”等 30 个元主题。随后，教师 LLM 利用这些元信息蒸馏出大量指令和对话，达到 150 万条实例的可观规模。UltraChat 以其词汇和主题上的多样性而著称。在这些数据上微调得到的 UltraLLaMA 模型始终优于其他开源模型。另一个值得注意的系列是 phi（Gunasekar et al., 2023; Li et al., 2023a; Mar, 2023），它专注于蒸馏更小但高质量、类似“教科书（textbooks）”的数据集。Phi-1（Gunasekar et al., 2023）尝试在编程领域合成“教科书质量”的数据。他们的方法是在随机主题或函数名的引导下（以增强多样性），从 LLM 中蒸馏出清晰、自成一体、有教学意义且均衡的内容。蒸馏得到的数据包括 10 亿 token 的 Python 教科书（附带自然语言解释和代码片段），以及 1.8 亿 token 带答案的 Python 练习题。值得注意的是，phi-1 模型尽管规模更小，在 HumanEval 和 MBPP 等编程基准上却超过了几乎所有开源模型，而其模型规模小 10 倍、数据集规模小 100 倍。MFTCoder（Liu et al., 2023d）利用数百个 Python 知识点作为元信息，构建了 CodeExercise 数据集。相比之下，Magicoder（Wei et al., 2023）和 WaveCoder（Yu et al., 2024）从开源代码数据集中获取原始代码集合，把它们作为元信息来生成指令数据。在 NLU 任务方面，某些研究（Ye et al., 2022; Gao et al., 2023a; Wang et al., 2021a）探索用标签作为元信息，合成相应样本用于数据增强。类似地，在信息检索任务中，也有工作把文档作为元信息来生成可能的查询，从而构建大规模的检索对（Bonifacio et al., 2022; Meng et al., 2023）。

总之，借助教师 LLM 进行数据整理已成为一种很有前景的技术，能够合成不仅高质量、多样化而且规模大的数据集。phi-1 等模型在专业领域的成功凸显了这种方法的有效性。创建合成数据集的能力将成为 AI 领域一项关键的技术能力和重点关注方向（Li et al., 2023a）。

#### 3.1.4 特征（Feature）

前面讨论的知识引出方法通常应用于强大的黑盒（black-box）模型，这类模型由于需要调用 API，成本高昂且在某种程度上难以复现。相比之下，白盒（white-box）蒸馏为研究者提供了一种更透明、更易获得的途径。它利用教师 LLM 的输出分布、中间特征或激活值，我们把这些统称为特征知识。白盒 KD 方法以往主要针对较小的基于编码器（encoder）的语言模型进行研究，这些模型通常参数少于 10 亿（详见 Gou et al.（2021））。然而，近期研究已开始在生成式 LLM 的语境下探索白盒蒸馏（Timiryasov and Tastet, 2023; Liang et al., 2023a; Gu et al., 2024; Agarwal et al., 2024; Liu et al., 2023a; Wen et al., 2023; Wan et al., 2024a; Zhao and Zhu, 2023; Qin et al., 2023b; Boizard et al., 2024; Zhong et al., 2024）。

获取这种特征知识的典型方法是由教师 LLM 用其内部表示来标注输出序列 $y$。然后用 Kullback-Leibler 散度（KLD）等方法把这些标注蒸馏进学生模型。引出特征知识的过程可以表述为：

$$
\mathcal{D}^{(\mathrm{feat})}=\{(x,y,\phi_{\mathrm{feat}}(x,y;\theta_T))\mid x\sim\mathcal{X},\ y\sim\mathcal{Y}\}. \tag{6}
$$

在这一公式中，$\mathcal{Y}$ 是输出集合，它可以由教师 LLM 或学生模型生成，也可以直接取自数据集。$\phi_{\mathrm{feat}}(\cdot;\theta_T)$ 表示从教师 LLM 中提取特征知识（如输出分布）的操作。

引出教师特征知识最直接的方法，是为一个固定的序列数据集标注 token 级的概率分布（Sanh et al., 2019; Wen et al., 2023）。为了利用教师模型中间层中丰富的语义和句法知识，TED（Liang et al., 2023a）设计了任务感知的逐层蒸馏（task-aware layer-wise distillation）。他们在每一层把学生的隐藏表示与教师的隐藏表示对齐，有选择地提取与目标任务相关的知识。Gu et al.（2024）和 Agarwal et al.（2024）提出了一种新方法：学生模型先生成序列（称为“自生成序列（self-generated sequences）”），然后学生利用教师在这些序列上给出的反馈（即输出分布）进行学习。当学生模型能力不足以模仿教师的分布时，这种方法尤其有益。此外，人们还提出了多种通过从教师 LLM 蒸馏特征知识来进行 LLM 量化的方法（Tao et al., 2022a; Liu et al., 2023a; Kim et al., 2023b）。这些方法旨在量化 LLM 时保留原有的输出分布，确保性能损失最小。另外，特征知识可以作为多教师知识蒸馏的有力来源。Timiryasov and Tastet（2023）利用 GPT-2 和 LLaMA 的集成作为教师模型来提取输出分布。类似地，FuseLLM（Wan et al., 2024a）创新性地通过对多个 LLM 的输出分布进行加权融合来结合它们的能力，并将其整合进单个 LLM。这种方法有潜力显著增强学生模型的能力，使其超越任何单个教师 LLM。

总之，特征知识为黑盒方法提供了一种更透明的替代方案，使人们能更深入地洞察和控制蒸馏过程。通过利用教师 LLM 的特征知识（如输出分布和中间层特征），白盒方法能够实现更丰富的知识迁移。尽管这类方法很有前景（尤其是在较小模型上），但它并不适用于内部参数无法获取的黑盒 LLM。此外，从白盒 LLM 蒸馏出的学生模型可能不如从黑盒教师蒸馏出的学生模型，因为黑盒教师 LLM（如 GPT-4）往往更强大。

> 译者注：“黑盒”指只能通过 API 拿到教师生成的文本，看不到其内部概率和参数（如 GPT-4）；“白盒”指能拿到教师每个 token 的概率分布甚至隐藏层（如开源的 LLaMA）。白盒蒸馏更接近 Hinton 2015 的原始做法。

#### 3.1.5 反馈（Feedback）

以往大多数工作主要关注从教师到学生的单向知识迁移以供模仿，并未考虑教师对学生生成结果的反馈。教师的反馈通常通过提供偏好、评估或纠正信息来指导学生生成的输出。例如，一种常见的反馈形式是由教师对学生的生成结果进行排序，并通过基于 AI 反馈的强化学习（Reinforcement Learning from AI Feedback, RLAIF）（Bai et al., 2022a）把这种偏好蒸馏进学生模型。下面是引出反馈知识的一般化公式：

$$
\mathcal{D}^{(\mathrm{fb})}=\{(x,y,\phi_{\mathrm{fb}}(x,y;\theta_T))\mid x\sim\mathcal{X},\ y\sim p_S(y\mid x)\}, \tag{7}
$$

其中 $y$ 表示学生模型针对 $x$ 生成的输出，$\phi_{\mathrm{fb}}(\cdot;\theta_T)$ 表示由教师 LLM 提供反馈。该操作在给定输入 $x$ 的情况下评估学生的输出 $y$，提供评估、纠正信息或其他形式的指导。这种反馈知识不仅可以蒸馏给学生，使学生也能生成反馈（例如构建学生偏好模型），更重要的是，它使学生能够根据反馈改进自己的回答。人们已探索了多种方法来引出这种高级知识（Bai et al., 2022a; Luo et al., 2023b; Cui et al., 2023a; Kwon et al., 2023; Jiang et al., 2023b; Chen et al., 2023a; Gu et al., 2024; Agarwal et al., 2024; Chen et al., 2024b; Guo et al., 2024; Ye et al., 2023; Hong et al., 2023; Lee et al., 2023a）。

如前所述，偏好（preference）是来自教师模型的一种重要反馈知识。通过以特定准则提示教师，可以从教师那里蒸馏出各种偏好知识。Bai et al.（2022a）提出 RLAIF，用于从 LLM 中蒸馏无害性偏好。具体做法是：用一个经过 SFT 训练的 LLM 为每个提示生成回答对，然后按无害性对它们排序，构建偏好数据集。该数据集被蒸馏进一个偏好模型（Preference Model, PM），再由该偏好模型指导一个更无害的 LLM 策略的强化学习训练。WizardMath（Luo et al., 2023b）侧重数学推理。他们使用 ChatGPT 作为教师，直接提供过程监督（process supervision），评估所生成解答中每一步的正确性。为了扩大高质量蒸馏偏好数据的规模，Cui et al.（2023a）构建了一个大规模偏好数据集 UltraFeedback，用于蒸馏更好的偏好模型。它汇集了多种指令和模型来生成对比数据，然后用 GPT-4 从偏好的多个方面（包括指令遵循、真实性、诚实性和有用性）为候选回答打分。

除了仅仅评估学生的生成结果外，教师还可以针对学生表现不佳的样例提供丰富的反馈。在 Lion（Jiang et al., 2023b）中，教师模型找出对学生模型有挑战性的指令，并生成新的、更难的指令，以增强学生的能力。PERsD（Chen et al., 2023a）展示了一种方法：教师根据学生生成的错误代码片段所遇到的具体执行错误，提供有针对性的改进反馈。类似地，SelFee（Ye et al., 2023）利用 ChatGPT 生成反馈，并根据反馈修改学生的答案。相比之下，FIGA（Guo et al., 2024）通过与真实（ground-truth）回答比较来修改学生的回答。此外，教师模型在学生生成结果上的分布本身就可以作为一种反馈。MiniLLM（Gu et al., 2024）和 GKD（Agarwal et al., 2024）提出了一种创新策略：学生模型先生成序列，再由教师模型给出输出分布作为反馈。这种方法利用教师的洞察来直接指导和完善学生模型的学习过程。

#### 3.1.6 自我知识（Self-Knowledge）

知识也可以从学生自身中引出，我们称之为自我知识。在这种设定下，同一个模型同时充当教师和学生，通过蒸馏和完善自己先前生成的输出来迭代地改进自身。这种知识独特地绕开了对外部（可能是专有的）强大教师模型（如 GPT 系列 LLM）的需求。此外，它还使模型能够突破传统师生方法中固有的局限或“天花板”。引出自我知识可以表述为：

$$
\mathcal{D}^{(\mathrm{sk})}=\{(x,y,\phi_{\mathrm{sk}}(x,y))\mid x\sim\mathcal{S},\ y\sim p_S(y\mid I\oplus x)\}, \tag{8}
$$

其中 $\phi_{\mathrm{sk}}(\cdot)$ 是一个广义函数，表示对自生成输出 $y$ 施加的额外处理，可以包括但不限于过滤、打奖励分，或任何其他增强或评估 $y$ 的机制。它可以由外部工具或学生本身 $\theta_S$ 来执行。该领域的近期研究提出了多种引出自我知识的创新方法，展示了其构建更高效、更自主的学习系统的潜力（Allen-Zhu and Li, 2020; Wang et al., 2022a; Sun et al., 2024b; Yang et al., 2024; Jung et al., 2023; Huang et al., 2023a; Gulcehre et al., 2023; Yuan et al., 2024a; Xu et al., 2023b; Zelikman et al., 2022; Chen et al., 2024a; Zheng et al., 2024; Li et al., 2024c; Zhao et al., 2024; Singh et al., 2023; Chen et al., 2024c; Hosseini et al., 2024）。

这种方法的一个著名例子是 Self-Instruct（Wang et al., 2022a），它利用 GPT-3 通过扩展方法进行数据增强，生成额外的数据样本来充实数据集，随后再用这个充实后的数据集微调原模型。其他方法则旨在通过修改提示从学生模型中引出有针对性的知识，并利用这些数据进一步完善模型。在 Self-Align（Sun et al., 2024b）中，作者发现用 Self-Instruct 数据微调的模型倾向于生成简短或不直接的回答。他们用冗长详细的指令提示该模型，以产生深入、详尽的回答。然后，他们使用上下文蒸馏（context-distillation）（Askell et al., 2021）把这些回答与不冗长的指令配对，蒸馏回模型中。类似地，RLCD（Yang et al., 2024）提出用对比提示（contrasting prompts）从未对齐的 LLM 中生成偏好对，其中既包含较优样例，也包含较差样例。在这些对上训练得到的偏好模型随后通过强化学习指导未对齐模型的改进。另有若干方法采用过滤手段来完善自生成数据。例如，Impossible Distillation（Jung et al., 2023）针对句子摘要任务，基于蕴含（entailment）、长度和多样性设置过滤器，对自生成的摘要进行筛选。LMSI（Huang et al., 2023a）为每个问题生成多条 CoT 推理路径和答案，然后只保留那些得出最一致答案的路径。

注意，随着学生模型不断改进，可以迭代地获取经过完善的自我知识，从而进一步增强学生的能力。Gulcehre et al.（2023）提出了强化自训练（Reinforced Self-Training, ReST）框架，在“增长（Grow）”和“改进（Improve）”两个阶段之间循环交替，逐步获得更好的自我知识并完善学生模型。在增长阶段，学生模型生成多个输出预测。然后在改进阶段，用一个评分函数对这些自生成输出进行排序和过滤。随后，语言模型在这个整理好的数据集上采用离线强化学习（offline RL）目标进行微调。Self-Play（Chen et al., 2024a）提出了一个类似迭代 DPO 的框架，微调语言模型使其区分自生成回答与人工标注数据。这些自生成回答可以被看作“负知识（negative knowledge）”，促使学生更好地与目标分布对齐。Self-Rewarding（Yuan et al., 2024a）探索了一种新颖而有前景的方法：把语言模型本身用作奖励模型。它采用“LLM 作为评判者（LLM-as-a-Judge）”的提示方式，自主地为自生成的回答分配奖励。整个过程可以迭代进行，从而同时提升指令遵循能力和奖励建模能力。

### 3.2 蒸馏（Distillation）

本小节关注把从教师 LLM 中引出的知识有效迁移到学生模型中的方法。如图 3 所示，我们探讨一系列蒸馏技术，从通过监督微调、散度与相似度来增强模仿的策略，到强化学习和排序优化等高级方法。

#### 3.2.1 监督微调（Supervised Fine-Tuning）

监督微调（SFT），又称序列级知识蒸馏（Sequence-Level KD, SeqKD）（Kim and Rush, 2016），是蒸馏强大黑盒 LLM 最简单、也是最有效的方法之一。SFT 通过最大化教师 LLM 所生成序列的似然来微调学生模型，使学生的预测与教师的预测对齐。这一过程在数学上可以表述为最小化如下目标函数：

$$
\mathcal{L}_{\mathrm{SFT}}=\mathbb{E}_{x\sim\mathcal{X},\,y\sim p_T(y\mid x)}\big[-\log p_S(y\mid x)\big], \tag{9}
$$

其中 $y$ 是教师模型生成的输出序列。这种简单却非常有效的技术构成了该领域众多研究的基础。许多研究者已成功使用 SFT，利用教师 LLM 生成的序列训练学生模型（Taori et al., 2023; Chiang et al., 2023; Wu et al., 2023c; Xu et al., 2023a; Luo et al., 2023b）。此外，许多自蒸馏工作中也探索了 SFT（Wang et al., 2022a; Huang et al., 2023c; Xu et al., 2023b; Zelikman et al., 2022）。由于应用 SFT 的 KD 工作数量众多，这里只列出代表性工作。更详细的工作见 §4。

> 译者注：式（9）其实就是普通的“下一个词预测”交叉熵训练，只不过训练答案 $y$ 是教师写的，而不是人工写的。

#### 3.2.2 散度与相似度（Divergence and Similarity）

本小节主要关注为从白盒教师 LLM 中蒸馏特征知识（包括分布和隐藏状态特征）而设计的算法。这些算法大致可分为两类：最小化概率分布之间散度的算法，以及旨在提升隐藏状态相似度的算法。

**表 1：不同散度类型下 $D$ 的函数形式。$p$：参考分布**

| 散度类型（Divergence Type） | $D(p,q)$ 函数 |
|---|---|
| 前向 KLD（Forward KLD） | $\sum p(t)\log\dfrac{p(t)}{q(t)}$ |
| 反向 KLD（Reverse KLD） | $\sum q(t)\log\dfrac{q(t)}{p(t)}$ |
| JS 散度（JS Divergence） | $\dfrac{1}{2}\Big(\sum p(t)\log\dfrac{2p(t)}{p(t)+q(t)}+\sum q(t)\log\dfrac{2q(t)}{p(t)+q(t)}\Big)$ |

**表 2：知识蒸馏中相似度函数汇总**

| 相似度函数 $\mathcal{L}_F$ | 表达式 |
|---|---|
| L2 范数距离（L2-Norm Distance） | $\lVert\Phi_T(f_T(x,y))-\Phi_S(f_S(x,y))\rVert_2$ |
| L1 范数距离（L1-Norm Distance） | $\lVert\Phi_T(f_T(x,y))-\Phi_S(f_S(x,y))\rVert_1$ |
| 交叉熵损失（Cross-Entropy Loss） | $-\sum\Phi_T(f_T(x,y))\log\big(\Phi_S(f_S(x,y))\big)$ |
| 最大均值差异（Maximum Mean Discrepancy） | $\mathrm{MMD}\big(\Phi_T(f_T(x,y)),\Phi_S(f_S(x,y))\big)$ |

**散度。** 基于散度的方法最小化教师模型与学生模型概率分布之间的散度，用一个通用的散度函数 $D$ 表示：

$$
\mathcal{L}_{\mathrm{Div}}=\mathbb{E}_{x\sim\mathcal{X},\,y\sim\mathcal{Y}}\big[D\big(p_T(y\mid x),\,p_S(y\mid x)\big)\big], \tag{10}
$$

$D$ 的具体形式取决于所采用的散度类型。表 1 列出了不同散度度量下 $D$ 的函数形式。常用的标准 KD 目标本质上是最小化教师分布与学生分布之间近似的前向 Kullback-Leibler 散度（KLD）（Sanh et al., 2019; Wen et al., 2023; Timiryasov and Tastet, 2023; Liang et al., 2023a; Chen et al., 2024d），即迫使 $p_S$ 覆盖 $p_T$ 的所有众数（mode）。

> **图 6**（图不翻译）：用一条双峰的目标分布 $p$ 对比两种近似结果——最小化 $\mathrm{KL}(p\Vert q)$ 得到的 $q$（蓝色）铺开覆盖两个峰，最小化 $\mathrm{KL}(q\Vert p)$ 得到的 $q$（绿色）集中在最显著的一个峰上。
> 原图注：前向与反向 KL 散度在近似目标分布时的比较。前向 KL 散度方法倾向于覆盖目标分布的所有众数，但不够精确，即“众数覆盖（mode-covering）”行为。反向 KL 散度方法主要聚焦于最显著的众数，从而表现出“众数寻求（mode-seeking）”行为。

然而，当学生模型无法学会一个高度复杂的教师的所有众数时，由此产生的“众数覆盖”行为可能导致学生把概率质量分配给在教师分布下概率很低的 token（参见图 6 蓝色曲线）。这种众数覆盖现象可能导致幻觉（hallucination）和低质量的生成。另一方面，反向 KL 这类众数寻求型散度会优先考虑教师分配了高概率的 token（参见图 6 绿色曲线）。这种方法可以降低低质量输出的风险，促成更准确的生成，但往往以多样性降低为代价。Gu et al.（2024）采用反向 KL 散度，防止学生高估教师分布中的低概率区域，并使用策略梯度（Policy Gradient）方法进行优化。Agarwal et al.（2024）和 Sason and Verdú（2016）都评估了不同散度函数在 LLM 蒸馏中的效果，发现最优散度取决于任务。例如，前向 KL 散度更适合机器翻译这类输出的众数或变化较少的任务，而反向 KL 散度更适合对话生成和指令微调这类涉及多个众数、可能的回答范围更广的任务。因此，任务的性质在很大程度上影响着为获得最佳性能而对散度函数的选择。

> 译者注：直观地说，前向 KL 惩罚“教师认为可能、学生却没覆盖到”的情况，所以学生会“什么都沾一点”；反向 KL 惩罚“学生认为可能、教师却认为不可能”的情况，所以学生宁可只学好教师最确定的那部分。

**相似度。** 知识蒸馏中基于相似度的方法旨在把学生模型的隐藏状态或特征与教师的对齐。这些方法使用各种相似度度量来衡量和优化两个模型内部表示的一致性。其目标是确保学生模型不仅产生与教师相似的输出，而且以相似的方式处理信息。基于相似度的目标可以表述为：

$$
\mathcal{L}_{\mathrm{Sim}}=\mathbb{E}_{x\sim\mathcal{X},\,y\sim\mathcal{Y}}\big[\mathcal{L}_F\big(\Phi_T(f_T(x,y)),\,\Phi_S(f_S(x,y))\big)\big], \tag{11}
$$

其中 $f_T(x,y)$ 和 $f_S(x,y)$ 分别是教师模型和学生模型的特征图（feature map）。对这些特征图施加变换函数 $\Phi_T$ 和 $\Phi_S$，以确保它们形状相同，便于直接比较。相似度函数 $\mathcal{L}_F$ 用于匹配这些变换后的特征图。表 2 列出了 $\mathcal{L}_F$ 的常见选择。在 LLM 的 KD 中采用基于相似度方法的工作很少。其中，Liang et al.（2023a）提出了任务感知逐层蒸馏（Task-Aware Layer-Wise Distillation, TED），该方法使用任务感知过滤器。这些过滤器被设计用来从教师模型中有选择地捕获与特定任务最相关的信息。其关键目标是最小化教师和学生模型中经过滤波的表示之间的差异。虽然基于相似度的方法在基于编码器的语言模型中很常见（Sun et al., 2019, 2020; Jiao et al., 2020; Hou et al., 2020; Zuo et al., 2022; Liang et al., 2021），但它们在 LLM 知识蒸馏中的应用并不那么广泛。不过，考虑到它们的有效性，我们预计在不久的将来，探索将这些方法用于 LLM 蒸馏的研究会增多。

#### 3.2.3 强化学习（Reinforcement Learning）

本小节探讨用强化学习（RL）把知识蒸馏进学生模型的高级方法。这种方法尤其适用于利用教师的反馈来训练学生模型（Bai et al., 2022a; Cui et al., 2023a; Luo et al., 2023b; Agarwal et al., 2024; Chen et al., 2024b; Ma et al., 2023a; Pang et al., 2023; Du et al., 2023a）。基于 RL 的蒸馏过程通常包括两个主要阶段：

**蒸馏奖励模型训练（Distilled Reward Model Training）。** 第一阶段是利用教师 LLM 生成的反馈数据 $\mathcal{D}^{(\mathrm{fd})}$ 训练奖励模型 $r_\phi$。偏好数据作为一种典型的反馈，被用于训练学生奖励模型（Bai et al., 2022a; Cui et al., 2023a; Lee et al., 2023a; Kim et al., 2023a）。它们通常由输入-输出三元组 $(x,y_w,y_l)$ 组成。这里 $y_w$ 和 $y_l$ 分别表示相对于教师偏好而言“胜出（winning）”和“落败（losing）”的输出。奖励模型的损失函数定义为：

$$
\mathcal{L}_{\mathrm{RM}}\big(r_\phi,\mathcal{D}^{(\mathrm{fd})}\big)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}^{(\mathrm{fd})}}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big] \tag{12}
$$

该公式引导奖励模型根据教师的标准正确区分更可取和较不可取的输出。RLMEC（Chen et al., 2024b）没有学习实例级奖励，而是采用了另一种方法：训练一个生成式奖励模型。它在从教师 LLM 蒸馏得到的错误解答改写数据上进行训练。这个蒸馏得到的奖励模型可以为 RL 训练产生 token 级的奖励。

**强化学习优化（Reinforcement Learning Optimization）。** 在第二阶段，以策略 $\pi_\theta$ 表示的学生模型被优化，以最大化训练好的奖励模型给出的期望奖励。同时，它还要最小化与参考策略 $\pi_{\mathrm{ref}}$（通常是经过 SFT 训练的学生模型初始策略）之间的散度，由系数 $\beta$ 控制。RL 目标为：

$$
\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{X},\,y\sim\pi_\theta(y\mid x)}\big[r_\phi(x,y)\big]-\beta\,D_{\mathrm{KL}}\big[\pi_\theta(y\mid x)\,\Vert\,\pi_{\mathrm{ref}}(y\mid x)\big] \tag{13}
$$

这一 RL 框架不仅确保学生模型学到教师的显式内容，还使其有效地采纳教师的偏好模式。使用 RL，尤其是 PPO（Schulman et al., 2017）算法，为使学生模型的输出与教师对齐提供了稳健的机制。另外，教师 LLM 也可以直接充当奖励模型，在 RL 过程中直接分配奖励，从而无需训练奖励模型（Lee et al., 2023a; Kwon et al., 2023）。这种方法虽然可能表现更优，但与使用一个较小的蒸馏奖励模型相比，计算成本更高。

> 译者注：式（12）中的 $\sigma$ 是 sigmoid 函数；它要求奖励模型给“更好”的回答打出比“更差”的回答更高的分。式（13）里的 KL 项像一根“缰绳”，防止学生为了刷高奖励而偏离原来的 SFT 模型太远。

#### 3.2.4 排序优化（Ranking Optimization）

排序优化为向语言模型注入偏好反馈提供了一种比 RL 更稳定、计算更高效的替代方案（Rafailov et al., 2023; Song et al., 2023a; Yuan et al., 2023b）。这种方法不同于传统的 RL 方法，它在微调过程中直接把来自一个固定偏好数据集的排序信息融入语言模型。直观上，它直接更新策略，以提高较受偏好的回答相对于较不受偏好回答的相对似然。这种对偏好的直接优化无需对输出进行采样，使过程更加稳定和高效。最近，已有一些工作探索用排序优化把教师的偏好蒸馏进学生模型（Tunstall et al., 2023; Hong et al., 2023; Yuan et al., 2024a）。

Zephyr（Tunstall et al., 2023）利用直接偏好优化（Direct Preference Optimization, DPO）（Rafailov et al., 2023）来蒸馏教师 LLM 中的偏好对齐。DPO 把强化学习的目标（如式 13，即带 KL 散度约束的奖励最大化）简化为单阶段的策略训练。具体而言，DPO 的训练目标是最大化如下期望：

$$
\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}^{(\mathrm{fd})}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right], \tag{14}
$$

其中根据教师 LLM 的判断，$y_w$ 优于 $y_l$。Hong et al.（2023）采用了两种基于排序的优化目标——用于对齐人类反馈的回答排序（Rank Responses to align Human Feedback, RRHF）（Yuan et al., 2023b）和偏好排序优化（Preference Ranking Optimization, PRO）（Song et al., 2023a）——来进行偏好蒸馏。RRHF（Yuan et al., 2023b）关注如下定义的排序损失：

$$
\mathcal{L}_{\mathrm{RRHF}}=\sum_{r_i<r_j}\max(0,\,p_i-p_j), \tag{15}
$$

其中 $r_i$ 和 $r_j$ 分别是教师 LLM 为回答 $y_i$ 和 $y_j$ 给出的奖励分数，$p_i$、$p_j$ 是它们在策略 $\pi_\theta$ 下对应的条件对数概率。这种方法强调根据教师的偏好对回答进行直接比较和排序。PRO（Song et al., 2023a）把成对比较的概念扩展到可以处理任意长度的偏好排序。对于给定指令 $x$ 以及按教师偏好排序的回答序列 $y_1\succ y_2\succ\dots\succ y_n$，RPO（原文如此，应指 PRO）的训练目标为：

$$
\mathcal{L}_{\mathrm{PRO}}=-\sum_{k=1}^{n-1}\log\frac{\exp(p_k)}{\sum_{i=k}^{n}\exp(p_i)}, \tag{16}
$$

其中 $p_k$ 表示 $y_k$ 在学生策略 $\pi_\theta$ 下的条件对数概率。通过迭代地对比生成各个回答的似然，PRO 优化学生语言模型，使其把最受偏好的回答放在首位，同时按偏好递减的顺序逐步对其余回答进行排序。

## 4 技能蒸馏（Skill Distillation）

在第 3 节关于知识引出和蒸馏算法的基础之上，我们把关注点转向这些技术如何促进 LLM 中特定技能的蒸馏。我们的探讨将涵盖 LLM 展现的多种技能，包括上下文遵循（Context Following）、对齐（Alignment）、智能体（Agent）、NLP 任务专精（NLP Task Specialization）和多模态（Multi-Modality）。上下文遵循关注学生理解输入信息并有效作出回应的能力。对齐深入探讨学生使其输出与教师回答相一致的能力。进一步地，智能体强调语言模型的自主性。NLP 任务专精突出 LLM 在各种自然语言处理任务上实现专精的多面性，展示其适应能力。最后，多模态涵盖从教师 LLM 到多模态模型的知识迁移。表 3 汇总了代表性工作，包括所涉及的技能、种子知识、教师 LLM、学生模型、知识引出方法和训练目标等细节。

**表 3：技能蒸馏工作汇总。** IF：指令遵循（Instruction Following），MD：多轮对话（Multi-turn Dialogue），TP：思维模式（Think Pattern），RAG：检索增强生成，NLU：自然语言理解，NLG：自然语言生成，IR：信息检索，SFT：监督微调，D&S：散度与相似度，RL：强化学习，RO：排序优化。

| 方法 | 技能 | 种子知识 | 教师 LLM | 学生模型 | 知识引出方式 | 目标 |
|---|---|---|---|---|---|---|
| **上下文遵循（Context Following）** | | | | | | |
| Self-Instruct (Wang et al., 2022a) | IF | 175 个人工整理的任务 | GPT3 | GPT3 | 扩展 + 自我知识 | SFT |
| Alpaca (Taori et al., 2023) | IF | 175 个人工整理的任务 | GPT3 | LLaMA | 扩展 + 自我知识 | SFT |
| LaMini-LM (Wu et al., 2023c) | IF | 3.5K 个维基百科类别 + 混合数据集 | ChatGPT | 多种模型 | 扩展 | SFT |
| WizardLM (Xu et al., 2023a) | IF | Alpaca Data | ChatGPT | LLaMA | 扩展 | SFT |
| Lion (Jiang et al., 2023b) | IF | Alpaca Cata（原文如此，应为 Alpaca Data） | ChatGPT | LLaMA | 标注 + 扩展 + 反馈 | - |
| BabyLlama (Timiryasov and Tastet, 2023) | IF | 10M-word BabyLM dataset | GPT-2 + small LLaMA | 5800 万参数的 LLaMA | 特征 | D&S |
| MiniLLM (Gu et al., 2024) | IF | Dolly Dataset | GPT2 + OPT + LLaMA | GPT2 + OPT + LLaMA | 特征 | D&S |
| Self-Align (Sun et al., 2024b) | IF | 人工编写的原则 | LLaMA | LLaMA | 扩展 + 自我知识 | SFT |
| Self-Rewarding (Yuan et al., 2024a) | IF | 人工编写的样本 | LLaMA | LLaMA | 自我知识 | SFT + RL |
| STaR (Zelikman et al., 2022) | IF | Arithmetic + CommonsenseQA + GSM8K | GPT-J | GPT-J | 自我知识 | SFT |
| Llama-GPT4 (Peng et al., 2023a) | IF | Alpaca Dataset | GPT4 | LLaMA | 标注 | SFT |
| Reflection-Tuning (Li et al., 2023e) | IF | Alpaca/WizardLM Dataset | ChatGPT | LLaMA | 标注 | SFT |
| Selective Reflection-Tuning (Li et al., 2024d) | IF | Alpaca/WizardLM Dataset | ChatGPT | LLaMA | 标注 | SFT |
| Vicuna (Chiang et al., 2023) | IF/MD | 人类对话 | ChatGPT + GPT4 | LLaMA | 标注 | SFT |
| Koala (Geng et al., 2023) | IF/MD | 人类对话 | ChatGPT | LLaMA | 标注 | SFT |
| Baize (Xu et al., 2023b) | IF/MD | Quora + Stack Overflow | ChatGPT | LLaMA | 扩展 + 自我知识 | SFT |
| UltraChat (Ding et al., 2023b) | IF/MD | Wikidata + Text Material + C4 | ChatGPT | LLaMA | 整理 | SFT |
| Orca (Mukherjee et al., 2023) | IF/TP | FLAN-v2 | ChatGPT + GPT4 | LLaMA | 标注 | SFT |
| Orca2 (Mitra et al., 2023) | IF/TP | FLAN-v2 + Few-Shot/Math/Synthetic | GPT4 | LLaMA | 标注 | SFT |
| SelFee (Ye et al., 2023) | IF/TP | 人类对话、Flan/代码/数学集合 | ChatGPT | LLaMA | 标注 | SFT |
| CoT-Distill (Hsieh et al., 2023) | IF/TP | e-SNLI + ANLI + CQA + SVAMP | PaLM | T5 | 标注 | SFT |
| KnowPAT (Zhang et al., 2023a) | IF/TP | CPKG + QA Data | ChatGPT + ChatGLM + Vicuna-7B | LLaMA | 标注 | SFT |
| DEBATunE (Li et al., 2024e) | IF/TP | 争议话题 | ChatGPT | LLaMA | 标注 | SFT |
| Phi-1 (Gunasekar et al., 2023) | IF/Code | - | GPT3.5 | phi-1 | 整理 | SFT |
| Phi-1.5 (Li et al., 2023a) | IF/Code | 来自网络的 2 万个主题 | GPT3.5 | phi-1 | 整理 + 标注 | SFT |
| SAIL (Luo et al., 2023c) | IF/RAG | Alpaca 数据 + 网页内容 | GPT4 | LLaMA | 标注 | SFT |
| KARD (Kang et al., 2023b) | IF/RAG | MedQAUSMLE | ChatGPT | T5 + OPT | 标注 | SFT + D&S |
| Self-RAG (Asai et al., 2023) | IF/RAG | Open-Instruct | GPT4 | LLaMA | 标注 | SFT |
| **对齐（Alignment）** | | | | | | |
| OpenChat (Wang et al., 2023c) | IF/Preference | 人类对话 | ChatGPT + GPT4 | LLaMA | 标注 | SFT + RL |
| Zephyr (Tunstall et al., 2023) | IF/Preference | 混合数据集 | GPT4 | Mistral | 标注 + 反馈 | SFT + RO |
| ALMoST (Kim et al., 2023a) | IF/Preference | 人工编写的提示 | LLaMA | LLaMA | 扩展 + 标注 | SFT + RL |
| RLCD (Yang et al., 2024) | IF/Preference | 人工编写的提示 | LLaMA | LLaMA | 标注 | SFT + RL |
| RLAIF (Lee et al., 2023a) | IF/Preference | 人工编写的提示 | PaLM 2 | PaLM 2 | 标注 + 反馈 | RL |
| GPT3 Reward (Kwon et al., 2023) | Preference | 人工编写的提示 | GPT3 | GPT3 | 标注 | RL |
| ILF (Scheurer et al., 2023) | Preference | 特定任务数据集 | GPT3 + FeedME | GPT3 | 标注 | RL |
| ULTRAFEEDBACK (Cui et al., 2023a) | Preference | 混合数据集 | GPT4 | LLaMA | 标注 | RL |
| Constitutional AI (Bai et al., 2022a) | Preference/Value | 人工编写的提示 | 自定义学生模型 | 自定义模型 | 标注 + 扩展 + 反馈 | SFT + RL |
| SANDBOX (Liu et al., 2023b) | Value | 模拟（Simulation） | text-davinci-002/-003 + GPT4 + ChatGPT | LLaMA | 数据整理 | SFT + RL |
| **智能体（Agent）** | | | | | | |
| Toolformer (Schick et al., 2023) | Tool | CCNet | GPT-J | GPT-J | 标注 | SFT |
| Graph-ToolFormer (Zhang, 2023) | Tool | 混合图数据集 | ChatGPT | GPT-J + LLaMA | 标注 | SFT |
| Gorilla (Patil et al., 2023) | Tool | 在线 API 文档 | GPT4 | LLaMA | 扩展 | SFT |
| GPT4Tools (Yang et al., 2023b) | Tool | 图像内容 | ChatGPT | LLaMA | 整理 + 扩展 | SFT |
| ToolAlpaca (Tang et al., 2023a) | Tool | Public-apis 仓库 | ChatGPT | LLaMA | 整理 | SFT |
| ToolLLM (Qin et al., 2023a) | Tool | 真实世界 API | ChatGPT | LLaMA | 整理 | SFT |
| MLLM-Tool (Wang et al., 2024) | Tool | HuggingFace 模型卡 | GPT4 | LLaMA | 整理 | SFT |
| FireAct (Chen et al., 2023b) | Planning | 混合 QA 数据集 | GPT4 | LLaMA | 标注 | SFT |
| AgentTuning (Zeng et al., 2023a) | Planning | 6 个智能体任务 | GPT4 + ChatGPT | LLaMA | 标注 + 扩展 | SFT |
| Lumos (Yin et al., 2023a) | Planning | 混合交互任务 | GPT4 | LLaMA | 标注 | SFT |
| AUTOACT (Qiao et al., 2024) | Planning | 混合 QA 任务 | LLaMA | LLaMA | 标注 | SFT |
| **NLP 任务专精（NLP Task Specialization）** | | | | | | |
| AugGPT (Dai et al., 2023a) | NLU | Amazon/Symptoms/PubMed20k Dataset | ChatGPT | BERT | 标注 | SFT |
| TDG (He et al., 2023b) | NLU | SST + QQP + MNLI | GPT3 | BERT | 扩展 | SFT |
| SunGen (Gao et al., 2023a) | NLU | 文本分类任务 | GPT2 | DistilBERT | 整理 | SFT |
| UDG (Wang et al., 2021a) | NLU | NLU 任务 | GPT3 | BERT | 扩展 | SFT |
| InheritSumm (Xu et al., 2023c) | NLG | Pile + ArXiv + CNN/DM + WikiHow | GPT3.5 | ZCode++ | 标注 | SFT |
| DIMSUM+ (Jung et al., 2023) | NLG | 无 | GPT2 + CTRL + BioGPT | T5 | 整理 + 自我知识 | SFT |
| Genie (Yehudai et al., 2024) | NLG | ELI5 + ASQA + NQ + CNN/DM | Falcon + LLaMA | FLAN + LLaMA | 标注 | SFT |
| GKD (Agarwal et al., 2024) | NLG/NLU/IF | XSum+WMT14 en-de+GSM8K+FLAN2021 | T5-XL | T5 | 特征 + 反馈 | D&S + RL |
| QUILL (Srinivasan et al., 2022) | IR | IR 数据集 | T5 | 4 层 Transformer | 内部知识 | D&S |
| RankVicuna (Pradeep et al., 2023a) | IR | IR 数据集 | ChatGPT | LLaMA | 标注 | SFT |
| RankZephyr (Pradeep et al., 2023b) | IR | IR 数据集 | ChatGPT + GPT4 | Mistral | 标注 | SFT |
| NDR (Mysore et al., 2023) | Recommendation | 推荐数据集 | GPT3 | MPnet-110M | 标注 | SFT |
| InstrcutRec (Zhang et al., 2023b) | Recommendation | 39 个指令模板 | ChatGPT | Flan-T5 | 扩展 + 自我知识 | SFT |
| ONCE (Liu et al., 2023c) | Recommendation | 推荐数据集 | ChatGPT | LLaMA | 标注 | SFT |
| PandaLM (Wang et al., 2023b) | Evaluation | Alpaca Data | ChatGPT | LLaMA | 标注 | SFT |
| Prometheus (Kim et al., 2024) | Evaluation | 50 条种子评分细则 | GPT4 | LLaMA | 标注 | SFT |
| InstructScore (Xu et al., 2023d) | Evaluation | 混合数据集 | GPT4 | LLaMA | 标注 | SFT |
| WizardMath (Luo et al., 2023b) | Math | GSM8k + MATH | ChatGPT | LLaMA | 扩展 + 反馈 | SFT + RL |
| Mammoth (Yue et al., 2023a) | Math/TP | 混合数学数据集 | GPT4 | LLaMA | 标注 | SFT |
| Mixed Distill (Chenglin et al., 2023) | Math/TP | SVAMP + GSM8K + ASDIV + StrategyQA | ChatGPT | LLaMa | 标注 | SFT |
| WizardCoder (Luo et al., 2023a) | Code | Code Alpaca Data | ChatGPT | StarCoder | 扩展 | SFT |
| Magicoder (Wei et al., 2023) | Code | 现有源代码 | ChatGPT | LLaMa | 整理 | SFT |
| WaveCoder (Yu et al., 2024) | Code | 现有源代码 | GPT4 | LLaMa | 整理 | SFT |
| Code Alpaca (Chaudhary, 2023) | Code | 代码指令 | ChatGPT | LLaMA | 扩展 + 自我知识 | SFT |
| Code Llama (Rozière et al., 2023) | Code | 人工编写的指令 | LLaMA | LLaMA | 扩展 + 自我知识 | SFT |
| Code Clean (Jain et al., 2023) | Code | 代码数据集 | ChatGPT | LLaMA | 标注 | SFT |
| **多模态（Multi-Modality）** | | | | | | |
| LLaVA (Liu et al., 2023e) | Vision-Language | COCO | GPT4 | LLaMA | 标注 | SFT |
| SVIT (Zhao et al., 2023b) | Vision-Language | Visual Genome + COCO | GPT4 | LLaMA | 标注 | SFT |
| LVIS-Instruct4V (Wang et al., 2023e) | Vision-Language | LVIS | GPT4V | LLaMA | 标注 | SFT |
| LLaVAR (Zhang et al., 2023d) | Vision-Language | LAION | GPT4 | LLaMA | 标注 | SFT |
| Macaw-LLM (Lyu et al., 2023) | Multiple Modalities | 带字幕/描述的图像/视频 | ChatGPT | LLaMA | 标注 | SFT |
| MIMIC-IT (Li et al., 2023f) | Multiple Modalities | 图像/视频数据集 | ChatGPT | LLaMA | 标注 | SFT |
| ChatBridge (Zhao et al., 2023d) | Multiple Modalities | 特定任务/多模态对话数据 | GPT4 + ChatGPT | LLaMA | 标注 | SFT |

### 4.1 上下文遵循（Context Following）

这一部分关注从 LLM 中蒸馏上下文遵循技能。这一过程是把 LLM 处理各种复杂上下文的能力——例如少样本（few-shot）示例、复杂指令、对话历史和检索增强信息——迁移到较小的模型中。该领域的许多研究工作旨在为较小模型赋予这些复杂的上下文遵循能力。我们在此将剖析技能蒸馏的这一方面，按上下文的不同类型进行分类，并详细阐述每种类型如何被蒸馏并融入更小、更高效的模型中。

#### 4.1.1 指令遵循（Instruction Following）

指令遵循能力使 LLM 能够理解并遵循用户给出的指令。这种能力显著增强了人机交互，使模型能够顺畅地理解并按用户指示执行任务。获得这种技能的一种主要方法是构建类似指令的“提示-回答”对，并采用监督微调（SFT）进行模型训练。用于此目的的数据可以由人类专家手工整理，也可以借助模板把现有的 NLP 任务转换为指令格式，例如在机器翻译数据前加上“Translate this sentence to Spanish:”（把这句话翻译成西班牙语：）。然而，这些方法有其局限。手工构建数据费时费力，而基于模板的转换缺乏指令多样性，且可能与人类的自然输入不太吻合。GPT-4 等 LLM 凭借其上下文学习和指令遵循能力，为创建多样且可控的 SFT 数据提供了一种高效的替代方案。大多数相关工作使用 OpenAI 的 GPT 系列模型生成“提示-回答”数据对，然后通过监督微调训练学生 LLM（Wang et al., 2022a; Taori et al., 2023; Chiang et al., 2023; Wu et al., 2023c; Xu et al., 2023a; Mukherjee et al., 2023; Mitra et al., 2023; Luo et al., 2023b; Peng et al., 2023a）。

**基础指令（Basic Instructions）。** Self-Instruct（Wang et al., 2022a）利用 GPT-3 的上下文学习能力，把一个包含 175 个任务的种子池扩展为 52K 条与任务无关的指令，确保涵盖广泛的通用指令。此外，它还引入了过滤和后处理阶段，以去除冗余或相似的指令。值得注意的是，通过在这个充实后的数据集上训练，GPT-3 获得了遵循指令的能力，在零样本指令任务以及为新任务提供专家编写的指令时，其表现可与 InstructGPT 相媲美。基于 Self-Instruct 方法，Taori et al.（2023）在 52K 条指令遵循示例上用 LLaMA 7B 模型训练了 Alpaca 模型，这些示例以与 Self-Instruct 类似的方式生成，但使用了更强大的 text-davinci-003 模型。为了增强指令数据的多样性，Wu et al.（2023c）提出了一种称为主题引导指令生成（Topic-Guided Instruction Generation）的技术。该方法从维基百科收集 3.5K 个常见主题，作为生成过程中的引导。

**复杂指令（Complex Instructions）。** 一些工作促使学生去解决更复杂的指令（Xu et al., 2023a; Luo et al., 2023b,a; Guo et al., 2023c）。根据 Xu et al.（2023a），由人工编写的种子衍生出的指令数据集往往只有低到中等的复杂度。为了增强较小模型遵循复杂指令的能力，WizardLM（Xu et al., 2023a）提出了 Evol-Instruct。该方法通过多步演化过程逐步把指令转化为更复杂的形式，既提高难度，又扩展主题的多样性。他们使用 OpenAI ChatGPT API 进行了四轮演化，得到一个包含 250k 条复杂指令的数据集。随后，他们在该数据集上训练了 LLaMA 7B 模型，称为 WizardLM。在测试指令的高难度部分，WizardLM 甚至超过了 ChatGPT，胜率比 ChatGPT 高 7.9%。Zhao et al.（2023e）进一步开展了初步研究，揭示了提高指令复杂度的有效性。Instruction Fusion（Guo et al., 2023c）进一步利用教师 LLM，通过融合两条不同的演化指令来提高复杂度。此外，这种“演化”指令的思路已被推广到蒸馏编程（Luo et al., 2023a）和数学（Luo et al., 2023b）等特定技能。

**人类指令（Human Instructions）。** 依赖 ChatGPT 生成的指令可能缺乏多样性，并与真实的人类指令存在差距。与这些工作不同，Vicuna（Chiang et al., 2023）和 Koala（Geng et al., 2023）利用来自社区贡献对话中的人类对话和自然指令，展现了令人瞩目的性能。这些对话来自 ShareGPT 等平台，用户在那里分享他们与 ChatGPT 的交互。不过需要注意的是，在这类自然对话上训练的模型可能会模仿原教师的风格，却未必能完全捕捉其推理过程（Gudibande et al., 2023; Mukherjee et al., 2023）。

**系统指令（System Instructions）。** 为了鼓励学生模型学习推理过程，Orca 和 Orca 2（Mukherjee et al., 2023; Mitra et al., 2023）通过引入系统消息（例如“explain like I’m five, think step-by-step”（像给五岁孩子讲解那样解释，一步一步思考））来增强“提示-回答”数据对，促使学生模型掌握推理过程。该系统消息促使 GPT-4 提供解释轨迹（explanation traces），阐明教师的推理过程。Orca 2（Mitra et al., 2023）进一步在 Orca 表现的指导下，训练学生模型为每个任务找出最有效的求解策略。这种方法显著提高了较小模型遵循涉及推理的指令的能力。

**高质量指令（High-Quality Instructions）。** 正如 Zhou et al.（2023a）和（Li et al., 2024f）所展示的，数据质量对指令遵循训练至关重要。UltraChat（Ding et al., 2023b）借助各种元信息，从教师 LLM 中蒸馏出具有高质量、多样化指令的大规模数据。在这些数据上微调的 UltraLLaMA 模型始终优于其他开源模型。Phi 系列模型（Gunasekar et al., 2023; Li et al., 2023a; Mar, 2023）把数据质量放在首位，采用合成方法生成“教科书质量”的数据，以提升较小模型的学习效果。值得注意的是，即使没有专门的指令微调，Phi 也展现出有效遵循指令的能力。尤其引人注目的是，仅有 27 亿参数的 Phi-2 在各种基准评测中超过了 7B 和 13B 参数的 Mistral 和 Llama-2 模型。

**改进的指令（Improved Instructions）。** 另一类工作关注改进现有指令数据的质量，包括改进指令本身和相应的回答。SelFee（Ye et al., 2023）利用 ChatGPT 迭代地提升回答质量。ExpertLLaMA（Xu et al., 2023f）通过为普通指令补充专门的专家身份（Expert Identity）描述来提高回答质量。Reflection-Tuning（Li et al., 2023e）通过依据特定标准进行反思，依次改进指令和回答。DEITA（Liu et al., 2023h）提出从复杂度、质量和多样性三个方向增强指令并为其打分，以获得高质量的蒸馏数据。MUFFIN（Lou et al., 2023）提出根据输入来扩展指令，即用多种输入层面（input facets）使这些任务多样化。Selective Reflection-Tuning（Li et al., 2024d）首次通过一个新颖的学生选择模块让学生模型参与数据改进流程，学生模型能够自己决定从哪些数据中学习。

总之，从教师中蒸馏指令数据为训练低成本、可复现的指令遵循语言模型提供了一条很有前景的途径。当前的小模型已在提升指令遵循能力的多个方面（如多样性、复杂度和解释）取得进展。然而，在 ChatGPT 扩展出的指令数据上训练的学生模型，往往模仿了 ChatGPT 的风格，却没有复制其事实准确性（Gudibande et al., 2023）。要获得更强的指令遵循能力，需要更强的教师 LLM（Gudibande et al., 2023），以及获取多样化、高质量的指令数据，例如 Orca 中使用的数据（Mukherjee et al., 2023; Mitra et al., 2023），其中包含了来自 Flan 2022 Collection（Longpre et al., 2023）的大量任务指令。

#### 4.1.2 多轮对话（Multi-turn Dialogue）

指令遵循关注单次命令的执行，而多轮对话则把它扩展为在持续的交互中理解并保持上下文。这一技能对于模型有意义地参与类人对话、并在连续的对话轮次中作出连贯回应至关重要。一些工作致力于通过从教师 LLM 中蒸馏多轮知识来训练小型聊天模型（Chiang et al., 2023; Xu et al., 2023b; Ding et al., 2023b; Li et al., 2023b; Wang et al., 2023c; Tunstall et al., 2023）。

ShareGPT 是一个供用户分享其与 ChatGPT 对话的平台，提供了一个现成的海量多轮对话库。一些小型聊天模型利用这些数据进行训练，以获得进行多轮对话的能力（Chiang et al., 2023; Ye et al., 2023; Wang et al., 2023c）。例如，Vicuna（Chiang et al., 2023）是一个完全在 ShareGPT 数据上训练的聊天模型。尽管其唯一的训练来源是 ShareGPT，Vicuna 仍获得了由 GPT-4 评定的较高 MT-Bench（Zheng et al., 2023a）分数³。在 Wang et al.（2023c）开展的研究中，他们利用 ShareGPT 数据，用 GPT-3.5 和 GPT-4 生成混合回答。他们为 GPT-4 生成的回答分配更高的奖励，旨在激励学生模型产生高质量的回答。此外，Ye et al.（2023）通过对模型回答生成自我反馈并基于收到的反馈迭代地改进回答，提升了 ShareGPT 多轮数据的质量。

> 脚注 3：MT-Bench：一个多轮问题集，其中模型的生成结果由 GPT-4 等 LLM 进行评估。

为了增强学生模型的多轮能力，另一类研究聚焦于通过自我对话扩展对话数据集，并用它们训练较小的模型（Xu et al., 2023b; Ding et al., 2023b; Tunstall et al., 2023）。例如，Xu et al.（2023b）以来自 Quora 和 Stack Overflow 的问题作为种子开始工作，通过自我对话收集了 111.5k 条对话。随后，他们采用参数高效微调（parameter-efficient tuning）训练出一个名为 Baize 的聊天模型。Ding et al.（2023b）首先构建了一个大得多的数据集 UltraChat，包含 150 万条高质量的多轮对话。他们是通过从 ChatGPT 中蒸馏指令和对话来实现的。值得注意的是，UltraChat 涵盖了广泛的主题和指令。在 UltraChat 数据集的基础上，他们微调了一个 LLaMA 模型，得到了一个强大的聊天模型 UltraLLaMA。UltraLLaMA 始终优于其他开源聊天模型，包括 Vicuna 和 Baize。此外，UltraChat 还被用于一个与 AI 偏好对齐的聊天模型 Zephyr（Tunstall et al., 2023）。Zephyr 通过应用蒸馏直接偏好优化（distilled direct preference optimization, dDPO）增强了意图对齐。

#### 4.1.3 RAG 能力（RAG Capability）

众所周知，LLM 缺乏利用最新知识的能力，而且由于仅依赖参数化知识，常常产生包含事实错误的回答。检索增强生成（RAG）是缓解这一问题的一种很有前景的技术。处理由检索信息增强的上下文，也是 LLM 的一项不简单的技能。人们已提出了几种蒸馏 RAG 能力的方法（Kang et al., 2023a; Luo et al., 2023c; Asai et al., 2023）。

SAIL（Luo et al., 2023c）首先使用搜索 API 为每个训练样例检索搜索结果，构造同时包含指令和事实依据（grounding）信息的搜索增强指令。为了鼓励语言模型优先利用信息量大的检索结果，他们把每个检索到的段落连同真实回答一起输入蕴含模型（entailment model），为每个检索结果标注相关性。随后，把搜索增强指令和相关性标签输入教师 LLM（如 GPT-4）以生成回答。在这个训练集上微调之后，学生模型就能熟练地对搜索结果去噪并生成准确的回答。KARD（Kang et al., 2023b）针对问题 $x$ 从教师 LLM 中蒸馏出推理依据（rationale）$r$。这些推理依据随后被用于训练两个模型：一个学生语言模型和一个重排序器（Reranker）。在训练学生语言模型时，推理依据被用来检索相关知识 $d$，随后用推理依据连同问题和知识一起对学生语言模型进行微调。然而，在推理阶段只有问题可用。为了解决这一问题，他们训练重排序器去模仿检索器如何借助推理依据为段落打分，方法是最小化 $\mathrm{Retriever}(d\mid r)$ 与 $\mathrm{Reranker}(d\mid x)$ 之间的 KL 散度。然而，在语言模型中整合固定数量的段落而不考虑其必要性或相关性，可能会降低通用性并导致生成无用的回答。为了让学生语言模型具备自适应的 RAG 能力，Self-RAG（Asai et al., 2023）把这种自适应能力从教师 LLM 蒸馏到一个小型评判模型（critic model）中。该评判模型通过生成“反思 token（reflection tokens）”来判断是否需要检索，并评估检索结果的质量。例如，Self-RAG 在生成反思 token「Retrieve」时启动检索操作。为了蒸馏这些评判数据，他们用少样本示例 $I$、任务输入 $x$ 和输出 $y$ 提示 GPT-4 评估是否需要检索，以如下方式预测反思 token $r$：$p(r\mid I,x,y)$。

### 4.2 对齐（Alignment）

#### 4.2.1 思维模式（Thinking Pattern）

大多数现有方法主要关注把学生模型的直接回答与教师模型的回答直接对齐（Taori et al., 2023）。这些模型虽然有效，但可能存在这样的问题：它们倾向于学会模仿教师模型的回答风格，而非推理过程（Mukherjee et al., 2023）。因此，为了更好地从教师模型中蒸馏，人们提出了不仅模仿单纯回答、还模仿一些新颖思维模式的方法（Ye et al., 2023; Mukherjee et al., 2023; Mitra et al., 2023; Wang et al., 2023d; Cheng et al., 2023; Zhang et al., 2023a）。

受 LLM 能在不依赖外部模型的情况下生成自身反馈这一有效性的启发（Schick et al., 2022; Madaan et al., 2023; Saunders et al., 2022），SelFee（Ye et al., 2023）提出训练一个经过微调的模型，使其不断修改自己的答案，直到在单次推理中给出高质量的回答。在训练过程中，它同时把最终回答和反馈链作为拟合目标。这种“带修改过程的回答”模式展现出可观的性能提升。继 SelFee 之后，Reflection-Tuning（Li et al., 2023e, 2024d）也把反思过程作为学习模式。注意到以往方法缺乏对推理的模仿，Orca（Mukherjee et al., 2023）首先提出了解释微调（Explanation tuning），旨在从教师模型中学习推理步骤，包括解释轨迹、逐步的思考过程以及其他复杂指令，而不仅仅是普通的风格。大量实验验证了用这种思维模式进行蒸馏的有效性。随后的 Orca2（Mitra et al., 2023）受较小模型与较大模型之间能力差异的启发，进一步提出让学生模型具备针对不同任务使用不同求解策略的能力。通过采用这种训练模式，学生模型能够获得更好的推理能力。除了结合相应的修改或反思过程进行学习外，最近出现的另一种思维模式是同时生成回答和偏好。Zhang et al.（2023a）提出利用 LLM 为特定领域问答同时学习知识及相应的偏好。最近，DEBATunE（Li et al., 2024e）提出提高 LLM 在争议话题上生成陈述的可控性。通过让两个智能体围绕争议话题进行结构化的多轮辩论，可以获得突出而深入的陈述，并进一步蒸馏进学生模型。

#### 4.2.2 偏好（Preference）

前面提到的方法主要关注学生模型产生严格准确的结果这一基本能力，但这些结果可能与人类偏好不一致；在这一层面上达成对齐，能让这些模型协助完成各种任务，却无法满足更高层次的需求。早期方法主要利用人类反馈来对齐人类偏好（Ziegler et al., 2019; Stiennon et al., 2020; Wu et al., 2021; Ouyang et al., 2022; Bai et al., 2022b; Köpf et al., 2023; Yuan et al., 2023b）。然而，获取人类反馈成本高昂且费时费力，因此人们也提出了从 AI 反馈中学习以对齐人类偏好的方法（Bai et al., 2022a; Kwon et al., 2023; Scheurer et al., 2023; Kim et al., 2023a; Roit et al., 2023; Yang et al., 2024; Lee et al., 2023a; Tunstall et al., 2023; Cui et al., 2023a; Wang et al., 2023f）。

Bai et al.（2022a）提出的 RLAIF 概念，是把 LLM 标注的偏好与人类标注的偏好结合起来。这种方法旨在同时优化两个关键目标：确保输出的有用性，并尽量减少任何潜在危害，使 LLM 的回答更符合人类偏好。Kwon et al.（2023）利用 GPT-3 等 LLM 构建了一个代理奖励函数（proxy reward function）：先向 LLM 提供用户期望行为的描述以及少量示例，然后 LLM 通过评估模型输出与所提供描述的吻合程度来给出奖励，本质上是衡量输出与既定参考答案的相关性。Scheurer et al.（2023）提出了基于语言反馈的模仿学习（Imitation Learning from Language Feedback），利用一个语言模型来改进另一个模型生成的各种输出。这种改进以人类提供的参考为依据。在此过程之后，选出改进效果最好的输出，用于进一步的监督微调。按照 Kim et al.（2023a）的描述，ALMoST 把人类偏好浓缩为一组启发式准则。这类规则的一个例子是：使用更全面、更高质量提示的更大 LLM 可能会产生更好的回答。基于这些既定准则，利用不同规模、不同提示的 LLM 的回答生成对比数据，再用这些数据训练奖励模型。Yang et al.（2024）提出了基于对比蒸馏的强化学习（Reinforcement Learning from Contrast Distillation），旨在不依赖人类反馈的情况下对齐语言模型。这种方法利用模拟的偏好对来训练偏好模型，这些偏好对包括通过正、负对比提示生成的高质量和低质量样例。

Lee et al.（2023a）进一步强调了 RLAIF 的有效性。该工作提出，RLAIF 不仅能与 RLHF 相当，在某些情况下甚至超过 RLHF；有趣的是，RLAIF 还能提升监督微调的性能。另一个值得注意的发现是，在强化学习过程中直接提示 LLM 给出奖励分数，可能比基于 LLM 偏好训练奖励模型的传统方法更有效。Wang et al.（2023f）提出了 Conditioned-RLFT，它把不同数据来源视为粗粒度的奖励标签，并构建一个类别条件策略（class-conditioned policy），以有效利用质量各异的数据；这是一种无需强化学习的监督学习方法。Cui et al.（2023a）提出了一个由 GPT-4 标注、用于全面反馈的大规模、高质量、多样化偏好数据集。Tunstall et al.（2023）在 UltraFeedback 上提出了蒸馏直接偏好优化（distilled Direct Preference Optimization）（Rafailov et al., 2023），得到了一个小而强大的 LLM。

#### 4.2.3 价值（Value）

与人类偏好对齐，使大模型能够以符合人类偏好的方式运行，从而最大化人类满意度。然而，为了建立可信赖的 LLM，人们提出了“使 LLM 与人类价值观对齐”的概念，对齐的关键原则通常被概括为“HHH”标准：有用（helpful）、无害（harmless）、诚实（honest）（Weidinger et al., 2021; Askell et al., 2021）。人们已经采取了许多方法来构建可信赖的 LLM。然而，由于这一目标本身的困难——即使对专有模型而言这仍是一个未解决的问题（Sun et al., 2024a）——大多数现有方法依赖于构建高质量的人类偏好数据集（Ji et al., 2023b; Solaiman and Dennison, 2021; Bai et al., 2022b; Qiu et al., 2022; Kiesel et al., 2022; Liu et al., 2022a）、利用人工编写的规则作为约束（Glaese et al., 2022; Sun et al., 2023b, 2024b）等。关于可信赖 LLM 的详细进展，请进一步参阅 Yao et al.（2023a）、Liu et al.（2023i）和 Sun et al.（2024a）。

尽管探索还略显不足，但通过蒸馏使 LLM 与人类价值观对齐仍是可能的（Bai et al., 2022a; Cui et al., 2023a; Yang et al., 2024; Sun et al., 2024b）。例如，Bai et al.（2022a）提出 RLAIF，利用 AI 生成的标签交互式地同时提升有用性和无害性。Sun et al.（2024b）用 16 条原则作为准则来提示学生模型，以生成有用、合乎伦理且可靠的回答。类似地，可以通过修改提示同时引出无害和有害的生成结果，然后用它们训练偏好模型（Yang et al., 2024）。Cui et al.（2023a）利用 GPT-4 从有用性、真实性和诚实性方面对生成结果进行排序。Liu et al.（2023b）通过在训练过程中融入模拟的社会交互，推进 LLM 与社会价值观的对齐。这种方法包含一系列要素，包括与社会规范相符和相冲突的示例，以及集体评分、深入的反馈和经过迭代修改的回答。

### 4.3 智能体（Agent）

#### 4.3.1 工具使用（Tool Using）

尽管近期的 LLM 已展现出解决各种任务的能力，但在处理较大的数值或执行复杂的数学计算时，它们仍容易出错（Qian et al., 2022; She et al., 2023; Manikandan et al., 2023; Liang et al., 2023b; Mialon et al., 2023）。因此，使 LLM 智能体具备使用工具的能力日益受到关注。常用方法主要依赖人工整理的数据进行训练（Parisi et al., 2022; Nakano et al., 2022; Qin et al., 2023c; Song et al., 2023b）或依赖提示设计（Cai et al., 2023; Shen et al., 2023a; Hao et al., 2024）。最近，也有人提出了基于蒸馏的方法（Schick et al., 2023; Zhang, 2023; Patil et al., 2023; Tang et al., 2023a; Qin et al., 2023a; Yuan et al., 2023a; Gao et al., 2023b; Wang et al., 2024; Shen et al., 2024; Yuan et al., 2024b）。

Toolformer（Schick et al., 2023）采用自监督方式，避免了大量人工标注，来获取最需要使用的 API，并进一步把这种能力蒸馏到模型自身中。基于 GPT-J 的 Toolformer 的性能大幅超过 OPT（66B）（Zhang et al., 2022）和 GPT3（175B）（Brown et al., 2020）。Graph-ToolFormer（Zhang, 2023）旨在使 LLM 具备处理复杂图数据并在其上推理的能力；它通过采用 ChatGPT 标注并扩充一个更大的图推理语句数据集用于训练，借助外部图推理 API 工具来增强 LLM 的图推理技能。Gorilla（Patil et al., 2023）针对当前 LLM 在生成准确输入参数方面的局限，减少“幻觉”即生成错误 API 用法的问题；它从 HuggingFace 和 Torch Hub 等平台收集数千个模型作为 API 调用，并利用 GPT-4 生成合成指令数据用于训练。GPT4Tools（Yang et al., 2023b）旨在使 LLaMA 和 OPT 等开源 LLM 能够使用多模态工具，而这种能力以前仅限于 ChatGPT 和 GPT-4 等先进的专有模型。该方法通过用多模态上下文提示一个先进的教师模型来生成指令遵循数据集，并使用低秩适配（Low-Rank Adaptation）进行优化。ToolAlpaca（Tang et al., 2023a）提出了一个框架，旨在增强面向具身智能的紧凑语言模型的工具使用能力。它基于 50 个类别中 400 多个真实世界工具 API 构建了一个包含 3938 个实例的数据集，并利用 ChatGPT 为每个提示生成文档以供之后训练。ToolLLM（Qin et al., 2023a）提出了一个通过从 ChatGPT 蒸馏来增强 LLM 工具使用能力的全面框架，涵盖数据构建、模型训练和评估。他们的 ToolLLaMA 在执行复杂指令和处理新 API 方面表现出色，可与 ChatGPT 相媲美。CRAFT（Yuan et al., 2023a）构建了一个通用的工具创建与检索框架，利用 GPT-4 生成代码片段作为所创建的工具。在推理时，其他小型 LLM 可以从生成的代码片段中选择和检索，以执行这些片段或基于给定片段生成其他方法。Confucius（Gao et al., 2023b）引入了一种分层训练策略，让 LLM 通过循序渐进的课程掌握工具使用，并提出了一种名为“基于内省反馈的迭代自指令（Iterative Self-instruction from Introspective Feedback, ISIF）”的创新方法，动态增强数据集以处理复杂工具。MLLM-Tool（Wang et al., 2024）是一个多模态工具智能体，通过把多模态编码器与开源大语言模型集成，能够理解嵌入在视觉或音频内容中的指令。作为一种可训练的方法，其初始的“指令-答案”对是利用 GPT-4 生成的。Shen et al.（2024）表明小型 LLM 是较弱的工具学习者，并提出了一个多 LLM 框架，把单个模型的工具使用能力分解为规划器（planner）、调用器（caller）和总结器（summarizer），取得了极佳的性能。该工作提出的两阶段训练策略借助 ChatGPT 和 GPT-4 来收集执行轨迹，构成训练集。Yuan et al.（2024b）注意到当前冗长的工具文档可能带来的问题——它妨碍 LLM 理解如何使用工具——因此提出了 EASYTOOL，从大量文档中提炼重要信息。训练文档的参考摘要是用 ChatGPT 获得的。

#### 4.3.2 规划（Planning）

LLM 智能体的另一个重要方面，是把高层任务分解为一组选定的可执行步骤的能力（Huang et al., 2022b），这在交互式环境中行动时尤其有用。Huang et al.（2022b）首先证明了 LLM 无需训练即可生成合理的目标驱动行动计划，引入了非侵入式工具来增强模型的可执行性，并通过人工评估来权衡可执行性与语义准确性。大多数现有方法利用提示策略进行任务规划（Singh et al., 2022; Zhou et al., 2023b; Song et al., 2023c; Wang et al., 2023g; Yao et al., 2023b; Liu et al., 2023j; Hao et al., 2023; Hu et al., 2023a），或构建人工整理的数据进行训练（Lin et al., 2023a; Valmeekam et al., 2023）。最近，也出现了一些蒸馏方法（Chen et al., 2023b; Zeng et al., 2023a; Yin et al., 2023a; Qiao et al., 2024; Kong et al., 2023）。

FireAct（Chen et al., 2023b）提出了一种改进 LLM 的创新方法。该方法利用源自多种任务和提示技术的智能体轨迹来微调较小规模的 LLM。实验表明，使用 GPT-4 生成的轨迹应用该方法可以持续提升性能。AgentTuning（Zeng et al., 2023a）旨在提升 LLM 执行智能体任务的表现，同时不牺牲其广泛的通用能力。它利用一个名为 AgentInstruct、包含高质量交互轨迹的新数据集，采用一种混合指令微调方法，把这些轨迹与通用领域指令合并。Lumos（Yin et al., 2023a）是一个新颖的框架，旨在基于开源 LLM、以统一的数据格式和模块化架构训练智能体。该系统由三个关键模块组成：规划、落地（grounding）和执行，能够把任务分解为子目标和可执行步骤。TPTU-v2（Kong et al., 2023）利用人类专家或 LLM 生成的数据，专注于提升 LLM 在真实场景中的任务规划和工具使用能力。它引入了一个由三个组件构成的框架：API 检索器（API Retriever）、LLM 微调器（LLM Finetuner）和示例选择器（Demo Selector）。AUTOACT（Qiao et al., 2024）提出了一个智能体学习框架，不需要大规模标注数据，也不需要来自 GPT-4 等高资源模型的合成轨迹。相反，它使用 self-instruct 方法，在有限的初始数据下生成自己的规划轨迹，然后采用分工策略，创建专门负责任务完成过程中不同方面的子智能体。

蒸馏同样适用于训练具身多模态智能体（Sumers et al., 2023; Yang et al., 2023c; Ma et al., 2023a; Du et al., 2023a; Sumers et al., 2023）。例如，Sumers et al.（2023）旨在增强 AI 智能体遵循指令的能力：他们利用预训练的视觉-语言模型提供监督，帮助智能体在其运行环境中理解语言并据此行动，并借助模型蒸馏和事后经验回放（hindsight experience replay），在模拟的 3D 环境中教会它们与上下文相关的交互。Emma（Yang et al., 2023c）评估了在没有专家指导的情况下、在嘈杂的视觉世界中训练具身智能体所面临的挑战和低效问题，并提出在模拟环境中用模仿学习训练它们，由一个专家语言模型（如 ChatGPT）进行指导，该专家在一个相应的基于文本的模拟中运行，专注于相同的任务。

### 4.4 NLP 任务专精（NLP Task Specialization）

NLP 任务常常面临数据稀缺、可解释性问题、隐私担忧和噪声数据等挑战。本综述的“知识”部分阐述了从 LLM 中蒸馏知识的各种方法，有效地为学生模型适应一系列 NLP 任务奠定了基础。这些知识通过信息增强（如 CoT 和解释）、数据增强和语义表示，为学生模型的训练提供监督。通过迁移从 LLM 蒸馏得到的知识，学生模型能更好地应对各种 NLP 挑战，提升任务性能，并更稳健地应对数据局限。

#### 4.4.1 自然语言理解（Natural Language Understanding）

自然语言理解（NLU）是一项基础的 NLP 任务，涉及对人类语言的理解和解释。从 LLM 蒸馏得到的知识（例如通过数据标注或增强）通常被迁移到 BERT（Vaswani et al., 2017）和 RoBERTa（Liu et al., 2019）等基于编码器的语言模型中。在分类任务方面，有一些研究值得关注（Dai et al., 2023a; Gilardi et al., 2023; He et al., 2023b; Gao et al., 2023a; Chenglin et al., 2023; Li et al., 2023g）。AugGPT（Dai et al., 2023a）同时关注通用领域和临床领域的文本分类。小规模临床数据集往往缺乏专家标注且受严格的隐私法规约束；为了应对其局限，AugGPT 利用教师 LLM 的知识改写训练样本中的每个句子。这一过程创造出多个概念相似但语义不同的样本，增强了数据集的丰富性和多样性。Gilardi et al.（2023）展示了另一种方法：他们用 ChatGPT 作为标注者对输入进行分类。实验表明，该方法在相关性、立场、主题和框架检测（frame detection）等多项任务上优于众包工作者。此外，He et al.（2023b）提出了目标数据生成（TDG），这是一种识别数据集中有挑战性子群的新方法。TDG 利用 LLM 并结合人在回路（human-in-the-loop），专门为这些子群生成新数据，从而充实数据集，提升模型在情感分析和自然语言推断任务上的性能。为了促进临床信息抽取任务，Tang et al.（2023b）通过提供示例和不同的临床实体种子（即数据整理方式），从 LLM 中引出多样的样本。

也有若干研究关注多个 NLU 任务（Ding et al., 2023a; He et al., 2023a; Wang et al., 2021a; He et al., 2022; Ye et al., 2022; Meng et al., 2022）。例如，He et al.（2023a）利用 GPT-3.5 中的知识，为多种 NLU 任务（包括用户输入与关键词相关性评估、BoolQ 和 WiC）的输入标注标签和解释。Wang et al.（2021a）使用少样本提示，借助 GPT-3 扩展高质量的训练数据，即扩展方式。Ding et al.（2023a）不局限于用单一方法引出 NLP 任务知识，而是探索结合标注、扩展和整理三种方法从 GPT-3 中提取知识，为序列级和 token 级 NLP 任务蒸馏数据。

#### 4.4.2 自然语言生成（Natural Language Generation）

自然语言生成（NLG）是评估 LLM 能力的一个关键方面，涵盖摘要、机器翻译以及其他开放式文本生成等任务。LLM 以其强大的生成能力和创造力而著称，在这些领域表现出色，因而是把知识蒸馏进面向 NLG 任务的学生模型的首选来源（Xu et al., 2023c, 2024b; Ramnath et al., 2023; Agarwal et al., 2024）。此外，从 LLM 蒸馏得到的知识还可以有效地用于特定 NLG 任务的数据增强（Jung et al., 2023; Wang et al., 2021b; Guo et al., 2023a; Yang and Nicolai, 2023; Wang et al., 2023h; Yang et al., 2023d）。前面各节已关注开放式生成和多轮对话方面的工作，这一部分将特别着重介绍与其他 NLG 任务相关的蒸馏技术。

在摘要任务中，虽然自动评测指标往往偏向较小的微调模型，但人类评估者倾向于更喜欢 LLM 生成的摘要。针对这一差异，Xu et al.（2023c）通过蒸馏 GPTSUMM 数据集开发了一个学生摘要模型，该数据集包含通过查询 GPT-3.5 生成的 400 多万个“段落-摘要”对。Jung et al.（2023）则采用了另一种方法，提出“不可能的蒸馏（Impossible Distillation）”，从较弱的教师 LLM 中创建高质量的摘要专用数据集。该方法在生成的数据集上训练学生模型，并通过自我知识增强其能力。再看机器翻译任务，传统上构建平行语料既昂贵又耗时，Yang and Nicolai（2023）提出了一个三步蒸馏过程：生成动词和名词种子、组成句子，然后翻译这些句子。他们的发现表明，虽然蒸馏得到的数据集可能缺乏多样性，但它能有效增强用于训练学生翻译模型的翻译信号。为了自动蒸馏高质量的、基于内容的（content-grounded）数据，Genie（Yehudai et al., 2024）提出了一种通用方法，包含三个关键步骤：（a）准备内容；（b）从教师 LLM 中蒸馏与内容对应的回答；（c）通过过滤机制确保生成数据的质量和忠实性。Genie 表明，用这种蒸馏数据训练的学生模型可以媲美甚至超过在人工生成数据上训练的模型。

#### 4.4.3 信息检索（Information Retrieval）

信息检索（IR）是计算机科学的一个重要分支，关注如何从海量信息库中高效地检索与用户查询相关的信息（Cai et al., 2022; Liu et al., 2022b; Feng et al., 2023; Shen et al., 2023b）。一个典型的 IR 系统包含三个主要组件：查询改写器（query rewriter）、检索器（retriever）和重排序器（reranker）。近期研究凸显了在 IR 系统中使用 LLM 的有效性，例如通过逐点（point-wise）和列表式（list-wise）排序方法来增强重排序阶段（Ma et al., 2023b; Sun et al., 2023a; Qin et al., 2023d）。然而，LLM 在 IR 系统中的实际应用面临挑战，主要是因为其生成速度较慢，与 IR 任务的低延迟要求相冲突（Sun et al., 2023a）。因此，LLM 的 KD 成为 IR 中更有前景的方法，它提供了一种在不牺牲速度的前提下把从 LLM 蒸馏的知识注入 IR 流程各个阶段的途径。已有大量工作展示了从 LLM 蒸馏的知识如何使 IR 系统的各个组件受益，包括查询改写器（Srinivasan et al., 2022; Ma et al., 2023c）、检索器（Dai et al., 2023b; Sachan et al., 2022, 2023; Schick and Schütze, 2021; Meng et al., 2023; Peng et al., 2023b）以及重排序器（Bonifacio et al., 2022; Sun et al., 2023a; Pradeep et al., 2023a,b; Saad-Falcon et al., 2023; Ferraretto et al., 2023; Jeronymo et al., 2023; Sun et al., 2023c）。

**查询改写器（Query Rewriter）。** 查询改写器（QR）是 IR 系统中的关键组件，负责通过改进或修改初始查询，使其更准确地契合用户的信息需求，从而提升用户查询的精确性和表达力。一个值得注意的方法是 QUILL（Srinivasan et al., 2022），它为查询意图理解提出了一种两阶段蒸馏方法。首先，把一个充当“教授（professor）”的检索增强 LLM 蒸馏进一个不带检索增强的教师 LLM，以增强其理解能力。随后，再用一个大型数据集把这个增强后的教师 LLM 蒸馏进最终的学生模型，进一步完善这一过程。为了把 QR 融入 IR 系统，Ma et al.（2023c）开发了一个“改写-检索-阅读（Rewrite-Retrieve-Read）”框架。该过程首先由 LLM 通过提示改写查询，然后进入检索增强的阅读阶段。为了把改写后的查询有效地融入 IR 系统，从 LLM 中获取的知识被蒸馏进一个紧凑的学生改写器。随后，借助强化学习，利用来自 LLM 阅读器的反馈对该改写器进行微调。

**检索器与重排序器（Retriever and Reranker）。** 在 IR 系统中，检索器被设计用于从大型语料库中高效地找出 top-k 个相关文本。它把查询和文档都编码为向量表示，并通过计算这些向量之间的点积来执行检索。重排序器进一步调整检索到的文档的顺序，以提升输出的整体质量。这主要通过两种方式实现：逐点重排序器（Pointwise Reranker）和列表式重排序器（Listwise Reranker）。逐点重排序器以查询和单个候选文档为输入，直接生成相关性分数。列表式重排序器则直接按相关性对输入的文档列表进行重新排序。

**检索器与逐点重排序器。** 对于检索器和逐点重排序器，从 LLM 进行 KD 的一种常见应用是为给定文档生成伪查询（pseudo-queries）。这种方法旨在扩展成对数据，加强稠密检索器或重排序器的训练。例如，InPars（Bonifacio et al., 2022）利用 GPT-3 为一个无标注文档生成多个伪查询。为了确保这些查询的相关性，系统根据“以文档为条件生成查询”的对数概率最高者进行过滤。随后，InPars 基于 monoT5（Raffel et al., 2020）微调一个重排序器。另一种类似方法 Promptagator（Dai et al., 2023b）提出了一种少样本稠密检索方法，利用来自目标领域的少量示例来生成伪查询。Sachan et al.（2022）不依赖无标注文档，而是从 GPT-4 中蒸馏知识，为近 100 种语言的文本嵌入任务整理出多样的合成数据。他们在这些合成数据上使用标准对比损失（contrastive loss）微调强大的仅解码器（decoder-only）LLM，如 Mistral-7b（Jiang et al., 2023a）。值得注意的是，该方法在没有任何标注数据的情况下，在文本嵌入和多语言检索基准上表现强劲。除了生成伪查询之外，教师 LLM 还可以用来生成相关性分数作为软标签。如 Sachan et al.（2023）所探索的，这些分数通过最小化教师分布与学生分布之间的 KL 散度损失来训练检索器。

**列表式重排序器（Listwise Reranker）。** 另一组研究聚焦于列表式重排序，其优势在于同时比较多个文档以确定最佳排序。RankGPT（Sun et al., 2023a）利用 GPT-4 为一组候选段落生成排列。为了把这种列表式排序知识蒸馏进一个逐点式的学生重排序器，人们采用了多种训练损失函数，如列表式交叉熵（Listwise Cross-Entropy）（Bruch et al., 2019）、RankNet（Burges et al., 2005）和 LambdaLoss（Wang et al., 2018）。在 RankGPT 框架的基础上，RankVicuna（Pradeep et al., 2023a）和 RankZephyr（Pradeep et al., 2023b）进一步改进了这一方法，直接用教师生成的文本形式的排列来微调列表式重排序器。这使学生重排序器能够直接产生排序结果序列，跳过计算单个相关性分数的中间步骤。

#### 4.4.4 推荐（Recommendation）

推荐系统对于提升各种在线服务中的用户体验不可或缺，它根据用户偏好和行为提供个性化内容。许多工作已证明，LLM 可以不经微调直接用作推荐器（Wang et al., 2023i; Dai et al., 2023c），或生成辅助性的文本特征来使推荐系统受益（Xi et al., 2023; Ren et al., 2023; Wei et al., 2024）（Wang et al., 2023j; Ren et al., 2023; Wei et al., 2024）。然而，在线推荐系统的实时性要求快速响应，而 LLM 固有的推理延迟对此构成了挑战。为了解决这一问题，若干研究探索了如何把 LLM 的知识蒸馏并整合进推荐系统，从而在利用其先进能力的同时缓解延迟问题，实现高效的实时推荐（Mysore et al., 2023; Zhang et al., 2023b; Liu et al., 2023c）。

Mysore et al.（2023）解决了叙述驱动推荐（narrative-driven recommendation, NDR）中的数据稀缺问题，在这类推荐中用户会详细描述自己的偏好。他们利用 GPT-3，通过少样本提示从“用户-物品”交互中创建合成的叙述式查询，然后把这些数据蒸馏进用于 NDR 的检索模型。类似地，GENRE（Liu et al., 2023c）利用 GPT-3.5 为数据集补充关于新闻摘要、用户画像和个性化内容的新知识，帮助训练基于内容的推荐模型。为了弥合语言模型与推荐系统之间的鸿沟，一些研究把行为建模视为语言建模的延伸（Cui et al., 2022; Liu et al., 2023k）。例如，InstructRec（Zhang et al., 2023b）把推荐解释为指令遵循。他们使用 ChatGPT，基于真实的历史交互，蒸馏出大量反映不同偏好和意图的用户个性化指令数据。随后，这些数据被用于专门为推荐目的微调一个 3B 的学生语言模型。

#### 4.4.5 文本生成评估（Text Generation Evaluation）

文本生成评估，即 NLG 评估，关注对生成内容质量的评估。传统的 NLG 评估指标，如 BLEU（Papineni et al., 2002）或 ROUGE（Lin, 2004），主要依赖表层的文本比较；与之不同，在海量语料上训练并通过 RLHF 等技术完善的 LLM 能提供更符合人类判断的评估。这种先进性使得 LLM 越来越多地被用于 NLG 评估（详见（Li et al., 2024b））。通过对 LLM 进行 KD，学生评估器可以提升推理效率，并实现更灵活、高度定制化的评估（Wang et al., 2023b; Kim et al., 2024; Xu et al., 2023d; Jiang et al., 2023c; Li et al., 2024a）。

PandaLM（Wang et al., 2023b）专注于一个用于比较两段生成内容的成对评估器。它利用教师 LLM（GPT-3.5）判断对于给定的指令和输入哪个回答更好，并给出判断理由。为了满足真实用户需求对定制化、灵活评估标准的要求，Prometheus（Kim et al., 2024）蒸馏 GPT-4，构建了一个包含参考答案和多种定制评分细则（scoring rubrics）的训练数据集。该数据集随后被用于微调 LLaMA，以评估模型生成的回答。Instructscore（Xu et al., 2023d）采取了更细粒度的方法，用 GPT-4 创建详细的分析数据。这些数据被用于微调 LLaMA，使其能够对比参考文本对生成文本进行错误分析。该系统还通过用真实的“模型生成回答-参考”对进行自训练，进一步完善其评估能力。针对跨不同领域的无参考评估，TigerScore（Jiang et al., 2023c）从各种文本生成数据集（如摘要、翻译和数据到文本）中采样数据。它从 GPT-4 中蒸馏错误分析知识，并用于微调 LLaMA。最后，为了使评估适应传统 NLP 任务之外的真实场景，Auto-J（Li et al., 2024a）收集了真实的用户查询以及教师 LLM 对它们的评估。这一海量的真实场景数据集随后通过微调把评估知识蒸馏进 LLaMA，提升了其实际适用性。

#### 4.4.6 代码（Code）

在包含代码的海量语料上训练的 LLM 以其在代码相关任务上的熟练程度而著称。它们的能力不仅限于直接生成代码，还包括提供外部知识和数据，这对于把它们的专长蒸馏进更小、更高效的模型至关重要。若干工作已成功地把代码知识从 LLM 蒸馏进紧凑的专用代码模型中（Chaudhary, 2023; Rozière et al., 2023; Gunasekar et al., 2023; Wei et al., 2023; Chen et al., 2023a; Liu et al., 2023d; Yu et al., 2024; Jain et al., 2023; Su and McMillan, 2023; Guo et al., 2023d）。

这些学生代码模型的主要关注点是代码生成，这是一项既常用又具有实际意义的任务。例如，Code Alpaca（Chaudhary, 2023）使用 self-instruct 方法，利用 ChatGPT 蒸馏出的专门面向代码生成任务的指令来微调 Llama。类似地，Code Llama-instruct（Rozière et al., 2023）通过 self-instruct 进行微调：用编程问题提示 Llama-2（Touvron et al., 2023），并用单元测试进一步完善。Phi-1（Gunasekar et al., 2023）旨在通过从教师 LLM 中提取“教科书质量”的数据（包括 Python 教科书和练习数据）来提升蒸馏代码数据的质量。Magicoder（Wei et al., 2023）通过参考大量开源代码来应对教师 LLM 中可能存在的偏差，为代码生成产出更多样、更有依据的数据。为了考虑学生模型的能力并利用教师的反馈，PERsD（Chen et al., 2023a）提出了一种个性化蒸馏（Personalized Distillation）方法：教师 LLM 根据执行器的执行反馈来改进学生生成的代码。

然而，这些模型主要针对代码生成任务，缺乏在更广泛的代码相关任务上的泛化性。为了解决这一问题，MFTCoder（Liu et al., 2023d）利用 self-instruct 从教师 LLM 中为多种任务（如代码补全和文本到代码生成）蒸馏多样的代码数据，并通过多任务学习训练学生模型。相比之下，WaveCoder（Yu et al., 2024）创建了一个全面的指令微调数据集，涵盖从 GPT-3.5-turbo 蒸馏出的四种通用代码相关任务。WaveCoder 首先使用 KCenterGreedy（Sener and Savarese, 2018）聚类方法选出一个多样化的原始数据核心集（coreset），然后利用教师 LLM 生成任务定义和输出。教师模型还在评估和过滤这些数据中发挥作用。值得注意的是，与其他开源模型相比，WaveCoder 在不同的代码相关任务上展现出更强的泛化能力。

### 4.5 多模态（Multi-Modality）

多模态大语言模型（Multimodal Large Language Models, MLLMs）能够理解和处理跨多种模态的信息，超越了传统的纯语言 LLM，更接近人类的感知方式，并支持更广泛的真实世界应用。开发能遵循多模态指令的 MLLM 正成为一种日益增长的趋势，以支持交互程度更高的任务。为了解决多模态指令遵循数据稀缺的问题，并利用教师 LLM 中蕴含的常识和世界知识，许多研究聚焦于从 LLM 进行多模态知识蒸馏（Liu et al., 2023e; Zhao et al., 2023b; Wang et al., 2023e; Chen et al., 2023c; Park et al., 2023; Pi et al., 2023; Zhao et al., 2023c; Liu et al., 2023f; Wu et al., 2023b; Luo et al., 2023d; Jiang et al., 2023d; Li et al., 2023c; Xu et al., 2023e）。

**视觉-语言（Vision-Language）。** 在视觉-语言领域，LLaVA（Liu et al., 2023e）率先把 Self-Instruct 方法从语言领域扩展到多模态领域。它把图像转换为文本描述（包括图像说明和边界框），并在种子样例的语境下蒸馏 GPT-4 来生成新数据。这一方法创建了 LLaVA-Instruct-150k 数据集，为 LLaVA-1.5（Liu et al., 2023l）和 GPT4ROI（Zhang et al., 2023e）等后续发展奠定了基础，增强了 MLLM 的指令遵循能力。为了扩大数据集规模，SVIT（Zhao et al., 2023b）利用人工图像标注从 GPT-4 中蒸馏，推出了一个 420 万规模的图像数据集。它采用一种新颖的数据配方，挑选出信息丰富、多样且均衡的训练数据子集。LVIS-Instruct4V（Wang et al., 2023e）利用强大的大型多模态模型 GPT-4V（OpenAI, 2023）作为教师，蒸馏出一个更准确、更具上下文感知能力的指令遵循数据集，侧重细粒度理解。进一步的进展包括在基于图像的指令遵循中融入对特定区域的指代。例如，Shikra（Chen et al., 2023c）使用 GPT-4 从 Flickr30K（Plummer et al., 2015）数据集中蒸馏指代性的问答对，增强对图像中被指代区域的理解。LSKD（Park et al., 2023）引入了对特定图像区域的局部指代，提示教师 LLM 生成关于这些区域的常识推断。为了用富含文字的图像增强视觉指令微调流程，LLaVAR（Zhang et al., 2023d）以纯文本的 GPT-4 作为教师，利用识别出的文字和图像说明，为富含文字的图像生成了 16K 个对话对。由此得到的学生 MLLM 在结合文字与图像的内容上展现出更强的交互能力。

**多种模态（Multiple Modalities）。** 为了把 LLM 的知识蒸馏扩展到音频和视频等更多模态，人们提出了若干创新方法。这些方法通常先把这些模态转换为教师 LLM 能理解的文本格式，然后再对教师进行蒸馏。Macaw-LLM（Lyu et al., 2023）利用 GPT-4 生成与图像或视频内容相对应的“指令-回答”对。MIMIC-IT（Li et al., 2023f）旨在把范围扩展到语言、图像和视频理解，创建了一个包含 280 万个从 ChatGPT 蒸馏出的多模态“指令-回答”对的大型数据集。另一方面，ChatBridge（Zhao et al., 2023d）代表了多模态语言建模中的一种新方法。它把各种非文本模态转换为文本，结合细粒度描述和全局描述。随后，这些信息通过上下文学习过程被用来从 ChatGPT 或 GPT-4 中蒸馏回答，有效地弥合了不同模态之间的鸿沟。

**其他（Others）。** 除了蒸馏指令遵循数据之外，还出现了若干专注于利用 LLM 知识其他方面的方法。例如，EMMA（Yang et al., 2023c）训练一个 MLLM，使其在视觉环境中充当具身反射智能体（embodied reflex agent）。它通过在一个平行的文本世界中蒸馏 GPT-4 的技能（生成动作并提供反思性反馈）来实现这一点。Silkie（Li et al., 2023h）采取了一种独特的方法，从 GPT-4V 中蒸馏偏好，关注有用性和视觉忠实性等标准。Ha et al.（2023）代表了另一个创新方向：它通过 LLM 生成、标注并蒸馏多样的以机器人为中心的探索经验，将其蒸馏进一个多任务的视觉-语言-运动（visuo-linguo-motor）策略中。

## 5 特定领域的垂直蒸馏（Domain-Specified Vertical Distillation）

本节从技能蒸馏转向考察 LLM 的 KD 在各个垂直领域中的应用，包括法律、医疗健康、金融和科学等。本节深入探讨如何为这些领域定制蒸馏后的 LLM，展示其在增强特定领域 AI 应用中的重要作用。这些工作的分类见图 7。

> **图 7**（图不翻译）：垂直化蒸馏的分类树，按法律、医疗健康、金融、科学和其他（Miscellaneous）五类列出代表性模型——法律：LawyerLLaMA（Huang et al., 2023b）、LawGPT（Cui et al., 2023b）、Fuzi（Wu et al., 2023d）；医疗健康：HuatuoGPT（Zhang et al., 2023c）、HuatuoGPT-II（Chen et al., 2023d）、DoctorGLM（Xiong et al., 2023）、AlpaCare（Zhang et al., 2023f）、Huatuo（Wang et al., 2023a）、ChatDoctor（Li et al., 2023i）、MedAlpaca（Han et al., 2023）、PMC-LLaMA（Wu et al., 2023e）、DISC-MedLLM（Bao et al., 2023a）；金融：XuanYuan（Zhang and Yang, 2023）；科学：DARWIN（Xie et al., 2023a）、SciGLM（Zhang et al., 2024）、WizardMath（Luo et al., 2023b）、MAmmoTH（Yue et al., 2023a）、TORA（Gou et al., 2024）、AstroLLaMA-Chat（Perkowski et al., 2024）、G-LLaVA（Gao et al., 2023c）、GIMLET（Zhao et al., 2023f）、LLM-Prop（Rubungo et al., 2023）、InstructMol（Cao et al., 2023a）、Prot2Text（Abdine et al., 2023）、BioMedGPT（Luo et al., 2023e）、xTrimoPGLM（Chen et al., 2024e）、K2（Deng et al., 2023）、OceanGPT（Bi et al., 2023）、MarineGPT（Zheng et al., 2023b）、GeoGalactica（Lin et al., 2024）；其他：EduChat（Dan et al., 2023）、Owl（Guo et al., 2023b）。
> 原图注：垂直化蒸馏的分类体系。

### 5.1 法律（Law）

法律在塑造社会、规范人际交往和确保正义得以伸张方面占据着关键地位。专业人士作出明智的决策、进行法律解释以及提供法律建议，都依赖于准确且最新的信息。不同场景中的法律智能应用通常需要组合法律文本检索、理解、推理和生成等多种基础能力（Zhang et al., 2023g; Sun, 2023; Lai et al., 2023）。法律术语、细微的解释差异以及立法的不断演变等方面带来了独特的挑战，需要定制化的解决方案。为应对上述挑战，若干研究探讨了为智能法律服务定制 LLM（Cui et al., 2023b; Yue et al., 2023b; Huang et al., 2023b; Wu et al., 2023d）。这通常包括在大量法律语料上进行继续预训练（continued pre-training），然后用自建指令或利用先进 LLM 增强的数据进行微调。

Huang et al.（2023b）推出了一个名为 LawyerLLaMA 的中文法律大模型。该模型首先在大量法律语料上进行预训练，系统地吸收中国法律体系的知识。随后，通过分析国家统一法律职业资格考试（Chinese National Judicial Examination）（Zhong et al., 2020）中的客观题，并使用 ChatGPT 收集法律咨询的回答来进行微调。这一过程使模型具备了把法律知识应用于具体场景的能力。Cui et al.（2023b）提出了建立在 OpenLLaMA 基础上的 LawGPT。该模型的训练数据构建过程融入了真实世界的法律文本、法律法规、司法解释和实际法律咨询数据。此外，作者还利用 ChatGPT API 辅助构建，基于现有数据集生成补充数据。Wu et al.（2023d）以 ChatGLM 为基础开发了一个大规模中文法律模型（名为“夫子”，Fuzi）。该模型在大量中文法律语料上训练，其中包含无监督的司法语言数据，如各类裁判文书和法律法规。此外，它还用涵盖法律问答和类案检索的数据进行了有监督的司法微调。Fuzi 的训练还同时用到了通用指令微调数据集（如 Alpaca）以及来自 LawyerLLaMA（Huang et al., 2023b）和 LawGPT（Cui et al., 2023b）的领域特定指令微调数据集。

### 5.2 医疗健康（Medical and Healthcare）

LLM 的融入在变革医学和医疗健康方面具有巨大潜力。大量研究聚焦于把通用 LLM 适配到医学领域（Singhal et al., 2023），如电子健康档案，以及患者护理等医疗健康应用（Zhu et al., 2023）。近期工作关注用先进的教师 LLM 增强医学指令遵循数据，以更好地契合复杂的用户指令。鉴于医学数据丰富，大多数研究把真实世界数据与从教师 LLM 蒸馏的指令数据结合起来（Zhang et al., 2023c; Xiong et al., 2023; Zhang et al., 2023f; Wang et al., 2023a; Li et al., 2023i; Han et al., 2023; Wu et al., 2023f; Bao et al., 2023a; Chen et al., 2023d）。

现有研究主要集中于使用专门的医学对话数据集进行训练，这些数据集由医学教科书（Wu et al., 2023e）、生物医学论文（Luo et al., 2023e）、医学知识图谱（Bao et al., 2023b）或真实的医患交互（Bao et al., 2023b）构成；与此同时，越来越多的研究开始探索用先进 LLM 增强医学指令遵循数据，以提升与实际用户指令的契合度。Zhang et al.（2023c）推出了专门面向医疗咨询的 HuatuoGPT（华佗GPT）。该模型在监督微调阶段同时利用了从 ChatGPT 蒸馏的数据和来自医生的真实数据。与此同时，Xiong et al.（2023）借助 ChatGPT 构建了一个中文医学对话数据集。他们的方法涵盖多种技术，用于训练 DoctorGLM——一个易于部署、面向诊断、药物推荐及其他医疗建议等任务的 LLM。Zhang et al.（2023f）使用 52k 条多样的、机器生成的医学指令遵循数据（名为 MedInstruct-52k）微调 LLaMA 系列模型。这一工作产出了 AlpaCare 模型，该模型在通用领域和医学专用领域的自由形式指令评估中都展现出强大的医学能力和泛化性。另一方面，Wang et al.（2023a）提出了 HuaTuo（华驼），这是一个基于 LLaMA、用生成的问答实例进行监督微调的模型。这一改进过程使模型掌握了更可靠的医学知识。Li et al.（2023i）推出了 ChatDoctor，它首先基于 LLaMA 被训练为一个通用对话模型，使用了来自斯坦福大学 Alpaca 项目（Taori et al., 2023）的 52K 条指令遵循数据。随后，该对话模型在从一个在线医疗咨询网站收集的 10 万条医患对话数据集上进行微调。这一两步训练过程凸显了模型对各种对话情境（尤其是医患交互特有的情境）的适应能力。

MedAlpaca（Han et al., 2023）在现有数据集的基础上，提出用 GPT-3.5-Turbo 重构数据，然后用其微调 LLM，以实现有效的医学应用。此外，PMC-LLaMA（Wu et al., 2023f）提出了一个训练框架（即继续预训练和领域特定的多任务监督微调），把通用 LLM 适配到医学领域，其中在 SFT 中利用 GPT-4 撰写同义句进行数据增强。为了使 LLM 适应真实世界的医疗咨询，DISC-MedLLM（Bao et al., 2023a）利用 GPT-3.5：1）以少样本方式构建 50K 个问答对；2）基于真实病例重新生成 420k 段对话；然后以监督微调方式用它们训练 LLM。最近，HuatuoGPT-II（Chen et al., 2023d）提出了一种用于 LLM 医学适配的单阶段训练方法，把领域数据收集统一为指令格式，其中利用 GPT-4 把医学问题整理成微调指令。

这些多样的研究共同推动了医学领域的发展，而这得益于从先进 LLM 进行的知识蒸馏。通过探索各种方法，这些工作为前沿语言模型与医学应用交叉领域中的挑战和潜在突破提供了宝贵的见解。

### 5.3 金融（Finance）

LLM 在金融领域的应用（Xue et al., 2023）显著改变了金融数据的分析方式、决策方式以及客户交互的管理方式。在金融领域，LLM 在理解复杂金融文档、预测市场趋势和自动化风险评估方面提供了前所未有的能力，从而使决策过程更加明智、迅速。通过处理和分析大量非结构化金融数据（如新闻文章、报告和实时市场信息流），LLM 能够识别出以前难以获得的模式和洞见，带来更准确的预测和战略性的财务规划。此外，LLM 还通过个性化理财建议、自动化客户服务以及能处理复杂查询的高级聊天机器人来提升客户体验。这种程度的自动化和洞察有潜力提高金融机构的效率、降低运营成本，并改进合规与风险管理实践，使 LLM 成为金融行业的一股变革性力量。从专有 LLM 进行知识蒸馏在这一领域仍探索不足，大多数现有工作专注于通过在金融专用语料上继续预训练（Wu et al., 2023g; Lu et al., 2023），或在多任务金融专用指令上进行监督微调（Yang et al., 2023e; Xie et al., 2023b; Wang et al., 2023k），把 LLM 适配到金融应用中。

具体而言，XuanYuan（轩辕）（Zhang and Yang, 2023）在种子数据上使用 self-instruct，在结构化/非结构化数据上使用 self-QA，生成金融领域的指令数据，用于训练一个金融 LLM。

### 5.4 科学（Science）

LLM 融入科学领域（Taylor et al., 2022; Yin et al., 2023b）代表着科研、知识发现和科学信息传播方面的范式转变。在科学领域，LLM 被用于消化和整合海量文献，帮助发现新的研究机会并加速科学突破。它们通过总结研究论文、生成假设，甚至起草研究计划和论文手稿，帮助人们理解复杂的科学概念，从而显著减少研究者花在文献综述上的时间，使他们能更专注于实验工作。LLM 还通过为复杂研究成果提供面向普通人的摘要，使科学知识的获取更加民主化，让非专业人士也更容易接触科学，促进公众对科学进展的更广泛理解。通过提升研究工作流程的效率并促进跨学科合作，LLM 有望加快各领域科学发现与创新的步伐。为了从 LLM 中蒸馏知识，DARWIN 系列（Xie et al., 2023a）采用一种半 self-instruct 方法为科学论文生成指令，再用这些指令微调 LLM。SciGLM（Zhang et al., 2024）提出训练一个科学 LLM：提示教师 LLM 为无标注的科学问题生成详细答案，并采用一种自我反思的“批评-修改（critic-and-revise）”机制来提升数据质量。除了上述把 LLM 适配到科学领域的知识蒸馏方法之外，我们还将深入探讨蒸馏如何在数学、航天、化学等子领域中进行。

**数学（Mathematics）。** LLM 在数学子领域的应用预示着数学研究、教育和问题求解的变革时代（Azerbayev et al., 2023; Yu et al., 2023b）。数学领域的 LLM 通过提供直观的解释、证明和解答，促进对复杂数学理论和问题的探索与理解，能够弥合高深数学概念与不同水平学习者之间的鸿沟。这些模型已展现出猜想新的数学定理和规律的潜力，从而开辟了仅凭人类自身可能难以触及的研究与发现新途径。在教育方面，它们充当个性化导师，为学生解数学题提供逐步指导，并根据学习者的理解水平调整讲解。这使高质量数学教育的获取更加民主化，并在更广泛的受众中促进对数学更深的欣赏和理解。通过生成新想法和简化复杂概念来加强协作，LLM 有望显著推动数学领域的发展，使其更易获取、更高效、更具创新性。WizardMath（Luo et al., 2023b）通过应用新颖的基于 Evol-Instruct 反馈的强化学习（Reinforcement Learning from Evol-Instruct Feedback, RLEIF）方法增强了 Llama-2 的数学推理能力，在 GSM8k 和 MATH 基准上显著优于其他开源 LLM，还超过了包括 ChatGPT-3.5 和 Minerva 在内的若干闭源 LLM。MAmmoTH（Yue et al., 2023a）是一系列专为通用数学问题求解开发的开源 LLM，在九个数学推理数据集上取得了优异的性能。借助一个名为 MathInstruct、结合了思维链和思维程序推理依据的新型指令微调数据集，MAmmoTH 模型相对现有模型展现出显著提升。TORA（Gou et al., 2024）是一系列工具集成推理智能体（Tool-integrated Reasoning Agents），通过把自然语言推理与外部计算工具的使用相结合，显著推进了数学问题求解。它在 10 个数学推理数据集上明显优于现有开源模型，相比基于推理依据和基于程序的方法都有显著提升，并引入了输出空间塑形（output space shaping）等创新训练技术来增强模型的推理能力。G-LLaVA（Gao et al., 2023c）利用结合文本和图像数据的多模态方法，在 LLM 几何问题求解方面取得了重大进展。该模型使用包含超过 17 万个几何“图像-描述”对和问答对的 Geo170K 数据集，在 MathVista 基准上相比 GPT-4V 有显著提升。

**航天（Astronautics）。** LLM 在航天领域的应用（Nguyen et al., 2023）推动了该领域向前发展。AstroLLaMA-Chat（Perkowski et al., 2024）是 AstroLLaMA 模型的进阶版本，它基于 70 亿参数的 LLaMA-2 模型，在精心整理的天文学语料上进行有针对性的继续预训练，以提升在天文学问答上的表现。该模型在专业主题理解方面表现出显著提升，并为天文学社区推出了支持对话的版本，凸显了特定领域知识蒸馏在专业主题上取得卓越性能的有效性。

**化学与材料科学（Chemistry and Materials Science）。** LLM 融入化学与材料科学，彻底改变了研究人员发现和开发新化合物与新材料的方式。通过分析海量数据集和科学文献，LLM 可以预测物质的性质和行为，显著加快创新周期。

GIMLET（Zhao et al., 2023f），即基于图指令的分子零样本学习（Graph Instruction based MolecuLe zEro-shoT learning），是一种用于分子性质预测的新方法，它在单一语言模型框架内整合图数据和文本数据，旨在改进分子任务上基于指令的零样本学习。借助带有广义位置嵌入和解耦注意力的 Transformer 机制，GIMLET 在零样本学习场景中显著优于传统的“分子-文本”基线，展示了该模型无需事先进行显式的特定任务训练、即可从指令泛化到广泛分子相关任务的有效性。LLM-Prop（Rubungo et al., 2023）基于 T5 模型，展示了 LLM 如何在根据文本描述预测晶体固体的物理和电子性质方面超越最先进（SoTA）的图神经网络。这一方法凸显了基于文本的方法在材料科学中的潜力，在显著提升预测准确性的同时，还贡献了一个基准数据集 TextEdge，以促进这一新兴领域的进一步研究。InstructMol（Cao et al., 2023a）整合多模态数据，把分子结构与自然语言指令对齐，用于药物发现任务。通过一种新颖的两阶段指令微调方法，它显著提升了在分子相关任务上的表现，建立了一个可靠的分子助手，其表现优于现有 LLM，并缩小了与专用模型之间的性能差距。这证明了多模态整合在为药物发现等复杂领域开发通用工具方面的价值。

**生物学（Biology）。** 在生物学领域，尤其是在蛋白质、DNA 和 RNA 研究方面，LLM 正在彻底改变我们对生命基本分子的理解。通过分析生物序列和结构的海量数据集，LLM 能够以超越传统计算方法的规模和速度预测蛋白质的三维结构、潜在功能和相互作用。这种能力对于揭示生物系统的复杂性、通过识别靶点和高精度设计分子来推进药物发现，以及通过解读基因组变异来理解遗传疾病都至关重要。

Prot2Text（Abdine et al., 2023）提出了一个新颖的多模态框架，通过结合图神经网络（GNN）和 LLM，以自由文本形式生成蛋白质功能描述。这种整合了蛋白质结构信息和序列信息的方法，凸显了通过融合 GNN 与 LLM 进行知识蒸馏在准确预测蛋白质功能方面的变革性影响，有望彻底改变生物信息学和生物科学的研究。BioMedGPT（Luo et al., 2023e）提出了一个专为生物医学领域设计的多模态生成式预训练 Transformer，强调对齐分子、蛋白质和自然语言模态对于增强生物医学问答、分子问答和蛋白质问答任务的重要性。该框架展示了知识蒸馏在弥合复杂生物数据与人类语言之间鸿沟方面的关键作用，从而促进药物发现和治疗靶点识别方面的突破性进展。xTrimoPGLM（Chen et al., 2024e）是一个统一的千亿（100B）规模预训练 Transformer 模型，通过整合自编码和自回归预训练目标，同时处理蛋白质理解和生成任务。它在 18 个蛋白质理解基准上相对现有模型的显著进步，以及其从头（de novo）生成蛋白质序列的能力，凸显了该模型在通过知识蒸馏推进蛋白质科学方面的重要性。

**地理学、地质学与环境科学（Geography, Geology, and Environmental Science）。** LLM 融入地理学、地质学和环境科学，正在通过增强数据分析、预测建模和跨学科研究来彻底改变这些领域（Roberts et al., 2023; Lin et al., 2023b; Wang et al., 2023l）。K2（Deng et al., 2023）是首个专门面向地球科学领域的 LLM，展示了知识蒸馏在垂直领域专精中的重要影响。通过用 55 亿 token 的地球科学语料适配通用领域的 LLaMA-7B 模型，并引入 GeoSignal 指令微调数据集，K2 在地球科学知识的理解和运用方面表现出更强的性能。该模型的开发展示了一种高效收集领域特定数据、并使模型回答契合专业用户查询的新方法。OceanGPT（Bi et al., 2023）作为首个面向海洋科学任务的 LLM，凸显了知识蒸馏在海洋学这一垂直领域中的重要作用。它利用 DOINSTRUCT——一个通过多智能体协作生成领域特定指令数据的新框架——并建立了 OCEANBENCH，一个用于评估海洋领域 LLM 的基准。MarineGPT（Zheng et al., 2023b）利用一个为海洋科学定制的新型视觉-语言模型，展示了知识蒸馏在海洋领域的变革潜力。借助包含 500 多万个海洋“图像-文本”对的 Marine-5M 数据集，MarineGPT 能出色地提供详细、准确且具有领域针对性的回答。GeoGalactica（Lin et al., 2024）代表了将 LLM 专精于地球科学的开创性一步，它基于一个在海量地球科学语料上预训练的 300 亿参数模型。该模型因是地球科学领域同类模型中规模最大的而引人注目。

### 5.5 其他（Miscellaneous）

除了前面讨论的领域之外，LLM 的知识蒸馏在其他各种垂直领域也有巨大潜力，凸显了其多面性以及对不同行业的变革性影响。例如，在教育领域，EduChat（Dan et al., 2023）是一个为教师、学生和家长提供定制化支持的聊天机器人系统的典范。KD 是其设计的核心：先在教育数据上进行预训练，再用定制指令进行微调，以提供作文评估和情感支持等能力。类似地，Owl（Guo et al., 2023b）是一个面向 IT 运维的 LLM，它使用从 ChatGPT 蒸馏得到的 Owl-Instruct 数据集来提升运维效率。通过采用混合适配器（mixture-of-adapter）策略进行领域特定微调，它增强了在 IT 相关任务中的分析能力和表现。

## 6 开放问题（Open Problems）

**进一步的数据选择（Further Data Selection）。** LLM 蒸馏需要多少数据、如何过滤掉低质量数据，仍然是开放性问题。在指令微调（最常用的蒸馏方法之一）领域，Zhou et al.（2023a）提出只需 1000 条人工整理的高质量数据就足以实现 LLM 的对齐，并假设 LLM 已在预训练中学到了所需知识，对齐只需要少量数据。这一发现进一步引出了一个新问题：如何自动选择数据以实现更好的蒸馏？Chen et al.（2023e）直接用 ChatGPT 为每个数据样本打分并给出解释，然后根据评分选择数据。Cao et al.（2023b）把现有的指令微调数据集进行划分，并训练一个线性函数，根据数据的统计特性选出最有效的数据。Li et al.（2023j）提出了一个类似自蒸馏的数据选择流程：LLM 首先从一小部分数据中学习以获得基本能力，然后再用这个学到的模型为原始数据集打分。Du et al.（2023b）提出在过滤过程中考虑质量、覆盖度和必要性三个方面。Li et al.（2023k）通过评估指令数据在一个留出集上带来的单样本（one-shot）提升来选择指令数据。Li et al.（2024f）最近提出了 Superfiltering，它能够利用 GPT-2 这样的小语言模型从给定的高质量数据集中筛选出高质量子集。尽管已出现这些关于数据过滤的工作，如何为 LLM 高效地选择最优蒸馏数据，以及蒸馏需要多少数据，仍未得到解决。

**降低蒸馏成本（轻量级方法）（Reduce the Distillation Cost (Lightweight Methods)）。** 尽管最新的 LLM 能力卓越，但其巨大的资源需求凸显了寻找高效方案来克服这些挑战的迫切性。进一步降低蒸馏成本的常见途径包括模型压缩和高效微调。在模型压缩方面，常用的有量化（Frantar et al., 2023; Dettmers et al., 2022; Kim et al., 2023c; Tao et al., 2022b; Yao et al., 2022; Xiao et al., 2023）、参数剪枝（Ma et al., 2023d; Zhang et al., 2023h; Frantar and Alistarh, 2023）以及低秩近似（Xu et al., 2023g; Li et al., 2023l）。在高效微调方面，常用的有参数高效微调（Parameter Efficient Fine-Tuning）（Hu et al., 2023b; Liu et al., 2022c; Wang et al., 2022b; Hu et al., 2021; Li and Liang, 2021; Liu et al., 2022d）以及内存高效微调（Memory Efficient Fine-Tuning）（Dettmers et al., 2023; Kim et al., 2023d; Malladi et al., 2024）。关于高效大语言模型的详细综述可参见 Wan et al.（2024b）。仍然存在的问题是：我们如何能进一步压缩模型并构建有效的蒸馏算法。

**多教师蒸馏（Multi-Teacher Distillation）。** 现有的大多数蒸馏模型都是从单个教师模型蒸馏而来的，然而人们普遍认为，用不同来源的数据训练的模型具有不同的能力。于是出现了一个问题：能否把来自不同教师模型的知识蒸馏进同一个学生模型？BabyLlama（Timiryasov and Tastet, 2023）提出把 GPT-2 和 LLaMA 两者的知识蒸馏进小规模学生模型。Ensemble-Instruct（Lee et al., 2023b）尝试以 RougeL 为指标，集成多个不同 LLM 来生成指令和回答。FUSELLM（Wan et al., 2024a）通过利用不同 LLM 的生成分布，把它们的集体知识和独特优势外化出来，旨在训练出超越任何单个源 LLM 的学生模型。尽管这一主题近来有所进展，但它仍是一个探索不足的方向。

**从教师 LLM 中探索更丰富的知识（Explore Richer Knowledge from Teacher LLMs）。** 如表 3 所示，由于能力先进，大多数教师 LLM 是闭源的。因此，当前方法主要把这些模型的生成结果用作硬标签（hard labels），通过简单的监督微调来训练学生模型。然而，除了通过硬标签直接模仿输出行为之外，人们越来越有兴趣利用教师 LLM 中更丰富的知识，包括反馈知识和特征知识，并探索知识引出方法的各种组合。正如反馈一节所强调的，教师可以根据学生的输出提供各种类型的反馈（Lee et al., 2023a; Jiang et al., 2023b; Chen et al., 2023a）。同样，特征一节讨论了基于特征的知识（如作为软标签的 logits）如何能提供关于教师模型更深层、更内在的洞察（Gu et al., 2024; Agarwal et al., 2024）。这些探索已展现出可喜的结果，表明获取更广泛的知识可以显著提升学生模型的性能，超出仅靠简单 SFT 蒸馏所能达到的水平。这凸显了进一步研究从教师 LLM 中提取知识的多种方法、以增强 KD 过程有效性的迫切需要。

**克服蒸馏过程中的灾难性遗忘（Overcoming Catastrophic Forgetting During Distillation）。** 先前的研究借助 LLM 技术的进步，深入探讨了通过微调 LLM 来获得遵循指令的能力，或为即将面对的任务、技能或领域迁移知识。然而，研究发现，在特定数据集（技能、领域）上持续微调 LLM 可能导致一种称为灾难性遗忘（catastrophic forgetting）的现象，即先前获得的知识和针对早期任务的问题求解能力受到损害（Chen et al., 2023f; Kotha et al., 2023; Koloski et al., 2023; Wu et al., 2024; Luo et al., 2023f）。机器学习和深度学习领域的早期研究探讨了在微调或持续学习过程中帮助减轻遗忘的各种技术，例如复习（rehearsal），即定期回顾并在过去的数据上训练（Kirkpatrick et al., 2017; Rostami et al., 2019; Rolnick et al., 2019），以及弹性权重巩固（elastic weight consolidation）（Lee et al., 2017）等正则化方法，或动态架构方法（Mallya et al., 2018; Wang et al., 2022c; Hu et al., 2023c; Chen et al., 2023f）。为了应对 LLM 知识蒸馏中灾难性遗忘的挑战并增强所生成指令的多样性，Jiang et al.（2023b）从简单指令中随机采样一条指令，并提示生成器生成一条与所采样指令属于同一领域的新指令。类似地，Li et al.（2023m）研究了多模态 LLM 知识蒸馏中的指令微调问题，并提出了一个竞争式蒸馏框架。在多模态增强阶段，模型尝试生成内容不同但与原始图片难度相近的新指令，从而缓解模型的灾难性遗忘并增强指令微调池的多样性。Chen et al.（2023f）基于通用语言模型提出了 Lifelong-MoE（混合专家，Mixture-of-Experts）架构，通过添加经过正则化预训练的专家来动态增加模型容量。此外，该模型还通过蒸馏旧专家和门控（gating）的知识引入隐式正则化，以有效保留旧知识。Zeng et al.（2023b）提出了一种新的基于生成的复习方法——狄利克雷持续学习（Dirichlet Continual Learning, DCL）。该方法结合任务分布建模和知识蒸馏，在无需访问旧数据的情况下减轻灾难性遗忘。为了评估指令微调在持续学习任务中的有效性，Zhang et al.（2023i）提出了一个更具挑战性但更实际的问题——持续指令微调（Continual Instruction Tuning, CIT），并建立了一个由学习和评估协议组成的基准套件。尽管当前研究已探索了一些在模型微调或知识蒸馏过程中缓解知识遗忘的简单方法，但在不同领域和技能之间有效避免灾难性遗忘仍是一个具有挑战性的问题。如何在知识蒸馏或迁移过程中有效保留原模型的能力，仍然是一个难题。

**可信的知识蒸馏（Trustworthy Knowledge Distillation）。** LLM 的可信性至关重要，涵盖真实性、安全性、公平性、鲁棒性、隐私以及遵守机器伦理等属性（Sun et al., 2024a）。LLM 的快速发展使人们对其可信性的担忧日益突出，这些担忧源于其复杂的输出、海量训练数据集中存在的偏差，以及可能包含的隐私信息。当前 LLM 的 KD 工作主要关注从 LLM 中蒸馏各种技能，对可信性方面的关注相对较少。现有研究往往只关注可信性的部分方面，如有用性、诚实性和无害性（Bai et al., 2022a; Yang et al., 2024; Cui et al., 2023a）。因此，在蒸馏过程中，学生模型可能会从其教师 LLM 那里继承与可信性相关的问题。根据 Sun et al.（2024a）的评估，较小的开源 LLM 在可信性指标上通常不及专有模型。因此，在把能力蒸馏进学生模型的同时考虑可信性至关重要。未来的 KD 研究不仅要增强学生模型的能力，还必须确保更广泛的可信性方面得到细致的处理。

**弱到强蒸馏（Weak-to-strong Distillation）。** LLM 中“弱到强泛化（weak-to-strong generalization）”的概念（Burns et al., 2023）强调了利用弱监督来引出更强大模型高级能力的潜力。这种方法挑战了传统的蒸馏范式，它表明即使监督有限或不完美，也有可能显著提升 LLM 的性能。这就需要探索创新的策略，使较弱的模型能有效地指导较强模型的学习过程，凸显了开发能弥合这些模型之间差距的方法的重要性。此类研究可以为提升 LLM 的效率和有效性开辟新途径，使对“弱到强蒸馏”的追求成为 LLM 时代未来研究的一个关键领域。最初，Burns et al.（2023）研究了弱模型的监督能否释放强得多的模型的全部能力。通过在 NLP、国际象棋和奖励建模任务上对 GPT-4 系列预训练语言模型进行实验，他们发现在弱标签上微调强模型，能得到比其弱监督者更好的性能，展示了弱到强泛化。随后，Li et al.（2024g）提出了 Superfiltering，该方法使用 GPT-2 等较小、较弱的模型来选择高质量数据，用于微调 LLaMA2 等更大、更强的模型。这种方法的依据是发现了不同规模的模型在评估指令微调数据难度方面具有很强的一致性。最近，Ji et al.（2024）提出了 Aligner，这是一种使 LLM 与人类价值观和意图对齐的新方法，它利用来自较小模型的弱监督信号来提升较大模型的性能。然而，Burns et al.（2023）发现，要实现强模型的全部能力，仅靠简单微调是不够的，这表明该领域需要进一步研究。因此，仍有一些开放问题：1）弱到强蒸馏在理论上和实践上的极限是什么？弱监督能否在所有领域中可靠地提取并增强更强模型的全部能力，还是存在基于模型架构或任务特殊性的固有局限？2）如何识别或设计用于把知识蒸馏进更强模型的最优弱监督者？是否存在某种框架或准则，可以预测哪些弱模型在针对特定任务指导更复杂模型的学习过程时最有效？3）弱到强蒸馏技术在不同规模和类型的模型之间可迁移和可扩展到什么程度？如何调整这些方法，以确保从非常大的模型向小得多的模型蒸馏知识时的效果和效率，尤其是在资源受限的环境中？

**自我对齐（Self-Alignment）。** 传统上，对齐 LLM 严重依赖人类或教师 LLM 提供大量偏好数据。因此，学生模型的对齐受限于蒸馏偏好数据的数量和教师的能力。自我对齐提供了一种很有前景的替代方案，旨在突破教师所提供偏好的限制来增强对齐。在自我对齐中，学生模型努力自主地改进其回答并使之与期望行为对齐，包括生成由模型撰写的反馈、批评和解释。若干研究探索了利用学生模型的内在能力来生成用于对齐的知识（Bai et al., 2022a; Sun et al., 2024b; Li et al., 2024c; Yuan et al., 2024a）。除了仅仅产生改进后的回答（Bai et al., 2022a; Sun et al., 2024b）之外，自我对齐的实现还包括把学生用作自身的奖励模型来提供反馈（Yuan et al., 2024a），这一策略融合了引出知识的自我知识方法与反馈方法。我们主张越来越多地利用学生模型自身来提供反馈，从而增强自我对齐能力。这种方法不仅有助于超越传统的基于人类/教师偏好的奖励，还为持续的自我改进和对齐开辟了道路。

## 7 结论与讨论（Conclusion and Discussion）

本综述探讨了 LLM 知识蒸馏的多元版图，重点阐述了关键技术、应用和挑战。KD 在使先进 LLM 能力的获取民主化方面发挥着至关重要的作用，在无需承担高昂训练和部署成本的情况下提供前沿进展。我们的综述强调了各种 KD 方法，从算法创新到技能增强再到垂直蒸馏。值得注意的是，KD 中的数据增强与数据合成已成为改进蒸馏的重要工具，揭示了丰富的训练数据与有效的模型蒸馏之间强大的协同作用。随着 AI 版图的演进，模型架构和训练方法的快速进步既给 LLM 的 KD 带来了挑战，也带来了研究机遇。未来的创新需要在保持模型可信性的同时，着力实现效率、透明度和伦理。此外，弱到强泛化、自我对齐和多模态 LLM 等有前景的方向，为增强蒸馏模型的能力提供了潜力。总之，LLM 的 KD 必将在未来的 AI 研究中发挥关键作用。正如本综述所强调的，持续的研究努力对于为所有人开发易获取、高效且负责任的 AI 至关重要。重要的是，在对 ChatGPT 或 Llama 等 LLM 进行 KD 时，必须遵守模型提供方的条款⁴，例如关于开发竞争性产品的限制。

> 脚注 4：OpenAI Business Terms（OpenAI 商业条款）：https://openai.com/policies/business-terms

> 译者注：原文未设单独的“致谢（Acknowledgments）”部分，故本译本无致谢一节。

## 参考文献（References）

> 以下参考文献按原文顺序保留英文原文（共 398 条，由 PDF 文本自动拼接换行，个别条目的排版断字可能与原 PDF 略有差异）。

- L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in Neural Information Processing Systems, vol. 35, pp. 27 730–27 744, 2022.
- OpenAI, :, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine, G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko, M. Boyd, A.-L. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen, R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry, N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gibson, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton, K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, Łukasz Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim, H. Kirchner, J. Kiros, M. Knight, D. Kokotajlo, Łukasz Kondraciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan, T. Lee, J. Leike, J. Leung, D. Levy, C. M. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov, Y. Markovski, B. Mar4. OpenAI Business Terms: https://openai.com/policies/businessterms tin, K. Mayer, A. Mayne, B. McGrew, S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Medina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. Mossing, T. Mu, M. Murati, O. Murk, D. Mély, A. Nair, R. Nakano, R. Nayak, A. Neelakantan, R. Ngo, H. Noh, L. Ouyang, C. O’Keefe, J. Pachocki, A. Paino, J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, Michael, Pokorny, M. Pokrass, V. Pong, T. Powell, A. Power, B. Power, E. Proehl, R. Puri, A. Radford, J. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. Saltarelli, T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler, M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song, N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. Tezak, M. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone, A. Vijayvergiya, C. Voss, C. Wainwright, J. J.
- Wang, A. Wang, B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter, S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao, T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph, “Gpt4 technical report,” 2023.
- G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth et al., “Gemini: a family of highly capable multimodal models,” arXiv preprint arXiv:2312.11805, 2023.
- J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus, “Emergent abilities of large language models,” Trans.
- Mach. Learn. Res., vol. 2022, 2022. [Online]. Available: https://openreview.net/forum?id=yzkSU5zdwD
- J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in Neural Information Processing Systems, vol. 35, pp. 24 824– 24 837, 2022.
- X. Xu, C. Tao, T. Shen, C. Xu, H. Xu, G. Long, and J. guang Lou, “Re-reading improves reasoning in large language models,” 2024.
- P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, B. Newman, B. Yuan, B. Yan, C. Zhang, C. Cosgrove, C. D. Manning, C. Ré, D. Acosta-Navas, D. A. Hudson, E. Zelikman, E. Durmus, F. Ladhak, F. Rong, H. Ren, H. Yao, J. Wang, K. Santhanam, L. J. Orr, L. Zheng, M. Yüksekgönül, M. Suzgun, N. Kim, N. Guha, N. S. Chatterji, O. Khattab, P. Henderson, Q. Huang, R. Chi, S. M. Xie, S. Santurkar, S. Ganguli, T. Hashimoto, T. Icard, T. Zhang, V. Chaudhary, W. Wang, X. Li, Y. Mai, Y. Zhang, and Y. Koreeda, “Holistic evaluation of language models,” CoRR, vol. abs/2211.09110, 2022. [Online]. Available: https://doi.org/10.48550/arXiv.2211.09110
- X. Wu, R. Duan, and J. Ni, “Unveiling security, privacy, and ethical concerns of chatgpt,” Journal of Information and Intelligence, 2023.
- H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom, “Llama 2: Open foundation and fine-tuned chat models,” 2023.
- A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mistral 7b,” 2023.
- L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” CoRR, vol. abs/2306.05685, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2306.05685
- L. Sun, Y. Huang, H. Wang, S. Wu, Q. Zhang, C. Gao, Y. Huang, W. Lyu, Y. Zhang, X. Li, Z. Liu, Y. Liu, Y. Wang, Z. Zhang, B. Kailkhura, C. Xiong, C. Xiao, C. Li, E. Xing, F. Huang, H. Liu, H. Ji, H. Wang, H. Zhang, H. Yao, M. Kellis, M. Zitnik, M. Jiang, M. Bansal, J. Zou, J. Pei, J. Liu, J. Gao, J. Han, J. Zhao, J. Tang, J. Wang, J. Mitchell, K. Shu, K. Xu, K.-W. Chang, L. He, L. Huang, M. Backes, N. Z. Gong, P. S. Yu, P.-Y. Chen, Q. Gu, R. Xu, R. Ying, S. Ji, S. Jana, T. Chen, T. Liu, T. Zhou, W. Wang, X. Li, X. Zhang, X. Wang, X. Xie, X. Chen, X. Wang, Y. Liu, Y. Ye, Y. Cao, Y. Chen, and Y. Zhao, “Trustllm: Trustworthiness in large language models,” 2024.
- J. Gou, B. Yu, S. J. Maybank, and D. Tao, “Knowledge distillation: A survey,” International Journal of Computer Vision, vol. 129, pp. 1789–1819, 2021.
- M. Gupta and P. Agrawal, “Compression of deep learning models for text: A survey,” ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 16, no. 4, pp. 1–55, 2022.
- S. Y. Feng, V. Gangal, J. Wei, S. Chandar, S. Vosoughi, T. Mitamura, and E. Hovy, “A survey of data augmentation approaches for nlp,” arXiv preprint arXiv:2105.03075, 2021.
- R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto, “Stanford alpaca: An instruction-following llama model,” https://github.com/ tatsu-lab/stanford alpaca, 2023.
- Y. Gu, L. Dong, F. Wei, and M. Huang, “MiniLLM: Knowledge distillation of large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum? id=5h0qf7IBZZ
- R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. R. Garea, M. Geist, and O. Bachem, “On-policy distillation of language models: Learning from self-generated mistakes,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=3zKtaqxLhW
- W. Yuan, R. Y. Pang, K. Cho, S. Sukhbaatar, J. Xu, and J. Weston, “Self-rewarding language models,” 2024.
- Z. Chen, Y. Deng, H. Yuan, K. Ji, and Q. Gu, “Self-play fine-tuning converts weak language models to strong language models,” 2024.
- Y. Huang, Y. Chen, Z. Yu, and K. McKeown, “In-context learning distillation: Transferring few-shot learning ability of pre-trained language models,” 2022.
- G. Cui, L. Yuan, N. Ding, G. Yao, W. Zhu, Y. Ni, G. Xie, Z. Liu, and M. Sun, “Ultrafeedback: Boosting language models with high-quality feedback,” arXiv preprint arXiv:2310.01377, 2023.
- S. Mukherjee, A. Mitra, G. Jawahar, S. Agarwal, H. Palangi, and A. Awadallah, “Orca: Progressive learning from complex explanation traces of gpt-4,” arXiv preprint arXiv:2306.02707, 2023.
- B. Ding, C. Qin, L. Liu, Y. K. Chia, B. Li, S. Joty, and L. Bing, “Is GPT-3 a good data annotator?” in ACL (1). Association for Computational Linguistics, 2023, pp. 11 173– 11 195.
- S. Chaudhary, “Code alpaca: An instruction-following llama model for code generation,” https://github.com/ sahil280114/codealpaca, 2023.
- H. Wang, C. Liu, N. Xi, Z. Qiang, S. Zhao, B. Qin, and T. Liu, “Huatuo: Tuning llama model with chinese medical knowledge,” arXiv preprint arXiv:2304.06975, 2023.
- LawGPT. GitHub, 2023.
- D. Zhang, Z. Hu, S. Zhoubian, Z. Du, K. Yang, Z. Wang, Y. Yue, Y. Dong, and J. Tang, “Sciglm: Training scientific language models with self-reflective instruction annotation and tuning,” CoRR, vol. abs/2401.07950, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2401. 07950
- W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,” March 2023. [Online]. Available: https://lmsys.org/blog/2023-03-30vicuna/
- C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, and D. Jiang, “Wizardlm: Empowering large language models to follow complex instructions,” arXiv preprint arXiv:2304.12244, 2023.
- W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, Y. Du, C. Yang, Y. Chen, Z. Chen, J. Jiang, R. Ren, Y. Li, X. Tang, Z. Liu, P. Liu, J.-Y. Nie, and J.-R. Wen, “A survey of large language models,” 2023.
- X. He, Z. Lin, Y. Gong, A. Jin, H. Zhang, C. Lin, J. Jiao, S. M. Yiu, N. Duan, W. Chen et al., “Annollm: Making large language models to be better crowdsourced annotators,” arXiv preprint arXiv:2303.16854, 2023.
- Y. Wang, Z. Yu, Z. Zeng, L. Yang, C. Wang, H. Chen, C. Jiang, R. Xie, J. Wang, X. Xie, W. Ye, S. Zhang, and Y. Zhang, “Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization,” 2023.
- C. Hsieh, C. Li, C. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C. Lee, and T. Pfister, “Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes,” in ACL (Findings). Association for Computational Linguistics, 2023, pp. 8003–8017.
- A. Mitra, L. D. Corro, S. Mahajan, A. Codas, C. Simoes, S. Agarwal, X. Chen, A. Razdaibiedina, E. Jones, K. Aggarwal, H. Palangi, G. Zheng, C. Rosset, H. Khanpour, and A. Awadallah, “Orca 2: Teaching small language models how to reason,” 2023.
- C. Xu, D. Guo, N. Duan, and J. J. McAuley, “Baize: An opensource chat model with parameter-efficient tuning on selfchat data,” in EMNLP. Association for Computational Linguistics, 2023, pp. 6268–6278.
- X. Yue, X. Qu, G. Zhang, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen, “Mammoth: Building math generalist models through hybrid instruction tuning,” arXiv preprint arXiv:2309.05653, 2023.
- L. Chenglin, C. Qianglong, W. Caiyu, and Z. Yin, “Mixed distillation helps smaller language model better reasoning,” 2023.
- Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning language model with self generated instructions,” arXiv preprint arXiv:2212.10560, 2022.
- Z. Sun, Y. Shen, Q. Zhou, H. Zhang, Z. Chen, D. Cox, Y. Yang, and C. Gan, “Principle-driven self-alignment of language models from scratch with minimal human supervision,” Advances in Neural Information Processing Systems, vol. 36, 2024.
- Z. Luo, C. Xu, P. Zhao, Q. Sun, X. Geng, W. Hu, C. Tao, J. Ma, Q. Lin, and D. Jiang, “Wizardcoder: Empowering code large language models with evol-instruct,” arXiv preprint arXiv:2306.08568, 2023.
- H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, and D. Zhang, “Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct,” arXiv preprint arXiv:2308.09583, 2023.
- H. Dai, Z. Liu, W. Liao, X. Huang, Y. Cao, Z. Wu, L. Zhao, S. Xu, W. Liu, N. Liu, S. Li, D. Zhu, H. Cai, L. Sun, Q. Li, D. Shen, T. Liu, and X. Li, “Auggpt: Leveraging chatgpt for text data augmentation,” 2023.
- Z. He, M. T. Ribeiro, and F. Khani, “Targeted data generation: Finding and fixing model weaknesses,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 8506–8520. [Online]. Available: https://aclanthology.org/2023.acl-long.474
- N. Ding, Y. Chen, B. Xu, Y. Qin, S. Hu, Z. Liu, M. Sun, and B. Zhou, “Enhancing chat language models by scaling high-quality instructional conversations,” in EMNLP. Association for Computational Linguistics, 2023, pp. 3029– 3051.
- S. Gunasekar, Y. Zhang, J. Aneja, C. C. T. Mendes, A. D. Giorno, S. Gopi, M. Javaheripi, P. Kauffmann, G. de Rosa, O. Saarikivi, A. Salim, S. Shah, H. S. Behl, X. Wang, S. Bubeck, R. Eldan, A. T. Kalai, Y. T. Lee, and Y. Li, “Textbooks are all you need,” 2023.
- Y. Li, S. Bubeck, R. Eldan, A. Del Giorno, S. Gunasekar, and Y. T. Lee, “Textbooks are all you need ii: phi-1.5 technical report,” arXiv preprint arXiv:2309.05463, 2023. Phi-2: The surprising power of small language models, December 2023. [Online]. Available: https://www.microsoft.com/en-us/research/blog/ phi-2-the-surprising-power-of-small-language-models/
- Y. Wei, Z. Wang, J. Liu, Y. Ding, and L. Zhang, “Magicoder: Source code is all you need,” 2023.
- Z. Yu, X. Zhang, N. Shang, Y. Huang, C. Xu, Y. Zhao, W. Hu, and Q. Yin, “Wavecoder: Widespread and versatile enhanced instruction tuning with refined data generation,” 2024.
- J. Ye, J. Gao, Q. Li, H. Xu, J. Feng, Z. Wu, T. Yu, and L. Kong, “Zerogen: Efficient zero-shot learning via dataset generation,” in EMNLP. Association for Computational Linguistics, 2022, pp. 11 653–11 669.
- J. Gao, R. Pi, Y. Lin, H. Xu, J. Ye, Z. Wu, W. Zhang, X. Liang, Z. Li, and L. Kong, “Self-guided noise-free data generation for efficient zero-shot learning,” in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, 2023. [Online]. Available: https://openreview.net/pdf?id=h5OpjGd lo6
- L. H. Bonifacio, H. Q. Abonizio, M. Fadaee, and R. F. Nogueira, “Inpars: Data augmentation for information retrieval using large language models,” CoRR, vol. abs/2202.05144, 2022. [Online]. Available: https://arxiv. org/abs/2202.05144
- I. Timiryasov and J.-L. Tastet, “Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty,” in Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, A. Warstadt, A. Mueller, L. Choshen, E. Wilcox, C. Zhuang, J. Ciro, R. Mosquera, B. Paranjabe, A. Williams, T. Linzen, and R. Cotterell, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 279–289. [Online]. Available: https://aclanthology.org/2023.conllbabylm.24
- C. Tao, L. Hou, W. Zhang, L. Shang, X. Jiang, Q. Liu, P. Luo, and N. Wong, “Compression of generative pretrained language models via quantization,” arXiv preprint arXiv:2203.10705, 2022.
- Z. Liu, B. Oguz, C. Zhao, E. Chang, P. Stock, Y. Mehdad, Y. Shi, R. Krishnamoorthi, and V. Chandra, “Llm-qat: Data-free quantization aware training for large language models,” arXiv preprint arXiv:2305.17888, 2023.
- Y. Bai, S. Kadavath, S. Kundu, A. Askell, J. Kernion, A. Jones, A. Chen, A. Goldie, A. Mirhoseini, C. McKinnon, C. Chen, C. Olsson, C. Olah, D. Hernandez, D. Drain, D. Ganguli, D. Li, E. Tran-Johnson, E. Perez, J. Kerr, J. Mueller, J. Ladish, J. Landau, K. Ndousse, K. Lukosuite, L. Lovitt, M. Sellitto, N. Elhage, N. Schiefer, N. Mercado, N. DasSarma, R. Lasenby, R. Larson, S. Ringer, S. Johnston, S. Kravec, S. E. Showk, S. Fort, T. Lanham, T. TelleenLawton, T. Conerly, T. Henighan, T. Hume, S. R. Bowman, Z. Hatfield-Dodds, B. Mann, D. Amodei, N. Joseph, S. McCandlish, T. Brown, and J. Kaplan, “Constitutional ai: Harmlessness from ai feedback,” 2022.
- L. Tunstall, E. Beeching, N. Lambert, N. Rajani, K. Rasul, Y. Belkada, S. Huang, L. von Werra, C. Fourrier, N. Habib et al., “Zephyr: Direct distillation of lm alignment,” arXiv preprint arXiv:2310.16944, 2023.
- J. Hong, Q. Tu, C. Chen, X. Gao, J. Zhang, and R. Yan, “Cyclealign: Iterative distillation from black-box llm to white-box models for better human alignment,” arXiv preprint arXiv:2310.16271, 2023.
- H. Lee, S. Phatale, H. Mansoor, K. Lu, T. Mesnard, C. Bishop, V. Carbune, and A. Rastogi, “Rlaif: Scaling reinforcement learning from human feedback with ai feedback,” arXiv preprint arXiv:2309.00267, 2023.
- Y. Jiang, C. Chan, M. Chen, and W. Wang, “Lion: Adversarial distillation of closed-source large language model,” arXiv preprint arXiv:2305.12870, 2023.
- H. Chen, A. Saha, S. Hoi, and S. Joty, “Personalized distillation: Empowering open-sourced LLMs with adaptive learning for code generation,” in The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. [Online]. Available: https://openreview. net/forum?id=alxWMBcNVN
- K. Yang, D. Klein, A. Celikyilmaz, N. Peng, and Y. Tian, “RLCD: Reinforcement learning from contrastive distillation for LM alignment,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=v3XXtxWKi6
- J. Jung, P. West, L. Jiang, F. Brahman, X. Lu, J. Fisher, T. Sorensen, and Y. Choi, “Impossible distillation: from low-quality model to high-quality dataset & model for summarization and paraphrasing,” 2023.
- J. Huang, S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han, “Large language models can self-improve,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 1051–1068. [Online]. Available: https://aclanthology.org/2023.emnlp-main.67
- C. Gulcehre, T. L. Paine, S. Srinivasan, K. Konyushkova, L. Weerts, A. Sharma, A. Siddhant, A. Ahern, M. Wang, C. Gu, W. Macherey, A. Doucet, O. Firat, and N. de Freitas, “Reinforced self-training (rest) for language modeling,” 2023.
- E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman, “Star: Bootstrapping reasoning with reasoning,” in NeurIPS, 2022.
- V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter,” arXiv preprint arXiv:1910.01108, 2019.
- Y. Wen, Z. Li, W. Du, and L. Mou, “f-divergence minimization for sequence-level knowledge distillation,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 10 817–10 834. [Online]. Available: https: //aclanthology.org/2023.acl-long.605
- C. Liang, S. Zuo, Q. Zhang, P. He, W. Chen, and T. Zhao, “Less is more: Task-aware layer-wise distillation for language model compression,” in International Conference on Machine Learning. PMLR, 2023, pp. 20 852–20 867.
- M. Kwon, S. M. Xie, K. Bullard, and D. Sadigh, “Reward design with language models,” in ICLR. OpenReview.net, 2023.
- B. Peng, C. Li, P. He, M. Galley, and J. Gao, “Instruction tuning with gpt-4,” 2023.
- G. Li, H. A. A. K. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem, “Camel: Communicative agents for” mind” exploration of large scale language model society,” arXiv preprint arXiv:2303.17760, 2023.
- G. Wang, S. Cheng, X. Zhan, X. Li, S. Song, and Y. Liu, “OpenChat: Advancing Open-source Language Models with Mixed-Quality Data,” Sep. 2023, arXiv:2309.11235 [cs]. [Online]. Available: http://arxiv.org/abs/2309.11235
- M. Kang, S. Lee, J. Baek, K. Kawaguchi, and S. J. Hwang, “Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks,” arXiv preprint arXiv:2305.18395, 2023.
- H. Luo, Y.-S. Chuang, Y. Gong, T. Zhang, Y. Kim, X. Wu, D. Fox, H. Meng, and J. Glass, “Sail: Search-augmented instruction learning,” arXiv preprint arXiv:2305.15225, 2023.
- A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, “Selfrag: Learning to retrieve, generate, and critique through self-reflection,” arXiv preprint arXiv:2310.11511, 2023.
- S. Ye, Y. Jo, D. Kim, S. Kim, H. Hwang, and M. Seo, “Selfee: Iterative self-revising llm empowered by self-feedback generation,” Blog post, May 2023. [Online]. Available: https://kaistai.github.io/SelFee/
- P. Wang, L. Li, L. Chen, F. Song, B. Lin, Y. Cao, T. Liu, and Z. Sui, “Making large language models better reasoners with alignment,” 2023.
- D. Cheng, S. Huang, and F. Wei, “Adapting large language models via reading comprehension,” 2023.
- Y. Zhang, Z. Chen, Y. Fang, L. Cheng, Y. Lu, F. Li, W. Zhang, and H. Chen, “Knowledgeable preference alignment for llms in domain-specific question answering,” 2023.
- J. Scheurer, J. A. Campos, T. Korbak, J. S. Chan, A. Chen, K. Cho, and E. Perez, “Training language models with language feedback at scale,” 2023.
- S. Kim, S. Bae, J. Shin, S. Kang, D. Kwak, K. Yoo, and M. Seo, “Aligning large language models through synthetic feedback,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 13 677–13 700. [Online]. Available: https://aclanthology. org/2023.emnlp-main.844
- P. Roit, J. Ferret, L. Shani, R. Aharoni, G. Cideron, R. Dadashi, M. Geist, S. Girgin, L. Hussenot, O. Keller, N. Momchev, S. Ramos Garea, P. Stanczyk, N. Vieillard, O. Bachem, G. Elidan, A. Hassidim, O. Pietquin, and I. Szpektor, “Factually consistent summarization via reinforcement learning with textual entailment feedback,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 6252–6272. [Online]. Available: https://aclanthology.org/2023.acl-long.344
- Y. Yang, E. Chern, X. Qiu, G. Neubig, and P. Liu, “Alignment for honesty,” arXiv preprint arXiv:2312.07000, 2023.
- R. Liu, R. Yang, C. Jia, G. Zhang, D. Zhou, A. M. Dai, D. Yang, and S. Vosoughi, “Training socially aligned language models on simulated social interactions,” 2023.
- T. Schick, J. Dwivedi-Yu, R. Dessı̀, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” 2023.
- J. Zhang, “Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt,” arXiv preprint arXiv:2304.11116, 2023.
- S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, “Gorilla: Large language model connected with massive apis,” 2023.
- Q. Tang, Z. Deng, H. Lin, X. Han, Q. Liang, B. Cao, and L. Sun, “Toolalpaca: Generalized tool learning for language models with 3000 simulated cases,” 2023.
- Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, S. Zhao, L. Hong, R. Tian, R. Xie, J. Zhou, M. Gerstein, D. Li, Z. Liu, and M. Sun, “Toolllm: Facilitating large language models to master 16000+ realworld apis,” 2023.
- L. Yuan, Y. Chen, X. Wang, Y. R. Fung, H. Peng, and H. Ji, “Craft: Customizing llms by creating and retrieving from specialized toolsets,” 2023.
- S. Gao, Z. Shi, M. Zhu, B. Fang, X. Xin, P. Ren, Z. Chen, J. Ma, and Z. Ren, “Confucius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum,” 2023.
- C. Wang, W. Luo, Q. Chen, H. Mai, J. Guo, S. Dong, Xiaohua, Xuan, Z. Li, L. Ma, and S. Gao, “Mllm-tool: A multimodal large language model for tool agent learning,” 2024.
- W. Shen, C. Li, H. Chen, M. Yan, X. Quan, H. Chen, J. Zhang, and F. Huang, “Small llms are weak tool learners: A multillm agent,” 2024.
- B. Chen, C. Shu, E. Shareghi, N. Collier, K. Narasimhan, and S. Yao, “Fireact: Toward language agent fine-tuning,” 2023.
- A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y. Dong, and J. Tang, “Agenttuning: Enabling generalized agent abilities for llms,” 2023.
- D. Yin, F. Brahman, A. Ravichander, K. Chandu, K.-W. Chang, Y. Choi, and B. Y. Lin, “Lumos: Learning agents with unified data, modular design, and open-source llms,” 2023.
- S. Qiao, N. Zhang, R. Fang, Y. Luo, W. Zhou, Y. E. Jiang, C. Lv, and H. Chen, “Autoact: Automatic agent learning from scratch via self-planning,” 2024.
- Y. Kong, J. Ruan, Y. Chen, B. Zhang, T. Bao, S. Shi, G. Du, X. Hu, H. Mao, Z. Li, X. Zeng, and R. Zhao, “Tptu-v2: Boosting task planning and tool usage of large language model-based agents in real-world systems,” 2023.
- F. Gilardi, M. Alizadeh, and M. Kubli, “Chatgpt outperforms crowd workers for text-annotation tasks,” Proceedings of the National Academy of Sciences, vol. 120, no. 30, Jul. 2023. [Online]. Available: http: //dx.doi.org/10.1073/pnas.2305016120
- Z. Wang, A. W. Yu, O. Firat, and Y. Cao, “Towards zero-label language learning,” 2021.
- Y. Xu, R. Xu, D. Iter, Y. Liu, S. Wang, C. Zhu, and M. Zeng, “InheritSumm: A general, versatile and compact summarizer by distilling from GPT,” in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 13 879–13 892. [Online]. Available: https: //aclanthology.org/2023.findings-emnlp.927
- F. Xu, W. Shi, and E. Choi, “RECOMP: Improving retrievalaugmented LMs with context compression and selective augmentation,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=mlJLVigNHp
- S. Ramnath, B. Joshi, S. Hallinan, X. Lu, L. H. Li, A. Chan, J. Hessel, Y. Choi, and X. Ren, “Tailoring self-rationalizers with multi-reward distillation,” 2023.
- S. Wang, Y. Liu, Y. Xu, C. Zhu, and M. Zeng, “Want to reduce labeling cost? GPT-3 can help,” in Findings of the Association for Computational Linguistics: EMNLP 2021, M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, Eds. Punta Cana, Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp. 4195–4205. [Online]. Available: https: //aclanthology.org/2021.findings-emnlp.354
- Z. Guo, P. Wang, Y. Wang, and S. Yu, “Improving small language models on pubmedqa via generative data augmentation,” 2023.
- W. Yang and G. Nicolai, “Neural machine translation data generation and augmentation using chatgpt,” 2023.
- K. Srinivasan, K. Raman, A. Samanta, L. Liao, L. Bertelli, and M. Bendersky, “QUILL: Query intent with large language models using retrieval augmentation and multi-stage distillation,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: Industry Track, Y. Li and A. Lazaridou, Eds. Abu Dhabi, UAE: Association for Computational Linguistics, Dec. 2022, pp. 492–501. [Online]. Available: https://aclanthology.org/2022.emnlp-industry.50
- Z. Dai, V. Y. Zhao, J. Ma, Y. Luan, J. Ni, J. Lu, A. Bakalov, K. Guu, K. B. Hall, and M. Chang, “Promptagator: Few-shot dense retrieval from 8 examples,” in The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, 2023. [Online]. Available: https://openreview.net/pdf?id=gmL46YMpu2J
- R. Meng, Y. Liu, S. Yavuz, D. Agarwal, L. Tu, N. Yu, J. Zhang, M. Bhat, and Y. Zhou, “Augtriever: Unsupervised dense retrieval by scalable data augamentation,” 2023.
- W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, and Z. Ren, “Is chatgpt good at search? investigating large language models as re-ranking agents,” 2023.
- R. Pradeep, S. Sharifymoghaddam, and J. Lin, “Rankvicuna: Zero-shot listwise document reranking with open-source large language models,” 2023.
- ——, “Rankzephyr: Effective and robust zero-shot listwise reranking is a breeze!” 2023.
- F. Ferraretto, T. Laitz, R. Lotufo, and R. Nogueira, “Exaranker: Synthetic explanations improve neural rankers,” in Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, ser. SIGIR ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 2409–2414. [Online]. Available: https://doi.org/10.1145/3539618.3592067
- S. Mysore, A. Mccallum, and H. Zamani, “Large language model augmented narrative driven recommendations,” in Proceedings of the 17th ACM Conference on Recommender Systems, ser. RecSys ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 777–783. [Online]. Available: https://doi.org/10.1145/3604915.3608829
- J. Zhang, R. Xie, Y. Hou, W. X. Zhao, L. Lin, and J.-R. Wen, “Recommendation as instruction following: A large language model empowered recommendation approach,” 2023.
- Q. Liu, N. Chen, T. Sakai, and X.-M. Wu, “Once: Boosting content-based recommendation with both open- and closed-source large language models,” 2023.
- S. Kim, J. Shin, Y. Cho, J. Jang, S. Longpre, H. Lee, S. Yun, S. Shin, S. Kim, J. Thorne, and M. Seo, “Prometheus: Inducing evaluation capability in language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=8euJaTveKw
- W. Xu, D. Wang, L. Pan, Z. Song, M. Freitag, W. Wang, and L. Li, “INSTRUCTSCORE: Towards explainable text generation evaluation with automatic feedback,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 5967–5994. [Online]. Available: https://aclanthology.org/2023.emnlp-main.365
- D. Jiang, Y. Li, G. Zhang, W. Huang, B. Y. Lin, and W. Chen, “Tigerscore: Towards building explainable metric for all text generation tasks,” 2023.
- J. Li, S. Sun, W. Yuan, R.-Z. Fan, hai zhao, and P. Liu, “Generative judge for evaluating alignment,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum? id=gtkFw6sZGS B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E.
- Tan, Y. Adi, J. Liu, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov, J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez, J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve, “Code llama: Open foundation models for code,” 2023.
- B. Liu, C. Chen, C. Liao, Z. Gong, H. Wang, Z. Lei, M. Liang, D. Chen, M. Shen, H. Zhou, H. Yu, and J. Li, “Mftcoder: Boosting code llms with multitask fine-tuning,” 2023.
- N. Jain, T. Zhang, W. Chiang, J. E. Gonzalez, K. Sen, and I. Stoica, “Llm-assisted code cleaning for training accurate code generators,” CoRR, vol. abs/2311.14904, 2023. [Online]. Available: https://doi.org/10.48550/ arXiv.2311.14904
- H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in NeurIPS, 2023.
- B. Zhao, B. Wu, M. He, and T. Huang, “Svit: Scaling up visual instruction tuning,” 2023.
- J. Wang, L. Meng, Z. Weng, B. He, Z. Wu, and Y.-G. Jiang, “To see is to believe: Prompting gpt-4v for better visual instruction tuning,” 2023.
- K. Chen, Z. Zhang, W. Zeng, R. Zhang, F. Zhu, and R. Zhao, “Shikra: Unleashing multimodal llm’s referential dialogue magic,” 2023.
- J. S. Park, J. Hessel, K. R. Chandu, P. P. Liang, X. Lu, P. West, Y. Yu, Q. Huang, J. Gao, A. Farhadi, and Y. Choi, “Localized symbolic knowledge distillation for visual commonsense models,” 2023.
- R. Pi, J. Gao, S. Diao, R. Pan, H. Dong, J. Zhang, L. Yao, J. Han, H. Xu, L. Kong, and T. Zhang, “DetGPT: Detect what you need via reasoning,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 14 172–14 189. [Online]. Available: https: //aclanthology.org/2023.emnlp-main.876
- L. Zhao, E. Yu, Z. Ge, J. Yang, H. Wei, H. Zhou, J. Sun, Y. Peng, R. Dong, C. Han, and X. Zhang, “Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning,” 2023.
- F. Liu, K. Lin, L. Li, J. Wang, Y. Yacoob, and L. Wang, “Mitigating hallucination in large multi-modal models via robust instruction tuning,” 2023.
- S. Wu, H. Fei, L. Qu, W. Ji, and T.-S. Chua, “Next-gpt: Anyto-any multimodal llm,” 2023.
- R. Luo, Z. Zhao, M. Yang, J. Dong, D. Li, P. Lu, T. Wang, L. Hu, M. Qiu, and Z. Wei, “Valley: Video assistant with large language model enhanced ability,” 2023.
- Y. Jiang, E. Schoop, A. Swearngin, and J. Nichols, “Iluvui: Instruction-tuned language-vision modeling of uis from machine conversations,” 2023.
- Y. Li, C. Zhang, G. Yu, Z. Wang, B. Fu, G. Lin, C. Shen, L. Chen, and Y. Wei, “Stablellava: Enhanced visual instruction tuning with synthesized image-dialogue data,” 2023.
- R. Xu, X. Wang, T. Wang, Y. Chen, J. Pang, and D. Lin, “Pointllm: Empowering large language models to understand point clouds,” 2023.
- Q. Huang, M. Tao, Z. An, C. Zhang, C. Jiang, Z. Chen, Z. Wu, and Y. Feng, “Lawyer llama technical report,” arXiv preprint arXiv:2305.15062, 2023.
- J. Cui, Z. Li, Y. Yan, B. Chen, and L. Yuan, “Chatlaw: Opensource legal large language model with integrated external knowledge bases,” arXiv preprint arXiv:2306.16092, 2023.
- H. Zhang, J. Chen, F. Jiang, F. Yu, Z. Chen, G. Chen, J. Li, X. Wu, Z. Zhiyi, Q. Xiao, X. Wan, B. Wang, and H. Li, “HuatuoGPT, towards taming language model to be a doctor,” in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 10 859– 10 885. [Online]. Available: https://aclanthology.org/ 2023.findings-emnlp.725
- J. Chen, X. Wang, A. Gao, F. Jiang, S. Chen, H. Zhang, D. Song, W. Xie, C. Kong, J. Li, X. Wan, H. Li, and B. Wang, “Huatuogpt-ii, one-stage training for medical adaption of llms,” CoRR, vol. abs/2311.09774, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2311.09774
- X. Zhang and Q. Yang, “Xuanyuan 2.0: A large chinese financial chat model with hundreds of billions parameters,” in Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, CIKM 2023, Birmingham, United Kingdom, October 2125, 2023, I. Frommholz, F. Hopfgartner, M. Lee, M. Oakes, M. Lalmas, M. Zhang, and R. L. T. Santos, Eds. ACM, 2023, pp. 4435–4439. [Online]. Available: https://doi.org/10.1145/3583780.3615285
- T. Xie, Y. Wan, W. Huang, Z. Yin, Y. Liu, S. Wang, Q. Linghu, C. Kit, C. Grazian, W. Zhang, I. Razzak, and B. Hoex, “DARWIN series: Domain specific large language models for natural science,” CoRR, vol. abs/2308.13565, 2023. [Online]. Available: https: //doi.org/10.48550/arXiv.2308.13565
- Y. Dan, Z. Lei, Y. Gu, Y. Li, J. Yin, J. Lin, L. Ye, Z. Tie, Y. Zhou, Y. Wang, A. Zhou, Z. Zhou, Q. Chen, J. Zhou, L. He, and X. Qiu, “Educhat: A large-scale language model-based chatbot system for intelligent education,” CoRR, vol. abs/2308.02773, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2308.02773
- H. Guo, J. Yang, J. Liu, L. Yang, L. Chai, J. Bai, J. Peng, X. Hu, C. Chen, D. Zhang, X. Shi, T. Zheng, L. Zheng, B. Zhang, K. Xu, and Z. Li, “OWL: A large language model for IT operations,” CoRR, vol. abs/2309.09298, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2309.09298
- Y. Kim and A. M. Rush, “Sequence-level knowledge distillation,” arXiv preprint arXiv:1606.07947, 2016.
- S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding,” International Conference on Learning Representations (ICLR), 2016.
- V. Gangal, S. Y. Feng, M. Alikhani, T. Mitamura, and E. Hovy, “Nareor: The narrative reordering problem,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 10, 2022, pp. 10 645–10 653.
- S. Longpre, Y. Lu, Z. Tu, and C. DuBois, “An exploration of data augmentation and sampling techniques for domainagnostic question answering,” in Proceedings of the 2nd Workshop on Machine Reading for Question Answering, A. Fisch, A. Talmor, R. Jia, M. Seo, E. Choi, and D. Chen, Eds. Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 220–227. [Online]. Available: https://aclanthology.org/D19-5829
- P. West, C. Bhagavatula, J. Hessel, J. Hwang, L. Jiang, R. Le Bras, X. Lu, S. Welleck, and Y. Choi, “Symbolic knowledge distillation: from general language models to commonsense models,” in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, Eds.
- Seattle, United States: Association for Computational Linguistics, Jul. 2022, pp. 4602–4625. [Online]. Available: https://aclanthology.org/2022.naacl-main.341
- Z. Li, X. Xu, T. Shen, C. Xu, J.-C. Gu, and C. Tao, “Leveraging large language models for nlg evaluation: A survey,” 2024.
- S. Li, J. Chen, Y. Shen, Z. Chen, X. Zhang, Z. Li, H. Wang, J. Qian, B. Peng, Y. Mao, W. Chen, and X. Yan, “Explanations from large language models make small reasoners better,” 2022.
- N. Ho, L. Schmid, and S. Yun, “Large language models are reasoning teachers,” in ACL (1). Association for Computational Linguistics, 2023, pp. 14 852–14 882.
- L. C. Magister, J. Mallinson, J. Adamek, E. Malmi, and A. Severyn, “Teaching small language models to reason,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 1773–1781. [Online]. Available: https://aclanthology.org/2023.acl-short.151
- Y. Fu, H. Peng, L. Ou, A. Sabharwal, and T. Khot, “Specializing smaller language models towards multi-step reasoning,” 2023.
- L. H. Li, J. Hessel, Y. Yu, X. Ren, K.-W. Chang, and Y. Choi, “Symbolic chain-of-thought distillation: Small models can also “think” step-by-step,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 2665–2679. [Online]. Available: https://aclanthology.org/2023.acllong.150
- W. Liu, G. Li, K. Zhang, B. Du, Q. Chen, X. Hu, H. Xu, J. Chen, and J. Wu, “Mind’s mirror: Distilling selfevaluation capability and comprehensive thinking from large language models,” 2023.
- S. Longpre, L. Hou, T. Vu, A. Webson, H. W. Chung, Y. Tay, D. Zhou, Q. V. Le, B. Zoph, J. Wei et al., “The flan collection: Designing data and methods for effective instruction tuning,” arXiv preprint arXiv:2301.13688, 2023.
- Y. Anand, Z. Nussbaum, B. Duderstadt, B. Schmidt, and A. Mulyar, “Gpt4all: Training an assistant-style chatbot with large scale data distillation from gpt-3.5-turbo,” GitHub, 2023.
- Q. Si, T. Wang, Z. Lin, X. Zhang, Y. Cao, and W. Wang, “An empirical study of instruction-tuning large language models in chinese,” in EMNLP (Findings). Association for Computational Linguistics, 2023, pp. 4086–4107.
- Y. Ji, Y. Deng, Y. Gong, Y. Peng, Q. Niu, L. Zhang, B. Ma, and X. Li, “Exploring the impact of instruction data scaling on large language models: An empirical study on real-world use cases,” 2023.
- M. Wu, A. Waheed, C. Zhang, M. Abdul-Mageed, and A. F. Aji, “Lamini-lm: A diverse herd of distilled models from large-scale instructions,” 2023.
- W. Guo, J. Yang, K. Yang, X. Li, Z. Rao, Y. Xu, and D. Niu, “Instruction fusion: Advancing prompt evolution through hybridization,” 2023.
- Y. Yu, Y. Zhuang, J. Zhang, Y. Meng, A. Ratner, R. Krishna, J. Shen, and C. Zhang, “Large language model as attributed training data generator: A tale of diversity and bias,” 2023.
- F. Wan, X. Huang, D. Cai, X. Quan, W. Bi, and S. Shi, “Knowledge fusion of large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum? id=jiDsk12qcz
- Q. Zhao and B. Zhu, “Towards the fundamental limits of knowledge transfer over finite domains,” in NeurIPS 2023 Workshop on Mathematics of Modern Machine Learning, 2023. [Online]. Available: https: //openreview.net/forum?id=9qxoXqxa0N
- C. Qin, W. Xia, F. Jiao, and S. Joty, “Improving in-context learning via bidirectional alignment,” 2023.
- N. Boizard, K. El-Haddad, C. Hudelot, and P. Colombo, “Towards cross-tokenizer distillation: the universal logit distillation loss for llms,” arXiv preprint arXiv:2402.12030, 2024.
- Q. Zhong, L. Ding, L. Shen, J. Liu, B. Du, and D. Tao, “Revisiting knowledge distillation for autoregressive language models,” 2024.
- M. Kim, S. Lee, J. Lee, S. Hong, D.-S. Chang, W. Sung, and J. Choi, “Token-scaled logit distillation for ternary weight generative language models,” arXiv preprint arXiv:2308.06744, 2023.
- Z. Chen, K. Zhou, W. X. Zhao, J. Wan, F. Zhang, D. Zhang, and J.-R. Wen, “Improving large language models via finegrained reinforcement learning with minimum editing constraint,” 2024.
- G. Guo, R. Zhao, T. Tang, X. Zhao, and J.-R. Wen, “Beyond imitation: Leveraging fine-grained quality signals for alignment,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=LNLjU5C5dK
- Z. Allen-Zhu and Y. Li, “Towards understanding ensemble, knowledge distillation and self-distillation in deep learning,” arXiv preprint arXiv:2012.09816, 2020.
- T. Zheng, S. Guo, X. Qu, J. Guo, W. Zhang, X. Du, C. Lin, W. Huang, W. Chen, J. Fu et al., “Kun: Answer polishment for chinese self-alignment with instruction backtranslation,” arXiv preprint arXiv:2401.06477, 2024.
- X. Li, P. Yu, C. Zhou, T. Schick, O. Levy, L. Zettlemoyer, J. E. Weston, and M. Lewis, “Self-alignment with instruction backtranslation,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=1oijHJBRsT
- B. Zhao, H. Hajishirzi, and Q. Cao, “Apt: Adaptive pruning and tuning pretrained language models for efficient training and inference,” arXiv preprint arXiv:2401.12200, 2024.
- A. Singh, J. D. Co-Reyes, R. Agarwal, A. Anand, P. Patil, P. J. Liu, J. Harrison, J. Lee, K. Xu, A. Parisi et al., “Beyond human data: Scaling self-training for problem-solving with language models,” arXiv preprint arXiv:2312.06585, 2023.
- W. Chen, D. Song, and B. Li, “Grath: Gradual self-truthifying for large language models,” 2024.
- A. Hosseini, X. Yuan, N. Malkin, A. Courville, A. Sordoni, and R. Agarwal, “V-star: Training verifiers for self-taught reasoners,” 2024.
- A. Askell, Y. Bai, A. Chen, D. Drain, D. Ganguli, T. Henighan, A. Jones, N. Joseph, B. Mann, N. DasSarma, N. Elhage, Z. Hatfield-Dodds, D. Hernandez, J. Kernion, K. Ndousse, C. Olsson, D. Amodei, T. Brown, J. Clark, S. McCandlish, C. Olah, and J. Kaplan, “A general language assistant as a laboratory for alignment,” 2021.
- J. Huang, S. Gu, L. Hou, Y. Wu, X. Wang, H. Yu, and J. Han, “Large language models can self-improve,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 1051–1068. [Online]. Available: https://aclanthology.org/2023.emnlp-main.67
- H. Chen, X. Quan, H. Chen, M. Yan, and J. Zhang, “Knowledge distillation for closed-source language models,” arXiv preprint arXiv:2401.07013, 2024.
- I. Sason and S. Verdú, “f -divergence inequalities,” IEEE Transactions on Information Theory, vol. 62, no. 11, pp. 5973– 6006, 2016.
- S. Sun, Y. Cheng, Z. Gan, and J. Liu, “Patient knowledge distillation for bert model compression,” 2019.
- Z. Sun, H. Yu, X. Song, R. Liu, Y. Yang, and D. Zhou, “MobileBERT: a compact task-agnostic BERT for resource-limited devices,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, Jul. 2020, pp. 2158–2170. [Online]. Available: https://aclanthology.org/2020.acl-main.195
- X. Jiao, Y. Yin, L. Shang, X. Jiang, X. Chen, L. Li, F. Wang, and Q. Liu, “TinyBERT: Distilling BERT for natural language understanding,” in Findings of the Association for Computational Linguistics: EMNLP 2020, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 4163–4174. [Online]. Available: https://aclanthology.org/2020.findings-emnlp.372
- L. Hou, Z. Huang, L. Shang, X. Jiang, X. Chen, and Q. Liu, “Dynabert: Dynamic bert with adaptive width and depth,” Advances in Neural Information Processing Systems, vol. 33, pp. 9782–9793, 2020.
- S. Zuo, Q. Zhang, C. Liang, P. He, T. Zhao, and W. Chen, “Moebert: from bert to mixture-of-experts via importanceguided adaptation,” arXiv preprint arXiv:2204.07675, 2022.
- K. J. Liang, W. Hao, D. Shen, Y. Zhou, W. Chen, C. Chen, and L. Carin, “Mixkd: Towards efficient distillation of largescale language models,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. [Online]. Available: https://openreview.net/forum?id=UFGEelJkLu5
- Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-level reward design via coding large language models,” 2023.
- J.-C. Pang, P. Wang, K. Li, X.-H. Chen, J. Xu, Z. Zhang, and Y. Yu, “Language model self-improvement by reinforcement learning contemplation,” 2023.
- Y. Du, O. Watkins, Z. Wang, C. Colas, T. Darrell, P. Abbeel, A. Gupta, and J. Andreas, “Guiding pretraining in reinforcement learning with large language models,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 8657–8677. [Online]. Available: https://proceedings.mlr.press/v202/du23f.html
- J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017.
- R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” 2023.
- F. Song, B. Yu, M. Li, H. Yu, F. Huang, Y. Li, and H. Wang, “Preference ranking optimization for human alignment,” arXiv preprint arXiv:2306.17492, 2023.
- Z. Yuan, H. Yuan, C. Tan, W. Wang, S. Huang, and F. Huang, “Rrhf: Rank responses to align language models with human feedback without tears,” arXiv preprint arXiv:2304.05302, 2023.
- M. Li, L. Chen, J. Chen, S. He, and T. Zhou, “Reflection-tuning: Recycling data for better instructiontuning,” in NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following, 2023. [Online]. Available: https://openreview.net/forum?id=xaqoZZqkPU
- M. Li, L. Chen, J. Chen, S. He, J. Gu, and T. Zhou, “Selective reflection-tuning: Student-selected data recycling for llm instruction-tuning,” 2024. [Online]. Available: https: //api.semanticscholar.org/CorpusID:267682220
- X. Geng, A. Gudibande, H. Liu, E. Wallace, P. Abbeel, S. Levine, and D. Song, “Koala: A dialogue model for academic research,” Blog post, April 2023. [Online]. Available: https://bair.berkeley.edu/blog/2023/04/03/ koala/
- M. Li, J. Chen, L. Chen, and T. Zhou, “Can llms speak for diverse people? tuning llms via debate to generate controllable controversial statements,” 2024.
- M. Kang, S. Lee, J. Baek, K. Kawaguchi, and S. J. Hwang, “Knowledge-augmented reasoning distillation for small language models in knowledge-intensive tasks,” 2023.
- R. Yang, L. Song, Y. Li, S. Zhao, Y. Ge, X. Li, and Y. Shan, “Gpt4tools: Teaching large language model to use tools via self-instruction,” 2023.
- A. Yehudai, B. Carmeli, Y. Mass, O. Arviv, N. Mills, A. Toledo, E. Shnarch, and L. Choshen, “Genie: Achieving human parity in content-grounded datasets generation,” 2024.
- Y. Zhang, R. Zhang, J. Gu, Y. Zhou, N. Lipka, D. Yang, and T. Sun, “Llavar: Enhanced visual instruction tuning for text-rich image understanding,” 2023.
- C. Lyu, M. Wu, L. Wang, X. Huang, B. Liu, Z. Du, S. Shi, and Z. Tu, “Macaw-llm: Multi-modal language modeling with image, audio, video, and text integration,” arXiv preprint arXiv:2306.09093, 2023.
- B. Li, Y. Zhang, L. Chen, J. Wang, F. Pu, J. Yang, C. Li, and Z. Liu, “Mimic-it: Multi-modal in-context instruction tuning,” 2023.
- Z. Zhao, L. Guo, T. Yue, S. Chen, S. Shao, X. Zhu, Z. Yuan, and J. Liu, “Chatbridge: Bridging modalities with large language model as a language catalyst,” 2023.
- Y. Zhao, B. Yu, B. Hui, H. Yu, F. Huang, Y. Li, and N. L. Zhang, “A preliminary study of the intrinsic relationship between complexity and alignment,” 2023.
- A. Gudibande, E. Wallace, C. Snell, X. Geng, H. Liu, P. Abbeel, S. Levine, and D. Song, “The false promise of imitating proprietary llms,” arXiv preprint arXiv:2305.15717, 2023.
- C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. YU, S. Zhang, G. Ghosh, M. Lewis, L. Zettlemoyer, and O. Levy, “LIMA: Less is more for alignment,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=KBMOKmX2he
- M. Li, Y. Zhang, S. He, Z. Li, H. Zhao, J. Wang, N. Cheng, and T. Zhou, “Superfiltering: Weak-to-strong data filtering for fast instruction-tuning,” 2024. [Online]. Available: https://api.semanticscholar.org/CorpusID:267365346
- B. Xu, A. Yang, J. Lin, Q. Wang, C. Zhou, Y. Zhang, and Z. Mao, “Expertprompting: Instructing large language models to be distinguished experts,” 2023.
- W. Liu, W. Zeng, K. He, Y. Jiang, and J. He, “What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning,” 2023.
- R. Lou, K. Zhang, J. Xie, Y. Sun, J. Ahn, H. Xu, Y. Su, and W. Yin, “Muffin: Curating multi-faceted instructions for improving instruction-following,” 2023.
- T. Schick, J. Dwivedi-Yu, Z. Jiang, F. Petroni, P. Lewis, G. Izacard, Q. You, C. Nalmpantis, E. Grave, and S. Riedel, “Peer: A collaborative language model,” 2022.
- A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark, “Self-refine: Iterative refinement with self-feedback,” 2023.
- W. Saunders, C. Yeh, J. Wu, S. Bills, L. Ouyang, J. Ward, and J. Leike, “Self-critiquing models for assisting human evaluators,” 2022.
- D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving, “Fine-tuning language models from human preferences,” arXiv preprint arXiv:1909.08593, 2019.
- N. Stiennon, L. Ouyang, J. Wu, D. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. F. Christiano, “Learning to summarize with human feedback,” Advances in Neural Information Processing Systems, vol. 33, pp. 3008–3021, 2020.
- J. Wu, L. Ouyang, D. M. Ziegler, N. Stiennon, R. Lowe, J. Leike, and P. Christiano, “Recursively summarizing books with human feedback,” 2021.
- Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan et al., “Training a helpful and harmless assistant with reinforcement learning from human feedback,” arXiv preprint arXiv:2204.05862, 2022.
- A. Köpf, Y. Kilcher, D. von Rütte, S. Anagnostidis, Z.-R. Tam, K. Stevens, A. Barhoum, N. M. Duc, O. Stanley, R. Nagyfi, S. ES, S. Suri, D. Glushkov, A. Dantuluri, A. Maguire, C. Schuhmann, H. Nguyen, and A. Mattick, “Openassistant conversations – democratizing large language model alignment,” 2023.
- G. Wang, S. Cheng, X. Zhan, X. Li, S. Song, and Y. Liu, “Openchat: Advancing open-source language models with mixed-quality data,” 2023.
- L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P.S. Huang, M. Cheng, M. Glaese, B. Balle, A. Kasirzadeh, Z. Kenton, S. Brown, W. Hawkins, T. Stepleton, C. Biles, A. Birhane, J. Haas, L. Rimell, L. A. Hendricks, W. Isaac, S. Legassick, G. Irving, and I. Gabriel, “Ethical and social risks of harm from language models,” 2021.
- J. Ji, M. Liu, J. Dai, X. Pan, C. Zhang, C. Bian, C. Zhang, R. Sun, Y. Wang, and Y. Yang, “Beavertails: Towards improved safety alignment of llm via a human-preference dataset,” 2023.
- I. Solaiman and C. Dennison, “Process for adapting language models to society (palms) with values-targeted datasets,” Advances in Neural Information Processing Systems, vol. 34, pp. 5861–5873, 2021.
- L. Qiu, Y. Zhao, J. Li, P. Lu, B. Peng, J. Gao, and S.-C. Zhu, “Valuenet: A new dataset for human value driven dialogue system,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 10, 2022, pp. 11 183– 11 191.
- J. Kiesel, M. Alshomary, N. Handke, X. Cai, H. Wachsmuth, and B. Stein, “Identifying the human values behind arguments,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 4459–4471. [Online]. Available: https://aclanthology.org/2022.acl-long.306
- R. Liu, G. Zhang, X. Feng, and S. Vosoughi, “Aligning generative language models with human values,” in Findings of the Association for Computational Linguistics: NAACL 2022, M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, Eds. Seattle, United States: Association for Computational Linguistics, Jul. 2022, pp. 241– 252. [Online]. Available: https://aclanthology.org/2022. findings-naacl.18
- A. Glaese, N. McAleese, M. Trebacz, J. Aslanides, V. Firoiu, T. Ewalds, M. Rauh, L. Weidinger, M. Chadwick, P. Thacker et al., “Improving alignment of dialogue agents via targeted human judgements,” arXiv preprint arXiv:2209.14375, 2022.
- H. Sun, Z. Zhang, F. Mi, Y. Wang, W. Liu, J. Cui, B. Wang, Q. Liu, and M. Huang, “MoralDial: A framework to train and evaluate moral dialogue systems via moral discussions,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 2213–2230. [Online]. Available: https://aclanthology.org/2023.acl-long.123
- J. Yao, X. Yi, X. Wang, J. Wang, and X. Xie, “From instructions to intrinsic human values – a survey of alignment goals for big models,” 2023.
- Y. Liu, Y. Yao, J.-F. Ton, X. Zhang, R. G. H. Cheng, Y. Klochkov, M. F. Taufiq, and H. Li, “Trustworthy llms: a survey and guideline for evaluating large language models’ alignment,” arXiv preprint arXiv:2308.05374, 2023.
- J. Qian, H. Wang, Z. Li, S. Li, and X. Yan, “Limitations of language models in arithmetic and symbolic induction,” 2022.
- X. She, Y. Liu, Y. Zhao, Y. He, L. Li, C. Tantithamthavorn, Z. Qin, and H. Wang, “Pitfalls in language models for code intelligence: A taxonomy and survey,” 2023.
- H. Manikandan, Y. Jiang, and J. Z. Kolter, “Language models are weak learners,” 2023.
- Y. Liang, C. Wu, T. Song, W. Wu, Y. Xia, Y. Liu, Y. Ou, S. Lu, L. Ji, S. Mao, Y. Wang, L. Shou, M. Gong, and N. Duan, “Taskmatrix.ai: Completing tasks by connecting foundation models with millions of apis,” 2023.
- G. Mialon, R. Dessı̀, M. Lomeli, C. Nalmpantis, R. Pasunuru, R. Raileanu, B. Rozière, T. Schick, J. DwivediYu, A. Celikyilmaz, E. Grave, Y. LeCun, and T. Scialom, “Augmented language models: a survey,” 2023.
- A. Parisi, Y. Zhao, and N. Fiedel, “Talm: Tool augmented language models,” 2022.
- R. Nakano, J. Hilton, S. Balaji, J. Wu, L. Ouyang, C. Kim, C. Hesse, S. Jain, V. Kosaraju, W. Saunders, X. Jiang, K. Cobbe, T. Eloundou, G. Krueger, K. Button, M. Knight, B. Chess, and J. Schulman, “Webgpt: Browser-assisted question-answering with human feedback,” 2022.
- Y. Qin, Z. Cai, D. Jin, L. Yan, S. Liang, K. Zhu, Y. Lin, X. Han, N. Ding, H. Wang, R. Xie, F. Qi, Z. Liu, M. Sun, and J. Zhou, “WebCPM: Interactive web search for Chinese long-form question answering,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 8968–8988. [Online]. Available: https://aclanthology.org/2023.acl-long.499
- Y. Song, W. Xiong, D. Zhu, W. Wu, H. Qian, M. Song, H. Huang, C. Li, K. Wang, R. Yao, Y. Tian, and S. Li, “Restgpt: Connecting large language models with realworld restful apis,” 2023.
- T. Cai, X. Wang, T. Ma, X. Chen, and D. Zhou, “Large language models as tool makers,” 2023.
- Y. Shen, K. Song, X. Tan, D. Li, W. Lu, and Y. Zhuang, “Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face,” 2023.
- S. Hao, T. Liu, Z. Wang, and Z. Hu, “Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings,” 2024.
- S. Yuan, K. Song, J. Chen, X. Tan, Y. Shen, R. Kan, D. Li, and D. Yang, “Easytool: Enhancing llm-based agents with concise tool instruction,” 2024.
- S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, “Opt: Open pre-trained transformer language models,” 2022.
- T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.
- W. Huang, P. Abbeel, D. Pathak, and I. Mordatch, “Language models as zero-shot planners: Extracting actionable knowledge for embodied agents,” in International Conference on Machine Learning. PMLR, 2022, pp. 9118–9147.
- I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg, “Progprompt: Generating situated robot task plans using large language models,” 2022.
- D. Zhou, N. Schärli, L. Hou, J. Wei, N. Scales, X. Wang, D. Schuurmans, C. Cui, O. Bousquet, Q. Le, and E. Chi, “Least-to-most prompting enables complex reasoning in large language models,” 2023.
- C. H. Song, J. Wu, C. Washington, B. M. Sadler, W.-L. Chao, and Y. Su, “Llm-planner: Few-shot grounded planning for embodied agents with large language models,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2998–3009.
- Z. Wang, S. Cai, A. Liu, X. Ma, and Y. Liang, “Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents,” arXiv preprint arXiv:2302.01560, 2023.
- S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan, “Tree of thoughts: Deliberate problem solving with large language models,” arXiv preprint arXiv:2305.10601, 2023.
- B. Liu, Y. Jiang, X. Zhang, Q. Liu, S. Zhang, J. Biswas, and P. Stone, “Llm+ p: Empowering large language models with optimal planning proficiency,” arXiv preprint arXiv:2304.11477, 2023.
- S. Hao, Y. Gu, H. Ma, J. J. Hong, Z. Wang, D. Z. Wang, and Z. Hu, “Reasoning with language model is planning with world model,” arXiv preprint arXiv:2305.14992, 2023.
- M. Hu, Y. Mu, X. Yu, M. Ding, S. Wu, W. Shao, Q. Chen, B. Wang, Y. Qiao, and P. Luo, “Tree-planner: Efficient close-loop task planning with large language models,” arXiv preprint arXiv:2310.08582, 2023.
- B. Y. Lin, C. Huang, Q. Liu, W. Gu, S. Sommerer, and X. Ren, “On grounded planning for embodied tasks with language models,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 11, 2023, pp. 13 192– 13 200.
- K. Valmeekam, M. Marquez, S. Sreedharan, and S. Kambhampati, “On the planning abilities of large language models - a critical investigation,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=X6dEqXIsEW
- T. Sumers, K. Marino, A. Ahuja, R. Fergus, and I. Dasgupta, “Distilling internet-scale vision-language models into embodied agents,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023.
- Y. Yang, T. Zhou, K. Li, D. Tao, L. Li, L. Shen, X. He, J. Jiang, and Y. Shi, “Embodied multi-modal agent trained by an llm from a parallel textworld,” 2023.
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.
- Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized bert pretraining approach,” 2019.
- J. Li, L. Gui, Y. Zhou, D. West, C. Aloisi, and Y. He, “Distilling chatgpt for explainable automated student answer assessment,” in EMNLP (Findings). Association for Computational Linguistics, 2023, pp. 6007–6026.
- R. Tang, X. Han, X. Jiang, and X. Hu, “Does synthetic data generation of llms help clinical text mining?” arXiv preprint arXiv:2303.04360, 2023.
- X. He, I. Nassar, J. Kiros, G. Haffari, and M. Norouzi, “Generate, annotate, and learn: NLP with synthetic text,” Trans. Assoc. Comput. Linguistics, vol. 10, pp. 826–842, 2022. [Online]. Available: https://transacl.org/ojs/index. php/tacl/article/view/3811
- Y. Meng, J. Huang, Y. Zhang, and J. Han, “Generating training data with language models: Towards zero-shot language understanding,” in Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. [Online]. Available: http://papers.nips.cc/paper files/ paper/2022/hash/0346c148ba1c21c6b4780a961ea141dcAbstract-Conference.html
- J. Wang, Z. Yao, A. Mitra, S. Osebe, Z. Yang, and H. Yu, “UMASS BioNLP at MEDIQA-chat 2023: Can LLMs generate high-quality synthetic note-oriented doctorpatient conversations?” in Proceedings of the 5th Clinical Natural Language Processing Workshop, T. Naumann, A. Ben Abacha, S. Bethard, K. Roberts, and A. Rumshisky, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 460–471. [Online]. Available: https://aclanthology.org/2023.clinicalnlp-1.49
- Z. Yang, S. Cherian, and S. Vucetic, “Data augmentation for radiology report simplification,” in Findings of the Association for Computational Linguistics: EACL 2023, A. Vlachos and I. Augenstein, Eds. Dubrovnik, Croatia: Association for Computational Linguistics, May 2023, pp. 1922–1932. [Online]. Available: https: //aclanthology.org/2023.findings-eacl.144
- Z. Cai, C. Tao, T. Shen, C. Xu, X. Geng, X. A. Lin, L. He, and D. Jiang, “Hyper: Multitask hyper-prompted training enables large-scale retrieval generalization,” in The Eleventh International Conference on Learning Representations, 2022.
- C. Liu, C. Tao, X. Geng, T. Shen, D. Zhao, C. Xu, B. Jiao, and D. Jiang, “Adam: Dense retrieval distillation with adaptive dark examples,” arXiv preprint arXiv:2212.10192, 2022.
- J. Feng, C. Tao, X. Geng, T. Shen, C. Xu, G. Long, D. Zhao, and D. Jiang, “Knowledge refinement via interaction between search engines and large language models,” arXiv preprint arXiv:2305.07402, 2023.
- T. Shen, G. Long, X. Geng, C. Tao, T. Zhou, and D. Jiang, “Large language models are strong zero-shot retriever,” arXiv preprint arXiv:2304.14233, 2023.
- X. Ma, X. Zhang, R. Pradeep, and J. Lin, “Zero-shot listwise document reranking with a large language model,” 2023.
- Z. Qin, R. Jagerman, K. Hui, H. Zhuang, J. Wu, J. Shen, T. Liu, J. Liu, D. Metzler, X. Wang, and M. Bendersky, “Large language models are effective text rankers with pairwise ranking prompting,” 2023.
- X. Ma, Y. Gong, P. He, H. Zhao, and N. Duan, “Query rewriting in retrieval-augmented large language models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 5303–5315. [Online]. Available: https://aclanthology.org/2023.emnlp-main.322
- D. Sachan, M. Lewis, M. Joshi, A. Aghajanyan, W.t. Yih, J. Pineau, and L. Zettlemoyer, “Improving passage retrieval with zero-shot question generation,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 3781–3797. [Online]. Available: https://aclanthology.org/2022.emnlp-main.249
- D. S. Sachan, M. Lewis, D. Yogatama, L. Zettlemoyer, J. Pineau, and M. Zaheer, “Questions are all you need to train a dense passage retriever,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 600–616, 2023. [Online]. Available: https://aclanthology. org/2023.tacl-1.35
- T. Schick and H. Schütze, “Generating datasets with pretrained language models,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, Eds. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp. 6943–6951. [Online]. Available: https: //aclanthology.org/2021.emnlp-main.555
- Z. Peng, X. Wu, and Y. Fang, “Soft prompt tuning for augmenting dense retrieval with large language models,” arXiv preprint arXiv:2307.08303, 2023.
- J. Saad-Falcon, O. Khattab, K. Santhanam, R. Florian, M. Franz, S. Roukos, A. Sil, M. A. Sultan, and C. Potts, “UDAPDR: unsupervised domain adaptation via LLM prompting and distillation of rerankers,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, 2023, pp. 11 265–11 279. [Online]. Available: https://aclanthology.org/2023.emnlp-main.693
- V. Jeronymo, L. Bonifacio, H. Abonizio, M. Fadaee, R. Lotufo, J. Zavrel, and R. Nogueira, “Inpars-v2: Large language models as efficient dataset generators for information retrieval,” arXiv preprint arXiv:2301.01820, 2023.
- W. Sun, Z. Chen, X. Ma, L. Yan, S. Wang, P. Ren, Z. Chen, D. Yin, and Z. Ren, “Instruction distillation makes large language models efficient zero-shot rankers,” 2023.
- C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” J. Mach. Learn. Res., vol. 21, no. 1, jan 2020.
- S. Bruch, X. Wang, M. Bendersky, and M. Najork, “An analysis of the softmax cross entropy loss for learningto-rank with binary relevance,” in Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval, ICTIR 2019, Santa Clara, CA, USA, October 2-5, 2019, 2019, pp. 75–78. [Online]. Available: https://doi.org/10.1145/3341981.3344221
- C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, and G. Hullender, “Learning to rank using gradient descent,” in Proceedings of the 22nd International Conference on Machine Learning, ser. ICML ’05. New York, NY, USA: Association for Computing Machinery, 2005, p. 89–96. [Online]. Available: https: //doi.org/10.1145/1102351.1102363
- X. Wang, C. Li, N. Golbandi, M. Bendersky, and M. Najork, “The lambdaloss framework for ranking metric optimization,” in Proceedings of the 27th ACM International Conference on Information and Knowledge Management, ser. CIKM ’18. New York, NY, USA: Association for Computing Machinery, 2018, p. 1313–1322. [Online]. Available: https://doi.org/10.1145/3269206.3271784
- W. Wang, X. Lin, F. Feng, X. He, and T.-S. Chua, “Generative recommendation: Towards next-generation recommender paradigm,” 2023.
- S. Dai, N. Shao, H. Zhao, W. Yu, Z. Si, C. Xu, Z. Sun, X. Zhang, and J. Xu, “Uncovering chatgpt’s capabilities in recommender systems,” in Proceedings of the 17th ACM Conference on Recommender Systems, ser. RecSys ’23. New York, NY, USA: Association for Computing Machinery, 2023, p. 1126–1132. [Online]. Available: https://doi.org/10.1145/3604915.3610646
- Y. Xi, W. Liu, J. Lin, X. Cai, H. Zhu, J. Zhu, B. Chen, R. Tang, W. Zhang, R. Zhang, and Y. Yu, “Towards openworld recommendation with knowledge augmentation from large language models,” 2023.
- X. Ren, W. Wei, L. Xia, L. Su, S. Cheng, J. Wang, D. Yin, and C. Huang, “Representation learning with large language models for recommendation,” 2023.
- W. Wei, X. Ren, J. Tang, Q. Wang, L. Su, S. Cheng, J. Wang, D. Yin, and C. Huang, “Llmrec: Large language models with graph augmentation for recommendation,” 2024.
- L. Wang, S. Zhang, Y. Wang, E.-P. Lim, and Y. Wang, “LLM4Vis: Explainable visualization recommendation using ChatGPT,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, M. Wang and I. Zitouni, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 675–692. [Online]. Available: https://aclanthology.org/ 2023.emnlp-industry.64
- Z. Cui, J. Ma, C. Zhou, J. Zhou, and H. Yang, “M6-rec: Generative pretrained language models are open-ended recommender systems,” 2022.
- P. Liu, L. Zhang, and J. A. Gulla, “Pre-train, prompt and recommendation: A comprehensive survey of language modelling paradigm adaptations in recommender systems,” 2023.
- K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for automatic evaluation of machine translation,” in Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, ser. ACL ’02. USA: Association for Computational Linguistics, 2002, p. 311–318. [Online]. Available: https://doi.org/10.3115/1073083.1073135
- C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out. Barcelona, Spain: Association for Computational Linguistics, Jul. 2004, pp. 74–81. [Online]. Available: https://aclanthology.org/W04-1013
- C. Su and C. McMillan, “Distilled GPT for source code summarization,” CoRR, vol. abs/2308.14731, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2308. 14731
- W. Guo, J. Yang, K. Yang, X. Li, Z. Rao, Y. Xu, and D. Niu, “Instruction fusion: Advancing prompt evolution through hybridization,” CoRR, vol. abs/2312.15692, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2312.15692
- O. Sener and S. Savarese, “Active learning for convolutional neural networks: A core-set approach,” in 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings, 2018. [Online]. Available: https://openreview.net/forum?id=H1aIuk-RW
- H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” 2023.
- S. Zhang, P. Sun, S. Chen, M. Xiao, W. Shao, W. Zhang, Y. Liu, K. Chen, and P. Luo, “Gpt4roi: Instruction tuning large language model on region-of-interest,” 2023.
- OpenAI, “Gpt-4v(ision) system card,” 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID: 263218031
- B. A. Plummer, L. Wang, C. M. Cervantes, J. C. Caicedo, J. Hockenmaier, and S. Lazebnik, “Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2641– 2649.
- L. Li, Z. Xie, M. Li, S. Chen, P. Wang, L. Chen, Y. Yang, B. Wang, and L. Kong, “Silkie: Preference distillation for large visual language models,” arXiv preprint arXiv:2312.10665, 2023.
- H. Ha, P. Florence, and S. Song, “Scaling up and distilling down: Language-guided robot skill acquisition,” in Conference on Robot Learning. PMLR, 2023, pp. 3766–3777.
- S. Wu, Z. Liu, Z. Zhang, Z. Chen, W. Deng, W. Zhang, J. Yang, Z. Yao, Y. Lyu, X. Xin, S. Gao, P. Ren, Z. Ren, and Z. Chen, “fuzi.mingcha,” https://github.com/irlabsdu/fuzi.mingcha, 2023.
- H. Xiong, S. Wang, Y. Zhu, Z. Zhao, Y. Liu, Q. Wang, and D. Shen, “Doctorglm: Fine-tuning your chinese doctor is not a herculean task,” arXiv preprint arXiv:2304.01097, 2023.
- X. Zhang, C. Tian, X. Yang, L. Chen, Z. Li, and L. R. Petzold, “Alpacare: Instruction-tuned large language models for medical application,” arXiv preprint arXiv:2310.14558, 2023.
- Y. Li, Z. Li, K. Zhang, R. Dan, S. Jiang, and Y. Zhang, “Chatdoctor: A medical chat model fine-tuned on a large language model meta-ai (llama) using medical domain knowledge,” Cureus, vol. 15, no. 6, 2023.
- T. Han, L. C. Adams, J. Papaioannou, P. Grundmann, T. Oberhauser, A. Löser, D. Truhn, and K. K. Bressem, “Medalpaca - an open-source collection of medical conversational AI models and training data,” CoRR, vol. abs/2304.08247, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2304.08247
- C. Wu, W. Lin, X. Zhang, Y. Zhang, Y. Wang, and W. Xie, “Pmc-llama: Towards building open-source language models for medicine,” arXiv preprint arXiv:2305.10415, vol. 6, 2023.
- Z. Bao, W. Chen, S. Xiao, K. Ren, J. Wu, C. Zhong, J. Peng, X. Huang, and Z. Wei, “Disc-medllm: Bridging general large language models and real-world medical consultation,” CoRR, vol. abs/2308.14346, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2308.14346
- Z. Gou, Z. Shao, Y. Gong, yelong shen, Y. Yang, M. Huang, N. Duan, and W. Chen, “ToRA: A toolintegrated reasoning agent for mathematical problem solving,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https: //openreview.net/forum?id=Ep0TtjVoap
- E. Perkowski, R. Pan, T. D. Nguyen, Y. Ting, S. Kruk, T. Zhang, C. O’Neill, M. Jablonska, Z. Sun, M. J. Smith, H. Liu, K. Schawinski, K. Iyer, I. Ciuca, and UniverseTBD, “Astrollama-chat: Scaling astrollama with conversational and diverse datasets,” CoRR, vol. abs/2401.01916, 2024. [Online]. Available: https: //doi.org/10.48550/arXiv.2401.01916
- J. Gao, R. Pi, J. Zhang, J. Ye, W. Zhong, Y. Wang, L. Hong, J. Han, H. Xu, Z. Li, and L. Kong, “G-llava: Solving geometric problem with multi-modal large language model,” CoRR, vol. abs/2312.11370, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2312.11370
- H. Zhao, S. Liu, C. Ma, H. Xu, J. Fu, Z.-H. Deng, L. Kong, and Q. Liu, “GIMLET: A unified graph-text model for instruction-based molecule zero-shot learning,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview. net/forum?id=Tt6DrRCgJV
- A. N. Rubungo, C. Arnold, B. P. Rand, and A. B. Dieng, “Llm-prop: Predicting physical and electronic properties of crystalline solids from their text descriptions,” CoRR, vol. abs/2310.14029, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2310.14029
- H. Cao, Z. Liu, X. Lu, Y. Yao, and Y. Li, “Instructmol: Multi-modal integration for building a versatile and reliable molecular assistant in drug discovery,” CoRR, vol. abs/2311.16208, 2023. [Online]. Available: https: //doi.org/10.48550/arXiv.2311.16208
- H. Abdine, M. Chatzianastasis, C. Bouyioukos, and M. Vazirgiannis, “Prot2text: Multimodal protein’s function generation with GNNs and transformers,” in Deep Generative Models for Health Workshop NeurIPS 2023, 2023. [Online]. Available: https://openreview.net/forum?id=EJ7YNgWYFj
- Y. Luo, J. Zhang, S. Fan, K. Yang, Y. Wu, M. Qiao, and Z. Nie, “Biomedgpt: Open multimodal generative pre-trained transformer for biomedicine,” arXiv preprint arXiv:2308.09442, 2023.
- B. Chen, X. Cheng, P. Li, Y. Geng, J. Gong, S. Li, Z. Bei, X. Tan, B. Wang, X. Zeng, C. Liu, A. Zeng, Y. Dong, J. Tang, and L. Song, “xtrimopglm: Unified 100b-scale pre-trained transformer for deciphering the language of protein,” CoRR, vol. abs/2401.06199, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2401.06199
- C. Deng, T. Zhang, Z. He, Y. Xu, Q. Chen, Y. Shi, L. Fu, W. Zhang, X. Wang, C. Zhou, Z. Lin, and J. He, “K2: A foundation language model for geoscience knowledge understanding and utilization,” 2023.
- Z. Bi, N. Zhang, Y. Xue, Y. Ou, D. Ji, G. Zheng, and H. Chen, “Oceangpt: A large language model for ocean science tasks,” CoRR, vol. abs/2310.02031, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2310.02031
- Z. Zheng, J. Zhang, T. Vu, S. Diao, Y. H. W. Tim, and S. Yeung, “Marinegpt: Unlocking secrets of ocean to the public,” CoRR, vol. abs/2310.13596, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2310.13596
- Z. Lin, C. Deng, L. Zhou, T. Zhang, Y. Xu, Y. Xu, Z. He, Y. Shi, B. Dai, Y. Song, B. Zeng, Q. Chen, T. Shi, T. Huang, Y. Xu, S. Wang, L. Fu, W. Zhang, J. He, C. Ma, Y. Zhu, X. Wang, and C. Zhou, “Geogalactica: A scientific large language model in geoscience,” CoRR, vol. abs/2401.00434, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2401.00434
- D. Zhang, A. Petrova, D. Trautmann, and F. Schilder, “Unleashing the power of large language models for legal applications,” in Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 2023, pp. 5257–5258.
- Z. Sun, “A short survey of viewing large language models in legal aspect,” arXiv preprint arXiv:2303.09136, 2023.
- J. Lai, W. Gan, J. Wu, Z. Qi, and P. S. Yu, “Large language models in law: A survey,” arXiv preprint arXiv:2312.03718, 2023.
- S. Yue, W. Chen, S. Wang, B. Li, C. Shen, S. Liu, Y. Zhou, Y. Xiao, S. Yun, W. Lin et al., “Disc-lawllm: Fine-tuning large language models for intelligent legal services,” arXiv preprint arXiv:2309.11325, 2023.
- H. Zhong, C. Xiao, C. Tu, T. Zhang, Z. Liu, and M. Sun, “Jec-qa: a legal-domain question answering dataset,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020, pp. 9701–9708.
- K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, L. Hou, K. Clark, S. Pfohl, H. Cole-Lewis, D. Neal, M. Schaekermann, A. Wang, M. Amin, S. Lachgar, P. A. Mansfield, S. Prakash, B. Green, E. Dominowska, B. A. y Arcas, N. Tomasev, Y. Liu, R. Wong, C. Semturs, S. S. Mahdavi, J. K. Barral, D. R. Webster, G. S. Corrado, Y. Matias, S. Azizi, A. Karthikesalingam, and V. Natarajan, “Towards expert-level medical question answering with large language models,” CoRR, vol. abs/2305.09617, 2023. [Online]. Available: https://doi. org/10.48550/arXiv.2305.09617
- W. Zhu, X. Wang, H. Zheng, M. Chen, and B. Tang, “Promptcblue: A chinese prompt tuning benchmark for the medical domain,” arXiv preprint arXiv:2310.14151, 2023.
- C. Wu, X. Zhang, Y. Zhang, Y. Wang, and W. Xie, “Pmcllama: Further finetuning llama on medical papers,” CoRR, vol. abs/2304.14454, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2304.14454
- Z. Bao, W. Chen, S. Xiao, K. Ren, J. Wu, C. Zhong, J. Peng, X. Huang, and Z. Wei, “Disc-medllm: Bridging general large language models and real-world medical consultation,” arXiv preprint arXiv:2308.14346, 2023.
- S. Xue, F. Zhou, Y. Xu, H. Zhao, S. Xie, Q. Dai, C. Jiang, J. Zhang, J. Zhou, D. Xiu, and H. Mei, “Weaverbird: Empowering financial decision-making with large language model, knowledge base, and search engine,” CoRR, vol. abs/2308.05361, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2308.05361
- S. Wu, O. Irsoy, S. Lu, V. Dabravolski, M. Dredze, S. Gehrmann, P. Kambadur, D. S. Rosenberg, and G. Mann, “Bloomberggpt: A large language model for finance,” CoRR, vol. abs/2303.17564, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2303.17564
- D. Lu, H. Wu, J. Liang, Y. Xu, Q. He, Y. Geng, M. Han, Y. Xin, and Y. Xiao, “Bbt-fin: Comprehensive construction of chinese financial domain pre-trained language model, corpus and benchmark,” CoRR, vol. abs/2302.09432, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2302. 09432
- Y. Yang, Y. Tang, and K. Y. Tam, “Investlm: A large language model for investment using financial domain instruction tuning,” CoRR, vol. abs/2309.13064, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2309.13064
- Q. Xie, W. Han, X. Zhang, Y. Lai, M. Peng, A. LopezLira, and J. Huang, “PIXIU: A large language model, instruction data and evaluation benchmark for finance,” CoRR, vol. abs/2306.05443, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2306.05443
- N. Wang, H. Yang, and C. D. Wang, “Fingpt: Instruction tuning benchmark for open-source large language models in financial datasets,” CoRR, vol. abs/2310.04793, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2310. 04793
- R. Taylor, M. Kardas, G. Cucurull, T. Scialom, A. Hartshorn, E. Saravia, A. Poulton, V. Kerkez, and R. Stojnic, “Galactica: A large language model for science,” CoRR, vol. abs/2211.09085, 2022. [Online]. Available: https://doi.org/10.48550/arXiv.2211.09085
- J. Yin, S. Dash, F. Wang, and M. Shankar, “FORGE: pre-training open foundation models for science,” in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2023, Denver, CO, USA, November 12-17, 2023, D. Arnold, R. M. Badia, and K. M. Mohror, Eds.
- ACM, 2023, pp. 81:1–81:13. [Online]. Available: https: //doi.org/10.1145/3581784.3613215
- Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Q. Jiang, J. Deng, S. Biderman, and S. Welleck, “Llemma: An open language model for mathematics,” CoRR, vol. abs/2310.10631, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2310.10631
- F. Yu, A. Gao, and B. Wang, “Outcome-supervised verifiers for planning in mathematical reasoning,” CoRR, vol. abs/2311.09724, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2311.09724
- T. D. Nguyen, Y. Ting, I. Ciuca, C. O’Neill, Z. Sun, M. Jablonska, S. Kruk, E. Perkowski, J. W. Miller, J. Li, J. Peek, K. Iyer, T. Rózanski, P. Khetarpal, S. Zaman, D. Brodrick, S. J. R. Méndez, T. Bui, A. Goodman, A. Accomazzi, J. P. Naiman, J. Cranney, K. Schawinski, and UniverseTBD, “Astrollama: Towards specialized foundation models in astronomy,” CoRR, vol. abs/2309.06126, 2023. [Online]. Available: https: //doi.org/10.48550/arXiv.2309.06126
- J. Roberts, T. Lüddecke, S. Das, K. Han, and S. Albanie, “Gpt4geo: How a language model sees the world’s geography,” 2023.
- Z. Lin, C. Deng, L. Zhou, T. Zhang, Y. Xu, Y. Xu, Z. He, Y. Shi, B. Dai, Y. Song, B. Zeng, Q. Chen, T. Shi, T. Huang, Y. Xu, S. Wang, L. Fu, W. Zhang, J. He, C. Ma, Y. Zhu, X. Wang, and C. Zhou, “Geogalactica: A scientific large language model in geoscience,” 2023.
- C. Wang, D. Engler, X. Li, J. Hou, D. J. Wald, K. Jaiswal, and S. Xu, “Near-real-time earthquake-induced fatality estimation using crowdsourced data and large-language models,” 2023.
- L. Chen, S. Li, J. Yan, H. Wang, K. Gunaratna, V. Yadav, Z. Tang, V. Srinivasan, T. Zhou, H. Huang, and H. Jin, “Alpagasus: Training a better alpaca with fewer data,” 2023.
- Y. Cao, Y. Kang, and L. Sun, “Instruction mining: Highquality instruction data selection for large language models,” 2023.
- M. Li, Y. Zhang, Z. Li, J. Chen, L. Chen, N. Cheng, J. Wang, T. Zhou, and J. Xiao, “From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning,” ArXiv, vol. abs/2308.12032, 2023. [Online]. Available: https://api.semanticscholar. org/CorpusID:261076515
- Q. Du, C. Zong, and J. Zhang, “Mods: Model-oriented data selection for instruction tuning,” 2023.
- Y. Li, B. Hui, X. Xia, J. Yang, M. Yang, L. Zhang, S. Si, J. Liu, T. Liu, F. Huang, and Y. Li, “One shot learning as instruction data prospector for large language models,” 2023.
- E. Frantar, S. P. Singh, and D. Alistarh, “Optimal brain compression: A framework for accurate post-training quantization and pruning,” 2023.
- T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, “GPT3.int8(): 8-bit matrix multiplication for transformers at scale,” in Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. [Online]. Available: https://openreview.net/ forum?id=dXiGWqBoxaD
- Y. J. Kim, R. Henry, R. Fahim, and H. H. Awadalla, “Finequant: Unlocking efficiency with fine-grained weight-only quantization for llms,” 2023.
- C. Tao, L. Hou, W. Zhang, L. Shang, X. Jiang, Q. Liu, P. Luo, and N. Wong, “Compression of generative pre-trained language models via quantization,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 4821–4836. [Online]. Available: https://aclanthology.org/2022.acllong.331
- Z. Yao, R. Yazdani Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He, “Zeroquant: Efficient and affordable post-training quantization for large-scale transformers,” Advances in Neural Information Processing Systems, vol. 35, pp. 27 168– 27 183, 2022.
- G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “Smoothquant: Accurate and efficient post-training quantization for large language models,” 2023.
- X. Ma, G. Fang, and X. Wang, “Llm-pruner: On the structural pruning of large language models,” 2023.
- M. Zhang, H. Chen, C. Shen, Z. Yang, L. Ou, X. Yu, and B. Zhuang, “Loraprune: Pruning meets low-rank parameter-efficient fine-tuning,” 2023.
- E. Frantar and D. Alistarh, “Sparsegpt: Massive language models can be accurately pruned in one-shot,” 2023.
- M. Xu, Y. L. Xu, and D. P. Mandic, “Tensorgpt: Efficient compression of the embedding layer in llms based on the tensor-train decomposition,” 2023.
- Y. Li, Y. Yu, Q. Zhang, C. Liang, P. He, W. Chen, and T. Zhao, “Losparse: Structured compression of large language models based on low-rank and sparse approximation,” 2023.
- Z. Hu, L. Wang, Y. Lan, W. Xu, E.-P. Lim, L. Bing, X. Xu, S. Poria, and R. K.-W. Lee, “Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models,” 2023.
- H. Liu, D. Tam, M. Mohammed, J. Mohta, T. Huang, M. Bansal, and C. Raffel, “Few-shot parameterefficient fine-tuning is better and cheaper than incontext learning,” in Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. [Online]. Available: https: //openreview.net/forum?id=rBCvMG-JsPd
- Y. Wang, S. Agarwal, S. Mukherjee, X. Liu, J. Gao, A. H. Awadallah, and J. Gao, “AdaMix: Mixtureof-adaptations for parameter-efficient model tuning,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Y. Goldberg, Z. Kozareva, and Y. Zhang, Eds. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, Dec. 2022, pp. 5744–5760. [Online]. Available: https://aclanthology.org/2022.emnlp-main.388
- E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” 2021.
- X. L. Li and P. Liang, “Prefix-tuning: Optimizing continuous prompts for generation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong, F. Xia, W. Li, and R. Navigli, Eds. Online: Association for Computational Linguistics, Aug. 2021, pp. 4582–4597. [Online]. Available: https://aclanthology.org/2021.acllong.353
- X. Liu, K. Ji, Y. Fu, W. Tam, Z. Du, Z. Yang, and J. Tang, “Ptuning: Prompt tuning can be comparable to fine-tuning across scales and tasks,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 61–68. [Online]. Available: https://aclanthology.org/2022.acl-short.8
- T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “Qlora: Efficient finetuning of quantized llms,” 2023.
- J. Kim, J. H. Lee, S. Kim, J. Park, K. M. Yoo, S. J. Kwon, and D. Lee, “Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization,” 2023.
- S. Malladi, T. Gao, E. Nichani, A. Damian, J. D. Lee, D. Chen, and S. Arora, “Fine-tuning language models with just forward passes,” 2024.
- Z. Wan, X. Wang, C. Liu, S. Alam, Y. Zheng, J. Liu, Z. Qu, S. Yan, Y. Zhu, Q. Zhang, M. Chowdhury, and M. Zhang, “Efficient large language models: A survey,” 2024.
- Y.-S. Lee, M. Sultan, Y. El-Kurdi, T. Naseem, A. Munawar, R. Florian, S. Roukos, and R. Astudillo, “Ensembleinstruct: Instruction tuning data generation with a heterogeneous mixture of LMs,” in Findings of the Association for Computational Linguistics: EMNLP 2023, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 12 561–12 571. [Online]. Available: https://aclanthology. org/2023.findings-emnlp.836
- W. Chen, Y. Zhou, N. Du, Y. Huang, J. Laudon, Z. Chen, and C. Cui, “Lifelong language pretraining with distributionspecialized experts,” in International Conference on Machine Learning. PMLR, 2023, pp. 5383–5395.
- S. Kotha, J. M. Springer, and A. Raghunathan, “Understanding catastrophic forgetting in language models via implicit inference,” arXiv preprint arXiv:2309.10105, 2023.
- B. Koloski, B. Škrlj, M. Robnik-Šikonja, and S. Pollak, “Measuring catastrophic forgetting in cross-lingual transfer paradigms: Exploring tuning strategies,” arXiv preprint arXiv:2309.06089, 2023.
- T. Wu, L. Luo, Y.-F. Li, S. Pan, T.-T. Vu, and G. Haffari, “Continual learning for large language models: A survey,” arXiv preprint arXiv:2402.01364, 2024.
- Y. Luo, Z. Yang, F. Meng, Y. Li, J. Zhou, and Y. Zhang, “An empirical study of catastrophic forgetting in large language models during continual fine-tuning,” arXiv preprint arXiv:2308.08747, 2023.
- J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.
- M. Rostami, S. Kolouri, and P. K. Pilly, “Complementary learning for overcoming catastrophic forgetting using experience replay,” arXiv preprint arXiv:1903.04566, 2019.
- D. Rolnick, A. Ahuja, J. Schwarz, T. Lillicrap, and G. Wayne, “Experience replay for continual learning,” Advances in Neural Information Processing Systems, vol. 32, 2019.
- S.-W. Lee, J.-H. Kim, J. Jun, J.-W. Ha, and B.-T. Zhang, “Overcoming catastrophic forgetting by incremental moment matching,” Advances in neural information processing systems, vol. 30, 2017.
- A. Mallya, D. Davis, and S. Lazebnik, “Piggyback: Adapting a single network to multiple tasks by learning to mask weights,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 67–82.
- Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, G. Su, V. Perot, J. Dy, and T. Pfister, “Learning to prompt for continual learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 139–149.
- Z. Hu, Y. Li, J. Lyu, D. Gao, and N. Vasconcelos, “Dense network expansion for class incremental learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 11 858–11 867.
- X. Li, L. Lin, S. Wang, and C. Qian, “Unlock the power: Competitive distillation for multi-modal large language models,” arXiv preprint arXiv:2311.08213, 2023.
- M. Zeng, W. Xue, Q. Liu, and Y. Guo, “Continual learning with dirichlet generative-based rehearsal,” arXiv preprint arXiv:2309.06917, 2023.
- Z. Zhang, M. Fang, L. Chen, and M.-R. Namazi-Rad, “Citb: A benchmark for continual instruction tuning,” arXiv preprint arXiv:2310.14510, 2023.
- C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet, M. Joglekar, J. Leike, I. Sutskever, and J. Wu, “Weak-to-strong generalization: Eliciting strong capabilities with weak supervision,” CoRR, vol. abs/2312.09390, 2023. [Online]. Available: https://doi.org/10.48550/arXiv.2312.09390
- M. Li, Y. Zhang, S. He, Z. Li, H. Zhao, J. Wang, N. Cheng, and T. Zhou, “Superfiltering: Weak-tostrong data filtering for fast instruction-tuning,” CoRR, vol. abs/2402.00530, 2024. [Online]. Available: https: //doi.org/10.48550/arXiv.2402.00530
- J. Ji, B. Chen, H. Lou, D. Hong, B. Zhang, X. Pan, J. Dai, and Y. Yang, “Aligner: Achieving efficient alignment through weak-to-strong correction,” CoRR, vol. abs/2402.02416, 2024. [Online]. Available: https: //doi.org/10.48550/arXiv.2402.02416