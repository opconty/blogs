# **`laygin`** 博客文章记录及资源（Blog Post Records & Resources）

## 多模态大模型（MultiModal Large Models）
### 多模态预训练模型串烧1：CLIP、ViLT、ALBEF、VLMo
[:page_facing_up:原文 blog post](https://mp.weixin.qq.com/s/bUkQA27OphCleiqREM8C3A)

所谓多模态就是融合了不止一种模态的信息，比如图像、文本、音频和视频等，现阶段最常见的就是Vision+Language的形式。
本文记录一下基于Transformer 的图文多模态预训练（Vision-and-Language Pre-training (VLP) ）基础模型（该模型一旦训练好就可以用于VL下游任务，比如图文检索、视觉问答等，还有比较实用的Document Understanding，分析文档布局、提取文档信息并结构化输出）。
本文分析了几个经典模型（CLIP、ViLT、ALBEF、VLMo）的架构：视觉编码器、文本编码器和特征融合，以及使用的目标函数、主要贡献等等。

### 多模态预训练模型串烧2：BLIP
[:page_facing_up:原文 blog post](https://mp.weixin.qq.com/s/ndgl8NsoD5SdmKK-iJ86rw)

其主要贡献有：
- （模型方面）多模态混合的编码器-解码器结构（Multimodal mixture of Encoder-Decoder (MED)）
- （数据方面）描述生成和过滤（Captioning and Filtering (CapFilt)）

### [多模态预训练模型串烧3：CoCa](https://mp.weixin.qq.com/s/5fGxhdW0TuI8aJi-t8U0XQ)
值得一提的是，CoCa采用了非常吸引眼球和震撼的蜘蛛网图（也称为多边形图或雷达图）来展现多个任务上的结果。CoCa和ALBEF的结构很相似，ALBEF和CoCa在图像侧的编码是相同的，区别在于文本侧，ALBEF采用encoder，将其拆分为两部分，一个用于文本特征编码，一个用于多模态融合，而CoCa采用的是decoder。

### [多模态预训练模型串烧4：BEIT-3](https://mp.weixin.qq.com/s/z9fRrhOeB5jKbCgUSxfcyg)
作者团队都来自 Microsoft Corporation，如果还有印象，该文作者曾提出过VLMo那篇工作，所以在BEIT-3中会发现有一些VLMo的痕迹（实际上VLMo中的MoME，在BEIT-3中成为Multiway Transformer，只是名字变了而已；两者不同的是预训练的任务，VLMo用了三个，BEIT-3只用一个）
文中开篇就提出语言、视觉和多模态预训练的大一统趋势。该文从三个方面来推进大一统：骨干网络结构，预训练任务和模型扩大。

### [多模态预训练模型串烧5：PaLI、SigLIP、PaLI-X、PaLI-3](https://mp.weixin.qq.com/s/blTSJ5APtyZxvJX7Dv5-5g)
介绍一下来自Google的PaLI系列工作：PaLI、SigLIP、PaLI-X和PaLI-3，重点在于PaLI-3，这是最新的一篇，但是为了理解PaLI-3，就要先了解一下之前的PaLI、SigLIP、PaLI-X系列工作。
总的来说，其要点在于：
- 采用seq2seq的架构，重用训练好的基于Transformer的权重文件进行初始化，从而高效训练多模态的模型。
- 对视觉模块和语言模块进行缩放（PaLI-X是放大，PaLI-3是缩小）。
- 构造更大的数据集（WebLI）
- 将对比学习的损失由softmax改为sigmoid（SigLIP及其之后）

### [多模态预训练模型串烧6：MDETR、GLIP](https://mp.weixin.qq.com/s/dTQ2FlFNi8pNEn9Uay4cHw)
除了将图像与文本对齐之外，还可以在物体和像素级别与文本进行对齐，因此，接下来的研究点将放在粒度更细的预训练模型上：区域级预训练和像素级预训练。
GLIP受到了MDETR的启发，而MDETR基于DETR物体检测算法，DETR的损失设计主要采用了匈牙利算法的思路。所以，在介绍MDETR和GLIP之前，先了解两个预备知识：DETR和匈牙利算法。

### [多模态预训练模型串烧7：GroundingDINO及与SAM的联合使用例子](https://mp.weixin.qq.com/s/ldGn6redNcjaX2G61RF7eQ)
本文简介GroundingDINO，主要是与SAM（segment anything model）的结合使用，通过文本实现物体的检测和分割（实例分割）。

### [聊聊文档智能（Document AI）](https://mp.weixin.qq.com/s/Y7eiBCYEuaWlJqu_XE8ggw)
<font size=2>本文简介文档智能处理领域的六大任务，本文内容关键要点都整理在一张思维导图上，包括文档智能处理的六大任务、评估指标、常用数据集、经典算法、文档类别和当前挑战等内容，导图有点大，如果比较模糊，原图和PDF的百度盘链接：[文档智能处理（Document AI）.jpg: 提取码:layg](https://pan.baidu.com/s/1VXkCO91RPJlGx9PZWLfMDA?pwd=layg)、[文档智能处理（Document AI）.pdf: 提取码:layg](https://pan.baidu.com/s/1bqzsBFbQJ42ANBP6b1DUxg?pwd=layg)

虽然文档智能处理似乎已经被研究得很透了，但文档的类型繁多、应用广泛，无论是生活还是生产活动中都存在大量文档，要能够高效并精准对文档进行处理需要不断深入研究和长期探索。
文档AI算法旨在阅读、分析和理解文档，文档类型十分丰富，我们可以将信件、发票、票证、简历、身份证、名片、表格、报告、收据和传单以及科学研究文献都视为文档。研究领域主要包括六大任务：光学字符识别（OCR）、文档图像分类、文档布局分析、视觉信息提取（有关键信息提取、文档解析等叫法）、表格检测提取和结构识别、文档视觉问答等。  由此可见，关于文档的算法种类很多，既包括计算机视觉，又包括自然语言处理。
</font>
### [多模态大模型：miniGPT-4、miniGPT-v2](https://mp.weixin.qq.com/s/KC8-5Fj4RRbI1_Df5mOa8g)
本文介绍同类型的多模态大模型miniGPT-4和miniGPT-v2，在强大的大语言模型中引入视觉编码器，将视觉信息融入到LLM中。在图像理解，视觉问答方面进行了应用，不支持OCR。

### [多模态大模型：LLaVA系列及应用示例](https://mp.weixin.qq.com/s/-W3b85LdcTfXJMvjKT0cpA)
LLaVA(Large Language and Vision Assistant)，即大型语言和视觉助手，是一个端到端训练的大型多模态模型，将视觉编码器和大语言模型连接起来实现通用的视觉和语言理解。本文先简要介绍LLaVA系列模型（LLaVA、LLaVA-1.5、LLAVA-PLUS）的设计和改进点，然后进行了试用，探索了在图像理解、OCR、KIE等方面的效果。

### [CLIP论文笔记及简单的使用示例](https://mp.weixin.qq.com/s/1nXo8_br-1z0jqfo3vKwAQ)
CLIP(Contrastive Language-Image Pre-Training) 在论文 “Learning Transferable Visual Models From Natural Language Supervision”提出的多模态神经网络模型，在4亿条（图像，文本）数据集上进行训练。采用Resnet或者ViT（vision transformer）得到视觉特征，使用语言模型得到文本特征，然后将视觉特征和文本特征映射到相同维度的向量空间，二者点积用作相似度评分。

## 大语言模型（LLM）
### [大语言模型微调（Fine-tune LLMs）](https://mp.weixin.qq.com/s/VSJDvalAlS5LgZgOwPwFow)
本文介绍大语言模型的微调，包括全参数微调和使用LoRA的参数高效微调方法，采用HuggingFace实现，最后通过一个具体的证件关键信息提取实例对本文方法进行应用，当然也可以用到别的任务上去，使用的LLM是EleutherAI的pythia系列模型以及Llama3-8b（也可以使用其他模型）。
通过本文可以学习到：
- 如何准备微调大语言模型（LLM）的数据集、指令微调数据集格式
- 什么是tokenization以及如何使用tokenizer
- 如何批量加载数据
- 如何采用Huggingface的Transformers库实现LLM的全参数微调和LoRA高效微调
- 在资源有限的情况下如何使用bitsandbytes量化参数以便于微调较大的LLM

同时每部分都提供录制视频，通过notebook一行行敲代码实现，即使没有接触过LLM也没有任何难度。

### [基于Llama 3的few-shot实现证件关键信息提取](https://mp.weixin.qq.com/s/5NpUOg_atXaHmokTOYnUXA)
本文首先介绍 Llama 3的In-Context Learning（zero-shot、one-shot和few-shot），然后利用few-shot实现证件关键信息提取（key information extraction，KIE），当然也可以用在自己感兴趣的其他地方，本文仅以证件KIE作为一个例子。
通过本文可以学到如何使用现成的大语言模型API服务、如何进行zero-shot、one-shot和few-shot，最后通过证件关键信息提取进行实例应用。

### [LLM部署和基于RAG的应用](https://mp.weixin.qq.com/s/YD2KlgbTRJN30c63VY6IbQ)
之前介绍了大语言模型的few-shot和微调，本文介绍大语言模型的部署、客户端连接、发送请求和检索增强生成（Retrieval-Augmented Generation, RAG），最后提供一个视频对本文内容进行演示，以法律知识问答为例。

完成了模型训练、部署和应用的闭环，本文主要内容有：
- 什么是RAG？RAG工作原理和应用场景介绍？
- 如何部署LLM服务（使用vllm）？
- 如何在客户端连接LLM服务并发送请求（使用openai）？
- 如何从PDF文档创建向量库（使用langchain和Chroma或者FAISS）？
- RAG的实现和应用，以中文法律知识问答为例。

### [预训练大语言模型（Pretrain LLMs）](https://mp.weixin.qq.com/s/WP6A-jNMzFzZ5RD-AiGcgw)
一般情况下，我们不会从头开始预训练一个新的大语言模型，采用前面几种方法通常就能满足我们的实际需求，然而预训练自己的大语言模型是不可避免的，比如在一些垂直领域实现特定的任务或者小语种应用场景等。
即使从头开始预训练LLMs，也有一些方法和技巧可以尽可能地降低资源消耗，同时获得不错的效果。
本文就模型创建、预训练数据集等方面进行简要介绍，希望能为对预训练LLM感兴趣的朋友提供一些入门级别的常识吧。

### [大语言模型的函数调用（Function calling with LLMs）](https://mp.weixin.qq.com/s/HRGFMzbpKAzi9wSjsXOvoA)
[:pen:代码Code](https://pan.baidu.com/s/1hsH55kG59DIan95WEztcYA?pwd=lay0)

本文介绍大语言模型中的函数调用（function calling），有些地方又称为工具使用（tool use / tool calling），本文中我们不做概念区分，一律使用函数调用这个术语，函数调用允许我们通过自然语言指令调用外部函数，在私有化部署和应用中发挥重要作用，我们一般在网页聊天窗口中感受不到函数调用的能力，可能会感受到联网搜索、文档解析、Excel表格分析、制作PPT等等功能。
本文我们从头开始，搭建一个用于演示的学生成绩数据库，该数据库仅包含一个数据表：100名学生的语数英、物化生成绩，基于该数据库实现几个查询和数据筛选函数，用于在LLM中进行调用。
通过LLM的函数调用，非技术人员也能使用自然语言与数据库进行交互而不用写SQL代码，不需要了解任何技术上的东西，一切都在函数中进行实现；另外，对于技术人员，需要了解LLM的函数调用原理，才能有助于更好地实现相关函数。
通过本文，我们会学习到：
1、详细了解如何使用函数调用，如何定义调用函数的形式，并使用LLM的文本指令调用这些函数
2、完整生命周期的实操演示，从头搭建示例数据库，实现几个简单的SQL查询函数，抛砖引玉，阐明要点
3、选择两个LLM，一个是开放平台提供的API（glm-4），另一个是本地部署的 Qwen2.5-7B-Instruct。以开放平台API为主，自己部署的API也是类似的。

### [Llama 3 简介及使用demo](https://mp.weixin.qq.com/s/0vrqtIfCdPymQz2Yh4SLrg)
前两天Meta AI发布了最新的大语言模型Llama 3，目前只发布了参数量为8B和70B的模型，包括预训练和指令微调版本，可以通过huggingface和Meta官网获取权重，需要输入一些信息进行申请。本文分享一下如何获取权重？如何在线使用和本地使用？Llama 3 与Llama 2 有哪些不同？

### [AI Agent？智能体简介](https://mp.weixin.qq.com/s/oLUN17A1Mz2gmoopjMOOeQ)
> 人工智能技术每年都有一些个热词出现，2022年应该是AIGC，彼时文生图模型效果惊艳；2023年应该是大语言模型LLM、ChatGPT，AI算法对人类意图的理解更进一步；今年一开始AI Agent，也就是智能体到处被提及，其实这些技术或者术语并不是新事物，就像深度学习一样早就存在，只不过只有技术水平和其他资源能够跟得上的时候，这些技术才有可能创造出价值，也就能够真正的实现。
> 所以，本文就来简介一下什么是智能体？需要哪些技术？有什么应用？等等。
> AI Agent，或称人工智能体，是一种能够感知环境、进行决策、执行动作完成既定目标的智能实体。不同于传统的人工智能，AI Agent 具备通过独立思考、调用工具或使用技能去逐步完成给定目标的能力。AI Agent 和大模型的区别在于，大模型与人类之间的交互是基于提示（Prompt）实现的，用户提示是否清晰明确会影响大模型回答的效果，而 AI Agent 的工作仅需给定一个目标，它就能够针对目标独立思考并做出行动。大语言模型作为目前 AI Agent 的核心，以巨大参数规模捕捉复杂语言结构，实现上下文理解和连贯文本输出。这一“能力涌现”现象体现在大模型能进行高级认知任务，如抽象思考和创造性写作。AI Agent 不仅理解和生成语言，还整合规划、记忆、工具使用能力，扩展其能力边界。
> 智能体通常具备以下几个关键特性：
> - 自主性：能够在没有人类直接干预的情况下执行任务和做出决策。
> - 社交能力：能够与人类或其他智能体进行交流和协作。
> - 反应性：能够感知其所处的环境变化，并根据这些变化做出快速响应。
> - 主动性：不仅能够响应环境，还能主动采取行动以实现特定的目标或适应环境变化。
> - 智能：能够运用知识、推理和规划来解决问题和执行任务。

### [可以运行在CPU上的聊天大模型：GPT4ALL](https://mp.weixin.qq.com/s/E_G_5SwsaUSbP3mwbGi3Pg)
### [大模型微调：LoRA和DoRA的原理及pytorch实现](https://mp.weixin.qq.com/s/Jk5FHMihFqh2mnaHajyQvg)

## 视觉大模型（LVM）
### [视觉大模型：SAM（Segment Anything）及示例](https://mp.weixin.qq.com/s/vNhuXUcJXPuG5Zs0JJN-Sg)
<font size=2>图像的分割问题是计算机视觉领域的核心问题之一，旨在判断哪个像素属于哪个类别，在像素级进行分类。有广泛的应用，比如自动驾驶、医学图像分析、卫星遥感图像分析等。
不同于一般的图像分割算法，SAM旨在设计一个图像分割的基础模型，即使是在训练阶段没有见过的物体类别也能够分割，即zero-shot。SAM的核心是减少图像分割对特定任务建模的专业知识、训练计算量和自定义数据的标注的需求，实现方法就是采用可提示的方法，根据不同的数据进行训练，并且可以适应特定的任务。实现这一目标取决于三个部分：任务、模型和数据，即三个方面的问题：什么任务可以实现零样本泛化？相应的模型架构是什么？什么数据可以为这个任务和模型提供动力？因此，对应的三个组成部分的设计：
1. 可提示分割任务，其目标是在给定任何分割提示的情况下返回有效的分割掩码。
2. 模型必须支持灵活的提示，需要分摊实时计算掩码以允许交互使用，并且必须具有模糊性意识。强大的图像编码器计算图像嵌入，提示编码器编码提示信息（点、框、掩模和文本等）， 然后将两个信息源组合在一个轻量级掩模解码器中，该解码器预测分割掩模；SAM 具有三个组件：图像编码器、灵活的提示编码器和快速掩模解码器。
3. 数据引擎分为三个阶段：辅助手动（人工最多，此时模型效果还不行）、半自动（半人工半自动，模型效果持续提升中）和全自动（人工参与最少甚至不需要，都由模型生成掩膜）。
</font>

### [视觉大模型：PerSAM（一张图定制SAM）、MobileSAM（更小更快的SAM）](https://mp.weixin.qq.com/s/6VpX0D8l7qFpUe5wlkMkHw)
基于SAM衍生出很多X+Anything/Everything的应用和研究，比如Caption Anything、Inpaint Anything、Edit Everything、Label Anything、Track Anything等等，将SAM与其他基础模型联合起来使用，扩展了交互的方式，覆盖更多的任务。另一些研究工作是对SAM进行微调或优化，比如PerSAM使用one-shot的方式个性化地定制SAM，无需用户输入提示词（点、框等），可以用于自动分割特定的物体；FastSAM和MobileSAM专注于提升SAM的推理速度同时减少模型参数量。本文记录一下PerSAM和MobileSAM的研究思路和应用效果。最后对PerSAM使用证件图像进行了实验。

## AIGC
### [空间可控的文生图模型：ReCo、GLIGEN、ControlNet](https://mp.weixin.qq.com/s/1ZnrsIiRrQ-_wANG9mMtNg)
图像生成模型的发展（1）从最初的特定领域的图像生成，比如人脸生成，在人脸数据集上进行训练，模型只能生成人脸，并且可控性很低，基本就是改改随机数，比如GAN、SNGAN、SAGAN、DCGAN、StyleGAN、DDPM等。（2）渐渐地，将自然语言纳入其中，通过文本内容来控制生成的图像，即将文本作为提示词让生成模型的内容向文本对齐，可控性提高了不少，如GLIDE、Stable Diffusion、DreamBooth、DALL·E、Imagen等等。（3）更进一步的，除了通过文本来控制生成图像之外，再加入更多控制条件，使生成的图像更加符合用户意图，比如ReCo在文生图模型上加入边界框控制物体的空间位置；GLIGEN通过文本、边界框、关键点、深度图和分割图等来控制图像生成；ControlNet通过视觉信息控制图像生成，这些模型对图像的生成控制得更加精细。有点儿类似于之前介绍的多模态系列模型的发展，粒度越来越细，从图像级到区域级再到像素级。
> 本文介绍三个区域可控的文生图模型：Reco、GLIGEN和ControlNet，重点在ControlNet。这三个模型都是基于已经训练好的文生图模型，如之前介绍的Stable Diffusion（Stable Diffusion笔记及使用示例），分别引入少量可训练参数将额外的控制条件注入原有T2I模型，但引入的参数和网络结构设计有所不同，本文进行详细分解，同时试用了GLIGEN和ControlNet的生成效果。

### [空间可控的文生图模型：UniControl、Uni-ControlNet](https://mp.weixin.qq.com/s/uksiOm4u0OE_5ajQ2cvo3w)
本篇介绍的两项工作就是为了解决这个问题所提出来的，UniControl和Uni-ControlNet，同一个模型权重可以处理不同的输入条件，不仅能处理单个条件输入，还能将不同的条件进行组合输入。
虽然UniControl和Uni-ControlNet都能同时接受不同控制条件输入，但在网络结构上有所不同，本文进行剖析并进行了试用。

### [聊聊Prompt Engineering，文生图模型如何生成惊艳的图片，以Stable Diffusion为例](https://mp.weixin.qq.com/s/x7oFGKktQgQ2jiACT9KKFg)
文生图模型，比如 DALL·E 3、Midjourney 和 Stable Diffusion等，通过简单的文本提示就能生成图片，不过构造一个好的文本提示词需要下点工夫，这就是提示工程，之前在CLIP的文章中略提了一嘴提示工程（Prompt Engineering），但是那是在图像-语言模型中的应用。提示工程在LLM中应用广泛，也有很多研究工作，不过本文主要关注文生图模型的提示工程，以Stable Diffusion为例，分享一下如何构造合适的提示词以生成惊艳的符合自己意图和期望的图片。首先需要了解一个好的提示词应该由哪些部分组成，然后给一些小技巧，最后是demo，利用SDXL生成满意图像。
> 一个提示词可以划分为几个部分：
> - 主体：比如风景照、人物照、宠物照等主要的内容，这是必须的，例如：cat and duck。
> - 类型：比如照片、手绘、水彩、素描等，例如：digital painting
> - 风格：艺术风格，这个对整张图片影响很大，可以是某种风格，也可以是一些图片网站，还可以是画家名字，例如 artstation、 concept art、 surrealism、 in the style of Van Gogh
> - 灯光和色彩：指定光照的一些特征，例如：surrealist lighting、 dark lighting、studio lighting、 muted color palette
> - 相机视角：比如鸟瞰图，超广角，广角，近景，特写等，例如：aerial view, ultra wide-angle, wide-angle, massive scale, street level view, close-up
> - 其他：比如细节、分辨率等，例如：extremely detailed、 ornate、highly detailed、grainy、 realistic、 4k、 8k

在构造提示词的时候，可以参照上面几个部分看看是不是都有所涉及。

### [用AI零代码制作音乐MV，分享免费工具，详细教程](https://mp.weixin.qq.com/s/peMyjJfL7zGzF73vL7jwVg)
### [如何使用AI制作诗词朗诵、讲故事视频？需要哪些技术？](https://mp.weixin.qq.com/s/9UwHtkIM8sZbRwwy2oZx8Q)
### [啥是扩散模型？](https://mp.weixin.qq.com/s/zPhNSttjneu1kPseDEqYjQ)
### [Stable Diffusion笔记及使用示例](https://mp.weixin.qq.com/s/4lswN9zrXtAfmm3nqp-sEA)

## Pytorch & Python
### [深入理解Transformer中的自注意力机制，step by step，in code](https://mp.weixin.qq.com/s/MsoOqrRm1j1eNmOiwGgphQ)
> 通过pytorch代码深度理解transformer中自注意力机制的实现，一步一步进行推导，便于记忆和理解，记录于此，备忘。

### [扫清障碍， 通过pytorch代码深入理解transformer](https://mp.weixin.qq.com/s/xblHmWpiEyEMxik6YNF1jQ)
### [通过代码了解加速pytorch训练的方法（一）：DP & DDP](https://mp.weixin.qq.com/s/ZV09KfEj8BJb2Y86VSyXVQ)
### [通过代码了解加速pytorch训练的方法（二）：自动混合精度AMP](https://mp.weixin.qq.com/s/t2nOJtUYwgBuUbTsZz8EvA)
### [通过代码了解加速pytorch训练的方法（三）：预加载数据](https://mp.weixin.qq.com/s/b3YNS6nE3osD_-8lkK79Og)
### [通过代码了解加速pytorch训练的方法（四）：DDP（通过命令行launch脚本）](https://mp.weixin.qq.com/s/VMxdrYUuAc-JtdHdbpxsrg)
### [通过代码了解加速pytorch训练的方法（五）：accelerate](https://mp.weixin.qq.com/s/izXv0vQhJZXABb4qQ5PPhQ)
### [简单抓站的N种方式（一）-urllib与bs4](https://mp.weixin.qq.com/s/mkHp7xq-eqOVvwdSUQBnPw)
### [简单抓站的N种方式（二）-requests与re](https://mp.weixin.qq.com/s/Vek-qV73PiStTJ2sIuMSMA)
### [简单抓站的N种方式（三）-lxml与xpath](https://mp.weixin.qq.com/s/FeetoMtMGJtehoZd-0Q03Q)
### [超参数优化之前,让我们先来了解下贝叶斯优化方法](https://mp.weixin.qq.com/s/v8R9DyJxXHU0gQruyrBaNw)
### [用python把玩一下数字签名和消息加密](https://mp.weixin.qq.com/s/CXdhsthtaZ_ZHx9WHR6HcA)
### [用numpy一步步解释pytorch损失函数](https://mp.weixin.qq.com/s/QJsSINLYozYynhm5xQ1nmw)
### [重新认识矩阵乘法 - 站在吃饭和建模的角度](https://mp.weixin.qq.com/s/sieP8lwMrperQEuw8gwrxQ)
### [文本识别应用CTC的束搜索解码方法](https://mp.weixin.qq.com/s/94pETZUde78zspQThcHhJA)
### [文本识别中的CTC损失](https://mp.weixin.qq.com/s/a_ahIwxiCaO7Bxmj81HUTw)
### [OCR: WHAT AND HOW](https://mp.weixin.qq.com/s/VO42GzwwJBOabpPJOWVn4g)
### [证件检测算法的高效演进之路](https://mp.weixin.qq.com/s/3hMmVoGUitl9reW6lfbMMA)
### [深入理解证件图像信息自动化处理Pipeline](https://mp.weixin.qq.com/s/4fXY15ezZ9-MIR2FaSirpw)
### [将图像作为序列：在伪证鉴别算法应用中的探索](https://mp.weixin.qq.com/s/jL-tpcuAprWSRQK1NBa3Fw)
### [通过pytorch代码深入理解在伪证鉴别模型中所实现的注意力机制](https://mp.weixin.qq.com/s/b5x_9ZSYS3EyvfH76tP7OA)
### [STD（Semantic Text Detector）：一种基于语义特征的场景文本检测方法](https://mp.weixin.qq.com/s/F5twPy_6fjpwQuCRxMPGIA)
### [STD++：一种基于语义特征的任意方向场景文本检测方法](https://mp.weixin.qq.com/s/S6z4KCcrycWF4QigBHNahw)
### [如何打包发布Python项目，让全世界的人都能用](https://mp.weixin.qq.com/s/yDndcIU5nzyGpN08s4C3uw)
### [如何在有限标注数据条件下利用自监督学习提升文本识别模型性能](https://mp.weixin.qq.com/s/7osavofAbGj_HgYYl9vUUw)

## 数据科学（Data Science）
### [问鼎数据科学比赛之探索性数据分析（EDA）](https://mp.weixin.qq.com/s/X6dJErVgWsjQ-4Xt9tn2uQ)
### [问鼎数据科学比赛之特征工程（数值型特征）](https://mp.weixin.qq.com/s/ehTw9LtLMXEpOx0KEqZhDw)
> 特征工程非常重要，比赛能不能上分，除了之前说的EDA很重要外，最重要的还是特征工程，毕竟EDA也是为了更深入理解数据，理解问题，从而才能发现好的特征。
> 由于特征工程涉及太多，并且不同类型的数据、不同的问题都有相应的方法。所以本系列主要是从特征的类型一一无死角详细的进行剖析，本篇首先分享下数值型特征处理的骚技能。

### [问鼎数据科学比赛之特征工程（类别特征）](https://mp.weixin.qq.com/s/IhWwmz-9qRZ9jMGhrRzg3w)
### [问鼎数据科学比赛之特征工程（时间空间特征）](https://mp.weixin.qq.com/s/_hJvkuUKP7HKNC8jepMcVw)
### [问鼎数据科学比赛之缺失值的处理](https://mp.weixin.qq.com/s/ziA9l_3z7SGUTK0iW_kafw)

## 视频教程（Video Tutorial）
### [深入理解证件图像信息自动化处理Pipeline](https://mp.weixin.qq.com/s/vlZdt-v4Edrqofp8SgDVyw)
> 证件图像信息自动化处理包括很多算法，除了常见的OCR（文本检测+文本识别）之外，还有证件定位和伪证鉴别，文档信息结构化等等。本视频深入浅出介绍整个pipeline，重点在于证件检测和伪证鉴别的算法设计

### [证件定位算法：What and How？](https://mp.weixin.qq.com/s/Ws1hJ1BXl0rOOsVdj0FrsA)
> 整个OCR Pipeline包括许多算法，主干任务就是证件检测==》文本检测==》文本识别==》文档信息结构化。主要记录一下在证件检测方面的探索心得。

### [图像作为序列：伪证鉴别（英文，patented）](https://mp.weixin.qq.com/s/-thupptcI303LGDdTKnsZg)
> Image as Sequence for Classification， English version and patented.

### [图像作为序列：伪证鉴别（中文）](https://mp.weixin.qq.com/s/bImVhQ1J8YoPsA2p4dRZxg)
> 总结一下最近在伪造身份证件鉴别方面的研究探索。通过将图像特征转换为序列，从而有效结合CNN和LSTM对图像进行分类。使用attention，聚焦在局部比较有辨识度的区域，从而实现真伪证件的鉴别。详细阐述两个技巧：将CNN提取的空间特征转换为序列特征、将双向LSTM的输出序列使用基于attention的方法转换为向量。

### [深度学习算法在工业上的落地应用之不良品检测](https://mp.weixin.qq.com/s/TU4OjJMvFPqISV-lnR55aw)
> 深度学习算法广泛应用，除了互联网等行业，AI+工业也在不断发展，本视频是AI在流水线上不良品检测中的落地应用，极大提高生产效率，降低成本。

### [Grounding DINO和SAM结合使用的demo](https://mp.weixin.qq.com/s/kk-Q-jWzeF5GXPilVhlx1g)
### [PerSAM+MobileSAM证件分割应用案例](https://mp.weixin.qq.com/s/Ik2ZGxZK25nzjivmjy_ORA)
> PerSAM在证件分割中的应用案例，包括不需要训练版本和需要优化两个权重参数的版本，进行效果对比。

### [cuda环境配置及pytorch安装](https://mp.weixin.qq.com/s/28VC-J3Xsr3fMBrcCYg4SA)
> 这是学习pytorch深度学习框架的入门知识：环境搭建。包括CUDA驱动器选择，CUDA版本选择和安装，CUDNN安装，python安装（anaconda），pytorch版本选择和安装，Docker基础（强烈建议使用Docker）等等

### [LLM部署和RAG-检索增强生成实践（完整视频）](https://mp.weixin.qq.com/s/4PjleDqLKty8nLrKkE3yCQ)
> 本视频是文章《LLM部署和基于RAG（Retrieval-Augmented Generation，检索增强生成）的应用》的实操部分，主要内容包括：
>  1、什么是RAG？RAG工作原理和应用场景介绍？
>  2、如何部署大语言模型服务（使用vllm）？
>  3、如何在客户端连接LLM服务并发送请求（使用openai）？
>  4、如何从PDF文档创建向量库（使用langchain和Chroma或者FAISS）？
>  5、RAG的实现和应用，以中文法律知识问答为例。

### [大语言模型微调（完整视频）](https://mp.weixin.qq.com/s/z75Ct1a9U7foZ24fJ0dr6Q)
### [菲律宾多种证件端到端信息提取算法演示](https://mp.weixin.qq.com/s/t8QJwvSZ4y1y2Ag5sJ6eIQ)
> 本演示视频分享最近在菲律宾多种证件上的端到端信息提取算法，已经放到[Huggingface](https://huggingface.co/spaces/laygin/card_kie_e2e)上了，感兴趣可以用一用。
证件信息提取具有广泛用途，[之前的方法](https://mp.weixin.qq.com/s/vlZdt-v4Edrqofp8SgDVyw)采用证件定位+文本检测+文本识别+信息结构化进行处理，模块多，并且不同证件版面很不一样，需要的关键字段也不一样，对信息结构化的要求很高[汗]。基于端到端的算法，可以将用户拍摄的原始大图输入模型，直接得到结构化的结果[耶]。后续分享一篇专门介绍文档智能处理领域（document AI）的任务和对应的算法、数据集等。

### [文献整理及论文笔记管理](https://mp.weixin.qq.com/s/k-cNlfVCDXBe7uPqzHRqYA)
> 在算法研究过程中，会看许许多多文献资料，时间久了就会很混乱；同时也会做很多笔记，刚开始用本子写，但不便于随时查阅，效率低下，写论文时需要看自己曾经的笔记也会非常不便。使用NoteExpress文献管理软件，方便做笔记，也方便用latex写论文时插入到论文中。

## 其他（Others）
### [Latex备忘录](https://mp.weixin.qq.com/s/z1585WXTv6qCNEc1Ga92nw)
### [How to plot pie chart in LaTex](https://mp.weixin.qq.com/s/OUehpG8HaLWzV6VwPf4mZw)
### [Latex备忘录：复杂表格、emoji、代码块等](https://mp.weixin.qq.com/s/1EHXDhCx-ZLKrjZJaewFew)
### [换脸-人脸动作生成-人脸属性编辑](https://mp.weixin.qq.com/s/dxV_hfSoV92xtynZ6uFD_g)
> 是因为乐趣使然，二是最近需要调研一下人脸动作生成的 deepfake 工具，合成张嘴闭嘴的人脸动作用于测试动作活体检测算法。当然，主要的目的是为了好玩儿。
调研了音频+图像生成视频、图像+图像换脸和文本+图像进行人脸属性编辑的方法，并生成了一些样例，以供娱乐。

### [善用markdown让写文档如虎添翼](https://mp.weixin.qq.com/s/WH9dTa8utntMGo2VNyNltg)
> 经常写文档都知道用word最头疼就是对排版格式的调整，往往写内容花40%的时间，格式和润色就要花60%的时间。
所以多年前就弃word转向markdown，这是一种标记语言，解决了写文档的排版格式问题，写作时专注于内容，只需要掌握一些简单的语法即可。本文记录一下markdown中的标题、表格、图像、流程图、表情emoji、代码块、列表、数学公式等语法，当然还能输入html代码，应该能满足日常工作大部分需要。

### [微小说：路边草](https://mp.weixin.qq.com/s/75WuKr4QoFVk9mLEizX6dw)

**欢迎关注公众号：laygin (Welcome to follow the official account: laygin)**
![](qrcode_for_ggzh.jpg)
