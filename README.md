
# <p align="center"> Axon: AI research Lab.🔬 </p>
<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/NN.jpg" alt="Your Image Description" width="250" height=250">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2307.09288">
    <img src="https://img.shields.io/badge/Arxiv-llama paper%20-red" alt="https://arxiv.org/abs/2307.09288">
  </a>
  <a href="https://arxiv.org/abs/1706.03762">
    <img src="https://img.shields.io/badge/Arxiv-Transformer%20-red" alt="https://arxiv.org/abs/1706.03762">
  </a>
  <a href="https://arxiv.org/abs/2305.13245">
    <img src="https://img.shields.io/badge/Arxiv-GQA%20-red" alt="https://arxiv.org/abs/2305.13245">
  </a>
  <a href="https://arxiv.org/abs/2104.09864">
    <img src="https://img.shields.io/badge/Arxiv-RoFormer%20-red" alt="https://arxiv.org/abs/2104.09864">
  </a>

  <a href="https://arxiv.org/abs/1910.07467">
    <img src="https://img.shields.io/badge/Arxiv-RMSNorm%20-red" alt="https://arxiv.org/abs/1910.07467">
  </a>
  <a href="https://arxiv.org/abs/2104.12470">
    <img src="https://img.shields.io/badge/Arxiv-Easy and Effecient Transformer%20-red" alt="https://arxiv.org/abs/2104.12470">
  </a>
  <a href="https://arxiv.org/abs/2203.02155">
    <img src="https://img.shields.io/badge/Arxiv-InstructGPT%20-red" alt="https://arxiv.org/abs/2203.02155">
  </a>
  <a href="https://arxiv.org/abs/1707.06347">
    <img src="https://img.shields.io/badge/Arxiv-PPO%20-red" alt="https://arxiv.org/abs/1707.06347">
  </a>
  <a href="https://arxiv.org/abs/2305.18290">
    <img src="https://img.shields.io/badge/Arxiv-DPO%20-red" alt="https://arxiv.org/abs/2305.18290">
  </a>
</p>


Welcome to **Axon: AI Research Lab!** This repository serves as a collaborative platform for implementing cutting-edge AI research papers and conducting novel research in various areas of artificial intelligence. Our mission is to bridge the gap between theoretical research and practical applications by providing high-quality, reproducible implementations of seminal and contemporary AI papers: InstructGPT, llama, transformers, diffusion models, RLHF, etc...



---

## Papers implemented:
- attention is all you need.
   - > The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best-performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
- InstructGPT.
   - > Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.
- Llama.
   - > In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.
- Multi-Head attention.
- Multi-Query attention.
- Grouped-Query attention.
   - > Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.
```
      ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
      │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
      └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │         │        │                 │
      ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
      │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
      └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
      ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
      │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
      └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
      ◀️──────────────────▶️  ◀️──────────────────▶️  ◀️──────────────────▶️
              MHA                    GQA                   MQA
        n_query_groups=4       n_query_groups=2      n_query_groups=1
```

|         MHA          |                  GQA                  |         MQA          |
| :------------------: | :-----------------------------------: | :------------------: |
|     High quality     | A good compromise between quality and |   Loss in quality    |
| Computationally slow |                 speed                 | Computationally fast |

- reinforcement learning from human feedback.
  - > A promising approach to improve the robustness and exploration in Reinforcement Learning is collecting human feedback and that way incorporating prior knowledge of the target environment. It is, however, often too expensive to obtain enough feedback of good quality. To mitigate the issue, we aim to rely on a group of multiple experts (and non-experts) with different skill levels to generate enough feedback. Such feedback can therefore be inconsistent and infrequent. In this paper, we build upon prior work -- Advise, a Bayesian approach attempting to maximise the information gained from human feedback -- extending the algorithm to accept feedback from this larger group of humans, the trainers, while also estimating each trainer's reliability. We show how aggregating feedback from multiple trainers improves the total feedback's accuracy and make the collection process easier in two ways. Firstly, this approach addresses the case of some of the trainers being adversarial. Secondly, having access to the information about each trainer reliability provides a second layer of robustness and offers valuable information for people managing the whole system to improve the overall trust in the system. It offers an actionable tool for improving the feedback collection process or modifying the reward function design if needed. We empirically show that our approach can accurately learn the reliability of each trainer correctly and use it to maximise the information gained from the multiple trainers' feedback, even if some of the sources are adversarial.

---

## Axon's Packages:
**packages with their papers implemented:**

```
┌───────────────┐        ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
│  Transformer  │        │    X-Llama    │        │      Dali     │        │  InstructGPT  │
└───────┬───────┘        └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
        │                        │                        │                        │
        │                        │                        │                        │
        ▼                        ▼                        ▼                        ▼
┌───────────────────┐   ┌───────────────────────────┐   ┌───────────────────┐   ┌────────────────────────┐
│ "Attention is All │   │ "Llama2"                  │   │ "DDPM"            |   │ "RLHF Survey"          │
│ You Need"         │   │ "RoFormer"                │   └───────────────────┘   │ "PPO"                  │
└───────────────────┘   │ "GQA"                     │                           │ "DPO"                  │
                        │ "Attention is All         │                           └────────────────────────┘                        
                        │ You Need"                 |                                                  
                        │ "KV-cache", RMSNorm       |                                                   
                        └───────────────────────────┘                           
```
- [Transformer model](https://github.com/Esmail-ibraheem/Axon/tree/main/Transformer%20model)
   - **Abstract.** The Transformer neural network is a powerful deep learning model that was introduced in a landmark paper titled "attention is all you need" by Vaswani et al. in 2017. It revolutionized the field of natural language processing (NLP) and has since found applications in various other domains. The Transformer architecture is based on the concept of attention, enabling it to capture long-range dependencies and achieve state-of-the-art performance on a wide range of tasks. The transformer is a neural network component that can be used to learn useful represen tations of sequences or sets of data-points [Vaswani et al., 2017]. The transformer has driven recent advances in natural language processing [Devlin et al., 2019], computer vision [Dosovitskiy et al., 2021], and spatio-temporal modelling [Bi et al., 2022].
- [X-Llama](https://github.com/Esmail-ibraheem/Axon/tree/main/X-Llama)
  - X-Llama is an advanced language model framework, inspired by the original Llama model but enhanced with additional features such as Grouped Query Attention (GQA), Multi-Head Attention (MHA), and more. This project aims to provide a flexible and extensible platform for experimenting with various attention mechanisms and building state-of-the-art natural language processing models.

project structure: The [model](https://github.com/Esmail-ibraheem/Axon/blob/main/X-Llama/X-Llama/model.py) was constructed in approximately ~500 lines of code, and you have the model's [configuration](https://github.com/Esmail-ibraheem/Axon/blob/main/X-Llama/X-Llama/config.py).
```
X-Llama/
│
├── images/
│
├── models/
│   ├── attentions/
│   ├── rotary_embeddings/
│   └── transformer/
│
├── model
│
└── config
│
└── inference

```
- [DDPM](https://github.com/Esmail-ibraheem/Axon/tree/main/Dali) 
  - Diffusion Models are generative models, meaning that they are used to generate data similar to the data on which they are trained. Fundamentally, Diffusion Models work by destroying training data through the successive addition of Gaussian noise and then learning to recover the data by reversing this noising process. After training, we can use the Diffusion Model to generate data by simply passing randomly sampled noise through the learned denoising process. Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).
- [InstructGPT](https://github.com/Esmail-ibraheem/Axon/tree/main/RLHF)
  - AI alignment: A large language model typically is pre-trained on a massive amount of data, for example, the entire Wikipedia and billions of web pages. This gives the language model a vast “knowledge” of information to complete any prompt in a reasonable way. However, to use an LLM as a chat assistant (for example ChatGPT) we want to force the language model to follow a particular style. For example, we may want the following:

     - Do not use offensive language
     - Do not use racist expressions
     - Answer questions using a particular style The goal of AI alignment is to align the model’s behavior with a desired behavior.

---

## Usage:
first to download the repo:
```
https://github.com/Esmail-ibraheem/Axon.git
```

Then you have this built tree. Check the README file for each package to gain a better understanding.:
```
Axon/
│
├── Transformer model/
│   ├── transformer/
│   ├── translator/
│   └── assets/
|   └── Readme.md
|
├── X-Llama/
│   ├── models/
│   ├── X-Llama/
│   └── assets/
|   └── Readme.md
│
├── Dali/
|   
│
└── RLHF (InstructGPT)/
│
└── Readme.md
│
└── NN.jpg
```
---

## Citation
```BibTex
@misc{Gumaan2024-Axon,
  title   = "Axon",
  author  = "Gumaan, Esmail",
  howpublished = {\url{https://github.com/Esmail-ibraheem/Axon}},
  year    = "2024",
  month   = "May",
  note    = "[Online; accessed 2024-05-24]",
}
```

---


## Notes and Acknowledgments:
I built this AI research lab, Axon, as an ecosystem for implementing research papers on topics ranging from transformers and x-Llama to diffusion models. The lab also focuses on understanding the theoretical and mathematical aspects of the research, as detailed in the README file for each package. This project contains multiple packages, each offering different implementations of various papers. If you want to add an implementation of a research paper, please make a pull request. This project is open to implementations of more papers.

**papers**:
- [llama 2 research paper](https://arxiv.org/abs/2307.09288)
- [attention is all you need research paper](https://arxiv.org/abs/1706.03762)
- [Grouped Query Attention research paper](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding research paper](https://arxiv.org/abs/2104.09864)
- [RMSNorm](https://arxiv.org/abs/1910.07467)
- [Easy and Efficient Transformer](https://arxiv.org/abs/2104.12470)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Proximal Policy Optimization algorithm](https://arxiv.org/abs/1707.06347)
- [Direct Perfernces Optimization algorithm](https://arxiv.org/abs/2305.18290)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Reinforcement Learning from Human Feedback survey](https://arxiv.org/abs/2312.14925)




---
