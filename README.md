
# <p align="center"> Axon: AI research Lab.🔬 </p>
<p align="center"> <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/NN.jpg" alt="Your Image Description" width="250" height=250"></p> 

Welcome to **Axon: AI Research Lab!** This repository serves as a collaborative platform for implementing cutting-edge AI research papers and conducting novel research in various areas of artificial intelligence. Our mission is to bridge the gap between theoretical research and practical applications by providing high-quality, reproducible implementations of seminal and contemporary AI papers: InstructGPT, llama, transformers, diffusion models, RLHF, etc...
 
---

## Papers implemented:
- attention is all you need.
   - > The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best-performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
- InstructGPT.
   - > Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.
- Llama.
- Multi-Head attention.
- Multi-Query attention.
- Grouped-Query attention.
- reinforcement learning from human feedback.

## Axon's Packages:
- [Transformer model](https://github.com/Esmail-ibraheem/Axon/tree/main/Transformer%20model)
   - **Abstract.** The Transformer neural network is a powerful deep learning model that was introduced in a landmark paper titled "attention is all you need" by Vaswani et al. in 2017. It revolutionized the field of natural language processing (NLP) and has since found applications in various other domains. The Transformer architecture is based on the concept of attention, enabling it to capture long-range dependencies and achieve state-of-the-art performance on a wide range of tasks. The transformer is a neural network component that can be used to learn useful represen tations of sequences or sets of data-points [Vaswani et al., 2017]. The transformer has driven recent advances in natural language processing [Devlin et al., 2019], computer vision [Dosovitskiy et al., 2021], and spatio-temporal modelling [Bi et al., 2022].
- [X-Llama](https://github.com/Esmail-ibraheem/Axon/tree/main/X-Llama)
- [DDPM](https://github.com/Esmail-ibraheem/Axon/tree/main/Dali)
- [InstructGPT](https://github.com/Esmail-ibraheem/Axon/tree/main/RLHF)
