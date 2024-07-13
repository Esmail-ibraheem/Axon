const packages = [
  {
    id: 0,
    name: 'Transformer',
    url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/Transformer%20model',
    description: ` <p>The Transformer neural network, introduced by Vaswani et al. in the 2017 paper "Attention is All You Need," has revolutionized natural language processing (NLP) and other fields. Based on the concept of attention, it captures long-range dependencies and achieves state-of-the-art performance in various tasks. It is a neural network component that learns useful representations of sequences or sets of data points, driving advances in NLP, computer vision, and spatio-temporal modeling.</p>
        `
  },
  {
    id: 1,
    name: 'X-Llama',
    url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/X-Llama',
    description: `<p>Large Language Models (LLMs) are advanced AI assistants excelling in complex reasoning tasks across various fields, including programming and creative writing. Their intuitive chat interfaces have led to rapid public adoption. LLMs, like auto-regressive transformers, are pre-trained on large datasets and aligned with human preferences using techniques like Reinforcement Learning with Human Feedback (RLHF). Despite their simple training methodology, high computational demands limit development to a few players. Publicly released models (e.g., BLOOM, LLaMa-1, Falcon) rival closed models (e.g., GPT-3, Chinchilla) in performance but lack the fine-tuning for usability and safety found in closed "product" LLMs like ChatGPT, BARD, and Claude. This fine-tuning, though costly and non-transparent, is crucial for enhanced usability and safety, posing challenges for AI alignment research.</p>
        `
  },
  {
    id: 2,
    name: 'Dali package',
    url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/Dali',
    description: `<p>A generative model learns the probability distribution of a dataset, allowing for the creation of new instances by sampling from this distribution. For example, a model trained on cat images can generate new cat images. Diffusion models are a type of generative model that have gained popularity recently, surpassing GANs in image synthesis. They have been prominently used in applications like OpenAI's DALL-E 2, an image generation model.</p>
        `
  },
  {
    id: 3,
    name: 'InstructGPT ',
    url: 'https://github.com/Esmail-ibraheem/Axon/tree/main/RLHF',
    description: `<p>To align large language models (LLMs) with desired behaviors, various techniques are used:</p>
    <ul class="list-disc list-inside my-5 space-y-2">
        <li><strong>Reinforcement Learning from Human Feedback (RLHF)</strong>: Aligns model behavior with human preferences.</li>
        <li><strong>Proximal Policy Optimization (PPO)</strong>: An algorithm for reinforcement learning.</li>
        <li><strong>Direct Preference Optimization (DPO)</strong>: Another method for optimizing model preferences.</li>
        <li><strong>Supervised Fine-tuning</strong>: Refines the model's performance based on specific training data.</li>
    </ul>
    <p>AI alignment aims to ensure the model's responses adhere to specific guidelines, such as avoiding offensive language and maintaining a particular style. This process involves pretraining on vast datasets and further tuning for specific applications like chat assistants. Reinforcement learning focuses on optimizing actions to maximize cumulative rewards in an environment.</p>
`
  }
];

export { packages };
