> **Introduction**. Large Language Models (LLMs) have shown great promise as highly capable AI assistants that excel in complex reasoning tasks requiring expert knowledge across a wide range of fields, including in specialized domains such as programming and creative writing. They enable interaction with humans through intuitive chat interfaces, which has led to rapid and widespread adoption among the general public. The capabilities of LLMs are remarkable considering the seemingly straightforward nature of the training methodology. Auto-regressive transformers are pretrained on an extensive corpus of self-supervised data, followed by alignment with human preferences via techniques such as Reinforcement Learning with Human Feedback (RLHF). Although the training methodology is simple, high computational requirements have limited the development of LLMs to a few players. There have been public releases of pretrained LLMs (such as BLOOM(Scao et al., 2022), LLaMa-1 (Touvron et al., 2023), and Falcon (Penedo et al., 2023)) that match the performance of closed pretrained competitors like GPT-3 (Brown et al., 2020) and Chinchilla (Hoffmannet al., 2022), but none of these models are suitable substitutes for closed “product” LLMs, such as ChatGPT, BARD, and Claude. These closed product LLMs are heavily fine-tuned to align with human preferences, which greatly enhances their usability and safety. This step can require significant costs in compute and humanannotation, and is often not transparent or easily reproducible, limiting progress within the community to advance AI alignment research.

---

1. RMS Normalization 
	 
	 Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and significantly slows the underlying network, e.g. RNNin particular, RMSNorm regularizes the summed inputs to a neuron in one layer ac cording to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm.
	 
	Awell-known explanation of the success of LayerNorm is its re-centering and re-scaling invariance property. The former enables the model to be insensitive to shift noises on both inputs and weights, and the latter keeps the output representations intact when both inputs and weights are randomly scaled.
	
	RMSNorm which only focuses on re-scaling invariance and regularizes the summed inputs simply according to the root mean square (RMS) statistic:
	$$a_i' = \frac{a_i}{\text{RMS}(a)} \cdot g_i, \quad \text{where } \text{RMS}(a) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}$$
	 Intuitively, RMSNorm simplifies LayerNorm by totally removing the mean statistic at the cost of sacrificing the invariance that mean normalization affords.
	 
	**Why RMSNorm?**
	- Requires less computation compared to Layer Normalization. 
	- It works well in practice.

2. Rotary Positional Embeddings 
	 
	 - **Absolute Positional Encodings** 
	 
		 are fixed vectors that are added to the embedding of a token to represent its absolute position in the sentence. So, it deals with **one token at a time**. You can think of it as the pair (latitude, longitude) on a map: each point on earth will have a unique pair.
	 
	 - **Relative Position Encoding**
	 
		Relative positional encodings, on the other hand, deals with **two tokens** at a time and it is involved when we calculate the attention: since the attention mechanism captures the “intensity” of how much two words are related two each other, relative positional encodings tells the attention mechanism the distance between the two words involved in it. So, given two tokens, we create a vector that represents their distance.
		[Shaw et al. (2018)](https://arxiv.org/abs/1803.02155)) incorporated relative positional information into $𝑊^𝑘$ and $𝑊^𝑣$. Maximum relative position is clipped to a maximum absolute value of $𝑘$ and this clipping operation enables the model to generalize to unseen sequence lengths. Therefore, $2𝑘+1$ unique edge labels are considered and let us denote $\mathbf{P}^k, \mathbf{P}^v \in \mathbb{R}^{2k+1}$ as learnable relative position representations.
		$$A_{ij}^k = P^k_{\text{clip}(j - i, k)} \quad
A_{ij}^v = P^v_{\text{clip}(j - i, k)} \quad
\text{where }\text{clip}(x, k) = \text{clip}(x, -k, k)$$
		[Transformer-XL](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#transformer-xl) ([Dai et al., 2019](https://arxiv.org/abs/1901.02860)) proposed a type of relative positional encoding based on re-parametrization of dot-product of keys and queries. To keep the positional information flow coherently across segments, Transformer-XL encodes the _relative_ position instead, as it could be sufficient enough to know the position offset for making good predictions, i.e., between one key vector and its query.

	 - **Rotary Position Embedding**
		
		Rotary position embedding (_RoPE_; [Su et al. 2021](https://arxiv.org/abs/2104.09864)) encodes the absolution position with a [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) and multiplies key and value matrices of every attention layer with it to inject relative positional information at every layer.
		
		When encoding relative positional information into the inner product of the $𝑖-th$ key and the $𝑗-th$ query, we would like to formulate the function in a way that the inner product is only about the relative position 𝑖−𝑗. Rotary Position Embedding (RoPE) makes use of the rotation operation in Euclidean space and frames the relative position embedding as simply rotating feature matrix by an angle proportional to its position index.
		
		Given a vector 𝑧, if we want to rotate it counterclockwise by 𝜃, we can multiply it by a rotation matrix to get 𝑅𝑧 where the rotation matrix 𝑅 is defined as:
	![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/a959cf9a-50fc-4db6-be5c-1d383f69f2cb)
	![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/9f05f971-206b-43dc-ba82-e249616772b4)
When generalizing to higher dimensional space, RoPE divide the 𝑑-dimensional space into 𝑑/2 subspaces and constructs a rotation matrix 𝑅 of size 𝑑×𝑑 for token at position 𝑖:
![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/4708c8a1-78ac-433d-89f5-56e6768dd840)
where in the paper we have Θ=𝜃𝑖=10000−2(𝑖−1)/𝑑,𝑖∈[1,2,…,𝑑/2]. Note that this is essentially equivalent to sinusoidal positional encoding but formulated as a rotation matrix.



