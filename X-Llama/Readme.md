# [Llama](https://arxiv.org/abs/2307.09288),  [Original repo](https://github.com/Esmail-ibraheem/X-Llama) 
  
> **Introduction**. Large Language Models (LLMs) have shown great promise as highly capable AI assistants that excel in complex reasoning tasks requiring expert knowledge across a wide range of fields, including in specialized domains such as programming and creative writing. They enable interaction with humans through intuitive chat interfaces, which has led to rapid and widespread adoption among the general public. The capabilities of LLMs are remarkable considering the seemingly straightforward nature of the training methodology. Auto-regressive transformers are pre-trained on an extensive corpus of self-supervised data, followed by alignment with human preferences via techniques such as Reinforcement Learning with Human Feedback (RLHF). Although the training methodology is simple, high computational requirements have limited the development of LLMs to a few players. There have been public releases of pre-trained LLMs (such as BLOOM(Scao et al., 2022), LLaMa-1 (Touvron et al., 2023), and Falcon (Penedo et al., 2023)) that match the performance of closed pre-trained competitors like GPT-3 (Brown et al., 2020) and Chinchilla (Hoffmannet al., 2022), but none of these models are suitable substitutes for closed “product” LLMs, such as ChatGPT, BARD, and Claude. These closed-product LLMs are heavily fine-tuned to align with human preferences, which greatly enhances their usability and safety. This step can require significant costs in computing and humanannotation, and is often not transparent or easily reproducible, limiting progress within the community to advance AI alignment research.   
  
<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/X-Llama/assets/Llama.png" alt="Your Image Description" >
</p>
 
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
		[Shaw et al. (2018)](https://arxiv.org/abs/1803.02155)) incorporated relative positional information into 𝑊^𝑘 and 𝑊^𝑣. Maximum relative position is clipped to a maximum absolute value of 𝑘 and this clipping operation enables the model to generalize to unseen sequence lengths. Therefore, 2𝑘+1 unique edge labels are considered.
		$$A_{ij}^k = P^k_{\text{clip}(j - i, k)} \quad
A_{ij}^v = P^v_{\text{clip}(j - i, k)} \quad
\text{where }\text{clip}(x, k) = \text{clip}(x, -k, k)$$
		[Transformer-XL](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/#transformer-xl) ([Dai et al., 2019](https://arxiv.org/abs/1901.02860)) proposed a type of relative positional encoding based on re-parametrization of dot-product of keys and queries. To keep the positional information flow coherently across segments, Transformer-XL encodes the _relative_ position instead, as it could be sufficient enough to know the position offset for making good predictions, i.e., between one key vector and its query.

	 - **Rotary Position Embedding**
		
		Rotary position embedding (_RoPE_; [Su et al. 2021](https://arxiv.org/abs/2104.09864)) encodes the absolution position with a [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix) and multiplies key and value matrices of every attention layer with it to inject relative positional information at every layer.
		
		When encoding relative positional information into the inner product of the 𝑖-th key and the 𝑗-th query, we would like to formulate the function in a way that the inner product is only about the relative position 𝑖−𝑗. Rotary Position Embedding (RoPE) makes use of the rotation operation in Euclidean space and frames the relative position embedding as simply rotating feature matrix by an angle proportional to its position index.
		
		Given a vector 𝑧, if we want to rotate it counterclockwise by 𝜃, we can multiply it by a rotation matrix to get 𝑅𝑧 where the rotation matrix 𝑅 is defined as:
	![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/a959cf9a-50fc-4db6-be5c-1d383f69f2cb)
When generalizing to higher dimensional space, RoPE divide the 𝑑-dimensional space into 𝑑/2 subspaces and constructs a rotation matrix 𝑅 of size 𝑑×𝑑 for token at position 𝑖:
![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/4708c8a1-78ac-433d-89f5-56e6768dd840)
where in the paper we have Θ=𝜃𝑖=10000−2(𝑖−1)/𝑑,𝑖∈[1,2,…,𝑑/2]. Note that this is essentially equivalent to sinusoidal positional encoding but formulated as a rotation matrix.

	<p align="center">
	  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/X-Llama/assets/RoPE.png" alt="Your Image Description" >
	</p>


3. KV-Cache 
	
	Recall the definition of Attention given in the [“Attention Is All You Need”](https://arxiv.org/pdf/1706.03762.pdf) paper:

	$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

	where 𝑄, 𝐾, and 𝑉 are three matrices that are trained during the training process. The embeddings of each token (a vector) is multiplied by these three matrices to obtain three vectors 𝑞𝑛, 𝑘𝑛, and 𝑣𝑛.
	
	When computing self-attention, we compute the dot product of the query vector 𝑞𝑛 with the key vector of every other token before it in the input sequence 𝑘_𝑛,𝑘_{𝑛+1},…,𝑘_𝑁.
	
	Each product 𝑞𝑖𝑇⋅𝑘𝑗 is divided by the square root of the dimension of the key vectors 𝑑𝑘 in order to have more stable gradients. Eventually, everything is passed through a softmax to normalize the scores:
	$$a_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{t=1}^{i}\exp(q_i^T k_t / \sqrt{d_k})}$$
The final output is derived by computing the weighted average over the value vectors:
	$$o_i = \sum_{j=1}^{i} a_{ij} v_j$$

**The autoregressive nature of transformers**
	
transformer-based models are **autoregressive models**, meaning essentially that they use the past to predict the future.

Given a prompt $(x_1, …, x_n)$

Since the tokens (𝑥1,…,𝑥𝑛) are all known, computing 𝑃(𝑥𝑛+1|𝑥1,…,𝑥𝑛) can be made with matrix-matrix multiplication and thus benefit from GPU parallelism.

Instead, when we get to compute the remaining tokens 𝑃(𝑥𝑛+𝑡+1|𝑥1,…,𝑥𝑛+𝑡), the data dependency forces us to use a matrix-vector multiplication, which is less efficient and leads to an **underutilization of the GPU**.
	
**In the process we described above**, one can notice that the key and value vectors 𝑘1,…,𝑘𝑛+𝑡−1 and 𝑣1,…,𝑣𝑛+𝑡−1 seem to be re-computed every time a new token is taken into consideration. Of course, this would be a waste of resources.

 Consider the below illustration:
 	![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/7e12fe2c-b230-4e92-9765-7e2f9bf15f0f)


 The 𝐾 and 𝑉 matrices contain information about all the sequences, while the query vector contains just the information about the last token. The dot product between 𝑞 and 𝐾 corresponds to paying attention between the last token (i.e. “blue” in our example) and all the previous ones.

Note two things:
	- during the sequence generation one token at a time, the two matrices 𝐾 and 𝑉 do not change very much
	- once we computed the embedding for the new token, it’s not going to change, no matter how many more tokens we generate

 That is why the key and value vectors of existing tokens are often cached for generating future tokens. This approach leads to what is called the **KV cache**. Note that the KV cache of one token depends on all its previous tokens, hence if we have the same token appearing in two different positions inside the sequence, the corresponding KV caches will be different as well.

![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/f26ff8fe-2964-4bc2-89e3-610781ffa02c)


 **How much memory does KV cache use?**

Let’s consider a 13B parameter [OPT model](https://arxiv.org/pdf/2205.01068.pdf)


In OPT a sequence can be made of up to 2048 tokens, hence we would need 800∗2048≈1.6GB per single request.

4. Grouped-Query Attention
	
	The standard practice for autoregressive decoding is to cache the keys and values of the previous tokens in the sequence to speed up attention computation. However, as the context window or batch size increases, the memory cost associated with the size of the key-value cache(kv cache) in the multi-head attention(MHA) model significantly increases.

	Multi-Query attention(MQA) is a mechanism that uses only a single key-value head for multiple queries, which can save memory and greatly speed up decoder inference.
	
	However, MQA may lead to a decrease in quality. In fact, we not only want fast inference, but also want the quality to be on par with MHA, so Grouped-query attention(GQA)[1] comes into play.

	Grouped-query attention(GQA) is an interpolation of multi-query and multi-head attention. It achieves a quality similar to multi-head attention while maintaining a comparable speed to multi-query attention.
	
	Grouped-query attention divides query heads into G-groups, each of which shares a single key head and value head. GQA-G refers to grouped query with G groups. GQA-1, with a single group and therefore single key and value head, is equivalent to MQA, while GQA-H, with groups equal to a number of heads, is equivalent to MHA. The following Figure shows a comparison of grouped-query attention and multi-head/multi-query attention. When converting a multi-head checkpoint to a GQA checkpoint, we construct each group key and value head by mean pooling all the original heads within that group. An intermediate number of groups leads to an interpolated model that is of higher quality than MQA but faster than MHA, and, as we will show, represents a favorable trade-off. Going from MHA to MQA reduces H key and value heads to a single key and value head, reducing the size of the key-value cache and therefore amount of data that needs to be loaded by a factor of H. However, larger models generally scale the number of heads, such that multi-query attention represents a more aggressive cut in both memory bandwidth and capacity. GQA lets us keep the same proportional decrease in bandwidth and capacity as model size increases.

<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/X-Llama/assets/GQA.png" alt="Your Image Description" >
</p>

Overview of the grouped-query method. Multi-head attention has H query, key, and value heads. Multi-query attention shares single key and value heads across all query heads. Grouped-query attention instead shares single key and value heads for each group of query heads, interpolating between multi-head and multi-query attention.

**MHA vs GQA vs MQA :**

<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/X-Llama/assets/MHA%2CGQA%2CMQA.png" alt="Your Image Description" width=500, height=400>
</p>

**Uptraining steps** Figure shows how performance varies with uptraining proportion for T5 XXL with MQA and GQA. First, we note that GQA already achieves reasonable performance after conversion while MQA requires uptraining to be useful. Both MQA and GQA gain from 5% uptraining with diminishing returns from 10%.


<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/X-Llama/assets/MHA%2CGQA%2CMQA2.png" alt="Your Image Description" width=500, height=400>
</p>

Time per sample for GQA-XXL as a function of the number of GQA groups with input length 2048 and output length 512. Going from 1 (MQA) to 8 groups adds modest inference overhead, with increasing cost to adding more groups.
	
**Number of groups**: The figure demonstrates the effect of the number of GQA groups on inference speed. For larger models, the memory bandwidth overhead from the KV cache is less con-straining (Shazeer, 2019), while the reduction in key-value size is sharper due to the increased number of heads. As a result, increasing the number of groups from MQA only results in modest slowdowns initially, with increasing costs as we move closer to MHA. We selected 8 groups as a favorable middle ground.

|         MHA          |                  GQA                  |         MQA          |
| :------------------: | :-----------------------------------: | :------------------: |
|     High quality     | A good compromise between quality and |   Loss in quality    |
| Computationally slow |                 speed                 | Computationally fast |

> MHA enables a nuanced understanding of the relationships between different parts of the input. Nevertheless, this complexity comes at a cost — a significant demand on memory bandwidth, especially during decoder inference. In multi-query attention, we average the heads for keys and values so that all query heads share the same key and value head. This is achieved by replicating the mean-pooled “head” H times, where H is the number of query heads. However, MQA is not without its drawbacks. The reduced complexity can lead to quality degradation and training instability. Grouped-query attention (GQA) is a simple approach that blends elements of multi-head attention (MHA) and multi-query attention (MQA) to create a more efficient attention mechanism.

---

## Acknowledgment: 

**papers**:
- [llama 2 research paper](https://arxiv.org/abs/2307.09288)
- [attention is all you need research paper](https://arxiv.org/abs/1706.03762)
- [Grouped Query Attention research paper](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding research paper](https://arxiv.org/abs/2104.09864)

**other**
- [llama from scratch](https://youtu.be/oM4VmoabDAI?si=rDegyrnSghByUEnK)
- [huggingFace transformers lib](https://github.com/huggingface/transformers)
