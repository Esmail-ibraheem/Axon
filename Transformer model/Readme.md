# [attention is all you need](https://arxiv.org/abs/1706.03762), [Original repo](https://github.com/Esmail-ibraheem/LlTRA-Model)

### Transformers:
 
**Abstract.** The Transformer neural network is a powerful deep learning model that was introduced in a landmark paper titled **"attention is all you need"** by Vaswani et al. in 2017. It revolutionized the field of natural language processing (NLP) and has since found applications in various other domains. The Transformer architecture is based on the concept of attention, enabling it to capture long-range dependencies and achieve state-of-the-art performance on a wide range of tasks.
The transformer is a neural network component that can be used to learn useful represen tations of sequences or sets of data-points [Vaswani et al., 2017]. The transformer has driven recent advances in natural language processing [Devlin et al., 2019], computer vision [Dosovitskiy et al., 2021], and spatio-temporal modelling [Bi et al., 2022].

<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/transformer.png" alt="Your Image Description" >
</p>

**Introduction.** Before the emergence of Transformer Neural Networks (TNNs), Recurrent Neural Networks (RNNs) were commonly employed for sequential processing tasks, including machine translation. However, RNNs were characterized by slow processing speeds, limited accuracy, and challenges in handling large datasets. ^20bbbe

**here how Recurrent Neural Network woks:** 
 is designed to process sequential data, where the current input not only depends on the current state but also on the previous inputs and states.

suppose we have this sentence "I work at the university.", and we want to translate it to Arabic "انا اعمل في الجامعة" .

In the translation task, the RNN analyzes each word ('I', 'work', 'at', 'the', 'university') one by one, updating the hidden state at each step. The output at each time step is influenced by the current word and the hidden state, which captures the historical information from previous words. The final output is a sequence of translated words ('انا', 'اعمل', 'في', 'الجامعة') in Arabic.

<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/rnn2.png" alt="Your Image Description" >
</p>

#### Problems with RNN:
1. Slow computation for long sequences 
2. Vanishing or exploding gradients 
3. Difficulty in accessing information from a long time ago
4. Complexity per layer: $O(nd^2)$, meanwhile transformer's is $O(n^2d)$

Indeed, RNNs tend to be slow and can struggle with handling large datasets, which can lead to potential confusion or difficulties in processing extensive data. However, 

The transformer Neural Network (TNN) introduced a breakthrough solution called "Self-Attention" in the paper "Attention is All You Need." This innovation addressed these issues and paved the way for subsequent advancements such as GPT, Bert, **Llama**, stable diffusion, and more.

so first we have the left architecture which is the "encoder" and the right is the "decoder":

1. **Input Embeddings:**  
	- Word Embedding: Represent each word as a “vector” of numbers
	
    The input sequence is transformed into fixed-dimensional embeddings, typically composed of word embeddings and positional encodings. Word embeddings capture the semantic meaning of each word.
	<p align="center">
	  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/embedding.png" alt="Your Image Description" >
	</p>
 2. while **positional encodings** indicate the word's position in the sequence using the sin and cos waves.
    	 <p align="center">
	  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/pe1.png" alt="Your Image Description" >
	</p>
$$\text{PE}(i,\delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta'\\
\cos(\frac{i}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1\\
\end{cases}$$
	<p align="center">
	  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/pe2.png" alt="Your Image Description" >
	</p>
 ### Why trigonometric functions? 
     
Trigonometric functions like cos and sin naturally represent a pattern that the model can recognize as continuous, so relative positions are easier to see for the model. By watching the plot of these functions, we can also see a regular pattern, so we can hypothesize that the model will see it too.

3. **Encoder and Decoder:**  
    The Transformer model consists of an encoder and a decoder. Both the encoder and decoder are composed of multiple layers. Each layer has two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network.
    
      - **_Encoder:_** The encoder takes the input sequence and processes it through multiple layers of self-attention and feed-forward networks. It captures the contextual information of each word based on the entire sequence.
	        <p align="center">
		  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/encoder.png" alt="Your Image Description" >
		</p>
      - **_Decoder:_** The decoder generates the output sequence word by word, attending to the encoded input sequence's relevant parts. It also includes an additional attention mechanism called "encoder-decoder attention" that helps the model focus on the input during decoding.
        	<p align="center">
		  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/decoder.png" alt="Your Image Description" >
		</p>
4. **Self-Attention Mechanism:**
   	- __first what is self-attention:__ it is the core of the Transformer model is the self-attention mechanism. It allows each word in the input sequence to attend to all other words, capturing their relevance and influence, works by seeing how similar and important each word is to all of the words in a sentence, including itself.
   	  	<p align="center">
		  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/attention.png" alt="Your Image Description" >
		</p>
   	- **Second the Mechanism:** 
		- **_Multi-head attention in the encoder block_:** plays a crucial role in capturing different types of information and learning diverse relationships between words. It allows the model to attend to different parts of the input sequence simultaneously and learn multiple representations of the same input.
	   		<p align="center">
			  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/transformer op.png" alt="Your Image Description" >
			</p>
		- **_Masked Multi-head attention in the decoder block_:** the same as Multi-head  attention in the encoder block but this time for the translation sentence, is used to ensure that during the decoding process, each word can only attend to the words before it. This masking prevents the model from accessing future information, which is crucial for generating the output sequence step by step.
   	   		<p align="center">
			  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/attention op2.png" alt="Your Image Description" >
			</p>
   	 	- **Multi-head attention in the decoder block_:** do the same as the Multi-head attention in the encoder block but between the input sentence and the translation sentence, is employed to capture different relationships between the input sequence and the generated output sequence. It allows the decoder to attend to different parts of the encoder's output and learn multiple representations of the context.
   	    		<p align="center">
			  <img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/attention op3.png" alt="Your Image Description" >
			</p>
---

### self attention mechanism:

The core of the Transformer model is the self-attention mechanism. It allows each word in the input sequence to attend to all other words, capturing their relevance and influence. Self-attention computes three vectors for each word: Query, Key, and Value.
<p align="center">
	<img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/attetion2.png" alt="Your Image Description" >
</p>
- Allows to “focus attention” on particular aspects of the input text

- Done by using a set of parameters, called "weights," that determine how much attention should be paid to each input at each time step

- These weights are computed using a combination of the input and the current hidden state of the model

- Attention weights are computed (dot product of the query, key and value matrix), then a Softmax function is applied to the dot product

 - Query (Q): Each word serves as a query to compute the attention scores.
	- Q: what I am looking for.
 - Key (K): Each word acts as a key to determine its relevance to other words.
	 - K: what I can offer.
 - Value (V): Each word contributes as a value to the attention-weighted sum.
	  - what I actually offer.

 ### Analogy for Q, K, V:

Library system

Imagine you're looking for information on a specific topic (query)

Each book in the library has a summary (key) that helps identify if it contains the information you're looking for

Once you find a match between your query and a summary, you access the book to get the detailed information (value) you need
<p align="center">
	<img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/q.png" alt="Your Image Description" >
</p>
<p align="center">
	<img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/q (2).png" alt="Your Image Description" >
</p>

---

<p align="center">
	<img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/heads.png" alt="Your Image Description" >
</p>

Attention vector for every word using this formula: 
$$Z = \text{softmax}\left(\frac{QK^T}{\sqrt{\text{Dimension of vector } Q, K \text{ or } V}}\right)V$$
Self-attention is calculated by taking the dot product of the query and key, scaled by a factor, and applying a Softmax function to obtain attention weights. These attention weights determine the importance of each word's value for the current word. 
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
$$\text{self attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_{\text{model}}}}+M\right)$$
### Output:

The final layer of the decoder is a linear projection followed by a softmax activation function. It produces a probability distribution over the vocabulary, allowing the model to generate the output word by sampling from this distribution.

### Softmax:
<p align="center">
	<img src="https://github.com/Esmail-ibraheem/Axon/blob/main/Transformer%20model/assets/matmul.png" alt="Your Image Description" >
</p>
The softmax function is a mathematical function that converts a vector of K real numbers into a probability distribution of K possible outcomes. It is a generalization of the logistic function to multiple dimensions, and used in multinomial logistic regression. The softmax function is often used as the last activation function of a neural network to normalize the output of a network to a probability distribution over predicted output classes.
​

### Linear 
convert the embeddings to word again (**_it just has weights not biases._**) 
