# attention is all you need

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
