# Reinforcement Learning from Human Feedback:

- ##### Proximal Policy Optimization algorithm
- ##### Direct Preference Optimization algorithm
- ##### Supervised Fine-tuning 

**AI alignment:** A large language model typically is pretrained on a massive amount of data, for example the entire Wikipedia and billions of web pages. This gives the language model a vast “knowledge” of information to complete any prompt in a reasonable way. However, to use an LLM as a chat assistant (for example ChatGPT) we want to force the language model to follow a particular style. For example, we may want the following: 

- Do not use offensive language 
- Do not use racist expressions 
- Answer questions using a particular style The goal of AI alignment is to align the model’s behavior with a desired behavior.

**Reinforcement Learning:** Reinforcement Learning is concerned with how an intelligent agent should take actions in an environment to maximize the cumulative reward.


![RL2](https://github.com/Esmail-ibraheem/Axon/assets/113830751/21108b4e-c2ce-4572-acfe-0d37312ca221)

**The RL setup:**

**Agent:** the cat 

**State:** the position of the cat (x, y) in the grid 

**Action:** at each position, the cat can move to one of the 4-directionally connected cells. If a move is invalid, the cell will not move and remain in the same position. Every time the cat makes a move, it results in a new state and a reward. 

![rlgame](https://github.com/Esmail-ibraheem/Axon/assets/113830751/781fea3d-de78-46e3-bcd9-8f596d28da30)

**Reward model:** 
- A move to another empty cell results in a reward of 0. 
- A move towards the broom, will result in a reward of -1. 
- A move towards the bathtub will result in a reward of -10 and the cat fainting (episode over). The cat will be respawned at the initial position again. 
- A move towards the meat will result in a reward of +100 

**Policy:** a policy rules how the agent selects the action to perform given the state it is in: $a_𝑡 \sim \pi(\cdot | 𝑠𝑡)$ The goal in RL is to select a policy that maximizes the expected return

![RL1](https://github.com/Esmail-ibraheem/Axon/assets/113830751/4ba3fb44-61ac-4e48-b6f7-396a64c2aa72)

**The RL setup: connection to language models** 

**Agent:** the language model itself State: the prompt (input tokens) 

**Action:** which token is selected as the next token 

**Reward model:** the language model should be rewarded for generating “good responses” and should not receive any reward for generating “bad responses”. 

**Policy:** In the case of language models, the policy is the language model itself! Because it models the probability of the action space given the current state of the agent: $a_𝑡 \sim \pi(\cdot | 𝑠𝑡)$

---

## History: RLHF for decision making

![deep_rl3](https://github.com/Esmail-ibraheem/Axon/assets/113830751/d6a7baaf-047a-4659-a6a0-748f4a3045df)
![Pasted image 20240429151951](https://github.com/Esmail-ibraheem/Axon/assets/113830751/d67ab0d9-27af-43eb-b485-82c2931114ff)
![Pasted image 20240429122909](https://github.com/Esmail-ibraheem/Axon/assets/113830751/3101840c-7dc6-4e90-bafc-ef5218b91613)

## RLHF: Let’s take it step by step

Reinforcement learning from Human Feedback (also referenced as RL from human preferences) is a challenging concept because it involves a multiple-model training process and different stages of deployment. In this blog post, we’ll break down the training process into three core steps:

1. Pretraining a language model (LM),
2. gathering data and training 
3. a reward model, and
4. fine-tuning the LM with reinforcement learning.

![Pasted image 20240430122335](https://github.com/Esmail-ibraheem/Axon/assets/113830751/790eba4f-b02b-4d15-81a0-fad8ad958633)

Figure: A diagram illustrating the three steps of our method: (1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO) on this reward model. Blue arrows indicate that this data is used to train one of our models. In Step 2, boxes A-D are samples from our models that get ranked by labelers. See Section 3 for more details on our method.

**Step 1:** Collect demonstration data, and train a supervised policy. Our labelers provide demon strations of the desired behavior on the input prompt distribution (see Section 3.2 for details on this distribution). We then fine-tune a pretrained GPT-3 model on this data using supervised learning. 

**Step 2:** Collect comparison data, and train a reward model. We collect a dataset of comparisons between model outputs, where labelers indicate which output they prefer for a given input. We then train a reward model to predict the human-preferred output. 

**Step 3:** Optimize a policy against the reward model using PPO. We use the output of the RM as a scalar reward. We fine-tune the supervised policy to optimize this reward using the PPO algorithm (Schulman et al., 2017). 

**Steps 2 and 3** can be iterated continuously; more comparison data is collected on the current best policy, which is used to train a new RM and then a new policy. In practice, most of our comparison data comes from our supervised policies, with some coming from our PPO policies.

---

### train models with three different techniques:

![rlhf3](https://github.com/Esmail-ibraheem/Axon/assets/113830751/b1932c86-b9cd-4bbb-ac73-e3e022f54bd5)

**Supervised fine-tuning (SFT)**. We fine-tune GPT-3on our labeler demonstrations using supervised learning. We trained for 16 epochs, using a cosine learning rate decay, and residual dropout of 0.2. We do our final SFT model selection based on the RM score on the validation set. Similarly to Wu et al. (2021), we find that our SFT models overfit on validation loss after 1 epoch; however, we find that training for more epochs helps both the RM score and human preference ratings, despite this overfitting. 

#### 1. Language Model Pretraining:

![rlhf_step11](https://github.com/Esmail-ibraheem/Axon/assets/113830751/dc87e901-5467-4c1e-8091-5ff7efbfd223)
As a starting point RLHF use a language model that has already been pretrained with the classical pretraining objectives (see this [blog post](https://huggingface.co/blog/how-to-train) for more details). OpenAI used a smaller version of GPT-3 for its first popular RLHF model, [InstructGPT](https://openai.com/blog/instruction-following/). In their shared papers, Anthropic used transformer models from 10 million to 52 billion parameters trained for this task. DeepMind has documented using up to their 280 billion parameter model [Gopher](https://arxiv.org/abs/2112.11446). It is likely that all these companies use much larger models in their RLHF-powered products.

This initial model _can_ also be fine-tuned on additional text or conditions, but does not necessarily need to be. For example, OpenAI fine-tuned on human-generated text that was “preferable” and Anthropic generated their initial LM for RLHF by distilling an original LM on context clues for their “helpful, honest, and harmless” criteria. These are both sources of what we refer to as expensive, _augmented_ data, but it is not a required technique to understand RLHF. Core to starting the RLHF process is having a _model that responds well to diverse instructions_.

In general, there is not a clear answer on “which model” is the best for the starting point of RLHF. This will be a common theme in this blog – the design space of options in RLHF training are not thoroughly explored.

Next, with a language model, one needs to generate data to train a **reward model**, which is how human preferences are integrated into the system

common training techniques in NLP:- Unsupervised sequence prediction- Data scraped from web- No single answer on “best” model size (examples in industry range 10B-280B parameters)

**Reward modeling (RM).** Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward. In this paper we only use 6B RMs, as this saves a lot of compute, and we found that 175B RM training could be unstable and thus was less suitable to be used as the value function during RL (see Appendix C for more details). In Stiennon et al. (2020), the RM is trained on a dataset of comparisons between two model outputs on the same input. They use a cross-entropy loss, with the comparisons as labels—the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler. In order to speed up comparison collection, we present labelers with anywhere between $K = 4 and K =9$ responses to rank. This produces K 2 comparisons for each prompt shown to a labeler. Since comparisons are very correlated within each labeling task, we found that if we simply shuffle the comparisons into one dataset, a single pass over the dataset caused the reward model to overfit.5 Instead, we train on all K 2 comparisons from each prompt as a single batch element. This is much more computationally efficient because it only requires a single forward pass of the RM for each completion (rather than K 2 forward passes for K completions) and, because it no longer overfits, it achieves much improved validation accuracy and log loss. Specifically, the loss function for the reward model is: $$loss(\theta) = \frac{1}{2K} E{(x, y_w, y_l) \sim D} \left[ \log\left( \sigma\left(r_{\theta}(x, y_w) - r_{\theta}(x, y_l)\right) \right) \right]$$ is the scalar output of the reward model for prompt x and completion y with parameters , $y_w$ is the preferred completion out of the pair of yw and $y_l$, and $D$ is the dataset of human comparisons.

#### 2. Reward Model Training:

![rlhf_step22](https://github.com/Esmail-ibraheem/Axon/assets/113830751/fee39892-9ac3-4177-88c7-fd78d6749d34)

Generating a reward model (RM, also referred to as a preference model) calibrated with human preferences is where the relatively new research in RLHF begins. The underlying goal is to get a model or system that takes in a sequence of text, and returns a scalar reward which should numerically represent the human preference. The system can be an end-to-end LM, or a modular system outputting a reward (e.g. a model ranks outputs, and the ranking is converted to reward). The output being a **scalar** **reward** is crucial for existing RL algorithms being integrated seamlessly later in the RLHF process.

These LMs for reward modeling can be both another fine-tuned LM or a LM trained from scratch on the preference data. For example, Anthropic has used a specialized method of fine-tuning to initialize these models after pretraining (preference model pretraining, PMP) because they found it to be more sample efficient than fine-tuning, but no one base model is considered the clear best choice for reward models.

The training dataset of prompt-generation pairs for the RM is generated by sampling a set of prompts from a predefined dataset (Anthropic’s data generated primarily with a chat tool on Amazon Mechanical Turk is [available](https://huggingface.co/datasets/Anthropic/hh-rlhf) on the Hub, and OpenAI used prompts submitted by users to the GPT API). The prompts are passed through the initial language model to generate new text.

Human annotators are used to rank the generated text outputs from the LM. One may initially think that humans should apply a scalar score directly to each piece of text in order to generate a reward model, but this is difficult to do in practice. The differing values of humans cause these scores to be uncalibrated and noisy. Instead, rankings are used to compare the outputs of multiple models and create a much better regularized dataset.

There are multiple methods for ranking the text. One method that has been successful is to have users compare generated text from two language models conditioned on the same prompt. By comparing model outputs in head-to-head matchups, an [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) system can be used to generate a ranking of the models and outputs relative to each-other. These different methods of ranking are normalized into a scalar reward signal for training.

An interesting artifact of this process is that the successful RLHF systems to date have used reward language models with varying sizes relative to the text generation (e.g. OpenAI 175B LM, 6B reward model, Anthropic used LM and reward models from 10B to 52B, DeepMind uses 70B Chinchilla models for both LM and reward). An intuition would be that these preference models need to have similar capacity to understand the text given to them as a model would need in order to generate said text.

At this point in the RLHF system, we have an initial language model that can be used to generate text and a preference model that takes in any text and assigns it a score of how well humans perceive it. Next, we use **reinforcement learning (RL)** to optimize the original language model with respect to the reward model.

How to capture human sentiments in samples and curated text? What is the loss! **Goal:** get a model that maps input text → scalar reward

> We aim for our model to emulate human responses more realistically. While the transformer model typically selects tokens with the highest prediction probability, we avoid this approach as it yields robotic-sounding answers. Instead, we employ various decoding strategies such as Top-K, Nucleus, Temperature, Greedy, etc. In this project, I utilize temperature decoding, where adjusting the temperature constant affects token selection: lowering the constant favors tokens with higher probabilities, while raising it favors those with lower probabilities.

**Reinforcement learning (RL).** Once again following Stiennon et al. (2020), we fine-tuned the SFT model on our environment using PPO (Schulman et al., 2017). The environment is a bandit environment which presents a random customer prompt and expects a response to the prompt. Given the prompt and response, it produces a reward determined by the reward model and ends the episode. In addition, we add a per-token KL penalty from the SFT model at each token to mitigate over optimization of the reward model. The value function is initialized from the RM. We call these models “PPO.” 

We also experiment with mixing the pretraining gradients into the PPO gradients, in order to fix the performance regressions on public NLP datasets. We call these models “PPO-ptx.” We maximize the following combined objective function in RL training:
![Pasted image 20240429125320](https://github.com/Esmail-ibraheem/Axon/assets/113830751/63230496-607d-45a4-a93f-1e1acbe3bbce)

where $ \begin{align*} \pi^{\text{RL}} \\ \phi \end{align*} $ is the learned RL policy, SFT is the supervised trained model, and D pretrain is the pretraining distribution. The KL reward coefficient, , and the pretraining loss coefficient, , control the strength of the KL penalty and pretraining gradients respectively. For "PPO" models, is set to 0. Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models.
