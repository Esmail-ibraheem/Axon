# [InstructGPT](https://arxiv.org/abs/2203.02155)
## [Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2312.14925), [PPO](https://arxiv.org/abs/1707.06347), [DPO](https://arxiv.org/abs/2305.18290)

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

where ![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/e8ce8cff-4498-4baa-bfbb-592ac052a709)
 is the learned RL policy, SFT is the supervised trained model, and D pretrain is the pretraining distribution. The KL reward coefficient, , and the pretraining loss coefficient, , control the strength of the KL penalty and pretraining gradients respectively. For "PPO" models, is set to 0. Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models.

#### 3. Fine-tuning with RL
![Pasted image 20240429132323](https://github.com/Esmail-ibraheem/Axon/assets/113830751/dd1de40b-e9d2-4058-9e89-9e07ff0800c6)

Training a language model with reinforcement learning was, for a long time, something that people would have thought as impossible both for engineering and algorithmic reasons. What multiple organizations seem to have gotten to work is fine-tuning some or all of the parameters of a **copy of the initial LM** with a policy-gradient RL algorithm, Proximal Policy Optimization (PPO). Some parameters of the LM are frozen because fine-tuning an entire 10B or 100B+ parameter model is prohibitively expensive (for more, see Low-Rank Adaptation ([LoRA](https://arxiv.org/abs/2106.09685)) for LMs or the [Sparrow](https://arxiv.org/abs/2209.14375) LM from DeepMind) -- depending on the scale of the model and infrastructure being used. The exact dynamics of how many parameters to freeze, or not, is considered an open research problem. PPO has been around for a relatively long time – there are [tons](https://spinningup.openai.com/en/latest/algorithms/ppo.html) of [guides](https://huggingface.co/blog/deep-rl-ppo) on how it works. The relative maturity of this method made it a favorable choice for scaling up to the new application of distributed training for RLHF. It turns out that many of the core RL advancements to do RLHF have been figuring out how to update such a large model with a familiar algorithm (more on that later).

Let's first formulate this fine-tuning task as a RL problem. First, the **policy** is a language model that takes in a prompt and returns a sequence of text (or just probability distributions over text). The **action space** of this policy is all the tokens corresponding to the vocabulary of the language model (often on the order of 50k tokens) and the **observation space** is the distribution of possible input token sequences, which is also quite large given previous uses of RL (the dimension is approximately the size of vocabulary ^ length of the input token sequence). The **reward function** is a combination of the preference model and a constraint on policy shift.

The reward function is where the system combines all of the models we have discussed into one RLHF process. Given a prompt, _x_, from the dataset, the text _y_ is generated by the current iteration of the fine-tuned policy. Concatenated with the original prompt, that text is passed to the preference model, which returns a scalar notion of “preferability”, 𝑟𝜃rθ​. In addition, per-token probability distributions from the RL policy are compared to the ones from the initial model to compute a penalty on the difference between them. In multiple papers from OpenAI, Anthropic, and DeepMind, this penalty has been designed as a scaled version of the Kullback–Leibler [(KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between these sequences of distributions over tokens, 𝑟KLrKL​. The KL divergence term penalizes the RL policy from moving substantially away from the initial pretrained model with each training batch, which can be useful to make sure the model outputs reasonably coherent text snippets. Without this penalty the optimization can start to generate text that is gibberish but fools the reward model to give a high reward. In practice, the KL divergence is approximated via sampling from both distributions (explained by John Schulman [here](http://joschu.net/blog/kl-approx.html)). The final reward sent to the RL update rule is r=r_θ​−λr_{KL}

Some RLHF systems have added additional terms to the reward function. For example, OpenAI experimented successfully on InstructGPT by mixing in additional pre-training gradients (from the human annotation set) into the update rule for PPO. It is likely as RLHF is further investigated, the formulation of this reward function will continue to evolve.

Finally, the **update rule** is the parameter update from PPO that maximizes the reward metrics in the current batch of data (PPO is on-policy, which means the parameters are only updated with the current batch of prompt-generation pairs). PPO is a trust region optimization algorithm that uses constraints on the gradient to ensure the update step does not destabilize the learning process. DeepMind used a similar reward setup for Gopher but used [synchronous advantage actor-critic](http://proceedings.mlr.press/v48/mniha16.html?ref=https://githubhelp.com) (A2C) to optimize the gradients, which is notably different but has not been reproduced externally.

_Technical detail note: The above diagram makes it look like both models generate different responses for the same prompt, but what really happens is that the RL policy generates text, and that text is fed into the initial model to produce its relative probabilities for the KL penalty. This initial model is untouched by gradient updates during training_.

Optionally, RLHF can continue from this point by iteratively updating the reward model and the policy together. As the RL policy updates, users can continue ranking these outputs versus the model's earlier versions. Most papers have yet to discuss implementing this operation, as the deployment mode needed to collect this type of data only works for dialogue agents with access to an engaged user base. Anthropic discusses this option as _Iterated Online RLHF_ (see the original [paper](https://arxiv.org/abs/2204.05862)), where iterations of the policy are included in the ELO ranking system across models. This introduces complex dynamics of the policy and reward model evolving, which represents a complex and open research question.

Kullback–Leibler (KL) divergence: $D_{KL}(P || Q)$ Distance between distributions

Constrains the RL fine-tuning to not result in a LM that outputs gibberish (to fool the reward model). Note: DeepMind did this in RL Loss (not reward), see GopherCite

#### 3.1. Fine-tuning with RL - PPO

Proximal Policy Optimization (PPO)- on-policy algorithm,- works with discrete or continuous actions,- optimized for parallelization.

If we apply the “PPO” like described, the language model (our policy) may just learn to output whatever the reward model wants to see to maximize its return. We for sure want the language model to receive good rewards, but as the same time we want the language model to output something that still looks like the training data it was trained upon. For this reason, for every reward generated by the model, we penalize the reward by the KL-Divergence between the logits generated by the policy being optimized and a frozen version of the language model.

**Baselines**. We compare the performance of our PPO models to our SFT models and GPT-3. We also compare to GPT-3 when it is provided a few-shot prefix to ‘prompt’ it into an instruction-following mode (GPT-3-prompted). This prefix is prepended to the user-specified instruction.6 We additionally compare InstructGPT to fine-tuning 175B GPT-3 on the FLAN (Wei et al., 2021) and T0 (Sanh et al., 2021) datasets, which both consist of a variety of NLP tasks, combined with natural language instructions for each task (the datasets differ in the NLP datasets included, and the style of instructions used). We fine-tune them on approximately 1 million examples respectively and choose the checkpoint which obtains the highest reward model score on the validation set. See Appendix C for more training details.

---

### Recapping RLHF techniques:

![Pasted image 20240430131111](https://github.com/Esmail-ibraheem/Axon/assets/113830751/732d6662-1c7b-446b-9966-f0be505484d4)

![Pasted image 20240430125633](https://github.com/Esmail-ibraheem/Axon/assets/113830751/c9dfe260-25f0-4ba1-9017-28fef498a5dc)

---

## Let’s talk about trajectories…

As said previously, the goal in RL is to select a policy which maximizes the expected

![Scheme-of-Deep-Reinforcement-Learning](https://github.com/Esmail-ibraheem/Axon/assets/113830751/37bb58be-b8e7-4fdb-9412-76fbd544900f)

return when the agent acts according to it. More formally:
![Pasted image 20240429152538](https://github.com/Esmail-ibraheem/Axon/assets/113830751/1948b5d7-f049-4e27-b917-a2943f25d965)

The expected return of a policy is the expected return over all possible trajectories.
![Pasted image 20240429152616](https://github.com/Esmail-ibraheem/Axon/assets/113830751/12c96d6e-aaae-4ee3-b2d4-f882c4339e5c)

A trajectory is a series of (action, state), starting from an initial state
![Pasted image 20240429152631](https://github.com/Esmail-ibraheem/Axon/assets/113830751/c94207ae-fce7-4717-a4aa-164cda5c9fa3)

We will model the next state as being stochastic (suppose that the cat is drunk and doesn’t always succeed in moving correctly)
![Pasted image 20240429152649](https://github.com/Esmail-ibraheem/Axon/assets/113830751/3c880212-c2c5-4e82-adc5-fb48154d787e)

We can thus define the probability of a trajectory as follows:
![Pasted image 20240429152717](https://github.com/Esmail-ibraheem/Axon/assets/113830751/e2691e26-f7c2-4eeb-a129-a7299167ce39)

We will always work with discounted rewards (we prefer immediate rewards instead of future):
![Pasted image 20240429152733](https://github.com/Esmail-ibraheem/Axon/assets/113830751/f1aa6f65-cf06-45c5-8ead-6c563c5c1d9d)

This is an expectation, which means we can approximate it with a sample mean by collecting a set D of trajectories.

![Pasted image 20240429152913](https://github.com/Esmail-ibraheem/Axon/assets/113830751/205ea082-5e66-4275-a106-e7de54929d50)

---

### RLHF-Proximal Policy Optimization algorithm:

**The PPO loss:** 

**Algorithm** The surrogate losses from the previous sections can be computed and differentiated with a minor change to a typical policy gradient implementation. For implementations that use automatic differentation, one simply constructs the loss $L^{CLIP}$ or $L^{KLPEN}$ instead of $L^{PG}$, and one performs multiple steps of stochastic gradient ascent on this objective. Most techniques for computing variance-reduced advantage-function estimators make use a learned state-value function $V (s)$; for example, generalized advantage estimation [Sch+15a], or the 4 finite-horizon estimators in [Mni+16]. If using a neural network architecture that shares parameters between the policy and value function, we must use a loss function that combines the policy surrogate and a value function error term. This objective can further be augmented by adding an entropy bonus to ensure sufficient exploration, as suggested in past work [Wil92; Mni+16]. Combining these terms, we obtain the following objective, which is (approximately) maximized each iteration: 

![image](https://github.com/Esmail-ibraheem/Axon/assets/113830751/a64ce3e8-8bb3-492d-9f9c-0f61d46065b9)

where c1,c2 are coefficients, and S denotes an entropy bonus, and $L_{t}^{VF}$ is a squared-error loss $(V_θ(s_t) − V_{t}^{targ})^2$.

![Pasted image 20240429153634](https://github.com/Esmail-ibraheem/Axon/assets/113830751/468a0b77-f82f-4077-bd1d-1816ced7f270)


A proximal policy optimization (**PPO**) algorithm that uses **fixed-length trajectory** segments is shown below. Each iteration, each of **N** (parallel) actors collect **T** timesteps of data. Then we construct the surrogate loss on these **NT** timesteps of data, and optimize it with minibatch **SGD** (or usually for better performance, Adam [KB14]), for K epochs.

![Pasted image 20240501133952](https://github.com/Esmail-ibraheem/Axon/assets/113830751/18c7a3fe-04d9-4a4e-a1ff-fc8fb49c7eea)

### RLHF-Direct Preference Optimization algorithm:

![Pasted image 20240430135953](https://github.com/Esmail-ibraheem/Axon/assets/113830751/74de5375-ab9b-4566-af77-202625ab5012)

**Figure:** DPO optimizes for human preferences while avoiding reinforcement learning. Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward. In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification objective, fitting an implicit reward model whose corresponding optimal policy can be extracted in closed form.

>In this paper, we show how to directly optimize a language model to adhere to human preferences, without explicit reward modeling or reinforcement learning. We propose Direct Preference Optimization (DPO), an algorithm that implicitly optimizes the same objective as existing RLHF algorithms (reward maximization with a KL-divergence constraint) but is simple to implement and straight forward to train. Intuitively, the DPO update increases the relative log probability of preferred to dispreferred responses, but it incorporates a dynamic, per-example importance weight that prevents the model degeneration that we find occurs with a naive probability ratio objective. Like existing algorithms, DPO relies on a theoretical preference model (such as the Bradley-Terry model; [5]) that measures how well a given reward function aligns with empirical preference data. However, while existing methods use the preference model to define a preference loss to train a reward model and then train a policy that optimizes the learned reward model, DPO uses a change of variables to define the preference loss as a function of the policy directly. Given a dataset of human preferences over model responses, DPO can therefore optimize a policy using a simple binary cross entropy objective, producing the optimal policy to an implicit reward function fit to the preference data. 
>
>Our main contribution is Direct Preference Optimization (DPO), a simple RL-free algorithm for training language models from preferences. Our experiments show that DPO is at least as effective as existing methods, including PPO-based RLHF, for learning from preferences in tasks such as sentiment modulation, summarization, and dialogue, using language models with up to 6B parameters.

Motivated by the challenges of applying reinforcement learning algorithms on large-scale problems such as fine-tuning language models, our goal is to derive a simple approach for policy optimization using preferences directly. Unlike prior RLHF methods, which learn a reward and then optimize it via RL, our approach leverages a particular choice of reward model parameterization that enables extraction of its optimal policy in closed form, without an RL training loop. As we will describe next in detail, our key insight is to leverage an analytical mapping from reward functions to optimal policies, which enables us to transform a loss function over reward functions into a loss function over policies. This change-of-variables approach avoids fitting an explicit, standalone reward model, while still optimizing under existing models of human preferences, such as the Bradley-Terry model. In essence, the policy network represents both the language model and the (implicit) reward.

**Deriving the DPO objective.** We start with the same RL objective as prior work, Eq. 3, under a general reward function r. Following prior work [29, 28, 17, 15], it is straightforward to show that the optimal solution to the KL-constrained reward maximization objective in Eq. 3 takes the form:


$$π_r(y | x) = \frac{1}{Z(x)} π_{ref}(y | x)exp (\frac1βr(x,y)),$$
where $Z(x) = \sum_y π_{ref}(y | x) exp (\frac1βr(x,y))$ is the partition function. See Appendix A.1 for a complete derivation. Even if we use the MLE estimate rϕ of the ground-truth reward function r∗, it is still expensive to estimate the partition function Z(x) [17, 15], which makes this representation hard to utilize in practice. However, we can rearrange Eq. 4 to express the reward function in terms of its corresponding optimal policy πr, the reference policy πref, and the unknown partition function Z(·). Specifically, we first take the logarithm of both sides of Eq. 4 and then with some algebra we obtain:

$$r(x,y) = βlog \frac {π_r(y | x)} {π_{ref}(y | x)} + β logZ(x).$$

Wecan apply this reparameterization to the ground-truth reward r∗ and corresponding optimal model π∗. Fortunately, the Bradley-Terry model depends only on the difference of rewards between two completions, i.e.,$p∗(y_1 ≻ y_2 | x) = σ(r∗(x,y_1) − r∗(x,y_2)).$ Substituting the reparameterization in Eq. 5 for r∗(x,y) into the preference model Eq. 1, the partition function cancels, and we can express the human preference probability in terms of only the optimal policy π∗ and reference policy πref. Thus, the optimal RLHF policy π∗ under the Bradley-Terry model satisfies the preference model:

$$p∗(y_1 ≻ y_2 | x) = \frac 1  {1 +exp (βlog\frac {π^*(y_2|x)} {π_{ref}(y_2|x)} − β log \frac{π^∗(y_1|x)} {π_{ref}(y_1|x)})}$$
The derivation is in Appendix A.2. While Eq. 6 uses the Bradley-Terry model, we can similarly derive expressions under the more general Plackett-Luce models [30, 21], shown in Appendix A.3. Nowthat we have the probability of human preference data in terms of the optimal policy rather than the reward model, we can formulate a maximum likelihood objective for a parametrized policy πθ. Analogous to the reward modeling approach (i.e. Eq. 2), our policy objective becomes:


![Pasted image 20240501142641](https://github.com/Esmail-ibraheem/Axon/assets/113830751/b9776cc1-3c58-4727-bfb5-f50c8dfbcd40)
![Pasted image 20240501142731](https://github.com/Esmail-ibraheem/Axon/assets/113830751/f8ba8ae2-6050-44a3-8e7d-e5e349dfc355)

where $\hat r_θ(x,y) = β log \frac{πθ(y|x)}{π_{ref}(y|x)}$ is the reward implicitly defined by the language model $π_θ$ and reference model $π_{ref}$. the gradient of the loss function $L_{DPO}$ increases the likelihood of the preferred completions yw and decreases the likelihood of dispreferred completions $y_l$. Importantly, the examples are weighed by how much higher the implicit reward model $\hat r_θ$ rates the dispreferred completions, scaled by β, i.e, how incorrectly the implicit reward model orders the completions, accounting for the strength of the KL constraint. Our experiments suggest the importance of this weighting, as a naïve version of this method without the weighting coefficient can cause the language model to degenerate

**DPOoutline.** The general DPO pipeline is as follows: 1) Sample completions $y_1,y_2 ∼ π_{ref}(· | x)$ for every prompt x, label with human preferences to construct the offline dataset of preferences D ={x^{(i)},y^{(i)}_w ,y_l)^{(i)}}^{N} _{i=1} and 2)optimize the language model π_θ to minimize L_{DPO} for the given πref and D and desired β. In practice, one would like to reuse preference datasets publicly available, rather than generating samples and gathering human preferences. Since the preference datasets are sampled using πSFT, we initialize π_{ref} = π_{SFT} whenever available. However, when $π_{SFT}$ is not available, we initialize $π_{ref}$ by maximizing likelihood of preferred completions $(x,y_w)$, that is, $π_{ref} = argmax_π E_{x,y_w}∼D [logπ(y_w | x)]$. This procedure helps mitigate the distribution shift between the true reference distribution which is unavailable, and πref used by DPO.

![Pasted image 20240501144218](https://github.com/Esmail-ibraheem/Axon/assets/113830751/bf142482-5507-44b4-8015-a8e6adb65fdc)
![Pasted image 20240501144222](https://github.com/Esmail-ibraheem/Axon/assets/113830751/af6c3025-1ff4-4b6e-8b2f-21a6b2142163)
![Pasted image 20240501144245](https://github.com/Esmail-ibraheem/Axon/assets/113830751/56ac960c-6fa8-4fd8-98f6-abf876514db9)

>  **DPO-algorithm in a simple way**
> - Sample good/bad response, -
> - Run pairs through 2 models (active and reference), 
> - Backprop, 
> - Profit

### Preliminaries:

We review the **RLHF** pipeline in Ziegler et al. (and later [38, 1, 26]). It usually includes three phases: 1) **supervised fine-tuning (SFT)**; 2) **preference sampling and reward learning** and 3) **RL optimization.** 

**SFT:** RLHF typically begins by fine-tuning a pre-trained LM with supervised learning on high-quality data for the downstream task(s) of interest (dialogue, summarization, etc.), to obtain a model $π^{SFT}$.

**Reward Modelling** Phase: In the second phase the SFT model is prompted with prompts x to produce pairs of answers $(y_1,y_2) ∼ π_{SFT}(y | x)$. These are then presented to human labelers who express preferences for one answer, denoted as $y_w ≻ y_l | x$ where $y_w$ and $y_l$ denotes the preferred and dispreferred completion amongst $(y_1,y_2)$ respectively. The preferences are assumed to be generated by some latent reward model $r∗(y,x)$, which we do not have access to. There are a number of approaches used to model preferences, the Bradley-Terry (BT) [5] model being a popular choice (although more general Plackett-Luce ranking models [30, 21] are also compatible with the framework if we have access to several ranked answers). The BT model stipulates that the human preference distribution p∗ can be written as:


$$p∗(y_1 ≻ y_2 | x) = \frac {exp(r^∗(x,y_1))} {exp(r^∗(x,y_1)) + exp(r^∗(x,y_2))} .$$
Assuming access to a static dataset of comparisons $D = (x^{(i)},y^{(i)}_w ,y^{(i)} _l)^N _{i=1}$ sampled from $p^∗$, we can parametrize a reward model $r_ϕ(x,y)$ and estimate the parameters via maximum likelihood. Framing the problem as a binary classification we have the negative log-likelihood loss:

$$L_R(r_ϕ,D) = −E(x,y_w,y_l)∼D [logσ(r_ϕ(x,y_w) − r_ϕ(x,y_l))]$$

where σ is the logistic function. In the context of LMs, the network $r_ϕ(x,y)$ is often initialized from the SFT model $π_{SFT}(y | x)$ with the addition of a linear layer on top of the final transformer layer that produces a single scalar prediction for the reward value [49]. To ensure a reward function with lower variance, prior works normalize the rewards, such that $Ex,y∼D [r_ϕ(x,y)] = 0$ for all x.

**RL Fine-Tuning Phase:** During the RL phase, we use the learned reward function to provide feedback to the language model. In particular, we formulate the following optimization problem

$$max_{π_θ} Ex∼D,y∼π_θ(y|x) [r_ϕ(x,y)] − βD_{KL}[π_θ(y | x) || π_{ref}(y | x)]$$

where β is a parameter controlling the deviation from the base reference policy πref, namely the initial SFT model $π^{SFT}$. In practice, the language model policy πθ is also initialized to $π^{SFT}.$ The added constraint is important, as it prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse to single high-reward answers. Due to the discrete nature of language generation, this objective is not differentiable and is typically optimized with reinforcement learning. The standard approach [49, 38, 1, 26] has been to construct the reward function $r(x,y) = r_ϕ(x,y) −β(log_{π_θ}(y | x) −logπ_{ref}(y | x))$, and maximize using PPO.


### PPO vs DPO: 

![Pasted image 20240501132016](https://github.com/Esmail-ibraheem/Axon/assets/113830751/34829721-c183-4299-b41d-29f451f6972a)

**Figure**: **Left.** The frontier of expected reward vs KL to the reference policy. DPO provides the highest expected reward for all KL values, demonstrating the quality of the optimization. **Right.** TL; DR summarization win rates vs. human-written summaries, using GPT-4 as an evaluator. DPO exceeds PPO’sbest-case performance on summarization while being more robust to changes in the sampling temperature.
