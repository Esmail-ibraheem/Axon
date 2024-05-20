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
