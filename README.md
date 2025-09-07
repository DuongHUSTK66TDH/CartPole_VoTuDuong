# [PYTORCH] Reinforcement Learning and Behaviour Clonning for playing CartPole 

## Introduction
Here is my python source code for training an agent to play CartPole by using Reinforcement, Behavior Cloning and the combination of two techniques. 

For only Reinforcement, I had used a Deep Q-learning network.

For Behavior Cloning, I had used a simple Deep Neural Network trained on the dataset from [NathanGavenski](https://huggingface.co/datasets/NathanGavenski/CartPole-v1/viewer/default/train?p=1&views%5B%5D=train).

For RL + BC, I had pretrained using same dataset from BC part, then using Cycle-of-Learning (CoL) to learn from the environment. The method CoL is presented in the paper **Integrating Behavior Cloning and Reinforcement Learning for Improved Performance in Dense and Sparse Reward Environments ** [paper](https://arxiv.org/abs/1910.04281).

<a align="center">
  <figure style="display: inline-block; margin: 10px;">
    <img src="Gif demo\RL_only_LR=0.0001.gif" width="200">
    <figcaption style="text-align: center;">RL_only_LR=0.0001</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 10px;">
    <img src="Gif demo\RL_only_LR=0.001.gif" width="200">
    <figcaption style="text-align: center;">RL_only_LR=0.001</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 10px;">
    <img src="Gif demo\BC_only.gif" width="200">
    <figcaption style="text-align: center;">BC_only</figcaption>
  </figure>
  <figure style="display: inline-block; margin: 10px;">
    <img src="Gif demo\RL_BC.gif" width="200">
    <figcaption style="text-align: center;">RL_BC</figcaption>
  </figure><br/>
  <i>Sample results</i>
</a>

## Motivation

This is a small project, but it is very meaningful to me. It has given me my first insights into a field I haven't had the opportunity to work with before. I've spent a lot of time learning about Reinforcement Learning and Behavior Cloning methods. Teaching an agent how to interact in different environments is truly beneficial. Starting with an introduction to the differences between Reinforcement Learning, Behavior Cloning, and how to combine them, I've gained a more intuitive understanding of these methods when they're placed side by side. Furthermore, I've also had the chance to learn about Deep Q-learning Network and Actor-Critic (A3C). The code below was written and used by me to train a model using three different methods: RL-only, BC-only, and RL+BC.

## Explanation in layman's term
If you are already familiar to reinforcement learning in general, Behavior Cloning or A3C in particular, you could skip this part. I write this part for explaining what is A3C algorithm, how and why it works, to people who are interested in or curious about A3C or my implementation, but do not understand the mechanism behind. Therefore, you do not need any prerequiste knowledge for reading this part :relaxed:

### Deep Q-learning Network
The agent will output Q-values for actions when given the environment's input state. Subsequently, the agent will execute an action and receive a reward and a new environment state. Based on this, the agent will adjust itself to make more accurate judgments. Simply put, when the agent is provided with information, it will score the actions it can take based on that information. From there, it will naturally choose the action with the highest score. After performing that action, if the outcome is good, the agent will be rewarded to be more likely to repeat that action in similar states. Conversely, if the action leads to a bad outcome, the agent will be penalized to avoid that action in similar situations in the future.
A major advantage of a Deep Q-learning Network is that it can be used when the environment's input state is continuous. For example, in the game Mario, the position of the character or environmental agents are continuous coordinates. This gives the Deep Q-learning Network a superior and more memory-efficient capability compared to a Q-value table.

### Actor-Critic
Your agent has 2 parts called **actor** and **critic**, and its goal is to make both parts perfom better over time by exploring and exploiting the environment. Let imagine a mischievous student (**actor**) is discovering set of new exercises, while his teacher (**critic**) oversees him, to make sure that he does not do missunderstood. Whenever the student does anything good, his teacher will praise and encourage him to repeat that action in the future. And of course, when the student does anything wrong, he will get warning from his teacher. The more the student done his exercises, and takes different actions, the more feedback, both positive and negative, he gets from his teacher. The goal of the student is, to collect as many positive feedback as possible from his teacher, while the goal of the teacher is to evaluate his student's action better. In other word, we have a win-win relationship between the student and his teacher, or equivalently between **actor** and **critic**.

Rather than rewarding every one of the actor's actions, the critic is capable of giving out heavy penalties if the actor performs poorly. This helps the actor limit useless early exploration, avoiding many consecutive mistakes.
### Behavior Cloning
The agent will receive experience from an "expert" and learn from it. For example, a student receives a series of detailed instructions from a top industry expert on how to perform Task A. All the student has to do is follow the given instructions. Of course, the great advantage of this method is that the student doesn't waste time exploring Task A, as everything he needs to know is in the expert's instructions. This method is very advantageous if the expert's experience is extensive and good enough, and the environment is not overly complex. However, conversely, the student could make many mistakes, and may even fail to ever complete the task, if they only rote learn from the expert's experience without exploring the task themselves.

### Combination of Behavior Cloning and Reinforcement Learning
By combining these two methods, we can leverage the advantages of both. The agent can both learn from the expert's dataset, accumulating experience, and "skip" the initial fumbling period to focus on exploiting and perfecting its skills once it has some experience. This improves both the training speed and the quality of training, as the agent can accumulate its own knowledge in addition to what it learned from the expert.


## How to use my code
With my code, you can:
* **Train your model** by running **python train_BC_only.py**,**python train_RL_only.py** or **python train_RL_BC.py**
* **Test your trained model** by running **python Simulation.py**

Remember to **install all** nescessary lib by running **pip install -r setup.txt**
If you want to change the hyperparameter, please adjust in specific training file.
## Trained models

You could find some trained models I have trained in [CartPole trained models](https://drive.google.com/drive/folders/19yAoZ-YLlJE8cJl7cUTXUWy7J_fVdM4v?usp=drive_link)
 
## Requirements

* **gymnasium**
* **gymnasium[classic-control]**
* **torch**
* **numpy**
* **pandas**
* **tqdm**
* **huggingface_hub**
* **scikit-learn**
* **matplotlib**
* **opencv-python**

## Acknowledgements
For training Behavior Cloning, I had used dataset from [NathanGavenski](https://huggingface.co/datasets/NathanGavenski/CartPole-v1/viewer/default/train?p=1&views%5B%5D=train).