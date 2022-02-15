
<H1 align="center">
Playing Super Mario Bros with the Double Deep Q Learning algorithm
<br>
</H1>

<p align="right"> Giorgio Strano - 1809528 </p>
<p align="right"> Machine Learning course  2021/2022</p>
<br>

### **Index:**

- ### Introduction, aims and goals 

- ### An introduction to Q learning
  
- ### The double deep Q algorithm

- ### The implementation 

- ### Results and conclusions

<br>
<br>

# Introduction, aims and goals


For this machine learning project, I decided to give my best shot at the implementation of an algorithm that fascinated me for a long time, and that I consider to be one of the most elegant and intriguing algorithms in the field of reinforcement learning: the Double Deep Q algorithm.

I first discovered this algorithm many years ago, but I tried to understand way before I had any of the sufficient knowledge to do so. Years later I discovered *Gym*, by *OpenAI*, which is an amazing framework for developing and testing RL algorithms.

In the words of the authors,
> Gym is a toolkit for developing and comparing reinforcement learning algorithms.
> 
> The gym library is a collection of test problems — environments — that you can use to work out your reinforcement learning algorithms. These environments have a shared interface, allowing you to write general algorithms.

This framework relit my curiosity for reinforcement learning, and led me to learn and implement a few basic algorithms, from random searches, to hill-climbing and simple policy gradient methods. The obvious next topic, that until now I did not have a chance to explore yet, was deep Q learning and its double-network version.

I intend to understand and implement the algorithm, and test my implementation on different games to experience how it behaves and how it can perform in various environments. The final goal is to train it to learn its way through an extremely complex environment and beat a level of the classic Super Mario Bros game.

<br>
<br>

# Technologies and hardware

- Python
- Pytorch
- CUDA
- gym
- neofetch

- OOP
- easy to read, tweak and extend
- conda environment


<br>
<br>

# Introduction to Q learning

Before explaining the Double Deep Q algorithm, I belive it is important to take a step back and understand its origins and the core concepts behind Q learning. I will try to give an overview of a few basic topics of reinforcement learning that are fundamental to understand the following ones, starting from its origin with TD learning. I will not go too deep in the details of each one, to keep this section clear and concise, so a basic knowledge of reinforcement learning will be assumed.

<br>

## Temporal Difference Learning

The Temporal Difference (TD) learning algorithm was introduced by Sutton in 1988. Its main idea is to *predict a quantity that depends on future values of a given signal*. In the context of reinforcement learning, where we try to predict some value, this means to approximate the current estimate based on the previously learned ones.

<br>

## TD prection

In **TD prediction**, the value we try to predict is the value $V(s)$ of a given state $s$. TD learning uses in this case the following update function:

$V(s) := V(s) + \alpha (r + \gamma \text{ } V(s') - V(s))$ ,

meaning that the value of a previous state gets updated by (*reward + discounted current state value - previous state value*) multiplied by the learning rate $\alpha$.

In this formulation, $V(s)$ is the expected reward, and $(r + \gamma \text{ } V(s'))$ is the actual reward, so the difference between the two is simply the prediction error, called *TD error*, which we want to minimise.

<br> 

## TD control and basic Q learning

In **TD control** algorithms, what we try to estimate is not the value of a state per se, but rather the value of performing an action $a$ while in a specific state $s$, which we call *Q value*. This value can be updated using the following equation:

$Q(s,a) := Q(s,a) + \alpha (r+\gamma \text{ } max(Q(s',a') - Q(s,a)))$

the difference from the TD prediction formula is that here the target value is computed as the reward received $r$, plus the discounted value of the *best possible* state-action pair in the resulting state $s'$. 

An alorithm based on Q learning would therefore keep a table (Q table), assinging a value to each possible state-action pair, and proceed with the following actions: 

1. Initialise the $Q$ table to arbitrary values
2. Use the *epsilon-greedy policy* (see following section) to select an action to perform.
3. Perform the action to progress to the next state
4. Update the $Q$ table using the update rule
5. If the terminal state is not reached, go back to step 2

<br> 

## Epsilon-greedy policy

The epsilon-greedy policy is a way to balance the *explorative* and *exploitative* approaches to the selection of an action in a reinforcement learning algorithm. 

**Exploitation** means to select the action that we believe has the best chance of success. This has the obvious advantage of letting the agent progress through the environment with a positive reward, and strengthen its knowledge of the possible ramifications of the action it currently believes to be the best. This, however, limits the agent's knowledge of the environment, as it prevents it from choosing different actions that may lead to better long-term results. 

**Exploration**, on the other hand, means to choose an action regardless of its immediate chance of success, with the goal of obtaining more information about the environment. 

The epsilon-greedy policy simply states that the agent at each step should **explore** the environment with probability $\epsilon$, by choosing a random action, and **exploit** its knowledge with probability $1-\epsilon$, by choosing the best known action.

<br>

## Deep Q learning

To recap, we defined a Q function (or state-action value function) as a function that specifies how good is an action $a$ in the state $s$. In classic Q learning, every possible value of this function is stored in a table, but in most environments that are not extremely simple and limited this may not be computationally feasible. A better approach would be to approximate the Q function with some parameters $\theta$ as $Q(s,a;\theta) \approx Q*(s,a)$. One of the best tools at our disposal to learn an arbitrary unknown function are neural networks, so we can use a deep network parametrised by $\theta$ to approximate the Q function, making it a *deep Q network (DQN)*.

We can therefore define the loss function of the network as the square of the error mentioned earlier:

$Loss = (y_i - Q(s,a;\theta))^2$,

where the target value $y_i$ is equal to the reward received plus the discounted future reward: 

$y_i = r + \gamma \text{ } max_{a'} Q(s', a'; \theta)$.

Having defined the loss, we can minimise it by updating the weights $\theta$ through gradient descent.

<br>

## DQN: the deep Q network

Since we want the algorithm to be independent from the game we choose to train it on, we can not use a game-specific state, so the network has to process the image of the current game frame and extract features from it on its own. The best choice in this case is clearly a convolutional network that takes as input a fixed-size image of the game and extracts features through a series of convolutions. The ouput of these convolutions is then fed as input to a multi-layer perceptron structure, that ends with an output layer with the same amount of nodes as the number of possible actions to take: the values in these output nodes (possibly passed through a *softmax* function) are the predicted Q values of each possible action in the current game state.

![](images/convnet.png)  
(image from S.Ravichandiran - *Hands-on Reinforcement Learning with Python* [2])

Lastly, a key component in most videogames is movement (for example, in the game of Super Mario, both Mario and its enemies move almost constantly), which cannot be extrapolated from a single image. This is why the state is composed by not only the current game screen, but also one or more screens of the past few frames, to extract the direction and speed of every moving component of the game.


<br> 

## Avoiding correlation: experience replay

Usually in videogames (and RL environments in general) the states and state-action values are strongly correlated between directly consecutive frames. This correlation vastly hurts the training process, and is known to cause severe overfitting in deep Q networks. A popular technique to reduce this problem is to introduce a *replay buffer* (or *replay memory*). 

As the agent plays the game, this buffer stores *transitions*, meaning tuples of state, action performed, reward received, and next state. When the agent wants to learn and update its weights, it randomly (or not, see the *prioritised experience replay* variants) samples memories from this buffer and uses those to execute the learning step. This buffer is usually implemented as a queue structure, so that when it reaches its maximum capacity, older experiences get deleted to make room for new ones.

<br>

## A variant: fixed Q value targets

In the traditional deep Q learning algorithm, the same network is used to both make predictions and set its own targets.
>This can lead to a situation analogous to a dog chasing its own tail. This feedback loop can make the network unstable: it can diverge, oscillate, freeze, and so on. [1]

This problem was solved in a paper published from DeepMind in 2013 [3], which used two identical DQNs instead of one, called *online* and *target* models. The online model performs the usual tasks of determining the agent's actions and learning at each step, while the target model is only used to generate targets. Its weights do not update at each step, but instead they get *synchronised* with the online weights at fixed intervals (in the original paper, every 10,000 steps).

With this improvement, the loss function now becomes:

$Loss = (r + \gamma \text{ } max_{a'} Q(s',a';\theta') - Q(s,a;\theta))^2$.

We notice that the function to estimate the best action in the next state (which is part of the target value) is now by parametrised by $\theta'$. These are the weights of the target network, which remain "frozen" a set number of steps behind those of the online networks.  
This improvement greatly stabilises the training process and improves the overall performance.


<br>
<br>

# The double deep Q algorithm

Having understood the basics of deep Q learning, we are only missing a small modification to obtain the final double deep Q network algorithm. In this section, this last variant is going to be explained, and then the whole algorithm presented in pseudocode and summarised.

<br>

## Avoiding overestimation of Q values: double DQN

Because of the $max$ operator in the DQN learning equation, the algorithm (to be precise, the target network) heavily tends to *overestimate* Q values.  

To briefly explain what that means:
the action values always contain some kind of random noise. In the  $max_{a'}Q(s',a';\theta')$ section of the equation, the target network has to choose the next best action, meaning the best action for the next state. If, for example, there were multiple equally good actions, they would all have the same *true* Q value. The target network would then always choose the one favoured by the random noise, and compute a Q value that is above the average (above the aforementioned *true* Q). Since this phenomemon is very recurrent, the network tends to systematically overestimate the Q values of the next best action.

> [... this is ] a bit like counting the height of the tallest random wave when measuring the depth of a pool. [1]
> 


This problem was solved in 2016 [4] by DeepMind, who performed a very simple modification on their previous DQN algorithm, increasing its performance and stabilising training: they proposed using the online model to *choose* the next best action, and the target model just to predict its Q value. This allows the random noise in Q values to sort of cancel out, as the next best action chosen is not always the one with the most amount of additive noise.

The new loss is therefore:

$Loss = (r + \gamma \text{ } Q(s', argmax_{a'} \text{ } Q(s',a';\theta); \text{ } \theta ') - Q(s,a;\theta))^2$.

Here, $\text{ }argmax_{a'} \text{ } Q(s',a';\theta)\text{ }$ selects the best next action using the online model (weights $\theta$), and $\text{ }Q(s', ...\text{ }; \theta ')\text{ }$ computes the value of said action using the target model (weights $\theta '$).

<br>

## Summary of the final algorithm

<br>

`
Algorithm: double-DQN

Initialise online network `$Q_\theta` and target network `Q_{\theta '}` with same weights
`   







<!-- # The implementation


# Results and conclusions -->



<!-- 


## Model the environment
- gym
- the environment class

## Classic Q learning vs Deep Q-Networks
- Q values
- Q table
- Q network

## The double network architecture


## Improvements

- Frequency of weight update (4 steps?)
- Frequency of copy (100 steps?)
- Initialization technique -->

<!-- ### Sources


- S. Ravichandiran - Hands-on Reinforcement Learning with Python -->
<!-- - A. Géron - Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems -->


<br>
<br>
<br>

# References

[1] A. Géron - Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow Concepts, Tools, and Techniques to Build Intelligent Systems

[2] S. Ravichandiran - Hands-on Reinforcement Learning with Python

[3] Volodymyr Mnih and Koray Kavukcuoglu and David Silver and Alex Graves and Ioannis Antonoglou and Daan Wierstra and Martin A. Riedmiller (2013). Playing Atari with Deep Reinforcement Learning 

[3] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1)