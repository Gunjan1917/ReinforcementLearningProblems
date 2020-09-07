# Background

The Frozen Lake environment is a 4×4 grid which contain four possible areas  — Safe (S), Frozen (F), Hole (H) and Goal (G). The agent moves around the grid until it reaches the
goal or the hole. If it falls into the hole, it has to start from the beginning and is rewarded the value 0. The process continues until it learns from every mistake and reaches 
the goal eventually, and is rewarded the value 1. Here is visual description of the Frozen Lake grid (4×4):

![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/Gunjan1917-patch-1/FrozenLake/frozenlake_grid.PNG)

The agent in the environment has four possible moves — Up, Down, Left and Right. This environment will allow the agent to move accordingly. 
There could be a random action happening once every a few episodes — let’s say the agent is slipping in different directions because it is hard to walk on a frozen surface. 
Considering this situation, we need to allow some random movement at first, but eventually try to reduce its probability. This way we can correct the error caused by minimising 
the loss.

This grid has 16 possible blocks where the agent will be at a given time. At the current state, the agent will have four possibilities of deciding the next state. 
From the current state, the agent can move in four directions as mentioned which gives us a total of 16×4 possibilities. These are our weights and we will update them
according to the movement of our agent. Once all the resources are obtained, a suitable RL algorithm can be implemented to train the agent for the Frozen-Lake situation.

# Problem Statement
Implement Q-learning and SARSA algorithm on the frozen-lake problem and obtain and plot the optimal state-value function, action-value function, and policy for
FrozenLake. What can be said about on-line performance during training in comparison to the performance of the optimal policy? Explore how the results for both the algorithms change if you switch to the non-slippery version (i.e. deterministic environment).

Implement n-step Sarsa and evaluate it on the 8x8 Frozen Lake environment. Evaluate the performance for different choices of n and alpha. Visualize your results
(plot the performance over alpha for different choices of n.

# Results and Observations

![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/Gunjan1917-patch-1/FrozenLake/sarsa.PNG)
![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/Gunjan1917-patch-1/FrozenLake/q_learning.PNG)

The difference between these two algorithms is that SARSA chooses an action following the same current policy and updates its Q-values whereas Q-learning chooses the greedy action, that is, the action that gives the maximum Q-value for the state, that is, it follows an optimal policy. Under some common conditions, they both converge to the real value function, but at different rates. Q-Learning tends to converge a little slower, but has the capability to continue learning while changing policies. Also, Q-Learning is not guaranteed to converge when combined with linear approximation.
If the goal is to train an optimal agent in simulation, or in a low-cost and fast-iterating environment, then Q-learning is a good choice, due to the first point (learning optimal policy directly). If the agent learns online, and you care about rewards gained while it is learning, then SARSA would a better choice.

# Non-Slippery version (Deterministic Environment)
![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/Gunjan1917-patch-1/FrozenLake/q_learning_nonslippery.PNG)
![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/Gunjan1917-patch-1/FrozenLake/sarsa_nonslippery.PNG)

# n-step SARSA on Frozen Lake
![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/Gunjan1917-patch-1/FrozenLake/nstep_sarsa_results.PNG)
