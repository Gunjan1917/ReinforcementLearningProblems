# Background

For this problem, we use the MountainCar environment from gym: https://gym.openai.com/envs/MountainCar-v0/. Starting from the bottom of a valley, an underpowered car has to gain enough momentum to reach the top of a
mountain. The objective is to minimize the number of time steps to reach the goal. There are three possible values of action a:
1. full throttle forward (+1)
2. full throttle reverse (-1)
3. zero throttle (0)

The continuous state space is defined by xt = (xt; x^t), where the bounded state variables xt lies within the range [-1.2, 0.6] and x^t lies within the range [-0.07; 0.07] are 
respectively the position and velocity of the car. The reward in this problem is -1 on all time steps. An episode terminates when the car moves past its goal position xt+1 >= 0.5 at the top of the mountain, or when the
episode length is greater than 200.

# Problem Statement
Implement Sarsa(lambda) with linear function approximation (e.g. tile-coding or RBFs) on the mountain car problem

# Results and Observations

![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/MountainCar/mountaincar.py)
