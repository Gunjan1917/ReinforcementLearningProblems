# Background
For this problem, we will use the Cart-Pole environment from gym: https://gym.openai.com/envs/CartPole-v1/ The task is to apply forces to a cart moving a long a track in order to keep the pole balanced. If the 
pole falls apart a given angle or an episode length of 500 is reached, the episode terminates. The state consists of 4 continous variables (position and velocity of cart and pole). There are 2 actions corresponding to left and right.
# Problem Statement
Implement the REINFORCE algorithm on the Cart-Pole example using the softmax action selection strategy. Track the mean of the 100 latest episode lengths. 
Tune the parameters and try to achieve a mean >= 495. How many episodes are needed? Plot the mean over the episode count

# Results and Observations
To achieve the required mean, the number of episodes can vary. For this particular run, the number of episodes required to achieve a mean of 495 was 158.

![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/master/REINFORCE_cartpole/mean.PNG)
![alt text](https://github.com/gunjan1917/ReinforcementLearningProblems/blob/master/REINFORCE_cartpole/ex8_output.PNG)
