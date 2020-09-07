import gym
import numpy as np
import matplotlib.pyplot as plt


def nstep_sarsa(env, n=8, alpha=0.8, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # episode_reward = []
    # total_reward = 0

    for _ in range(num_ep):
        # self.reset()
        t = 0
        T = np.inf
        state = env.reset()
        action = np.random.choice(env.action_space.n)
        '''
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.argmax(Q[state, :])
        '''
        actions = [action]
        states = [state]
        rewards = [0]
        while True:
            if t < T:
                state_next, reward, done, info = env.step(action)
                states.append(state_next)
                rewards.append(reward)

                if done:
                    T = t + 1
                else:
                    action_next = np.random.choice(env.action_space.n)
                    ''' 
                    if np.random.uniform(0, 1) < epsilon:
                        action_next = np.random.randint(env.action_space.n)
                    else:
                        action_next = np.argmax(Q[state, :])
                    '''
                    actions.append(action_next)  # next action
            # state tau being updated
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + n + 1, T + 1)):
                    G += np.power(gamma, i - tau - 1) * rewards[i]
                if tau + n < T:
                    state_action = (states[tau + n], actions[tau + n])
                    G += np.power(gamma, n) * Q[state_action[0]][state_action[1]]
                # update Q values
                state_action = (states[tau], actions[tau])
                Q[state_action[0]][state_action[1]] += alpha * (G - Q[state_action[0]][state_action[1]])

            if tau == T - 1:
                break

            t += 1
            state = state_next
            action = action_next

        # episode_reward.append(np.sum(rewards))
    # total_reward = np.sum(episode_reward)
    return Q


env = gym.make('FrozenLake-v0', map_name="8x8")
# TODO: run multiple times, evaluate the performance for different n and alpha
Q = nstep_sarsa(env)

# Optimal real values obtained from exercise 3 using value iteration
actual_state_values = [2.23747107e-04, 3.91558432e-04, 7.35847221e-04, 1.38030491e-03,
                       2.47307413e-03, 3.86416505e-03, 5.54943466e-03, 6.54514275e-03,
                       2.05340185e-04, 3.40939392e-04, 6.43275809e-04, 1.32276519e-03,
                       2.93678953e-03, 5.07701994e-03, 8.71580325e-03, 1.14540005e-02,
                       1.63283129e-04, 2.43689401e-04, 3.53672742e-04, 0.00000000e+00,
                       3.46286738e-03, 6.45885716e-03, 1.56808277e-02, 2.27826987e-02,
                       1.39093160e-04, 2.19224027e-04, 4.39308427e-04, 1.07451057e-03,
                       3.59010661e-03, 0.00000000e+00, 2.73046024e-02, 4.69715943e-02,
                       9.56825699e-05, 1.24034810e-04, 1.50224725e-04, 0.00000000e+00,
                       8.92552232e-03, 1.98525064e-02, 3.97398375e-02, 1.01867282e-01,
                       4.45207127e-05, 0.00000000e+00, 0.00000000e+00, 2.89829863e-03,
                       1.00280961e-02, 2.57815395e-02, 0.00000000e+00, 2.33163432e-01,
                       2.67500328e-05, 0.00000000e+00, 2.53665989e-04, 8.40523887e-04,
                       0.00000000e+00, 6.68001708e-02, 0.00000000e+00, 5.39332157e-01,
                       2.90422239e-05, 5.08239232e-05, 1.10723595e-04, 0.00000000e+00,
                       8.17160368e-02, 2.24719101e-01, 5.36261491e-01, 0.00000000e+00]
actual_state_values = np.array(actual_state_values)

estimate_state_values = [np.mean(list(v)) for v in Q]
rms_error = np.mean([er ** 2 for er in actual_state_values - np.array(estimate_state_values)])

# Plotting
x1 = [0, 0.2, 0.4, 0.6, 0.8]
y1 = [0.011, 0.007, 0.006, 0.005, 0.007]  # Values obtained after running multiple times of n=1

x2 = [0, 0.2, 0.4, 0.6, 1]
y2 = [0.011, 0.0068, 0.0047, 0.004, 0.011]  # Values obtained after running multiple times of n=8

x3 = [0, 0.2, 0.4, 0.6, 0.8, 1]
y3 = [0.011, 0.0059, 0.0015, 0.0038, 0.0066, 0.0078]  # Values obtained after running multiple times of n=32

x4 = [0, 0.2, 0.4, 0.6, 0.8, 1]
y4 = [0.011, 0.0047, 0.0019, 0.0034, 0.0042, 0.011]  # Values obtained after running multiple times of n=64

x5 = [0, 0.2, 0.4, 0.6]
y5 = [0.011, 0.0077, 0.0027, 0.011]  # Values obtained after running multiple times of n=128

plt.plot(x1, y1)
plt.plot(x2, y2)
plt.plot(x3, y3)
plt.plot(x4, y4)
plt.plot(x5, y5)

plt.xlabel('Learning Rate')
plt.ylabel('RMS error for 1e4 episodes')
plt.legend(["n=1", "n=8", "n=32", "n=64", "n=128"])
plt.show()
