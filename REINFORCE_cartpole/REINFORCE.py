import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    x = np.dot(state, theta)
    x = np.reshape(x, (2,))
    a = max(x)
    return np.exp(x - a) / np.sum(np.exp(x - a))
    # return [0.5, 0.5]  # both actions with 0.5 probability => random


def generate_episode(env, theta, display=False):
    """ enerates one episode and returns the list of states, the list of rewards and the list of actions
    of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    step_size = 1.0
    targetMean = 495
    episode_reward = None
    rewards1 = []
    episode_rewards = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)

        # print("episode: " + str(e) + " length: " + str(len(states)))

        # TODO: keep track of previous 100 episode lengths and compute mean
        reward = np.sum(rewards)
        rewards1.append(reward)
        episode_reward = np.mean(rewards1[-100:])
        episode_rewards.append(episode_reward)

        if not e % 100:
            print("Episode: {}. Episode Mean reward: {}".format(e + 1, episode_reward))

        if e >= 99 and episode_reward >= targetMean:
            print("Episode: {}. Episode Mean reward: {}".format(e + 1, episode_reward))
            print("Ran {} episodes. Reached the Target after {} episodes.".format(e + 1, e - 100 + 1))
            break

        # TODO: implement the reinforce algorithm to improve the policy weights
        total = len(states)
        for i in range(len(states)):
            estimated_return = total - i - 1
            grad = np.zeros((4, 2))
            a = actions[i]
            s = states[i]
            product = np.dot(s, theta)
            c = np.sum(np.exp(product))

            if a == 0:  # origina a == 0
                grad[:, 0] = s * (1 - np.exp(product[0]) / c)
                grad[:, 1] = -s * np.exp(product[1]) / c
            else:
                grad[:, 0] = -s * np.exp(product[0]) / c
                grad[:, 1] = s * (1 - np.exp(product[1]) / c)

            theta += step_size * grad * estimated_return

    plt.plot(episode_rewards)
    plt.xlabel('No of Episodes')
    plt.ylabel('Mean Rewards')
    plt.show()


def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
