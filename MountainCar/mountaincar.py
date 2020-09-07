import gym
import matplotlib.pyplot as plt
import numpy as np
from math import floor

env = gym.make("MountainCar-v0")
env._max_episode_steps = 3000  # Increase upper time limit so we can plot full behaviour.
np.random.seed(6)  # Make plots reproducible
MAX_STEPS = 200


class IHT:
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates


def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles


class QEstimator():
    def __init__(self, step_size, num_tilings=8, max_size=4096, tiling_dim=None, trace=False):

        self.trace = trace
        self.max_size = max_size
        self.num_tilings = num_tilings
        self.tiling_dim = tiling_dim or num_tilings
        self.alpha = step_size / num_tilings
        self.iht = IHT(max_size)
        self.weights = np.zeros(max_size)
        if self.trace:
            self.z = np.zeros(max_size)
        self.position_scale = self.tiling_dim / (env.observation_space.high[0]
                                                 - env.observation_space.low[0])
        self.velocity_scale = self.tiling_dim / (env.observation_space.high[1]
                                                 - env.observation_space.low[1])

    def featurize_state_action(self, state, action):
        featurized = tiles(self.iht, self.num_tilings,
                           [self.position_scale * state[0],
                            self.velocity_scale * state[1]],
                           [action])
        return featurized

    def predict(self, s, a=None):
        if a is None:
            features = [self.featurize_state_action(s, i) for
                        i in range(env.action_space.n)]
        else:
            features = [self.featurize_state_action(s, a)]

        return [np.sum(self.weights[f]) for f in features]

    def update(self, s, a, target):
        features = self.featurize_state_action(s, a)
        estimation = np.sum(self.weights[features])  # Linear FA
        delta = (target - estimation)

        if self.trace:
            # self.z[features] += 1  # Accumulating trace
            self.z[features] = 1  # Replacing trace
            self.weights += self.alpha * delta * self.z
        else:
            self.weights[features] += self.alpha * delta

    def reset(self, z_only=False):
        if z_only:
            assert self.trace, 'q-value estimator has no z to reset.'
            self.z = np.zeros(self.max_size)
        else:
            if self.trace:
                self.z = np.zeros(self.max_size)
            self.weights = np.zeros(self.max_size)


def make_epsilon_greedy_policy(estimator, epsilon, num_actions):
    def policy_fn(observation):
        action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(observation)
        best_action_idx = np.argmax(q_values)
        action_probs[best_action_idx] += (1.0 - epsilon)
        return action_probs
    return policy_fn


def sarsa_lambda(lmbda, env, estimator, gamma=1.0, epsilon=0):

    estimator.reset(z_only=True)

    policy = make_epsilon_greedy_policy(
        estimator, epsilon, env.action_space.n)

    state = env.reset()
    action_probs = policy(state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

    ret = 0
    for t in range(MAX_STEPS):
        next_state, reward, done, _ = env.step(action)
        ret += reward

        if done:
            target = reward
            estimator.update(state, action, target)
            break

        else:
            next_action_probs = policy(next_state)
            next_action = np.random.choice(
                np.arange(len(next_action_probs)), p=next_action_probs)

            q_new = estimator.predict(
                next_state, next_action)[0]
            target = reward + gamma * q_new

            estimator.update(state, action, target)
            estimator.z *= gamma * lmbda

        state = next_state
        action = next_action

    return t, ret


def get_state(env):
    segmentation_factor = 100
    pos_segment = (env.high[0] - env.low[0]) / segmentation_factor
    vel_segment = (env.high[1] - env.low[1]) / segmentation_factor
    state = env.state
    coarse_state = np.zeros(2*segmentation_factor)

    coarse_state[int((state[0] - env.low[0]) / pos_segment)] = 1
    coarse_state[int((state[1] - env.low[1]) / vel_segment) + segmentation_factor] = 1

    return coarse_state


def value_approx(env, state, action, weights):
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1

    s_a = np.zeros(len(weights))

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            s_a[w_i] = s_i*a_i
            w_i = w_i + 1

    return np.dot(s_a, weights)


def value_approx_grad(env, state, action, weights):
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1
    gradient = np.zeros(len(weights))

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            gradient[w_i] = s_i*a_i
            w_i = w_i + 1

    return gradient


def select_action_e_greedy(env, Q_values, epsilon=0.2):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    else:
        return np.argmax(Q_values)


def learning_episode(env, weights, alpha, gamma):
    total_reward = 0
    s = get_state(env)
    Q_values = [value_approx(env, s, action, weights) for action in range(env.action_space.n)]
    a = select_action_e_greedy(env, Q_values)
    Q_prev = Q_values[a]

    for t in range(MAX_STEPS):

        _, reward, done, _ = env.step(a)

        total_reward = total_reward + reward

        if done:
            weights = weights + alpha*(reward - Q_prev) * value_approx_grad(env, s, a, weights)
            break

        s_new = get_state(env)
        Q_values_s_new = [value_approx(env, s_new, action, weights) for action in range(env.action_space.n)]
        a_next = select_action_e_greedy(env, Q_values_s_new)
        Q_next_a = Q_values_s_new[a_next]

        #updata rule
        weights = weights + alpha*(reward + gamma*Q_next_a - Q_prev) * value_approx_grad(env, s, a, weights)

        s = s_new
        a = a_next
        Q_values = Q_values_s_new
        Q_prev = Q_next_a

    return t, total_reward


def main():
    runs = 10
    step_size = 0.5  # Fraction of the way we want to move towards target
    lmbda = 0.92  # Level of bootstrapping (set to intermediate value)
    gamma = 1  # discount_factor = 1.0
    alpha = 0.1
    num_episodes = 500
    steps = np.zeros(num_episodes)
    returns = np.zeros(num_episodes)
    total_steps = np.zeros(num_episodes)
    total_returns = np.zeros(num_episodes)

    for run in range(runs):
        estimator_lambda = QEstimator(step_size=step_size, trace=True)
        print("Runs: " + str(run))
        for i in range(num_episodes):
            episode_steps, episode_reward = sarsa_lambda(lmbda=lmbda, env=env, estimator=estimator_lambda)
            steps[i] += episode_steps
            returns[i] += episode_reward
            print("\rEpisode {}/{} Return {}".format(
                i + 1, num_episodes, episode_reward), end="")
        print("\n")

    total_steps = steps / runs
    total_returns = returns / runs

    plt.xlabel('Number of Episodes')
    plt.ylabel('Avg number of steps per episode')
    plt.plot(np.arange(num_episodes), total_steps)
    plt.show()

    steps = np.zeros(num_episodes)
    returns = np.zeros(num_episodes)
    total_steps = np.zeros(num_episodes)
    total_returns = np.zeros(num_episodes)

    env.reset()
    env_dim = len(get_state(env))
    weights = np.zeros(env_dim * env.action_space.n)

    for run in range(runs):
        print("Runs: " + str(run))
        for i in range(num_episodes):
            env.reset()
            episode_steps, episode_reward = learning_episode(env, weights, alpha, gamma)
            steps[i] += episode_steps
            returns[i] += episode_reward
            print("\rEpisode {}/{} Return {}".format(
                i + 1, num_episodes, episode_reward), end="")
        print("\n")

    total_steps = steps / runs
    total_returns = returns / runs

    plt.xlabel('Number of Episodes')
    plt.ylabel('Avg number of steps per episode')
    plt.plot(np.arange(num_episodes), total_steps)
    plt.show()

    env.close()


if __name__ == "__main__":
    main()