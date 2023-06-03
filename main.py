import gym
import numpy as np
import matplotlib.pyplot as plt
from fuzzy import InputStateVariable
from fuzzy import Trapeziums
from fuzzy import Build
from fuzzy import FQLModel


GLOBAL_SEED = 1
LOCAL_SEED = 42
np.random.seed(GLOBAL_SEED)


# Define membership functions for MountainCar problems
def get_fis_env():
    p = InputStateVariable(Trapeziums(-1.2, -1.2, -1.2, -0.775),
                           Trapeziums(-1.2, -0.775, -0.775, -0.35),
                           Trapeziums(-0.775, -0.35, -0.35, 0.075),
                           Trapeziums(-0.35, 0.075, 0.075, 0.5),
                           Trapeziums(0.075, 0.5, 0.5, 0.5))
    v = InputStateVariable(Trapeziums(-0.07, -0.07, -0.07, -0.035),
                           Trapeziums(-0.07, -0.035, -0.035, 0.),
                           Trapeziums(-0.035, 0., 0., 0.035),
                           Trapeziums(0., 0.035, 0.035, 0.07),
                           Trapeziums(0.035, 0.035, 0.035, 0.07))
    env = gym.make("MountainCar-v0")
    env = env.unwrapped
    fis = Build(p, v)
    return env, fis


def train_env(model=None, max_eps=500):
    env, fis = get_fis_env()
    print('Observation shape:', env.observation_space.shape)
    print('Action length:', env.action_space.n)
    action_set_length = env.action_space.n

    # Create Model
    if model is None:
        model = FQLModel(gamma=0.99,
                         alpha=0.1,
                         ee_rate=1.,
                         action_set_length=action_set_length,
                         fis=fis)
    rewards = []
    r = 0
    done = True
    iteration = 0
    best_mean_rewards = []
    while iteration < max_eps:
        if done:
            state_value, _ = env.reset()
            action = model.get_initial_action(state_value)
            rewards.append(r)
            mean_reward = np.mean(rewards[-50:])
            best_mean_rewards.append(mean_reward)
            if len(best_mean_rewards) > 2:
                epsilon = best_mean_rewards[-1] - best_mean_rewards[-2]
            else:
                epsilon = 0
            print('EPS=', iteration, ' reward=', r,
                  ' epsilon=', model.ee_rate, ' best mean eps=', epsilon)
            iteration += 1
            r = 0
            # Epsilon decay
            model.ee_rate -= model.ee_rate * 0.01
            if model.ee_rate <= 0.2:
                model.ee_rate = 0.2
        state_value, reward, done, done_1, _ = env.step(action)
        # Change the rewards to -1
        if reward == 0:
            reward = -1
        action = model.run(state_value, reward)
        r += reward
        # Reach to 2000 steps --> done
        if r <= -2000:
            done = True
    print(model.q_table)
    print('Epsilon=', model.ee_rate)
    plt.figure(figsize=(14, 5))
    plt.plot(best_mean_rewards[1:])
    plt.ylabel('Rewards')
    plt.show()
    return model


if __name__ == '__main__':
    model = train_env(max_eps=500)
