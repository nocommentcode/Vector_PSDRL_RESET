import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper, Wrapper


class DeepSeaWrapper(Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            0, 1, (np.prod(self.env.observation_space.shape),)
        )

    def reset(self):
        obs = self.env.reset()
        return self.flatten_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.flatten_obs(obs), reward, done, info

    def flatten_obs(self, obs):
        return obs.flatten()


class MiniGridWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        image_space = env.observation_space["image"]

        self.observation_space = gym.spaces.Box(
            0,
            255,
            (np.prod(env.observation_space["image"].shape) + 1,),
        )

    def observation(self, observation):
        obs = observation["image"].flatten()
        direction = np.array([observation["direction"]])
        return np.concatenate((obs, direction))

    def step(self, action):
        observation, reward, done, trucated, _ = super().step(action)
        return observation, reward, done or trucated, _

    def reset(self, *, seed=None, options=None):
        obs, _ = super().reset(seed=seed, options=options)
        return obs
