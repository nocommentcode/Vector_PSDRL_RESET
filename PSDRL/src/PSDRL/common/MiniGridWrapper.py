import gymnasium as gym
import numpy as np
from gymnasium.core import ObservationWrapper


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
