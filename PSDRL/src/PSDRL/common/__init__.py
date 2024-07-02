from gymnasium.envs.registration import register

register(
    id="MiniGrid-Empty-5x5-v1",
    entry_point="minigrid.envs:EmptyEnv",
    kwargs={"size": 5, "max_steps": 300},
)


register(
    id="MiniGrid-Empty-8x8-v1",
    entry_point="minigrid.envs:EmptyEnv",
    kwargs={"size": 8, "max_steps": 700},
)
