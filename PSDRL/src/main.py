import os
import argparse
import wandb
import numpy as np
import torch
from ruamel.yaml import YAML
import gym


from PSDRL.common.data_manager import DataManager
from PSDRL.common.utils import init_env, load, preprocess_image
from PSDRL.common.logger import Logger
from PSDRL.agent import Agent

from PSDRL.common.plot_minigrid import (
    TrajectoryImages,
    TransitionModelImage,
    ValueHeatmap,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_test_episode(env: gym.Env, agent: Agent, time_limit: int):
    current_observation = env.reset()
    episode_step = 0
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(current_observation, episode_step)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        current_observation = observation
        episode_step += 1
        done = done or episode_step == time_limit
    return episode_reward


def early_stop(dataset) -> bool:
    n_episodes = 20
    last_n_episodes = dataset.episodes[-n_episodes:]

    episode_returns = [ep["cum_rew"] for ep in last_n_episodes]
    av_return = sum(episode_returns) / n_episodes

    return av_return >= 0.98


def run_experiment(
    env: gym.Env,
    agent: Agent,
    logger: Logger,
    test_env: gym.Env,
    steps: int,
    test: int,
    test_freq: int,
    time_limit: int,
    save: bool,
    save_freq: int,
):
    ep = 0
    experiment_step = 0

    while experiment_step < steps:
        episode_step = 0
        episode_reward = 0

        current_observation = env.reset()
        done = False
        while not done:

            if test and experiment_step % test_freq == 0:
                test_reward = run_test_episode(test_env, agent, time_limit)
                logger.log_episode(
                    experiment_step, train_reward=np.nan, test_reward=test_reward
                )
                print(
                    f"Episode {ep}, Timestep {experiment_step}, Test Reward {test_reward}"
                )

            action = agent.select_action(current_observation, episode_step)
            observation, reward, done, _ = env.step(action)
            done = done or episode_step == time_limit

            agent.update(
                current_observation,
                action,
                reward,
                observation,
                done,
                ep,
                experiment_step,
            )

            episode_reward += reward
            current_observation = observation
            episode_step += 1
            experiment_step += 1

            if ep and save and experiment_step % save_freq == 0:
                logger.data_manager.save(agent, experiment_step)
        print(
            f"Episode {ep}, Timestep {experiment_step}, Train Reward {episode_reward}"
        )

        if ep % 25 == 0:
            image_generators = [
                TransitionModelImage(env, agent),
                TrajectoryImages(env, agent),
                ValueHeatmap(env, agent),
            ]
            for generator in image_generators:
                generator.generate_and_log(logger, experiment_step)

        ep += 1
        logger.log_episode(
            experiment_step, train_reward=episode_reward, test_reward=np.nan
        )

        if early_stop(agent.dataset):
            break


def main(config: dict):
    data_manager = DataManager(config)
    logger = Logger(data_manager)
    exp_config = config["experiment"]

    env, actions, test_env = init_env(
        exp_config["suite"], exp_config["env"], exp_config["test"]
    )

    agent = Agent(
        config,
        actions,
        logger,
        (
            config["representation"]["embed_dim"]
            if config["visual"]
            else env.observation_space.shape[0]
        ),
        config["experiment"]["seed"],
    )
    if config["load"]:
        load(agent, config["load_dir"])

    run_experiment(
        env,
        agent,
        logger,
        test_env,
        exp_config["steps"],
        exp_config["test"],
        exp_config["test_freq"],
        exp_config["time_limit"],
        config["save"],
        config["save_freq"],
    )


def run_on_seed(config):
    with open(args.config, "r") as f:
        yaml = YAML(typ="rt")
        config = yaml.load(f)

        config["experiment"]["env"] = args.env
        config["experiment"]["seed"] = args.seed
        config["experiment"]["suite"] = args.suite
        config["experiment"]["name"] = args.experiment_name
        if config["experiment"]["suite"] == "bsuite":
            config["replay"]["sequence_length"] = int(args.env)

    main(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/config_psdrl_vector.yaml"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="3",
        help="Currently if you put an integer it makes DeepSea with the size of that integer.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--suite", type=str, default="bsuite")
    parser.add_argument("--experiment_name", type=str, default="")

    args = parser.parse_args()
    envs = args.env
    for seed in args.seed:
        args.seed = seed
        run_on_seed(args)
        wandb.finish()
