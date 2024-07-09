import os
import argparse
import wandb
import numpy as np
import torch
from ruamel.yaml import YAML
import gym

from PSDRL.common.data_manager import DataManager
from PSDRL.common.utils import init_env, load
from PSDRL.common.logger import Logger
from PSDRL.agent import Agent
from PSDRL.common.plot_deep_sea import (
    log_deep_shallow_expl,
    log_shallow_effect,
    log_trajectories,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_test_episode(env: gym.Env, agent: Agent, time_limit: int):
    current_observation = env.reset()
    episode_step = 0
    episode_reward = 0
    done = False
    while not done:
        action = agent.exploite(current_observation, episode_step)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        current_observation = observation
        episode_step += 1
        done = done or episode_step == time_limit
    return episode_reward


# def log_correct_path(env: gym.Env, agent):
#     def get_right_action():
#         row = env._row
#         col = env._column
#         return env._action_mapping[row, col]

#     agent.model.prev_state[:] = torch.zeros(agent.model.transition_network.gru_dim)
#     obs = env.reset()
#     for time in range(env._size):
#         right_a = get_right_action()
#         print(right_a)

#         obs = torch.from_numpy(obs).float().to(agent.device)
#         if is_image:
#             obs = agent.model.autoencoder.embed(obs)
#         states, rewards, terminals, h = agent.model.predict(
#             obs, agent.model.prev_state, batch=True
#         )

#         obs, reward, done, _ = env.step(right_a)

#         pred_state = states[right_a]
#         pred_state = (
#             pred_state.detach().cpu().numpy().reshape((env._size, env._size)).round(0)
#         )
#         pred_rew = rewards[right_a]
#         pred_rew = pred_rew.detach().cpu().numpy().round(3)[0]

#         pred_terminals = terminals[right_a]
#         pred_terminals = pred_terminals.detach().cpu().numpy().round(3)[0]

#         agent.model.prev_state = h[right_a]

#         print(f"Time {time}:")
#         print(pred_rew)
#         print(f"{reward},{done} {' '*env._size}{str(pred_rew)}, {str(pred_terminals)}")
#         for act, pred in zip(obs, pred_state):
#             print(act, pred)


def early_stop(dataset) -> bool:
    n_episodes = 500
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

            # if done and reward >= 0.99:
            #     log_shallow_effect(env, agent, logger, experiment_step)

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

        # log_correct_path(env, agent)
        # log_trajectories(env, agent, 10, logger, experiment_step)

        ep += 1
        logger.log_episode(
            experiment_step, train_reward=episode_reward, test_reward=np.nan
        )

        # if ep % 5 == 0:
        #     log_deep_shallow_expl(env, agent, logger, experiment_step)

        if early_stop(agent.dataset):
            break


def main(config: dict):
    data_manager = DataManager(config)
    logger = Logger(data_manager)
    exp_config = config["experiment"]

    env, actions, test_env = init_env(
        exp_config["suite"], exp_config["env"], exp_config["test"], exp_config
    )

    agent = Agent(
        config,
        actions,
        logger,
        (
            config["representation"]["embed_dim"]
            if config["visual"]
            else np.prod(env.observation_space.shape)
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
        config["experiment"]["name"] = args.experiment_name
        config["experiment"]["suite"] = args.suite
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
    parser.add_argument("--suite", type=str, default="bsuite")
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--experiment_name", type=str, default="")

    args = parser.parse_args()
    envs = args.env
    for seed in args.seed:
        args.seed = seed
        run_on_seed(args)
        wandb.finish()
