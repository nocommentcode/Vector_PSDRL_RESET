# def generate_trajectory():
from io import BytesIO
import os
import uuid
import gym
from matplotlib import pyplot as plt
import numpy as np
import torch

from ..agent.psdrl import PSDRL

from ..common.settings import TP_THRESHOLD
from ..common.utils import preprocess_image
from PIL import Image
import seaborn as sns

from matplotlib.patches import Rectangle


def make_frame(obs, reward, terminal):
    direction = obs[-1]
    obs = obs[:-1].reshape((7, 7, 3))
    obs = obs / obs.max()

    plt.close()
    plt.imshow(obs)
    plt.title(
        f"{str(round(direction, 2))}|{str(round(reward, 3))}|{str(round(terminal, 2))}"
    )

    # save figure to PIL image
    fig = plt.gcf()
    # fig.set_size_inches((12, 5))
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def compile_frames(true_frames, pred_frames):
    padding = 10
    # Calculate the total width and height for the output image
    total_width = sum(image.width for image in true_frames) + padding * (
        len(true_frames) - 1
    )
    max_height = max(image.height for image in true_frames)

    # Create a new blank image with the calculated dimensions
    new_image = Image.new("RGB", (total_width, max_height * 2 + padding))

    # Paste each image into the new image at the correct position
    x_offset = 0
    for true, pred in zip(true_frames, pred_frames):
        new_image.paste(pred, (x_offset, 0))
        new_image.paste(true, (x_offset, max_height + padding))
        x_offset += true.width + padding

    # new_image.show()
    return new_image


def simulate_trajectory(env: gym.Env, agent: PSDRL, logger, timestep):
    true_frames = []
    pred_frames = []

    obs = env.reset()
    agent.model.prev_state[:] = torch.zeros(agent.model.transition_network.gru_dim)
    for time in range(20):
        states, rewards, terminals, h = agent.model.predict(
            torch.from_numpy(obs).float().to(agent.model.device),
            agent.model.prev_state,
            batch=True,
        )

        action = agent.select_action(obs, time)

        pred_obs = states.detach().cpu().numpy()[action]
        pred_rew = rewards.detach().cpu().numpy()[action][0]
        pred_term = terminals.detach().cpu().numpy()[action][0]

        pred_frames.append(make_frame(pred_obs, pred_rew, pred_term))

        obs, reward, terminal, *_ = env.step(action)
        true_frames.append(make_frame(obs, reward, terminal))

        if terminal:
            break

    traj = compile_frames(true_frames, pred_frames)
    logger.data_manager.log_images("Trajectory", [traj], timestep)


FORWARD = 2
TURN_RIGHT = 1


def get_action_to_coords(x, y):
    curr_x = 1
    curr_y = 1
    actions = []
    while curr_x != x:
        actions.append(FORWARD)
        curr_x += 1
    actions.append(TURN_RIGHT)
    while curr_y != y:
        actions.append(FORWARD)
        curr_y += 1
    return actions


def move_to_coord_and_rotate(env, agent, x, y):
    obs = env.reset()
    agent.model.prev_state[:] = torch.zeros(agent.model.transition_network.gru_dim)

    actions = get_action_to_coords(x, y)
    # move agent to desired square
    for action in actions:
        obs = torch.from_numpy(obs).float().to(agent.device)
        states, rewards, terminals, h = agent.model.predict(
            obs, agent.model.prev_state, batch=True
        )
        agent.model.prev_state = h[action]
        obs, *_ = env.step(action)

    hiddens = torch.zeros((4, *agent.model.prev_state.shape)).to(agent.device)
    hiddens[0] = agent.model.prev_state
    observations = torch.zeros((4, *obs.shape)).to(agent.device)
    observations[0] = torch.from_numpy(obs).float().to(agent.device)

    # rotate 3 times to get remaining directions
    for i in range(1, 4):
        obs = torch.from_numpy(obs).float().to(agent.device)
        states, rewards, terminals, h = agent.model.predict(
            obs, agent.model.prev_state, batch=True
        )
        agent.model.prev_state = h[TURN_RIGHT]
        obs, *_ = env.step(TURN_RIGHT)
        hiddens[i] = h[TURN_RIGHT]
        observations[i] = torch.from_numpy(obs).float().to(agent.device)
        print(env.unwrapped.agent_dir)

    return hiddens, observations


def get_value_for_obs(obs, hiddens, agent: PSDRL):
    v = agent.value_network.predict(torch.cat((obs, hiddens), dim=1))
    return v


def get_obs_for(env, x, y, dir):
    env.unwrapped.agent_pos = (x, y)
    env.unwrapped.agent_dir = dir
    obs = env.gen_obs()

    # image = obs["image"] / obs["image"].max()
    # plt.imshow(image)
    # plt.show()

    image_obs = obs["image"].flatten()
    direction = np.array([obs["direction"]])
    final_obs = np.concatenate((image_obs, direction))
    return final_obs


def plot_value_heatmap(env: gym.Env, agent: PSDRL, logger, timestep):
    values = np.zeros((3 * 2, 3 * 2))
    env.reset()
    plt.close()
    for x in range(1, 4):
        for y in range(1, 4):
            hidden, obs = move_to_coord_and_rotate(env, agent, x, y)
            v = np.tan(get_value_for_obs(obs, hidden, agent).detach().cpu().numpy())
            x_c = (x - 1) * 2
            y_c = (y - 1) * 2
            values[x_c, y_c] = v[0]
            values[x_c + 1, y_c] = v[1]
            values[x_c, y_c + 1] = v[2]
            values[x_c + 1, y_c + 1] = v[3]

    ax = sns.heatmap(values, linewidth=0.5, xticklabels=[], yticklabels=[])
    ax.vlines([x * 2 for x in range(5)], 0, 3 * 2, colors="black")
    ax.hlines([y * 2 for y in range(5)], 0, 3 * 2, colors="black")

    fig = plt.gcf()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)

    logger.data_manager.log_images("Values", [image], timestep)
