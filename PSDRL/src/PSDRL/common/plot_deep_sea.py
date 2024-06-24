from io import BytesIO
import os
import uuid
import gym
from matplotlib import pyplot as plt
import numpy as np
import torch

from ..common.settings import TP_THRESHOLD
from ..common.utils import preprocess_image
from PIL import Image
import seaborn as sns

from matplotlib.patches import Rectangle


def plot_origin_state(state, ax):
    ax = sns.heatmap(
        state, linewidth=0.5, ax=ax, xticklabels=[], yticklabels=[], cbar=False
    )


def reset_env_to(env, prev_actions):
    obs, reward, done = env.reset(), 0, False
    for a in prev_actions:
        obs, reward, done, _ = env.step(a)

    return obs, reward, done


def plot_state(
    axis,
    actual_state,
    predicted_state,
    pred_rew,
    act_rew,
    pred_term,
    act_term,
    pred_value,
):
    ax = sns.heatmap(
        predicted_state, linewidth=0.5, ax=axis, xticklabels=[], yticklabels=[]
    )
    try:
        # plot actual position
        x_pos, y_pos = np.where(actual_state == 1)
        axis.add_patch(
            Rectangle(
                (int(y_pos), int(x_pos)), 1, 1, fill=False, edgecolor="blue", lw=3
            )
        )
    except:
        # in absorbtion state, no ned to plot actual position
        pass
    reward = f"R: {str(round(pred_rew, 3))}({str(round(act_rew, 3))})"
    terminal = f"T: {str(round(pred_term, 3))}({str(round(act_term, 3))})"
    value = f"V: {str(round(pred_value, 3))}"
    title = f"{reward} | {terminal} | {value}"
    axis.set_title(title)


def simulate_action_and_plot(
    env: gym.Env,
    prev_actions,
    action,
    pred_state,
    pred_rew,
    pred_term,
    pred_value,
    axis,
):
    obs, reward, done = reset_env_to(env, [*prev_actions, action])

    pred_state = pred_state.detach().cpu().numpy().reshape((env._size, env._size))
    pred_rew = pred_rew.detach().cpu().numpy()[0]
    pred_term = pred_term.detach().cpu().numpy()[0]
    pred_value = pred_value[0]

    plot_state(
        axis,
        obs,
        pred_state,
        pred_rew,
        reward,
        pred_term,
        1 if done else 0,
        pred_value=pred_value,
    )


def plot_frame(
    env,
    step,
    prev_actions,
    pred_states,
    pred_rews,
    pred_terms,
    pred_values,
    chosen_action,
):
    plt.close()

    def get_subplot_idx(a):
        row = env._row
        col = env._column
        return a == env._action_mapping[row, col]

    # get current obs
    current_obs, *_ = reset_env_to(env, prev_actions)

    # plot starting obs on top
    ax = plt.subplot(2, 1, 1)
    going_right = get_subplot_idx(chosen_action)
    ax.set_title(f"Time {step}, going {'right' if going_right else 'left'}")
    plot_origin_state(current_obs, ax)

    # simulate each action and plot on bottom
    left_action = 0 if get_subplot_idx(1) else 1
    right_action = 1 if get_subplot_idx(1) else 0
    for i, a in enumerate([left_action, right_action]):
        ax = plt.subplot(2, 2, 3 + i)
        simulate_action_and_plot(
            env,
            prev_actions,
            a,
            pred_states[a],
            pred_rews[a],
            pred_terms[a],
            pred_values[a],
            ax,
        )

    # save figure to PIL image
    fig = plt.gcf()
    fig.set_size_inches((12, 5))
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def compile_frames(frames):
    padding = 10
    # Calculate the total width and height for the output image
    total_width = sum(image.width for image in frames) + padding * (len(frames) - 1)
    max_height = max(image.height for image in frames)

    # Create a new blank image with the calculated dimensions
    new_image = Image.new("RGB", (total_width, max_height))

    # Paste each image into the new image at the correct position
    x_offset = 0
    for image in frames:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width + padding

    # new_image.show()
    return new_image


def simulate_trajectory(env: gym.Env, agent):

    def process_obs(obs):
        obs, is_image = preprocess_image(obs)
        obs = torch.from_numpy(obs).float().to(agent.device)
        if is_image:
            obs = agent.model.autoencoder.embed(obs)
        return obs

    agent.model.prev_state[:] = torch.zeros(agent.model.transition_network.gru_dim)
    taken_actions = []
    frames = []
    for time in range(env._size):
        # get model predictions
        obs, *_ = reset_env_to(env, taken_actions)

        obs = process_obs(obs)
        states, rewards, terminals, h = agent.model.predict(
            obs, agent.model.prev_state, batch=True
        )
        v = agent.discount * (
            agent.value_network.predict(torch.cat((states, h), dim=1))
            * (terminals < TP_THRESHOLD)
        )
        values = (rewards + v).detach().cpu().numpy()
        selected_action = agent.random_state.choice(
            np.where(np.isclose(values, max(values)))[0]
        )

        frame = plot_frame(
            env,
            time,
            taken_actions,
            states,
            rewards,
            terminals,
            values,
            selected_action,
        )
        frames.append(frame)

        taken_actions.append(selected_action)
        agent.model.prev_state = h[selected_action]

    return compile_frames(frames)


def log_trajectories(env: gym.Env, agent, n_traj, logger, timestep):
    trajs = []
    for _ in range(n_traj):
        trajs.append(simulate_trajectory(env, agent))
        agent.model.sample()

    logger.data_manager.log_images("Trajectories", trajs, timestep)
