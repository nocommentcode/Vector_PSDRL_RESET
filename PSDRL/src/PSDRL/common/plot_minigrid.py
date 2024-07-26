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

from matplotlib.patches import Rectangle, Circle


class MinigridImagePlot:
    def __init__(self, env: gym.Env, agent: PSDRL, name: str):
        self.env = env
        self.agent = agent
        self.name = name

    def generate_and_log(self, logger, timestep):
        image = self.generate_image()
        logger.data_manager.log_images(self.name, [image], timestep)

    def generate_image(self):
        pass

    def compile_frames(self, *rows):
        padding = 10

        def generate_blank_image():
            total_padding = padding * (len(rows[0]) - 1)
            width = sum(image.width for image in rows[0]) + total_padding
            height = rows[0][0].height * len(rows) + total_padding

            return Image.new(
                "RGB",
                (
                    width,
                    height,
                ),
            )

        compiled_image = generate_blank_image()

        y_offset = 0
        for row in rows:
            x_offset = 0
            for image in row:
                compiled_image.paste(image, (x_offset, y_offset))
                x_offset += image.width + padding
            y_offset += image.height + padding

        return compiled_image

    def reset_env_and_agent(self):
        self.agent.model.prev_state[:] = torch.zeros(
            self.agent.model.transition_network.gru_dim
        )
        return self.env.reset()

    def predict_from_agent(self, obs):
        return self.agent.model.predict(
            torch.from_numpy(obs).float().to(self.agent.model.device),
            self.agent.model.prev_state,
            batch=True,
        )

    def plot_to_img(self):
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return Image.open(buf)


class TransitionModelImage(MinigridImagePlot):
    def __init__(self, env: gym.Env, agent: PSDRL):
        super().__init__(env, agent, "TransitionModel")

    def make_frame(self, obs, reward, terminal):
        direction = obs[-1]
        obs = obs[:-1].reshape((7, 7, 3))
        obs = obs / obs.max()

        plt.close()
        plt.imshow(obs)
        plt.title(
            f"{str(round(direction, 2))}|{str(round(reward, 3))}|{str(round(terminal, 2))}"
        )

        return self.plot_to_img()

    def generate_image(self):
        true_frames = []
        pred_frames = []

        obs = self.reset_env_and_agent()
        for time in range(20):
            states, rewards, terminals, h = self.predict_from_agent(obs)
            action = self.agent.select_action(obs, time)

            pred_obs = states.detach().cpu().numpy()[action]
            pred_rew = rewards.detach().cpu().numpy()[action][0]
            pred_term = terminals.detach().cpu().numpy()[action][0]

            obs, reward, terminal, *_ = self.env.step(action)

            # make frames
            true_frames.append(self.make_frame(obs, reward, terminal))
            pred_frames.append(self.make_frame(pred_obs, pred_rew, pred_term))

            if terminal:
                break

        return self.compile_frames(pred_frames, true_frames)


class TrajectoryImages(MinigridImagePlot):
    def __init__(self, env: gym.Env, agent: PSDRL):
        super().__init__(env, agent, "Trajectories")

    def make_frame(self, coords):
        size = self.env.unwrapped.width
        x_coords = [x for x, _ in coords]
        y_coords = [y for _, y in coords]

        plt.close()
        plt.plot(
            x_coords,
            y_coords,
            color=("green" if coords[-1] == (size - 2, size - 2) else "blue"),
        )
        plt.grid()
        plt.xlim(0, size)
        plt.ylim(0, size)

        ax = plt.gca()

        starting_point = Circle((1, 1), radius=0.3, color="b")
        ax.add_patch(starting_point)

        goal_point = Circle((size - 2, size - 2), radius=0.3, color="g")
        ax.add_patch(goal_point)

        return self.plot_to_img()

    def generate_image(self):
        trajs = []
        for _ in range(10):
            obs = self.reset_env_and_agent()
            coords = [self.env.unwrapped.agent_pos]
            for t in range(self.env.unwrapped.max_steps):
                a = self.agent.select_action(obs, t)
                obs, _, terminal, *_ = self.env.step(a)
                coords.append(self.env.unwrapped.agent_pos)
                if terminal:
                    break
            trajs.append(self.make_frame(coords))

        return self.compile_frames(trajs)


class ValueHeatmap(MinigridImagePlot):
    def __init__(self, env: gym.Env, agent: PSDRL):
        super().__init__(env, agent, "Value")

    def get_action_to_coords(self, x, y):
        FORWARD = 2
        TURN_RIGHT = 1

        curr_x = 1
        curr_y = 1
        actions = []

        # move in x
        while curr_x != x:
            actions.append(FORWARD)
            curr_x += 1

        # turn
        actions.append(TURN_RIGHT)

        # move in y
        while curr_y != y:
            actions.append(FORWARD)
            curr_y += 1

        # turn 3 more times to get all 4 directions
        for _ in range(3):
            actions.append(TURN_RIGHT)
        return actions

    def get_obs_and_hidden_for_coords(self, x, y):
        obs = self.reset_env_and_agent()

        # move agent to desired square
        observations = []
        hiddens = []
        for action in self.get_action_to_coords(x, y):
            *_, h = self.predict_from_agent(obs)
            self.agent.model.prev_state = h[action]
            obs, *_ = self.env.step(action)

            observations.append(obs)
            hiddens.append(h[action])

        # last 4 obs and hiddens are the ones to predict values for
        def make_tensor(array):
            combined_tensor = torch.zeros((4, *array[0].shape)).to(self.agent.device)
            for i, tensor in enumerate(array[-4:]):
                if type(tensor) == np.ndarray:
                    tensor = torch.from_numpy(tensor).float().to(self.agent.device)
                combined_tensor[i] = tensor

            return combined_tensor

        hiddens = make_tensor(hiddens)
        observations = make_tensor(observations)

        return hiddens, observations

    def get_value_for_obs(self, obs, hiddens):
        return self.agent.value_network.predict(torch.cat((obs, hiddens), dim=1))

    def build_values(self):
        size = self.env.unwrapped.width - 2
        values = np.zeros((size * 2, size * 2))

        for x in range(1, size + 1):
            for y in range(1, size + 1):
                hidden, obs = self.get_obs_and_hidden_for_coords(x, y)
                v = self.get_value_for_obs(obs, hidden).detach().cpu().numpy()
                for i in range(len(v)):
                    x_offset = 1 if i in (1, 3) else 0
                    y_offset = 1 if i in (2, 3) else 0
                    values[(x - 1) * 2 + x_offset, (y - 1) * 2 + y_offset] = v[i]

        return values

    def generate_image(self):
        values = self.build_values()
        size = self.env.unwrapped.width - 2

        plt.close()
        ax = sns.heatmap(values, linewidth=0.5, xticklabels=[], yticklabels=[])

        # add gridlines and ticks
        ax.vlines(
            [x * 2 for x in range(size)],
            0,
            size * 2,
            colors="black",
            linewidth=5,
            zorder=1,
        )
        ax.hlines(
            [y * 2 for y in range(size)],
            0,
            size * 2,
            colors="black",
            linewidth=5,
            zorder=1,
        )
        ax.set_xticks(
            [i for i in range(size * 2)],
            [str((x // 2) + 1) if x % 2 != 0 else "" for x in range(size * 2)],
        )
        ax.set_yticks(
            [i for i in range(size * 2)],
            [str((x // 2) + 1) if x % 2 != 0 else "" for x in range(size * 2)],
        )

        # plot goal sqaure
        goal = Rectangle(
            ((size - 1) * 2, (size - 1) * 2),
            2,
            2,
            linewidth=5,
            edgecolor="green",
            facecolor="none",
            zorder=2,
        )
        ax.add_patch(goal)

        return self.plot_to_img()
