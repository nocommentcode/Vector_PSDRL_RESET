import numpy as np
import torch

from ..common.replay import Dataset
from ..common.utils import create_state_action_batch


class EnsembleModel:
    def __init__(
        self,
        config: dict,
        state_size: int,
        actions: torch.tensor,
        transition_network: torch.nn.Module,
        terminal_network: torch.nn.Module,
        autoencoder: torch.nn.Module,
        device: str,
    ):
        self.device = device
        self.transition_network = transition_network
        self.terminal_network = terminal_network
        self.autoencoder = autoencoder
        self.actions = torch.tensor(actions).to(self.device)
        self.num_actions = len(self.actions)
        self.state_size = state_size
        self.prev_state = torch.zeros(self.transition_network.gru_dim).to(self.device)

        self.ensemble_size = config["ensemble_size"]

        # Random sampling from the prior
        self.sample()

    def predict(
        self,
        x: torch.tensor,
        h: torch.tensor,
        batch: bool,
    ):
        """
        Simulate one timestep using the current sampled model(s).
        """
        if batch:
            x, h = create_state_action_batch(
                x, self.actions, h, self.num_actions, self.device
            )

        # replicate obs and hidden for each ensemble
        obs = torch.concatenate([x for _ in range(self.ensemble_size)], 0)
        h = torch.concatenate([h for _ in range(self.ensemble_size)], 0)

        # predict for all ensemble
        prediction, h1 = self.transition_network.predict(obs, h)

        def sample_ensemble(pred):
            pred = pred.view((self.ensemble_size, -1, *pred.shape[1:]))
            return pred[self.sampled_index]

        # sample ensemble currently sampled
        prediction = sample_ensemble(prediction)
        h1 = sample_ensemble(h1)

        states, rewards = prediction[:, :-1], prediction[:, -1]

        terminals = self.terminal_network.predict(states)
        return states, rewards.reshape(-1, 1), terminals, h1

    def update(self, dataset: Dataset):
        self.sample()

    def sample(self):
        self.sampled_index = np.random.randint(0, self.ensemble_size)
