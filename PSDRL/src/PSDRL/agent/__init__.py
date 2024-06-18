from .psdrl import NeuralLinearPSDRL, EnsemblePSDRL
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.logger import Logger


def Agent(
    config: dict,
    actions: list,
    logger: "Logger",
    env_dim: int,
    seed: int = None,
):
    algorithm = config["algorithm"]["name"]
    if algorithm == "PSDRL":
        agent = NeuralLinearPSDRL
    elif algorithm == "Ensemble":
        agent = EnsemblePSDRL
    else:
        raise ValueError(f"algorithm {algorithm} is not supported")

    return agent(
        config,
        actions,
        logger,
        env_dim,
        seed,
    )
