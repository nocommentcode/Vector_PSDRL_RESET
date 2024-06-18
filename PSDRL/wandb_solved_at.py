import argparse
import wandb
import numpy as np

PROJECT = "qmul_research/VectorPSDRLNew"
N_EPISODES = 20
THRESHOLD = 0.98

ALGORITHMS = ["PSDRL", "Ensemble", "ShallowEnsemble"]


def set_solved_at_episode(env, algorithm):
    api = wandb.Api()
    runs = api.runs(
        path=PROJECT,
        filters={"config.algorithm.name": algorithm, "config.experiment.env": env},
    )

    for run in runs:
        hist = run.scan_history(keys=["Reward/Train_Reward"], page_size=1e6)
        train_reward = np.array([row["Reward/Train_Reward"] for row in hist])
        running_av = np.convolve(
            train_reward, np.ones(N_EPISODES) / N_EPISODES, mode="valid"
        )

        solved = running_av >= THRESHOLD

        if not np.any(solved):
            solved_episode = len(train_reward)

        else:
            solved_after = np.argmax(solved)
            solved_episode = solved_after + N_EPISODES

        print(f"{run.url}: {solved_episode}")

        run.summary["solved_after"] = solved_episode
        run.summary.update()


def run(envs):
    for env in envs:
        for alg in ALGORITHMS:
            print(f"{env} - {alg}: ")
            set_solved_at_episode(env, alg)
            print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", nargs="+", required=True, type=str)
    args = parser.parse_args()

    run(args.e)
