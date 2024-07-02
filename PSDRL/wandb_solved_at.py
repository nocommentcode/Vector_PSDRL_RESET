import argparse
from matplotlib import pyplot as plt
import wandb
import numpy as np

PROJECT = "qmul_research/VectorPSDRLNew"
N_EPISODES = 20
THRESHOLD = 0.98

ALGORITHMS = ["PSDRL", "Ensemble", "ShallowEnsemble"]


class WandbRunCollection:
    def __init__(self, algorithms, envs) -> None:
        self.algorithms = algorithms
        self.envs = envs

    def fetch_runs(self, algorithm, env):
        api = wandb.Api()
        runs = api.runs(
            path=PROJECT,
            filters={
                "config.algorithm.name": algorithm,
                "config.experiment.env": env,
            },
        )

        return runs

    def get_solved_at(self):
        def make_algo():
            return {env: [] for env in self.envs}

        results = {algo: make_algo() for algo in self.algorithms}

        for algo in self.algorithms:
            for env in self.envs:
                runs = self.fetch_runs(algo, str(env))
                for run in runs:
                    if "solved_after" not in run.summary:
                        continue
                    solved_at = run.summary["solved_after"]
                    results[algo][env].append(solved_at)

        return results

    def plot_solved_vs_env(self):
        results = self.get_solved_at()

        fig, ax = plt.subplots()
        colors = ["b", "g", "r", "c", "m"]
        for i, algo in enumerate(self.algorithms):
            envs = []
            means = []
            stds = []

            for env, solved_at in results[algo].items():
                envs.append(int(env))
                means.append(np.mean(solved_at))
                stds.append(np.std(solved_at))

            envs = np.array(envs)
            means = np.array(means)
            stds = np.array(stds)

            ax.plot(envs, means, color=colors[i], label=algo)
            ax.fill_between(
                envs, means - stds, means + stds, color=colors[i], alpha=0.3
            )

        ax.plot(self.envs, [2 ** (e + 1) for e in self.envs], "--", label="2^N+1")
        ax.set_title("DeepSea # Episodes to solve")
        ax.set_xlabel("Env size")
        ax.set_xticks(self.envs)
        ax.set_ylabel("Episodes")
        ax.legend()

        # Show the plot
        plt.show()


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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", nargs="+", required=True, type=str)
    # args = parser.parse_args()

    # run(args.e)
    w = WandbRunCollection(ALGORITHMS, envs=[4, 6, 8, 10])
    w.plot_solved_vs_env()
    g = 0
