"""Run policies on the default scene and print metrics.

Usage: python -m scripts.run_episode
"""
from wanderai.scene import default_scene
from wanderai.environment import SceneSearchEnv, EnvConfig
from wanderai.policies import OraclePolicy, RandomPolicy, run_episode
from wanderai.metrics import summarize


def main():
    for name, policy in [("oracle", OraclePolicy()), ("random", RandomPolicy(seed=0))]:
        results = [
            run_episode(SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=400)), policy)
            for _ in range(5)
        ]
        print(name, summarize(results))


if __name__ == "__main__":
    main()
