#!/usr/bin/self._env python3
"""Training for the OpenAI Gym CartPole V1 problem.
1. Train q learner with random initial conditions.
2. Attempt to find weak spots for trained model with Gremlin's adversarial training.
3. Retrain q learner with weak initial conditions identified by Gremlin.
"""
import json
import argparse
import re

from q_learner_cartpolev1 import CartPoleV1_Q_Learner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-fpath",
        required=True,
        help="Path to model file",
    )
    parser.add_argument(
        "--num-episodes",
        required=True,
        help="Number of episodes to train the q learner for",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training status",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the CartPole visualization",
    )
    parser.add_argument(
        "--stats",
        required=False,
        help="Optional JSON file containing stats for all 4 genes",
    )
    args = parser.parse_args()

    model = CartPoleV1_Q_Learner()

    if not args.stats:
        print("Training Q Learner with random initial states")
        progress = model.train(
            args.verbose, args.render, episodes=int(args.num_episodes)
        )
        progress_fpath = f"./{args.model_fpath.replace('_q-table.pkl', '')}_q_learner_progress.csv"
        new_model_fpath = args.model_fpath
    else:
        with open(args.stats, "r") as stats_file:
            stats = json.loads(stats_file.read())
        print("Training Q Learner with adversarial examples from Gremlin")
        model.load(args.model_fpath)
        progress = model.train(
            args.verbose,
            args.render,
            initial_state_dist=stats,
            episodes=int(args.num_episodes),
        )
        # Form new model file
        if "baseline" in args.model_fpath:
            new_model_fpath = args.model_fpath.replace(
                "baseline", "gremlin-enhanced-iter0"
            )
        else:
            regex = re.compile(r"\d+")
            gremlin_run_id = int(regex.findall(args.model_fpath)[1])
            new_model_fpath = args.model_fpath.replace(
                f"iter{gremlin_run_id}", f"iter{gremlin_run_id+1}"
            )
        progress_fpath = f"./{new_model_fpath.replace('_q-table.pkl', '')}_q_learner_progress.csv"

    # Save model file
    model.save(new_model_fpath)
    print(f"Training progress written to {progress_fpath}")
    progress.to_csv(progress_fpath, index=False)
    print(f"Model written to {new_model_fpath}")
