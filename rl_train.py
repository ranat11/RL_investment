import argparse
import difflib
import importlib
import os

import gym
import numpy as np
import seaborn
import torch as th
from stable_baselines3.common.utils import set_random_seed

# Register custom envs
from utils.exp_manager import ExperimentManager
from utils.utils import ALGOS, StoreDict

seaborn.set()

def input_args():

    parser = argparse.ArgumentParser()
    # Algo & Env 
    parser.add_argument("--algo", default="a2c", help="RL Algorithm", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env", default="ForexEnv-v0", help="environment ID", type=str)
    parser.add_argument("--env-kwargs", default={"env_data": "data/EURUSDdays_ti.csv"}, type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("--vec-env", default="dummy", help="VecEnv type", type=str, choices=["dummy", "subproc"])

    # Train
    parser.add_argument("-n", "--n-timesteps", default=-1, help="Overwrite the number of timesteps", type=int)
    parser.add_argument("-i", "--trained-agent", default="", help="Path to a pretrained agent to continue training", type=str)
    # parser.add_argument("-i", "--trained-agent", default="logs/models/a2c/ForexEnv-v0_5/ForexEnv-v0.zip", help="Path to a pretrained agent to continue training", type=str)

    # Save & tensorboard
    parser.add_argument("--save-freq", default=-1, help="Save the model every n steps (if negative, no checkpoint)", type=int)
    parser.add_argument("--save-replay-buffer", default=False, help="Save the replay buffer too (when applicable)", action="store_true")
    parser.add_argument("-f", "--log-folder", default="logs/models", help="Log folder", type=str)
    parser.add_argument("-tb", "--tensorboard-log", default="./logs/tb", help="Tensorboard log dir", type=str)
    parser.add_argument("--log-interval", default=1, help="Override log interval (default: -1, no change)", type=int)

    # evaluate
    parser.add_argument("--eval-freq", default=10000, help="Evaluate the agent every n steps (if negative, no evaluation). During hyperparameter optimization n-evaluations is used instead", type=int,)
    parser.add_argument("--eval-episodes", default=20, help="Number of episodes to use for evaluation", type=int)
    parser.add_argument("--n-eval-envs", default=4, help="Number of environments for evaluation", type=int)
    
    # hyperparameters optimizer
    parser.add_argument("-optimize", "--optimize-hyperparameters", default=False, action="store_true", help="Run hyperparameters search")
    parser.add_argument("--study-name", default="test", help="Study name for distributed optimization", type=str)
    parser.add_argument("--n-trials", default=10, help="Number of trials for optimizing hyperparameters", type=int)
    parser.add_argument("--n-evaluations", default=20, help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization",type=int)
    parser.add_argument("--optimization-log-path", default="logs/optimizer", help="Path to save the evaluation log and optimal policy for each hyperparameter tried during optimization. Disabled if no argument is passed.", type=str)
    parser.add_argument("--n-jobs", default=4, help="Number of parallel jobs when optimizing hyperparameters", type=int)
    parser.add_argument("--sampler", default="tpe", help="Sampler to use when optimizing hyperparameters", type=str, choices=["random", "tpe", "skopt"],)
    parser.add_argument("--n-startup-trials", default=10, help="Number of trials before using optuna sampler", type=int)

    # Rarely use
    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict, help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)

    parser.add_argument("--no-optim-plots", action="store_true", default=False, help="Disable hyperparameter optimization plots")
    parser.add_argument("--pruner", help="Pruner to use when optimizing hyperparameters", type=str, default="median", choices=["halving", "median", "none"],)
    parser.add_argument("--storage", help="Database storage path if distributed optimization should be used", type=str, default=None)

    parser.add_argument("--truncate-last-trajectory", help="When using HER with online sampling the last trajectory in the replay buffer will be truncated after reloading the replay buffer.", default=True, type=bool,)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)

    return parser.parse_args()


def check_env(env_id):
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    return


def check_trainagent(trained_agent):
    if trained_agent != "":
        assert trained_agent.endswith(".zip") and os.path.isfile(trained_agent), "The trained_agent must be a valid path to a .zip file"
        
        
def set_tread(num_threads, verbose):
    # Setting num threads to 1 makes things run faster on cpu
    if num_threads > 0:
        if verbose > 1:
            print(f"Setting torch.num_threads to {num_threads}")
        th.set_num_threads(num_threads)

    return


if __name__ == "__main__": 
    args = input_args()

    env_id = args.env; check_env(env_id)
    seed = np.random.randint(2 ** 32 - 1, dtype="int64").item(); set_random_seed(seed)

    set_tread(args.num_threads, args.verbose)
    check_trainagent(args.trained_agent)

    print("=" * 10, env_id, "=" * 10); print(f"Seed: {seed}")


    exp_manager = ExperimentManager(
        args,
        args.algo,
        env_id,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        seed=seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
        )


    if not args.optimize_hyperparameters:
        model = exp_manager.setup_experiment()
        exp_manager.learn(model)
        exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()
