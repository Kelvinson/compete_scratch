import gym
import gym_compete
import pickle
import sys
import os.path as osp
import argparse
import tensorflow as tf
import numpy as np
import logging
from policy import LSTMPolicy
from baselines.common import set_global_seeds, Dataset, explained_variance, fmt_row, zipsame
import baselines.common.tf_util as U
from baselines import logger, bench
from learn import compete_learn


def train(env, seed):
    if env == "sumo-ants":
        env = gym.make("sumo-ants-v0")
    else:
        print("right now I only support sumo-ants-v0")
        sys.exit()
    # seed = args.seed
    num_timesteps = 10
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    # env = bench.Monitor(env, logger.get_dir())
    set_global_seeds(seed)

    # policy = []
    # for i in range(2):
    #     scope = "policy" + str(i)
    #     policy.append(MlpPolicyValue(scope=scope, reuse=False,
    #                                  ob_space=env.observation_space.spaces[i],
    #                                  ac_space=env.action_space.spaces[i],
    #                                  hiddens=[64, 64], normalize=True))

    def policy_fn(pi_name, ob_space, ac_space, placeholder_name):
        scope = pi_name
        return LSTMPolicy(scope=scope, reuse=False,
                                     ob_space=ob_space,
                                     ac_space=ac_space,
                                     hiddens=[128, 128], normalize=True
                                     ,placeholder_name=placeholder_name)


    gym.logger.setLevel(logging.WARN)
    compete_learn(env, policy_fn,
            max_timesteps=num_timesteps,
            #TODO:remember to change it to 2048
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='constant', #T
        )

    env.close()

if __name__ == "__main__":
    # p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    # p.add_argument("--env", default="sumo-ants", type=str, help="competitive environment: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
    # p.add_argument("--seed", default=123, required=True, type=str)
    #
    # args = p.parse_args()
    train(env="sumo-ants", seed = 123)