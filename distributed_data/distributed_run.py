import logging
import sys
import threading
import tensorflow as tf
import baselines.common.tf_util as U
import gym
from baselines.common import set_global_seeds

from distributed_data.distributed_utils import traj_segment_generator
from distributed_data.distributed_learn  import compete_learn
from policy import MlpPolicy
import threading, queue

N_WORKER = 3

def train(env, seed):

    # seed = args.seed
    num_timesteps = 0
    U.make_session(num_cpu=0).__enter__()
    # bench.Monitor(env, "log")
    # env.seed(seed)
    # env = bench.Monitor(env, logger.get_dir())
    set_global_seeds(seed)
    if env == "you-shall-not-pass":
        #whe I use MLP policy, it can be applied to the environment "run-to-goal" and "you-shall-not-pass environments"
        env = gym.make("you-shall-not-pass-humans-v0")
    else:
        print("right now I only support run-to-goal-humans-v0")
        sys.exit()

    # policy = []
    # for i in range(2):
    #     scope = "policy" + str(i)
    #     policy.append(MlpPolicyValue(scope=scope, reuse=False,
    #                                  ob_space=env.observation_space.spaces[i],
    #                                  ac_space=env.action_space.spaces[i],
    #                                  hiddens=[64, 64], normalize=True))

    def policy_fn(pi_name, ob_space, ac_space, placeholder_name):
        return MlpPolicy(name=pi_name,
                                     ob_space=ob_space,
                                     ac_space=ac_space,
                                     hid_size=64, num_hid_layers=2,
                                     placeholder_name=placeholder_name)


    # env = bench.Monitor(env, logger.get_dir())
    # print("the loogger data is stored at:",logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    compete_learn(env, policy_fn,
            max_timesteps=num_timesteps,
            #TODO:remember to change it to 2048
            timesteps_per_batch=5120,
            clip_param=0.2, entcoeff=0.0,
            max_iters=45,
            optim_epochs=6, optim_stepsize=3e-4, optim_batchsize=5120,
            gamma=0.995, lam=0.95, schedule='constant', #T
        )

    env.close()

if __name__ == "__main__":
    # p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    # p.add_argument("--env", default="sumo-ants", type=str, help="competitive environment: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
    # p.add_argument("--seed", default=123, required=True, type=str)
    #
    # args = p.parse_args()
    len1 = 2
    env = gym.make("you-shall-not-pass-humans-v0")
    ob_space = env.observation_space.spaces
    ac_space = env.action_space.spaces
    def policy_fn(pi_name, ob_space, ac_space, placeholder_name):
        return MlpPolicy(name=pi_name,
                                     ob_space=ob_space,
                                     ac_space=ac_space,
                                     hid_size=64, num_hid_layers=2,
                                     placeholder_name=placeholder_name)



    pi = [policy_fn("pi" + str(i), ob_space[i], ac_space[i], placeholder_name="observation" + str(i)) for i in range(len1)]
    GLOBAL_PPO = train(env, seed = 123)
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear() #not udpate now
    UPDATE_EVENT.set() # start to roll out
    seggen = traj_segment_generator(pi, env,  stochastic=True)
    workers = [seggen for i in N_WORKER]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO, ))
    threads[-1].start()
    COORD.join(threads)
