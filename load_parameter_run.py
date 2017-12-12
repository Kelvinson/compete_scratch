from policy import LSTMPolicy, MlpPolicy
import gym
import gym_compete
import pickle
import sys
import argparse
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np


def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

def setFromFlat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})

def run(config):
    if config.env == "kick-and-defend":
        env = gym.make("kick-and-defend-v0")
        policy_type = "lstm"
    elif config.env == "run-to-goal-humans":
        env = gym.make("run-to-goal-humans-v0")
        policy_type = "mlp"
    elif config.env == "run-to-goal-ants":
        env = gym.make("run-to-goal-ants-v0")
        policy_type = "mlp"
    elif config.env == "you-shall-not-pass":
        env = gym.make("you-shall-not-pass-humans-v0")
        policy_type = "mlp"
    elif config.env == "sumo-humans":
        env = gym.make("sumo-humans-v0")
        policy_type = "lstm"
    elif config.env == "sumo-ants":
        env = gym.make("sumo-ants-v0")
        policy_type = "lstm"
    else:
        print("unsupported environment")
        print("choose from: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
        sys.exit()

    param_paths = "/tmp/model.ckpt"

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()
    ##
    # return MlpPolicy(name=pi_name,
    #                  ob_space=ob_space,
    #                  ac_space=ac_space,
    #                  hid_size=64, num_hid_layers=2,
    #                  placeholder_name=placeholder_name)

    policy = []
    for i in range(2):
        scope = "pi" + str(i)
        if policy_type == "lstm":
            policy.append(MlpPolicy(name=scope,
                                     ob_space=env.observation_space.spaces[i],
                                     ac_space=env.action_space.spaces[i],
                                     hid_size=64, num_hid_layers=2, placeholder_name="observation"+str(i)))
        else:
            policy.append(MlpPolicy(name=scope,
                                    ob_space=env.observation_space.spaces[i],
                                    ac_space=env.action_space.spaces[i],
                                    hid_size=64, num_hid_layers=2, placeholder_name="observation" + str(i)))


    pi0_variables = slim.get_variables(scope="pi0")
    pi1_variables = slim.get_variables(scope="pi1")
    parameters_to_save_list0 = [v for v in pi0_variables]
    parameters_to_save_list1 = [v for v in pi1_variables]
    parameters_to_save_list = parameters_to_save_list0 + parameters_to_save_list1
    # saver = tf.train.Saver(parameters_to_save_list)
    saver = tf.train.Saver()

    # initialize uninitialized variables
    sess.run(tf.variables_initializer(tf.global_variables()))
    saver.restore(sess, "saveparameter/60/60.pkl")
    # params = [load_from_file(param_pkl_path=path) for path in param_paths]
    # for i in range(len(policy)):
    #     setFromFlat(policy[i].get_variables(), params[i])

    max_episodes = config.max_episodes
    num_episodes = 0
    nstep = 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]
    # total_scores = np.asarray(total_scores)
    observation = env.reset()
    print("-"*5 + " Episode %d " % (num_episodes+1) + "-"*5)
    while num_episodes < max_episodes:
        env.render()
        # ac = [pi[i].act(ob=ob[i], stochastic=stochastic)[0] for i in range(len1)]
        # vpred = [pi[i].act(ob=ob[i], stochastic=stochastic)[1] for i in range(len1)]
        #
        action = tuple([policy[i].act(stochastic=True, ob=observation[i])[0]
                        for i in range(len(policy))])
        observation, reward, done, infos = env.step(action)
        nstep += 1
        for i in range(len(policy)):
            total_reward[i] += reward[i]
        if done[0]:
            num_episodes += 1
            draw = True
            for i in range(len(policy)):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, num_episodes))
            if draw:
                print("Game Tied: Agent {}, Scores: {}, Total Episodes: {}".format(i, total_scores, num_episodes))
            observation = env.reset()
            nstep = 0
            total_reward = [0.0  for _ in range(len(policy))]
            for i in range(len(policy)):
                policy[i].reset()
            if num_episodes < max_episodes:
                print("-"*5 + "Episode %d" % (num_episodes+1) + "-"*5)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--env", default="sumo-humans", type=str, help="competitive environment: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
    p.add_argument("--param-paths", nargs='+', required=True, type=str)
    p.add_argument("--max-episodes", default=10, help="max number of matches", type=int)

    config = p.parse_args()
    run(config)