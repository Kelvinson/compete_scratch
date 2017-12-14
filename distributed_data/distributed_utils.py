import pickle
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

# note after I revised this function, the trajectory segment generator will receive
# the double-view observation and output the joint actions of the competitive two agents


"""
input: pi: a tuple of two polices,
       env, horizon: the specified timesteps, stochastic: whether the process
       should be stochastic
"""
def traj_segment_generator(pi, env, stochastic):
    global episode_length
    assert(len(pi) == 2,"this policy network in the environment must be a tuple of multiple agetns")

    #len1 = len(env.agents)
    len1= 2
    t = 0
    # ac is a tuple of multiple agents' actions
    # ac should a tuple of two agents' actions
    ac = env.action_space.sample() # not used, just so we have the datatype
    #for the two agents whether this T timestamps should end(one agent dies or ..)
    new = [True for i in range(len1)] # marks if we're on first timestep of an episode
    # observation being set
    ob = env.reset()   # ob is a tuple

    #TODO: in this file: I change to use list as the data structure to store the information of the two agents
    cur_ep_ret = [0.0 for i in range(len1)] # return in current episode
    cur_ep_len = [0.0 for i in range(len1)]# len of current episode
    ep_rets = [[] for i in range(len1)] # returns of completed episodes in this segment
    ep_lens = [[] for i in range(len1)] # lengths of ...

    #TODO: now all the T timestamps produced data is converted to data within a dnoe episode.
    obs = [[] for i in range(len1)]
    rews = [[] for i in range(len1)]
    vpreds = [[] for i in range(len1)]
    news = [[]for i in range(len1)]
    # in python3 list also have copy method the original is a numpy copy now it is a list copy
    acs = [[] for i in range(len1)] # array of T timestamps of action tuples
    prevacs = acs.copy()

    #a helper variable to store the number of runs already.
    episode_now = 0

    while True:
        # because prevac is a list copy of ac so it is safe here to use "=" assignment
        prevac = ac
        # don't know how it will differ from get ac and predicted value simutanously
        # I think it should be done in that way, however I don't know how to get tuple of tuples
        ac = [pi[i].act(ob=ob[i], stochastic=stochastic)[0] for i in range(len1)]
        vpred = [pi[i].act(ob=ob[i], stochastic=stochastic)[1] for i in range(len1)]
        # ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value

        if t > 0 and (new[0] or new[1]):

            yield {"ob" : obs[:t], "rew" : rews[:t], "vpred" : vpreds[:t], "new" : news[:t],
                    "ac" : acs[:t], "prevac" : prevacs[:t], "nextvpred": [vpred[k] * (1 - new[k]) for k in range(len1)],
                   "ep_rets": ep_rets, "ep_lens": ep_lens}
            ep_rets = [[] for i in range(len1)]  # returns of completed episodes in this segment
            ep_lens = [[] for i in range(len1)]
            episode_length += 1
        #   obs = [[] for i in range(len1)]
            rews = [[] for i in range(len1)]
            vpreds = [[] for i in range(len1)]
            news = [[]for i in range(len1)]
            # in python3 list also have copy method the original is a numpy copy now it is a list copy
            acs = [[] for i in range(len1)] # array of T timestamps of action tuples
            prevacs = acs.copy()

        #TODO: now i is the timestamp index and j becomes the index of the agents, previously i is the index of the two agents
        for j in range(len1):
            obs[j].append(ob[j])
            vpreds[j].append(vpred[j])
            news[j].append(new[j])
            acs[j].append(ac[j])
            prevacs[j].append(prevac[j])

        #now it is time to turn ac to tuple to feed into env.step function
        # print("at timestamp {} the rewards are {} {}.".format(t,rewrd[0], rewrd[1]));
        ac = tuple(ac)
        ob, rew, new, info = env.step(ac)

        for j in range(len1):
            #TODO: I have to revise the reward to be like it in the papar just like in the My_Simple_PPO_LSTM_New.py file
            # now because the data is collected distributedly, the episode should be declared globally

            exploration_reward = info[j]['reward_move']* (1 - (episode_length + episode_now) * 0.002)
            competition_reward = info[j]['reward_remaining'] * (episode_length + episode_now) * 0.002
            #
            rewrd = exploration_reward + competition_reward
            rews[j].append(rewrd)

            cur_ep_ret[j] += rewrd
            cur_ep_len[j] += 1

        print("t{} rewards {} {}.".format(t, rews[0][-1], rews[1][-1]))
        if new[0] or new[1]:
            for j in range(len1):
                ep_rets[j].append(cur_ep_ret[j])
                ep_lens[j].append(cur_ep_len[j])
                cur_ep_ret[j] = 0
                cur_ep_len[j] = 0
            ob = env.reset()
        t += 1

# TODO: had to fix this function to function right.
def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    len1 = 2
    tmpadv = [np.zeros(()) for i in range(len1)]
    tmplamret = [np.zeros(()) for i in range(len1)]
    for j in range(len1):
        new = np.append(seg["new"][j], 0)
        vpred = np.append(seg["vpred"][j], seg["nextvpred"][j])
        T = len(seg["rew"][j])
       # using this consecutive assign operator, gaelam and tmpadv[j] are the numpy array, so if you change gaelam you also have
        # changed tmpadv[j]
        tmpadv[j] =gaelam = np.empty(T, 'float32')
        rew = seg["rew"][j]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - new[t+1]
            delta = rew[t] + gamma  * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam =delta + gamma * lam * nonterminal * lastgaelam
        # the following two equations should be the same
        # tmplamret[j] = gaelam + seg["vpred"][j]
        tmplamret[j] = tmpadv[j] + seg["vpred"][j]

    # after the two operations, assign the tmpadv and tmplamret to seg["adv"] and seg["tdlamret"]
    seg["adv"] = tmpadv
    seg["tdlamret"] = tmplamret

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

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
        size = int(np.prod(shape)  )
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})



