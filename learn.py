
from policy import LSTMPolicy
from baselines.common import set_global_seeds, Dataset, explained_variance, fmt_row, zipsame
from baselines import bench
from baselines import logger
from utils import flatten_lists
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
from baselines import logger, bench
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from utils import traj_segment_generator, add_vtarg_and_adv

def compete_learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    # at this stage, the ob_space and reward_space is
    #TODO: all of this tuples are not right ? becuase items in tuple is not mutable
    #TODO: another way to store the two agents' states is to use with with tf.variable_scope(scope, reuse=reuse):
    len1 = len(env.agents)
    ob_space = env.observation_space.spaces
    ac_space = env.action_space.spaces
    pi = tuple([policy_func("pi" + str(i), ob_space[i], ac_space[i]) for i in range(len1)])
    oldpi = tuple([policy_func("oldpi" + str(i), ob_space[i], ac_space[i]) for i in range(len1)])
    atarg = tuple([tf.placeholder(dtype=tf.float32, shape=[None]) for i in range(len1)])
    ret = tuple([tf.placeholder(dtype=tf.float32, shape=[None]) for i in range(len1)])
    lrmult = 1.0 # here for simple I only use constant learning rate multiplier
    clip_param = clip_param * lrmult

    #TODO: this is the places I dont understand fully
    ob = U.get_placeholder_cached(name="ob") # Note: I am not sure about this point

    ac = tuple([pi[i].pdtype.sample_placeholder([None]) for i in range(len1)])


    kloldnew = tuple([oldpi[i].pd.kl(pi[i].pd) for i in range(len1)])
    ent = tuple([pi[i].pd.entropy for i in range(len1)])
    meankl = tuple(U.mean(kloldnew[i]) for i in range(len1))
    meanent = tuple([U.mean(ent[i]) for i in range(len1)])

    pol_entpen = tuple([(-entcoeff) * meanent[i]] for i in range(len1))
    ratio = tuple(tf.exp(pi[i].pd.logp(ac[i] - oldpi.pd.logp(ac[i]))) for i in range(len1)) #pnew / pold
    surr1 = tuple([ratio * atarg[i] for i in range(len1)])
    # U.clip = tf.clip_by_value(t, clip_value_min, clip_value_max,name=None):
    # among which t is A 'Tensor' so
    surr2 = tuple([U.clip(ratio[i], 1.0 - clip_param, 1.0 + clip_param) for i in range(len1)])
    pol_surr = tuple([-U.mean(tf.minimum(surr1[i], surr2[i])) for i in range(len1)])
    vf_loss = tuple([U.mean(tf.square(pi[i].vpred - ret[i])) for i in range(len1)])
    total_loss = tuple([pol_surr[i] + pol_entpen[i] + vf_loss[i] for i in range(len1)])
    losses = tuple([[pol_surr[i], pol_entpen[i], vf_loss[i], meankl[i], meanent[i]] for i in range(len1)])
    loss_names = ["pol_sur", "pol_entpen","vf_loss", "kl", "ent"]
    var_list = tuple([pi[i].get_trainable_variables() for i in range(len1)])
    lossandgrad = tuple([U.function([ob[i], ac[i], atarg[i], ret[i], lrmult[i]], losses[i] + [U.flatgrad(total_loss[i], var_list[i])]) for i in range(len1)])
    adam = tuple([MpiAdam(var_list[i], epsilon=adam_epsilon) for i in range(2)])

    #TODO: I wonder this cannot function as expected because the result is a list of functions, not will not execute automatically
    assign_old_eq_new = [U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi[i].get_variables(), pi[i].get_variables())]) for i in range(len1)]

    # compute_losses is a function, so it should not be copied to copies, nevertheless the parameters should be
    # passed into it as the two agents
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    # [adam[i].sync() for i in range(2)]
    adam[0].sync()
    adam[1].sync()
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        vpredbefore = []
        tdlamret = []
        for i in range(len1):
            ob[i], ac[i], atarg[i], tdlamret_temp= seg["ob"][i], seg["ac"][i], seg["adv"][i], seg["tdlamret"][i]
            tdlamret.append(tdlamret_temp)
            vpredbefore.append(seg["vpred"][i]) # predicted value function before udpate
            atarg[i] = (atarg[i] - atarg[i].mean()) / atarg[i].std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob[i], ac=ac[i], atarg=atarg[i], vtarg=tdlamret[i]), shuffle=not pi[i].recurrent)
            optim_batchsize = optim_batchsize or ob[i].shape[0]

            if hasattr(pi[i], "ob_rms"): pi[i].ob_rms.update(ob[i]) # update running mean/std for policy

            #TODO: I have to make suer how assign_old_ea_new works and whether to assign it for each agent
            assign_old_eq_new[i]() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names[i]))
        # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses[i] = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam[i].update(g, optim_stepsize * cur_lrmult)
                    losses[i].append(newlosses)

            # logger.log("Evaluating losses...")
            losses[i] = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses[i] = compute_losses(batch["ob"][i], batch["ac"][i], batch["atarg"][i], batch["vtarg"][i], cur_lrmult)
                losses[i].append(newlosses)
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            # logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names[i]):
                logger.record_tabular("loss_"+name, lossval)
            # logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore[i], tdlamret[i]))
            lrlocal = (seg["ep_lens"][i], seg["ep_rets"][i]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            # logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            # logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            # logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            # logger.record_tabular("EpisodesSoFar", episodes_so_far)
            # logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            # logger.record_tabular("TimeElapsed", time.time() - tstart)
            # if MPI.COMM_WORLD.Get_rank()==0:
            #     logger.dump_tabular()


