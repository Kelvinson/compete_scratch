
from policy import LSTMPolicy
from baselines.common import set_global_seeds, Dataset, explained_variance, fmt_row, zipsame
from baselines import bench
from baselines import logger
import tensorflow.contrib.slim as slim
from utils import flatten_lists
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
from baselines import logger, bench
import time
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from mpi4py import MPI
from utils import traj_segment_generator, add_vtarg_and_adv
from baselines.common.mpi_moments import  mpi_moments

def compete_learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=20, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-3,
        schedule='linear' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    # at this stage, the ob_space and reward_space is
    #TODO: all of this tuples are not right ? becuase items in tuple is not mutable
    #TODO: another way to store the two agents' states is to use with with tf.variable_scope(scope, reuse=reuse):
    len1 = 2
    ob_space = env.observation_space.spaces
    ac_space = env.action_space.spaces
    pi = [policy_func("pi" + str(i), ob_space[i], ac_space[i],placeholder_name="observation"+str(i)) for i in range(len1)]
    oldpi = [policy_func("oldpi" + str(i), ob_space[i], ac_space[i], placeholder_name="observation"+str(i)) for i in range(len1)]
    atarg = [tf.placeholder(dtype=tf.float32, shape=[None]) for i in range(len1)]
    ret = [tf.placeholder(dtype=tf.float32, shape=[None]) for i in range(len1)]
    tdlamret = [[] for i in range(len1)]
    # TODO: here I should revise lrmult to as it was before
    # lrmult = 1.0 # here for simple I only use constant learning rate multiplier
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult

    #TODO: this point I cannot finally understand it, originally it is
    # ob=U.get_placeholder_cached(name="ob")
    #TODO: here it is a bug to fix, I think the get_placeholder_cached is global, you can only cache observation once and the next time if it finds the name placeholder it will return the previous placeholder, I don't know whether different namescope have  effect on this.
    # ob1 = U.get_placeholder_cached(name="observation1") # Note: I am not sure about this point
    # # ob2 = U.get_placeholder_cached(name="observation2")
    # ob1 = U.get_placeholder_cached(name="observation0")  # Note: I am not sure about this point
    # ob2 = U.get_placeholder_cached(name="observation1")
    #TODO: the only one question now is that pi network and oldpi networ both have the ob_ph named "observation", even in the original baseline implementation, does pi and oldpi share the observation placeholder, I think it is not

    ob = [U.get_placeholder_cached(name="observation"+str(i)) for i in range(len1)]
    # ac = tuple([pi[i].act(stochastic=True, observation=env.observation_space[i])[0]
    #      for i in range(len1)])
    # TODO: here for the policy to work I changed the observation parameter passed into the pi function to s which comes from env.reset()
    # s = env.reset()
    # ac = tuple([pi[i].act(stochastic=True, observation=s[i])[0]
    #             for i in range(len1)])

    ac = [pi[i].pdtype.sample_placeholder([None]) for i in range (len1)]
    kloldnew = [oldpi[i].pd.kl(pi[i].pd) for i in range(len1)]
    ent = [pi[i].pd.entropy() for i in range(len1)]
    print("ent1 and ent2 are {} and {}".format(ent[0], ent[1]))
    meankl = [U.mean(kloldnew[i]) for i in range(len1)]
    meanent = [U.mean(ent[i]) for i in range(len1)]

    pol_entpen = [(-entcoeff) * meanent[i] for i in range(len1)]
    ratio = [tf.exp(pi[i].pd.logp(ac[i]) - oldpi[i].pd.logp(ac[i])) for i in range(len1)]
    # ratio = [tf.exp(pi[i].pd.logp(ac) - oldpi[i].pd.logp(ac[i])) for i in range(len1)] #pnew / pold
    surr1 = [ratio[i] * atarg[i] for i in range(len1)]
    # U.clip = tf.clip_by_value(t, clip_value_min, clip_value_max,name=None):
    # # among which t is A 'Tensor' so
    surr2 = [U.clip(ratio[i], 1.0 - clip_param, 1.0 + clip_param) for i in range(len1)]
    pol_surr = [-U.mean(tf.minimum(surr1[i], surr2[i])) for i in range(len1)]
    vf_loss = [U.mean(tf.square(pi[i].vpred - ret[i])) for i in range(len1)]
    total_loss = [pol_surr[i] + pol_entpen[i] + vf_loss[i] for i in range(len1)]
    # here I ccome to realize that the following miscelleous losses are just operations not tensors so they should be
    # # be made to a list to contain the info of the two agents
    # surr2 = U.clip(ratio[i], 1.0 - clip_param, 1.0 + clip_param)
    # pol_surr = -U.mean(tf.minimum(surr1[i], surr2[i]))
    # vf_loss = U.mean(tf.square(pi[i].vpred - ret[i]))
    # total_loss = pol_surr + pol_entpen + vf_loss

    #TODO: in another way I choose to revise losses to following:
    losses = [[pol_surr[i], pol_entpen[i], vf_loss[i], meankl[i], meanent[i]] for i in range(len1)]
    loss_names = ["pol_sur", "pol_entpen","vf_loss", "kl", "ent"]
    var_list = [pi[i].get_trainable_variables() for i in range(len1)]

    lossandgrad = [U.function([ob[i], ac[i], atarg[i], ret[i], lrmult], losses[i] + [U.flatgrad(total_loss[i], var_list[i])]) for i in range(len1)]
    adam = [MpiAdam(var_list[i], epsilon=adam_epsilon) for i in range(2)]

    #TODO: I wonder this cannot function as expected because the result is a list of functions, not will not execute automatically
    # assign_old_eq_new = [U.function([],[], updates=[tf.assign(oldv, newv)
    #     for (oldv, newv) in zipsame(oldpi[i].get_variables(), pi[i].get_variables())]) for i in range(len1)]

    # compute_losses is a function, so it should not be copied to copies, nevertheless the parameters should be
    # passed into it as the two agents
    compute_losses = [U.function([ob[i], ac[i], atarg[i], ret[i], lrmult], losses[i]) for i in range(len1)]
    # sess = U.get_session()
    # writer = tf.summary.FileWriter(logdir='log-mlp',graph=sess.graph)
    # now when the training iteration ends, save the trained model and test the win rate of the two.
    pi0_variables = slim.get_variables(scope="pi0")
    pi1_variables = slim.get_variables(scope="pi1")
    parameters_to_save_list0 = [v for v in pi0_variables]
    parameters_to_save_list1 = [v for v in pi1_variables]
    parameters_to_save_list = parameters_to_save_list0 + parameters_to_save_list1
    saver = tf.train.Saver(parameters_to_save_list)
    # restore = tf.train.Saver(parameters_to_save_list)
    U.initialize()
    # restore.restore(U.get_session(), "saveparameter/20/20.pkl")
    U.get_session().run
    # [adam[i].sync() for i in range(2)]
    adam[0].sync()
    adam[1].sync()
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, horizon=timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    lenbuffer = [deque(maxlen=100) for i in range(len1)]  # rolling buffer for episode lengths
    rewbuffer = [deque(maxlen=100) for i in range(len1)] # rolling buffer for episode rewards

    parameters_savers = []
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



        # saver.restore()

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        #TODO: I got to fix this function to let it return the right seg["adv"] and seg["lamret"]
        add_vtarg_and_adv(seg, gamma, lam)

        losses = [[] for i in range(len1)]
        meanlosses = [[] for i in range(len1)]
        for i in range(len1):
            ob[i], ac[i], atarg[i], tdlamret[i]= seg["ob"][i], seg["ac"][i], seg["adv"][i], seg["tdlamret"][i]
            # ob_extend = np.expand_dims(ob[i],axis=0)
            # ob[i] = ob_extend
            vpredbefore = seg["vpred"][i] # predicted value function before udpate
            atarg[i] = (atarg[i] - atarg[i].mean()) / atarg[i].std() # standardized advantage function estimate
            d= Dataset(dict(ob=ob[i], ac=ac[i], atarg=atarg[i], vtarg=tdlamret[i]), shuffle=not pi[i].recurrent)
            optim_batchsize = optim_batchsize or ob[i].shape[0]

            if hasattr(pi[i], "ob_rms"): pi[i].ob_rms.update(ob[i]) # update running mean/std for policy

            #TODO: I have to make suer how assign_old_ea_new works and whether to assign it for each agent
            #Yes I can assure it will work now

            # save network parameters using tf.train.Saver
            #     saver_name = "saver" + str(iters_so_far)

            U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in zipsame(oldpi[i].get_variables(), pi[i].get_variables())])()
             # set old parameter values to new parameter values
        # Here we do a bunch of optimization epochs over the data
            logger.log("Optimizing the agent{}...".format(i))
            logger.log(fmt_row(13, loss_names))
            for _ in range(optim_epochs):
                losses[i] = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    # batch["ob"] = np.expand_dims(batch["ob"], axis=0)
                    *newlosses, g = lossandgrad[i](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam[i].update(g, optim_stepsize * cur_lrmult)
                    losses[i].append(newlosses)
                    logger.log(fmt_row(13, np.mean(losses[i], axis=0)))

            logger.log("Evaluating losses of agent{}...".format(i))
            losses[i] = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses[i](batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses[i].append(newlosses)
            meanlosses[i], _, _ = mpi_moments(losses[i], axis=0)
            logger.log(fmt_row(13, meanlosses[i]))
            for (lossval, name) in zipsame(meanlosses[i], loss_names):
                logger.record_tabular("loss_" + name, lossval)
            logger.record_tabular("ev_tdlam_before{}".format(i), explained_variance(vpredbefore, tdlamret[i]))

            lrlocal = (seg["ep_lens"][i], seg["ep_rets"][i]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer[i].extend(lens)
            rewbuffer[i].extend(rews)
            logger.record_tabular("EpLenMean {}".format(i), np.mean(lenbuffer[i]))
            logger.record_tabular("EpRewMean {}".format(i), np.mean(rewbuffer[i]))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        temp_pi = policy_func("temp_pi"+str(iters_so_far) , ob_space[0], ac_space[0], placeholder_name="temp_pi_observation" + str(iters_so_far))
        U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                    zipsame(temp_pi.get_variables(), pi[0].get_variables())])()
        parameters_savers.append(temp_pi)
        if iters_so_far % 5 == 0:
            sample_iteration = int(np.random.uniform(iters_so_far / 2, iters_so_far))
            print("now assign the {}th parameter of agent0 to agent1".format(sample_iteration))
            pi_restore = parameters_savers[sample_iteration]
            U.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                    zipsame(pi[1].get_variables(), pi_restore.get_variables())])()

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()


    # # now when the training iteration ends, save the trained model and test the win rate of the two.
    # pi0_variables = slim.get_variables(scope = "pi0")
    # pi1_variables = slim.get_variables(scope = "pi1")
    # parameters_to_save_list0 = [v for v in pi0_variables]
    # parameters_to_save_list1 = [v for v in pi1_variables]
    # parameters_to_save_list = parameters_to_save_list0 + parameters_to_save_list1
    # saver = tf.train.Saver(parameters_to_save_list)
    # parameters_path = 'parameter/'
    # tf.train.Saver()
    save_path = saver.save(U.get_session(), "saveparameter/20/20.pkl")