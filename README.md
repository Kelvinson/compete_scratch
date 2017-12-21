# compete_scratch
try to implement the training phase code for two agents in the competitive environment in the paper paper [Emergent Complexity via Multi-agent Competition](https://arxiv.org/abs/1710.03748) using PPO algorithm and it's imcomplete.

I imported the competitive environment from OpenAI's repo and revised the ppo algorithm from the baseline to adapt to the competitive environment. As you can see, the dependencies are the dependencies of the two repo above. baseline etc have to be installed at first. Now there is a problem of the old policy network's state_in placeholder.


- run.py: the main program of training, you can specify the environment and the policy network used in the training as you like.

- policy.py: the policy network

- learn.py: learn and update the network. Specifically, intialize all the relevant placeholders fo the network and 
the intermediate variables in the PPO algorithm. Then call the seg_gen function which uses a generator to produce a spefic amount of 
timesteps of running data(in a dict)when called once. Next call the add_vtarg_and_adv function to calculate the target value and advantage
to the dict of data. (see the paper for details of the calculating process). Then uses Adam optimizer to apply the gradients to 
the networks. At the end of the program. Save the parameters of the policy(and value prediction) networks in this iteration to a list.
After some anther iterations. Assign the paramters of a random iteration( of latter half iterations) to the opponent agent. Do this in iteratively 
until any one of max_steps, max_episodes, max_iters, max_seconds exceeds the threshold.  

- load_parameter_run.py: load the trained paramters of the action and value network of the two agents
and run in the multiagent competitive environment. Print out the rewards for the two
agents in every step, add the win rate if you like.

- policy.py: contains the policy network definition as used in the original paper. Specifically, LSTM and MLP policy are used. You can define
your policy network if you like. 

- utils.py: trajectory geneator function and  advantage generator function. In other words, it is used to produce the training data of a specific 
timesteps (full rollout in the self play paper). A generator generator a dict of such data every time when called.

- parameters: this folder contains some pretrained paramter I have trained for 200 rollouts(actually not the rollouts as in the paper)
saveparameters: parameters produced with early versions of the training program. The result is very bad.
 
## TO DO
distributed_data: reimplement the utils.py to produce the rollout data in multithreads, ideally to be the same as
in the paper says: **run on 4 GPUs.Each iteration, we collect 409600 samples from the parallel rollouts and perform multiple epochs of PPO training in mini-batches consisting of 5120 samples.** 