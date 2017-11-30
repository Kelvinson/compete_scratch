# compete_scratch
try to implement the training phase code for two agents in the competitive environment in the paper paper [Emergent Complexity via Multi-agent Competition](https://arxiv.org/abs/1710.03748) using PPO algorithm and it's imcomplete.

I imported the competitive environment from OpenAI's repo and revised the ppo algorithm from the baseline to adapt to the competitive environment. As you can see, the dependencies are the dependencies of the two repo above. baseline etc have to be installed at first. Now there is a problem of the old policy network's state_in placeholder.

The error is as follows:
<img src="https://github.com/Kelvinson/compete_scratch/error.png">
<img src="https://github.com/Kelvinson/compete_scratch/srcode.png">
