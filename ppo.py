import gym
import numpy as np
from torchsummary import summary
from torch import nn, optim
import torch
import glob, datetime, sys
# from utils import Tee
import itertools
from collections import deque
import time, os

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/main.py
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage

from baselines.bench import Monitor

from .base import RLAlgorithm
from .utils import make_vec_envs_custom

class PPO(RLAlgorithm):

    constants = {
        "env": None,
        "clear_logs": True,
        "log_interval": 10,
        "episode_r_deque_size": 10,
        "seed": 1,
        "num_processes": 1,
        "gamma": 0.99,
        "log_dir": '/tmp/gym',
        "clip_param": 0.1,
        "ppo_epoch": 4,
        "minibatch_size": 4,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "lr": 2.5e-4,
        "eps": 1e-5,
        "max_grad_norm": 0.5,
        "forward_steps": 200,
        "num_env_steps": 1e6,
        "episodes_per_epoch": np.inf,
        "use_gae": True,
        "gae_lambda": 0.95,
        "use_proper_time_limits": False,
        "recurrent_policy": False,
    }

    def __init__(self, new_constants={}):
        # Update constants
        assert(type(new_constants) == dict)
        for c in new_constants.keys():
            if c in self.constants.keys():
                self.constants[c] = new_constants[c]
        # Clear existing logs
        if self.constants["clear_logs"]:
            for fname in glob.glob("*_log_*.txt"):
                os.remove(fname)
                print('Removed: %s' % fname)
        # set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(self.constants["seed"])
        torch.cuda.manual_seed_all(self.constants["seed"])
        # construct envs
        args = self.args_k("seed", "num_processes", "gamma", "log_dir")
        assert(self.constants["env"] != None)
        if type(self.constants["env"]) == str: # Gym env
            self.envs = make_vec_envs(self.constants["env"], *args, self.device, False)
        else: # Custom env (TODO: multi-process?)
            self.envs = make_vec_envs_custom(self.constants, self.device, self.constants["env"])
        # construct actor critic
        env_args = (self.envs.observation_space.shape, self.envs.action_space)
        self.actor_critic = Policy(*env_args, base_kwargs={'recurrent': self.constants["recurrent_policy"]}).to(self.device)
        # construct PPO
        args = self.args_k("clip_param", "ppo_epoch", "minibatch_size", "value_loss_coef", "entropy_coef")
        kwargs = self.kwargs_k("lr", "eps", "max_grad_norm")
        self.agent = algo.PPO(self.actor_critic, *args, **kwargs)
        # rollout storage / experiences
        args = self.args_k("forward_steps", "num_processes")
        self.rollouts = RolloutStorage(*args, *env_args, self.actor_critic.recurrent_hidden_state_size)

    def log(self, fname):
        self.f = open('%s.txt' % fname, 'w')

    def epoch_setup(self):
        obs = self.envs.reset()
        self.rollouts.obs.copy_(obs)
        self.rollouts.to(device)
        self.episode_rewards = deque(maxlen=self.constants["episode_r_deque_size"])
        self.current_episode_rewards = []
        self.num_episodes = 0
        self.start = time.time()
        self.num_updates = int(self.constants["num_env_steps"]) // self.constants("forward_steps") // self.constants("num_processes")

    def epoch(self):
        for j in range(self.num_updates):
            # decrease learning rate linearly
            utils.update_linear_schedule(
                self.agent.optimizer, j, self.num_updates, self.constants["lr"])
            # forward steps
            for step in range(self.constants["forward_steps"]):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])
                # Observe reward and next obs
                obs, reward, done, infos = self.envs.step(action)
                reward = self.modify_reward(reward) # This should be extended in outside files.
                self.current_episode_rewards.append(reward)
                # Termination
                if done:
                    self.episode_rewards.append(sum(self.current_episode_rewards))
                    self.current_episode_rewards.clear()
                    self.num_episodes += 1
                    self.envs.reset()
                # Max number of episodes
                if num_episodes >= self.constants["episodes_per_epoch"]: return
                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                # Add experience
                self.rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()
            # compute returns, update agent
            rolloutargs = self.args_k("use_gae", "gamma", "gae_lambda", "use_proper_time_limits")
            self.rollouts.compute_returns(next_value, *rolloutargs)
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
            self.rollouts.after_update()
            # TODO: save model here
            if j % self.constants["log_interval"] == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * self.constants["num_processes"] * self.constants["forward_steps"]
                self.end = time.time()
                if total_num_steps >= 1000 and total_num_steps < 1000000:
                    tot_str = "%gk" % round(total_num_steps/1000.0, 1)
                elif total_num_steps >= 1000000:
                    tot_str = "%gM" % round(total_num_steps/1000.0, 3)
                else:
                    tot_str = "%g" % total_num_steps
                time_s = int(end-start)
                if time_s < 60:
                    time_str = "%gs" % time_s
                elif time_s < 60*60:
                    time_str = "%gmin%ss" % (int(time_s/60.0), time_s%60)
                else:
                    time_str = "%gh%gmin%gs" % (int(time_s/3600.0), int((time_s%3600)/60.0), time_s%60)
                # Construct summary
                summary_str = "MeanR: %.2f\t MedR: %.2f\t MinR: %.2f\t MaxR: %.2f\t Entr:%.2f\t VLoss:%.2f\t ALoss:%.2f\t Steps:%s\t Time:%s\t\n" % (
                    np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards), dist_entropy, 
                    value_loss, action_loss, tot_str, time_str))
                if hasattr(self, 'f'):
                    self.f.write(summary_str)
                    self.f.flush()
                print(summary_str)
                # Max number of episodes
                if num_episodes >= self.constants["episodes_per_epoch"]: return

        # f.close()
        print('Time:%ss\n' % (time.time()-start))