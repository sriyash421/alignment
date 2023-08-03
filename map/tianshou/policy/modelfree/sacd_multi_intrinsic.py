from copy import deepcopy
from typing import Dict, List, Union, Optional, Callable

import numpy as np
import os
import torch
import torch.nn.functional as F

from tianshou.policy import BasePolicy
from tianshou.data import Batch, to_torch_as, ReplayBuffer, to_torch
from tianshou.policy import SACDMultiPolicy

class SACDMultiIntPolicy(BasePolicy):
    """Implementation of Multiagent Soft Actor-Critic.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic1: the first critic network. (s -> Q(s,
        a))
    :param torch.optim.Optimizer critic1_optim: the optimizer for the first
        critic network.
    :param torch.nn.Module critic2: the second critic network. (s -> Q(s,
        a))
    :param torch.optim.Optimizer critic2_optim: the optimizer for the second
        critic network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float alpha: entropy regularization coefficient, default to 0.2.
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.
    :param int estimation_step: the number of estimation step, should be an int
            greater than 0, defaults to 1.
    :param bool grads_logging: whether to log gradients or not

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 actors: List[torch.nn.Module],
                 actor_optims: List[torch.optim.Optimizer],
                 global_critic1s: List[torch.nn.Module],
                 global_critic1_optims: List[torch.optim.Optimizer],
                 global_critic2s: List[torch.nn.Module],
                 global_critic2_optims: List[torch.optim.Optimizer],
                 local_critic1s: List[torch.nn.Module],
                 local_critic1_optims: List[torch.optim.Optimizer],
                 local_critic2s: List[torch.nn.Module],
                 local_critic2_optims: List[torch.optim.Optimizer],
                 int_critic1s: List[torch.nn.Module],
                 int_critic1_optims: List[torch.optim.Optimizer],
                 int_critic2s: List[torch.nn.Module],
                 int_critic2_optims: List[torch.optim.Optimizer],
                 dist_fn: torch.distributions.Distribution
                 = torch.distributions.Categorical,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 grads_logging: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert 0 <= tau <= 1, 'tau should in [0, 1]'
        self._tau = tau
        assert 0 <= gamma <= 1, 'gamma should in [0, 1]'
        self._gamma = gamma
        self._rew_norm = reward_normalization
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step
        self._rm_done = ignore_done
        self.dist_fn = dist_fn

        assert len(actors) == len(global_critic1s) == len(global_critic2s)
        self.actors, self.actor_optims = actors, actor_optims
        
        self.global_critic1s, self.global_critic1_olds = global_critic1s, deepcopy(global_critic1s)
        for global_critic1_old in self.global_critic1_olds:
            global_critic1_old.eval()
        self.global_critic1_optims = global_critic1_optims
        self.global_critic2s, self.global_critic2_olds = global_critic2s, deepcopy(global_critic2s)
        for global_critic2_old in self.global_critic2_olds:
            global_critic2_old.eval()
        self.global_critic2_optims = global_critic2_optims

        self.local_critic1s, self.local_critic1_olds = local_critic1s, deepcopy(local_critic1s)
        for local_critic1_old in self.local_critic1_olds:
            local_critic1_old.eval()
        self.local_critic1_optims = local_critic1_optims
        self.local_critic2s, self.local_critic2_olds = local_critic2s, deepcopy(local_critic2s)
        for local_critic2_old in self.local_critic2_olds:
            local_critic2_old.eval()
        self.local_critic2_optims = local_critic2_optims

        self.int_critic1s, self.int_critic1_olds = int_critic1s, deepcopy(int_critic1s)
        for int_critic1_old in self.int_critic1_olds:
            int_critic1_old.eval()
        self.int_critic1_optims = int_critic1_optims
        self.int_critic2s, self.int_critic2_olds = int_critic2s, deepcopy(int_critic2s)
        for int_critic2_old in self.int_critic2_olds:
            int_critic2_old.eval()
        self.int_critic2_optims = int_critic2_optims

        self._alpha = alpha
        self.__eps = np.finfo(np.float32).eps.item()
        self.model_list = [('actor_', self.actors),
                           ('global_critic1_', self.global_critic1s),
                           ('global_critic2_', self.global_critic2s),
                           ('local_critic1_', self.local_critic1s),
                           ('local_critic2_', self.local_critic2s),
                           ('int_critic1_', self.int_critic1s),
                           ('int_critic2_', self.int_critic2s)]
        self.grads_logging = grads_logging
        self.beta = kwargs['beta']
        self.temp = kwargs['temp']

        self.critic1s = [*self.global_critic1s, *self.local_critic1s, *self.int_critic1s]
        self.critic2s = [*self.global_critic2s, *self.local_critic2s, *self.int_critic2s]
        self.critic1_olds = [*self.global_critic1_olds, *self.local_critic1_olds, *self.int_critic1_olds]
        self.critic2_olds = [*self.global_critic2_olds, *self.local_critic2_olds, *self.int_critic2_olds]
    
    def save(self, logdir, type="best"):
        for prefix, models in self.model_list:
            if type == 'final':
                prefix = 'final_' + prefix
            for i, model in enumerate(models):
                path = os.path.join(logdir, prefix + str(i) + '.pkl')
                torch.save(model.state_dict(), path)
    
    def load(self, logdir, type="best"):
        for prefix, models in self.model_list:
            if type == 'final':
                prefix = 'final_' + prefix
            for i, model in enumerate(models):
                path = os.path.join(logdir, prefix + str(i) + '.pkl')
                model.load_state_dict(torch.load(
                    path, map_location=lambda storage, _: storage))
        self.critic1_olds = deepcopy(self.critic1s)
        for critic1_old in self.critic1_olds:
            critic1_old.eval()
        self.critic2_olds = deepcopy(self.critic2s)
        for critic2_old in self.critic2_olds:
            critic2_old.eval()

    def train(self) -> None:
        self.training = True
        for actor in self.actors:
            actor.train()
        for critic1 in self.critic1s:
            critic1.train()
        for critic2 in self.critic2s:
            critic2.train()

    def eval(self) -> None:
        self.training = False
        for actor in self.actors:
            actor.eval()
        for critic1 in self.critic1s:
            critic1.eval()
        for critic2 in self.critic2s:
            critic2.eval()

    def sync_weight(self) -> None:
        for i in range(len(self.actors)):
            critic1 = self.critic1s[i]
            critic2 = self.critic2s[i]
            critic1_old = self.critic1_olds[i]
            critic2_old = self.critic2_olds[i]
            for o, n in zip(critic1_old.parameters(), critic1.parameters()):
                o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
            for o, n in zip(critic2_old.parameters(), critic2.parameters()):
                o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                input: str = 'obs', **kwargs) -> Batch:
        obs = getattr(batch, input)
        logits = []
        acts = []
        log_probs = []
        for i, actor in enumerate(self.actors):
            logit, h = actor(obs[:, i])
            dist = self.dist_fn(logit)
            act = dist.sample()
            log_prob = torch.log(logit + self.__eps)
            acts.append(act.unsqueeze(1))
            logits.append(logit.unsqueeze(1))
            log_probs.append(log_prob.unsqueeze(1))
        acts = torch.cat(acts, dim=1)
        logits = torch.cat(logits, dim=1)
        log_probs = torch.cat(log_probs, dim=1)
        return Batch(
            logits=logits, act=acts, state=h, dist=logits, log_prob=log_probs)


    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        if self._rm_done:
            batch.done = np.zeros_like(batch.done)

        int_rewards = self.calculate_intrinsic_reward(buffer, indice)
        
        local_targets = self.compute_nstep_return(
            deepcopy(batch), buffer, indice, self._target_local_q,
            self._gamma, self._n_step, self._rew_norm
        )
        global_targets = self.compute_nstep_return(
            deepcopy(batch), buffer, indice, self._target_global_q,
            self._gamma, self._n_step, self._rew_norm
        )
        
        int_batch = deepcopy(batch)
        int_batch.rews = int_rewards 
        int_targets = self.compute_nstep_return(
            deepcopy(batch), buffer, indice, self._target_int_q,
            self._gamma, self._n_step, False
        )

        batch.int_targets = int_targets
        batch.local_targets = local_targets
        batch.global_targets = global_targets
        return batch, {}

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray, critic1_olds, critic2_olds, cc=True) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        with torch.no_grad():
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            batch.act = to_torch_as(batch.act, a_)
            target_qs = []
            for i in range(len(self.actors)):
                critic1_old = critic1_olds[i]
                critic2_old = critic2_olds[i]
                if cc:
                    critic_input = (to_torch(batch.obs_next, device=a_.device), a_)
                else:
                    critic_input = (batch.obs_next[:, i])
                target_q = torch.min(
                        critic1_old(*critic_input),
                        critic2_old(*critic_input),
                ) - self._alpha * obs_next_result.log_prob[:, i]
                target_q = (obs_next_result.dist[:, i] * target_q).sum(
                            dim=1, keepdim=True)
                target_qs.append(target_q)
            target_qs = torch.cat(target_qs, dim=1)
        return target_qs

    def _target_local_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        return self._target_q(buffer, indice, self.local_critic1_olds, self.local_critic2_olds, False)

    def _target_global_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        return self._target_q(buffer, indice, self.global_critic1_olds, self.global_critic2_olds)
    
    def _target_int_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        return self._target_q(buffer, indice, self.int_critic1_olds, self.int_critic2_olds)
    
    def calculate_intrinsic_reward(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs: s_{t+n}
        with torch.no_grad():
            int_rews = []
            obs_next_result = self(batch, input='obs_next')
            a_ = obs_next_result.act
            for i in range(len(self.actors)):
                critic1_old = self.global_critic1_olds[i](to_torch(batch.obs_next, device=a_.device), a_)
                critic2_old = self.global_critic2_olds[i](to_torch(batch.obs_next, device=a_.device), a_)

                local_critic1_old = self.local_critic1_olds[i](batch.obs_next[:, i])
                local_critic2_old = self.local_critic2_olds[i](batch.obs_next[:, i])

                rews1 = torch.exp(-torch.abs(critic1_old-local_critic1_old)/self.temp)
                rews2 = torch.exp(-torch.abs(critic2_old-local_critic2_old)/self.temp)
                rew = torch.min(rews1, rews2)
                int_rews.append(rew)
            int_rews = torch.cat(int_rews, dim=1)
        return int_rews
    
    def train_critic(self, batch, i, critic1, critic2, critic1_optim, critic2_optim):
        # critic 1
        current_q1 = critic1(batch.obs[:, i])
        a_ = to_torch_as(batch.act[:, i], current_q1).long().unsqueeze(1)
        current_q1 = current_q1.gather(1, a_)
        target_q = to_torch_as(batch.returns[:, i], current_q1)[:, None]
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic1_optim.zero_grad()
        critic1_loss.backward()
        critic1_optim.step()

        # critic 2
        current_q2 = critic2(batch.obs[:, i])
        current_q2 = current_q2.gather(1, a_)
        critic2_loss = F.mse_loss(current_q2, target_q)
        critic2_optim.zero_grad()
        critic2_loss.backward()
        critic2_optim.step()

        return critic1_loss, critic2_loss

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        total_actor_loss = np.zeros(len(self.actors))
        total_global_critic1_loss = np.zeros(len(self.global_critic1s))
        total_global_critic2_loss = np.zeros(len(self.global_critic2s))
        total_local_critic1_loss = np.zeros(len(self.local_critic1s))
        total_local_critic2_loss = np.zeros(len(self.local_critic2s))
        total_int_critic1_loss = np.zeros(len(self.int_critic1s))
        total_int_critic2_loss = np.zeros(len(self.int_critic2s))
        total_returns = np.zeros(len(self.actors))

        for i in range(len(self.actors)):
            actor = self.actors[i]
            actor_optim = self.actor_optims[i]

            global_critic1_loss, global_critic2_loss = self.train_critic(batch, i, self.global_critic1s[i], self.global_critic2s[i], self.global_critic1_optims[i], self.global_critic2_optims[i])
            local_critic1_loss, local_critic2_loss = self.train_critic(batch, i, self.local_critic1s[i], self.local_critic2s[i], self.local_critic1_optims[i], self.local_critic2_optims[i])
            int_critic1_loss, int_critic2_loss = self.train_critic(batch, i, self.int_critic1s[i], self.int_critic2s[i], self.int_critic1_optims[i], self.int_critic2_optims[i])

            critic1 = self.global_critic1s[i]
            critic2 = self.global_critic2s[i]
            # actor
            current_q1a = critic1(batch.obs[:, i]) + self.beta * self.int_critic1s[i](batch.obs[:, i])
            current_q2a = critic2(batch.obs[:, i]) + self.beta * self.int_critic2s[i](batch.obs[:, i])
            logit, _ = actor(batch.obs[:, i])
            log_prob = torch.log(logit + self.__eps)
            actor_loss = self._alpha * log_prob - torch.min(
                    current_q1a, current_q2a)
            actor_loss = (logit * actor_loss).sum(dim=1).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            total_actor_loss[i] = actor_loss.item()
            total_global_critic1_loss[i] = global_critic1_loss.item()
            total_global_critic2_loss[i] = global_critic2_loss.item()
            total_local_critic1_loss[i] = local_critic1_loss.item()
            total_local_critic2_loss[i] = local_critic2_loss.item()
            total_int_critic1_loss[i] = int_critic1_loss.item()
            total_int_critic2_loss[i] = int_critic2_loss.item()
            total_returns[i] = batch.returns[:, i].mean()
        self.sync_weight()

        output = {}
        for i in range(len(self.actors)):
            output[f'models/actor_{i}'] = total_actor_loss[i]
            output[f'models/global_critic1_{i}'] = total_global_critic1_loss[i]
            output[f'models/global_critic2_{i}'] = total_global_critic2_loss[i]
            output[f'models/local_critic1_{i}'] = total_local_critic1_loss[i]
            output[f'models/local_critic2_{i}'] = total_local_critic2_loss[i]
            output[f'models/int_critic1_{i}'] = total_int_critic1_loss[i]
            output[f'models/int_critic2_{i}'] = total_int_critic2_loss[i]
            output[f'models/returns_{i}'] = total_returns[i]
            if self.grads_logging:
                for tag, value in self.actors[i].named_parameters():
                    if value.grad is not None:
                        output[f'grads/actor_{i}_{tag}'] = value.grad
            
                # for (tag1, value1), (tag2, value2) in zip(self.critic1s[i].named_parameters(), self.critic2s[i].named_parameters()):
                #     if value1.grad is not None:
                #         output[f'grads/critic1_{i}_{tag1}'] = value1.grad
                #     if value2.grad is not None:
                #         output[f'grads/critic2_{i}_{tag2}'] = value2.grad

        output[f'loss/actor'] = total_actor_loss.sum()
        output[f'loss/critic1'] = total_global_critic1_loss.sum()
        output[f'loss/critic2'] = total_global_critic2_loss.sum()

        return output
