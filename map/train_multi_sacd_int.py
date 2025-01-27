from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import json
import os
import torch
import pprint
import argparse
import wandb
import numpy as np

from models.sac_nets import Actor, Critic, CentralizedCritic
from tianshou.data import Collector, ReplayBuffer
from tianshou.env.multiagent.multi_discrete import MultiDiscrete
from tianshou.env import (
    make_multiagent_env,
    VectorEnv,
    BaseRewardLogger,
    SimpleTagRewardLogger,
    SimpleTagBenchmarkLogger,
    SimpleAdversaryBenchmarkLogger,
    SimpleSpreadBenchmarkLogger,
)
from tianshou.policy import (
    SACDMultiIntPolicy
)
from tianshou.trainer import offpolicy_trainer


def get_args():
    parser = argparse.ArgumentParser()

    # State arguments.
    parser.add_argument('--task', type=str, default='simple_spread_in')
    parser.add_argument('--save-video', action='store_true', default=False)
    parser.add_argument('--save-models', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--video-file', type=str, default='videos/simple.mp4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Training arguments
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=4000)
    parser.add_argument('--collect-per-step', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--training-num', type=int, default=20)
    parser.add_argument('--test-num', type=int, default=100)

    # Model arguments
    parser.add_argument('--layer-num', type=int, default=2)

    # SAC special
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--ignore-done', type=int, default=0)
    parser.add_argument('--n-step', type=int, default=1)

    # Task specific
    parser.add_argument('--num-good-agents', type=int, default=0)
    parser.add_argument('--num-adversaries', type=int, default=0)
    parser.add_argument('--obs-radius', type=float, default=float('inf'))
    parser.add_argument('--amb-init', type=int, default=0)
    parser.add_argument('--rew-shape', action='store_true', default=False)

    # Enable wandb logging or not
    parser.add_argument('--wandb-enabled', action='store_true', default=False)

    # Enable grads logging or not
    parser.add_argument('--grads-logging', action='store_true', default=False)

    # Specify the intrinsic reward type or no intrinsic reward
    # options include ['no', 'elign_self', 'elign_team', 'elign_adv', 'elign_both', 'curio_self', 'curio_team']
    parser.add_argument('--intr-rew', type=str, default='intrinsic')
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--temp', type=float, default=1)

    args = parser.parse_known_args()[0]
    return args


def train_multi_sacd(args=get_args()):
    wandb_dir = '/home/sriyash/scratch/ELIGN_FINAL/'
# if torch.cuda.is_available() else 'log/'
    if args.wandb_enabled:
        wandb.init(project='ELIGN-LOG', entity='sriyash-mila', dir=wandb_dir, config=args, mode="offline",  sync_tensorboard=True)
    torch.set_num_threads(4)  # 1 for poor CPU
    task_params = {'num_good_agents': args.num_good_agents,
                   'num_adversaries': args.num_adversaries,
                   'obs_radius': args.obs_radius,
                   'amb_init': args.amb_init,
                   'rew_shape': args.rew_shape}
    env = make_multiagent_env(
        args.task, benchmark=args.benchmark, optional=task_params)
    num_agents = len(env.world.agents)
    args.state_shape = (env.observation_space[0].shape or
                        env.observation_space[0].n)
    args.action_shape = (env.action_space[0].shape or
                         env.action_space[0].n)
    # to account for setups where each agent might have a different action space
    action_space_n = []
    for act_space in env.action_space:
        if isinstance(act_space, MultiDiscrete):
            total_dim = np.sum(act_space.high - act_space.low + 1)
        else:
            total_dim = act_space.n
        action_space = max(total_dim, max(action_space_n)
                           if len(action_space_n) > 0 else total_dim)
        action_space_n.append(action_space)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = VectorEnv(
        [lambda: make_multiagent_env(args.task, benchmark=args.benchmark, optional=task_params)
            for _ in range(args.training_num)])
    test_envs = VectorEnv(
        [lambda: make_multiagent_env(args.task, benchmark=args.benchmark, optional=task_params)
            for _ in range(args.test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    actors = [Actor(args.layer_num, args.state_shape, action_space_n[i],
                    softmax=True, device=args.device).to(args.device)
              for i in range(num_agents)]
    global_critic1s = [CentralizedCritic(args.layer_num, args.state_shape, action_space_n[i], num_agents,
                                  device=args.device).to(args.device)
                for i in range(num_agents)]
    
    global_critic2s = [CentralizedCritic(args.layer_num, args.state_shape, action_space_n[i], num_agents,
                                  device=args.device).to(args.device)
                for i in range(num_agents)]
    
    int_critic1s = [CentralizedCritic(args.layer_num, args.state_shape, action_space_n[i], num_agents,
                                  device=args.device).to(args.device)
                for i in range(num_agents)]
    
    int_critic2s = [CentralizedCritic(args.layer_num, args.state_shape, action_space_n[i], num_agents,
                                  device=args.device).to(args.device)
                for i in range(num_agents)]

    local_critic1s = [Critic(args.layer_num, args.state_shape, action_space_n[i], device=args.device).to(args.device)
                for i in range(num_agents)]
    local_critic2s = [Critic(args.layer_num, args.state_shape, action_space_n[i], device=args.device).to(args.device)
                for i in range(num_agents)]

    # Optimizers.
    actor_optims = [Adam(actor.parameters(), lr=args.actor_lr)
                    for actor in actors]
    global_critic1_optims = [Adam(global_critic1.parameters(), lr=args.critic_lr)
                      for global_critic1 in global_critic1s]
    global_critic2_optims = [Adam(global_critic2.parameters(), lr=args.critic_lr)
                      for global_critic2 in global_critic2s]
    local_critic1_optims = [Adam(local_critic1.parameters(), lr=args.critic_lr)
                      for local_critic1 in local_critic1s]
    local_critic2_optims = [Adam(local_critic2.parameters(), lr=args.critic_lr)
                      for local_critic2 in local_critic2s]
    int_critic1_optims = [Adam(int_critic1.parameters(), lr=args.critic_lr)
                      for int_critic1 in int_critic1s]
    int_critic2_optims = [Adam(int_critic2.parameters(), lr=args.critic_lr)
                      for int_critic2 in int_critic2s]

    # Policy
    dist = torch.distributions.Categorical
    policy = SACDMultiIntPolicy(
        actors, actor_optims,
        global_critic1s, global_critic1_optims,
        global_critic2s, global_critic2_optims,
        local_critic1s, local_critic1_optims,
        local_critic2s, local_critic2_optims,
        int_critic1s, int_critic1_optims,
        int_critic2s, int_critic2_optims,
        dist, args.tau, args.gamma, args.alpha,
        reward_normalization=args.rew_norm,
        ignore_done=args.ignore_done,
        estimation_step=args.n_step,
        grads_logging=args.grads_logging,
	beta=args.beta, temp=args.temp
    )

   # Load existing models if checkpoint is specified.
    if args.checkpoint:
        policy.load(args.checkpoint)

    # Setup the reward loggers
    competitive_tasks = ['simple_tag_in',
                         'simple_adversary_in', 'simple_push_in']
    if args.task in competitive_tasks:
        def reward_logger(): return SimpleTagRewardLogger(len(
            [a for a in env.world.agents if a.adversary]))
    else:
        reward_logger = BaseRewardLogger

    # Setup the benchmark loggers.
    train_benchmark_logger = None
    test_benchmark_logger = None
    vis_benchmark_logger = None
    if args.benchmark:
        if 'simple_tag' in args.task:
            train_benchmark_logger = SimpleTagBenchmarkLogger(
                args.training_num, len(
                    [a for a in env.world.agents if a.adversary]),
                max_world_steps=env.world.max_steps)
            test_benchmark_logger = SimpleTagBenchmarkLogger(
                args.test_num, len(
                    [a for a in env.world.agents if a.adversary]),
                max_world_steps=env.world.max_steps)
            vis_benchmark_logger = SimpleTagBenchmarkLogger(
                1, len([a for a in env.world.agents if a.adversary]),
                max_world_steps=env.world.max_steps)
        elif 'simple_adversary' in args.task or 'simple_push' in args.task:
            train_benchmark_logger = SimpleAdversaryBenchmarkLogger(
                args.training_num, len(
                    [a for a in env.world.agents if a.adversary]),
                max_world_steps=env.world.max_steps)
            test_benchmark_logger = SimpleAdversaryBenchmarkLogger(
                args.test_num, len(
                    [a for a in env.world.agents if a.adversary]),
                max_world_steps=env.world.max_steps)
            vis_benchmark_logger = SimpleAdversaryBenchmarkLogger(
                1, len([a for a in env.world.agents if a.adversary]),
                max_world_steps=env.world.max_steps)
        elif 'spread' in args.task:
            train_benchmark_logger = SimpleSpreadBenchmarkLogger(
                args.training_num, max_world_steps=env.world.max_steps)
            test_benchmark_logger = SimpleSpreadBenchmarkLogger(args.test_num,
                                                                max_world_steps=env.world.max_steps)
            vis_benchmark_logger = SimpleSpreadBenchmarkLogger(1,
                                                               max_world_steps=env.world.max_steps)
        else:
            raise NotImplementedError

    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size),
        num_agents=num_agents,
        reward_logger=reward_logger,
        benchmark_logger=train_benchmark_logger)
    test_collector = Collector(
        policy, test_envs,
        num_agents=num_agents,
        reward_logger=reward_logger,
        benchmark_logger=test_benchmark_logger)

    # log
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    writer = SummaryWriter(args.logdir)

    def save_fn(): return policy.save(args.logdir) if args.save_models else None
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, save_fn=save_fn, writer=writer)
    train_collector.close()
    test_collector.close()

    if args.save_models:
        policy.save(args.logdir, type='final')

    if args.save_video:
        pprint.pprint(result)
        # Let's watch its performance!
        env = make_multiagent_env(
            args.task, benchmark=args.benchmark, optional=task_params)
        # Change max steps for a longer visualization
        env.world.max_steps = 1000
        collector = Collector(policy, env, num_agents=num_agents,
                              reward_logger=reward_logger,
                              benchmark_logger=vis_benchmark_logger)
        result = collector.collect(
            n_episode=1, render=args.render, render_mode='rgb_array')
        from tianshou.env import create_video
        create_video(result['frames'], args.video_file)
        print('Final reward: ', result["rew"])
        collector.close()


if __name__ == '__main__':
    train_multi_sacd()
