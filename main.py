#!/usr/bin/env python3
import argparse
from itertools import count
import gym
import scipy.optimize
import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

def main():
    args = parse_arg()
    torch.manual_seed(args.seed)

    env = gym.make(args.env_name)
    env.seed(args.seed)
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)

    policy_net = Policy(num_inputs, num_actions)
    value_net = Value(num_inputs)

    n_update = 10
    for i_update in range(n_update):
        memory = Memory(); num_steps = 0; reward_batch = 0; num_episodes = 0

        while num_steps < args.batch_size:
            state = env.reset()
            state = running_state(state)

            reward_sum = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(policy_net, state)
                action = action.data[0].numpy()
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                next_state = running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                memory.push(state, np.array([action]), mask, next_state, reward)

                if args.render:
                    env.render()
                if done:
                    break

                state = next_state
            num_steps += (t-1)
            num_episodes += 1
            reward_batch += reward_sum

        batch = memory.sample()
        update_params(policy_net, value_net, batch, args)

        reward_batch /= num_episodes
        if (i_update==0) or (i_update % args.log_interval == 0) or (i_update==n_update-1):
            print('===== update {} ====='.format(i_update))
            print('Last episode return= {}\tAverage return= {:.2f}'.format(reward_sum, reward_batch))

def select_action(policy_net, state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(policy_net, value_net, batch, args):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0; prev_value = 0; prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))
        targets = Variable(returns)
        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        l2_reg = 1e-3
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))

        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

def parse_arg():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    parser.add_argument('--env-name', default="Reacher-v2",help='name of the environment to run')
    parser.add_argument('--batch-size', type=int, default=2000, help='batch-size')
    parser.add_argument('--gamma', type=float, default=0.995, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.97, help='gae')
    parser.add_argument('--max-kl', type=float, default=1e-2, help='max kl value')
    parser.add_argument('--damping', type=float, default=1e-1, help='damping')
    parser.add_argument('--log-interval', type=int, default=1, help='interval between training status logs')
    parser.add_argument('--render', action='store_true', help='render the environment')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
