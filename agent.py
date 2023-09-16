import argparse
from collections import deque
import logging
import random
import timeit

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np

import gymnasium as gym

from ale_py import ALEInterface

ale = ALEInterface()

import utils


torch.set_num_threads(4)


logging.basicConfig(format=(
        "[%(levelname)s:%(asctime)s] " "%(message)s"), level=logging.INFO)


class DQN(nn.Module):
    def __init__(self, naction, args):
        super().__init__()
        self.iH, self.iW, self.iC = 210, 160, 3
        self.conv1 = nn.Conv2d(self.iC, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # the flattened size is 8960 assuming dims and convs above
        self.fc1 = nn.Linear(8960, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, naction)

        self.epsilon = args.epsilon_start

    def forward(self, X):
        """
        X - bsz x T x iC x iH x iW observations (in order)
        returns:
          bsz x T x naction action q_values
        """
        bsz, T = X.size()[:2]

        Z = F.relu(self.conv3( # bsz*T x hidden_dim x H3 x W3
              F.relu(self.conv2(
                F.relu(self.conv1(X.view(-1, self.iC, self.iH, self.iW)))))))

        # flatten with MLP
        Z = F.relu(self.fc1(Z.view(bsz*T, -1))) # bsz*T x hidden_dim
        Z = Z.view(bsz, T, -1)

        return self.fc2(Z)

    def get_action(self, x, prev_state):
        """
        x - 1 x 1 x ic x iH x iW
        returns:
          int index of action
        """
        q_values = self(x)
        # take highest scoring action
        action = q_values.argmax(-1).squeeze().item()

        if random.uniform(0, 1) < self.epsilon:
            # Choose a random action
            action = Categorical(logits=q_values).sample()

        return action, prev_state


def dqn_step(stepidx, q_network, target_network, buffer, optimizer, scheduler, envs, observations, bsz=4, mbsz=32):
    if envs is None:
        envs = [gym.make(args.env) for _ in range(bsz)]
        observations = [env.reset(seed=i)[0] for i, env in enumerate(envs)]
        observations = torch.stack( # bsz x ic x iH x iW -> bsz x 1 x ic x iH x iW
            [utils.preprocess_observation(obs) for obs in observations]).unsqueeze(1)

    not_terminated = torch.ones(bsz) # agent is still alive

    # for each time step
    for t in range(args.unroll_length):

        # set epsilon to be maximum of minimum possible epsilon value and epsilon found through epsilon decay
        q_network.epsilon = max(args.epsilon_end, args.epsilon_start - ((stepidx + t) / args.epsilon_decay))

        # append q_values for the time step to the q_values for the batch
        q_values_t = q_network(observations) # q_values are bsz x 1 x naction

        # append sample actions for the time step to sample actions for the batch
        actions_t = [epsilon_greedy(q_values_t_b, q_network.epsilon) for q_values_t_b in q_values_t.squeeze(1)]

        # get outputs for each env, which are (observation, reward, terminated, truncated, info) tuples
        env_outputs = [env.step(actions_t[b].item()) for b, env in enumerate(envs)]
        next_obs = torch.stack([utils.preprocess_observation(eo[0]) for eo in env_outputs]).unsqueeze(1)
        rewards_t = torch.tensor([eo[1] for eo in env_outputs])
        dones_t = torch.tensor([eo[2] for eo in env_outputs])

        # if we lose a life, zero out all subsequent rewards
        still_alive = torch.tensor([env.ale.lives() == args.start_nlives for env in envs])
        not_terminated.mul_(still_alive.float())

        # add to buffer
        for _ in range(bsz):
            buffer.append([observations[_], actions_t[_], rewards_t[_], next_obs[_], dones_t[_]])

        # set observations equal to the next observations
        observations = next_obs

    # selects the mini batch
    mini_batch = random.sample(buffer, mbsz)
    obs, actions, rewards, next_obs, dones = zip(*mini_batch)
    obs = torch.stack(list(obs), dim=0)
    actions = torch.stack(list(actions), dim=0)
    rewards = torch.stack(list(rewards), dim=0)
    next_obs = torch.stack(list(next_obs), dim=0)
    dones = torch.stack(list(dones), dim=0)

    # finds the expected q-values of the mini batch according to the current q network and current observations
    expected_q_values, _ = torch.max(q_network(obs).squeeze(1), dim=1)

    # calculates the target q-values for the next observations by adding the rewards to the
    next_q_values, _ = torch.max(target_network(next_obs).squeeze(1), dim=1)
    target_q_values = rewards + (~dones) * args.discounting * next_q_values

    mse = F.mse_loss(expected_q_values, target_q_values, reduction='none')
    dqn_loss = (mse.view_as(actions)).mean()
    total_loss = dqn_loss

    stats = {"mean_return": sum(r.mean() for r in rewards)/mbsz,
             "dqn_loss": total_loss.item()}

    total_loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), args.grad_norm_clipping)
    optimizer.step()
    scheduler.step()

    # reset any environments that have ended
    for b in range(bsz):
        if not_terminated[b].item() == 0:
            obs = envs[b].reset(seed=stepidx+b)[0]
            observations[b].copy_(utils.preprocess_observation(obs))

    return stats, envs, observations


def epsilon_greedy(action_values, epsilon):
    if random.uniform(0, 1) < epsilon:
        # Choose a random action
        action = Categorical(logits=action_values).sample()
    else:
        # Choose the action with the highest Q-value
        action = torch.argmax(action_values)

    return action



def train(args):
    T = args.unroll_length
    B = args.batch_size
    N = args.mini_batch_size

    args.device = torch.device("cpu")
    env = gym.make(args.env)
    naction = env.action_space.n
    args.start_nlives = env.ale.lives()
    del env

    q_network = DQN(naction, args)
    target_network = DQN(naction, args)
    target_network.load_state_dict(q_network.state_dict())
    buffer = deque(maxlen=args.buffer_cap)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=args.learning_rate)

    def lr_lambda(epoch): # multiplies learning rate by value returned; can be used to decay lr
        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def checkpoint():
        if args.save_path is None:
            return
        logging.info("Saving checkpoint to {}".format(args.save_path))
        torch.save({"q_network_state_dict": q_network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": args}, args.save_path)


    timer = timeit.default_timer
    last_checkpoint_time = timer()
    envs, observations = None, None
    frame = 0
    while frame < args.total_frames:
        start_time = timer()
        start_frame = frame
        stats, envs, observations = dqn_step(
            frame, q_network, target_network, buffer, optimizer, scheduler, envs, observations, bsz=B, mbsz=N)
        frame += T*B # here steps means number of observations
        if timer() - last_checkpoint_time > args.min_to_save * 60:
            checkpoint()
            last_checkpoint_time = timer()

        sps = (frame - start_frame) / (timer() - start_time)
        logging.info("Frame {:d} @ {:.1f} FPS: dqn_loss {:.3f} | mean_ret {:.3f}".format(
          frame, sps, stats['dqn_loss'], stats["mean_return"]))

        if frame > 0 and frame % (args.eval_every*T*B) == 0:
            utils.validate(q_network, render=args.render)
            q_network.train()

        # update target network
        if frame % args.target_network_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())


parser = argparse.ArgumentParser()

parser.add_argument("--env", type=str, default="ALE/MsPacman-v5", help="gym environment")
parser.add_argument("--mode", default="train", choices=["train", "valid",],
                    help="training or validation mode")
parser.add_argument("--total_frames", default=1000000, type=int,
                    help="total environment frames to train for")
parser.add_argument("--buffer_cap", default=10000, type=int, help="learner buffer cap.")
parser.add_argument("--batch_size", default=8, type=int, help="learner batch size.")
parser.add_argument("--mini_batch_size", default=32, type=int, help="sample batch size.")
parser.add_argument("--unroll_length", default=80, type=int,
                    help="unroll length (time dimension)")
parser.add_argument("--hidden_dim", default=256, type=int, help="policy net hidden dim")
parser.add_argument("--discounting", default=0.99, type=float, help="discounting factor")
parser.add_argument("--epsilon_start", default=1, type=float, help="initial probability of selecting a random action during exploration at start")
parser.add_argument("--epsilon_end", default=0.01, type=float, help="final probability of selecting a random action during exploration")
parser.add_argument("--epsilon_decay", default=200, type=float, help="final probability of selecting a random action during exploration")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--target_network_update_freq", default=250, type=int, help="replacement rate of target weights")
parser.add_argument("--grad_norm_clipping", default=10.0, type=float,
                    help="Global gradient norm clip.")
parser.add_argument("--save_path", type=str, default=None, help="save q_network here")
parser.add_argument("--load_path", type=str, default=None, help="load q_network from here")
parser.add_argument("--min_to_save", default=1, type=int, help="save every this many minutes")
parser.add_argument("--eval_every", default=50, type=int, help="eval every this many updates")
parser.add_argument("--render", action="store_true", help="render game-play at validation time")


if __name__ == "__main__":
    torch.manual_seed(59006)
    np.random.seed(59006)
    args = parser.parse_args()
    logging.info(args)
    if args.mode == "train":
        train(args)
    else:
        assert args.load_path is not None
        checkpoint = torch.load(args.load_path)
        saved_args = checkpoint["args"]
        env = gym.make(args.env)
        naction = env.action_space.n
        saved_args.start_nlives = env.ale.lives()
        del env

        q_network = DQN(naction, saved_args)
        q_network.load_state_dict(checkpoint["q_network_state_dict"])
        q_network = q_network
        args = saved_args

        utils.validate(q_network, args)