import numpy as np
import itertools
from scipy.stats import norm

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import itertools
from scipy.stats import norm
from tqdm import tqdm

import argparse

from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--dim", default=4, type=int)
parser.add_argument("--side", default=20, type=int)
parser.add_argument("--name", default='base', type=str)

parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--hidden_size", default=256, type=int)

parser.add_argument("--reg_coef", default=0.001, type=float) # state flow regularization coefficient lambda

parser.add_argument("--loss", default='DB', type=str) # DB / StableDB
parser.add_argument("--loss_scale", default='LogFlow', type=str) # LogFlow / Flow
parser.add_argument("--stable_db_eps", default=1.0, type=float)
parser.add_argument("--stable_db_eta", default=0.001, type=float)

parser.add_argument("--save_dir", default='results', type=str)

# we fix uniform distribution over first state PF(s1 | s0)
def get_uniform_initial_state(args):
    return torch.randint(high=args.side, size=(args.dim,), dtype=torch.int32)

# standard reward
def Reward(args, state):
    ax = abs(state / (args.side - 1) - 0.5)
    return (ax > 0.25).prod(-1) * 0.5 + ((ax < 0.4) * (ax > 0.3)).prod(-1) * 2 + 1e-3

# MLP network takes ohe of states as input
def ohe(args, state):
    state_ohe = torch.zeros(args.dim * args.side, dtype=torch.float32)
    state_ohe[torch.arange(args.dim) * args.side + state] = 1.0
    return state_ohe

def batch_ohe(args, states):
    return torch.stack([ohe(args, state) for state in states])

# compute normalized reward distribution
def compute_true_distribution(args):
    probs = np.zeros((args.side,) * args.dim)
        
    for state in itertools.product(list(range(args.side)), repeat=args.dim):
        state = np.array(state)
        probs[tuple(int(v) for v in state)] = Reward(args, state)

    true_Z = probs.sum()
    return torch.tensor(probs / true_Z), true_Z

# L1 difference between true distribution and empirical distribution of samples
def compute_empirical_distribution_error(true_dist, samples):
    empirical_dist = true_dist * 0.0
    for samp in samples:
        empirical_dist[tuple(int(v) for v in samp)] += 1.0
    empirical_dist /= empirical_dist.sum()
    l1 = torch.abs(empirical_dist - true_dist).sum().item()
    return l1

def process_logits(args, states, logits):        
    # we mask invalid actions: ones that make us exit the grid limits
    mask_minus = (states == 0)
    mask_plus = (states == args.side - 1)

    forward_logits = logits[:, :args.dim + args.dim + 1]
    mask = torch.hstack((mask_minus, mask_plus, torch.zeros(logits.shape[0], 1).bool()))
    forward_logits[mask] = -float("inf")

    backward_logits = logits[:, args.dim + args.dim + 1:-1]
    mask = torch.hstack((mask_plus, mask_minus, torch.zeros(logits.shape[0], 1).bool()))
    backward_logits[mask] = -float("inf")
    
    # compute log forward policy and log flows
    log_pfs = torch.nn.functional.log_softmax(forward_logits, dim=-1)
    log_flows = logits[:, -1]
    
    # compute log backward policy 
    log_pbs = torch.nn.functional.log_softmax(backward_logits, dim=-1)

    return log_pfs, log_pbs, log_flows
    

def train(args, model, opt, log_Z, log_Z_opt, true_distribution, batch_size=16, steps=10000):
    all_samples = []
    l1_history = []
    traj_len_history = []
    mean_traj_len_history = []
    log_Z_history = []

    reg_coef = args.reg_coef
    
    for it in tqdm(range(steps)):
        opt.zero_grad()
        log_Z_opt.zero_grad()

        # we fix uniform distribution over first state PF(s1 | s0)
        states = torch.stack([get_uniform_initial_state(args) for i in range(batch_size)]).type(torch.int32)
        dones = torch.tensor([False] * batch_size)
        loss = 0.0
        total = 0
        logits = None

        logits = model(batch_ohe(args, states).detach())
        log_pfs, log_pbs, log_flows = process_logits(args, states, logits)
        backward_stop_log_probs = log_pbs[:, -1]

        # - args.dim * np.log(args.side) corresponds to log PF(s1 | s0) = log 1 / number of states in grid
        if args.loss_flow_scale == "LogFlow":
            forward_flow = torch.zeros_like(log_flows) + log_Z.sum() - args.dim * np.log(args.side)
            backward_flow = log_flows + backward_stop_log_probs
        elif args.loss_flow_scale == "Flow":
            forward_flow = torch.exp(torch.zeros_like(log_flows) + log_Z.sum() - args.dim * np.log(args.side))
            backward_flow = torch.exp(log_flows + backward_stop_log_probs)

        if args.loss == "DB":
            step_losses = (forward_flow - backward_flow) ** 2
        elif args.loss == "StableDB":
            step_losses = torch.log1p(args.stable_db_eps * ((forward_flow - backward_flow) ** 2))
            step_losses *= 1 + args.stable_db_eta * torch.exp(log_flows) 

        step_losses += reg_coef*torch.exp(log_flows)
        loss += step_losses.sum()
        total += batch_size
        
        # we sample trajectories and compute loss simultaneously
        while not all(dones):
            with torch.no_grad():
                actions = Categorical(logits=log_pfs.detach()).sample()
                
            # update states and compute GFlowNet rewards
            log_rewards = torch.zeros(batch_size)
            dones_new = torch.clone(dones)
            for i in range(batch_size):
                if dones[i]:
                    continue
                if actions[i] == args.dim + args.dim:
                    # store the flag for trajectories in the batch that are already terminated
                    dones_new[i] = True
                    log_rewards[i] = torch.log(Reward(args, states[i])).detach()
                else:
                    if actions[i] < args.dim: 
                        states[i][actions[i]] -= 1
                    else:
                        states[i][actions[i] - args.dim] += 1
            
            assert states.max() < args.side
            assert states.min() >= 0
            
            # make predictions for next state and replace log flow in terminal states with log reward
            new_logits = model(batch_ohe(args, states).detach())
            log_pfs_next, log_pbs_next, log_flows_next = process_logits(args, states, new_logits)
            log_flows_next[dones_new] = log_rewards[dones_new]

            log_forward_policy = log_pfs[range(batch_size), actions]
            log_backward_policy = log_pbs_next[range(batch_size), actions]
            log_backward_policy[dones_new] = 0.0
            
            # mask for the loss calculation to avoid trajectories that already finished
            loss_mask = ~dones

            if args.loss_flow_scale == "LogFlow":
                forward_flow = log_flows + log_forward_policy
                backward_flow = log_flows_next + log_backward_policy
            elif args.loss_flow_scale == "Flow":
                forward_flow = torch.exp(log_flows + log_forward_policy)
                backward_flow = torch.exp(log_flows_next + log_backward_policy)

            if args.loss == "DB":
                step_losses = (forward_flow - backward_flow) ** 2
            elif args.loss == "StableDB":
                step_losses = torch.log1p(args.stable_db_eps * ((forward_flow - backward_flow) ** 2))
                step_losses *= 1 + args.stable_db_eta * torch.exp(log_flows) 
 
            step_losses += reg_coef * torch.exp(log_flows) 
            loss += (step_losses * loss_mask).sum()
            total += loss_mask.sum()
            
            dones = dones_new
            log_pfs, log_pbs, log_flows = log_pfs_next, log_pbs_next, log_flows_next

        for state in states:
            all_samples.append(state)
        traj_len_history.append(total / batch_size)

        loss = loss / total
        loss.backward()
        opt.step()
        log_Z_opt.step()
            
        if (it + 1) % 100 == 0:
            l1  = compute_empirical_distribution_error(true_distribution, all_samples[-200000:])
            print(f"states visited: {(it + 1) * batch_size}, current_loss: {loss.item()}")
            print(f"empirical L1 distance: {l1}, mean traj len: {np.mean(traj_len_history[-1000:])}")
            print(f"estimated log Z: {log_Z.sum().detach().item()}")
                
            l1_history.append(l1)
            mean_traj_len_history.append(np.mean(traj_len_history[-1000:]))
            log_Z_history.append(log_Z.sum().detach().item())
            
            np.save(f"{args.save_dir}/{args.name}_{args.seed}_{args.loss}_{args.loss_flow_scale}_seed{args.seed}_standard_{args.dim}_{args.side}_l1.npy", np.array(l1_history))
            np.save(f"{args.save_dir}/{args.name}_{args.seed}_{args.loss}_{args.loss_flow_scale}_seed{args.seed}_standard_{args.dim}_{args.side}_trajlen.npy", np.array(mean_traj_len_history))
            np.save(f"{args.save_dir}/{args.name}_{args.seed}_{args.loss}_{args.loss_flow_scale}_seed{args.seed}_standard_{args.dim}_{args.side}_logz.npy", np.array(log_Z_history))
                
    return l1_history, mean_traj_len_history, log_Z_history


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    true_distribution, true_Z = compute_true_distribution(args)
    print("true log Z:", np.log(true_Z))

    model = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(args.dim * args.side, args.hidden_size)),
          ('relu1', nn.ReLU()),
          ('linear2', nn.Linear(args.hidden_size, args.hidden_size)),
          ('relu2', nn.ReLU()), 
          ('linear3', nn.Linear(args.hidden_size, 4 * args.dim + 2 + 1)),
    ]))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # For explanation see: https://gist.github.com/MJ10/59bfcc8bce4b5fce9c1c38a81b1105ae#file-bit_tb-py-L78 
    log_Z = nn.Parameter(torch.zeros(64, requires_grad=True))
    log_Z_opt = torch.optim.Adam([log_Z], 1e-2)

    l1_history, mean_traj_len_history, log_Z_history = train(args, model, opt, log_Z, log_Z_opt, true_distribution, 
                                                             batch_size=args.batch_size, steps=2000000 // args.batch_size)