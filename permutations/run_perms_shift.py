from tqdm import tqdm
import torch
from torch.distributions.categorical import Categorical
import numpy as np 
from env import PermutationEnv
import argparse
import torch.nn as nn
from losses import GFlowLoss
from logger import GFlowLogger
from metric import EmpiricalDistributionMetric

parser = argparse.ArgumentParser()

parser.add_argument("--p", default=10, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--reg_coef", default=0.0, type=float)
parser.add_argument("--hidden_size", default=64, type=int)
parser.add_argument("--name", default="exp", type=str)
parser.add_argument("--loss", default="DB", type=str)
parser.add_argument("--sdb_eps", default=1.0, type=float)
parser.add_argument("--sdb_beta", default=1.0, type=float)
parser.add_argument("--sdb_alpha", default=2.0, type=float)
parser.add_argument("--sdb_eta", default=0.001, type=float)
parser.add_argument("--loss_scale", default=True, type=bool, action=argparse.BooleanOptionalAction) # log_scale or not
parser.add_argument("--steps", default=100000, type=int)
parser.add_argument("--save_dir", default="permutations/results", type=str)
parser.add_argument("--reward", default="sum", type=str)
parser.add_argument("--t", default=2.0, type=float)
parser.add_argument("--seed", default=1337, type=int)


def to_ohe(states, bs):
    return torch.nn.functional.one_hot(states).reshape(bs, -1).to(torch.float32)


def train(model, opt, log_Z_opt, log_Z, device, args):
    logger = GFlowLogger(save_dir=args.save_dir, args=args)
    all_samples = []
    traj_len_history = []
    mean_traj_len_history=[]
    mean_reward = []
    log_Z_history = []
    relative_empirical_l1 = []
    empirical_l1 = []
    reward_history = []
    env = PermutationEnv(p=args.p,
                    batch_size=args.batch_size,
                    reward_type=args.reward,
                    tau=args.t)
    inverse_factorial = -torch.lgamma(torch.tensor(args.p+1))
    loss_func = GFlowLoss(args.loss)
    metric = EmpiricalDistributionMetric(p=args.p, tau=args.t)



    for it in tqdm(range(args.steps)):
        opt.zero_grad()
        log_Z_opt.zero_grad()

        states = env.get_initial_state().to(device)
        dones = torch.tensor([False] * args.batch_size).to(device)
        loss = 0.0
        total = 0
        logits = None

        # compute first step loss 
    
        logits = model(to_ohe(states, args.batch_size).detach())
    
        log_flows = logits[:, -1]
        backward_logits = logits[:, args.p + 1:-1]
        backward_log_probs = torch.nn.functional.log_softmax(backward_logits, dim=-1)
        backward_stop_log_probs = backward_log_probs[:, -1]
        loss += loss_func(log_flow=log_Z.sum() * torch.ones_like(backward_stop_log_probs).to(device),
                          log_forward=inverse_factorial,
                          log_flow_next=log_flows,
                          log_backward=backward_stop_log_probs,
                          loss_mask=torch.ones_like(backward_stop_log_probs).to(device), 
                          args=args, is_first=True)
        total += args.batch_size

        
        # we sample trajectories and compute loss simultaneously
        while not all(dones):
            forward_logits = logits[:, :args.p + 1] # all actions + stop action
            log_flows = logits[:, -1] 

            # sample batch of actions
            with torch.no_grad():
                actions = Categorical(logits=forward_logits).sample()

            log_policy = forward_logits[torch.arange(args.batch_size), actions] - torch.logsumexp(forward_logits, dim=-1)
                
            log_rewards = env.batch_log_reward(states)
            dones_new = dones.clone()
            dones_new[actions == args.p] = True
            next_states = states.clone()
            next_states[~dones_new] = env.get_next_state_extended(states[~dones_new], actions[~dones_new])
            assert next_states.sum() == args.batch_size * (args.p - 1) * args.p // 2        
            # compute log forward policy
            
            # make predictions for next state and replace log flow in terminal states with log reward
            
            new_logits = model(to_ohe(next_states, args.batch_size).detach())
            
            log_flows_next = new_logits[:, -1]
            log_flows_next[dones_new] = log_rewards[dones_new]

            backward_logits = new_logits[:, args.p + 1:-1]
            log_pbs = backward_logits[range(args.batch_size), actions] - torch.logsumexp(backward_logits, dim=-1)
            log_pbs[dones_new] *= 0.0
            
            

            # mask for the loss calculation to avoid trajectories that already finished
            loss_mask = ~dones

            loss += loss_func(log_policy, log_pbs, log_flows, log_flows_next, loss_mask, args, is_first=False)
            total += loss_mask.sum()
            dones = dones_new
            logits = new_logits
            states = next_states

        metric.add_to_buffer(states=states.cpu())
        traj_len_history.append(total.cpu().detach() / args.batch_size)
        reward_history += [torch.exp(env.batch_log_reward(states).cpu().detach()).sum().item() / args.batch_size]

        loss = loss / total
        loss.backward()
        opt.step()
        log_Z_opt.step()
            
        if (it + 1) % 500 == 0:
            mean_reward += [np.mean(reward_history[-1000:])]
            mean_traj_len_history += [np.mean(traj_len_history[-1000:])]
            log_Z_history += [log_Z.cpu().detach().sum().item()]
            relative, absolute = metric.compute_metric()
            empirical_l1 += [absolute]
            relative_empirical_l1 += [relative]
            dct = {"reward": np.array(mean_reward),
                    "trajlen": np.array(mean_traj_len_history),
                    "logz": np.array(log_Z_history),
                    "empirical_l1": np.array(empirical_l1),
                    "relative_l1": np.array(relative_empirical_l1)}
            logger.log(dct)

            print(f"states visited: {(it + 1) * args.batch_size}")
            print(f"current_loss: {loss.item()}")
            print(f"mean traj len: {np.mean(traj_len_history)}")
            print(f"mean reward: {mean_reward[-1]}")
            print(f"estim log_Z: {log_Z_history[-1]}")
            print(f"empirical l1: {empirical_l1[-1]}")
            print(f"relative empirical l1: {relative_empirical_l1[-1]}")
            
    return mean_traj_len_history, all_samples,
            

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = nn.Sequential(
        nn.Linear(args.p*args.p, args.hidden_size),
        nn.LeakyReLU(),
        nn.Linear(args.hidden_size, args.hidden_size),
        nn.LeakyReLU(),
        nn.Linear(args.hidden_size, 2*args.p + 3)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    log_Z = torch.nn.Parameter(torch.zeros(64, requires_grad=True, device=device))
    log_Z_opt = torch.optim.Adam([log_Z], 1e-2)

    a, b = train(model, opt, log_Z_opt, log_Z, device, args=args)
