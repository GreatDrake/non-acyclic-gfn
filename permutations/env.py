import torch 
import numpy as np
import itertools

class PermutationEnv:
    def __init__(self, p=3, batch_size=16, reward_type="indicator", tau=2):
        self.initial_state = torch.from_numpy(np.array([np.random.permutation(np.arange(p)) 
                                       for _ in range(batch_size)]).reshape(batch_size, -1))
        self.p = p
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.true_perm = torch.arange(p).to(self.device)
        self.reward_type = reward_type
        self.tau = tau
    
    def get_initial_state(self):
        return torch.from_numpy(np.array([np.random.permutation(np.arange(self.p)) 
                                       for _ in range(self.batch_size)]).reshape(self.batch_size, -1))
    
    def get_fix_initial_state(self):
        return torch.from_numpy(
                        np.tile(
                            np.arange(self.p), self.batch_size
                            ).reshape(self.batch_size, -1)).flip(dims=(0, 1))
    
    def get_next_state(self, prev_state, action):
        """
        Assume the set of generators is (1 2), (2, 3), ... , (n-1, n)
        """
        new_states = prev_state.clone()
        bs_idx = torch.arange(prev_state.shape[0])
        new_states[bs_idx, action],new_states[bs_idx, action+1] = prev_state[bs_idx, action+1], prev_state[bs_idx, action]
        return new_states
    
    def get_next_state_extended(self, prev_state, action):
        """
        Assume the set of generators is (1 2), (2, 3), ... , (n-1, n), (cyclic-shift)
        """
        next_state = torch.clone(prev_state)
        mask_roll = action == self.p-1
        mask_roll = mask_roll.to(self.device)
        next_state[mask_roll] = torch.roll(prev_state[mask_roll], shifts=1, dims=-1)
        bs_idx = torch.arange(prev_state.shape[0], device=self.device)[~mask_roll]
        next_state[bs_idx, action[~mask_roll]], next_state[bs_idx, action[~mask_roll] + 1] = prev_state[bs_idx, action[~mask_roll] + 1], prev_state[bs_idx, action[~mask_roll]]
        return next_state

    def batch_reward(self, states):
        return torch.exp(torch.sum((states == self.true_perm), dim=-1)/self.tau)

    def batch_log_reward(self, states):
        return torch.log(self.batch_reward(states))
    
    def true_vals(self):
        all_perms_batch = torch.tensor(list(itertools.permutations([i for i in range(self.p)])))
        all_rewards = self.batch_reward(all_perms_batch)
        Z = all_rewards.sum()
        return np.log(Z), ((all_rewards/Z)*all_rewards).sum().item()