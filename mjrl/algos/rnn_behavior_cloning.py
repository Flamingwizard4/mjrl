"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging

from torch.nn.modules.loss import BCEWithLogitsLoss
logging.disable(logging.CRITICAL)
import numpy as np
import time as timer
import torch
from torch.autograd import Variable
from mjrl.utils.logger import DataLog
from tqdm import tqdm


class rnn_BC:
    def __init__(self, expert_paths,
                 policy,
                 env,
                 epochs = 5,
                 seed=None,
                 batch_size = 32,
                 lr = 1e-3,
                 optimizer = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 save_logs = True,
                 set_transforms = False,
                 **kwargs,
                 ):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.seed = seed
        self.mb_size = batch_size
        self.logger = DataLog()
        self.loss_type = loss_type
        self.save_logs = save_logs
        self.env = env

        if set_transforms:
            in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
            self.set_transformations(in_shift, in_scale, out_shift, out_scale)
            self.set_variance_with_data(out_scale)

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=lr) if optimizer is None else optimizer

        # Loss criterion if required
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

        # make logger
        if self.save_logs:
            self.logger = DataLog()

    def compute_transformations(self):
        # get transformations
        if self.expert_paths == [] or self.expert_paths is None:
            in_shift, in_scale, out_shift, out_scale = None, None, None, None
        else:
            observations = np.concatenate([path["observations"] for path in self.expert_paths])
            actions = np.concatenate([path["actions"] for path in self.expert_paths])
            in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
        return in_shift, in_scale, out_shift, out_scale

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # set scalings in the target policy
        self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_variance_with_data(self, out_scale):
        # set the variance of gaussian policy based on out_scale
        params = self.policy.get_param_values()
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        self.policy.set_param_values(params)

    def loss(self, data, idx=None):
        if self.loss_type == 'MLE':
            return self.mle_loss(data, idx)
        elif self.loss_type == 'MSE':
            return self.mse_loss(data, idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # use indices if provided (e.g. for mini-batching)
        # otherwise, use all the data
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) == torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]
        LL, mu, log_std = self.policy.new_dist_info(obs, act)
        # minimize negative log likelihood
        return -torch.mean(LL)

    def mse_loss(self, ob, ac):
        '''idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)'''
        if type(ob) is not torch.Tensor:
            obs = Variable(torch.from_numpy(ob).float(), requires_grad=False)
            acts = Variable(torch.from_numpy(ac).float(), requires_grad=False)

        '''print(torch.unsqueeze(obs, 1).shape)
        print("Observations Dim: ", obs.shape)
        print("Observation Dim: ", obs[0].shape)
        print("Actions Dim: ", acts.shape)'''
        obs = torch.unsqueeze(obs, 1)
        acts_pi = self.policy.model(obs)
        #print(acts_pi[0].shape)
        #print(acts_pi[1].shape)
        return self.loss_criterion(acts_pi[0], torch.unsqueeze(acts.detach(), 1))

    def old_mse_loss(self, data, idx=None): #mse loss on whole trajectories (action pairs)
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)   
        obs = np.array(data['observations'], dtype=float)[idx]
        actx = np.array(data['expert_actions'], dtype=float)[idx]
        if type(data['observations']) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
            actx = Variable(torch.from_numpy(actx).float(), requires_grad=False)
        for o in range(obs.shape[1]): #rolling out policy
            act_pi, _ = self.policy.model(obs[:,:o+1])
            act_pi = torch.unsqueeze(act_pi, 1)
            acts_pi = torch.cat((acts_pi, act_pi), 1) if o > 0 else act_pi
        return self.loss_criterion(acts_pi, actx.detach())
    
    def selfrolled_mse_loss(self, data, idx=None): #doesn't train well
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)   
        obs = np.array(data['observations'], dtype=float)[idx]
        act_expert = np.array(data['expert_actions'], dtype=float)[idx]
        if type(data['observations']) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
            act_expert = Variable(torch.from_numpy(act_expert).float(), requires_grad=False)
        
        self.env.reset()
        t, done, performed, terminate_at_done, mean_action = 0, False, False, False, True
        while len(obs.shape) < 3:
            obs = torch.unsqueeze(obs, 0)
        #obs, acts = torch.Tensor(1, 1, self.observation_dim), torch.Tensor(1, 1, self.action_dim)
        while t < obs.shape[1] and (done == False or terminate_at_done == False):
            for b in range(obs.shape[0]): #roll out each sample from batch separately
                b_obs = new_obs[b, :, :] if t > 0 else obs[b, 0, :]
                ob = torch.unsqueeze(b_obs, 0)
                if t == 0:
                    ob = torch.unsqueeze(ob, 0)
                a = self.policy.get_action(ob.detach())[1]['evaluation'] #if mean_action is True else self.policy.get_action(obs)[0]
                o, r, done, _ = self.env.step(a)
                o = self.env.get_obs()
                if type(o) is not torch.Tensor:
                    ob = Variable(torch.from_numpy(o).float(), requires_grad=True)
                else:
                    ob = o 
                new_o = torch.unsqueeze(ob, 0)
                new_o = torch.unsqueeze(new_o, 0)
                new_ob = torch.cat((new_ob, new_o), 0) if b > 0 else new_o
                if type(a) is not torch.Tensor:
                    ac = Variable(torch.from_numpy(a).float(), requires_grad=True)
                else:
                    ac = a
                act_pi = torch.unsqueeze(ac, 0)
                act_pi = torch.unsqueeze(act_pi, 0)
                new_act = torch.cat((new_act, act_pi), 0) if b > 0 else act_pi
                #print(new_act.shape)
            
            new_obs = torch.cat((new_obs, new_ob), 1) if t > 0 else new_ob
            
            acts_pi = torch.cat((acts_pi, new_act), 1) if t > 0 else new_act
            #print(acts_pi.shape)
            t += 1

        if _['goal_achieved']:
            performed = True
            print("Accomplished")
        
        return self.loss_criterion(acts_pi, act_expert.detach()), performed
    
    def pad_paths(self, obs, act, m):
        new_obs, new_act = [], []
        for i in range(len(obs)):
            o, a = obs[i], act[i]
            n = o.shape[0]
            last_o = np.reshape(o[-1][:], (1, o.shape[1])) #should be outside the loop but too lazy
            null_a = np.zeros((1, a.shape[1]))
            while n < m:
                o = np.concatenate((o, last_o))
                a = np.concatenate((a, null_a))
                n += 1
            new_obs.append(o)
            new_act.append(a)
        return new_obs, new_act

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        ts = timer.time()
        num_samples = len(data["observations"])

        # log stats before
        '''if self.save_logs:
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            self.logger.log_kv('loss_before', loss_val)'''
        losses = []
        rng = np.random.default_rng(seed=self.seed)
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            rdx = rng.integers(low=0, high=len(self.expert_paths), size=self.mb_size)
            self.optimizer.zero_grad()
            #print(type(data['observations'][0]))
            #print(rdx)
            #print(type(int(rdx)))
            loss = self.old_mse_loss(data, idx=rdx) #(data['observations'])[rdx], (data['expert_actions'])[rdx]) 
            loss.backward()
            losses.append(loss.detach())
            self.optimizer.step()    
        '''# train loop
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            for mb in range(int(num_samples / self.mb_size)):
                rand_idx = np.random.choice(num_samples, size=self.mb_size)
                self.optimizer.zero_grad()
                loss = self.loss(data, idx=rand_idx)
                loss.backward()
                self.optimizer.step()'''
        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)
        return losses
        # log stats after
        '''if self.save_logs:
            self.logger.log_kv('epoch', self.epochs)
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            self.logger.log_kv('loss_after', loss_val)
            self.logger.log_kv('time', (timer.time()-ts))'''

    def train(self, **kwargs):
        observations = [path["observations"] for path in self.expert_paths]
        max_obs_len = len(max(observations, key=len))
        #print(max_obs_len)
        #print(self.expert_paths[0]["observations"].shape[1])
        #print(self.expert_paths[0]["actions"].shape[1])    
        expert_action_paths = [path["actions"] for path in self.expert_paths]
        max_act_len = len(max(expert_action_paths, key=len))
        #print(len(observations))
        #print(len(expert_action_paths))
        assert max_obs_len == max_act_len
        #print(type(expert_action_paths[0][0]))
        #print(expert_action_paths[0][0].shape)
        obs, act = self.pad_paths(observations, expert_action_paths, max_act_len)
        #print([path["actions"].shape for path in self.expert_paths])
        data = dict(observations=obs, expert_actions=act)
        return self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)