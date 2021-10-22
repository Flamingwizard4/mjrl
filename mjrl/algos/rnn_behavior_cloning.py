"""
Minimize bc loss (MLE, MSE, RWR etc.) with pytorch optimizers
"""

import logging
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
                 epochs = 5,
                 seed=None,
                 batch_size = 1,
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

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        ts = timer.time()
        '''num_samples = data["observations"].shape[0]

        # log stats before
        if self.save_logs:
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            self.logger.log_kv('loss_before', loss_val)'''
        losses = []
        rng = np.random.default_rng(seed=self.seed)
        for ep in range(self.epochs):
            rdx = int(rng.integers(low=0, high=len(self.expert_paths), size=1))
            self.optimizer.zero_grad()
            #print(type(data['observations'][0]))
            #print(rdx)
            #print(type(int(rdx)))
            loss = self.loss((data['observations'])[rdx], (data['expert_actions'])[rdx])
            loss.backward()
            losses.append(loss)
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
        #print(self.expert_paths[0]["observations"].shape[1])
        #print(self.expert_paths[0]["actions"].shape[1])    
        expert_action_paths = [path["actions"] for path in self.expert_paths]
        #print([path["actions"].shape for path in self.expert_paths])
        data = dict(observations=observations, expert_actions=expert_action_paths)
        return self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)