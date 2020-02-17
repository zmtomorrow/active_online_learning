import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tools import *
import operator
import itertools
import matplotlib.pyplot as plt
import math


class LogisticNet(nn.Module):
    def __init__(self,opt):
        super(LogisticNet, self).__init__()
        self.last_weight_dim=784
        self.device=opt['device']
        self.if_cuda=opt['if_cuda']
        self.mode='erf'
        self.q_rank=opt['q_rank']
        self.prior_mu=torch.zeros(self.last_weight_dim, dtype=torch.float, requires_grad=False, device=self.device)
        self.prior_sigma=torch.tensor(1.0, requires_grad=False, device=self.device)

        self.q_mu=torch.nn.Parameter(data=torch.randn(self.last_weight_dim)*0.1, requires_grad=True)
        self.q_sigma=torch.nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.q_L=torch.nn.Parameter(data=torch.randn(self.last_weight_dim,self.q_rank)*0.1, requires_grad=True)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.online_optimizer = optim.Adam(self.parameters(), lr=opt['online_lr'])

   
    def predictive_entropy(self,x,mode='erf'):
        p=self.marginal_probability(x,mode)
        return -p*torch.log(p)-(1-p)*torch.log(1-p)

    def predict(self,x,mode='erf'):
        p=self.marginal_probability(x,mode)
        return (p>0.5).float()

    
    def mc_forward(self,x,mc_num):
        final_weight_sample= low_rank_gaussian_sample(self.q_mu,self.q_L,self.q_sigma,amount=mc_num,cuda=self.if_cuda)
        probs=torch.mean(torch.sigmoid(x@final_weight_sample.t()),dim=-1)
        return probs
        
        
        
    def marginal_probability(self,x,mode='erf',sample_num=1000):
        x=x.view(-1,784).to(self.device)
        if mode=='erf':
            with torch.no_grad():
                cov=self.q_L@self.q_L.t()+torch.eye(784).to(self.device)*(self.q_sigma**2)
                pre_act_var=torch.bmm((x@cov).view(-1,1,784),x.view(-1,784,1)).squeeze()
                pre_act_mu=x@self.q_mu
                ks=(1+math.pi*pre_act_var/8)**(-0.5)
                probs=torch.sigmoid(ks*pre_act_mu)
                return probs
        elif mode=='mc':
            with torch.no_grad():
                probs=self.mc_forward(x,sample_num)
                return probs
        else:
            raise NotImplementedError


    def online_train(self,x,label,mc_num,online_step):
        x=x.view(-1,784).to(self.device)
        train_losses = []
        prob_list = []
        entropy_list = []
        total_size=x.size(0)
        curr_prior_mu = self.q_mu.clone().detach()
        curr_prior_L = self.q_L.clone().detach()
        curr_prior_sigma = self.q_sigma.clone().detach()

   
        for i in range(0,online_step):
            self.online_optimizer.zero_grad()
            probs=self.mc_forward(x,mc_num)
            nll_loss=F.binary_cross_entropy(probs, label.type_as(probs))*total_size
            kl=KL_low_rank_gaussian_with_low_rank_gaussian(self.q_mu,self.q_L,self.q_sigma,curr_prior_mu,curr_prior_L,curr_prior_sigma,cuda=self.if_cuda)
            neg_elbo=kl+nll_loss
            neg_elbo.backward()
            self.online_optimizer.step()
            train_losses.append(neg_elbo.item())
            entropy_list.append(self.predictive_entropy(x,self.mode))
            prob_list.append(self.marginal_probability(x,self.mode))
            
        return train_losses
    
    def train(self,x,label,mc_num):
        x=x.view(-1,784).to(self.device)
        train_losses = []
        if x.size(0)<100:
            batch_size=x.size(0)
            iteration=1
        else:
            batch_size=100
            iteration=int(x.size(0)/batch_size)
        for epoch in range(0,30):
            print('epoch',epoch)
            for it in range(0,iteration):
                index=np.random.choice(x.size(0),batch_size)
                self.optimizer.zero_grad()
                probs=self.mc_forward(x[index],mc_num)
                nll_loss=F.binary_cross_entropy(probs, label[index].type_as(probs),reduction='mean')*x.size(0)

                kl=KL_low_rank_gaussian_with_diag_gaussian(self.q_mu,self.q_L,self.q_sigma,self.prior_mu,self.prior_sigma,cuda=self.if_cuda)
                neg_elbo=kl+nll_loss
                neg_elbo.backward()
                self.optimizer.step()
                train_losses.append(neg_elbo.item())
        return train_losses
        

    def test(self,test_data,test_label):
        with torch.no_grad():
            pred=self.predict(test_data)
            correct_list=pred.eq(test_label.type_as(pred))
            correct=correct_list.sum().float()
            ratio=correct/torch.tensor(test_label.size(0))
            return ratio.item()
    
