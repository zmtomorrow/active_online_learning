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
np.random.seed(0)
torch.manual_seed(0)
opt= {}
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opt['device']= torch.device('cuda:0')
    opt['if_cuda']=True
else:
    opt['device']= torch.device('cpu')
    opt['if_cuda']=False


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

    def marginal_probability(self,x,mode='erf',sample_num=1000):
        x=x.view(-1,784)
        if mode=='erf':
            with torch.no_grad():
                cov=self.q_L@self.q_L.t()+torch.eye(784)*(self.q_sigma**2)
                pre_act_var=torch.bmm((x@cov).view(-1,1,784),x.view(-1,784,1)).squeeze()
                pre_act_mu=x@self.q_mu
                ks=(1+math.pi*pre_act_var/8)**(-0.5)
                probs=torch.sigmoid(ks*pre_act_mu)
                return probs
        elif mode=='mc':
            with torch.no_grad():
                final_weight_samples=low_rank_gaussian_sample(self.q_mu,self.q_L,self.q_sigma,sample_num,cuda=self.if_cuda)
                probs = torch.mean(torch.sigmoid(x@final_weight_samples.t()),dim=-1)
                return probs
        else:
            raise NotImplementedError


    def online_train(self,x,label,sample_num=1,iteration=200):
        x=x.view(-1,784)
        train_losses = []
        prob_list = []
        entropy_list = []
        total_size=x.size(0)
        curr_prior_mu = self.q_mu.clone().detach()
        curr_prior_L = self.q_L.clone().detach()
        curr_prior_sigma = self.q_sigma.clone().detach()

   
        for i in range(0,iteration):
            self.online_optimizer.zero_grad()
            final_weight_sample= low_rank_gaussian_one_sample(self.q_mu,self.q_L,self.q_sigma,cuda=self.if_cuda)
            logits=x@final_weight_sample
            probs=torch.sigmoid(logits)
            nll_loss=F.binary_cross_entropy_with_logits(logits, label.type_as(logits))*total_size
            kl=KL_low_rank_gaussian_with_low_rank_gaussian(self.q_mu,self.q_L,self.q_sigma,curr_prior_mu,curr_prior_L,curr_prior_sigma)
            neg_elbo=kl+nll_loss
            neg_elbo.backward()
            self.online_optimizer.step()
            train_losses.append(neg_elbo.item())
            entropy_list.append(self.predictive_entropy(x,self.mode))
            prob_list.append(self.marginal_probability(x,self.mode))
            
        return prob_list
    
    def train(self,x,label):
        x=x.view(-1,784)
        train_losses = []
        if x.size(0)<100:
            batch_size=x.size(0)
            iteration=1
        else:
            batch_size=100
            iteration=int(x.size(0)/batch_size)
        for epoch in range(0,10):
            for it in range(0,iteration):
                index=np.random.choice(x.size(0),batch_size)
                self.optimizer.zero_grad()
                final_weight_sample= low_rank_gaussian_one_sample(self.q_mu,self.q_L,self.q_sigma,cuda=self.if_cuda)
                logits=x[index]@final_weight_sample
                probs=torch.sigmoid(logits)
                nll_loss=F.binary_cross_entropy_with_logits(logits, label[index].type_as(logits))*x.size(0)
                #nll_loss= -torch.mean(label[index]*torch.log(probs)+(1-label[index])*torch.log(1-probs))*x.size(0)
                kl=KL_low_rank_gaussian_with_diag_gaussian(self.q_mu,self.q_L,self.q_sigma,self.prior_mu,self.prior_sigma,cuda=self.if_cuda)
                neg_elbo=kl+nll_loss
                neg_elbo.backward()
                self.optimizer.step()
                train_losses.append(neg_elbo.item())
            print('accuracy',neg_elbo)
        #plt.plot(train_losses)
        #plt.show()
        return train_losses
        
    #def test(self):
    #    correct=0
    #    for data, target in test_loader:
    #        pred = self.predict(data,'erf')
    #        correct += pred.eq(target.data.view_as(pred)).sum()
    #        correct_ratio= float(correct)/len(test_loader.dataset)
    #    return correct_ratio

    def test(self,test_data,test_label):
        pred=self.predict(test_data)
        correct=pred.eq(test_label).sum().detach().numpy()
        ratio=float(correct/test_label.size())
        return ratio
    
