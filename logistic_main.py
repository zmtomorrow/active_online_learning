import os
import argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tools import *
from utils import *
import operator
import itertools
from logisticnet import LogisticNet


def main(opt):
    train_data=torchvision.datasets.MNIST('../data/', train=True, download=True,transform=torchvision.transforms.ToTensor())
    test_data=torchvision.datasets.MNIST('../data/', train=False, download=True,transform=torchvision.transforms.ToTensor())

    train_data_list=[]
    train_label_list=[]
    for x,y in train_data:
        if y ==7 :
            train_data_list.append(x)
            train_label_list.append(0)
        elif y ==9:
            train_data_list.append(x)
            train_label_list.append(1)
        else:
            pass
    test_data_list=[]
    test_label_list=[]
    for x,y in test_data:
        if y ==7:
            test_data_list.append(x)
            test_label_list.append(0)
        elif y ==9:
            test_data_list.append(x)
            test_label_list.append(1)
        else:
            pass 
    test_data_tensor=torch.stack(test_data_list)
    test_label_tensor=torch.tensor(test_label_list)

    net=LogisticNet(opt).to(opt['device'])

    if opt['active_learning']==False:
        train_data_tensor=torch.stack(train_data_list)
        train_label_tensor=torch.tensor(train_label_list)
        print('train_data_size',train_label_tensor.size(0))
        net.train(train_data_tensor,train_label_tensor)
        accuracy=net.test(test_data_tensor,test_label_tensor)
        print(accuracy)
        return accuracy
        
    else:
        init_data_size=opt['init_data_size']
        all_data_size=len(train_data_list)
        print(all_data_size)
        all_data_tensor=torch.stack(train_data_list)
        all_label_tensor=torch.tensor(train_label_list)

        init_data_tensor=torch.stack(train_data_list[0:init_data_size])

        init_label_tensor=torch.tensor(train_label_list[0:init_data_size])

        test_accuracy_list=[]
        train_accuracy_list=[]
        net.train(init_data_tensor,init_label_tensor)

        train_accuracy=net.test(init_data_tensor,init_label_tensor)
        train_accuracy_list.append(train_accuracy)
        print('initial train accuracy',train_accuracy)

        test_accuracy=net.test(test_data_tensor,test_label_tensor)
        test_accuracy_list.append(test_accuracy)
        print('initial test accuracy',test_accuracy)

        if opt['acquisition']=='predictive_entropy':
            if opt['allow_revisit']==False:
                unlabelled_list=[i for i in range(init_data_size,len(train_data_list))]
                labelled_list=[i for i in range(0,init_data_size)]
                for i in range(0,all_data_size-init_data_size):
                    acq_tensor=net.predictive_entropy(all_data_tensor[unlabelled_list])
                    target_index=unlabelled_list[np.argmax(acq_tensor.cpu().numpy())]
                    net.online_train(all_data_tensor[target_index],all_label_tensor[target_index].view(-1))
                    labelled_list.append(target_index)
                    unlabelled_list.remove(target_index)
                    
                    labelled_data_tensor=all_data_tensor[labelled_list]
                    labelled_label_tensor=all_label_tensor[labelled_list]
                    
                    train_accuracy=net.test(labelled_data_tensor,labelled_label_tensor)
                    test_accuracy=net.test(test_data_tensor,test_label_tensor)
                    test_accuracy_list.append(test_accuracy)
                    print(i)
                    print('train_accuracy',train_accuracy)
                    print('test_accuracy',test_accuracy)
                    if i%100==0:
                        save_data(opt,i,train_accuracy_list,test_accuracy_list)
                return train_accuracy_list,test_accuracy_list

            else:
                labelled_list=[i for i in range(0,init_data_size)]
                index_list=np.arange(0,len(train_label_list))
                for i in range(0,all_data_size):
                    acq_tensor=net.predictive_entropy(all_data_tensor)
                    target_index=np.argmax(acq_tensor.cpu().numpy())
                    net.online_train(all_data_tensor[target_index],all_label_tensor[target_index].view(-1))

                    if target_index not in labelled_list:
                        labelled_list.append(target_index)

                    labelled_data_tensor=all_data_tensor[labelled_list]
                    labelled_label_tensor=all_label_tensor[labelled_list]
                    
                    train_accuracy=net.test(labelled_data_tensor,labelled_label_tensor)
                    test_accuracy=net.test(test_data_tensor,test_label_tensor)
                    test_accuracy_list.append(test_accuracy)
                    print(i,'labelled_data:',len(labelled_list))
                    print('train_accuracy',train_accuracy)
                    print('test_accuracy',test_accuracy)
                
                    if i%opt['log_time']==0:
                        save_data(opt,i,train_accuracy_list,test_accuracy_list)


                return train_accuracy_list,test_accuracy_list


        elif opt['acquisition']=='random':
            train_data_size=all_data_size-init_data_size
            train_data_tensor=torch.stack(train_data_list[init_data_size:])
            train_label_tensor=torch.tensor(train_label_list[init_data_size:])
            trained_data_tensor=init_data_tensor.clone()
            trained_label_tensor=init_label_tensor.clone()
            index_list=np.arange(0,train_data_size)
            np.random.shuffle(index_list)
            for i,index in enumerate(index_list):
                net.online_train(train_data_tensor[index],train_label_tensor[index].view(-1))

                trained_data_tensor=torch.cat((trained_data_tensor,train_data_tensor[index].unsqueeze(0)),0)
                trained_label_tensor=torch.cat((trained_label_tensor,train_label_tensor[index].unsqueeze(0)),0)


                train_accuracy=net.test(trained_data_tensor,trained_label_tensor)
                train_accuracy_list.append(train_accuracy)

                test_accuracy=net.test(test_data_tensor,test_label_tensor)
                test_accuracy_list.append(test_accuracy)
                print(i)
                print('train_accuracy',train_accuracy)
                print('test_accuracy',test_accuracy)


                if i%100==0:
                    save_data(opt,i,train_accuracy_list,test_accuracy_list)
            return train_accuracy_list, test_accuracy_list

        


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--q_rank', type=int, default=10)
    parser.add_argument('--if_active', type=bool, default=True)
    parser.add_argument('--if_revisit', type=bool, default=False)
    parser.add_argument('--acquisition', type=str, default='predictive_entropy')
    parser.add_argument('--log_time', type=int, default='100')

    parser.add_argument('--file', type=str, default='1.txt')
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    opt= {}
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        opt['device']= torch.device('cuda:0')
        opt['if_cuda']=True
    else:
        opt['device']= torch.device('cpu')
        opt['if_cuda']=False
       
    opt['init_data_size']=1
    opt['online_lr']=1e-4

    opt['log_time']=args.log_time
    opt['active_learning']=args.if_active
    opt['acquisition']=args.acquisition
    opt['allow_revisit']=args.if_revisit
    opt['q_rank']=args.q_rank
    
    save_file='./results/'+args.file
    opt['file']=save_file
    f=open(save_file,'w')
    f.write(str(opt))
    f.write('\n')
    f.close()
    train_list,test_list=main(opt)
