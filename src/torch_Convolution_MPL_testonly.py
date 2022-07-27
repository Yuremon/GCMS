from random import shuffle as r_shuffle
from turtle import forward
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import tools
import torch
from torch.nn.parameter import Parameter
import numpy as np
###################class
class NormalMeanVar(nn.Module):
    def __init__(self, num_input) -> None:
        super().__init__()
        self.mean = Parameter(torch.normal(0,0.1,size=(1,num_input)))
        self.var = Parameter(torch.ones((1,num_input)))
    def forward(self, X):
        res = (X-self.mean)/(self.var + 1e-5)
        return res**2
class MinusOneRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rl = F.relu
    def forward(self, X):
        return self.rl(X-1)
        
class Accumulator:
    def __init__(self,n) -> None:
        self.data = [0.0]*n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
    
    
class GCMS_Data_convo(Dataset):
    def __init__(self, x_train, y_train, binary = True) -> None:
        super().__init__()
        self.train = x_train
        self.type = y_train
        self.binary = binary
    def __len__(self):
        return len(self.train)
    def __getitem__(self, index):
        if self.binary:
            trainset = (torch.tensor(self.train[index],dtype=torch.float32).reshape(1,1,504),torch.tensor(self.type[index]))
            return trainset
        else:
            type = torch.zeros(6)
            if ";" in self.type[index]:
                for i in self.type[index].split(";") : 
                    type[int(i)] = 1.
            else:
                type[int(self.type[index])] = 1.
            trainset = (torch.tensor(self.train[index],dtype=torch.float32),type)
            return trainset

#####fonction
def dropout_layer(X, rate_dropout):
    assert 0 <= rate_dropout <= 1
    if rate_dropout == 1:
        return torch.zeros_like(X)
    if rate_dropout == 0:
        return X
    mask = (torch.randn(X.shape)>rate_dropout).float()
    return mask*X / (1.0 - rate_dropout)

def sgd(params, lr, batch_size):
    """小批量随机梯度下降
    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
def cross_entropy(y_hat,y):
    '''
    y_hat n éléments chaque élément -> probabilté de chaque class
    y list de class des n élément -> indice de class
    '''
    return -torch.log(y_hat[range(len(y_hat)),y])#-torch.log(y_hat[y])#
 
def square_loss(X, y):
    return 0.5*torch.square(X)/X.shape[1]

def accuracy(y_hat,y):
    '''
        nb bien prédit / nb total
    '''
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1) #indice de plus grand élément return list indice ex: [[0.9, 0.5, 0.1],[0.1, 0.2, 0.3],[0.1, 0.5, 0.4]] -> [0, 2, 1]
    #y_hat = y_hat.argmax()
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X,y in data_iter:
        metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def Precision_Recall(y_hat, y):
    '''
                            Prédiction
    actu-     true_non_normal   |  false_normal
    al        false_non_normal      |  true_normal
    '''
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        actual_normal = (y > 0).nonzero().numel()
        actaul_non_normal = y.numel()-actual_normal
        y_hat = y_hat.argmax(axis=1) 
        cmp = y_hat.type(y.dtype) == y # accuracy 
        
        res_non_normal = y - torch.ones_like(y) + y_hat
        
        
        true_non_normal = (res_non_normal < 0).nonzero().numel()
        false_normal = actaul_non_normal - true_non_normal
        true_normal = float(cmp.type(y.dtype).sum())-true_non_normal
        false_non_normal = actual_normal - true_normal
        
    return true_non_normal, false_normal, false_non_normal, true_normal

def Evaluate_Matrix_Confusion(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(5)
    for X,y in data_iter:
        true_non_normal, false_normal, true_normal, false_non_normal = Precision_Recall(net(X),y)
        metric.add(true_non_normal, false_normal, true_normal, false_non_normal, y.numel())
    return metric[0]/metric[4], metric[1]/metric[4], metric[2]/metric[4], metric[3]/metric[4]
 
def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    #metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.sum().backward()
            updater.step()
            #metric.add(float(l.sum()), accuracy(y_hat,y), y.numel())         
        else:
           l.sum().backward()
           updater(x.shape[0])
           #metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    #return metric[0]/metric[2], metric[1]/metric[2]
    
def trainbystep_epoch(prenet, net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    #metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(prenet(x))
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.sum().backward()
            updater.step()
            #metric.add(float(l.sum()), accuracy(y_hat,y), y.numel())         
        else:
           l.sum().backward()
           updater(x.shape[0])
def trainbystep(prenet, net, train_iter, test_iter, loss, num_epochs, updater, evaluate = False):
    for epoch in range(num_epochs): 
        trainbystep_epoch(prenet, net, train_iter, loss, updater)
        
    if evaluate:
       # print(net_label.state_dict())
        test_acc = evaluate_accuracy(nn.Sequential(prenet, net), test_iter)
        print(test_acc)
        print(Evaluate_Matrix_Confusion(nn.Sequential(prenet, net), test_iter))
    
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs): 
        train_epoch(net, train_iter, loss, updater)
    #test_acc = evaluate_accuracy(net, test_iter)
    #train_loss, train_acc = train_metrics
    test_acc = evaluate_accuracy(net, test_iter)
    print(test_acc)
    print("\n")
    
def train_6label(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs): 
        train_epoch(net, train_iter, loss, updater)


def K_fold_data(k, i, x, y):
    
    assert k > 1
    train_size = len(y)
    fold_size = train_size//k
    
    x_train,y_train = None, None
    for j in range(k):
        ind = slice(j*fold_size, (j+1)*fold_size)
        x_seq, y_seq = x[ind], y[ind]
        if j == i:
            x_validation = x_seq
            y_validation = y_seq
        elif x_train is None:
            x_train, y_train = x_seq, y_seq
        else:
            x_train = np.concatenate((x_train, x_seq), axis = 0) 
            y_train = np.concatenate((y_train, y_seq), axis = 0) 
    return x_train, y_train, x_validation, y_validation

def k_fold_train(net, k, x, y, loss, num_epochs, updater, train, binary = True):
    for ki in range(k):
        x_train, y_train, x_validation, y_validation = K_fold_data(k, ki, x, y)
        train_dataset = GCMS_Data_convo(x_train, y_train, binary)
        validation_dataset = GCMS_Data_convo(x_validation, y_validation, binary)
        train_batch = DataLoader(train_dataset, batch_size= 25, shuffle=True)
        validation_batch = DataLoader(validation_dataset, batch_size=len(y_validation), shuffle=False)
        #print(net.state_dict())
        train(net, train_batch, validation_batch, loss, num_epochs, updater)

def k_fold_trainstep(prenet, net, k, x, y, loss, num_epochs, updater, binary = True, evaluate= False):
    for ki in range(k):
        x_train, y_train, x_validation, y_validation = K_fold_data(k, ki, x, y)
        train_dataset = GCMS_Data_convo(x_train, y_train, binary)
        validation_dataset = GCMS_Data_convo(x_validation, y_validation, binary)
        train_batch = DataLoader(train_dataset, batch_size= 25, shuffle=True)
        validation_batch = DataLoader(validation_dataset, batch_size=len(y_validation), shuffle=False)
        trainbystep(prenet, net, train_batch, validation_batch, loss, num_epochs, updater, evaluate)      

def initial_weight(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.normal_(module.weight, std=0.1)
       
        torch.nn.init.zeros_(module.bias)


if __name__ == '__main__':
    path_train = "./data/train/"
    path_test = "./data/test/"
    momentum = 0.9
    lr = 0.02
    epochs = 45
    k = 8
    wd = 1e-4
    X_train, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
    X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)
    #print(len(y_test))
    X_train_label, y_train_label = tools.getDataTransformed(path_train + 'database.csv', path_train, binary=False)
    index = [i for i in range(len(y_train))]
    r_shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]
    X_train_label = X_train_label[index]
    y_train_label = y_train_label[index]
    
    index2 = (y_train > 0).nonzero()
    xmv = X_train[index2]
    ymv = y_train[index2]
    netmv = NormalMeanVar(504)
    trainer1 = torch.optim.SGD(netmv.parameters(), lr = 0.01, momentum=momentum)
    k_fold_train(netmv, k, xmv, ymv, square_loss, 2*epochs, trainer1, train_6label)
    
    net_label = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 128),nn.BatchNorm1d(128),nn.ReLU(),  nn.Dropout(dropout_rate2),nn.Linear(128,6),nn.Sigmoid())
    net_label.apply(initial_weight)
    trainer2 = torch.optim.SGD(net_label.parameters(),weight_decay=wd, lr = lr, momentum=momentum)
    k_fold_trainstep(prenet1, net_label, k, X_train_label, y_train_label, F.cross_entropy, epochs, trainer2, False)
    
    prenet2 = nn.Sequential(prenet1, net_label)
    net_type = nn.Sequential(nn.Linear(6,2), nn.Softmax(dim=1))
    net_type.apply(initial_weight)
    trainer3 = torch.optim.SGD(net_type.parameters(),weight_decay=wd, lr = 0.001, momentum=momentum)
    k_fold_trainstep(prenet2, net_type, k, X_train, y_train, F.cross_entropy, epochs, trainer3, True, evaluate=True)
    #torch.save(prenet1.state_dict(),'Parametre_net/netvm2.params')
    #torch.save(net_label.state_dict(),'Parametre_net/netlabel2.params')
    #torch.save(net_type.state_dict(),'Parametre_net/nettype2.params')
    
         
        
        


    
    net2 = torch.nn.Sequential(torch.nn.Conv1d(1,32,kernel_size=3, padding=1),torch.nn.BatchNorm1d(32), nn.ReLU(), nn.Conv1d(32,1,kernel_size=1), torch.nn.Flatten(),torch.nn.Linear(504,num_output), softmaxlayer())


    net2.apply(initial_weight)
