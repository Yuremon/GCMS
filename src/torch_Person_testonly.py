import torch.nn.functional as F
from torch.nn.parameter import Parameter

import tools
import torch
import torch.nn as nn
class Accumulator:
    def __init__(self,n) -> None:
        self.data = [0.0]*n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

class softmaxlayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, X):
        x_exp = torch.exp(X)
        partition = x_exp.sum()    
        return x_exp/partition

class Seuillayer(nn.Module):
    def __init__(self, num_input) -> None:
        super().__init__()
        self.b = Parameter(torch.normal(0, 0.01, size = (1, num_input),requires_grad=True))
    def forward(self, X):
        return X-self.b
    
class Sumlayer(nn.Module):
    def __init__(self, num_input, num_output) -> None:
        super().__init__()  
        self.w = torch.ones(size=(num_input, num_output), requires_grad=False)
        self.b = Parameter(torch.zeros(num_output, requires_grad=True))
    def forward(self, X):
        return torch.matmul(X, self.w) + self.b
    
class mylinear(nn.Module):
    def __init__(self, num_input, num_output) -> None:
        super().__init__()
        self.w = Parameter(torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True))
        self.b = Parameter(torch.zeros(num_output,requires_grad=True))
    def forward(self, X):
        return torch.matmul(X,self.w)+self.b

def sgd(params, lr, batch_size):
    """小批量随机梯度下降
    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
                 
def softmax(x):
    x_exp = torch.exp(x)
    partition = x_exp.sum()    
    return x_exp/partition
def cross_entropy(y_hat,y):
    '''
    y_hat n éléments chaque élément -> probabilté de chaque class
    y list de class des n élément -> indice de class
    '''
    y_hat = y_hat.view(-1)
    return -torch.log(y_hat[y])#-torch.log(y_hat[range(len(y_hat)),y])
    
def accuracy(y_hat,y):
    '''
        nb bien prédit / nb total
    '''
    #if len(y_hat.shape)>1 and y_hat.shape[1]>1:
    #y_hat = y_hat.argmax(axis=1) #indice de plus grand élément return list indice ex: [[0.9, 0.5, 0.1],[0.1, 0.2, 0.3],[0.1, 0.5, 0.4]] -> [0, 2, 1]
    y_hat = y_hat.view(-1)
    y_hat = y_hat.argmax()
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    for X,y in data_iter:
        x_reshape = X.view(1,1,X.shape[0])
        metric.add(accuracy(net(x_reshape),y),y.numel())
    return metric[0]/metric[1]
    


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    #metric = Accumulator(3)
    for x, y in train_iter:
        x_reshape = x.view(1,1,x.shape[0])
        y_hat = net(x_reshape)
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

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    #for epoch in range(num_epochs):
    train_epoch(net, train_iter, loss, updater)
    #test_acc = evaluate_accuracy(net, test_iter)
    #train_loss, train_acc = train_metrics
    test_acc = evaluate_accuracy(net, test_iter)
    print(test_acc)
    print("\n")
    
    

num_input = 504
num_output = 2
lr = 0.08
w = torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)

b = torch.zeros(num_output,requires_grad=True)
num_epochs = 20
def updater(batch_size):
    return sgd([w, b], lr, batch_size)

def updater2(batch_size):
    return sgd(net2.parameters(), lr, batch_size)


def initial_weight(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.normal_(module.weight, std=0.01)
        torch.nn.init.zeros_(module.bias)

path_train = "./data/train/"
path_test = "./data/test/"

X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)

net2 = nn.Sequential(Seuillayer(num_input),nn.ReLU() ,mylinear(num_input,num_output),softmaxlayer())

train_iter = zip(torch.tensor(X_train_origin,dtype=torch.float32),torch.tensor(y_train))
test_iter = zip(torch.tensor(X_test_origin,dtype=torch.float32),torch.tensor(y_test))
    
train(net2,train_iter, test_iter, cross_entropy, num_epochs,updater2)
