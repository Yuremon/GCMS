
from operator import index
import tools
import torch_gcmsdataset as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

class NormalMeanVar(nn.Module):
    def __init__(self, num_input) -> None:
        super().__init__()
        self.mean = Parameter(torch.normal(0,0.1,size=(1,num_input)))
        self.var = Parameter(torch.ones((1,num_input)))
    def forward(self, X):
        res = (X-self.mean)/(self.var + 1e-5)
        return res
    
    
def square_loss(X, y):
    return 0.5*torch.square(X)/X.shape[1]

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
   

def train(net, train_iter, loss, num_epochs, updater):
     for epoch in range(num_epochs): 
        train_epoch(net, train_iter, loss, updater)
    

if __name__ == "__main__":
    
    path_train = "./data/train/"
    path_test = "./data/test/"
    X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
    X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)
    index = (y_train > 0).nonzero()
    X_train_origin = X_train_origin[index]
    y_train = y_train[index]
    d = tg.GCMS_Data(X_train_origin, y_train)
    test_dataset = tg.GCMS_Data(X_test_origin,y_test)
    train_iter = DataLoader(d,batch_size=32, shuffle=True,)
    test_iter = DataLoader(test_dataset,batch_size = 128, shuffle=False)
    
    net = NormalMeanVar(504)
    trainer = torch.optim.Adagrad(net.parameters(), lr = 0.1)
    train(net, train_iter,square_loss,20, trainer)
    print(net.state_dict())
    torch.save(net.state_dict(),'regression.params')