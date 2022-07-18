import tools
import torch_gcmsdataset as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
class convolutionlayer(nn.Module):
    def __init__(self, nb_pass_in, nb_pass_out, start = False) -> None:
        super().__init__()
        self.start = start
        self.c1 = nn.Conv1d(nb_pass_in, nb_pass_out, kernel_size=1)
        self.bn = nn.BatchNorm1d(nb_pass_out)
    def forward(self, X):
        if self.start:
            X = X.view(1,1,X.shape[0])
            y = self.c1(X)
            return self.bn(y)
        else:
            y = self.c1(X)
            y = self.bn(y)
            return y.view(-1)
class Residual(torch.nn.Module):
    '''
    resnet
    '''
    def __init__(self, n_input, n_output, use_1x1conv=False, strides = 1) -> None:
        super().__init__()
        self.c1 = torch.nn.Conv1d(n_input,n_output,kernel_size=1, stride= strides)
        self.c2 = torch.nn.Conv1d(n_output,n_output,kernel_size=1,)
        if use_1x1conv:
            self.c3 = torch.nn.Conv1d(n_input, n_output, kernel_size=1, stride = strides)
        else:
            self.c3 = None
        self.relu = torch.nn.ReLU(inplace=True)
        self.bn1 = torch.nn.BatchNorm1d(n_output)# à tester 
        self.bn2 = torch.nn.BatchNorm1d(n_output)# à tester
    def forward(self,x):
        Y = self.relu(self.bn1(self.c1(x)))
        Y = self.bn2(self.c2(Y))
        if self.c3:
            x = self.c3(x)
        Y += x
        return self.relu(Y)

class resnet(nn.Module):
    def __init__(self, num_input, num_output) -> None:
        super().__init__()
        self.l1 = mylinear(num_input,num_output)
        
    def forward(self, X):
        y = F.relu(self.l1.forward(X))
        y = y+X
        return F.relu(y)
class mylinear(nn.Module):
    def __init__(self, num_input, num_output) -> None:
        super().__init__()
        self.w = Parameter(torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True))
        self.b = Parameter(torch.zeros(num_output,requires_grad=True))
    def forward(self, X):
        return torch.matmul(X,self.w)+self.b

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
        partition = x_exp.sum(1, keepdim=True)    
        return x_exp/partition
    
    
def sgd(params, lr, batch_size):
    """小批量随机梯度下降
    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            

            
def softmax(x):
    x_exp = torch.sigmoid(x)
    partition = x_exp.sum()    
    return x_exp/partition
def net(x):
    
    temp = torch.matmul(x,w)+b
    #temp2 = F.linear(x,w.transpose(0,1))
    test= nn.Sequential(softmaxlayer())
    testres = test(temp)
    res = softmax(temp)
    return res
def cross_entropy(y_hat,y):
    '''
    y_hat n éléments chaque élément -> probabilté de chaque class
    y list de class des n élément -> indice de class
    '''
    return -torch.log(y_hat[range(len(y_hat)),y])#-torch.log(y_hat[y])#
    
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
    


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    #metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(x)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            #metric.add(float(l.sum()), accuracy(y_hat,y), y.numel())         
        else:
           l.sum().backward()
           updater(x.shape[0])
           #metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    #return metric[0]/metric[2], metric[1]/metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs): 
        train_epoch(net, train_iter, loss, updater)
    #test_acc = evaluate_accuracy(net, test_iter)
    #train_loss, train_acc = train_metrics
    
    test_acc = evaluate_accuracy(net, test_iter)
    print(test_acc)
    print("\n")
    
    

num_input = 504
num_output = 2
lr = 0.05
w = torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)

b = torch.zeros(num_output,requires_grad=True)
num_epochs = 100
def updater(batch_size):
    return sgd([w, b], lr, batch_size)
def updater2(batch_size):
    return sgd(net2.parameters(), lr, batch_size)

def initial_weight(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.normal_(module.weight, std=0.1)
       
        torch.nn.init.zeros_(module.bias)

path_train = "./data/train/"
path_test = "./data/test/"

X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)
train_dataset = tg.GCMS_Data(X_train_origin,y_train)
test_dataset = tg.GCMS_Data(X_test_origin,y_test)
train_iter = DataLoader(train_dataset,batch_size= 25, shuffle=True)
test_iter = DataLoader(test_dataset,batch_size = 128, shuffle=False)
#train_iter = zip(torch.tensor(X_train_origin,dtype=torch.float32),torch.tensor(y_train))
#test_iter = zip(torch.tensor(X_test_origin,dtype=torch.float32),torch.tensor(y_test))

net2 = nn.Sequential(nn.BatchNorm1d(num_input) ,resnet(num_input, num_input),nn.BatchNorm1d(num_input), nn.Linear(num_input,num_output), softmaxlayer())

#net2 = torch.nn.Sequential(convolutionlayer(1,2,start=True), Residual(2,2), Residual(2,6, use_1x1conv=True), convolutionlayer(6,1), mylinear(num_input,num_output),softmaxlayer())
trainer = torch.optim.SGD(net2.parameters(), lr=0.1, momentum= 0.9)
#net2 = nn.Sequential(convolutionlayer(), mylinear(num_input,num_output),softmaxlayer())
train(net2,train_iter, test_iter, cross_entropy, num_epochs,trainer)

#net.state_dict()
#torch.save(net.state_dict(),'mlp.params')
#clone = net2 
#clone.load_state_dict(torch.load('mlp.params'))
#clone.eval

#重写convolution 重写resnet 重写linear 全连接层