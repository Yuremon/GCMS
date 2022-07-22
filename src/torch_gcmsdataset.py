from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tools
import torch

###################class
class Accumulator:
    def __init__(self,n) -> None:
        self.data = [0.0]*n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
class GCMS_Data(Dataset):
    def __init__(self, x_train, y_train, binary = True) -> None:
        super().__init__()
        self.train = x_train
        self.type = y_train
        self.binary = binary
    def __len__(self):
        return len(self.train)
    def __getitem__(self, index):
        if self.binary:
            trainset = (torch.tensor(self.train[index],dtype=torch.float32),torch.tensor(self.type[index]))
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
        cmpn = y_hat.type(y.dtype) == torch.zeros_like(y) # nombre non normal de prédiction
        res_non_normal = cmp == cmpn #prédit vraiement non normal  
        
        true_non_normal = float(res_non_normal.type(y.dtype).sum())
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
    



def K_fold_data(k, i, x, y):
    
    assert k > 1
    train_size = len(y)
    fold_size = train_size//k
    
    x_trian,y_train = None, None
    for j in range(k):
        ind = slice(j*fold_size, (j+1)*fold_size)
        x_seq, y_seq = x[ind], y[ind]
        if j == i:
            x_validation = x_seq
            y_validation = y_seq
        elif x_trian is None:
            x_trian, y_train = x_seq, y_seq
        else:
            x_trian = x_trian + x_seq
    return x_trian, y_train, x_validation, y_validation

def k_fold_train(net, k, x, y, loss, num_epochs, updater):
    for ki in k:
        x_train, y_train, x_validation, y_validation = K_fold_data(k, ki, x, y)
        train_dataset = GCMS_Data(x_train, y_train)
        validation_dataset = GCMS_Data(x_validation, y_validation)
        train_batch = DataLoader(train_dataset, batch_size= 25, shuffle=True)
        validation_batch = DataLoader(validation_dataset, batch_size=len(y_validation), shuffle=False)
        train(net, train_batch, validation_batch, loss, num_epochs, updater)
    return       

def initial_weight(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.normal_(module.weight, std=0.1)
       
        torch.nn.init.zeros_(module.bias)


if __name__ == '__main__':
    path_train = "../data/train/"
    path_test = "../data/test/"
    X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train,binary = False)
    X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test,binary = False)
    d = GCMS_Data(X_train_origin, y_train, binary=False)
    test_dataset = GCMS_Data(X_test_origin,y_test, binary=False)
    train_iter = DataLoader(d,batch_size=32, shuffle=True,)
    test_iter = DataLoader(test_dataset,batch_size = 128, shuffle=False)
    for X,y in test_iter:
        print("type = ", y)
        
         
        
        

