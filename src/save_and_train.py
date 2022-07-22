import tools
import torch_gcmsdataset as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

class DNP(nn.Module):
    def __init__(self, num_input, hide_num) -> None:
        super().__init__()
        self.weights = Parameter(torch.zeros((num_input, hide_num), requires_grad=True))
        self.mask = Parameter(torch.zeros(num_input))
    def forward(self, X):
        return X*self.weights
    def add_feature(self, index):
        self.mask[index] = 1.
        self.weights = self.weights*self.mask
    def maxgradindex(self):
        grads = self.weights.grad
#net.state_dict()
#torch.save(net.state_dict(),'mlp.params')
#clone = net2 
#clone.load_state_dict(torch.load('mlp.params'))
#clone.eval
dropout_rate1 = 0.3
dropout_rate2 = 0.15
num_epochs = 15
net = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout_rate2),nn.Linear(256,2), nn.Softmax(dim=1))
net.apply(tg.initial_weight)

net_6label = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout_rate2),nn.Linear(256,6), nn.Sigmoid(dim=1))
if __name__ == '__main__': 
    path_train = "./data/train/"
    path_test = "./data/test/"
    X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
    X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)
    
    d = tg.GCMS_Data(X_train_origin, y_train)
    test_dataset = tg.GCMS_Data(X_test_origin,y_test)
    
    train_iter = DataLoader(d,batch_size=32, shuffle=True,)
    test_iter = DataLoader(test_dataset,batch_size = 128, shuffle=False)
    trainer = torch.optim.Adagrad(net.parameters(), lr = 0.1)
    
    
    #tg.train(net,train_iter, test_iter, tg.cross_entropy, num_epochs,trainer)
    #torch.save(net.state_dict(),'dnp.params')
    
    #使用注意力机制去拟合函数504的散点函数 三维 index -> 均值方差
    #网络504 -> 6 
    #网络 6-> 2