from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch_Person_testonly as tpt
import tools
import torch
import torch_gcmsdataset as tg

x = torch.normal(0, 0.1, size=(2,6))
y_pre = torch.sigmoid(x)

type = torch.tensor([[1.,0.,0.,0.,0.,0.],[0.,1.,1.,0.,0.,1.]])

F.cross_entropy(y_pre, type)

prenet = tpt.regression(504)
prenet.load_state_dict(torch.load("regression.params"))
print(prenet.state_dict())
prenet.eval()
dropout_rate1 = 0.3
dropout_rate2 = 0.15
num_epochs = 20
net = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout_rate2),nn.Linear(256,2), nn.Softmax(dim=1))
net.apply(tg.initial_weight)
def train_epoch_test(prenet, net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    #metric = Accumulator(3)
    for x, y in train_iter:
        y_hat = net(prenet(x))
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
            #metric.add(float(l.sum()), accuracy(y_hat,y), y.numel())         
        else:
           l.sum().backward()
           updater(x.shape[0])
def test_train(prenet, net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs): 
        train_epoch_test(prenet, net, train_iter, loss, updater)
    #test_acc = evaluate_accuracy(net, test_iter)
    #train_loss, train_acc = train_metrics
    seq = nn.Sequential(prenet, net)
    test_acc = tg.evaluate_accuracy(seq, test_iter)
    print(test_acc)
    print("\n")
    
    
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
    test_train(prenet, net,train_iter, test_iter, tg.cross_entropy, num_epochs,trainer)
    