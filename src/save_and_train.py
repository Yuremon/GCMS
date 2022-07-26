import tools
import torch_gcmsdataset as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

dropout_rate1 = 0.3
dropout_rate2 = 0.15
num_epochs = 15
net = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout_rate2),nn.Linear(256,2), nn.Softmax(dim=1))
net.apply(tg.initial_weight)

net_6label = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout_rate2),nn.Linear(256,6), nn.Sigmoid())
if __name__ == '__main__': 
    path_train = "./data/train/"
    path_test = "./data/test/"
    #X_train_origin, y_train = tools.getDataTransformed(path_train + 'database.csv', path_train)
    #X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)
    #d = tg.GCMS_Data(X_train_origin, y_train)
    #test_dataset = tg.GCMS_Data(X_test_origin,y_test)
    #train_iter = DataLoader(d,batch_size=32, shuffle=True,)
    #test_iter = DataLoader(test_dataset,batch_size = 128, shuffle=False)
    
    
    trainer = torch.optim.Adagrad(net.parameters(), lr = 0.1)
    
    X_train_origin_6label, y_train_6label = tools.getDataTransformed(path_train + 'database.csv', path_train, binary=False)
    X_test_origin_6label, y_test_6label = tools.getDataTransformed(path_test + 'database.csv', path_test, binary=False)
    d_6label = tg.GCMS_Data(X_train_origin_6label, y_train_6label, binary = False)
    test_dataset_6label = tg.GCMS_Data(X_test_origin_6label,y_test_6label,binary= False)
    train_iter_6label = DataLoader(d_6label,batch_size=32, shuffle=True,)
    test_iter_6label = DataLoader(test_dataset_6label,batch_size = 128, shuffle=False)
    
    tg.train_6label(net_6label,train_iter_6label, test_iter_6label, F.cross_entropy, num_epochs,trainer)
    
    #tg.train(net,train_iter, test_iter, tg.cross_entropy, num_epochs,trainer)
    #torch.save(net.state_dict(),'dnp.params')

    