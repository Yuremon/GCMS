
import tools
import torch_gcmsdataset as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

def predict(net, X):
    y = net(X)
    return y

def result_dict(y):
    res = {"Normal": 0, "Non normal" : 0}
    res["Normal"] = y[0]
    res["Non normal"] = y[1]
    return res

dropout_rate1 = 0.3
dropout_rate2 = 0.15

net = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout_rate2),nn.Linear(256,2), nn.Softmax(dim=1))
clone = net
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()