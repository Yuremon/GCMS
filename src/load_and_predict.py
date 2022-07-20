
import tools
import torch_gcmsdataset as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

test= tools.readAndAdaptDataFromCSV("./data/all-data/","0").df['values'].to_numpy()
xc = test[:504]
def predict(net, X):
    xc = torch.tensor(X, dtype=torch.float32)
    y = net(xc.view((1,len(X))))
    return y

def result_dict(y):
    yr = y.view(-1)
    res = {"Normal": 0, "Non normal" : 0}
    res["Normal"] = yr[1].item()
    res["Non normal"] = yr[0].item()
    return res

dropout_rate1 = 0.3
dropout_rate2 = 0.15

net = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 256),nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout_rate2),nn.Linear(256,2), nn.Softmax(dim=1))
clone = net
clone.load_state_dict(torch.load('dnp.params'))
clone.eval()

if __name__ == '__main__':
    x = torch.normal(0,0.1,size=(1,504))
    y = predict(net,xc)
    print(result_dict(y))