
import src.tools
import src.torch_gcmsdataset as tg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

#test= src.tools.readAndAdaptDataFromCSV("../data/all-data/","10").df['values'].to_numpy()
#xc = test[:504]
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

def Label_dict(y):
    '''
    labels = [
    "normal",                          # 0
    "cétose",                          # 1
    "maladies métabolique connues",    # 2
    "métaboliques bactériens",         # 3
    "métabolique médicamenteux",       # 4
    "autre"                            # 5
    '''
    yr = y.view(-1)
    res = {"Normal": 0, "cétose" : 0, "maladies métabolique connues": 0, "métaboliques bactériens" : 0, "métabolique médicamenteux" : 0, "autre" : 0  }
    res["Normal"] = yr[0].item()
    res["cétose"] = yr[1].item()
    res["maladies métabolique connues"] = yr[2].item()
    res["métaboliques bactériens"] = yr[3].item()
    res["métabolique médicamenteux"] = yr[4].item()
    res["autre"] = yr[5].item() 
    return res
def MatrixConfusion_dict():
    '''
        true_non_normal : patient non normal qui est prédit non normal
        false_normal : patient non normal qui est prédit normal
        false_non_normal : patient normal qui est prédit non normal
        true_normal : patient normal qui est prédit normal
    '''
    res = { "true_non_normal": 0.3284671532846715, 
            "false_normal":0.145985401459854,
            "false_non_normal": 0.08759124087591241,  
            "true_normal":0.43795620437956206 }
    return res

path_net_mv = '../Parametre_net/netvm.params'
path_net_label = '../Parametre_net/netlabel.params'
path_net_type = '../Parametre_net/nettype.params'

dropout_rate1 = 0.5
dropout_rate2 = 0.3
net_mv = tg.NormalMeanVar(504)
net_label = nn.Sequential(nn.Linear(504,256),nn.BatchNorm1d(256),nn.ReLU(), nn.Dropout(dropout_rate1), nn.Linear(256, 128),nn.BatchNorm1d(128),nn.ReLU(),  nn.Dropout(dropout_rate2),nn.Linear(128,6),nn.Sigmoid())
net_type = nn.Sequential(nn.Linear(6,2), nn.Softmax(dim=1))  

net_mv.load_state_dict(torch.load(path_net_mv))
net_label.load_state_dict(torch.load(path_net_label))
net_type.load_state_dict(torch.load(path_net_type))
netforlabel = nn.Sequential(net_mv, net_label)
netfortype = nn.Sequential(net_mv, net_label, net_type)
netforlabel.eval()
netfortype.eval()
#clone = net
#clone.load_state_dict(torch.load('dnp.params'))
#clone.eval()
def calcul(path, name):
    x = src.tools.readAndAdaptDataFromCSV(path, name).df['values'].to_numpy()
    xc = x[:504]
    y = predict(netfortype,xc)
    y2 = predict(netforlabel,xc)

    labels6 = Label_dict(y2)
    type_patient = result_dict(y)

    return type_patient, labels6

    
if __name__ == '__main__':
    x = torch.normal(0,0.1,size=(1,504))
    y = predict(netfortype,xc)
    y2 = predict(netforlabel,xc)
    print("\n label")
    print(Label_dict(y2))
    print("\n type patient")
    print(result_dict(y))
    print("\n")
    print("matrix test")
    path_test = "./data/test/"
    X_test_origin, y_test = tools.getDataTransformed(path_test + 'database.csv', path_test)
    test_dataset = tg.GCMS_Data(X_test_origin,y_test)
    test_iter = DataLoader(test_dataset,batch_size = 128, shuffle=False)
    print(tg.Evaluate_Matrix_Confusion(netfortype, test_iter))
