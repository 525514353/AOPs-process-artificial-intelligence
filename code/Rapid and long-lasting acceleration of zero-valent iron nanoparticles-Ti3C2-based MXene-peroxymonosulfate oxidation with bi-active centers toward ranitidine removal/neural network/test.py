import torch
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import shap
def r2_adjust(y_test,y_predict,n,p):
    if n-p-1==0:
        return 0
    else:
        return 1-((1-r2_score(y_test,y_predict))*(n-1))/(n-p-1)


class nueral(nn.Module):
    def __init__(self):
        super(nueral, self).__init__()
        self.net=nn.Sequential(nn.Linear(4,8),nn.ELU(),nn.Linear(8,8),
                               nn.ELU(),nn.Linear(8,4),nn.Sigmoid(),
                               nn.Linear(4,1))
    def forward(self,x):
        return self.net(x)

data=pd.read_excel('data_new.xlsx')
s=StandardScaler()
x=s.fit_transform(data.iloc[:,:-1])
data.iloc[:,:-1]=x

net=nueral()
net.load_state_dict(torch.load('neural network2.pth'))
net.eval()


x_total=torch.from_numpy(data.iloc[:,:-1].values).type(torch.float32)
y_total=torch.from_numpy(data.iloc[:,-1].values).type(torch.float32).squeeze()
pre=net(x_total).squeeze()

x_train=torch.from_numpy(data.iloc[:-4,:-1].values).type(torch.float32)
y_train=torch.from_numpy(data.iloc[:-4,-1].values).type(torch.float32)


x_test=torch.from_numpy(data.iloc[-4:len(data)+1,:-1].values).type(torch.float32)
y_test=torch.from_numpy(data.iloc[-4:len(data)+1,-1].values).type(torch.float32).squeeze()

y_train_pre=net(x_train)
y_test_pre=net(x_test)

def plot():
    a=range(27)
    b=range(50,100)
    # plt.style.use('seaborn-whitegrid')
    plt.rc('font',family='Times New Roman',size=22)
    plt.rc('legend',fontsize=22)
    fig,ax=plt.subplots(dpi=300,figsize=(10,8))


    ax.scatter(y_train.detach().numpy(),y_train_pre.detach().numpy(),label='Train',linewidths=2)
    ax.scatter(y_test.detach().numpy(),y_test_pre.detach().numpy(),label='Test',linewidths=2)
    ax.legend()
    ax.plot(b,b,c='green')
    ax.set_xlabel('Predict value')
    ax.set_ylabel('True value')
    ax.axis('tight')
    ax.set_xlabel('True',fontsize=22)
    ax.set_ylabel('Predict',fontsize=22)

    plt.show()


    print(f'R_2total:{r2_score(y_total.detach().numpy(),pre.detach().numpy())}')
    print(f'R_2_adjust:{r2_adjust(y_total.detach().numpy(),pre.detach().numpy(),n=len(data),p=4)}')
    print(f'R_2predict:{r2_score(y_test.detach().numpy(),y_test_pre.detach().numpy())}')
    print(f'Test_mean_squared_error:{mean_squared_error(y_test.detach().numpy(),y_test_pre.detach().numpy())}')
    print(f'Total_mean_squared_error:{mean_squared_error(y_total.detach().numpy(),pre.detach().numpy())}')
    print(f'Residuals are {y_total.detach().numpy()-net(x_total).detach().numpy().squeeze()}')
    print(f'Residuals for train is {y_train.detach().numpy()-net(x_train).detach().numpy().squeeze()}')

def deep_shap_TOC(x_test=x_test):
    e=shap.DeepExplainer(net,x_train)
    # x_test = torch.tensor(s.inverse_transform(x_test.cpu()),dtype=torch.float32)
    shap_values=e.shap_values(torch.tensor(x_test,dtype=torch.float32))
    print(shap_values)
    print(x_test)
    feature_names = data.iloc[:, :-1].columns
    for i in range(1):
        shap.plots._waterfall.waterfall_legacy(e.expected_value[0], shap_values[i])
    # shap.summary_plot(shap_values, x_test, feature_names)
    # fig = plt.gcf()
    # fig.set_size_inches(10, 20)
    # fig.savefig('summary_plot.png',dpi=300)


def original_shap(x_train=x_train):
    net = nueral()
    net.load_state_dict(torch.load('neural network2.pth'))
    f=lambda x:net(torch.from_numpy(x).type(torch.float32)).cpu().detach().numpy()[:,-1]
    explainer=shap.Explainer(f,x_train.cpu().detach().numpy())
    shap_train=explainer(x_train.cpu().detach().numpy())
    shap_test=explainer(x_test.cpu().detach().numpy())
    feature_name = data.iloc[:, :-1].columns
    print(shap_test)
    for i in range(4):
        shap.plots.waterfall(shap_test[i])
        fig, ax = plt.gcf(), plt.gca()
        # ax.set_yticklabels('')
        # fig.set_size_inches(12, 6)
        fig.savefig(f'the{i}th.png',dpi=300)
        plt.show(block=True)


    x_train = torch.tensor(s.inverse_transform(x_train.cpu()), dtype=torch.float32)
    shap.summary_plot(shap_train,x_train,feature_name)
    # plt.show(block=True)
    fig = plt.gcf()
    # fig.set_size_inches(12, 8)
    fig.savefig(f'the summary for traun.png', dpi=300)
    plt.show(block=True)


original_shap()


