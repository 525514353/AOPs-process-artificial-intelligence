import torch
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import shap
import numpy as np
# plt.rcParams.update({'font.size': 15})
from torch.autograd import Variable


class neural(nn.Module):
    def __init__(self):
        super(neural, self).__init__()
        self.input=nn.Sequential(nn.Linear(5,10),nn.ELU(),nn.Linear(10,5),
                                 nn.ELU(),nn.Sigmoid(),nn.Linear(5,5),nn.ELU(),nn.Linear(5,1))
    def forward(self,x):
        return self.input(x)

data=pd.read_excel('data.xlsx')
s=StandardScaler()
data.iloc[:,:5]=s.fit_transform(data.iloc[:,:5])

# net.eval()

x_total=torch.from_numpy(data.iloc[:,:5].values).type(torch.float32).cuda()
y_total_TOC=torch.from_numpy(data.iloc[:,5].values).type(torch.float32).squeeze().cuda()
y_total_IP=torch.from_numpy(data.iloc[:,6].values).type(torch.float32).squeeze().cuda()

x_train=torch.from_numpy(data.iloc[:-4,:5].values).type(torch.float32).cuda()
y_train_TOC=torch.from_numpy(data.iloc[:-4,5].values).type(torch.float32).cuda()
y_train_IP=torch.from_numpy(data.iloc[:-4,6].values).type(torch.float32).cuda()

x_test=torch.from_numpy(data.iloc[-4:len(data)+1,:5].values).type(torch.float32).cuda()
y_test_TOC=torch.from_numpy(data.iloc[-4:len(data)+1,5].values).type(torch.float32).cuda()
y_test_IP=torch.from_numpy(data.iloc[-4:len(data)+1,6].values).type(torch.float32).cuda()

def r2_adjust(y_all,y_predict,n,p):
    if n-p-1==0:
        return 0
    else:
        return 1-((1-r2_score(y_all,y_predict))*(n-1))/(n-p-1)

def plot_TOC():
    net = neural().cuda()
    net.load_state_dict(torch.load('net_TOC_new.pth'))
    y_train_pre=net(x_train).squeeze()
    y_test_pre=net(x_test).squeeze()
    y_total_pre=net(x_total).squeeze()
    linewith=range(10, 100)
    # plt.style.use('seaborn-whitegrid')
    plt.rc('font',family='Times New Roman',size=22)
    plt.rc('legend',fontsize=22)
    fig,ax=plt.subplots(dpi=300,figsize=(10,8))

    ax.scatter(y_train_TOC.cpu().detach().numpy(),y_train_pre.cpu().detach().numpy(),label='Train',linewidths=2)
    ax.scatter(y_test_TOC.cpu().detach().numpy(),y_test_pre.cpu().detach().numpy(),label='Test',linewidths=2)
    ax.legend()
    ax.plot(linewith, linewith, c='green')
    ax.set_xlabel('Predict value')
    ax.set_ylabel('True value')
    ax.axis('tight')
    ax.set_xlabel('True',fontsize=22)
    ax.set_ylabel('Predict',fontsize=22)
    plt.show()

    print(f'R_2total:{r2_score(y_total_TOC.cpu().detach().numpy(),y_total_pre.cpu().detach().numpy())}')
    print(f'R_2predict:{r2_score(y_test_TOC.cpu().detach().numpy(),y_test_pre.cpu().detach().numpy())}')
    print(f'R_2adjust:{r2_adjust(y_total_TOC.cpu().detach().numpy(), y_total_pre.cpu().detach().numpy(),n=len(data),p=5)}')
    print(f'Test_mean_squared_error:{mean_squared_error(y_test_TOC.cpu().detach().numpy(),y_test_pre.cpu().detach().numpy())}')
    print(f'Total_mean_squared_error:{mean_squared_error(y_total_TOC.cpu().detach().numpy(),y_total_pre.cpu().detach().numpy())}')
    print(f'Residuals are {y_total_TOC.cpu().detach().numpy()-net(x_total).cpu().detach().numpy().squeeze()}')
    print(f'Residuals for train is {y_train_TOC.cpu().detach().numpy()-net(x_train).cpu().detach().numpy().squeeze()}')

def plot_IP():
    net = neural().cuda()
    net.load_state_dict(torch.load('net_IP_new.pth'))
    y_train_pre=net(x_train).squeeze()
    y_test_pre=net(x_test).squeeze()
    y_total_pre=net(x_total).squeeze()
    linewith=range(50, 100)
    # plt.style.use('seaborn-whitegrid')
    plt.rc('font',family='Times New Roman',size=22)
    plt.rc('legend',fontsize=22)
    fig,ax=plt.subplots(dpi=300,figsize=(10,8))

    ax.scatter(y_train_IP.cpu().detach().numpy(),y_train_pre.cpu().detach().numpy(),label='Train',linewidths=2)
    ax.scatter(y_test_IP.cpu().detach().numpy(),y_test_pre.cpu().detach().numpy(),label='Test',linewidths=1)
    ax.legend()
    ax.plot(linewith, linewith, c='green')
    ax.set_xlabel('Predict value')
    ax.set_ylabel('True value')
    ax.axis('tight')
    ax.set_xlabel('True',fontsize=22)
    ax.set_ylabel('Predict',fontsize=22)
    plt.show()

    print(f'R_2total:{r2_score(y_total_IP.cpu().detach().numpy(),y_total_pre.cpu().detach().numpy())}')
    print(f'R_2predict:{r2_score(y_test_IP.cpu().detach().numpy(),y_test_pre.cpu().detach().numpy())}')
    print(f'R_2adjust:{r2_adjust(y_total_IP.cpu().detach().numpy(), y_total_pre.cpu().detach().numpy(), n=len(data), p=5)}')
    print(f'Test_mean_squared_error:{mean_squared_error(y_test_IP.cpu().detach().numpy(),y_test_pre.cpu().detach().numpy())}')
    print(f'Total_mean_squared_error:{mean_squared_error(y_total_IP.cpu().detach().numpy(),y_total_pre.cpu().detach().numpy())}')
    print(f'Residuals are {y_total_IP.cpu().detach().numpy()-net(x_total).cpu().detach().numpy().squeeze()}')
    print(f'Residuals for train is {y_train_IP.cpu().detach().numpy()-net(x_train).cpu().detach().numpy().squeeze()}')


# def kernel_shap(x_test=x_test):
#     torch.set_grad_enabled(True)
#     f = lambda x: net( Variable( torch.from_numpy(x) ) ).cpu().detach().numpy()
#     net = neural()
#     net.load_state_dict(torch.load('net_TOC_new.pth'))
#     e = shap.KernelExplainer(f, x_train.cpu().detach().numpy())
#     shap_values = e.shap_values(x_test.cpu().detach().numpy())
#     print(shap_values)
#     print(e.expected_value[0])
#     # shap.plots._waterfall.waterfall_legacy(e.expected_value[0], shap_values[0][0])
#     # x_test=s.inverse_transform(x_test.cpu())
#
#     feature_names=data.iloc[:,:5].columns
#     print(feature_names)
#     # shap.summary_plot(shap_values, x_test, feature_names)
#     shap.force_plot(e.expected_value, shap_values[0], feature_names)
#     plt.show(block=True)
# def deep_shap_TOC(x_test=x_test):
#     net=neural().cuda()
#     net.load_state_dict(torch.load('net_TOC_new.pth'))
#     e=shap.DeepExplainer(net,x_train)
#     # x_test = torch.tensor(s.inverse_transform(x_test.cpu()),dtype=torch.float32)
#     shap_values=e.shap_values(torch.tensor(x_test,dtype=torch.float32))
#     print(shap_values)
#     feature_names = data.iloc[:, :5].columns
#     # for i in range(4):
#     #     shap.plots._waterfall.waterfall_legacy(e.expected_value[0], shap_values[i])
#     #     plt.show(block=True)
#     shap.summary_plot(shap_values, x_test, feature_names)
#     fig = plt.gcf()
#     fig.set_size_inches(20, 8)
#     fig.savefig('summary_plot.png', dpi=300)
#     plt.show(block=True)
#
#
# def deep_shap_IP(x_test=x_test):
#     net=neural().cuda()
#     net.load_state_dict(torch.load('net_IP_new.pth'))
#     e=shap.DeepExplainer(net,x_train)
#     # x_test = torch.tensor(s.inverse_transform(x_test.cpu()),dtype=torch.float32)
#     shap_values=e.shap_values(torch.tensor(x_test,dtype=torch.float32))
#     print(shap_values)
#     feature_names = data.iloc[:, :5].columns
#     for i in range(4):
#         shap.plots._waterfall.waterfall_legacy(e.expected_value[0], shap_values[i])
#     shap.summary_plot(shap_values, x_test, feature_names)
#     plt.show(block=True)

def original_shap(model,x_train=x_train):
    net = neural()
    if model=='IP':
        net.load_state_dict(torch.load('net_IP_new.pth'))
    else:
        net.load_state_dict(torch.load('net_TOC_new.pth'))
    f=lambda x:net(torch.from_numpy(x).type(torch.float32)).cpu().detach().numpy()[:,-1]
    explainer=shap.Explainer(f,x_train.cpu().detach().numpy())
    shap_train=explainer(x_train.cpu().detach().numpy())
    shap_test=explainer(x_test.cpu().detach().numpy())
    feature_name=data.iloc[:,:-1].columns
    # for i in range(4):
    #     shap.plots.waterfall(shap_test[i])
    #     fig =plt.gcf()
    #     fig.savefig(f'the{i}th.png',dpi=300)
    #     plt.show(block=True)
    #
    # x_train = torch.tensor(s.inverse_transform(x_train.cpu()), dtype=torch.float32)
    # shap.summary_plot(shap_train,x_train,feature_name)
    # # plt.show(block=True)
    # fig = plt.gcf()
    # # fig.set_size_inches(48, 12)
    # fig.savefig(f'the summary for traun.png', dpi=300)
    # plt.show(block=True)
    shap.force_plot(shap_train[0],show=False,matplotlib=True)
    plt.show(block=True)
    # shap.plots.heatmap(shap_train)
    # plt.show(block=True)

original_shap('TOC')
