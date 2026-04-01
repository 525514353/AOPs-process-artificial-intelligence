import math
import pickle
import time

import matplotlib.pyplot as plt
import optuna
from torch import optim
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import StratifiedKFold,KFold


data=pd.read_excel('data.xlsx')

data.iloc[:,:5]=StandardScaler().fit_transform(data.iloc[:,:5])

x_train_all=torch.from_numpy(data.iloc[:-4,:5].values).type(torch.float32).cuda()
y_train_TOC=torch.from_numpy(data.iloc[:-4,5].values).type(torch.float32).cuda()
y_train_IP=torch.from_numpy(data.iloc[:-4,6].values).type(torch.float32).cuda()

x_test_all=torch.from_numpy(data.iloc[-4:len(data)+1,:5].values).type(torch.float32).cuda()
y_test_TOC=torch.from_numpy(data.iloc[-4:len(data)+1,5].values).type(torch.float32).cuda()
y_test_IP=torch.from_numpy(data.iloc[-4:len(data)+1,6].values).type(torch.float32).cuda()

class set_TOC(Dataset):
    def __init__(self):
        self.x=x_train_all
        self.y_TOC=y_train_TOC
    def __getitem__(self, item):
        return self.x[item],self.y_TOC[item]
    def __len__(self):
        return len(self.x)
class set_IP(Dataset):
    def __init__(self):
        self.x=x_train_all
        self.y_IP=y_train_IP
    def __getitem__(self, item):
        return self.x[item],self.y_IP[item]
    def __len__(self):
        return len(self.x)
dataset_TOC=set_TOC()
Dataset_IP=set_IP()
dataloader_TOC=DataLoader(dataset=dataset_TOC,batch_size=5,shuffle=True)
dataloader_IP=DataLoader(dataset=Dataset_IP,batch_size=1,shuffle=True)

class neural(nn.Module):
    def __init__(self):
        super(neural, self).__init__()
        self.input=nn.Sequential(nn.Linear(5,10),nn.ELU(),nn.Linear(10,5),
                                 nn.ELU(),nn.Sigmoid(),nn.Linear(5,5),nn.ELU(),nn.Linear(5,1))
    def forward(self,x):
        return self.input(x)


def define_model(trial):
    # 定义超参数搜索范围
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layers = []
    in_features = 5
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 12,step=2)
        layers.append(nn.Linear(in_features, out_features))
        activation=trial.suggest_categorical('activation{}'.format(i),['ReLU','ELU'])
        layers.append(getattr(nn,activation)())
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))
    # layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)
    return model

def objective_TOC(trial):
    loss_func=nn.MSELoss()
    start_time=time.time()
    # best_loss=10
    loss_item_train=[]
    loss_item_test=[]
    # r2_loss=[]
    net=define_model(trial).cuda()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
    n_epochs = trial.suggest_int("n_epochs", 400,800)
    for epoch in range(n_epochs):
        for step, (x_train, y_train) in enumerate(dataloader_TOC):
            x_train = x_train.type(torch.float32).cuda()
            y_train=y_train.type(torch.float32).unsqueeze(-1).cuda()
            y_pre_train=net(x_train)
            loss_train=loss_func(y_pre_train, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # r2_temp=r2_score(y_train,y_pre_test)
        y_total_pre_train=net(x_train_all).squeeze()
        loss_total_train=loss_func(y_total_pre_train,y_train_TOC)
        y_total_pre_test=net(x_test_all).squeeze()
        loss_total_test=loss_func(y_total_pre_test,y_test_TOC)
        # print(f'{epoch}th:train_loss is {loss_total_train},{epoch}th test_loss is {loss_total_test}')
        if math.isnan(loss_total_train):
            return float('nan')
        loss_item_train.append(loss_total_train.item())
        loss_item_test.append(loss_total_test.item())

    print(f'the final loss for train is {loss_item_train[-1]}')
    print(f'the final loss for test is {loss_item_test[-1]}')
    trial.set_user_attr("training time", time.time() - start_time)
    return loss_item_test[-1]
        # r2_loss.append(r2_temp)

def objective_IP(trial):
    loss_func=nn.MSELoss()
    start_time=time.time()
    # best_loss=10
    loss_item_train=[]
    loss_item_test=[]
    # r2_loss=[]
    net=define_model(trial).cuda()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
    n_epochs = trial.suggest_int("n_epochs", 200,300)
    for epoch in range(n_epochs):
        for step, (x_train, y_train) in enumerate(dataloader_IP):
            x_train = x_train.type(torch.float32).cuda()
            y_train=y_train.type(torch.float32).unsqueeze(-1).cuda()
            y_pre_train=net(x_train)
            loss_train=loss_func(y_pre_train, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # r2_temp=r2_score(y_train,y_pre_test)
        y_total_pre_train=net(x_train_all).squeeze()
        loss_total_train=loss_func(y_total_pre_train,y_train_IP)
        y_total_pre_test=net(x_test_all).squeeze()
        loss_total_test=loss_func(y_total_pre_test,y_test_IP)
        # print(f'{epoch}th:train_loss is {loss_total_train},{epoch}th test_loss is {loss_total_test}')
        if math.isnan(loss_total_train):
            return float('nan')
        loss_item_train.append(loss_total_train.item())
        loss_item_test.append(loss_total_test.item())

    print(f'the final loss for train is {loss_item_train[-1]}')
    print(f'the final loss for test is {loss_item_test[-1]}')
    trial.set_user_attr("training time", time.time() - start_time)
    return loss_item_test[-1]
        # r2_loss.append(r2_temp)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_IP, n_trials=30)
    with open('example-study_IP.pkl', 'wb') as f:
        pickle.dump(study, f)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print('Best training time:', trial.user_attrs['training time'])

    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig.legend(loc='upper right')
    plt.show(block=True)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.show(block=True)
    # optuna.visualization.matplotlib.plot_contour(study)
    # plt.show(block=True)



    # with open('example-study.pkl', 'rb') as f:
    #     study = pickle.load(f)
    # # fig=optuna.visualization.matplotlib.plot_optimization_history(study)
    # # fig.legend(loc='upper right')
    # # plt.show(block=True)
    # optuna.visualization.matplotlib.plot_contour(study)
    # plt.show(block=True)






# def train_TOC():
#     loss_func=nn.MSELoss().cuda()
#     max_epochs=1000
#     for i in range(30):
#         net=neural().cuda()
#         optimizer=torch.optim.Adam(params=net.parameters(),lr=7e-3,weight_decay=0.05)
#         print(f'The {i+1}th expirement begin！')
#         train_item = []
#         test_item = []
#         for epoch in range(max_epochs):
#             for step,(x,y) in enumerate(dataloader_TOC):
#                 predict=net(x).squeeze()
#                 loss=loss_func(predict,y)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             loss_train=loss_func(net(x_train).squeeze(),y_train_TOC)
#             loss_test=loss_func(net(x_test).squeeze(),y_test_TOC)
#             print(f'The {epoch+1}th loss_train is {loss_train},loss_test is {loss_test}')
#             train_item.append(loss_train.cpu().detach().numpy())
#             test_item.append(loss_test.cpu().detach().numpy())
#             if loss_test.cpu().detach().numpy() < 3 and loss_train<2:
#                 torch.save(net.state_dict(),'net_TOC_new.pth')
#                 plt.rc('font',family='Times New Roman',size=22)
#                 plt.rc('legend', fontsize=22)
#                 fig,axis=plt.subplots(dpi=300,figsize=(10,8))
#                 axis.plot(range(epoch+1),train_item,label='Train_loss')
#                 axis.plot(range(epoch+1),test_item,label='Test_loss')
#                 axis.legend()
#                 axis.set_xlabel('Iterations',fontsize=22)
#                 axis.set_ylabel('Mean-Squared-error',fontsize=22)
#                 plt.show()
#
#                 plt.figure(dpi=300,figsize=(10,8))
#                 plt.plot(range(epoch-50,epoch),train_item[-50:],label='Train_loss')
#                 plt.plot(range(epoch-50,epoch),test_item[-50:],label='Test_loss')
#                 plt.legend()
#                 plt.xlabel('Iterations',fontsize=22)
#                 plt.ylabel('Mean-squared-error',fontsize=22)
#                 plt.show()
#                 return
#
# def train_IP():
#     loss_func=nn.MSELoss().cuda()
#     max_epochs=1000
#     for i in range(30):
#         net=neural().cuda()
#         optimizer=torch.optim.Adam(params=net.parameters(),lr=7e-3,weight_decay=0.05)
#         print(f'The {i+1}th expirement begin！')
#         train_item = []
#         test_item = []
#         for epoch in range(max_epochs):
#             for step,(x,y) in enumerate(dataloader_IP):
#                 predict=net(x).squeeze()
#                 loss=loss_func(predict,y)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#             loss_train=loss_func(net(x_train).squeeze(),y_train_IP)
#             loss_test=loss_func(net(x_test).squeeze(),y_test_IP)
#             print(f'The {epoch+1}th loss_train is {loss_train},loss_test is {loss_test}')
#             train_item.append(loss_train.cpu().detach().numpy())
#             test_item.append(loss_test.cpu().detach().numpy())
#             if loss_test.cpu().detach().numpy() < 0.5 and loss_train<0.5:
#                 torch.save(net.state_dict(),'net_IP_new.pth')
#                 plt.rc('font',family='Times New Roman',size=22)
#                 plt.rc('legend', fontsize=22)
#                 fig,axis=plt.subplots(dpi=300,figsize=(10,8))
#                 axis.plot(range(epoch+1),train_item,label='Train_loss')
#                 axis.plot(range(epoch+1),test_item,label='Test_loss')
#                 axis.legend()
#                 axis.set_xlabel('Iterations',fontsize=22)
#                 axis.set_ylabel('Mean-Squared-error',fontsize=22)
#                 plt.show()
#
#                 plt.figure(dpi=300,figsize=(10,8))
#                 plt.plot(range(epoch-50,epoch),train_item[-50:],label='Train_loss')
#                 plt.plot(range(epoch-50,epoch),test_item[-50:],label='Test_loss')
#                 plt.legend()
#                 plt.xlabel('Iterations',fontsize=22)
#                 plt.ylabel('Mean-squared-error',fontsize=22)
#                 plt.show()
#                 return



