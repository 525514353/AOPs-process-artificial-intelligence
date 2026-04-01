import math
import pickle
import time
import optuna
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd
import torch
from torch import optim
from sklearn.model_selection import StratifiedKFold,KFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


pd.set_option('display.max_columns', None)

data=pd.read_excel('data_new.xlsx')
data.iloc[:,:-1]=StandardScaler().fit_transform(data.iloc[:, :-1])

x_total_train=torch.from_numpy(data.iloc[:-4,:-1].values).type(torch.float32).cuda()
y_total_train=torch.from_numpy(data.iloc[:-4,-1].values).type(torch.float32).squeeze().cuda()

x_test=torch.from_numpy(data.iloc[-4:len(data)+1,:-1].values).type(torch.float32).cuda()
y_test=torch.from_numpy(data.iloc[-4:len(data)+1,-1].values).type(torch.float32).squeeze().cuda()

class set(Dataset):
    def __init__(self,data):
        self.x=data.iloc[:-4,:-1].values
        self.y=data.iloc[:-4,-1].values
    def __getitem__(self, item):
        return self.x[item],self.y[item]
    def __len__(self):
        return len(self.x)

dataset=set(data=data)
dataLoader=DataLoader(dataset=dataset,batch_size=1,shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net=nn.Sequential(nn.Linear(4,8),nn.ELU(),nn.Linear(8,8),
                               nn.ELU(),nn.Linear(8,4),nn.Sigmoid(),
                               nn.Linear(4,1))
    def forward(self,x):
        return self.net(x)


net=MLP().cuda()
loss_func=nn.MSELoss().cuda()
optimizer=torch.optim.Adam(params=net.parameters(),lr=7e-4)

#超参数调优
def define_model(trial):
    # 定义超参数搜索范围
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layers = []
    in_features = 4
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

def objective(trial):
    start_time=time.time()
    # best_loss=10
    loss_item_train=[]
    loss_item_test=[]
    # r2_loss=[]
    net=define_model(trial).cuda()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
    n_epochs = trial.suggest_int("n_epochs", 2500,3500)
    for epoch in range(n_epochs):
        for step, (x_train, y_train) in enumerate(dataLoader):
            x_train = x_train.type(torch.float32).cuda()
            y_train=y_train.type(torch.float32).unsqueeze(-1).cuda()
            y_pre_train=net(x_train)
            loss_train=loss_func(y_pre_train, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # r2_temp=r2_score(y_train,y_pre_test)
        y_total_pre_train=net(x_total_train).squeeze()
        loss_total_train=loss_func(y_total_pre_train,y_total_train)
        y_total_pre_test=net(x_test).squeeze()
        loss_total_test=loss_func(y_total_pre_test,y_test)
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
    # study = optuna.create_study(direction="minimize")
    # study.optimize(objective, n_trials=30)
    # with open('example-study.pkl', 'wb') as f:
    #     pickle.dump(study, f)
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: {}".format(trial.value))
    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print("    {}: {}".format(key, value))
    # print('Best training time:', trial.user_attrs['training time'])
    #
    # optuna.visualization.matplotlib.plot_optimization_history(study)
    # plt.show(block=True)
    # optuna.visualization.matplotlib.plot_param_importances(study)
    # plt.show(block=True)
    # optuna.visualization.matplotlib.plot_slice(study)
    # plt.show(block=True)
    # optuna.visualization.matplotlib.plot_contour(study)
    # plt.show(block=True)
    # optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    # plt.show(block=True)

    with open('example-study.pkl', 'rb') as f:
        study = pickle.load(f)
    # fig=optuna.visualization.matplotlib.plot_optimization_history(study)
    # fig.legend(loc='upper right')
    # plt.show(block=True)
    optuna.visualization.matplotlib.plot_contour(study)
    plt.show(block=True)



def train():
    loss_item_train=[]
    loss_item_test=[]
    # r2_loss=[]
    for epoch in range(3000):
        for step, (x_train, y_train) in enumerate(dataLoader):
            x_train = x_train.type(torch.float32).cuda()
            y_train=y_train.type(torch.float32).unsqueeze(-1).cuda()
            y_pre_train=net(x_train)
            loss_train=loss_func(y_pre_train, y_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            # r2_temp=r2_score(y_train,y_pre_test)

        y_total_pre_train=net(x_total_train).squeeze()
        loss_total_train=loss_func(y_total_pre_train,y_total_train)
        y_total_pre_test=net(x_test).squeeze()
        loss_total_test=loss_func(y_total_pre_test,y_test)


        print(f'{epoch}th:train_loss is {loss_total_train},{epoch}th test_loss is {loss_total_test}')
        loss_item_train.append(loss_total_train.item())
        loss_item_test.append(loss_total_test.item())
        # r2_loss.append(r2_temp)


    plt.rc('font',family='Times New Roman')
    plt.rc('legend', fontsize=22)
    # print(loss_item_train)
    plt.figure(dpi=300)
    plt.plot(range(3000), loss_item_train,c='g',label='train_loss')
    plt.plot(range(3000), loss_item_test,c='r',label='test_loss')
    plt.xlabel('Iterations',fontsize=22)
    plt.ylabel('Mean-squared-error',fontsize=22)
    plt.legend()
    plt.show()
    # print(net(x_total_train))


    plt.figure(dpi=300)
    plt.plot(range(500), loss_item_train[-500:],c='g',label='train_loss')
    plt.plot(range(500), loss_item_test[-500:],c='r',label='test_loss')
    plt.xlabel('Iterations',fontsize=22)
    plt.ylabel('Mean-squared-error',fontsize=22)
    plt.xticks(range(0,500,50),range(2500,3000,50))
    plt.legend()
    plt.show()

    torch.save(net.state_dict(),'neural network4.pth')



