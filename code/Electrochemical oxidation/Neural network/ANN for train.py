import math
import pickle

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import time
import optuna
from torch import optim
data=pd.read_excel('Electrochemical oxidation.xlsx')

data.iloc[:,:3]=StandardScaler().fit_transform(data.iloc[:,:3])

x_train=torch.from_numpy(data.iloc[:-1,:3].values).type(torch.float32).cuda()
y_train_COD=torch.from_numpy(data.iloc[:-1,3].values).type(torch.float32).cuda()
y_train_color_remove=torch.from_numpy(data.iloc[:-1,4].values).type(torch.float32).cuda()
y_train_energy_consume=torch.from_numpy(data.iloc[:-1,5].values).type(torch.float32).cuda()

x_test=torch.from_numpy(data.iloc[-1:len(data)+1,:3].values).type(torch.float32).cuda()
y_test_COD=torch.from_numpy(data.iloc[-1:len(data)+1,3].values).type(torch.float32).cuda()
y_test_color_remove=torch.from_numpy(data.iloc[-1:len(data)+1,4].values).type(torch.float32).cuda()
y_test_energy_consume=torch.from_numpy(data.iloc[-1:len(data)+1,5].values).type(torch.float32).cuda()



class muti_task(Dataset):
    def __init__(self):
        self.x=x_train
        self.y_train_COD=y_train_COD
        self.y_train_color_remove = y_train_color_remove
        self.y_train_energy_consume=y_train_energy_consume
    def __getitem__(self, item):
        return self.x[item],self.y_train_COD[item],self.y_train_color_remove[item],self.y_train_energy_consume[item]
    def __len__(self):
        return len(self.x)

dataset_muti_task=muti_task()
dataloader_muti_task=DataLoader(dataset=dataset_muti_task,batch_size=1,shuffle=True)


class neural_mutitask(nn.Module):
    def __init__(self):
        super(neural_mutitask, self).__init__()
        self.input=nn.Sequential(nn.Linear(3,5),nn.ELU(),nn.Linear(5,5),
                                 nn.ELU(),nn.Sigmoid(),nn.Linear(5,5),nn.ELU(),nn.Linear(5,3))
    def forward(self,x):
        return self.input(x)
def train_mutitask(loop=10):
    for i in range(loop):
        loss_func=nn.MSELoss().cuda()
        max_epochs=5000
        net=neural_mutitask().cuda()
        optimizer=torch.optim.Adam(params=net.parameters(),lr=5e-3,weight_decay=0.05)
        print(f'The {i+1}th expirement begin！')
        train_item = []
        test_item = []
        for epoch in range(max_epochs):
            for step,(x,y_COD,y_color,y_energy) in enumerate(dataloader_muti_task):
                predict=net(x).squeeze()
                y_matrix=torch.concat((y_COD,y_color,y_energy)).type(torch.float32).cuda().squeeze()
                loss=loss_func(predict,y_matrix)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_train=loss_func(net(x_train).flatten(),torch.stack((y_train_COD,y_train_color_remove,y_train_energy_consume),dim=1).flatten())
            loss_test=loss_func(net(x_test).flatten(),torch.stack((y_test_COD,y_test_color_remove,y_test_energy_consume),dim=1).flatten())
            train_item.append(loss_train.cpu().detach().numpy())
            test_item.append(loss_test.cpu().detach().numpy())
            print(f'The {epoch+1}th loss_train is {loss_train},loss_test is {loss_test}')
            if loss_test < 1.5 and loss_train < 1.5:
                torch.save(net.state_dict(), 'mutitask ANN model.pth')
                plt.rc('font',family='Times New Roman',size=22)
                plt.rc('legend', fontsize=22)
                fig,axis=plt.subplots(dpi=300,figsize=(10,8))
                axis.plot(range(epoch+1),train_item,label='Train_loss')
                axis.plot(range(epoch+1),test_item,label='Test_loss')
                axis.legend()
                axis.set_xlabel('Iterations',fontsize=22)
                axis.set_ylabel('Mean-Squared-error',fontsize=22)
                plt.show()

                plt.figure(dpi=300,figsize=(10,8))
                plt.plot(range(epoch-500,epoch),train_item[-500:],label='Train_loss')
                plt.plot(range(epoch-500,epoch),test_item[-500:],label='Test_loss')
                plt.legend()
                plt.xlabel('Iterations',fontsize=22)
                plt.ylabel('Mean-squared-error',fontsize=22)
                plt.show()
                return

def define_model(trial):
    # 定义超参数搜索范围
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layers = []
    in_features = 3
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 3, 7,step=2)
        layers.append(nn.Linear(in_features, out_features))
        activation=trial.suggest_categorical('activation{}'.format(i),['ReLU','ELU'])
        layers.append(getattr(nn,activation)())
        in_features = out_features
    layers.append(nn.Linear(in_features, 3))
    # layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)
    return model

def objective(trial):
    loss_func=nn.MSELoss()
    start_time=time.time()
    # best_loss=10
    train_item=[]
    test_item=[]
    # r2_loss=[]
    net=define_model(trial).cuda()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(net.parameters(), lr=lr)
    n_epochs = trial.suggest_int("n_epochs", 3500,4500)
    for epoch in range(n_epochs):
        for step, (x, y_COD, y_color, y_energy) in enumerate(dataloader_muti_task):
            predict = net(x).squeeze()
            y_matrix = torch.concat((y_COD, y_color, y_energy)).type(torch.float32).cuda()

            loss = loss_func(predict, y_matrix)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # r2_temp=r2_score(y_train,y_pre_test)
        loss_train = loss_func(net(x_train).flatten(),
                               torch.stack((y_train_COD, y_train_color_remove, y_train_energy_consume),
                                           dim=1).flatten())
        loss_test = loss_func(net(x_test).flatten(),
                              torch.stack((y_test_COD, y_test_color_remove, y_test_energy_consume), dim=1).flatten())

        # print(f'{epoch}th:train_loss is {loss_train},{epoch}th test_loss is {loss_test}')
        if math.isnan(loss_train):
            return float('nan')
    #     train_item.append(loss_train.cpu().detach().numpy())
    #     test_item.append(loss_test.cpu().detach().numpy())
    #
    # print(f'the final loss for train is {train_item[-1]}')
    # print(f'the final loss for test is {test_item[-1]}')
    # trial.set_user_attr("training time", time.time() - start_time)

    loss_COD_test = loss_func(net(x_test)[:,0], y_test_COD)
    loss_Color = loss_func(net(x_test)[:,1], y_test_color_remove)
    loss_energy = loss_func(net(x_test)[:,2], y_test_energy_consume)

    return (loss_COD_test,loss_Color,loss_energy)

if __name__ == "__main__":
    # sampler = optuna.samplers.NSGAIISampler()
    # study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)
    # study.optimize(objective, n_trials=30)
    #
    # with open('example-study.pkl', 'wb') as f:
    #     pickle.dump(study, f)
    #
    # print("Best params:", study.best_trials)
    # # print("Best values:", study.best_values)
    # optuna.visualization.matplotlib.plot_pareto_front(study, target_names=["COD", "Color",'Energy'])
    # plt.show(block=True)

    with open('example-study.pkl', 'rb') as f:
        study = pickle.load(f)
        fig=optuna.visualization.matplotlib.plot_pareto_front(study, target_names=["COD", "Color", 'Energy'])
        fig.patch.set_facecolor('white')
        # fig.axes[0].tick_params(axis='both', labelsize=12)
        plt.show(block=True)
        # fig=optuna.visualization.matplotlib.plot_optimization_history(study,target=lambda t: t.values[0])
        #
        # fig.legend(loc='upper right')
        # plt.show(block=True)
        # fig=optuna.visualization.matplotlib.plot_optimization_history(study,target=lambda t: t.values[1])
        #
        # fig.legend(loc='upper right')
        # plt.show(block=True)
        # fig=optuna.visualization.matplotlib.plot_optimization_history(study,target=lambda t: t.values[2])
        #
        # fig.legend(loc='upper right')
        # plt.show(block=True)

