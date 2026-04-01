from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axis3d import Axis


data=pd.read_excel('Electrochemical oxidation.xlsx')
data.iloc[:,:3]=StandardScaler().fit_transform(data.iloc[:,:3])


x_train=data.iloc[:-1,:3]
y_train_COD=data.iloc[:-1,3]
y_train_color_remove=data.iloc[:-1,4]
y_train_energy_consume=data.iloc[:-1,5]/100


x_test=data.iloc[-1:len(data)+1,:3]
y_test_COD=data.iloc[-1:len(data)+1,3]
y_test_color_remove=data.iloc[-1:len(data)+1,4]
y_test_energy_consume=data.iloc[-1:len(data)+1,5]/100

x_all=data.iloc[:len(data)+1,:3]
y_all_COD=data.iloc[:len(data)+1,3]
y_all_color_remove=data.iloc[:len(data)+1,4]
y_all_energy_consume=data.iloc[:len(data)+1,5]/100

pre_cod_train=[]
pre_cod_test=[]
pre_color_train=[]
pre_color_test=[]
pre_energy_train=[]
pre_energy_test=[]

def r2_adjust(y_test,y_predict,n,p):
    if n-p-1==0:
        return 0
    else:
        return 1-((1-r2_score(y_test,y_predict))*(n-1))/(n-p-1)

def main_COD():
    linear=LinearRegression()
    linear.fit(x_train,y_train_COD)

    print(f'R_2 for all dataset:{linear.score(x_all,y_all_COD)}')
    print(f'R_2adjust for all:{r2_adjust(y_all_COD,linear.predict(x_all),n=len(data),p=3)}')
    print(f'Mean-squared-error for all dataset:{mean_squared_error(y_all_COD,linear.predict(x_all))}')
    print(f'Mean-squared-error for test:{mean_squared_error(y_test_COD,linear.predict(x_test))}')
    print(f'{list(zip(data.columns.values[:3],linear.coef_))}')
    print(f'bias is {linear.intercept_}')
    print(f'predict for cod is{linear.predict(x_all)}')
    pre_cod_train.append(linear.predict(x_train))
    pre_cod_test.append(linear.predict(x_test))
    print(f'residual is{y_all_COD.values - [*linear.predict(x_train),*linear.predict(x_test)]}')

def main_Color():
    linear=LinearRegression()
    linear.fit(x_train,y_train_color_remove)

    print(f'R_2 for all dataset:{linear.score(x_all,y_all_color_remove)}')
    print(f'R_2adjust for all:{r2_adjust(y_all_color_remove,linear.predict(x_all),n=len(data),p=3)}')
    print(f'Mean-squared-error for all dataset:{mean_squared_error(y_all_color_remove,linear.predict(x_all))}')
    print(f'Mean-squared-error for test:{mean_squared_error(y_test_color_remove,linear.predict(x_test))}')
    print(f'{list(zip(data.columns.values[:3],linear.coef_))}')
    print(f'bias is {linear.intercept_}')
    print(f'predict for cod is{linear.predict(x_all)}')
    pre_color_train.append(linear.predict(x_train))
    pre_color_test.append(linear.predict(x_test))
    print(f'residual is{y_all_color_remove.values - [*linear.predict(x_train), *linear.predict(x_test)]}')


def main_energy():
    linear=LinearRegression()
    linear.fit(x_train,y_train_energy_consume)

    print(f'R_2 for all dataset:{linear.score(x_all,y_all_energy_consume)}')
    print(f'R_2adjust for all:{r2_adjust(y_all_energy_consume,linear.predict(x_all),n=len(data),p=3)}')
    print(f'Mean-squared-error for all dataset:{mean_squared_error(y_all_energy_consume,linear.predict(x_all))}')
    print(f'Mean-squared-error for test:{mean_squared_error(y_test_energy_consume,linear.predict(x_test))}')
    print(f'{list(zip(data.columns.values[:3],linear.coef_))}')
    print(f'bias is {linear.intercept_}')
    print(f'predict for cod is{linear.predict(x_all)}')
    pre_energy_train.append(linear.predict(x_train))
    pre_energy_test.append(linear.predict(x_test))
    print(f'residual is{y_all_energy_consume.values - [*linear.predict(x_train), *linear.predict(x_test)]}')


def matplot3D():
    plt.rc('font', family='Times New Roman', size=22)
    plt.rc('legend', fontsize=30)
    plt.figure(dpi=300, figsize=(26, 18))


    ax = plt.axes(projection='3d')
    # plt.style.use('ggplot')
    ax.scatter3D(y_train_COD, 1, pre_cod_train,label='Train_COD_removal', alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(y_test_COD, 1, pre_cod_test, label='Test_COD_removal',linewidths=8, alpha=1)

    ax.scatter3D(y_train_color_remove, 2, pre_color_train,label='Train_color_removal', alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(y_test_color_remove, 2, pre_color_test, label='Test_color_removal', linewidths=8, alpha=1)

    ax.scatter3D(y_train_energy_consume, 3, pre_energy_train,label='Train_energy_consume', alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(y_test_energy_consume, 3, pre_energy_test, label='Test_energy_consume', alpha=1, linewidths=8)

    ax.w_xaxis.set_pane_color((1, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1, 1.0, 1.0, 1.0))

    ax.view_init(12, 240)

    ax.legend(loc=(-85 / 200, 100 / 200),columnspacing=0.4)

    ax.plot3D(range(-40,200), [1] * 240, range(-40,200))
    ax.plot3D(range(-40,200), [2] * 240, range(-40,200))
    ax.plot3D(range(-40,200), [3] * 240, range(-40,200))
    ax.set(xlim=[-40, 200], ylim=[1, 3], zlim=[-40, 200])

    ax.set_yticks([1, 2, 3])
    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.3, 1, 0.3, 1]))
    ax.set_title('')
    plt.yticks(visible=False)
    plt.show()

main_COD()
main_Color()
main_energy()
matplot3D()
