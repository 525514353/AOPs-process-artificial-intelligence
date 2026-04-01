import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score


T=260
t=[0,30,60,90]
m=2.5
H=5.7
TOC_test_pre=[]
for i in t:
    EQU_TOC=-864.2047+5.91642*T+2.3739*i+0.5247*m-7.84e-3*T*i-9.189e-3*T*T-1.679e-3*i*i
    TOC_test_pre.append(EQU_TOC)

IP_test_pre=[]
for i in t:
    EQU_IP=-1242.1923+9.8191*T+2.4389*i-0.3192*H-8.757e-3*T*i-1.793e-2*T*T-1.403e-3*i*i
    IP_test_pre.append(EQU_IP)

TOC_train=[67.1,66.6,68,58.6,61.6,61.3,28.6,44.9,48.9,
            60.7,56,36.5,59.8,74.2,17,69.3,54.7,21.9,63.1,54.3,22.7,
            63.8,63.8,37.8,48.6,57.3,66.9,44.2,68.7,73,
            58,55.5,37.4,29.4,27.5,54.1,64.7,62.3,68.2,43.8,56.2,67,
            66,71.6,30.4,61.2,74.2,55.5,29,54.9]

TOC_train_pre=[64.636234313331,56.837531803484,68.670384234835,53.742012183278,
64.601779214539,61.978457502499,30.790299436654,48.97140445695,
51.594726168989,51.594726168989,56.837531803484,30.790299436654,
56.837531803484,76.964627437535,13.28718056617,68.670384234835,
56.837531803484,28.166977724615,64.601779214539,56.837531803484,
28.166977724615,61.978457502499,61.978457502499,48.97140445695,
48.97140445695,56.837531803484,64.601779214539,51.594726168989,
68.670384234835,71.293705946874,56.837531803484,59.93305142369,
30.790299436654,28.166977724615,30.790299436654,48.97140445695,
64.601779214539,56.837531803484,68.670384234835,56.837531803484,
56.837531803484,71.293705946874,61.978457502499,71.293705946874,
31.915573690373,56.837531803484,71.293705946874,51.594726168989,
28.166977724615,56.837531803484]

TOC_test=[60.6,62.85,70.63,71.64]


IP_train=[98.6,99.6,98.9,99.1,98.4,98.5,
75.4,87.7,90.8,90.9,97.4,71.7,99,99.2,52.2,
98.8,96.7,69.5,98.5,95.7,71.6,98.5,98.5,
88,87.5,96.8,98.7,89.9,98.2,99.5,95.2,
98,75.8,67.2,75.8,90.1,98.8,95.6,98.9,
97.4,97.5,98.7,98.1,99.2,84.2,98.5,99.1,
89.2,77.7,96.9]


IP_train_pre=[98.123649473545,97.63624528169,100.41396623283,96.104259987573,
99.311355528717,100.58800994048,74.595942193394,88.90774407398,
88.90774407398,90.184398485745,96.104259987573,73.31928778163,
96.104259987573,94.803947104682,51.700200866614,99.137311821067,
96.104259987573,73.31928778163,100.58800994048,96.104259987573,
74.595942193394,100.58800994048,99.311355528717,90.184398485745,
88.90774407398,96.104259987573,100.58800994048,90.184398485745,
99.137311821067,99.137311821067,96.104259987573,96.104259987573,73.31928778163,
73.31928778163,74.595942193394,90.184398485745,99.311355528717,
96.104259987573,100.41396623283,96.104259987573,96.104259987573,99.137311821067,99.311355528717,100.41396623283,
79.780498497751,96.104259987573,100.41396623283,88.90774407398,74.595942193394,94.572274693455,
]

IP_test=[97.35,98.59,98.60,99.54]

TOC_all=[*TOC_train,*TOC_test]
TOC_all_pre=[*TOC_train_pre,*TOC_test_pre]

IP_all=[*IP_train,*IP_test]
IP_all_pre=[*IP_train_pre,*IP_test_pre]

def r2_adjust(y_all,y_predict,n,p):
    if n-p-1==0:
        return 0
    else:
        return 1-((1-r2_score(y_all,y_predict))*(n-1))/(n-p-1)

def plot_TOC():
    plt.rc('font',family='Times New Roman',size=22)
    plt.rc('legend',fontsize=22)
    plt.figure(dpi=300,figsize=(10,8))
    plt.scatter(TOC_train,TOC_train_pre,label='Train')
    plt.scatter(TOC_test,TOC_test_pre,label='Test')
    plt.legend()
    # plt.xticks(range(50,110,10),range(50,110,10))
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.plot(range(10,100),range(10,100))
    plt.show()
    print(f'The R2 for all is {r2_score(TOC_all,TOC_all_pre)}')
    print(f'The R2 for test is {r2_score(TOC_test,TOC_test_pre)}')
    print(f'The R2adjust for all is {r2_adjust(TOC_all,TOC_all_pre,n=len(TOC_all),p=5)}')
    print(f'The mean-squared-error for all is {mean_squared_error(TOC_all,TOC_all_pre)}')
    print(f'The mean-squared-error for test is {mean_squared_error(TOC_test,TOC_test_pre)}')

def plot_IP():
    plt.rc('font',family='Times New Roman',size=22)
    plt.rc('legend',fontsize=22)
    plt.figure(dpi=300,figsize=(10,8))
    plt.scatter(IP_train,IP_train_pre,label='Train')
    plt.scatter(IP_test,IP_test_pre,label='Test')
    plt.legend()
    plt.xticks(range(50,110,10),range(50,110,10))
    plt.xlabel('True')
    plt.ylabel('Predict')
    plt.plot(range(50,100),range(50,100))
    plt.show()
    print(f'The R2 for all is {r2_score(IP_all,IP_all_pre)}')
    print(f'The r2 for test is {r2_score(IP_test,IP_test_pre)}')
    print(f'The r2adjust for all is {r2_adjust(IP_all,IP_all_pre,n=len(IP_all),p=5)}')
    print(f'The mean-squared-error for all is {mean_squared_error(IP_all,IP_all_pre)}')
    print(f'The mean-squared-error for test is {mean_squared_error(IP_test,IP_test_pre)}')

def residual():
    residual_list_TOC=np.array([*TOC_train,*TOC_test])-np.array([*TOC_train_pre,*TOC_test_pre])
    residual_list_IP = np.array([*IP_train, *IP_test]) - np.array([*IP_train_pre, *IP_test_pre])
    print(f'residual for TOC is {residual_list_TOC}')
    print(f'residual for IP is {residual_list_IP}')

residual()