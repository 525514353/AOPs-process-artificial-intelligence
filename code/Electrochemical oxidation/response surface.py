import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score


pH=5.49
t=79.55
i=1.66

COD_test_pre=0.2624-6.795e-3*pH-1.725e-3*t-4.683e-2*i+5.4e-5*pH*t+1.253e-3*pH*i+7.065e-6*t*t+8.647e-3*i*i

Color_test_pre=[17.9609+0.5389*pH+1.3852*t+26.6639*i-0.8169*pH*i+0.0708*t*i-9.651e-3*t*t-5.8392*i*i]

Energy_test_pre=[0.1544-2.675e-3*t-0.2902*i+6.905e-3*t*i+0.0894*i*i]
Energy_test_pre=[i*100 for i in Energy_test_pre]


COD_test_pre=[(1/COD_test_pre)**2]


COD_train=[0.14271159300493,0.12480514407483,
0.12480514407483,0.12480514407483,0.12480514407483,
0.19000285006413,0.13327411355101,0.16878989451394,0.16552117772047,
0.10733469768527,0.18349396085439,0.11501092655706,0.11194341570991,
0.12480514407483,0.14586499149789,0.13176156917368,0.14571006315731]

COD_train=[(1/i)**2 for i in COD_train ]


COD_train_pre=[0.14111070585458,0.12610784108826,0.12610784108826,0.12610784108826,
0.12610784108826,0.19306577442081,0.12752178981157,0.17127057947124,
0.16569633168319,0.11455282659665,0.17982668751659,0.11132498900508,
0.11235394651415,0.12610784108826,0.14192226933426,0.13012061003903,0.14613925817572]

COD_train_pre=[(1/i)**2 for i in COD_train_pre ]


COD_test=[80]

Color_train=[75.4,89.9,89.9,89.9,
89.9,35,78.3,76.5,
45.6,98.9,58.5,99.1,
96.1,89.9,72.7,97.4,56.4]

Color_train_pre=[73.959828573556,91.263157894737,91.263157894737,
91.263157894737,91.263157894737,39.668639998552,80.123615259264,
71.282276689602,47.028418776875,97.629475959968,57.572139897118,
100.4113039357,94.06996536604,91.263157894737,74.68925473829,
95.418490003894,51.230801327455]

Color_test=[97.25]

Energy_train=[0.04,0.35,0.35,
0.35,0.35,0.01,0.99,
0.04,0.2,1.78,0.07,
0.62,0.62,0.35,0.07,0.99,0.07]

Energy_train= [i*100 for i in Energy_train]



Energy_train_pre=[0.04,0.34777777777778,
0.34777777777778,0.34777777777778,
0.34777777777778,0.077942492942195,
0.99,0.04,
0.26843842354761,1.7115615764524,
0.0045873195328738,0.69096823602268,
0.69096823602268,0.34777777777778,
0.0020575070578046,0.99,
0.0045873195328738]

Energy_train_pre=[i*100 for i in Energy_train_pre]



Energy_test=[0.679]
Energy_test=[0.679*100]


COD_all=[*COD_train,*COD_test]
COD_all_pre=[*COD_train_pre,*COD_test_pre]

Color_all=[*Color_train,*Color_test]
Color_all_pre=[*Color_train_pre,*Color_test_pre]

Energy_all=[*Energy_train,*Energy_test]
Energy_all_pre=[*Energy_train_pre,*Energy_test_pre]



def r2_adjust(y_all,y_predict,n,p):
    if n-p-1==0:
        return 0
    else:
        return 1-((1-r2_score(y_all,y_predict))*(n-1))/(n-p-1)


print(f'The R2 for all is {r2_score(COD_all, COD_all_pre)}')
print(f'The r2adjust for all is {r2_adjust(COD_all, COD_all_pre, n=len(COD_all), p=3)}')
print(f'The mean-squared-error for all is {mean_squared_error(COD_all, COD_all_pre)}')
print(f'The mean-squared-error for test is {mean_squared_error(COD_test, COD_test_pre)}')
print(f'the residual is {np.array(COD_all)-np.array(COD_all_pre)}')

print(f'The R2 for all is {r2_score(Color_all, Color_all_pre)}')
print(f'The r2adjust for all is {r2_adjust(Color_all, Color_all_pre, n=len(Color_all), p=3)}')
print(f'The mean-squared-error for all is {mean_squared_error(Color_all, Color_all_pre)}')
print(f'The mean-squared-error for test is {mean_squared_error(Color_test, Color_test_pre)}')
print(f'the residual is {np.array(Color_all)-np.array(Color_all_pre)}')

print(f'The R2 for all is {r2_score(Energy_all, Energy_all_pre)}')
print(f'The r2adjust for all is {r2_adjust(Energy_all, Energy_all_pre, n=len(Energy_all), p=3)}')
print(f'The mean-squared-error for all is {mean_squared_error(Energy_all, Energy_all_pre)}')
print(f'The mean-squared-error for test is {mean_squared_error(Energy_test, Energy_test_pre)}')
print(f'the residual is {np.array(Energy_all)-np.array(Energy_all_pre)}')

def plot_3D():
    plt.rc('font', family='Times New Roman', size=22)
    plt.rc('legend', fontsize=30)
    plt.figure(dpi=300, figsize=(26, 18))

    ax = plt.axes(projection='3d')

    ax.scatter3D(COD_train, 1, COD_train_pre,label='Train_COD_removal', alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(COD_test, 1, COD_test_pre, label='Test_COD_removal',linewidths=8, alpha=1)

    ax.scatter3D(Color_train, 2, Color_train_pre,label='Train_color_removal', alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(Color_test, 2, Color_test_pre, label='Test_color_removal', linewidths=8, alpha=1)

    ax.scatter3D(Energy_train, 3, Energy_train_pre,label='Train_energy_consume', alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(Energy_test, 3, Energy_test_pre, label='Test_energy_consume', alpha=1, linewidths=8)

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


plot_3D()