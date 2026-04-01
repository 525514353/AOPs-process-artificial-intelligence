import torch
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import shap
import os
import re


class neural_mutitask(nn.Module):
    def __init__(self):
        super(neural_mutitask, self).__init__()
        self.input=nn.Sequential(nn.Linear(3,5),nn.ELU(),nn.Linear(5,5),
                                 nn.ELU(),nn.Sigmoid(),nn.Linear(5,5),nn.ELU(),nn.Linear(5,3))
    def forward(self,x):
        return self.input(x)


net=neural_mutitask().cuda()
net.load_state_dict(torch.load('mutitask ANN model.pth'))
# net.eval()


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

pre_train=net(x_train)
pre_test=net(x_test)

# true_train=torch.stack((y_train_COD, y_train_color_remove, y_train_energy_consume), dim=1)
# true_test=torch.stack((y_test_COD,y_test_color_remove,y_test_energy_consume),dim=1)


y_COD_all=y_train_COD.cpu().detach().numpy()+y_test_COD.cpu().detach().numpy()
y_COD_all_pre=pre_train[:,0].cpu().detach().numpy()+pre_test[:,0].cpu().detach().numpy()

y_Color_all=y_train_color_remove.cpu().detach().numpy()+y_test_color_remove.cpu().detach().numpy()
y_Color_all_pre=pre_train[:,1].cpu().detach().numpy()+pre_test[:,1].cpu().detach().numpy()

y_Energy_all=y_train_energy_consume.cpu().detach().numpy()+y_test_energy_consume.cpu().detach().numpy()
y_Energy_all_pre=pre_train[:,2].cpu().detach().numpy()+pre_test[:,2].cpu().detach().numpy()

y_Energy_all=y_Energy_all/100
y_Energy_all_pre=y_Energy_all_pre/100





def r2_adjust(y_all,y_predict,n,p):
    if n-p-1==0:
        return 0
    else:
        return 1-((1-r2_score(y_all,y_predict))*(n-1))/(n-p-1)

def result():
    print(f'The R2 for all is {r2_score(y_COD_all, y_COD_all_pre)}')
    print(f'The r2adjust for all is {r2_adjust(y_COD_all, y_COD_all_pre, n=len(data), p=3)}')
    print(f'The mean-squared-error for all is {mean_squared_error(y_COD_all, y_COD_all_pre)}')
    print(f'The mean-squared-error for test is {mean_squared_error(y_test_COD.cpu().detach().numpy(), pre_test[:,0].cpu().detach().numpy())}')
    print(f'residual is {y_COD_all-y_COD_all_pre}')

    print(f'The R2 for all is {r2_score(y_Color_all, y_Color_all_pre)}')
    print(f'The r2adjust for all is {r2_adjust(y_Color_all, y_Color_all_pre, n=len(data), p=3)}')
    print(f'The mean-squared-error for all is {mean_squared_error(y_Color_all, y_Color_all_pre)}')
    print(f'The mean-squared-error for test is {mean_squared_error(y_test_color_remove.cpu().detach().numpy(), pre_test[:,1].cpu().detach().numpy())}')
    print(f'residual is {y_Color_all - y_Color_all_pre}')

    print(f'The R2 for all is {r2_score(y_Energy_all, y_Energy_all_pre)}')
    print(f'The r2adjust for all is {r2_adjust(y_Energy_all, y_Energy_all_pre, n=len(data), p=3)}')
    print(f'The mean-squared-error for all is {mean_squared_error(y_Energy_all, y_Energy_all_pre)}')
    print(f'The mean-squared-error for test is {mean_squared_error(y_test_energy_consume.cpu().detach().numpy()/100, pre_test[:,2].cpu().detach().numpy()/100)}')
    print(f'residual is {y_Energy_all-y_Energy_all_pre}')

def plot3D():
    plt.rc('font', family='Times New Roman', size=22)
    plt.rc('legend', fontsize=30)

    plt.figure(dpi=300,figsize=(26,18))
    ax=plt.axes(projection='3d')

    ax.scatter3D(y_train_COD.cpu().detach().numpy(),1, pre_train[:,0].cpu().detach().numpy(),label='Train_COD_removal',alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(y_test_COD.cpu().detach().numpy(),1, pre_test[:,0].cpu().detach().numpy(), label='Test_COD_removal',linewidths=8,alpha=1)

    ax.scatter3D(y_train_color_remove.cpu().detach().numpy(), 2,pre_train[:,1].cpu().detach().numpy(), label='Train_color_removal',alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(y_test_color_remove.cpu().detach().numpy(), 2,pre_test[:,1].cpu().detach().numpy(), label='Test_color_removal',linewidths=8,alpha=1)

    ax.scatter3D(y_train_energy_consume.cpu().detach().numpy(), 3,pre_train[:,2].cpu().detach().numpy(), label='Train_energy_consume',c='r',alpha=0.7,linewidths=6,marker='*')
    ax.scatter3D(y_test_energy_consume.cpu().detach().numpy(), 3,pre_test[:,2].cpu().detach().numpy(), label='Test_energy_consume',linewidths=8,alpha=1)

    ax.xaxis.set_pane_color((1, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1, 1.0, 1.0, 1.0))

    ax.view_init(12, 240)

    ax.legend(loc=(-85 / 200, 100 / 200), columnspacing=0.4)

    ax.plot3D(range(-40, 200), [1] * 240, range(-40, 200))
    ax.plot3D(range(-40, 200), [2] * 240, range(-40, 200))
    ax.plot3D(range(-40, 200), [3] * 240, range(-40, 200))
    ax.set(xlim=[-40, 200], ylim=[1, 3], zlim=[-40, 200])

    ax.set_yticks([1,2,3],['COD_removal','Color_removal','Energy_consume'])
    plt.yticks(visible=False)

    plt.show(block=True)

def _sanitize_filename(text):
    return str(text).replace(' ', '_').replace('/', '_').replace('\\', '_')


def _enlarge_figure_fonts(fig, label_size=34, tick_size=30, adjust_axis_label=True):
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=tick_size)
        if adjust_axis_label:
            ax.xaxis.label.set_size(label_size)
            ax.yaxis.label.set_size(label_size)
        if ax.title is not None:
            ax.title.set_size(label_size)


def _is_numeric_text(text):
    return re.fullmatch(r'[-+]?\d+(\.\d+)?', text.strip()) is not None


def _refine_waterfall_text_layout(fig, base_value):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        min_y = None
        ax_bbox = ax.get_window_extent(renderer=renderer)
        bottom_cutoff = ax_bbox.y0 + ax_bbox.height * 0.14
        for text_obj in ax.texts:
            text_value = text_obj.get_text().strip()
            x_pos, y_pos = text_obj.get_position()
            if min_y is None or y_pos < min_y:
                min_y = y_pos

            text_bbox = text_obj.get_window_extent(renderer=renderer)
            normalized = re.sub(r'\s+', '', text_value).lower()

            if text_bbox.y1 <= bottom_cutoff and (('e[' in normalized and 'f(' in normalized) or _is_numeric_text(text_value)):
                text_obj.set_visible(False)
                continue

            if text_value.startswith('+') or text_value.startswith('-'):
                try:
                    magnitude = abs(float(text_value))
                except ValueError:
                    magnitude = 10
                if magnitude < 8:
                    text_obj.set_fontsize(16)
                elif magnitude < 15:
                    text_obj.set_fontsize(20)
                else:
                    text_obj.set_fontsize(26)
                text_obj.set_fontweight('bold')
                text_obj.set_clip_on(False)
                text_obj.set_zorder(10)
            if 'e[' in normalized and 'f(' in normalized and ')]' in normalized:
                text_obj.set_visible(False)
                continue

            if _is_numeric_text(text_value):
                try:
                    numeric_value = float(text_value)
                    if abs(numeric_value - float(base_value)) < 0.25 and (min_y is None or y_pos <= min_y + 0.8):
                        text_obj.set_visible(False)
                        continue
                except ValueError:
                    pass

            if 'f(' in normalized and 'x' in normalized:
                text_obj.set_fontsize(24)
                text_obj.set_fontweight('normal')

        ax.set_xlabel('')

    fig.text(
        0.50,
        0.035,
        f'E[f(x)] = {float(base_value):.3f}',
        ha='center',
        va='bottom',
        fontsize=24,
        color='black'
    )


def shap_test(i, target_name, x_test=x_test, x_train=x_train):
    net = neural_mutitask()
    net.load_state_dict(torch.load('mutitask ANN model.pth'))
    torch.set_grad_enabled(True)
    f = lambda x: net(torch.from_numpy(x).type(torch.float32)).cpu().detach().numpy()[:,-i]
    # f = lambda x: net(x).cpu().detach().numpy()[:, 1]
    x_train=x_train.cpu().detach().numpy()
    x_test = x_test.cpu().detach().numpy()
    feature_names = data.columns[:3].tolist()
    e=shap.Explainer(f, x_train, feature_names=feature_names)
    # print(e)
    # # x_test = torch.tensor(s.inverse_transform(x_test.cpu()),dtype=torch.float32)
    shap_values_train=e(x_train)
    shap_values_test=e(x_test)
    output_dir = 'shap_outputs'
    os.makedirs(output_dir, exist_ok=True)
    target_name_safe = _sanitize_filename(target_name)

    plt.rcParams.update({
        'font.size': 30,
        'axes.labelsize': 34,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 28,
        'figure.dpi': 300,
        'savefig.dpi': 600,
    })

    shap.plots.waterfall(shap_values_test[0], show=False)
    fig1 = plt.gcf()
    fig1.set_size_inches(10, 12)
    fig1.set_dpi(600)
    _enlarge_figure_fonts(fig1, label_size=30, tick_size=30, adjust_axis_label=False)
    base_value = float(np.array(shap_values_test[0].base_values).reshape(-1)[0])
    _refine_waterfall_text_layout(fig1, base_value)
    fig1.subplots_adjust(bottom=0.18, top=0.95)
    fig1.savefig(os.path.join(output_dir, f'shap_waterfall_{target_name_safe}.png'), dpi=600, bbox_inches='tight')
    plt.show(block=False)
    plt.close(fig1)

    shap.summary_plot(shap_values_train, x_train, feature_names=feature_names, show=False)
    fig2 = plt.gcf()
    fig2.set_size_inches(10, 12)
    fig2.set_dpi(600)
    _enlarge_figure_fonts(fig2, label_size=34, tick_size=30)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, f'shap_summary_{target_name_safe}.png'), dpi=600, bbox_inches='tight')
    plt.show(block=False)
    plt.close(fig2)

# def deep_shap(x_test=x_test):
#     net=neural_mutitask().cuda()
#     net.load_state_dict(torch.load('mutitask ANN model.pth'))
#     e=shap.DeepExplainer(net,x_train)
#     # x_test = torch.tensor(s.inverse_transform(x_test.cpu()),dtype=torch.float32)
#     shap_values=e.shap_values(torch.tensor(x_test,dtype=torch.float32))
#     feature_names = data.iloc[:, :-3].columns
#     # shap.plots._waterfall.waterfall_legacy(e.expected_value[0], shap_values[0][0])
#     # shap.summary_plot(shap_values, x_test, feature_names)
#     shap_interaction_values = e.shap_interaction_values(x_train)
#     shap.summary_plot(shap_interaction_values, x_train)
#     plt.show(block=True)

target_names = data.columns[3:6].tolist()
for idx, target_name in enumerate(target_names):
    shap_test(i=idx, target_name=target_name)
