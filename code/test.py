import numpy as np
import matplotlib.pyplot as plt

# # 创建一个2x2的网格布局
# fig, axes = plt.subplots(nrows=2, ncols=2)
#
# # 在当前活动的Figure对象上绘制图形
# plt.plot([1, 2, 3, 4], [10, 5, 20, 15], label='Data')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Example Plot')
# plt.legend()
#
# # 获取当前活动的Figure对象
# current_figure = plt.gcf()
#
# # 获取当前活动的Axes对象（例如，第一个子图）
# current_axes = plt.gca()
# current_axes.plot([1,2,3])
# current_axes.set_xlabel('ewq')
# current_axes.axis([1,5,2,3])
#
#
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示图形
# plt.show()


# x=np.linspace(1,10,10)
# y=np.linspace(1,10,10)
#
# X,Y=np.meshgrid(x, y)
# func=lambda x,y:np.sin(x)+np.cos(y)
# Z=func(X,Y)
#
# axs=plt.axes()
# axs.contourf(X,Y,Z,cmap='RdGy')
# plt.show()

# data=np.random.random(50)
#
# plt.hist(data,bins=100,histtype='barstacked',density=True)
# plt.show()


# mean = [0, 0]
# cov = [[1, 1], [1, 2]]
# x,y = np.random.multivariate_normal(mean, cov, 10000).T
# plt.hist2d(x,y,bins=30,cmap='Blues')
# plt.colorbar()
# plt.show()

# for area in [100, 300, 500]:
#     plt.scatter([1,2],[3,4],c='k', alpha=0.3, s=area,
#                 label=str(area) + ' km$^2$')
# plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
#
# plt.title('California Cities: Area and Population')
# plt.show()

# fig=plt.figure()
# ax=fig.add_subplot(2,2,2)
# ax.plot([1,2,3],[4,5,6])
# plt.show()

grid=plt.GridSpec(2,3)
ax1=plt.subplot(grid[:-1,1:])
ax1.plot([1,2,3],[4,5,6])
plt.show()
