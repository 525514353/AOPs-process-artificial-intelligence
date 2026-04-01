from sklearn.linear_model import LinearRegression,Ridge,Lasso,SGDRegressor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

data=pd.read_excel('data_new.xlsx')
data.iloc[:,:-1]=StandardScaler().fit_transform(data.iloc[:,:-1])

x_all=data.iloc[:,:-1]
y_all=data.iloc[:,-1]

x_train=data.iloc[:-4,:-1]
y_train=data.iloc[:-4,-1]
x_test=data.iloc[-4:len(data)+1,:-1]
y_test=data.iloc[-4:len(data)+1,-1]

linear=LinearRegression()
linear.fit(x_train,y_train)

def r2_adjust(y_test,y_predict,n,p):
    if n-p-1==0:
        return 0
    else:
        return 1-((1-r2_score(y_test,y_predict))*(n-1))/(n-p-1)



print(f'R_2 for all dataset:{linear.score(x_all,y_all)}')
print(f'R_2_adjust for all dataset:{r2_adjust(y_all,linear.predict(x_all),n=len(data),p=4)}')

print(f'R_2 for predict:{linear.score(x_test, y_test)}')
# print(f'R_2_adjust for predict:{r2_adjust(y_test,linear.predict(x_test),n=4,p=4)}')

print(f'Mean-squared-error for all dataset:{mean_squared_error(y_all,linear.predict(x_all))}')
print(f'Mean-squared-error for test:{mean_squared_error(y_test,linear.predict(x_test))}')
print(f'{list(zip(data.columns.values,linear.coef_))}')
print(f'bias is {linear.intercept_}')


# print([list(a) for a in list(zip(data.columns.values, linear.coef_))])
plt.rc('legend',fontsize=22)
plt.rc('font',family='Times New Roman',size=22)
fig,ax=plt.subplots(dpi=300,figsize=(10,8))
ax.set_xlabel('True')
ax.set_ylabel('Predict')
ax.scatter(y_train,linear.predict(x_train),label='Train',linewidths=2)
ax.scatter(y_test,linear.predict(x_test),label='Test',linewidths=2)
ax.plot(range(50,100),range(50,100),c='g')
ax.legend()
plt.show()


print(f'The residuals are {(linear.predict(x_all)-y_all).values}')





