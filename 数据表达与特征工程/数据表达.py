import pandas as pd
fruits = pd.DataFrame({'数值特征':[5,6,7,8,9], '类型特征':['西瓜','香蕉','橘子','苹果','葡萄']})
#上面我们创建了一个二维列表，接下来我们要把字符串转为数值
fruits_dum = pd.get_dummies(fruits)
print(fruits_dum)#False 代表0  True 代表1
#令程序将数值看做字符串
fruits['数值特征'] = fruits['数值特征'].astype(str)
fruits = pd.get_dummies(fruits, columns=['数值特征'])
print(fruits)


#对数据进行装箱化处理——离散化处理
import numpy as np
import matplotlib .pyplot as plt
#生成随机数列
rnd = np.random.RandomState(38)
x = rnd.uniform(-5,5,size=50)
#向数据里添加噪音
y_no_noise = (np.cos(6*x)+x)
X = x.reshape(-1,1)
y = (y_no_noise + rnd.normal(size=len(X)))/2
plt.plot(X,y,'o',c='r')
plt.show()

#接下来我们用KNN和MLP算法来对其进行回归分析
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
#生成一个等差数列
line = np.linspace(-5,5,1000,endpoint=False).reshape(-1,1)
#分别用两种算法拟合数据
mlpr = MLPRegressor().fit(X, y)
knr = KNeighborsRegressor().fit(X, y)
plt.plot(line, mlpr.predict(line), label='MLP')
plt.plot(line, knr.predict(line),label='KNN')
plt.plot(X,y,'o',c='r')
plt.legend(loc='best')
plt.show()

#为了让回归更加准确,我们来使用装箱处理
bins = np.linspace(-5,5,11)
#进行装箱操作
target_bin = np.digitize(X, bins=bins)
#打印装箱数据范围
print('装箱数据范围：\n{}'.format(bins))
#打印前十个数据点的特征值
print('\n前十个数据点的特征值：\n{}'.format(X[:10]))
#找到它们所在的箱子
print('\n前十个数据点所在的箱子：\n{}'.format(target_bin[:10]))
#我们来换一种方式来表达这个数据
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse= False)
onehot.fit(target_bin)
x_in_bin = onehot.transform(target_bin)
print('装箱后的数据形态：{}'.format(x_in_bin.shape))
print("\n装箱后的前十个数据点\n{}".format(x_in_bin[:10]))

#数据处理完了 我们再次拟合回归吧！
#使用独热编码进行数据表达
new_line = onehot.transform(np.digitize(line,bins=bins))
new_mlpr = MLPRegressor().fit(x_in_bin,y)
new_knr = KNeighborsRegressor().fit(x_in_bin,y)
plt.plot(line, new_mlpr.predict(new_line),label = 'New MLP')
plt.plot(line, new_knr.predict(new_line), label='New KNN')
plt.plot(X,y,'o',c='r')
plt.legend(loc='best')
plt.show()
