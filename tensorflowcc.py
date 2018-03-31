
import pandas as pd
import numpy as np
import seaborn as sns

import  matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.special import boxcox1p


train=pd.read_csv("./train.csv")
test=pd.read_csv("./test.csv")
print(train.head())
print(train.describe())
#print(train.loc[:,'SaleCondition'])
train=train[~((train['GrLivArea']>4000)&(train['SalePrice']<400000))]
all_data=pd.concat([train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']])
print(all_data['GrLivArea'].describe())

all_data.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd','GarageYrBlt'],axis=1,inplace=True)
train['SalePrice']=np.log1p(train['SalePrice'])
numeric_feats=all_data.dtypes[all_data.dtypes!='object'].index
skewed_feats=train[numeric_feats].apply(lambda x:skew(x.dropna()))
skewed_feats=skewed_feats[abs(skewed_feats)>0.7]
skewed=skewed_feats.index
from scipy.special import boxcox1p
all_data[skewed]=boxcox1p(all_data[skewed], 0.15)#需要继续了解
#all_data=all_data[numeric_feats]
all_data=pd.get_dummies(all_data)
all_data=all_data.fillna(all_data.mean())
xtr=all_data[:train.shape[0]]
xte=all_data[train.shape[0]:]
Y_=train['SalePrice']
xte=xte.astype(np.float32)
xtr=xtr.astype(np.float32)
Y_=Y_.astype(np.float32)
xtr=xtr.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#xte=xte.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(all_data.shape)

Xtr=xtr.as_matrix()
'''
corrmat=all_data.corr()
show2=corrmat[corrmat.abs()>0.8]
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(show2,vmax=.9,square=True)
'''

#plt.show()
print(Xtr.shape)
import tensorflow as tf
# 定义placeholder
x = tf.placeholder(tf.float32, [None, 283])#行不确定，只有一列
y = tf.placeholder(tf.float32, [None])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([283, 10]))#1行10列
biases_L1 = tf.Variable(tf.zeros([10]))
L1 = tf.nn.sigmoid(tf.matmul(x, Weights_L1) + biases_L1)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(initial_value=12.0,dtype=tf.float32)
prediction =tf.matmul(L1, Weights_L2)+biases_L2

#损失函数
loss= tf.reduce_mean(tf.square(y - prediction))
loss2=-tf.reduce_sum(y*tf.log(prediction))

# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    for _ in range(5001):
        #print(sess.run(Weights_L2))
        #sess.run(train_step, feed_dict={x: Xtr, y: Y_})
        for start, end in zip(range(0, len(Xtr), 150), range(128, len(Xtr) + 1, 150)):
            sess.run(train_step, feed_dict={x: Xtr[start:end], y: Y_[start:end]})

        print(_)

    # 获得预测值
    prediction_value=sess.run(prediction, feed_dict={x: Xtr})
    print(prediction_value)

    # 画图
    plt.figure()
    plt.scatter(prediction_value, Y_)#样本点
    plt.show()
