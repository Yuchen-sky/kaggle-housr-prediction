import pandas as pd
import  matplotlib.pyplot as plt
import  seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline # 为了在jupyter notebook里作图，需要用到这个命令
df_train=pd.read_csv('./train.csv')
print(df_train.columns)
print(df_train['SalePrice'].describe())
print(df_train['YearBuilt'].describe())

sns.distplot(df_train['SalePrice'])
#plt.show()
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %F" % df_train['SalePrice'].skew())

print("Skewness: %f" % df_train['YearBuilt'].skew())
print("Kurtosis: %F" % df_train['YearBuilt'].skew())

var='GrLivArea'
var2='TotalBsmtSF'
var3='OverallQual'
var4='YearBuilt'
putout='SalePrice'
'''
data=pd.concat([df_train[putout],df_train[var]],axis=1)
data.plot.scatter(x=var,y=putout,ylim=(0,800000))

data=pd.concat([df_train[putout],df_train[var2]],axis=1)
data.plot.scatter(x=var2,y=putout,ylim=(0,800000))

data=pd.concat([df_train[putout],df_train[var3]],axis=1)
f,ax=plt.subplots(figsize=(8,6))
fig=sns.boxplot(x=var3,y=putout,data=data)
fig.axis(ymin=0,ymax=800000)

data=pd.concat([df_train[putout],df_train[var4]],axis=1)
f,ax=plt.subplots(figsize=(16,8))
fig=sns.boxplot(x=var4,y=putout,data=data)
fig.axis(ymin=0,ymax=800000)
plt.xticks(rotation=90);

corrmat=df_train.corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=.9,square=True)

k=10
cols=corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm=np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)

sns.set()
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(df_train[cols],size=2.5)

#plt.show()
'''
total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
print(missing_data.head(20))

df_train=df_train.drop((missing_data[missing_data['Total']>1]).index,1)
df_train=df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
print(df_train.isnull().sum().max())

saleprice_scaled=StandardScaler().fit_transform(df_train[putout][:,np.newaxis])
low_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


df_train.sort_values(by=var,ascending=False)[:2]
df_train=df_train.drop(df_train[df_train['Id']==1299].index)
df_train=df_train.drop(df_train[df_train['Id']==524].index)
'''
data=pd.concat([df_train[putout],df_train[var]],axis=1)
data.plot.scatter(x=var,y=putout,ylim=(0,800000))

data=pd.concat([df_train[putout],df_train[var2]],axis=1)
data.plot.scatter(x=var2,y=putout,ylim=(0,800000))
'''


'''
df_train[putout]=np.log(df_train[putout])
sns.distplot(df_train[putout],fit=norm)
fig=plt.figure()
res=stats.probplot(df_train[putout],plot=plt)

df_train[var]=np.log(df_train[var])
sns.distplot(df_train[var],fit=norm)
fig=plt.figure()
res=stats.probplot(df_train[var],plot=plt)

sns.distplot(df_train[var2],fit=norm)
fig=plt.figure()
res=stats.probplot(df_train[var2],plot=plt)

var6='HasBsmt'
df_train[var6]=pd.Series(len(df_train[var2]),index=df_train.index)
df_train[var6]=0
df_train.loc[df_train[var2]>0,var6]=1
df_train.loc[df_train[var6]==1,var2]=np.log(df_train[var2])
sns.distplot(df_train[df_train[var2]>0][var2],fit=norm)
fig=plt.figure()
res=stats.probplot(df_train[df_train[var2]>0][var2],plot=plt)
'''
plt.scatter(df_train[var],df_train[putout])
df_train=pd.get_dummies(df_train)
plt.show()