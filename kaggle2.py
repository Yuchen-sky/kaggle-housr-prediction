
import pandas as pd
import numpy as np
import seaborn as sns

import  matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.special import boxcox1p


train=pd.read_csv("./train.csv")
test=pd.read_csv("./test.csv")

#data descirbe
print(train.head())
print(train.describe())
#print(train.loc[:,'SaleCondition'])

#SalePrice handling
train=train[~((train['GrLivArea']>4000)&(train['SalePrice']<400000))]

#data linking
all_data=pd.concat([train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']])
print(all_data['GrLivArea'].describe())

#relative variables change
all_data.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd','GarageYrBlt'],axis=1,inplace=True)
train['SalePrice']=np.log1p(train['SalePrice'])

all_data.BedroomAbvGr = all_data.BedroomAbvGr.astype(str)
all_data.MSSubClass = all_data.MSSubClass.astype(str)
all_data.MoSold = all_data.MoSold.astype(str)



'''
                             "BsmtCond": {"NA": np.nan, "No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "BsmtExposure": {"NA": np.nan, "No": 1, "Mn": 1, "Av": 2, "Gd": 3},
                             "BsmtQual": {"NA": np.nan, "No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "ExterCond": {"NA": np.nan, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "ExterQual": {"NA": np.nan, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "FireplaceQu": {"NA": np.nan, "No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "GarageCond": {"NA": np.nan, "No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "GarageQual": {"NA": np.nan, "No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "HeatingQC": {"NA": np.nan, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
                             "KitchenQual": {"NA": np.nan, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
'''




#skew proceeding
numeric_feats=all_data.dtypes[all_data.dtypes!='object'].index
skewed_feats=train[numeric_feats].apply(lambda x:skew(x.dropna()))
skewed_feats=skewed_feats[abs(skewed_feats)>0.7]
skewed=skewed_feats.index
from scipy.special import boxcox1p
all_data[skewed]=boxcox1p(all_data[skewed], 0.15)#需要继续了解
#all_data=all_data[numeric_feats]

Unf=['2.5Unf','1.5Unf']
all_data["HouseStypeFinish"]=all_data["HouseStyle"].map(lambda x:0 if x in Unf else 1)

all_data = all_data.replace({"Alley" : {"NA":np.nan,"Grvl" : 1.0, "Pave" : 2.0},
                             "Street": {"NA": np.nan, "Grvl": 1.0, "Pave": 2.0},

})


#ont_hot
all_data=pd.get_dummies(all_data)
all_data=all_data.fillna(all_data.mean())


from sklearn.preprocessing import RobustScaler
N = RobustScaler()
all_data = N.fit_transform(all_data)
print(all_data.shape)
xtr=all_data[:train.shape[0]]
xte=all_data[train.shape[0]:]
Y_=train['SalePrice']




'''
#format unify
xte=xte.astype(np.float32)
xtr=xtr.astype(np.float32)
Y_=Y_.astype(np.float32)
#normalization
xtr=xtr.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#xte=xte.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

'''
'''
#heatmap information
Xtr=xtr.as_matrix()
corrmat=all_data.corr()
show2=corrmat[corrmat.abs()>0.8]
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(show2,vmax=.9,square=True)
plt.show()
print(Xtr.shape)
'''


from  sklearn.model_selection import cross_val_score

def rmse_cv(model,xtr,Y_):
    rmse = np.sqrt(-cross_val_score(model,
                                    xtr,
                                    Y_,
                                    scoring="neg_mean_squared_error",
                                    cv=10))
    return rmse


from xgboost import  XGBRegressor

xgb_model=XGBRegressor(n_estimators=2000,max_depth=3,learning_rate=0.03,subsample=0.9,colsample_bytree=0.6)
xgb_rmse=rmse_cv(xgb_model,xtr,Y_)
print("{:.5f}:+/-{:.5f}".format(xgb_rmse.mean(),xgb_rmse.std()))


from sklearn.linear_model import Lasso,Ridge

alphas=[5,8,9,10.5,11,11.5,12,13,15,20]
ridge=[rmse_cv(Ridge(alpha=alpha),xtr,Y_).mean() for alpha in alphas]
print(ridge)
ridge=pd.Series(ridge,index=alphas)
ridge.plot()


alphas=[0.00033,0.00034,0.00035,0.00036]
lasso=[rmse_cv(Lasso(alpha=alpha),xtr,Y_).mean() for alpha in alphas]
print(lasso)
ridge=pd.Series(lasso,index=alphas)
ridge.plot()
plt.show()


test_xgb_model=XGBRegressor(n_estimators=6000,max_depth=3,learning_rate=0.02,subsample=0.9,colsample_bytree=0.6)
test_xgb_model.fit(xtr,Y_)

test_ridge=Ridge(alpha=11.5)
test_ridge.fit(xtr,Y_)

test_lasso=Lasso(alpha=0.00033)
test_lasso.fit(xtr,Y_)

a=test_xgb_model.predict(xte)
b=test_ridge.predict(xte)
c=test_lasso.predict(xte)

output=0.7*b+0.1*a+0.2*c

preds = np.expm1(output)
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("ridge.csv", index = False)
