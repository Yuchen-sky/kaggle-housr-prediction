import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import  matplotlib.pyplot as plt
from scipy.stats import  skew
from  scipy.stats.stats import pearsonr

train=pd.read_csv("./train.csv")
test=pd.read_csv("./test.csv")
print(train.head())
all_data=pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
sale="SalePrice"
'''

matplotlib.rcParams['figure.figsize']=(12.0,6.0)
prices=pd.DataFrame({"price":train[sale],"log(peice+1)":np.log(train[sale])})
prices.hist()
'''

train[sale]=np.log(train[sale])
numeric_feats=all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats=train[numeric_feats].apply(lambda  x: skew(x.dropna()))
skewed_feats=skewed_feats[skewed_feats>0.75]
skewed_feats=skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data=pd.get_dummies(all_data)
all_data=all_data.fillna(all_data.mean())
X_train=all_data[:train.shape[0]]
X_test=all_data[train.shape[0]:]
y=train.SalePrice



from sklearn.linear_model import Ridge, RidgeCV,ElasticNet,LassoLarsCV,LassoCV
from sklearn.model_selection import  cross_val_score

def rmse_cv(model):
    rmse=np.sqrt(-cross_val_score(model,X_train,y,scoring="neg_mean_squared_error",cv=5))
    return(rmse)

model_ridge=Ridge()
'''
alphas=[0.05,0.1,0.3,1,3,5,10,15,30,50,75]
cv_ridge=[rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge=pd.Series(cv_ridge,index=alphas)
cv_ridge.plot(title="Validation-Just do it")
plt.xlabel("alpha")
plt.ylabel("rmse")
'''
#print( cv_ridge.min())

model_lasso=LassoCV(alphas=[1,0.1,0.001,0.0005]).fit(X_train,y)

print(rmse_cv(model_lasso).mean())

coef=pd.Series(model_lasso.coef_,index=X_train.columns)
print("Lasso picked"+ str(sum(coef!=0))+"variables and eliminated the other"+ str(sum(coef==0))+"variables")

imp_coef=pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize']=(8.0,10.0)
imp_coef.plot(kind="barh")

matplotlib.rcParams['figure.figsize']=(6.0,6.0)
preds=pd.DataFrame({"preds":model_lasso.predict(X_train),"true":y})
preds["residuals"]=preds["true"]-preds["preds"]
preds.plot(x="preds",y="residuals",kind="scatter")



import xgboost as xgb

dtrain =xgb.DMatrix(X_train,label=y)
dtest=xgb.DMatrix(X_test)

params={"max_depth":2,"eta":0.1}
model=xgb.cv(params,dtrain,num_boost_round=901500, early_stopping_rounds=201300)
model.loc[30:,["test-rmse-mean","train-rmse-mean"]].plot()
model_xgb=xgb.XGBRegressor(n_estimators=360,max_depth=10,learning_rate=0.01)
model_xgb.fit(X_train,y)


xgb_preds=np.exp(model_xgb.predict(X_test))
lasso_preds=np.exp(model_lasso.predict(X_test))

predictions=pd.DataFrame({"xgb":xgb_preds,"lasso":lasso_preds})
predictions.plot(x="xgb",y="lasso",kind="scatter")
preds=0.7*lasso_preds+0.3*xgb_preds
plt.show()