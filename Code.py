import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


train_data = pd.read_csv("Train_ML_Models.csv")
test_data = pd.read_csv("Test_ML_Models.csv")
submission = pd.read_csv("sample_submission.csv")

le = preprocessing.LabelEncoder()

def pre_process(data):
    data["week"] = pd.to_datetime(data["week"])
    data['year'] = data.week.dt.year
    data['month'] = data.week.dt.month
    #data['day_name'] = data.week.dt.day_name()
    data['day'] = data.week.dt.day
    data['quarter'] = data.week.dt.quarter
    data['week_of_year'] = data.week.dt.week
    data['discount'] = (data['base_price'] -data['total_price'])/data['base_price']
    data['discount_flag'] = np.where(data['discount'] > 0, 1,0)
    data['store_id'] = le.fit_transform(data["store_id"])
    data['sku_id'] = le.fit_transform(data["sku_id"])
    #data['day_name'] = le.fit_transform(data["day_name"])
    data1 = data[['store_id', 'sku_id', 'is_featured_sku', 'is_display_sku']]
    cat = list(data1.columns.values)
    enc = OneHotEncoder(handle_unknown='ignore')

    enc_df = pd.DataFrame(enc.fit_transform(data[cat]).toarray())
    data = data.join(enc_df)

    data = data.drop(cat, axis = 1)
    return data

train = pre_process(train_data)
test = pre_process(test_data)


Y = train["units_sold"]
X = train.drop(['record_ID', 'week', "units_sold", 'total_price'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=123)

params = {
         'min_child_weight': [1, 3, 5],
         'gamma': [7 , 10],
         'subsample': [0.5, 0.6, 0.8],
         'colsample_bytree': [0.6, 0.7, 0.8],
         'max_depth': [5, 10]}
        
xg_reg = xgb.XGBRegressor(objective ='reg:linear', learning_rate = 0.02, n_estimators = 2000)
 
 
gsearch = GridSearchCV(xg_reg,  param_grid = params, n_jobs=-1, cv=5, verbose=3 )


gsearch.fit(X_train, y_train)
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
best_grid = gsearch.best_estimator_
preds = best_grid.predict(X_test)


xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.7, learning_rate = 0.02,
                max_depth = 10, gamma = 7 , n_estimators = 2000, subsample = 0.5, min_child_weight = 1)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

submission["units_sold"] = xg_reg.predict(test.drop(['record_ID', 'week', 'total_price'], axis = 1))
submission.to_csv("submission.csv", index = False)

