import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import os


types = {'Store':np.dtype(int),
         'DayOfWeek': np.dtype(int),
         'Sale' : np.dtype(float),
         'Promo' : np.dtype(int),
         'SchoolHoliday': np.dtype(int),
         'StateHoliday': np.dtype(int),
         'StoreType': np.dtype(int),
         'Assortment': np.dtype(int),
         'CompetitionDistance' : np.dtype(float),
         'IsPromo2Month': np.dtype(int),
         'Competition': np.dtype(int),
         'StoreRenovated': np.dtype(int),
         'DaysAfterRenovation' : np.dtype(int),
         'MonthSale': np.dtype(float),
         'Month':np.dtype(int),
         'Day':np.dtype(int)
}

train = pd.read_csv("~/kaggleout/rossman/train_normalized.csv", 
                    parse_dates=['Date'], dtype=types)
test = pd.read_csv("~/kaggleout/rossman/test_normalized.csv", 
                   parse_dates=['Date'], dtype=types)
                   
# Some new feature engineering:
def extendFeatures(df):
    df['Friday'] = 0
    df['EarlyInMonth'] = 0
    df['LateInMonth'] = 0
    df['December'] = 0
    df['ChristmasTime'] = 0
    df.loc[df.DayOfWeek == 5,'Friday'] = 1
    df.loc[df.Day < 5, 'EarlyInMonth'] = 1
    df.loc[df.Day > 25, 'LateInMonth'] = 1
    df.loc[df.Month == 12,'December'] = 1
    isXmas = (df.Month == 12) & (df.Day > 23)
    df.loc[isXmas, 'ChristmasTime'] = 1
    return df
    
train = extendFeatures(train)
test = extendFeatures(test)


# print train.columns.values


# Keeping out the last two month for cross validation
cvtest = train[train.Date > '2015-06-30']
train = train[train.Date < '2015-07-01']


features = ['Store', 'DayOfWeek', 'Promo',  'StateHoliday', 
            'SchoolHoliday', 'StoreType', 'Assortment',  
            'CompetitionDistance', 'Month', 'Year', 'Day',  
            'IsPromo2Month', 'Competition','StoreRenovated', 
            'DaysAfterRenovation','Friday','EarlyInMonth','LateInMonth',
            'December','ChristmasTime']

#XGboost params: (set for quick run, not high accuracy)
params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.2,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "thread": 1,
          "seed": 2015
          }
# For the final run should try 1000+ iteration at eta < 0.02
num_boost_round = 200


X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = np.array(X_train.Sales)
y_valid = np.array(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

def rmse(y, yhat):    
    return np.sqrt(np.mean((yhat - y) ** 2))

def rmse_xg(yhat, y):
    y = np.array(y.get_label())
    yhat = np.array(yhat)
    return "rmse", rmse(y,yhat)
    
def rmspe(y,yhat):
    return np.sqrt(np.mean( ((y-yhat) / y)**2 ) )

# Training the tree:
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, 
                early_stopping_rounds=100, feval=rmse_xg, verbose_eval=True)

print("Make predictions on the cvtest set")
dtest = xgb.DMatrix(cvtest[features])
test_preds = gbm.predict(dtest)
test_sales = (test_preds*np.array(cvtest.SaleSD) + np.array(cvtest.MonthTrend) 
              + np.array(cvtest.YearTrend))

error = rmspe(cvtest.Sales2, test_sales)
print('RMSPE: {:.6f}'.format(error))

os.chdir("/Users/arman")

plt.plot(cvtest.Sales2,test_sales,'bo',cvtest.Sales2,cvtest.Sales2,'k')
plt.title('Cross validation for 1115 stores \n for the last two month of train set')
plt.xlabel('Actual Sale')
plt.ylabel('Predicted Sale')
plt.savefig("cv.png",dpi=200)

# Predicting for the test set:
dtest = xgb.DMatrix(test[features])
test_preds = gbm.predict(dtest)
test_sales = (test_preds*np.array(test.SaleSD) + np.array(test.MonthTrend) 
              + np.array(test.YearTrend))

submissionDF = pd.DataFrame({'Id': test['Id'], 'Sales': test_sales})
submissionDF.to_csv("submission.csv",index=False)
