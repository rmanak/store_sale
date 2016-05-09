
import pandas as pd
import numpy as np

types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'CompetitionDistance' : np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'Promo2SinceYear': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
         
train = pd.read_csv('~/kaggledata/rossman/train.csv',
                    parse_dates=['Date'], dtype=types)

test = pd.read_csv('~/kaggledata/rossman/test.csv',
                   parse_dates=['Date'],dtype=types)
                   
store = pd.read_csv('~/kaggledata/rossman/store.csv')

def calcDates(df):
    df['Month'] = df.Date.dt.month
    df['Year'] = df.Date.dt.year
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.weekofyear
    # Year-Month 2015-08 
    # will be used for monthly sale calculation:
    df['YearMonth'] = df['Date'].apply(lambda x:(str(x)[:7]))
    return df


train = pd.merge(train,store,on='Store')
test = pd.merge(test,store,on='Store')

train = calcDates(train)
test = calcDates(test)

#train.head(10)
#test.head(10)

def cleanPromoCompetition(df,drop=False):
    # ========== Fixing promo2 ============
    df.PromoInterval.fillna(0,inplace=True)
    monthAsString = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                     7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    # Using string format of month names to extract info from promo interval column                 
    df['SMonth'] = df.Month.map(monthAsString)
    # Fixing NaN values in promo interval when there is no promotion
    df.loc[df.PromoInterval==0,'PromoInterval'] = ''
    
    # New feature: 
    #     IsPromo2Month: 
    #     0 if month is not among PromoInterval
    #     1 if it is
    
    
    df['IsPromo2Month'] = 0
    for interval in df.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                condmatch = (df.SMonth == month) & (df.PromoInterval == interval)
                # If promo started this year, Week of Year must be > Promo2SinceWeek
                cond1 = (condmatch & (df.Year == df.Promo2SinceYear)
                         & (df.WeekOfYear >= df.Promo2SinceWeek) )
                # Or If promo started previous year, Week of Year doesn't matter
                cond2 = condmatch & (df.Year > df.Promo2SinceYear)
                fullcond = cond1 | cond2
                df.loc[fullcond, 'IsPromo2Month'] = 1
                
     # ======= Fixing Competition =============
    df.CompetitionOpenSinceYear.fillna(0,inplace=True)
    df.CompetitionOpenSinceMonth.fillna(0,inplace=True)
    
    # New feature: 
    #    Competition:
    #    1 if there exist a compettion at date = today
    #    0 otherwise
    
    df['Competition'] = 0 
    cond1 = df.Year > df.CompetitionOpenSinceYear
    cond2 = ((df.Year == df.CompetitionOpenSinceYear) 
             & (df.Month >= df.CompetitionOpenSinceMonth))
    fullcond = cond1 | cond2
    df.loc[fullcond, 'Competition'] = 1 
    
    if (drop):
        df = df.drop(['SMonth','PromoInterval','Promo2SinceYear','Promo2SinceWeek'],1)
        df = df.drop(['CompetitionOpenSinceMonth','CompetitionOpenSinceYear'],1)
    
    return df

train = cleanPromoCompetition(train,drop=True)
test = cleanPromoCompetition(test,drop=True)   

trainOpen = train[train.Open == 1][['Store','YearMonth','Sales']]
monthlySale  = trainOpen.groupby(['Store','YearMonth'],as_index=False).mean() 


#====== Finding renovated stores ========

renovatedStores = []
for store in train.Store.unique():
    # Renovated stores are close before 2015 for more than 2 month
    if len(monthlySale[monthlySale.Store==store]) < 29:
        renovatedStores.append(store)
    

#print(renovatedStores)

def createRenovation(df,renovatedStores):
    
    # New features:
    # StoreRenovated: 1 if it is, 0 otherwise
    # DaysAfterRenovation: 0 if date is before renovation, 1 if it is after
    df['StoreRenovated'] = 0
    df['DaysAfterRenovation'] = 0
    for store in renovatedStores:
        df.loc[df.Store == store,'StoreRenovated'] = 1
        # Renovated stores are back to open state in 2015
        df.loc[(df.Store == store) & (df.Year == 2015), 'DaysAfterRenovation'] = 1

    return df


train = createRenovation(train,renovatedStores)
test  = createRenovation(test,renovatedStores)



monthlySale['MonthSale'] = monthlySale.Sales
monthlySale = monthlySale.drop(['Sales'],1)

# New feature: MonthSale:
# Average of monthly sale for each store
# Adding monthly sale to train set:
train = pd.merge(train,monthlySale,on=['Store','YearMonth'])


# Small NaN Fix on test, only 1 case which is in fact open
test.Open.fillna(1,inplace=True)


#train = train.sort_values(by = 'Date')
train.to_csv('~/kaggleout/rossman/trainCleaned.csv')
test.to_csv('~/kaggleout/rossman/testCleaned.csv')

