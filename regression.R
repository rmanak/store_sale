library(readr)
library(splines)

train <- read_csv("~/kaggleout/rossman/trainCleaned.csv")
test  <- read_csv("~/kaggleout/rossman/testCleaned.csv")

# Dropping the index column:
train <- train[,-c(1)]
test <- test[,-c(1)]


sapply(train,function(x)any(is.na(x)))
sapply(test,function(x)any(is.na(x)))


# There are few NAs in competition distance... imputing to the average value 
avg_comp_distance <- mean(train$CompetitionDistance,na.rm=TRUE)
na_idx <- is.na(train$CompetitionDistance)
train$CompetitionDistance[na_idx] <- avg_comp_distance
na_idx <- is.na(test$CompetitionDistance)
test$CompetitionDistance[na_idx] <- avg_comp_distance 

# Checking NA values, should return all false
table(is.na(train))
table(is.na(test))

names(train)
str(train)
summary(train)

names(test)
str(test)
summary(test)

# Dropping the days that stores are close
train <- train[ which(train$Open==1),]
train <- train[ which(train$Sales!=0),]

# Dropping Year-Month date:
train$YearMonth <- NULL
test$YearMonth <- NULL

feature.names <- names(train)
cat("Feature Names:\n")
feature.names


# Converting string and characters to categorical variables
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

# New feature: Year trend: 
# Average trend in the sale over the 3 years using a spline fit
train$YearTrend <- 0
test$YearTrend <- 0

# Measuring time in the unit of month:
train$TimeInMonth <- (train$Year - 2013)*12 + train$Month
test$TimeInMonth <- (test$Year - 2013)*12 + test$Month

#----------------------------------------------------------------------
# 6 Category of stores' sale trend
# Store never renovated => 2 category: sale with promotion and without
# Store *is* renovated => 4 category:
#                             sale w / wihtout promo before renovation
#                             sale w / without promo after renovation
-----------------------------------------------------------------------

growthfitnopromo <- NULL
growthfitwithpromo <- NULL
growthfit0nopromo <- NULL
growthfit0withpromo <- NULL
growthfit1nopromo <- NULL
growthfit1withpromo <- NULL

setwd("~/kaggleout/rossman")
for (i in unique(train$Store)) {
  
cat("fitting store i=",i,"\n")
  
# Removing December from linear reg. since it is an outlier in most of the stores, 
# The feature will be picked up by a decision tree model, aka. XGBoost 
Store <- train[train$Store==i & train$Month !=12,]
  
if (Store$StoreRenovated[1] == 0) {
    
  #=======Without Promo =============
  x <- as.integer(Store[Store$Promo==0,]$Date)
  y <- Store[Store$Promo==0,]$Sales

  growthfitnopromo[[i]] <- lm(y~ns(x,df=2))

  plot(Store$Date,Store$Sales,pch=19,col=Store$Promo+5,
       xlab="date",ylab="sale",main = paste0("Store = ",as.character(i)))
  
  xx <- as.integer(train[train$Store==i & train$Promo==0,]$Date)
  yy <- predict(growthfitnopromo[[i]],data.frame(x=xx))
  points(xx,yy,col="orange")
  train[train$Store==i & train$Promo==0,]$YearTrend <- yy
  
  if (i %in% unique(test$Store)) {
    xx <- as.integer(test[test$Store==i & test$Promo==0,]$Date)
    yy <- predict(growthfitnopromo[[i]],data.frame(x=xx))
    test[test$Store==i & test$Promo==0,]$YearTrend <- yy
  }
  
  #======= With Promo =============
  
  x <- as.integer(Store[Store$Promo==1,]$Date)
  y <- Store[Store$Promo==1,]$Sales
  
  growthfitwithpromo[[i]] <- lm(y~ns(x,df=2))
  
  
  xx <- as.integer(train[train$Store==i & train$Promo==1,]$Date)
  yy <- predict(growthfitwithpromo[[i]],data.frame(x=xx))
  points(xx,yy,col="blue")
  train[train$Store==i & train$Promo==1,]$YearTrend <- yy
  
  if (i %in% unique(test$Store)) {
    xx <- as.integer(test[test$Store==i & test$Promo==1,]$Date)
    yy <- predict(growthfitwithpromo[[i]],data.frame(x=xx))
    test[test$Store==i & test$Promo==1,]$YearTrend <- yy
    points(test[test$Store==i,]$Date,test[test$Store==i,]$YearTrend,col=24+test$Promo)
  }
  
} # End of store *not* renovated 
else {
    
    # Fitting before renovation:
  
    #=======Without Promo =============
    x <- as.integer(Store[Store$DaysAfterRenovation==0 &Store$Promo==0,]$Date)
    y <- Store[Store$DaysAfterRenovation==0&Store$Promo==0,]$Sales
    
    growthfit0nopromo[[i]] <- lm(y~ns(x,df=2))
    
    plot(as.integer(Store$Date),Store$Sales,pch=19,col=Store$Promo+5,
         xlab="date",ylab="sale",main = paste0("Store = ",as.character(i)))
    
    xx <- as.integer(train[train$Store==i & train$DaysAfterRenovation==0 & train$Promo==0,]$Date)
    yy <- predict(growthfit0nopromo[[i]],data.frame(x=xx))
    points(xx,yy,col="orange")
    
    train[train$Store==i & train$DaysAfterRenovation==0 &train$Promo==0,]$YearTrend <- yy
    
    
    #=======With Promo =============
    
    x <- as.integer(Store[Store$DaysAfterRenovation==0 &Store$Promo==1,]$Date)
    y <- Store[Store$DaysAfterRenovation==0&Store$Promo==1,]$Sales
    
    growthfit0withpromo[[i]] <- lm(y~ns(x,df=2))
    
    xx <- as.integer(train[train$Store==i & train$DaysAfterRenovation==0 & train$Promo==1,]$Date)
    yy <- predict(growthfit0withpromo[[i]],data.frame(x=xx))
    points(xx,yy,col="blue")
    
    train[train$Store==i & train$DaysAfterRenovation==0 &train$Promo==1,]$YearTrend <- yy
    
    
    # Fitting to after renovation: 
    
    #=======Without Promo =============
    x <- as.integer(Store[Store$DaysAfterRenovation==1 & Store$Promo==0,]$Date)
    y <- Store[Store$DaysAfterRenovation==1 & Store$Promo==0,]$Sales
    
    growthfit1nopromo[[i]] <- lm(y~ns(x,df=1))
    

    xx <- as.integer(train[train$Store==i & train$DaysAfterRenovation==1 & train$Promo==0,]$Date)
    yy <- predict(growthfit1nopromo[[i]],data.frame(x=xx))
    points(xx,yy,col="yellow")
    train[train$Store==i & train$DaysAfterRenovation==1 &train$Promo==0,]$YearTrend <- yy
    
    
    if (i %in% unique(test$Store)) {
      
      if (nrow(test[test$Store==i & test$DaysAfterRenovation==0,]) != 0) {
         # If this happens, there is a bug in data cleaning process
         stop("Something terrible happened!")
      }
    
      xx <- as.integer(test[test$Store==i & test$Promo==0,]$Date)
      yy <- predict(growthfit1nopromo[[i]],data.frame(x=xx))
      test[test$Store==i & test$Promo==0,]$YearTrend <- yy
    }
    #=======With Promo =============
    
    
    x <- as.integer(Store[Store$DaysAfterRenovation==1 & Store$Promo==1,]$Date)
    y <- Store[Store$DaysAfterRenovation==1 & Store$Promo==1,]$Sales
    
    growthfit1withpromo[[i]] <- lm(y~ns(x,df=1))
    
    
    xx <- as.integer(train[train$Store==i & train$DaysAfterRenovation==1 & train$Promo==1,]$Date)
    yy <- predict(growthfit1withpromo[[i]],data.frame(x=xx))
    points(xx,yy,col="green")
    train[train$Store==i & train$DaysAfterRenovation==1 &train$Promo==1,]$YearTrend <- yy
    
    if (i %in% unique(test$Store)) {
      
       xx <- as.integer(test[test$Store==i &  test$Promo==1,]$Date)
       yy <- predict(growthfit1withpromo[[i]],data.frame(x=xx))
    
       test[test$Store==i & test$Promo==1,]$YearTrend <- yy
       
       points(test[test$Store==i,]$Date,test[test$Store==i,]$YearTrend,col=24+test$Promo)
    }
  } #End of renovation = 1 
fname<-paste0("Store_",as.character(i),"_SaleTrend.png")
dev.copy(png,file=fname); dev.off()
}


listoffits <- c("growthfitnopromo","growthfitwithpromo","growthfit0nopromo", 
                "growthfit0withpromo","growthfit1nopromo","growthfit1withpromo") 

save(list=listoffits,file="~/all_fits_sep_pro_sep_renovation.RData")


# Subtracting the yearly trend of growth/decay of sale:
train$Sales2 <- train$Sales 
train$Sales <- train$Sales - train$YearTrend

#save(list=c("train"),file="~/train_set_with_Year_Trend.RData")
#save(list=c("test"),file="~/test_set_with_Year_Trend.RData")


# Exploring the monthly trend:
monthfit <- NULL
train$MonthTrend <- 0
test$MonthTrend <- 0
for (i in unique(train$Store)) {
  cat("doing Store =",i,"\n")
  Store <- train[train$Store==i ,]
  y <- by(Store$Sales, Store$Day, mean)
  x <- as.integer(names(y))
  monthfit[[i]] <- lm(y~ns(x,df=5))
  plot(x,y,pch=19,xlab="DayofMonth",ylab="sale variation",main = paste0("Store = ",as.character(i)))
  
  xx <- train[train$Store==i,]$Day
  yy <- predict(monthfit[[i]], data.frame(x=xx))
  train[train$Store==i,]$MonthTrend <- yy

  if (i %in% unique(test$Store)) {
    xx <- test[test$Store==i,]$Day
    yy <- predict(monthfit[[i]],data.frame(x=xx))
    test[test$Store==i,]$MonthTrend <- yy
    points(test[test$Store==i,]$Day,test[test$Store==i,]$MonthTrend,col="blue",pch=19)
  }
  fname<-paste0("Store_",as.character(i),"_MonthTrend.png")
  dev.copy(png,file=fname); dev.off()
}

save(list=c("monthfit"),file="~/all_monthfits.RData")

train$Sales3 <- train$Sales
train$Sales <- train$Sales - train$MonthTrend


#save(list=c("train"),file="~/train_set_with_Month_Trend.RData")
#save(list=c("test"),file="~/test_set_with_Month_Trend.RData")


train$SaleSD <- 1
test$SaleSD <- 1
for (i in unique(train$Store)) {
  sd_sale <- sd(train[train$Store==i ,]$Sales)
  cat("doing Store i=",i,"sd=",sd_sale,"\n")
  train[train$Store==i,]$SaleSD <- sd_sale
  if (i %in% unique(test$Store)) {
    test[test$Store==i,]$SaleSD <- sd_sale 
  }
}

train$Sales4 <- train$Sales 
# Normalizing sale of stores by their standard deviation
train$Sales <- train$Sales / train$SaleSD

write_csv(train, "~/kaggleout/rossman/train_normalized.csv")
write_csv(test,"~/kaggleout/rossman/test_normalized.csv")
