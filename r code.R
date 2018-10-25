##################################################################################################
###              SETTING UP
##################################################################################################

#to clear all list in the environment
rm(list=ls())
#to set the working directory
setwd("C:/Users/tamat/Desktop/JOB/Projects/Doing/Bike rental")
#loading the csv file
data=read.csv("day.csv")

#loading the required libraries
x = c("DMwR","corrplot","ggplot2","caret","lattice","rpart","ranger","xgboost")
lapply(x, require, character.only = TRUE)
rm(x)

##################################################################################################
###               UNDERSTANDING THE DATA
##################################################################################################

###Basic structure and dimensions of the data 
dim(data)
names(data)
str(data)

##################################################################################################
###                UNIVARIATE ANALYSIS
##################################################################################################

###Target variable/Outcome variable = cnt(count of the total bikes rented)
hist(data$cnt)
#the distribution is uniform. 

##Temperature
hist(data$temp)
#distribution is normalised

##feel temperature
hist(data$atemp)
#distribution is normalised

##humidity
hist(data$hum)
#distribution is normalised

##windspeed
hist(data$windspeed)
#distribution is normalised

##weather situation
hist(data$weathersit)
prop.table(table(data$weathersit))
#we can say that the weather situation is mostly 1 that is moslty clear
#and there are zero instances of situation 4

##################################################################################################
###                     MULTIVARIATE ANALYSIS
##################################################################################################

##Count of bikes rented with respect to different seasons
#season needs to be converted into factor 
data$season=as.factor(data$season)
boxplot(data$cnt~data$season,xlab="season",ylab="count",mail="distribution of bike rental with season")
#the above distribution shows that the bike rental is highest during seasons 3(fall) and 2(summer)
#followed by season 4(winter)
#during season 1(spring), the bike rental is at its lowest


##To test the hypothesis that the number of bike rental will increase with year
#variable year needs to be converted into factor
data$yr=as.factor(data$yr)
boxplot(data$cnt~data$yr,xlabel="year",ylabel="count")
#there is increase in bike rental with year
#this supports our hypothesis


##To check the distribution of bike rental on holidays and workingdays
#variable workingday needs to be converted into factor
data$workingday=as.factor(data$workingday)
boxplot(data$cnt~data$workingday,xlabel="Working day/Holiday",ylabel="Count")
#the median for the number of bike rented is somewhat similar on both working day and holiday
boxplot(data$registered~data$workingday,xlabel="Working day/Holiday",ylabel="Registered Users")
#We can infer from the above box plot that registered users are using the bike rental more on holidays as compared to working days
boxplot(data$casual~data$workingday,xlabel="Working day/Holiday",ylabel="Casual Users")
#We can infer from the above box plot that casual users are using the bike rental more on working days compared to holidays


##To check the distribution of bike rental according to the weather conditions
#converting variable weather sit into factor
data$weathersit=as.factor(data$weathersit)
boxplot(data$cnt~data$weathersit,xlabel="Weather situation",ylabel="Count")
#It can be inferred that more bikes are rented when the weather situation is 1 followed by situation 2
#Least number of bikes are rented during weather situation 3 and none during situation 4
#Situation 3 has light rain and situation 4 has heavy rain
#therefore it supports our hypotheses that rain effects the total number of bikes rented

##To check the relationship between count of bikes rented to temperature and feel temperature
plot(data$cnt,data$temp)
plot(data$cnt,data$atemp)
#from the above scatter plot we can infer that number of bikes rented is increasing with increase in temperature

##To check the relationship between count of bikes rented to the humidity
plot(data$cnt,data$hum)
#from the above scatter plot we can infer that the number of bikes rented isnt affected by humidity

##To check the distribution of registered users with year
boxplot(data$registered~data$yr,xlabel="Year",ylabel="Registered Users")
#We can infer that the number of registered users have increased with year

##################################################################################################
###               MISSING VALUE AND OUTLIER
##################################################################################################

colSums(is.na(data))
#There are no missing values in the dataframe

##Outlier analysis 
#We will check outliers of continous variables using boxplot method
#Temperature
boxplot(data$temp)

#Feel like temperature
boxplot(data$atemp)

#Humidity
boxplot(data$hum)
#there are two outliers
boxplot.stats(data$hum,coef=1.5)
#The two outlier values are 0.187 and 0.000
#the value of humidity can never be zero
#outliers has to be removed and knn imputation is to be used
data$hum[data$hum %in% boxplot.stats(data$hum)$out] <- NA

#Windspeed
boxplot(data$windspeed)
#there are a few outliers in windspeed
boxplot.stats(data$windspeed)
#there are 13 outliers in windspeed 
#these could be natural outliers due to certain weather situations like during thunderstorm and rain (weather sit 3)
data$weathersit[data$windspeed %in% boxplot.stats(data$windspeed)$out]
data$season[data$windspeed %in% boxplot.stats(data$windspeed)$out]
#the outlier in windspeed isn't during thunderstorm and rain 
#therefore we need to treat them for outliers
data$windspeed[data$windspeed %in% boxplot.stats(data$windspeed)$out] <- NA

data <- knnImputation(data)

#checking missing values again
colSums(is.na(data))

#Cross-check humidity and windspeed for outliers
boxplot(data$hum)
boxplot(data$windspeed)
#we can conclude that the outliers are safely removed

###################################################################################################
###               FEATURE ENGINEERING AND DATA PREPROCESSING
###################################################################################################

#the variables instant and date are not significant when training models as they are unique values to the data entry
#therefore removing the two variable from the dataset 
data <- subset(data, select= -c(instant,dteday))

str(data)

#Converting the data types from factor back to integer 
#As most of the machine learning algorithms produce better results with numeric variables only
data$season <- as.integer(data$season)
data$yr <- as.integer(data$yr)
data$workingday <- as.integer(data$workingday)
data$weathersit <- as.integer(data$weathersit)

#Checking the correlation of variables in the dataset

corr_data <- cor(subset(data,select=-c(cnt)))
corrplot(corr_data, method = "number")
#From the correlation plot above we can conclude the following
#Season and month are highly correlated hence month can be removed from the dataset
#temp and atemp are highly correlated hence atemp can be removed from the dataset

#As cnt varaible is a sum of registered and casual users, the variable registered and casual can also be considered as the target variable
#hence the variable registered and casual needs to be removed 

#Final dataset for training 
data <- subset(data,select = -c(mnth,atemp,registered,casual))

###################################################################################################
###             MODEL BUILDING
###################################################################################################

##We will use k-fold cross validation method in all the models to be trained below.
train_control <- trainControl(method="cv", number=5)

#####-----LINEAR REGRESSION-----#####

lr_model <- train(cnt~.,data=data,method="lm",trControl=train_control)
print(lr_model)
summary(lr_model)
##The accuracy of linear regression model is as follows
#RMSE=882.07  --  Rsquared=0.792  --  MAE=656.38


#####-----DECISION TREE-----#####

dt_model <- train(cnt~.,data=data,method="rpart",trControl=train_control)
plot(dt_model)
summary(dt_model)
print(dt_model)
##RMSE is used to select the optimal model using the smallest model .i.e. when cp=0.0801
#RMSE=1155.98  --  Rsquared=0.64  --  MAE=903.93

#####-----RANDOM FOREST-----#####

tgrid=expand.grid(.mtry=c(3:8),.splitrule ="variance",.min.node.size = c(5,10,15,20))
rf_model <- train(cnt~.,data=data,method="ranger",trControl=train_control,tuneGrid=tgrid,num.tree=200,importance="permutation")
plot(rf_model)
print(rf_model)
#From the plot above we could see that the optimal parameters are mtry=4, min.node.size=5
#RMSE=682.00  --  Rsquared=0.88  --  MAE=484.85


