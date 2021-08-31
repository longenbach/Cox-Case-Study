###################
### Packages & Data:
###################
library(ggplot2)
library(dplyr)
library(glmnet)
library(lubridate)
library(tidyr)
library(randomForest)

setwd("/Users/shlong/Desktop/Cox Case Study/input")
df = read.csv("ICO_modeling_data.csv")
df$offer_date = as.Date(df$offer_date,'%m/%d/%Y')
str(df)
summary(df)
quantile(df$trade_in_value) 

###################
### Basic Features:
###################
df = df %>% 
  mutate( Age = (as.numeric(format(offer_date, "%Y")) + yday(offer_date)/365 ) -  as.numeric(model_yr)) %>% 
  mutate( Year = format(offer_date, "%Y")) %>% 
  mutate( Month = format(df$offer_date, "%m") ) %>% 
  mutate( DayofWeek = weekdays(offer_date) ) %>% 
  mutate(accident_count = ifelse(is.na(accident_count),0,accident_count))
str(df)

numeric_features = c('mileage', 'reported_condition', 'auctionavgmiles_mmr', 'avgcond_mmr', 
                      'costfactor_condition', 'costfactor_color', 'accident_count', 'currdeprperweek', 
                      'ammr', 'forecast_mmr','Age')

df_numeric = df %>% select(c(numeric_features,trade_in_value))  
summary(df_numeric)

###################
## Simple Train / Test Split:
###################
split = 5  # 1/split
NR=nrow(df_numeric)
index = sample(1:split,NR,rep=T)
INDEX_test <- index==1 

train_has_na = df_numeric[!INDEX_test,]
test_has_na = df_numeric[INDEX_test,]

###################
## Impute NAs:
###################
colMeans_train = colMeans(train_has_na,na.rm=T) # Fit
colMeans(test_has_na,na.rm=T) # In theory, would not know test set 

train = replace_na(train_has_na,as.list(colMeans_train)) # Transform
test = replace_na(test_has_na,as.list(colMeans_train)) # Transform
sum(is.na(train))
sum(is.na(test))

###################
## Models
###################

###################
##### Linear Model
lm_fit = lm(trade_in_value ~., data = train) # Fit
summary(lm_fit)
pred.lm = predict(lm_fit,test, type="response") # Predict

## Error
#mean(test$trade_in_value - pred.lm )
mean(abs(test$trade_in_value - pred.lm )) 
mean((test$trade_in_value - pred.lm )^2)

# #Count Errors 
#Under
sum(ifelse(test$trade_in_value > pred.lm,1,0))
#Over
sum(ifelse(test$trade_in_value < pred.lm,1,0))

## Avg Residual Over/ Under 
Avg_Over = mean(ifelse(test$trade_in_value - pred.lm > 0,test$trade_in_value - pred.lm,NA),na.rm = TRUE)
Avg_Under = mean(ifelse(test$trade_in_value - pred.lm < 0,test$trade_in_value - pred.lm,NA),na.rm = TRUE)

## Residuals
test %>% 
  ggplot()+
  geom_point(aes(x=pred.lm,y=trade_in_value - pred.lm ,color=is.na(test_has_na$ammr) ),alpha = 0.5)+
  theme(legend.position = "none")+
  labs(x="ICO Predictions", y= "Actual ICO - Predicted ICO", title = 'Linear Regression Errors') +
  ylim(c(-2500,2500))+
  geom_hline(yintercept=0)+
  geom_hline(yintercept=Avg_Over,linetype = "dashed") +
  geom_hline(yintercept=Avg_Under,linetype = "dashed") 
  

###################
##### Lasso Model
NC=ncol(df_numeric) # last column is 'trade_in_value'

## Need in matrix format
train_mat = data.matrix(train)
test_mat = data.matrix(test)

x.train <- train_mat[,-NC]
y.train <- train_mat[,NC]
x.test <- test_mat[,-NC]
y.test <- test_mat[,NC]

## CV for Lambda (lasso penalty) 
numLambda <- 500
expVals <- seq(5,-2,length=numLambda)#seq(-4,4,length=numLambda)
lam.grid <- 10^expVals

cv.lasso <- cv.glmnet(x.train,y.train,alpha=1,lambda=lam.grid,nfolds = 5, standardize = TRUE)
plot(cv.lasso)

(lamb0 <- cv.lasso$lambda.min)

## Fit 
mod.lasso <- glmnet(x.train,y.train,alpha=1,lambda=lamb0, standardize = TRUE)
coef.lasso <- as.matrix(coef(mod.lasso))
pred.lasso <- predict(mod.lasso,newx=x.test)

## Error
mean(abs(y.test - pred.lasso ))
mean((y.test - pred.lasso )^2)

## Residuals
test %>% 
  ggplot()+
  geom_point(aes(x=pred.lasso,y=trade_in_value - pred.lasso ,color=is.na(test_has_na$ammr) ),alpha = 0.5)+
  theme(legend.position = "none")

##### Linear & Lasso coffecients close ( cause lamb0 ~ 0): 
data.frame(lm = round(lm_fit$coefficients,2), lasso = round(coef.lasso,2))

###################
##### Random Forest 
nc <- round((ncol(train)-1) / 3)
rf.model <- randomForest(trade_in_value~., data=train, mtry=nc, ntree=200)# importance =TRUE)
preds.rf <- predict(rf.model,newdata=test)

## Error
mean(abs(test$trade_in_value - preds.rf ))
mean((test$trade_in_value - preds.rf )^2)

# #Count Errors 
#Under
sum(ifelse(test$trade_in_value > preds.rf,1,0))
#Over
sum(ifelse(test$trade_in_value < preds.rf,1,0))

## Avg Residual Over/ Under 
#sum(ifelse(test$trade_in_value - pred.lm < 0,1,0))
Avg_Over_rf = mean(ifelse(test$trade_in_value - preds.rf > 0,test$trade_in_value - preds.rf,NA),na.rm = TRUE)
Avg_Under_rf = mean(ifelse(test$trade_in_value - preds.rf < 0,test$trade_in_value - preds.rf,NA),na.rm = TRUE)

## Residuals
test %>% 
  ggplot()+
  geom_point(aes(x=preds.rf,y=trade_in_value - preds.rf ,color=is.na(test_has_na$ammr) ),alpha = 0.5)+
  theme(legend.position = "none") +
  labs(x="ICO Predictions", y= "Actual ICO - Predicted ICO", title = 'Random Forest Errors')+
  geom_hline(yintercept=0)+
  geom_hline(yintercept=Avg_Over_rf, linetype = "dashed") +
  geom_hline(yintercept=Avg_Under_rf, linetype = "dashed") +
  ylim(c(-2500,2500))

test %>% 
  ggplot()+
    geom_boxplot(aes(x=trade_in_value - preds.rf))


# varImpPlot(rf.model )
# importance (rf.model)
# rf.model

## Importance Predictors
imp.vals <- rf.model$importance
nn <- length(imp.vals)
imp <- as.numeric(imp.vals)
rnks <- rank(imp)
ords <- order(imp)
vars <- rownames(imp.vals)[ords]

df00 <- data.frame(val=rnks,var=vars[rnks],importance=as.numeric(imp.vals))

df00%>%
  ggplot()+
  geom_segment(aes(x=val,xend=val,y=0,yend=importance),size=1,color="blue")+
  geom_point(aes(x=val,y=importance),size=2,color="blue")+
  scale_x_continuous("",labels=vars,breaks=1:nn)+
  scale_y_continuous("Importance")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  coord_flip()+
  theme(plot.title = element_text(hjust = 0))

#############
### Plots:
#############

## NA Over Time 
df %>% 
  ggplot() + 
  geom_histogram(aes(x=offer_date,color=is.na(forecast_mmr)))

df %>% 
  ggplot() + 
  geom_histogram(aes(x=offer_date),color='darkblue')+
  theme(legend.position = "none") +
  labs(y="Rows", x="Offer Date") 
  
df %>% 
  ggplot()+
  geom_histogram(aes(x=trade_in_value ))
  
### Missing Values: 
(sum(is.na(df$ammr))/length(df$ammr)) * 100
(sum(is.na(df$forecast_mmr))/length(df$forecast_mmr)) * 100
(sum(is.na(df$avgcond_mmr))/length(df$avgcond_mmr)) * 100

### NA filled with mean
df %>% 
  mutate( was_NA = is.na(ammr)) %>% 
  mutate(ammr = ifelse(is.na(ammr), mean(ammr, na.rm=TRUE), ammr)) %>% 
  ggplot(aes(y=trade_in_value,x=ammr, color = was_NA )) + 
  geom_point(alpha=0.5) +
  geom_smooth() +
  labs(y="Instant Cash Offer", x= "Ammr = Whole price estimate from Manheim Market Report" ) +
  theme(legend.position = "none") 



#######################################
########################## IGNORE:
#######################################

# Most Expensive Models: 
df %>% 
  ggplot() + 
  geom_boxplot(aes(x=reorder(make, trade_in_value, mean),y=trade_in_value))+
  coord_flip()+
  labs(x="Instant Cash Offer (ICO)")

## Add Car Group Feature
# Note if more time would cross validate to determine categories 
p1 = list(levels(reorder(df$make, df$trade_in_value, mean))[1:8])
p2 =levels(reorder(df$make, df$trade_in_value, mean))[9:25]
p3 = levels(reorder(df$make, df$trade_in_value, mean))[26:39]
p4 = levels(reorder(df$make, df$trade_in_value, mean))[40:46]

df$offer_date.day

## Vehicle make -> (make) 
df %>% # Tesla & Alfa Romeo most expensive 
  ggplot() + 
  geom_boxplot(aes(x=reorder(make, trade_in_value, mean),y=trade_in_value))+
  coord_flip()+
  labs(x="Instant Cash Offer (ICO)")

df %>% # Ford & Chevy most popular
  ggplot() + 
  geom_boxplot(aes(x=reorder(make, trade_in_value, length),y=trade_in_value))+
  coord_flip()

## Vehicle model -> (model)

df %>%  # Nissan Altima 4c & Sentra most popular 
  group_by(make,model) %>% 
  summarise(count=n(),avg_value = mean(trade_in_value)) %>% 
  arrange(desc(count))

df %>% # Mercedes AMG GT most expensive (small sample size)
  group_by(make,model) %>% 
  summarise(avg_value = mean(trade_in_value),count=n()) %>% 
  arrange(desc(avg_value))

## Vehicle color -> (color)
df %>%  # Black & White most popular
  group_by(color) %>% 
  summarise(avg_value = mean(trade_in_value),count=n()) %>% 
  arrange(desc(count))

# NA = 170 
# Unknown = 555
df %>%  
  filter(is.na(df$color)) %>% 
  group_by(make,model) %>% 
  summarise(count=n(),avg_value = mean(trade_in_value)) %>% 
  arrange(desc(count))

## Body Description -> (body_desc)
df %>% 
  ggplot() + 
  geom_boxplot(aes(x=reorder(body_desc, trade_in_value, length),y=trade_in_value))+
  coord_flip()

# Unknown - 420
df %>%  
  filter(body_desc == 'Unknown') %>% 
  group_by(make,model) %>% 
  summarise(count=n(),avg_value = mean(trade_in_value)) %>% 
  arrange(desc(count))

# NA - 591
df %>%  
  filter(is.na(df$body_desc)) %>% 
  group_by(make,model) %>% 
  summarise(count=n(),avg_value = mean(trade_in_value)) %>% 
  arrange(desc(count))

abc = df %>%  
  group_by(make,model,body_desc) %>% 
  summarise(count=n(),avg_value = mean(trade_in_value)) %>% 
  arrange(desc(count))

## Model Year -> (model_yr)
df %>% 
  ggplot() + 
  geom_boxplot(aes(x=as.factor(model_yr),y=trade_in_value))+
  coord_flip()

## Mileage -> (mileage)
df %>% 
  ggplot() + 
  geom_point(aes(x=mileage,y=trade_in_value))
df %>% 
  ggplot() + 
  geom_histogram(aes(x=mileage),bins=200)

## Report Condition 
df %>% 
  ggplot() + 
  geom_point(aes(x=reported_condition,y=trade_in_value))

# NA = 5938
df %>%  
  filter(is.na(df$reported_condition)) %>% 
  group_by(make,model) %>% 
  summarise(count=n(),avg_value = mean(trade_in_value)) %>% 
  arrange(desc(count))

## Auctionavgmiles_mmr
df %>% 
  ggplot() + 
  geom_point(aes(x=auctionavgmiles_mmr,y=trade_in_value))

## NA = 607
df %>%  
  filter(is.na(df$auctionavgmiles_mmr)) %>% 
  group_by(make) %>% 
  summarise(count=n(),avg_value = mean(trade_in_value)) %>% 
  arrange(desc(count))

## Accident count 
gg = df %>%  filter(df$accident_count==0) 

df %>% 
  ggplot(aes(x=ammr,y=forecast_mmr)) + 
  geom_point()

df %>% 
  ggplot(aes(x=ammr,y=trade_in_value)) + 
  geom_point()

df %>% 
  ggplot(aes(x=currdeprperweek,y=trade_in_value)) + 
  geom_point()

df %>% 
  ggplot(aes(x=ammr,y=forecast_mmr)) + 
  geom_point()

df %>% 
  ggplot(aes(x=ammr,y=trade_in_value)) + 
  geom_point()


df %>% 
  mutate( was_NA = is.na(forecast_mmr)) %>% 
  mutate(forecast_mmr = ifelse(is.na(forecast_mmr), mean(forecast_mmr, na.rm=TRUE), forecast_mmr)) %>% 
  ggplot(aes(y=trade_in_value,x=forecast_mmr, color = was_NA )) + 
  geom_point()


df %>% 
  mutate( was_NA = is.na(ammr)) %>% 
  mutate(ammr = ifelse(is.na(ammr), mean(ammr, na.rm=TRUE), ammr)) %>% 
  ggplot(aes(y=trade_in_value,x=ammr, color = was_NA )) + 
  geom_point()

mmr_df = df %>% 
  mutate( ammr_was_NA = is.na(ammr)) %>% 
  mutate(ammr = ifelse(is.na(ammr), mean(ammr, na.rm=TRUE), ammr)) %>% 
  mutate( forecast_mmr_NA = is.na(forecast_mmr)) %>% 
  mutate(forecast_mmr = ifelse(is.na(forecast_mmr), mean(forecast_mmr, na.rm=TRUE), forecast_mmr)) 

lm_model = lm(trade_in_value~ammr,data = mmr_df)
summary(lm_model)
lm_model$coefficients

lm_model = lm(trade_in_value~ammr+forecast_mmr,data = mmr_df)
summary(lm_model)
lm_model$coefficients

df %>% 
  mutate(day_of_week = weekdays(offer_date)) %>% 
  group_by(day_of_week) %>% 
  summarise(offer_count=n())

rr = df %>% 
  filter(offer_date > as.Date("01/01/2020",'%m/%d/%Y') ) %>% 
  mutate(day_of_week = weekdays(offer_date)) %>% 
  group_by(offer_date) %>% 
  summarise(offer_count=n()) %>% 
  ggplot() +
  geom_col(aes(x=offer_date,offer_count))

df %>% 
  mutate(day_of_week = weekdays(offer_date)) %>% 
  group_by(offer_date) %>% 
  summarise(na_count=sum(is.na(forecast_mmr)),offer_count= n(), per = na_count/offer_count) %>% 
  ggplot() +
  geom_col(aes(x=offer_date,per))

df %>% 
  mutate(day_of_week = weekdays(offer_date)) %>% 
  group_by(offer_date) %>% 
  summarise(na_count=sum(is.na(ammr)),offer_count= n(), per = na_count/offer_count) %>% 
  ggplot() +
  geom_col(aes(x=offer_date,per))
  
df %>% 
  mutate(day_of_week = as.numeric(format(offer_date, "%m"))) %>% 
  ggplot()+
  geom_point(aes(x=day_of_week,y=trade_in_value))

df %>% 
    ggplot() + 
    geom_histogram(aes(x=offer_date,color=is.na(ammr)))+
    theme(legend.position = "none")
  
df %>% 
  ggplot() + 
  geom_histogram(aes(x=offer_date,color=is.na(forecast_mmr)))

df %>% 
  ggplot() + 
  geom_histogram(aes(x=offer_date,color=is.na(ammr)))

df %>% 
  ggplot() + 
  geom_histogram(aes(x=offer_date))
