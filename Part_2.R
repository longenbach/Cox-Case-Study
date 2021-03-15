###################
### 0) Packages & Data:
###################
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ISLR)
library(lamW) 

setwd("/Users/shlong/Desktop/Cox Case Study/input")
df = read.csv("ICO_acceptance_data.csv")
df$d = df$kbb_price  - df$trade_in_value  

###################
### 1) Quick Stats:
###################
summary(df) # 4757 / 15243 ~ 31% Accepted ICO Offer 
mean(df$d * df$accepted) # $1627.97 avg profit, assuming sold at KBB 

###################
### 2) Plots: 
###################
# ICO Offer vs. Kelly Blue Brook (KBB) 
df %>% 
  ggplot()+
  geom_point(aes(x=trade_in_value, y=kbb_price, color=accepted),alpha=0.5)

# delta = (KBB - ICO) vs. Accepted
df %>% 
  ggplot()+
  geom_boxplot(aes(y=d,x=as.factor(accepted)),color=c("dark red","dark green"))+
  scale_y_reverse() +
  coord_flip() +
  labs(x="Accepted",y="Delta = KBB - ICO")

df %>% 
  group_by(accepted) %>% 
  summarise(mean(d))
  
## Bin deltas
BINS = 20
bins_df = df %>% 
  mutate(Delta_Bins = cut(d, breaks = seq(from = min(d)-1, to = max(d)+1, length.out= BINS), dig.lab = 5) ) %>% 
  group_by(Delta_Bins) %>% 
  summarise(Offer_Count=n(),Accept_Proportion = mean(accepted), Total_Profit = sum(d*accepted), Avg_Profit = mean(d*accepted)) %>% 
  arrange(desc(Delta_Bins)) 

# Accept_Proportion: As delta(d) goes down, Accept_Proportion goes up. 
# Offer_Count: Most offers are $2000 to $10000 less than KBB value
# Total Profit: Majority of profit comes from offers $2000 to $10000 less than KBB value
# Avg_Profit: Ignoring low counts, offers $3500 to $6500 less than KBB value result in best avg profit

bins_df %>% 
  gather(key = "Statistic", value = "Measure",2:5) %>% 
  ggplot(aes(x=Delta_Bins,y=Measure,color=Statistic))+
  geom_col()+
  coord_flip()+
  facet_wrap(~Statistic, scales = "free")+
  theme(legend.position = "none")

bins_df %>% 
  select(-c('Total_Profit')) %>% 
  filter(Delta_Bins != '(19219,20277]') %>% 
  gather(key = "Statistic", value = "Measure",2:4) %>% 
  ggplot(aes(x=Delta_Bins,y=Measure,color=Statistic))+
  geom_col()+
  coord_flip()+
  facet_wrap(~fct_relevel(Statistic, "Offer_Count", after = 0), scales = "free")+
  theme(legend.position = "none")

###################
### 3) Generalized Linear Models: 
###################
delta_data = data.frame(d=seq(from = 0, to = max(df$d),by = 1))
delta_df = data.frame(d = delta_data$d)

#########
## Linear Model
#########
# Fit: p(delta) = a * delta + B
lm_fit = lm(accepted ~ d, data = df)
summary(lm_fit)
B_lm = lm_fit$coefficient[1]
a_lm = lm_fit$coefficient[2]

# Predict: profit = delta * p(delta) 
delta_df$lm_prob = predict(lm_fit,delta_data, type="response") # p(delta)
delta_df$lm_profit = delta_df$d * delta_df$lm_prob # delta * p(delta)

# Solve: derivative[delta * p(delta) w.r.t delta]
lm_optimal_delta_approx = delta_df$d[which.max(delta_df$lm_profit)]
lm_optimal_delta_exact = -B_lm / (2*a_lm) 
lm_max_exp_profit = (a_lm*lm_optimal_delta_exact + B_lm) * lm_optimal_delta_exact

# Plot:
delta_df %>% 
  filter(d<12500) %>% 
  ggplot() +
  geom_point(aes(x=d,y=lm_profit),color='#F8766D')+
  geom_point(aes(x=lm_optimal_delta_exact, y= lm_max_exp_profit),color='black') +
  labs(x="Delta = (KBB - ICO)", y="Expected Profit = Delta * p(Delta) ", title="Expected Profit (Linear Fit)")+
  ylim(c(0,2000))+
  scale_x_reverse() 
  
# Summary:
## Max [ Expected Profit(delta_i) = delta_i * p(delta_i) ] -> lm_optimal_delta_exact = delta_i
# delta_i = $6491.142 
# p(delta_i) = 0.3062831 
# Expected Profit = $1988.127

#########
## Logistic Model
#########
# Fit: log( p(delta) / 1 - p(delta) ) = a * delta + B
log_fit <- glm( accepted ~ d, data = df, family = binomial)
summary(log_fit)
B_log = log_fit$coefficient[1]
a_log = log_fit$coefficient[2]

# Predict: profit = delta * p(delta) 
delta_df$log_prob = predict(log_fit,delta_data, type="response") # p(delta)
delta_df$log_profit = delta_df$d * delta_df$log_prob # delta * p(delta)

# Solve: derivative[delta * p(delta) w.r.t delta]
log_optimal_delta_approx = delta_df$d[which.max(delta_df$log_profit)]
log_optimal_delta_exact = ( -lambertW0(exp(B_log-1) ) - 1) / a_log # https://www.wolframalpha.com/input/?i=derivative+of+d%2F%281+%2B+e%5E-%28ad%2Bb%29%29+wrt+d

logit2prob <- function(logit){
  odds <- exp(logit)
  prob <- odds / (1 + odds)
  return(prob)
}

log_max_exp_profit = logit2prob(a_log*log_optimal_delta_exact + B_log) * log_optimal_delta_exact

# Plot:
delta_df %>% 
  filter(d<20000) %>% 
  ggplot() +
  geom_point(aes(x=d,y=log_profit), color='#00BFC4')+
  geom_point(aes(x=log_optimal_delta_approx, y= log_max_exp_profit),color='black') +
  labs(x="Delta = (KBB - ICO)", y="Expected Profit = Delta * p(Delta) ", title="Expected Profit (Logistic Fit)")+
  ylim(c(0,2000))+
  scale_x_reverse() 

# Summary:
## Max [ Expected Profit(delta_i) = delta_i * p(delta_i) ] -> log_optimal_delta_exact = delta_i
# delta_i = $5758.303  
# p(delta_i) = 0.3255126 
# Expected Profit = $1874.4 

############
## Compare: Linear vs. Logistic Fits: 
df %>% 
  ggplot(mapping = aes(x = d, y = accepted)) +
  geom_hline(yintercept = 0, linetype = "dotted") +
  geom_hline(yintercept = 1, linetype = "dotted") +
  geom_smooth(mapping = aes(color = "Linear"), method = "glm", formula = y ~ x, method.args = list(family = gaussian(link = "identity")), se = FALSE) +
  geom_smooth(mapping = aes(color = "Logistic"), method = "glm", formula = y ~ x, method.args = list(family = binomial(link = "logit")), se = FALSE) +
  geom_point(size = 3, alpha = 0.50) +
  # Intercept:
  geom_hline(aes(yintercept=B_lm),color='#F8766D') +
  geom_hline(aes(yintercept=logit2prob(B_log)),color='#00BFC4') +
  # Optimal Delta:
  geom_vline(aes(xintercept= lm_optimal_delta_exact),color='#F8766D')+
  geom_vline(aes(xintercept= log_optimal_delta_exact),color='#00BFC4')+
  # P(Delta) @ Optimal Delta:
  geom_hline(aes(yintercept=(a_lm*lm_optimal_delta_exact + B_lm)),color='#F8766D') +
  geom_hline(aes(yintercept=logit2prob(a_log*log_optimal_delta_exact + B_log)),color='#00BFC4') +
  labs(x="Delta = (KBB - ICO)", y="Accepted", title="Linear Fit vs. Logistic Fit")+
  theme(legend.position = "none")+
  scale_x_reverse() 

## Compare: Linear vs. Logistic Profit
delta_df %>% 
  select(c("d","lm_profit","log_profit") ) %>% 
  gather(key="model",value="metric",2:3) %>% 
  filter(metric>0) %>% 
  ggplot() +
  geom_point(aes(x=d,y=metric,color=model))+
  labs(x="Delta = (KBB - ICO)", y="Expected Profit = Delta * p(Delta) ", title="Expected Profit Linear vs. Logistic")+
  theme(legend.position = "none")+
  ylim(c(0,2100))+
  geom_hline(aes(yintercept=log_max_exp_profit),color='#00BFC4') +
  geom_hline(aes(yintercept=lm_max_exp_profit),color='#F8766D') +
  geom_vline(aes(xintercept= lm_optimal_delta_exact),color='#F8766D')+
  geom_vline(aes(xintercept= log_optimal_delta_exact),color='#00BFC4')+
  geom_point(aes(x=log_optimal_delta_approx, y= log_max_exp_profit),color='black') +
  geom_point(aes(x=lm_optimal_delta_exact, y= lm_max_exp_profit),color='black') +
  scale_x_reverse() 
  
###################
### 4) Bootstrap: 
###################
library(boot)

## Linear Model
# Sample data w/ replacement 
# Calc optimal delta on sampled data
boot_linear.fn =function(data,index){ 
  fit = lm(accepted ~ d, data = data, subset = index)
  B = fit$coefficient[1]
  a = fit$coefficient[2]
  optimal_delta = -B / (2*a) 
  return(optimal_delta) 
}

boot_linear.fn(df,sample(15243,15243,replace=T))
lm_boot_1000 = boot(df ,boot_linear.fn ,1000)
plot(lm_boot_1000)

## Logistic Model
# Sample data w/ replacement 
# Calc optimal delta on sampled data
boot_logistic.fn =function(data,index){ 
  fit = glm( accepted ~ d, data = data, family = binomial, subset = index )
  B = fit$coefficient[1]
  a = fit$coefficient[2]
  optimal_delta = ( -lambertW0(exp(B-1) ) - 1) / a
  return(optimal_delta) 
  }

boot_logistic.fn(df,sample(15243,15243,replace=T))
log_boot_1000 = boot(df ,boot_logistic.fn ,1000)
plot(log_boot_1000)

## Compare Optimal Delta: Linear vs. Logistic: 
boots_df = data.frame(d_est= c(lm_boot_1000$t, log_boot_1000$t), model= rep(c("Linear","Logistic"),each=1000))
boots_df %>% 
  ggplot()+
  geom_boxplot(aes(x=model,y=d_est))+
  labs(y="Optimal Delta")+
  scale_y_reverse() 

  

