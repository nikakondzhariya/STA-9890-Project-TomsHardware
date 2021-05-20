rm(list = ls())    #delete objects
cat("\014")        #clear console

### REQUIRED LIBRARIES ###
library(readr)
library(tidyverse)
library(modelr)
library(glmnet)
library(randomForest)
library(reshape)
library(ggthemes)
library(gridExtra)

### OPERATIONS WITH DATA ###

# 1 Read the Data 
TomsHardware = read_csv("/Users/nikakondzhariya/Desktop/9890\ Project\ /TomsHardware.data", 
                         col_names = FALSE)
TomsHardware=as.data.frame(TomsHardware)
str(TomsHardware)
dim(TomsHardware)
summary(TomsHardware)

# 2 Change the colname for response variable
colnames(TomsHardware)[97] = "Res.Var.ND" # our response variable

# 3 Subset data to 1500 with no zeros in the response variable
TomsHardwareS=TomsHardware %>% filter(Res.Var.ND >0)
dim(TomsHardwareS)
# Final Dataset
set.seed(1)
#TomsHardwareR = sample_n(TomsHardwareS,1500)
#summary(TomsHardwareR ) 
#dim(TomsHardwareR)

# 4 See distribution of Y (Res.Var.ND) and make necessary adjustments 
hist(TomsHardwareS$Res.Var.ND) # we see that it is right-skewed 
Y=log(TomsHardwareS$Res.Var.ND) # normalize taking log of Y 
#Y=TomsHardware$Res.Var.ND
hist(Y) # now Y is normalized 

# 5 Create X matrix
#X = model.matrix(Res.Var.ND~.,TomsHardware)[,-1]
X=data.matrix(TomsHardwareS%>%select(-Res.Var.ND))
dim(X)

### TRAIN AND TEST: LASSO, ELASTIC-NET, RIDGE, RANDOM FORREST GENERATION WITH REQUIRED CALCULATIONS ###

# 1 Obtain n (number of observations) and p (number of predictors)
n=nrow(X)
p=ncol(X)

# 2 Define Modeling Parameters including train (n.train) and test (n.test) number of observations
#d.rate=0.8 # division rate for train and test data
d.repit=100 # repeat the ongoing tasks 100 times 
n.train=1000
n.test=n-n.train

# 3 Prepare the following zero-filled matrices that we are going to fill as values are obtained 

# Matrices for train and test R-squared 
Rsq.train=matrix(0,d.repit,4)
colnames(Rsq.train)=c("Lasso","Elastic-Net","Ridge","Random Forrest")
Rsq.test=matrix(0,d.repit,4)
colnames(Rsq.test)=c("Lasso","Elastic-Net","Ridge","Random Forrest")

# Matrix for the time it takes to cross-validate Lasso/Elastic-Net/Ridge regression
Time.cv=matrix(0,d.repit,3)
colnames(Time.cv)=c("Lasso","Elastic-Net","Ridge")

# Matrices for estimated coefficients 
La.coef=matrix(0,nrow=p+1,ncol=d.repit)
row.names(La.coef) <- c("Intercept",colnames(X))
El.coef=matrix(0,nrow=p+1,ncol=d.repit)
row.names(El.coef) <- c("Intercept",colnames(X))
Ri.coef=matrix(0,nrow=p+1,ncol=d.repit)
row.names(Ri.coef) <- c("Intercept",colnames(X))

# 4 Fit Lasso, Elastic-Net, Ridge and Random forest (100 times repition)

for (i in c(1:d.repit)) {
  
  cat("d.repit = ", i, "\n")
  
  shuffled_indexes =  sample(n)
  train            =  shuffled_indexes[1:n.train]
  test             =  shuffled_indexes[(1+n.train):n]
  X.train          =  X[train, ]
  Y.train          =  Y[train]
  X.test           =  X[test, ]
  Y.test           =  Y[test]
  
  # Fit Lasso, calculate time, R-squared (both train and test), and estimated coefficients 
  
  time.start       =  Sys.time()
  cv.fit           =  cv.glmnet(X.train, Y.train, alpha = 1, nfolds = 10)
  time.end         =  Sys.time()
  Time.cv[i,1]        =  time.end - time.start
  fit              =  glmnet(X.train, Y.train, alpha = 1, lambda = cv.fit$lambda.min)
  Y.train.hat      =  predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  Y.test.hat       =  predict(fit, newx = X.test, type = "response")  # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test[i,1]   =  1-mean((Y.test - Y.test.hat)^2)/mean((Y.test - mean(Y.test))^2)
  Rsq.train[i,1]    =  1-mean((Y.train - Y.train.hat)^2)/mean((Y.train - mean(Y.train))^2)
  La.coef[,i]      =  predict(fit, newx = X.test, type = "coefficients")[,1]
  
  # Fit Elastic-Net , calculate time, R-squared (both train and test), and estimated coefficients 
  
  time.start       =  Sys.time()
  cv.fit           =  cv.glmnet(X.train, Y.train, alpha = 0.5, nfolds = 10)
  time.end         =  Sys.time()
  Time.cv[i,2]        =  time.end - time.start
  fit              =  glmnet(X.train, Y.train, alpha = 0.5, lambda = cv.fit$lambda.min)
  Y.train.hat      =  predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  Y.test.hat       =  predict(fit, newx = X.test, type = "response")  # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test[i,2]   =  1-mean((Y.test - Y.test.hat)^2)/mean((Y.test - mean(Y.test))^2)
  Rsq.train[i,2]    =  1-mean((Y.train - Y.train.hat)^2)/mean((Y.train - mean(Y.train))^2)
  El.coef[,i]      =  predict(fit, newx = X.test, type = "coefficients")[,1]
  
  
  # Fit Ridge , calculate time, R-squared (both train and test), and estimated coefficients 
  
  time.start       =  Sys.time()
  cv.fit           =  cv.glmnet(X.train, Y.train, alpha = 0, nfolds = 10)
  time.end         =  Sys.time()
  Time.cv[i,3]        =  time.end - time.start
  fit              =  glmnet(X.train, Y.train, alpha = 0, lambda = cv.fit$lambda.min)
  Y.train.hat      =  predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  Y.test.hat       =  predict(fit, newx = X.test, type = "response")  # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test[i,3]   =  1-mean((Y.test - Y.test.hat)^2)/mean((Y.test - mean(Y.test))^2)
  Rsq.train[i,3]    =  1-mean((Y.train - Y.train.hat)^2)/mean((Y.train - mean(Y.train))^2)
  Ri.coef[,i]      =  predict(fit, newx = X.test, type = "coefficients")[,1]
  
  # Fit Random Forest , calculate time, R-squared (both train and test)
  
  rf               =  randomForest(X.train, Y.train, mtry = p/3, importance = TRUE)
  Y.train.hat      =  predict(rf, X.train)
  Y.test.hat       =  predict(rf, X.test)
  Rsq.test[i,4]   =  1-mean((Y.test - Y.test.hat)^2)/mean((Y.test - mean(Y.test))^2)
  Rsq.train[i,4]    =  1-mean((Y.train - Y.train.hat)^2)/mean((Y.train - mean(Y.train))^2)
  
}

# Record average time it takes to cross-validate Lasso/Elastic-Net/Ridge regression
Time.cv.average=apply(Time.cv, 2, mean) # to be used in 4c

# Record average test R-squared for each model 
Rsq.test.average=apply(Rsq.test, 2, mean) # to be used in 5b

# 5 Side-by-Side Boxplots of R-squared 
#Rsq.train.boxplot=ggplot(melt(data.frame(Rsq.train)), aes(factor(variable), value,color=variable))+
#  geom_boxplot()+
#  ylim(0.5,1)+
#  ggtitle("R-squared for Train Data")+
#  theme(plot.title=element_text(hjust=0.5))+
#  theme_few()+
#  theme(legend.position="none")+
#  theme(axis.title.x=element_blank())+ 
#  theme(axis.title.y=element_blank())
  
#Rsq.test.boxplot=ggplot(melt(data.frame(Rsq.test)), aes(factor(variable), value,color=variable))+
#  geom_boxplot()+
#  ylim(0.5,1)+
#  ggtitle("R-squared for Test Data")+
#  theme(plot.title=element_text(hjust=0.5))+
#  theme_few()+
#  theme(legend.position="none")+
#  theme(axis.title.x=element_blank())+ 
#  theme(axis.title.y=element_blank())
  

#Rsq.Boxplot=grid.arrange(Rsq.train.boxplot,Rsq.test.boxplot, nrow = 1)

# ANOTHER way to draw Rsq boxplots (choose whichever is better)
par(mfrow=c(1,2))
a=boxplot(Rsq.train[,1], Rsq.train[,2],Rsq.train[,3],Rsq.train[,4],
                        main = "R-squared for Train Data",
                        names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
                        col=c("blue","yellow", "green","red"),
                        ylim=c(0.5,1))

b=boxplot(Rsq.test[,1], Rsq.test[,2],Rsq.test[,3],Rsq.test[,4],
                        main = "R-squared for Test Data",
                        names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
                        col=c("blue","yellow", "green","red"),
                        ylim=c(0.5,1))
  


# 6 For one on the 100 samples, create 10-fold CV curves for Lasso, Elastic-Net,Ridge

cv.la.1=cv.glmnet(X.train, Y.train, alpha = 1, nfolds = 10)
cv.el.1=cv.glmnet(X.train, Y.train, alpha = 0.5, nfolds = 10)
cv.ri.1=cv.glmnet(X.train, Y.train, alpha = 0, nfolds = 10)

par(mfrow=c(1,3))
plot(cv.la.1)
title('Lasso', line = 2.5)
plot(cv.el.1)
title('Elastic-Net', line = 2.5)
plot(cv.ri.1)
title('Ridge', line = 2.5)

# 7 Let's record residuals for every single model for one on the 100 samples
# and show the side-by-side boxplots of train and test residuals

# Lasso 
cv.la.2=cv.glmnet(X.train, Y.train, alpha = 1, nfolds = 10)
fit.la.2=glmnet(X.train, Y.train, alpha = 1, lambda = cv.la.2$lambda.min)
# Lasso Residuals 
Y.train.hat.la=predict(fit.la.2, newx = X.train, type = "response")
Y.test.hat.la=predict(fit.la.2, newx = X.test, type = "response")
Res.train.la=Y.train - Y.train.hat.la
Res.train.la=as.vector(Res.train.la)
Res.test.la=Y.test - Y.test.hat.la
Res.test.la=as.vector(Res.test.la)

# Elastic_Net
cv.el.2=cv.glmnet(X.train, Y.train, alpha = 0.5, nfolds = 10)
fit.el.2=glmnet(X.train, Y.train, alpha = 0.5, lambda = cv.el.2$lambda.min)
# Elastic_Net Residuals 
Y.train.hat.el=predict(fit.el.2, newx = X.train, type = "response")
Y.test.hat.el=predict(fit.el.2, newx = X.test, type = "response")
Res.train.el=Y.train - Y.train.hat.el
Res.train.el=as.vector(Res.train.el)
Res.test.el=Y.test - Y.test.hat.el
Res.test.el=as.vector(Res.test.el)

# Ridge
cv.ri.2=cv.glmnet(X.train, Y.train, alpha = 0, nfolds = 10)
fit.ri.2=glmnet(X.train, Y.train, alpha = 0, lambda = cv.ri.2$lambda.min)
# Ridge Residuals 
Y.train.hat.ri=predict(fit.ri.2, newx = X.train, type = "response")
Y.test.hat.ri=predict(fit.ri.2, newx = X.test, type = "response")
Res.train.ri=Y.train - Y.train.hat.ri
Res.train.ri=as.vector(Res.train.ri)
Res.test.ri=Y.test - Y.test.hat.ri
Res.test.ri=as.vector(Res.test.ri)

# Random Forrest 
rf.2=randomForest(X.train, Y.train, mtry = p/3, importance = TRUE)
# Random Forest Residuals 
Y.train.hat.rf=predict(rf.2, X.train)
Y.test.hat.rf=predict(rf.2, X.test)
Res.train.rf=Y.train - Y.train.hat.rf
Res.train.rf=as.vector(Res.train.rf)
Res.test.rf=Y.test - Y.test.hat.rf
Res.test.rf=as.vector(Res.test.rf)

# Create Residual Boxplots 
par(mfrow=c(1,2))
Res.train.boxplot=boxplot(Res.train.la, Res.train.el, Res.train.ri, Res.train.rf,
        main = "Residuals for Train Data",
        names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
        col=c("blue","yellow", "green","red"),
        ylim=c(-13,8))

Res.test.boxplot=boxplot(Res.test.la, Res.test.el, Res.test.ri, Res.test.rf,
        main = "Residuals for Test Data",
        names = c("Lasso", "Elastic-Net", "Ridge", "Random Forest"),
        col=c("blue","yellow", "green","red"),
        ylim=c(-13,8))


### FULL DATASET: LASSO, ELASTIC-NET, RIDGE, RANDOM FORREST GENERATION WITH REQUIRED CALCULATIONS ###

#Lasso 

time.start.la=Sys.time()
cv.fit.la=cv.glmnet(X,Y, alpha = 1, nfolds = 10)
fit.la=glmnet(X, Y, alpha = 1, lambda = cv.fit.la$lambda.min)
#Y.hat.la=predict(fit.la, newx = X, type = "response") # ASK QUESTION ABOUT IT
time.end.la=Sys.time()
Time.la=time.end.la - time.start.la

# Coefficients Lasso
betaS.la=data.frame(c(1:p), as.vector(fit.la$beta))
colnames(betaS.la)     =     c( "feature", "value")

# Elastic-Net
time.start.el=Sys.time()
cv.fit.el=cv.glmnet(X,Y, alpha = 0.5, nfolds = 10)
fit.el=glmnet(X, Y, alpha = 0.5, lambda = cv.fit.el$lambda.min)
#Y.hat.el=predict(fit.el, newx = X, type = "response") # ASK QUESTION ABOUT IT
time.end.el=Sys.time()
Time.el=time.end.el - time.start.el

# Coefficients Elastic-Net
betaS.el=data.frame(c(1:p), as.vector(fit.el$beta))
colnames(betaS.el)     =     c( "feature", "value")

# Ridge
time.start.ri=Sys.time()
cv.fit.ri=cv.glmnet(X,Y, alpha = 0, nfolds = 10)
fit.ri=glmnet(X, Y, alpha = 0, lambda = cv.fit.ri$lambda.min)
#Y.hat.ri=predict(fit.ri, newx = X, type = "response") # ASK QUESTION ABOUT IT
time.end.ri=Sys.time()
Time.ri=time.end.ri - time.start.ri

# Coefficients Ridge
betaS.ri=data.frame(c(1:p), as.vector(fit.ri$beta))
colnames(betaS.ri)     =     c( "feature", "value")

# Random Forrest
time.start.rf=Sys.time()
rf.wh=randomForest(X, Y, mtry = p/3, importance = TRUE)
time.end.rf=Sys.time()
Time.rf=time.end.rf - time.start.rf

# Coefficients Random Forrest
betaS.rf=data.frame(c(1:p), as.vector(rf.wh$importance[1:p]))
colnames(betaS.rf)     =     c( "feature", "value")

# Result for time (required for 5b)
Time.for.4m=rbind(Time.la,Time.el,Time.ri,Time.rf)
colnames(Time.for.4m)=c("Time Elapsed")

# Create 90% test R2 intervals based on the 100 samples for each model
interval.la=quantile(Rsq.test[,1], c(0.05,0.95))
interval.en=quantile(Rsq.test[,2], c(0.05,0.95))
interval.ri=quantile(Rsq.test[,3], c(0.05,0.95))
interval.rf=quantile(Rsq.test[,4], c(0.05,0.95))

# Result for intervals (required for 5b)
Inteval.90.testRsq=rbind(interval.la,interval.en,interval.ri,interval.rf)

### BAR-PLOTS OF THE ESTIMATED COEFFICIENTS (LASS, ELASTIC-NET, RIDGE), THE IMPORTANCE OF RF PARAMETERS ###

# we need to change the order of factor levels by specifying the order explicitly.
betaS.el$feature=factor(betaS.el$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.rf$feature=factor(betaS.rf$feature,levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.la$feature=factor(betaS.la$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])
betaS.ri$feature=factor(betaS.ri$feature, levels = betaS.el$feature[order(betaS.el$value, decreasing = TRUE)])

# Let's use elastic-net estimated coefficients to create an order based on largest to smallest coefficients, and 
# use this order to present bar-plots of the estimated coefficients of all 4 models

# Lasso Plot
laPlot =  ggplot(betaS.la, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Lasso))

# Elastic-Net Plot
elPlot =  ggplot(betaS.el, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Elastic-Net))+
  ylim(-50,40)

# Ridge 
riPlot =  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")+
  labs(x = element_blank(), y = "Coefficients", title = expression(Ridge))+
  ylim(-50,40)

# Random Forrest
rfPlot=ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  labs(x = element_blank(), y = "Importance", title = expression(Random.Forest))
  
Coef.Plot=grid.arrange(laPlot,elPlot, riPlot, rfPlot, nrow = 4)
  
  
  






