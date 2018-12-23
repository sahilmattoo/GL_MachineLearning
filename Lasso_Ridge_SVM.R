#Some libraries
library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
#library(ggord)
library(ggplot2)
library(Hmisc)
#library(klaR)
#library(klaR)
library(MASS)
#library(nnet)
library(plyr)
library(pROC)
library(psych)
library(scatterplot3d)
library(SDMTools)
library(dplyr)
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
library(randomForest)
#library(neuralnet)

#setwd( "")
Loan<-read.csv(file.choose(), header=T)
na.omit(Loan)

summary(Loan)
str(Loan)

levels(Loan$Purpose)
levels(Loan$Job)
#Define some dummies
Loan$Default<-ifelse(Loan$Status=="Default",1,0)
Loan$Female<-ifelse(Loan$Gender=="Female",1,0)
Loan$Management<-ifelse(Loan$Job=="Management",1,0)
Loan$Skilled<-ifelse(Loan$Job=="skilled",1,0)
Loan$Unskilled<-ifelse(Loan$Job=="unskilled",1,0)


Loan$CH.Poor<-ifelse(Loan$Credit.History=="Poor",1,0)
Loan$CH.critical<-ifelse(Loan$Credit.History=="critical",1,0)
Loan$CH.good<-ifelse(Loan$Credit.History=="good",1,0)
Loan$CH.verygood<-ifelse(Loan$Credit.History=="very good",1,0)

Loan$Purpose.car<-ifelse(Loan$Purpose=="car",1,0)
Loan$Purpose.cd<-ifelse(Loan$Purpose=="consumer.durable",1,0)
Loan$Purpose.education<-ifelse(Loan$Purpose=="education",1,0)
Loan$Purpose.personal<-ifelse(Loan$Purpose=="personal",1,0)


#Partitioning Data Sets
#Partition train and val
#We will use this throughout so that samples are comparable
set.seed(1234)
pd<-sample(2,nrow(Loan),replace=TRUE, prob=c(0.7,0.3))

train<-Loan[pd==1,]
val<-Loan[pd==2,]

str(Loan)
library(ISLR)
library(leaps)

#Use reg subset to evaluate best subset (default is 8)
Linear<-EMI.Ratio~Loan.Offered+Work.Exp+Credit.Score+Own.house+Dependents +Female+Management+Skilled+CH.Poor+CH.good+CH.critical+Purpose.car+Purpose.education+Purpose.personal
regfitfull<-regsubsets(Linear, data=Loan)
reg.summary<-summary(regfitfull)
reg.summary
#reg subset for all
regfit.full<-regsubsets(Linear, data=Loan,nvmax=24)
reg.summary.full<-summary(regfit.full)
reg.summary.full
names(reg.summary.full)
##### Statistical methods "which"  "rsq"    "rss"    "adjr2"  "cp"     "bic"    "outmat" "obj"

reg.summary.full$rsq
reg.summary.full$adjr2
###So how many?



par(mfrow =c(1,1))
plot(reg.summary.full$rss ,xlab=" Number of Variables ",ylab=" RSS",
     type="l")

# Min - RSS gives which we need to be included in model
rss.opt<-which.min(reg.summary.full$rss)
rss.opt
coef(regfit.full,rss.opt)


#Use Adjusted R Square
plot(reg.summary.full$adjr2 ,xlab =" Number of Variables ",
     ylab=" Adjusted RSq",type="l")

# Rsquare is max
Adrsq.opt<-which.max(reg.summary.full$adjr2)
Adrsq.opt
coef(regfit.full,Adrsq.opt)


#Use Cp
plot(reg.summary.full$cp,xlab="number of variables", ylab="Cp Statistic")
Cp.opt<-which.min(reg.summary.full$cp)
Cp.opt
coef(regfit.full,Cp.opt)

#Use BIC
plot(reg.summary.full$bic,xlab="number of variables", ylab="BIC")
BIC.opt<-which.min(reg.summary.full$bic)
BIC.opt
coef(regfit.full,BIC.opt)



##Forward and Backward Selection
#Forward Selection
regfit.fwd<-regsubsets(Linear, data=Loan, nvmax=11, method="forward")

coef(regfit.fwd,11)
#Backward Selection
regfit.bwd<-regsubsets(Linear, data=Loan, nvmax=11, method="backward")

coef(regfit.bwd,11)

####Best Model -- Used RSS

Loan.work2<-Loan[,c(6,2,4,5,9,11,13:15,18:19,21:23)]
set.seed(123)

### Preparing Training and Test Data 
pd<-sample(2,nrow(Loan.work2),replace=TRUE, prob=c(0.7,0.3))
train.G<-Loan.work2[pd==1,]
test.G<-Loan.work2[pd==2,]


regfit.best<-regsubsets(EMI.Ratio~., data=train.G, nvmax=14)

####Build test matrix
test.matrix<-model.matrix(EMI.Ratio~., data=test.G)
test.matrix
val.errors=rep(0,13)
for(i in 1:13){
  coeff=coef(regfit.best,id=i)
  pred=test.matrix[,names(coeff)]%*%coeff
  val.errors[i]=mean((test.G$EMI.Ratio-pred)^2)
}
val.errors
optsub<-which.min(val.errors)
optsub
coef(regfit.best,optsub)



####RIDGE and LASSO
#install.packages("glmnet")
library(glmnet)
x.G=model.matrix(EMI.Ratio~., data=Loan.work2)[,-1]
y.G=na.omit(Loan.work2$EMI.Ratio)

### Grid is used to find Lambda
grid=10^(seq(10,-2,length=100))
grid

#By default the glmnet() function performs ridge regression for an automatically
#selected range of λ values. However, here we have chosen to implement
#the function over a grid of values ranging from λ = 10^10 to λ = 10^2, essentially
#covering the full range of scenarios from the null model containing
#only the intercept, to the least squares fit. As we will see, we can also compute
#model fits for a particular value of λ that is not one of the original
#grid values.

###Ridge: alpha=0
ridge.model.G=glmnet(x.G,y.G,alpha=0,lambda=grid)
predict(ridge.model.G,s=50,type="coefficients")[1:14,]
###Test Train
set.seed(1)
train.Ridge.G<-sample(1:nrow(x.G),nrow(x.G)/2)
test.Ridge.G<-(-train.Ridge.G)
y.test.GR<-y.G[test.Ridge.G]
#cross val
set.seed(1)
cv.out.ridge.G=cv.glmnet(x.G[train.Ridge.G,],y.G[train.Ridge.G],alpha=0)
#plot(cv.out.ridge)
#names(cv.out.ridge)
bestlambda.ridge.G=cv.out.ridge.G$lambda.min
bestlambda.ridge.G 
#0.1599629
pred.ridge.G<-predict(ridge.model.G,  s=bestlambda.ridge.G, newx=x.G[test.Ridge.G,])
MSERidge.G<-mean((pred.ridge.G-y.test.GR)^2)
MSERidge.G
pred.ridge.G<-predict(ridge.model.G, type="coefficients", s=bestlambda.ridge.G, newx=x.G[test.Ridge.G,])
pred.ridge.G


###Lasso: alpha=1
lasso.model.G=glmnet(x.G,y.G,alpha=1,lambda=grid)
predict(lasso.model.G,s=.15,type="coefficients")[1:14,]
###Test Train
set.seed(1)
train.Lasso.G<-sample(1:nrow(x.G),nrow(x.G)/2)
test.Lasso.G<-(-train.Lasso.G)
y.test.GL<-y.G[test.Lasso.G]
#cross val
set.seed(1)
cv.out.lasso.G=cv.glmnet(x.G[train.Lasso.G,],y.G[train.Lasso.G],alpha=1)
#plot(cv.out.lasso)
#names(cv.out.lasso)
bestlambda.lasso.G=cv.out.lasso.G$lambda.min
bestlambda.lasso.G
pred.lasso.G<-predict(lasso.model.G,  s=bestlambda.lasso.G, newx=x.G[test.Lasso.G,])
MSELasso.G<-mean((pred.lasso.G-y.test.GL)^2)
MSELasso.G

lasso.coeff.G<-predict(lasso.model.G, type="coefficients", s=bestlambda.lasso.G, newx=x.G[test.Lasso.G,])
lasso.coeff.G

########What did we achieve
####MSE with Regression

###LM


trainLM.G<-Loan.work2[1:391,]
testLM.G<-Loan.work2[392:781,]

lm.G<-lm(EMI.Ratio~.,data=trainLM.G)
summary(lm.G)
vif(lm.G)

Linear.2<-EMI.Ratio~Loan.Offered+Work.Exp+Credit.Score+Management+Skilled+CH.good+CH.critical+Purpose.car+Purpose.education
lm.G<-lm(Linear.2,data=trainLM.G)
summary(lm.G)
vif(lm.G)


Pred_LM.G <- predict(lm.G,testLM.G)
MSE_LM.G<-mean((Pred_LM.G-testLM.G$EMI.Ratio)^2)
MSE_LM.G

MSERidge.G
MSELasso.G



###SVM
####2_D PLOT

set.seed(1234)
pd<-sample(2,nrow(Loan),replace=TRUE, prob=c(0.7,0.3))

train<-Loan[pd==1,]
val<-Loan[pd==2,]


sum(Loan$Default)
sum(val$Default)
sum(train$Default)



train.2fact<-train[,c(7,4,5)]
val.2fact<-val[,c(7,4,5)]

library(e1071)
svm.2<-svm(Status~., data=train.2fact, kernel="linear")
summary(svm.2)

plot(svm.2,data=train.2fact, Credit.Score~Work.Exp)


y_pred.svm<-predict(svm.2,newdata=val.2fact[-1])

#Confusion matrix



cm.SVMB.1=table(val.2fact[,1],y_pred.svm)
cm.SVMB.1
accuracy.svm<-sum(diag(cm.SVMB.1))/sum(cm.SVMB.1)
accuracy.svm



#####


###
#Tuning
#Full Model
train.svm<-train[,c(12,2,4:6, 9,11,13:24)]
val.svm<-val[,c(12,2,4:6, 9,11,13:24)]


svm.full<-svm(Default~., data=train.svm, kernel="radial")
summary(svm.full)
y_pred.svm.full<-predict(svm.full,newdata=val.svm[-1])

#Confusion matrix



cm.SVMB.full=table(val.svm[,1],y_pred.svm.full>0.5)
cm.SVMB.full
accuracy.svm.full<-sum(diag(cm.SVMB.full))/sum(cm.SVMB.full)
accuracy.svm.full
#Tune
set.seed(123)
##
#Best model choose by tuning: We can further improve our SVM model
#and tune it so that the error is even lower. 
#We will now go deeper into the SVM function and the tune function.
#We can specify the values for the cost parameter and epsilon 
#which is 0.1 by default. A simple way is to try for each value
#of epsilon between 0 and 1 (we will take steps of 0.1) and 
#similarly try for cost function from 4 to 2^9 
#(we will take exponential steps of 2 here). 
#we are taking 11 values of epsilon and 8 values of cost function. 
#we will thus be testing 88 models and see which ones performs best.

##

tune.svm<-tune(svm, Default~.,data=train.svm, 
               ranges=list(epsilon=seq(0,1,0.1), cost=2^(2:9)))

summary(tune.svm)

best.svm<-tune.svm$best.model
summary(best.svm)
best.par<-tune.svm$best.parameters
summary(best.par)

#CM

y_pred.svm.best<-predict(best.svm,newdata=val.svm[-1])

#Confusion matrix



cm.SVMB.best=table(val.svm[,1],y_pred.svm.best>0.5)
cm.SVMB.best
accuracy.svm.best<-sum(diag(cm.SVMB.best))/sum(cm.SVMB.best)
accuracy.svm.best

