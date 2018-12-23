
#Some libraries

library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(ggplot2)
library(Hmisc)
library(MASS)
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


sum(Loan$Default)
sum(val$Default)
sum(train$Default)




#Logistic Regression

#install.packages(c("SDMTools","pROC", "Hmisc"))
library(SDMTools)
library(pROC)
library(Hmisc)

#data frame
train.logit<-train[,c(2,4:6,9,11:24)]
val.logit<-val[,c(2,4:6,9,11:24)]

#####Retain only significant ones
Logit.eq.final<-Default  ~ Credit.Score+  EMI.Ratio+Dependents+Female+Skilled+CH.verygood+Purpose.education

Logit.final <- glm(Logit.eq.final   , train.logit, family = binomial)
summary(Logit.final)
vif(Logit.final)

varImp(Logit.final)
pred.logit.final <- predict.glm(Logit.final, newdata=val.logit, type="response")

#Classification
tab.logit<-table(val.logit$Default,pred.logit.final>0.5)
tab.logit
#Logit
accuracy.logit<-sum(diag(tab.logit))/sum(tab.logit)
accuracy.logit


#K FOLD VALIDATIONS

Loantrim<-Loan[,c(12,5,6,11,13,15,20,23)]
na.omit(Loantrim)
Loantrim$Default<-as.numeric(Loantrim$Default)
set.seed(1234)
pd<-sample(2,nrow(Loantrim),replace=TRUE, prob=c(0.7,0.3))

traintrim<-Loantrim[pd==1,]
valtrim<-Loantrim[pd==2,]


set.seed(123)
folds<-createFolds(Loantrim$Default,k=10)
str(folds)
######

#10-Fold for Logit
Eq.2<-Default  ~ Credit.Score+  EMI.Ratio +Dependents+ Female+Skilled+CH.verygood+Purpose.education

cv_logit<-lapply(folds,function(x){
  train<-traintrim[x,]
  test<-valtrim[-x,]
  Logit.1<-glm(Eq.2, train, family = binomial)
  pred.1 <- predict.glm(Logit.1, newdata=test, type="response")
  actual<-test$Default
  tab.logit.1<-confusion.matrix(actual,pred.1,threshold = 0.5)
  sum(diag(tab.logit.1))/sum(tab.logit.1)
})

str(cv_logit)
fit.logit<-mean(unlist(cv_logit))
fit.logit

#10 Fold validation with LPM
#LPM : a linear probability model is a special case of a binomial regression model.
#Here the dependent variable for each observation takes values 
#which are either 0 or 1. The probability of observing a 0 or 1 
in any one case is treated as depending on one or more explanatory variables.
cv_LPM<-lapply(folds,function(x){
  train<-traintrim[x,]
  test<-valtrim[-x,]
  LPM.1<-lm(Eq.2, train)
  LPM1.pred<-predict(LPM.1, test)
  tab.LPM<-table(test$Default, LPM1.pred>0.5)
  sum(diag(tab.LPM))/sum(tab.LPM)
})

str(cv_LPM)
fit.LPM<-mean(unlist(cv_LPM))
fit.LPM
##########
#10 Vold Validation with NB
traintrim$Default<-as.factor(traintrim$Default)
valtrim$Default<-as.factor(valtrim$Default)

cv_NB<-lapply(folds,function(x){
  train.NB.kval<-traintrim[x,]
  test.NB.kval<-valtrim[-x,]
  NB.kval<-naiveBayes(x=train.NB.kval[-1], y=train.NB.kval$Default)
  y_pred.NB.kval<-predict( NB.kval,newdata=test.NB.kval[-1])
  cm.NB.kval=table(test.NB.kval[,1],y_pred.NB.kval)
  sum(diag(cm.NB.kval))/sum(cm.NB.kval)
})

str(cv_NB)
fit.NB<-mean(unlist(cv_NB))
fit.NB

#####################


#####################

#10 Fold with LDA
library(MASS)
library(ISLR)

cv_LDA<-lapply(folds,function(x){
  train<-traintrim[x,]
  test<-valtrim[-x,]
  lda_1<-lda(Eq.2   , train)
  lda1.pred<-predict(lda_1, newdata=test)
  ldapredclass<-lda1.pred$class
  tab.LDA<-table(ldapredclass,test$Default)
  sum(diag(tab.LDA))/sum(tab.LDA)
})

str(cv_LDA)
fit.LDA<-mean(unlist(cv_LDA))
fit.LDA
#########
#10 Fold on Decision Trees

cv_DT<-lapply(folds,function(x){
  train<-traintrim[x,]
  test<-valtrim[-x,]
  DT<-rpart(Eq.2, method="class",train)
  pred = predict(DT, type="class",newdata=test)
  tabDT<-table( pred,test$Default)
  sum(diag(tabDT))/sum(tabDT)
})

str(cv_DT)
fit.DT<-mean(unlist(cv_DT))
fit.DT
########

##########
#SMOTE TO IMPROVE


Logit.eq.final<-Default  ~ Credit.Score+  EMI.Ratio+Dependents+Female+Skilled+CH.verygood+Purpose.education

train.log1<-train[,c(2,4:6,9,11:15,18:23)]
val.log1<-val[,c(2,4:6,9,11:15,18:23)]

#Logit.2<-Default  ~ Credit.Score+ Loan.Offered+Own.house+ EMI.Ratio+Dependents+Female+Skilled+Management+CH.verygood+Purpose.education

Logit.final <- glm(Logit.eq.final , train.log1, family = binomial)
summary(Logit.final)
vif(Logit.final)

varImp(Logit.final)
pred.logit.final <- predict.glm(Logit.final, newdata=val.logit, type="response")

tab.logit.SM<-confusion.matrix(val.log1$Default,pred.logit.final,threshold = 0.5)
tab.logit.SM
accuracy.logit.SM<-sum(diag(tab.logit.SM))/sum(tab.logit.SM)
accuracy.logit.SM


########
#SMOTE

#install.packages("DMwR")
library(DMwR)


train_SMOTE<-train.log1[,c(2,3,7)]
qplot(Credit.Score,Work.Exp,color=Default, data=train.log1)

table(train.log1$Default)
#SMOTE

#Two factors to see the plot
train_SMOTE$target <- as.factor(train.log1$Default)
trainSplit <- SMOTE(target ~ ., train_SMOTE, perc.over = 100, perc.under=100)
trainSplit$target <- as.numeric(trainSplit$target)
trainSplit$target<-ifelse(trainSplit$target==1,1,0)

print(prop.table(table(trainSplit$target)))
table(trainSplit$target)
qplot(Credit.Score,Work.Exp,color=Default, data=train.log1)



train_SMOTE_new<-train.log1


#train_SMOTE_new$winnew<-ifelse(train_SMOTE_new$win==1,1,0)
train_SMOTE_new <- SMOTE(Default ~ ., train_SMOTE_new, perc.over = 100, perc.under=300)
train_SMOTE_new$target <- as.factor(train_SMOTE_new$Default)
trainSplit <- SMOTE(target ~ ., train_SMOTE_new, perc.over = 100, perc.under=300)
trainSplit$target <- as.numeric(trainSplit$target)
trainSplit$target<-ifelse(trainSplit$target==1,1,0)




print(prop.table(table(trainSplit$target)))
#write.csv(train_SMOTE, "SMOTE.csv")


###### LOGIT with smote

summary(trainSplit$Default)
sum(val.log1$Default)

Logit.SM<-glm(Logit.eq.final,data=trainSplit, family=binomial)
summary(Logit.SM)
vif(Logit.SM)

Logit.SMOTE<-Default  ~ Credit.Score+ Work.Exp+ EMI.Ratio+Dependents +Own.house + CH.verygood+Female+ Skilled

Logit.SM<-glm(Logit.SMOTE,data=trainSplit, family=binomial)
summary(Logit.SM)
vif(Logit.SM)

varImp(Logit.SM)
pred.logit.SM <- predict.glm(Logit.SM, newdata=val.log1, type="response")

#Classification



tab.logit.SM<-confusion.matrix(val.log1$Default,pred.logit.SM,threshold = 0.5)
tab.logit.SM
accuracy.logit.SM<-sum(diag(tab.logit.SM))/sum(tab.logit.SM)
accuracy.logit.SM




#########
####ACTIONABLE INSIGHTS
#########
val.logit$LLhood<-log(pred.logit.SM/(1-pred.logit.SM))

val.logit$pred<-pred.logit.SM

Logit.SM$coefficients

summary(Logit.SM)$coeff[2,1]

maxCS<-max(val.logit$Credit.Score)
minEMI<-min(val.logit$EMI.Ratio)
minCS<-min(val.logit$Credit.Score)
maxEMI<-max(val.logit$EMI.Ratio )
maxDep<-max(val.logit$Dependents)
minDep<-min(val.logit$Dependents)
maxWork<-max(val.logit$Work.Exp)
minWork<-min(val.logit$Work.Exp)




CScoeff<-summary(Logit.SM)$coeff[2,1]
Workcoeff<-summary(Logit.SM)$coeff[3,1]
EMIcoeff<- summary(Logit.SM)$coeff[4,1]
Depcoeff<-summary(Logit.SM)$coeff[5,1]



val.logit$CS.new<-ifelse(val.logit$LLhood/(CScoeff)  +val.logit$Credit.Score>maxCS,"no",
                         ifelse(val.logit$LLhood/(CScoeff)  +val.logit$Credit.Score<minCS,"no",
                                abs(val.logit$LLhood/(CScoeff*val.logit$Credit.Score))*100))                  

val.logit$EMI.new<-ifelse(val.logit$LLhood/(EMIcoeff)  +val.logit$EMI.Ratio>maxEMI,"no",
                          ifelse(val.logit$LLhood/(EMIcoeff)  +val.logit$EMI.Ratio   <minEMI,"no",
                                 abs(val.logit$LLhood/(EMIcoeff*val.logit$EMI.Ratio))*100))                  


val.logit$Dep.new<-ifelse(val.logit$LLhood/(Depcoeff)  +val.logit$Dependents>maxDep,"no",
                          ifelse(val.logit$LLhood/(Depcoeff)  +val.logit$Dependents   <minDep,"no",
                                 abs(val.logit$LLhood/(Depcoeff*val.logit$Dependents))*100))                  


val.logit$Work.new<-ifelse(val.logit$LLhood/(Workcoeff)  +val.logit$Work.Exp>maxWork,"no",
                           ifelse(val.logit$LLhood/(Workcoeff)  +val.logit$Work.Exp   <minWork,"no",
                                  abs(val.logit$LLhood/(Workcoeff*val.logit$Work.Exp))*100))                  


####ACTIONABLE INSIGHTS: CONTINUOUS RESPONSE

###Let's say we are trying to Predict EMI Ratio

LoanLM<-Loan[,c(6,5,11,13,15,20,23)]
na.omit(LoanLM)

normalize<-function(x){
  +return((x-min(x))/(max(x)-min(x)))}
LoanLM$Credit.Score<-normalize(LoanLM$Credit.Score)
LoanLM$Dependents<-normalize(LoanLM$Dependents)



set.seed(1234)
pd<-sample(2,nrow(LoanLM),replace=TRUE, prob=c(0.7,0.3))

trainLM<-LoanLM[pd==1,]
valLM<-LoanLM[pd==2,]



lm.G<-lm(EMI.Ratio~.,data=trainLM)
summary(lm.G)
vif(lm.G)

Pred_LM.G <- predict(lm.G,valLM)
MSE_LM.G<-mean((Pred_LM.G-valLM$EMI.Ratio)^2)
MSE_LM.G

####
#Let's define average values
valLM$avCS<-mean(valLM$Credit.Score)
valLM$avDep<-mean(valLM$Dependents)
#Also the opposite values
valLM$NFem<-1-valLM$Female
valLM$NSkl<-1-valLM$Skilled
valLM$NCHVG<-1-valLM$CH.verygood
valLM$NPurEd<-1-valLM$Purpose.education
####3
maxCS<-max(valLM$Credit.Score)
minCS<-min(valLM$Credit.Score)
maxDep<-max(valLM$Dependents)
minDep<-min(valLM$Dependents)


CScoeff<-summary(lm.G)$coeff[2,1]
Depcoeff<-summary(Logit.SM)$coeff[3,1]


valLM$resid<-(Pred_LM.G-valLM$EMI.Ratio)^2
valLM$CS.new<-ifelse(valLM$resid/(CScoeff)  +valLM$Credit.Score>maxCS,"no",
                     ifelse(valLM$resid/(CScoeff)  +valLM$Credit.Score<minCS,"no",
                            abs(valLM$resid/(CScoeff*valLM$Credit.Score))))                  


valLM$Dep.new<-ifelse(valLM$resid/(Depcoeff)+valLM$Dependents   >maxDep,"no",
                      ifelse(valLM$resid/(Depcoeff)  +valLM$Dep<minDep,"no",
                             abs(valLM$resid/(Depcoeff*valLM$Dep))))                  



         