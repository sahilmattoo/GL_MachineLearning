#Importing the data
pdata = read.csv("data.csv") # Import Decision Tree Data Set from GitHub
head(pdata)

#Exploring the dataset
str(pdata)

#install.packages("caTools)
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("rattle")
#install.packages("RColorBrewer")
#install.packages("data.table")
#install.packages("ROCR")


#Loading all the required libraries
library(caTools)
library(rpart)
library(rpart.plot)
library(rattle)
library(RColorBrewer)
library(data.table)
library(ROCR)



#Paritioning the data into training and test dataset
set.seed(3000)
sample = sample.split(pdata,SplitRatio = 0.7)
p_train = subset(pdata,sample == TRUE)
p_test = subset(pdata,sample == FALSE)
nrow(p_train)
nrow(p_test)


str(p_train)

head(p_train)

#Checing the delinquency distribution
table(p_train$delinquent)


#Setting the control parameters
r.ctrl = rpart.control(minsplit=1000, minbucket = 100, cp = 0, xval = 10)

#Building the CART model
m1 <- rpart(formula = Sdelinquent~term+gender+FICO+age, data = p_train, method = "class", control = r.ctrl)
m1

#Displaying the decision tree
fancyRpartPlot(m1)


#Scoring/Predicting the training dataset
p_train$predict.class <- predict(m1, p_train, type="class")
p_train$predict.score <- predict(m1, p_train)
head(p_train)

#COnverting the dataset into deciles
p_train$deciles <- decile(p_train$predict.score[,2])


#Building the ROC curve and lift charts
pred <- prediction(p_train$predict.score[,2], p_train$Sdelinquent)
perf <- performance(pred, "tpr", "fpr")
plot(perf,main = "ROC curve")


#Model validation parameters
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)

#install.packages("ineq")
#library(ineq)

#Checking the classification error
with(p_train, table(Sdelinquent, predict.class))
nrow(p_train)

auc
KS


# Scoring test sample and validating the same
p_test$predict.class <- predict(m1, p_test, type="class")
p_test$predict.score <- predict(m1, p_test)
head(p_test)

with(p_test, table(Sdelinquent, predict.class))
nrow(p_test)

library(ROCR)
pred <- prediction(p_test$predict.score[,2], p_test$Sdelinquent)
perf <- performance(pred, "tpr", "fpr")
plot(perf,main = "ROC curve")


KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)

