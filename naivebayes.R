library(ISLR)
attach(Smarket)
summary(Smarket)
View(Smarket)
# Lag 1 - Percentage retrun previous day
# Lag 2/3/4 - Percentage retrun 2/3/4 days ago
# Volume - Vol of share traded in billions

train = (Year<2005)
Smarket.2005 = Smarket[!train,]
dim(Smarket.2005)
names(Smarket)
Direction.2005 = Direction[!train]
dim(Direction.2005)

library(e1071)
snb=naiveBayes(Direction~Lag1+Lag2,Smarket,subset=train)
snb.pred=predict(snb,Smarket.2005)
table(snb.pred,Direction.2005)
mean(snb.pred==Direction.2005)

#QDA - Quadratic Discriminant Analysis
library(MASS)
qda.fit=qda(Direction~Lag1+Lag2,data=Smarket,subset=train)
qda.class=predict(qda.fit,Smarket.2005)$class
table(qda.class,Direction.2005)
mean(qda.class==Direction.2005)
table(qda.class,snb.pred)
