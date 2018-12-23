# XGBoost
setwd("C:/Users/SahilMattoo/Documents/PGP-BABI/Machine Learning/Residency 2/RCodesFor2ndResidency")

# Importing the dataset
dataset = read.csv('Churn.csv')
dataset = dataset[4:14]
View(dataset)
str(dataset)

# Encoding the categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting XGBoost to the Training set
# install.packages('xgboost')
library(xgboost)
View(training_set)
View(training_set[-11]) # Except Exited
classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = as.matrix(test_set[-11]))
y_pred = (y_pred >= 0.5)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
cm #86.85%
# Applying k-Fold Cross Validation
# install.packages('caret')
library(caret)

help("createFolds")
# createFolds(y, k = 10, list = TRUE, returnTrain = FALSE)
folds = createFolds(training_set$Exited, k = 3) # Data Splitting functions


cv = lapply(folds, function(x) {
  training_fold = training_set[-x, ]
  test_fold = training_set[x, ]
  classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)
  y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
  y_pred = (y_pred >= 0.5)
  cm = table(test_fold[, 11], y_pred)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy # 0.875

for(i in 1:10)
{
  folds = createFolds(training_set$Exited, k = i)
  cv = lapply(folds, function(x) {
    training_fold = training_set[-x, ]
    test_fold = training_set[x, ]
    classifier = xgboost(data = as.matrix(training_set[-11]), label = training_set$Exited, nrounds = 10)
    y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]))
    y_pred = (y_pred >= 0.5)
    cm = table(test_fold[, 11], y_pred)
    accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
    return(accuracy)
  })
  accuracy[i] = mean(as.numeric(cv))
  accuracy[i]
  
  
}
for (i in 1:10) {
  print(accuracy[i])
}
