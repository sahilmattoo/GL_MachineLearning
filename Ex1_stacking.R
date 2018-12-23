library(mlbench)
library(caret)
library(caretEnsemble)

# Load the dataset
data(Ionosphere)
dataset <- Ionosphere
dataset <- dataset[,-2]
View(dataset)
dataset$V1 <- as.numeric(as.character(dataset$V1))

seed <- 7

# create submodels
control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, 
                        classProbs=TRUE)
algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')
set.seed(seed)
models <- caretList(Class~., data=dataset, trControl=control, methodList=algorithmList)
results <- resamples(models)
summary(results)
dotplot(results)

modelCor(results)
splom(results)

#Logistic and KNN are highly correlated
#Let us combine the predictions of the classifier using linear model
stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, 
                             classProbs=TRUE)
set.seed(seed)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)
"
Accuracy   Kappa    
  0.9525211  0.8966403
"
#Stacking using RandomForest
# stack using random forest
set.seed(seed)
stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf)
"
   mtry  Accuracy   Kappa    
  2     0.9547037  0.9006837
  3     0.9531132  0.8975998
  5     0.9502650  0.8914336
"
