# Machine Learning - Regulization
#install.packages("glmnet") 
#install.packages("lars") 

# Ridge Regression

# load the package
library(glmnet)
# load data
data(longley)
str(longley)
View(longley)


head(longley,10)
x <- as.matrix(longley[,1:6]) # Indepedent Variable
y <- as.matrix(longley[,7]) # Dependent Variable

# Correlation of all matrix variable
pairs(longley, main = "longley data")
cor(longley) # Correlation chart

# Linear Regression
summary(fm1 <- lm(Employed ~ ., data = longley))


opar <- par(mfrow = c(2, 2), oma = c(0, 0, 1.1, 0),
            mar = c(4.1, 4.1, 2.1, 1.1))
plot(fm1)
help(par)
#par can be used to set or query graphical parameters. Parameters can be set by specifying them 
#as arguments to par in tag = value form, or by passing them as a list of tagged values.
help("glmnet")
# fit model
"
glmnet
Fit a generalized linear model via penalized maximum likelihood. 
The regularization path is computed for the lasso or elasticnet penalty at a grid of values for 
the regularization parameter lambda. Can deal with all shapes of data, including very large sparse 
data matrices. Fits linear, logistic and multinomial, poisson, and Cox regression models.
"
fit <- glmnet(x, y, family="gaussian", alpha=0, lambda=0.001)
# summarize the fit
print(fit)


# make predictions
predictions <- predict(fit, x, type="link")
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)

# 3 components in GLM, random(The random component states the probability distribution
# of the response variable.), systematic(It specifies the linear combination of the explanatory
#variables, it consist in a vector of predictors) and link(It connects the random and the systematic component. It
#shows how the expected value of the response variable is connected to the linear predictor of explanatory variables)


# Least Absolute Shrinkage and Selection Operator

# load the package
library(lars)
# load data
data(longley)
x <- as.matrix(longley[,1:6])
y <- as.matrix(longley[,7])
# fit model
fit <- lars(x, y, type="lasso")
#summarize the fit
print(fit)
#Step size is how lambda changes between each calculation of the model.
#Best step is just labeling the chosen model from amongst the family of models. The example uses minimum RSS as the metric for best.
#select a step with a minimum error
best_step <- fit$df[which.min(fit$RSS)]
# make predictions
predictions <- predict(fit, x, s=best_step, type="fit")$fit
# summarize accuracy
mse <- mean((y - predictions)^2)
print(mse)
