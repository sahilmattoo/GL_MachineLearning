
library(recipes)
data("credit_data")
View(credit_data)
vars <- c("Home", "Seniority")
str(credit_data[, c(vars, "Status")])
#simple split
# a simple split
set.seed(2411)
in_train <- sample(1:nrow(credit_data), size = 3000)
train_data <- credit_data[ in_train,]
test_data  <- credit_data[-in_train,]
#fit a simple classification tree model
library(C50)
tree_mod <- C5.0(x = train_data[, vars], y = train_data$Status)
tree_mod
summary(tree_mod)
plot(tree_mod)
tree_boost <- C5.0(x = train_data[, vars], y = train_data$Status, trials = 3)
summary(tree_boost)
#Rule Based Boosting
rule_mod <- C5.0(x = train_data[, vars], y = train_data$Status, rules = TRUE)
rule_mod
summary(rule_mod)
#no pruning is guarenteed
predict(rule_mod, newdata = test_data[1:3, vars])
predict(tree_boost, newdata = test_data[1:3, vars], type = "prob")


