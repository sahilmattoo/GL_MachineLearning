# GL_MachineLearning
Machine Learning Examples


Data Files
Dafault.CSV

https://raw.githubusercontent.com/sahilmattoo/GL_MachineLearning/master/default.csv

Churn.CSV

https://raw.githubusercontent.com/sahilmattoo/GL_MachineLearning/master/Churn.csv

CLUSTERING

https://raw.githubusercontent.com/sahilmattoo/GL_MachineLearning/master/Clustering_Data.csv

Decision Tree

https://raw.githubusercontent.com/sahilmattoo/GL_MachineLearning/master/DecisionTree_data.csv


-------------------------------- READ DATA FROM GITHUB Into R ---------------------

library(readr)  # for read_csv

library(knitr)  # for kable

myfile <- "https://raw.githubusercontent.com/sahilmattoo/NeuralNets/master/Churn_Modelling.csv"

Affairs <- read_csv(myfile)

