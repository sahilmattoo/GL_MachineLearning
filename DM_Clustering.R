
data = read.csv("Clustering_Data.csv", header=TRUE)
str(data)
head(data)

# Credit limit is in 5 digit and rest of the data is in 1 digit - perform scalling

scaledata = scale(data[,3:7])
head(scaledata,10)

#Create the distance matrix
distdata <- dist(x=scaledata, method = 'euclidean') 
distdata

#Use Average linkage method to form the clusters
avgclus = hclust(distdata, method = 'average')

#Plot the dendogram
plot(avgclus, labels = as.character(data[,2]))
rect.hclust(avgclus, k=5, border="red")

## profiling the clusters
data$clus <- cutree(avgclus, k=3) # number of clusters

data[c(1,2,8)] # Cust Id, Name and Cluster
# Apply aggregate function 
aggr = aggregate(data[,-c(1,2, 8)],list(data$clus),mean)
View(aggr)

table(data$clus) # frequency of clusters
Cluster=aggr[,1]
Cluster
clus.profile <- data.frame( Cluster=aggr[,1],
                            Freq=as.vector(table(data$clus)),
                            aggr[,-1])

View(clus.profile)



## K Means Clustering
#Non-Hierarchical technique


KRCDF = read.csv("Clustering_Data.csv", header=TRUE)
str(KRCDF)
head(KRCDF)

## scale function standardizes the values
scaled.RCDF <- scale(KRCDF[,3:7])

# define wssplot function
#nrow(data)-1 = 9

wssplot <- function(data, nc=15, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  # for i = 2 to 15 - iterate
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}

wssplot(scaled.RCDF, nc=5)


#nstart = 25 it will do the clustering process 25 times for each observation to check consistency of the obsevation on every iteration
kmeans.clus = kmeans(x=scaled.RCDF, centers = 3, nstart = 25)
kmeans.clus

library(fpc)
library(cluster)

## profiling the clusters
KRCDF$Clusters <- kmeans.clus$cluster
KRCDF
aggr = aggregate(KRCDF[,-c(1,2, 8)],list(KRCDF$Clusters),mean)
clus.profile <- data.frame( Cluster=aggr[,1],
                            Freq=as.vector(table(KRCDF$Clusters)),
                            aggr[,-1])

#Output
View(clus.profile)
