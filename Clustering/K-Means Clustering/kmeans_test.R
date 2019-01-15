dataset = read.csv("Mall_Customers.csv")
X <- dataset[4:5]

#Using elbow Method
set.seed(6)
wcss <- vector()
for(i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type='b', main = paste('Clusters of Clients'),
     xlab='Number of clusters', ylab='WCSS')


#Applying kmeans
set.seed(29)
kmeans <- kmeans(X, 5, iter.max=300, nstart=10)

#Plotting
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines=0,
         shade=FALSE,
         color=TRUE,
         labels=2,
         plotchar=FALSE,
         span=TRUE,
         main = paste('Cluster of Clients'),
         xlab = 'Annual Income',
         ylab = 'Spending Score')