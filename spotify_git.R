set.seed(1)
library(corrplot)
#removing few columns which have junk values
songdat<-song_data[,-1:-2]
#finding correlation matrix for the songs data
s<-cor(songdat)
corrplot(s, method="circle")
s
#filtering out only unique data
newsongdata<-unique(song_data)
summary(song_data)
#dividing the data into 70:30 for training and the rest for testing
train.index <- sample(c(1:dim(newsongdata)[1]), dim(newsongdata)[1]*0.7)
train.df <- newsongdata[train.index, ]
valid.df <- newsongdata[-train.index, ]

summary(newsongdata)
#scaled data
#caret package for linear regression
library(caret)
scaled_data<-scale(newsongdata[,-1])
colMeans(scaled_data)
#Training data in linear regression model
songs_mod <- lm(song_popularity ~ song_duration_ms + acousticness + danceability + instrumentalness + key + liveness + loudness + audio_mode + speechiness + tempo + time_signature + audio_valence, newsongdata)
summary(songs_mod)

trainsong<-lm(song_popularity~danceability+instrumentalness+liveness+tempo, data=train.df)
#using the trained the model for prediction of validation data(test data)
predsong<-predict(trainsong,newdata=valid.df)
actual <- valid.df$song_popularity
#finding rmse of the result
rmse <- (mean((predsong - actual)^2))^0.5
rmse
#finding accuracy, Root mean error and error rate
library(plm)
rss <- sum((pre - acc) ^ 2)
tss <- sum((acc - mean(acc)) ^ 2)
rsq <- 1 - rss/tss
rm<-RMSE(pre,acc)
error<-rm/mean(acc)
summary(trainsong)
accuracy<-mean((pre-acc)/(acc))*100
c<-which(actual==0)
acc<-actual[-c]
pre<-predsong[-c]

#randomforest
#rpart is the package we use for randomforest 
install.packages("rpart")
library(randomForest)
library(reptree)
#traing data for regression random forest tree model
songs_rm<-randomForest(song_popularity~danceability+instrumentalness+liveness+tempo, data=train.df)
#ploting the tree 
plot(songs_rm)
#prediction on test data using the model
predictrf<-predict(songs_rm,newdata=valid.df)
actual1<-valid.df$song_popularity
#finding the accuracy
accuracyrm<-mean((pre1-acc)/(acc))*100
pre1<-predictrf[-c]
#plotting regression tree to see lowest level nodes
library(FFTrees)
tree <- FFTrees(formula =song_popularity~danceability+instrumentalness+liveness+tempo,
                data = train.df,
                data.test = valid.df,
                main = "sdd")
summary(tree)
plot(tree)
view(head(song_data))

#this is the second part of the project, so we have imported the dataset again from scratch as 'song'.


song_data<-unique(song)
View(song_data)
summary(song_data)
song_data$song_name <- NULL
song_data$audio_mode <- NULL
song_data$time_signature <- NULL
song_data$key <- NULL
View(song_data)
#scaling data 
scaled_data<-scale(song_data[,-12])
colMeans(scaled_data)
View(scaled_data)
#created new colum called liking to record users liking
newsong_data = cbind(scaled_data, song_data$liking)
#populating the liking data
colnames(newsong_data)[12] <- "liking"
View(newsong_data)
songs = newsong_data[1:149,]
View(songs)

#dividing the data into training and validation data
train.songs <- songs[1:100,]
valid.songs <- songs[101:149, ]
train.songs = data.frame(train.songs)
valid.songs = data.frame(valid.songs)
View(valid.songs)
testingdata = newsong_data[150:15004,-12]
testingdata = data.frame(testingdata)
#logistic regression model created on the data 
install.packages("stats")
library(stats)
songs = data.frame(songs)
logis.reg <- glm(liking ~ . , data = train.songs, family = "binomial")  
summary(logis.reg)

#prediting data on validation songs
log.reg <- glm(songs$liking~ ., data =valid.songs, family = "binomial")
summary(log.reg)
logpredict <- predict(log.reg,valid.songs,type="response")
#confusion matrix for liking column 
table(defaulttest, logpredict > 0.5)

prd = prediction(logpredict, labels = as.factor(defaulttest))
prf = performance(prd, measure = "tpr", x.measure = "fpr")
plot(prf, main = "ROC")

#knn
install.packages("class")
library(class)
as.factor(songs$liking)
as.factor(train.songs$liking)
as.factor(valid.songs$liking)

defaultclass <- train.songs$liking 

defaulttest <- valid.songs$liking
#knn model on train data
knn.pred = knn(train.songs, valid.songs, defaultclass, k=1)
summary(knn.pred)
d = table(knn.pred, defaulttest)
d
#prediction on test data
predictknn = knn(train.songs,c(1,1,1,0,1,1,0,1,1,0,1,0),defaultclass,k=1)
predictknn
install.packages("gains")
library(gains)cind
library(caret)
#confusion matrix for the results
confusionMatrix(as.factor(knn.pred),as.factor(defaulttest))
