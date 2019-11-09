# ML_datamining

Problem Definition : Here we will be learning how Spotify uses it’s algorithms to predict a song’s popularity . 
Song popularity is a ,etric based on the reach the song has to audience and with this model if we have the parameters beforehand we 
can predict its popularity. We also added one more column for like/dislike and recorded the likes of a single user . We used this data 
to to build a model to predict if a given song will be loved by the user

Data Description: There are 16 columns in this dataset which specify the characteristic of each song . 
The columns include features like song name and artist and other attributes which are more specific to the song type like dance ability, 
beats per min and energy etc. The number of records will be around 19000. The data will have to scaled and the divided into training and 
validation dataset. 
The correlation plot was also plotted to see correlation of all the variables in the same plot. From these plots we decided to remove 
some of the predictors in the dataset.

Data Mining Tasks: Linear regression, random forest were used to predict the songpolularity while knn and logistic regression was used 
for the classification of liked songs. The predictors were scaled to bring it in between -1 and 1. This was done because of different 
predictor ranges in the data. Once they were scaled, model was built using training data.
With 94.3% accuracy this model performs well for the data. 


User Prediction Trend: To recommend a user any songs based on his/her likes and dislikes, we added another column to the dataset 
called ’liking’ where the user inputs the values of 1 for like and 0 for dislike. We filled the values for 150 observations from which 
100 were used as training data and 50 were used as validation data. Normalization and dimension reduction were also performed to get 
accurate results. We decided to run the training data through logistic regression and KNN models to get a comparision.

LOGISTIC REGRESSION: Logistic regression was not an efficient model as the ROC curve almost follows a linear path.  Almost all the 
parameters were not significant as their p-value was more than 5%.

K-NEAREST NEIGHBOURS : KNN gives the best results in comparison with logistic regression model. K=1 is giving accurate results 
as compared with other values. The values for true positive class and false negative classes are higher and significant for k=1. 
Accuracy for KNN is 83%. Hence, this is a better model for prediction compared to Logistic regression. • Based on the user likings, 
the user would not like a particular with given parameter ratings.
